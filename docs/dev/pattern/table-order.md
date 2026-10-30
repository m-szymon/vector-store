# Substring search: ordering and table order

Status: implemented on the index node (vector-store, `pattern_index/order.rs` and the ordered walk
in `pattern_index/tantivy.rs`) and in ScyllaDB (`pattern_index::parse_order` and `check_order`,
`pattern_search::fit_ordering` and `index_for`, `ordered_pattern_select_statement`). It builds on follow-up 4 of
SCYLLADB-5644 and is taken as a requirement for the pattern index before any further
optimisation.

## What a pattern index answers

A pattern index replaces the filtered `WHERE col LIKE`. That gives two kinds of query, and the
`order_by` option says which kind an index serves.

- **`WHERE col LIKE p` without `ORDER BY`** keeps the meaning of the filtered `LIKE`: rows come in
  table order. Two kinds of index serve it, and an index is one or the other, never both:
  - **No `order_by` option (the default):** the index keeps its matches in table order. `LIMIT n`
    returns the first `n` rows of that order, so a smaller `LIMIT` is a prefix of a larger one,
    and the search pages with a cursor.
  - **`'order_by': 'unordered'`:** the index returns any `n` matches, with no paging. ScyllaDB
    still sorts the page into table order (`9743001621`, SCYLLADB-5644 itself), so the rows of a
    page are in table order; which rows make the page is unspecified. The option defines what an
    unspecified order means, and exists for cost (see below).
- **`WHERE col LIKE p ORDER BY ...`** is served only by an index declared with ordering columns,
  and only when the `ORDER BY` has a shape the index supports (below). A query without `ORDER BY`
  never uses an ordered index, and `ORDER BY` is rejected on an index without ordering columns.

Table order is the order of a scan of the table. Rows are ordered by token, then by partition key
(this matters only when two partitions share a token), then by clustering key with the schema's
`CLUSTERING ORDER BY`.

## Declaring the order

The order goes in the existing index option. The option's value is a list of columns, each
optionally followed by `ASC` or `DESC`:

```sql
CREATE CUSTOM INDEX ON ks.posts (title) USING 'pattern_index'
    WITH OPTIONS = {'order_by': 'published_at DESC, score ASC'};
```

- **Keyword:** an unquoted `unordered` is the keyword. A column named `unordered` is written
  `'"unordered"'`.
- **Column names:** follow CQL identifier rules, with quoting for case and special characters.
- **Column types:** each must be orderable, as today: integers, `date`, `time`, `timestamp`.

The index's order is (ordering columns, each in its declared direction, then table order).

### Query shapes, as for clustering columns

- **`ORDER BY` names a prefix of the ordering columns:** either all with their declared
  directions, or all reversed. With `published_at DESC, score ASC` declared, these are served:
  - `ORDER BY published_at DESC`;
  - `ORDER BY published_at DESC, score ASC`;
  - `ORDER BY published_at ASC, score DESC`.

  `ORDER BY published_at DESC, score DESC` is rejected, as CQL rejects it for clustering columns.
- **Direction:** the reversed shape is the exact reverse of the declared order, table-order ties
  included. So the declared direction decides which query direction returns ties in table order.
- **Ranges:** a range on the first ordering column (`<`, `<=`, `>`, `>=`) narrows the search, as
  today, in either direction. Ranges on later columns, or tuple ranges such as
  `(a, b) < (x, y)`, may come later.

### Other syntax, for later

ScyllaDB's grammar already parses `CREATE INDEX ... WITH CLUSTERING ORDER BY (a DESC, b ASC)`,
because index properties go through the table-property rule. `create_index_statement.cc` then
rejects it ("Indexes do not allow for specifying the clustering order").

That syntax fits the model:
- a global index is like a single partition clustered by the ordering columns;
- a local index, once there is one, is exactly partition plus clustering.

Lifting the rejection for `pattern_index` and translating it into the option is a candidate
upstream surface. It cannot say `unordered`, whose syntax needs more thought, so the PoC uses the
option.

## Nulls

An ordered index holds only rows whose ordering columns are all set, as a materialized view
declared with `WHERE col IS NOT NULL` does.

- **No filtered equivalent:** an `ORDER BY` on a regular column has no filtered counterpart in
  CQL, so leaving those rows out changes no existing query's result. The query without `ORDER BY`,
  which must match the filtered `LIKE`, is served by the table-order or unordered index, and those
  have no ordering columns.
- **Node handling:** a null ordering value is handled like a null indexed text. The row is left
  out, and setting the column to null removes the row from the index. An empty value (CQL allows
  one even for `int`) counts as null.
- **Writes are not rejected:** CQL has no `NOT NULL` for regular columns, and an index does not
  veto writes to its table.
- **Fixes two current bugs:** today the node stores a null as sort key 0. Nulls then tie with the
  lowest real value (the bigint minimum, day 0 of `date`), and a range `col < x` returns them,
  although a null satisfies no comparison in CQL.
- **Table order needs no null handling:** the token and the clustering key are never null.

## Representation on the index node

- **`token`:** a new `FAST` u64 column holding the row's token, biased like the signed sort keys.
  It is computed at ingestion from the partition key by the driver, which reads each table's
  partitioner from the schema and hashes with it (`ClusterState::compute_token`).
  - The partitioner is a cluster-wide setting, and ScyllaDB supports only Murmur3 for user
    tables; the random and byte-ordered partitioners are deprecated and need
    `enable_deprecated_partitioners`.
  - The only per-table partitioner is the CDC log's, which no pattern index targets. The driver
    handles that one too.
  - A table whose token the driver cannot compute is refused at index creation.
- **Ordering columns:** one `FAST` u64 column each, encoded so the index's declared order is
  ascending. A `DESC` column stores the complement.
- **The index's order:** the tuple (ordering columns, token, primary key) compared
  lexicographically.
  - Every element up to the token compares as u64 from columns.
  - The primary key is compared only when everything before it is equal. That happens only for
    rows of one partition with equal ordering values, or on a token collision. The compare uses
    the keys from the table map: partition key, then clustering key, each column by its CQL type's
    order, with clustering columns reversed where the schema says `DESC`.
  - The heap holds the u64 prefix, the primary id and the document's address, and resolves the key
    lazily.
- **Table-order index:** it has no ordering columns, so its order is (token, primary key).
- **Unordered index:** it keeps no token or ordering columns, and its walk stops after the first
  `n` live matches, as the walk without ordering does today.
- **`primary_id`:** stays as the document's identity. It is no longer the tie-break.
- **Clustering order:** the node reads it once per index from `system_schema.columns`
  (`clustering_order`), because the driver's metadata leaves it out.
  - The order is fixed for the table's lifetime: `ALTER TABLE` cannot change it, nor the type of
    a primary-key column.
  - Dropping and recreating the table drops the index too.
- **Supported key types:** an ordered or table-order index accepts only primary-key column types
  whose order the node implements, and refuses others at creation. The first set:
  - integers, `varint`;
  - `text`, `ascii`, `blob`;
  - `boolean`;
  - `timestamp`, `date`, `time`;
  - `uuid`, `timeuuid`;
  - `inet`;
  - `float`, `double`;
  - frozen tuples of these.

  `decimal`, collections and UDTs come later if anyone asks. An unordered index takes any key.

## Bounds, layout and the rewrite

- **Segment bounds:** they become tuples of the u64 prefix: the ordering columns and the token.
  The minimum and maximum tuples are found by one scan of those columns when a segment's columns
  are first cached.
- **Merge policy and rewrite:** both compare tuples. The rewrite's quantile ranges are cut over
  tuples, so a run of rows tied on the first column can be split by the next column or the token.
- **Low-cardinality columns prune again:** a column such as a 1–5 rating gives segments that each
  span one value and a slice of what follows it. Today every segment spans the whole range and
  nothing prunes.
- **Pruning and ranges:** the segment skip compares against the full tuple. Range restrictions
  apply to the first column, as today.

## Cursor

The cursor carries the last row's ordering values and its primary key.
- **Wire:** the ordering values are a list in place of today's single sort value. A table-order
  index sends the key alone.
- **Resume:** the node computes the key's token and resumes strictly after the full tuple. It no
  longer looks up the key's `primary_id`.
- **Deleted cursor row:** resumes exactly. Today it takes the tied rows again (`TIE_UNKNOWN`).

## ScyllaDB

- **Choosing an index:** a query without `ORDER BY` is served by a table-order or an unordered
  index on the column. A query with `ORDER BY` is served by an ordered index whose shape matches,
  as described above.
- **Table-order index:** its queries go through the ordered path
  (`ordered_pattern_select_statement`). They page, and `LIMIT` is optional.
- **Unordered index:** keeps today's path, with the table-order sort of the page.
- **Changes against the current PoC:**
  - An ordered query without `ORDER BY`, which today returns newest first, is no longer served by
    an ordered index.
  - The option accepts several columns with directions, and the `unordered` keyword.
  - The paging state carries a list of ordering values.
- **Same rules on both sides:** the option syntax and the supported key types are checked at index
  creation by ScyllaDB and by the node, like the orderable sort types.

## Why keep `unordered`

Table order costs a write-heavy table on both sides:

- **Ingestion:** inserts arrive in random token order. Every commit therefore lands across the
  whole token range, and every row is rewritten once more by the rewrite. That is the
  shuffled-stream case, and it holds the writer lock for seconds per range (the known
  ingestion stall).
- **Queries:** until the rewrite catches up, every query walks all its matches in that fresh, wide
  part of the index (L0). An unordered index stops after the first `n` alive matches wherever they
  are, and keeps no sort or token column.

An index ordered by a timestamp does not have this problem: new rows land at one end of the
range, which is where newest-first queries read. Table order is the adverse case on every write.

So table order is the default, and `unordered` is for tables where a search needs some matches but
neither a defined order nor paging. Measuring both under a steady insert stream decides whether the
option stays.

## Tests

- **Unit, node:**
  - ties ordered by token and then by clustering key, in both query directions, with a `DESC`
    clustering column;
  - several ordering columns with mixed directions;
  - a null or empty ordering value leaving the row out, and a later value bringing it in;
  - a range that today returns null rows, written first to fail;
  - a cursor that resumes inside a run of ties, including on a deleted row;
  - tuple bounds;
  - rewrite ranges cut inside a run of one value.
- **cqlpy with the mock:**
  - the option's syntax, and the rejected shapes;
  - a query without `ORDER BY` on an ordered index is not served by it;
  - a table-order index pages through table order, and `LIMIT n` is a prefix of `LIMIT 2n`.
- **Local accuracy (SCT):** new corpora with qrels, with every page in order:
  - a column with few values (a 1–5 rating) on a table with a clustering column;
  - two ordering columns;
  - a table-order index.
- **AWS:**
  - query capacity for a table-order index, an unordered one and a timestamp-ordered one;
  - each under a steady insert stream of 2k and 5k rows/s;
  - the low-cardinality column.

## As implemented on the node

- **The sort prefix:** five u64 slots (`SortPrefix`): at most four ordering columns
  (`MAX_ORDER_COLUMNS`), then the token, in fast fields `sort_<i>` and `token`. Unused slots are
  zero for every row, so whole prefixes compare as arrays. Walking in the declared order
  complements the whole prefix, as the single sort key was complemented before.
- **Layout key:** the merge policy, the wide/spread checks and the rewrite work on a u128 made of
  the first two slots. For one ordering column that is (value, token), which is what lets the
  rewrite split a run of one value; for table order, the token.
  - A segment's bounds are exact. When its rows hold more than one first slot, the second slot at
    either end comes from one pass over the two columns when the segment is first opened.
  - Bounds widened to the whole second slot at either end, as first implemented, made the rewrite
    of a column with few values endless. A rewritten range crossing from one value to the next
    lands in a piece holding the end of one and the start of the other. Widened, the piece spanned
    two whole values and read as spread. Spread segments merge among themselves whatever they
    span, so these pieces merged into segments spanning up to four values. The rewrite cut those
    into crossing pieces again. A local run with five values (2026-10-30) rewrote 75–95k rows every
    50 s, for as long as it ran.
- **The walk:**
  - It keeps the page as a vector sorted best first. It compares prefixes, and only on an exact
    prefix tie looks both primary keys up in the table and compares them in table order
    (`KeyLayout::compare_same_token`).
  - The page's ids come from the primary-id fast column the walk reads anyway, so an exact page
    reads nothing from the document store.
- **Partition keys on a token collision:** they compare by ScyllaDB's legacy byte form. That form
  is also what the node hashes for the token, with the driver's Murmur3 partitioner. Tests pin
  both to known ScyllaDB tokens and to the driver's serialization.
- **Nulls:** a null or empty ordering value leaves the row out (`read_prefix` returns none, and
  the actor skips the add). A test pins that a range no longer returns such rows; with the old
  "null is key 0" encoding it returns `[5, 4, 2, 3, 1]` instead of `[5, 4]`.
- **The cursor:** it carries the prefix, so resuming needs no lookup. `TableSearch::primary_id`
  is gone.
- **Merge policy, a fix found on the way:** spread segments are merged in runs by lowest key, and
  a full segment between two small ones used to end the run. For spread segments, which each span
  most of the range, that adjacency means nothing, and the old layout passed the tests only by
  luck. Full spread segments are now left out before the runs are cut.
- **Checks when the index is created:**
  - the option's syntax: an unparsable `order_by` refuses the index rather than falling back to
    table order;
  - the ordering columns' types;
  - the table's partitioner, Murmur3 (`DbDriver::uses_murmur3`);
  - the primary-key types (`is_key_type_supported`);
  - the clustering order, read from `system_schema.columns` (`DbDriver::clustering_order`).

## Implementation order

1. Node: the option's syntax, null exclusion, the token column, the tuple order in the walk, the
   cursor, the clustering order from the schema, and the type check. Unit tests.
2. Node: the table-order index as (token, key); tuple bounds in the merge policy and the rewrite;
   the `unordered` keyword.
3. ScyllaDB: choosing an index by query shape, the option syntax, paging with a list of values, the
   type check; cqlpy.
4. SCT: the new corpora, then local accuracy.
5. AWS, together with the profiles and per-window CPU prepared after run `0216688f`.
