# Substring search, stage 2: ordering and paging

Status: design proposal. Every number below is measured unless marked otherwise; the benchmark is
`crates/vector-store/benches/substring_order.rs` and the measurements were taken on corpora of
500,000 and 2,000,000 synthetic names.

## What stage 1 gives, and what is missing

Stage 1 answers `WHERE col LIKE '%keyword%' LIMIT n` from an n-gram index, in unspecified order.
That lets the search stop walking as soon as it has `n` matches, which is why a keyword matching a
fifth of the corpus costs the same as one matching a handful. On AWS at 10M rows it served 2,000
QPS at p99 1.49 ms, and topped out around 9,600 QPS at p99 20.7 ms, bounded by ScyllaDB's
base-table reads rather than by the index.

What it cannot do is the query the feature exists for:

```sql
SELECT user_id FROM users
 WHERE nickname LIKE '%将军%'
 ORDER BY register_time DESC
 LIMIT 20;
```

Ordering changes the problem completely. The best 20 rows can be anywhere in the match set, so
"stop after 20" stops being available, and paging compounds it: every page must re-answer the same
question with a different cut-off.

### Budget

10,000 QPS against a 4-core index node is **400 µs of CPU per query**. That is the number every
option below is judged against.

## Why the obvious implementation is not enough

The obvious implementation is a `FAST` sort column and `TopDocs::order_by_fast_field`. It is
correct, and it is linear in the match count, because Tantivy prunes only when sorting by score:
`sort_by_score.rs` overrides collection with `for_each_pruning` (Block-WAND) and every other sort
key falls through to `default_collect_segment_impl`, a plain `for_each` over the whole docset.
Index-level sorting, which would have made this cheap, was added upstream for exactly this purpose
(#1026) and then removed (#2434).

Measured cost per matching document:

| path | cost per match |
|---|---|
| within `max_gram` (one term, no verification) | 13.5 ns |
| past `max_gram` (gram intersection, verification) | 580–7,700 ns |

The verified path is dear because membership is only known after reading the stored text, and
ordering cannot select the top 20 before membership is settled. At 2M names a keyword matching 20%
of the corpus costs **9.75 ms** unverified and **233 ms** verified. Extrapolated to 10M that is
roughly 27 ms and over a second — 60× and 2,500× over budget.

Deep paging is worse: with no pruning, page 50 costs what page 1 costs, so reading *n* pages costs
*n* times the whole match set.

## Rejected: verification from a `FAST` text column

The verified path's cost is store reads, so the natural fix is to put the normalized text in a
columnar field. Measured, it is **13× worse**: a flat ~7,700 ns per candidate against 580–5,600 ns
for the document store. The column is dictionary-encoded and every name is distinct, so each read is
an sstable seek into a dictionary with one entry per document, while the document store gets
*cheaper* as matches grow denser because the walk picks up block locality. At 400,000 matches that
is 3,088 ms against 233 ms, and the column adds 31% to the index.

Kept reproducible behind `SUBSTRING_BENCH_FAST_TEXT` rather than deleted, so the result can be
re-checked against a later Tantivy.

## The design

### Target state

**Segments whose span of the sort column is narrow.** Documents need not be sorted *within* a
segment — only the bounds matter, which is a far weaker requirement than the index sorting Tantivy
removed, and it is why this is achievable at all.

A segment is already an immutable, independently searchable unit with its own postings and columns,
and `Column::min_value()`/`max_value()` give conservative bounds for a `FAST` field without touching
the data. Given narrow segments, a query can:

1. Sort segments by `max_value(sort_column)` descending.
2. Take the ordered top-k within each, maintaining a global k-th-best threshold.
3. Stop as soon as the next segment's `max_value` cannot beat the k-th best held.

This is correct **whatever order documents arrived in**, because the bounds are always valid. Arrival
order affects only how tight the bounds are, and therefore how early the walk stops.

Measured with segments spanning 5% of the range (500k names, 160 segments):

| matches | plain ordering | segment-pruned |
|---|---|---|
| 5,000 | 257 µs | 109 µs |
| 25,000 | 661 µs | 117 µs |
| 100,000 | 2,093 µs | **147 µs** |

The pruned column is flat: cost tracks segments visited, not matches.

### Paging

The cursor is the sort key of the previous page's last row, so a page is "the best `limit` strictly
below it". That prunes from both ends — a segment whose lower bound is at or above the cursor holds
only rows already returned, and one whose upper bound cannot beat the k-th best holds nothing that
can enter the page.

Measured: page 50 costs what page 1 costs (104 µs vs 105 µs at 5,000 matches; 147 µs vs 147 µs at
100,000). Offset paging has neither property and is not proposed.

### An optimisation that needs no preconditions

Compare a candidate's sort key against the running k-th best **before** reading it from the document
store. The sort key is a columnar read and the store read is not, so rejecting early avoids the
expensive half. Measured on a corpus whose segments span the whole range and can never be skipped:
100,000 matches fell from 1.40 ms to 0.35 ms, and a verified query at 5,000 matches from 12.1 ms to
0.90 ms — **4–13×, with no dependency on segment layout.**

The same reasoning applies to the primary key. It lives in the document store too, and the first
stage-2 build read it for every row that entered the top-k heap — most of which a later, higher row
then pushed out again. On AWS that put the ordered page-1 ceiling at 6.0k op/s against stage 1's
9.6k, with the index node at 95% CPU; locally the walk with those reads was 9–27× slower than the
same walk without them (`segment_pruned` vs `store_on_entry` in the benchmark). The heap now holds
document addresses and the store is read once, for the `limit` rows that survive. A keyword past
`max_gram` still reads each candidate to verify its text; that read cannot be deferred.

## Reaching and holding the target state

### Segments do not become tight on their own

With ingestion following the sort column exactly, the mean segment still spanned **100%** of the
corpus. The writer deals documents round-robin across `num_worker_threads`, so each worker's segment
receives an interleaved sample of everything. Only a commit forces every worker to flush, so
**commit cadence, not insertion order, is what bounds a segment's range.** Committing every 25,000
documents brought the mean span to 5%.

### Merging cannot narrow a range

A merge is a range *union*, and there is no split API. So:

- Merging value-adjacent segments preserves tightness; merging distant ones destroys it
  **irreversibly**. The default `LogMergePolicy` merges by size tier and will silently ruin the
  property queries depend on, so replacing it is not optional.
- A segment that is already wide can only be repaired by rewriting its documents.
- "Small segments now, merge later" does not work: a segment of a randomly ordered stream already
  spans everything, and merging such segments keeps it so.

`MergePolicy` is pluggable via `set_merge_policy`, but `SegmentMeta` exposes only id, `max_doc`,
`num_docs` and delete counts — no fast-field statistics. A range-aware policy therefore needs a side
map `SegmentId → (min, max)` maintained by the actor, since "adjacent" is defined in sort-value space.

### The distribution pass

The only route from unsorted data to tight segments. Each output range is one pass that reads the
`sort_key` column and re-adds only the documents falling in it, then commits. Memory is **constant**
— documents are never grouped in memory, which for a randomly ordered input would need memory
proportional to the whole dataset — and each document is re-indexed exactly once across all passes.
The price is that every range scans the column again.

Measured on fully unsorted corpora:

| corpus | ranges | time | throughput | mean span |
|---|---|---|---|---|
| 500k | 20 | 1.3 s | 398k docs/s | 99.9% → 5.0% |
| 2M | 80 | 9.1 s | 219k docs/s | 99.9% → 1.2% |

Four times the data took seven times as long: re-indexing is linear, but column scans are
`ranges × documents`, and ranges grows with the corpus at fixed segment size. Extrapolated to 10M
that is roughly 2.5 minutes, most of it scanning. It does not extrapolate much further — at 100M a
single pass would spend hours scanning, and the fix is the standard one: fan out by a bounded factor
per pass and take `log` passes, so the cost becomes `fanout × documents × passes`. Not designed here,
since nothing at the current target needs it.

### The resulting shape is an LSM

- **L0** takes whatever arrives, committed as it comes. Wide ranges, unprunable.
- **L1+** is value-aligned and narrow, produced by distribution passes over L0.
- A query scans all of L0 plus a pruned L1+.

A large backfill is simply a very large L0; continuous updates to historic rows are a continuously
refilled one. Same machinery, and queries work throughout, degrading in proportion to L0.

Measured, with L0 scattered across the whole sort range (the hard case — a rewrite of historic rows,
rather than recent traffic which sits at the top of the range a newest-first query reads anyway):

| L0 | mean span | cost at 100,000 matches |
|---|---|---|
| 0% | 5.0% | 135 µs |
| 1% | 9.8% | 134 µs |
| 5% | 10.0% | 148 µs |
| 20% | 25.0% | 200 µs |

Degradation is gradual. L0 costs roughly 3 ns per matching document in it, so at 10M rows a hot
keyword matching 2M rows can afford a few percent of the corpus in L0 before the query passes the
400 µs budget. **That is the compaction SLA.** (Extrapolated; measured only to 2M.)

## Index options

```sql
CREATE CUSTOM INDEX users_nickname_sub ON users (nickname)
  USING 'substring_index'
  WITH OPTIONS = {
    'min_gram': '1', 'max_gram': '3', 'case_sensitive': 'false',
    'order_by': 'register_time',
    'segments': 'append',
    'segment_docs': '25000',
    'max_l0_fraction': '0.05'
  };
```

| option | meaning |
|---|---|
| `order_by` | the sort column. Absent, the index behaves exactly as stage 1. |
| `segments` | `append` assumes arrivals mostly follow `order_by` and relies on commit cadence alone; `rewrite` runs distribution passes over L0. |
| `segment_docs` | target documents per segment. Smaller prunes better and costs index size; see Open questions. |
| `max_l0_fraction` | how far compaction may fall behind before it is forced. |

The sort column must be immutable in practice (`register_time` is; `nickname` is not) — a mutable
sort column moves documents between ranges on every change, which is correct but multiplies
compaction work.

`segments` is a create-time decision: changing it later needs a rebuild.

## CQL surface

- `ORDER BY <order_by column> DESC` accepted on a routed substring query, rejected otherwise. The
  existing `verify_ordering_is_allowed` guard, which rejects `ORDER BY` with secondary indexing,
  needs an exemption for this case.
- A range restriction on the sort column (`AND register_time < ?`) is the primitive; paging is a
  cursor expressed through it. Worth having in its own right, not only as a paging mechanism.
- The paging state carries the last sort value. It does not carry a primary key, so two rows
  sharing a sort value are not separated across a page boundary; see the tie gap below. Offset
  paging is not extended.
- `substring_index::check_target` stays as it is; the sort column is an option, not a target.

## Implementation order

| | scope | precondition |
|---|---|---|
| **P0** | Sort-key check before the store read | none — ship with anything |
| **P1** | `order_by`, sort column as `FAST`, plain ordered top-k; `size_hint()` branch so keywords under ~2,000 matches take the cheap path | none |
| **P2** | Segment-bound pruning, cursor paging, range restriction | P1 |
| **P3a** | `segments: append` — commit cadence, range-aware merge policy, side map | P2 |
| **P3b** | `segments: rewrite` — L0 accounting, distribution passes, compaction scheduling | P3a |

P0 and P1 together give correct global ordering with no new concepts, and are honest about being
slow for hot keywords; that limitation must be documented rather than discovered.

**Status: P0, P1, P2 and a first P3a are implemented** in both repos and measured on AWS at 10M
names (below). P3b is not started, so an index built by a full scan of an existing table — every
segment spanning the whole range — is *correct* but not *fast*, and stays so until rewritten.

## Measured at 10M names (AWS, 2026-09-25)

Four runs on the sequential corpus, one `i4i.xlarge` for ScyllaDB and one 4-core `c8g.xlarge` for
the index node, 20 rows per page, 10k queries/s offered. Stage 1 (unordered, stops after 20
matches) reached 9.7k on the same nodes, bounded by ScyllaDB's base-table reads.

| build | page 1, 2-char keyword | deep page (cursor at 5M) |
|---|---|---|
| store read per heap entrant | 6,176 | 1,066 |
| store read for the final page only | 7,326 | 4,451 |
| same, second layout | 6,795 | 3,085 |
| primary id as a column (A/B, one load) | 3,896 / 7,100 | 4,317 / 3,135 |

The per-query counters and the per-segment layout report (`substring_search_*_total`,
`substring_segment_*`) turned the throughput into an explanation:

- **Reading the store per heap entrant** was the first bottleneck: 9–27× the walk locally, and
  deferring the reads to the final page was worth +20% on page 1 and 4× on the deep page.
- **The scan is cheap**: 8–16 ns per posting, as benchmarked. A deep page costs whatever its
  segments hold — 76k postings, 1.2 ms, when the cursor lands in a 1.3M-row segment.
- **Resolving the page is cheap either way**: ~20 µs from the store, ~40 µs from a `FAST` primary
  id column. The column is not worth its bytes; it stays available behind `poc_option_1` only.
- **What capped page 1 was a fixed ~450 µs per query**, constant across 1–8 opened segments and
  2k–85k postings: every query opened every segment's sort column to read its bounds, and a column
  open costs time proportional to the segment's size (2.7 µs at 77k rows; seven 1.3M-row segments
  on a Graviton core add up). Fixed by caching the columns per segment (`columns_for`).
- **The layout the default merge policy produces** with CDC ingestion: ~20 segments, mean span
  8–9%, seven contiguous 1.3M-row slices and a tail of small segments that all reach the top of
  the range, so page 1 opens six to eight of them. Two indexes fed the same stream got different
  tails (one had a 1.7M-row segment reaching the top), which alone made page 1 cost 2×. Ingestion
  order does not give a layout; only a policy does.

### P3a as implemented

`poc_option_2` = the segment cap in rows. An ordered index with it installs `RangeMergePolicy`:
segments sorted by lowest sort key, runs of neighbours merged while under the cap, a segment at
the cap left alone. The side map of bounds the note called for is filled after every reload from
the column cache, and the actor asks the writer for the merges the policy would choose, since
Tantivy evaluates merges right after a commit, before the new segments' bounds are known; idle
ticks reload the reader so merges finishing after the writes stop become visible. Write
amplification on the tail (rewritten every commit, up to the cap) is the known cost; a levelled
tail is the refinement.

Measured the same evening, three indexes over one 10M load (default policy, 250k cap, 100k cap):

| | default | cap 250k | cap 100k |
|---|---|---|---|
| segments, mean span, widest | 16, 10.8%, 35.7% | 41, 3.3%, 5.2% | 100, 1.0%, 1.0% |
| caught up after the load | 15 s | 15 s | 15 s |
| page 1, 2-char keyword | 10.2k/s, 28 µs, 2 segments | 9.6k/s, 111 µs, 2 | 9.8k/s, 81 µs, 1 |
| unthrottled | — | 9.7k/s | 10.1k/s |
| deep page, cursor at 5M | — (3.1k–4.5k on earlier layouts) | 9.75k/s, 168 µs, 15k postings | **9.85k/s, 76 µs, 3k postings** |
| 1-char keyword | — | 10.0k/s | 10.0k/s |
| 4-char keyword (verified) | — | 9.6k/s, 112 store reads | 5.8k/s, 207 store reads |

The column cache alone took page 1 to the offered rate on the default layout; the policy makes
that independent of luck and gives the deep page the cost of page 1, which was the design's
claim. Neither cap slowed ingestion measurably at 5k rows/s. The ceiling is ScyllaDB's base-table
reads again, above stage 1's unordered 9.6k on the same nodes.

What remained was the verified path. In a single narrow newest segment scanned in ascending sort
order every later candidate beats the page's weakest entry, so all of them were read before the
heap could reject any -- 207 store reads for a 4-char keyword at the 100k cap. The walk now runs
two passes per segment for a keyword past `max_gram`: it gathers the candidates' sort keys from
the column, then verifies from the highest down and stops at the first that can no longer enter
the page, so the store is read for the page's rows plus the false positives above them (a unit
test: 30 matching rows, a page of 5, 5 verification reads). Not yet measured at 10M.

### P3b as implemented (stage 3, not yet measured)

`poc_option_3` turns on the rewrite of wide segments; it needs the cap. Three widths matter,
all measured in caps' worth of contiguous rows over the sort range:

- **Narrow**: a segment spanning at most two caps' worth. The merge policy folds narrow
  neighbours into runs, under the cap in rows and no wider than two caps. The pieces a commit's
  writer threads leave (a fraction of the commit's rows each, over the commit's whole key range)
  are narrow however many threads there are, since a commit is far smaller than a cap.
- **Spread**: anything wider than that, or over the cap. It never joins a narrow run, which
  would only widen it; spread segments merge among themselves with no limit on the span, so
  slivers grow into something the rewrite will find wide.
- **Wide**: spanning more than eight caps' worth -- a full-scan build, a shuffled load -- or over
  the cap, which no merge can bring under it. Wide segments holding half a cap or more are what
  triggers a plan.

On an idle tick (no rows pending; a plan made while a build is still committing samples the rows
so far and cuts too few, too wide ranges) with at least a cap's worth of rows in wide segments,
the backend samples their sort keys, splits the key space into ranges of about a cap each by
quantiles, and records a plan of ranges. Every idle tick then moves one range: under the writer
lock it commits, reads from the store each live row of a segment *that is spread at that moment*
whose key falls in the range, deletes it by primary id and re-adds it, and commits again. The
lock is what keeps a concurrent update or delete of one of those rows from being undone by the
re-add; searches take the reader and never wait. The range's rows land in pieces spanning no
more than the range, narrow by construction, which the policy folds into one segment. Progress
and the rows involved are on `/metrics` (`substring_rewrite_*`, `substring_l0_docs`), and the
plan, its end and every merge request are in the log.

Why "spread at that moment" rather than the wide segments the plan found: the policy retires
segment ids within a tick, so a plan naming them found them gone (789 rows planned, 19 moved in
the first docker smoke); and since ranges move in ascending key order, what remains in the
drained segments is a shrinking band of the highest keys, which stops counting as wide once it
spans under eight caps and would escape with a cap's rows over several caps' width, at the top
of the range where newest-first looks. In-process, thirty caps' worth of shuffled rows built in
one commit or in scan-sized commits end as one segment per cap's worth of keys, plus a narrow
remainder at the top.

The cost model is the design note's distribution pass: each range scans the spread segments'
sort column once (`ranges × rows` column reads) and every row is read from the store and
re-indexed once. Unmeasured at 10M; the plan for that is the `names_10M_backfill` dataset of
`aws_variants_config.yaml`, an index created after the load with the rewrite on against one
without.

### Measured at 10M names (AWS, 2026-09-27, one run, six index variants)

i4i.xlarge Scylla, c8g.xlarge vector-store (4 cores), 20 rows per page, 10k queries/s offered
with 64 in flight; the walk figures are the node's own counters per query. Four indexes ingested
the same load through CDC (all caught up 15 s after it); two were created on the loaded table
and built by full scan (136 s and 166 s).

| variant | segments / mean span | char2 page 1 | deep page (window 0.5) | char1 | char4 |
|---|---|---|---|---|---|
| default policy | 16 / 10.9%, widest 35.8% | 10.0k/s, 57 us | **8.5k/s, 402 us** | 10.0k/s, 76 us | 10.0k/s, 118 us, 40 reads |
| P3a cap 100k | 100 / 1.0% | 10.0k/s, 91 us | 9.5k/s, 92 us | 9.9k/s, 113 us | 10.0k/s, 134 us, 40 reads |
| P3a cap 250k | 41 / 3.1%, widest 5.0% | 10.0k/s, 44 us | 9.5k/s, 162 us | 10.0k/s, 51 us | 10.0k/s, 116 us, 40 reads |
| cap 100k, single-pass verification | 100 / 1.0% | 9.8k/s, 95 us | 9.7k/s, 78 us | 10.0k/s, 117 us | **5.5k/s, 667 us, 223 reads** |
| backfill, left wide | 23 / 100% | **1.5k/s, 2.6 ms** | **0.9k/s, 4.2 ms** | **1.1k/s, 3.6 ms** | not run |
| backfill, cap 100k + rewrite | 136 / 2.2% (see below) | 9.9k/s, 75 us | 9.3k/s, 136 us | 10.0k/s, 94 us | not run |

What it settles:

- **Segment balancing (P3a) is what makes deep pages cheap**: 402 us on the default layout
  against 92 us at the 100k cap, the one phase where the default falls under target. The 250k
  cap halves the short-keyword walk (44 vs 91 us: fewer segments to consider) at the price of a
  deep page that scans a bigger segment (162 us). Both meet the target; 100k is the safer choice
  for paging, 250k for page-1 throughput.
- **Two-pass verification is worth 4x on long keywords**: same layout, same 207 postings,
  40 store reads instead of 223, 134 us instead of 667, and 10.0k/s instead of 5.5k/s.
- **An index built after the load is unusable for ordered queries without the rewrite**: every
  segment spans the whole range, page 1 scans 300k postings per query and runs at 1.5k/s. With
  the cap and the rewrite the same build answers at 9.9k/s.
- **The rewrite's cost at 10M**: the plan of 100 ranges took 5 minutes, one range every 3 s,
  moving all 10M rows once (a store read and a re-index per row) while the queries were not yet
  running, and left 126 segments of one cap each. The build itself was no slower for the cap.
- **The rewrite did not converge on its own on this run.** The drained band's remains were
  sparse, and a plan over them cut ranges by row count that spanned half the key space; the
  pieces those rows landed in were as wide as the range, so the index kept planning 5 ranges of
  ~450k rows and moving ~48k. The queried layout had four such 21k-row pieces at 52.5% (mean
  span 2.2% all the same) and the loop's CPU under the queries; the numbers above are with it.
  Fixed after the run by bounding a range to one cap's worth of the key space (commit
  `a6cfcf7`), not yet measured at 10M.
- **Index size**: 34.7 bytes per name at the default policy, 35.8 at the 100k cap, 43.0 for
  the rewritten backfill while its loop kept deleted rows around.
- **p99 under a fixed offered rate**: the phases that reach exactly 10k/s show p99 of 12 to
  104 ms; any phase that falls short of the offered rate queues without bound in latte and its
  p99 runs to seconds. Read capacity from the throughput column, not p99, wherever throughput is
  under 10k.

### Measured at 10M names with names up to 32 characters (AWS, 2026-09-28)

The same cluster shape and the 100k cap, on a corpus a tenth of whose names are 11 to 32
characters (`generate_local_dataset.py --long-names`), with keyword sets of 8, 16 and 32
characters each cut from a different long name, so each matches that name and rarely another.
Ingestion caught up 15 s after the load as before; 49.8 bytes per name against 35.8.

| keyword | queries/s | walk us | segments opened | postings | store reads |
|---|---|---|---|---|---|
| 2 chars, page 1 | 10.0k | 92 | 1 | 4849 | 20 |
| 2 chars, deep page | 10.0k | 112 | 1 | 4841 | 20 |
| 1 char | 10.0k | 119 | 1 | 8017 | 20 |
| 4 chars | 9.8k | 158 | 1 | 469 | 40 |
| 8 chars | **5.0k** | **750** | 100 | 3 | 6 |
| 16 chars | **2.3k** | **1688** | 100 | 1 | 2 |
| 32 chars | **1.1k** | **3641** | 100 | 1 | 2 |
| 8 chars, deep page | 9.2k | 370 | 50 | 2 | 4 |

The design's argument that long keywords get cheaper (fewer candidates to verify) holds for the
candidates and misses the segments: a keyword matching fewer names than a page never fills the
page, so the walk opens every segment the bounds allow, and in each one looks up all of the
keyword's grams and builds the intersection before finding nothing. The cost is opened segments
times grams, about 120 us per gram over 100 segments, linear in keyword length, and it is the
price of the cap: the same keyword on the 16-segment default layout would cost a sixth.

The fix, not yet built: a per-index map from each gram to the set of segments containing it,
rebuilt on every reload from the segments' term dictionaries (about a million grams times 100
bits, a few megabytes, milliseconds to build). A query past `max_gram` intersects its grams'
sets first and opens only the segments that survive: at most as many as the keyword has
matches. The short-keyword path, which fills a page from the first segment, is untouched.

The same run was to re-measure the rewrite with ranges bounded to a cap's width; it was lost to a
network outage on the runner's side during the second load, so that stays at "reproduced and
fixed in-process, unmeasured at 10M".

## Open questions and risks

- **The sort-key encoding is written twice**, in `cql_types.rs` and in `index/substring_index.cc`,
  and the two must agree bit for bit: the node stores the key that ScyllaDB produces a bound for, so
  a disagreement would filter on one ordering and sort by another, dropping rows from the middle of
  a result rather than raising an error. The list of orderable types is duplicated the same way.
  Both couplings go away if the request carries typed values and the node converts them, which needs
  a JSON encoding for typed CQL values.
- **Ties are not separated across a page boundary.** The cursor is a sort value alone, so rows
  sharing one are taken or skipped together. A tie-break key in the cursor fixes it.

- **Write amplification under sustained historic rewrites** is unmeasured. One distribution pass is
  characterised; steady state is not, and it is what sizes `max_l0_fraction` on a rewrite-heavy
  table.
- **`segment_docs` is a guess.** Smaller segments prune better but cost index size — measured at
  **+49%** going from 8 to 160 segments over 500k documents — and lengthen the walk for rare
  keywords, which must visit more segments to fill a page. Justifying a value needs the match-count
  distribution of real two-character CJK substrings, which is corpus analysis rather than
  benchmarking.
- **Ingestion throughput under frequent commits** is unmeasured; the AWS run managed 5.5k rows/s
  with the current cadence.
- **Nothing is measured above 2M rows.** The 10M figures here are extrapolations, and the verified
  path is known to degrade with corpus size (the same 100,000 matches cost 54 ms at 500k names and
  94 ms at 2M, as the document store outgrows cache).
- **A rare keyword opens every segment the cap made.** Measured above: 5.0k/s at 8 characters,
  1.1k/s at 32, against the 10k target. The gram-to-segments map above is the answer; without
  it the cap trades long-keyword throughput for deep-page cost.
- **Replacing the merge policy is load-bearing.** Getting it wrong does not fail loudly; it silently
  widens segments until pruning stops working.
