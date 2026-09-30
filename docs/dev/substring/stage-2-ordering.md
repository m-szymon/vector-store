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

- `ORDER BY <order_by column> DESC` or `ASC` accepted on a routed substring query, rejected
  otherwise. The existing `verify_ordering_is_allowed` guard, which rejects `ORDER BY` with
  secondary indexing, has an exemption for this case; the direction travels to the node as
  `order`, and the node walks the complement of every key for `ASC`. (Stage 2 shipped `DESC`
  only; stage 4 added `ASC`.)
- A range restriction on the sort column (`AND register_time < ?`) is the primitive; paging is a
  cursor expressed through it. Worth having in its own right, not only as a paging mechanism.
- The paging state carries the node's cursor as an opaque string: the last row's sort key and
  primary key. Rows sharing a sort key are ordered by internal id and resumed exactly; a cursor
  whose row is gone resumes at its sort key inclusive. (Stage 2 carried the sort key alone and
  skipped ties; stage 4 fixed it.) Offset paging is not extended.
- `LIKE 'keyword%'` and `LIKE '%keyword'` are routed like `'%keyword%'` (stage 4): the node
  frames every value with a start and an end mark, and an anchored query is containment of the
  keyword with the mark. The request carries `kind`.
- A keyword past `max_gram` on a case-sensitive index is checked by ScyllaDB, not the node
  (stage 5): the request says `verify: false`, the node answers with the candidates in sort
  order and `verified: false`, and the coordinator applies the pattern to the value it reads
  with the row. See "Stage 5" below.
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

## Stage 5: verification on the coordinator (branch `substring-index-stage5`)

Past `max_gram` the grams nominate candidates and the stored text decides, at a document-store
read per candidate that beats the page; two-pass verification (stage 2) keeps those reads to
the page's worth plus the false positives above it. ScyllaDB reads every returned row from the
base table anyway, so the check can happen there at no extra read.

How it works:

- ScyllaDB sends `verify: false` when the pattern's framed length (characters, plus one for the
  anchor mark of a prefix or suffix) is past the index's `max_gram` and the index is
  case-sensitive. The walk then takes the candidates as exact matches -- the cheap branch that a
  keyword within `max_gram` already uses -- and the page says `verified: false`.
- The coordinator fetches the target column with the row (a non-serialized selection column, so
  it does not reach the client) and drops the rows that do not hold the pattern, byte for byte,
  anchored as the kind says. That is the node's own test for a case-sensitive index.
- A page that comes back short is topped up from the node's cursor up to three times, then
  handed over short with a cursor; the LIMIT counts rows returned. A short page is a correct
  page to a paging client.
- The node's word is final: a page it reports verified (or an older node that never heard of
  the flag) is taken as it is.

The choice is an index option read by ScyllaDB only, `verify_candidates`: absent, ScyllaDB
checks whenever it can (a case-sensitive index with `order_by`) and the node otherwise;
`'index'` keeps the check on the node, which is what lets one run compare the two on one index
and one load (`aws_stage5_ab_config.yaml`); `'scylla'` insists on ScyllaDB and is refused at
`CREATE INDEX` where it cannot check. An index without `order_by` always leaves the check on the
node: it reports no cursor, so a page ScyllaDB left short could not resume.

The static rule is a first step. Where the check belongs depends on the candidates: many false
candidates, especially concentrated in one segment, are cheaper to reject on the node, which
reads each from a block it has open, than to ship to ScyllaDB as base-table reads and extra
round trips; a handful of candidates spread over many segments is cheaper to ship. A later
value of the option can let the node decide per query from what the walk saw (candidates per
segment, the share that failed on the pages it did verify) and report the decision in
`verified`, which ScyllaDB already obeys.

Why only case-sensitive indexes: a case-insensitive index lowercases values and keywords with
Rust's Unicode tables, which ScyllaDB does not share. Folding on the coordinator with different
tables would be the sort-key coupling all over again, silent on the exotic code points where
the two disagree. A case-insensitive index therefore keeps verifying on the node. A search box
over CJK names has nothing to fold, so such an index can be created case-sensitive at no cost
to it; a Latin search box that wants case folding keeps today's cost.

What it changes and what it does not:

- The document-store reads leave the query path for long keywords on case-sensitive indexes;
  the page is resolved from the store (or the FAST id column) as before.
- The index size does not change: the stored text is still there for the rewrite and for
  case-insensitive indexes. Dropping it is the second step, once this one is measured.
- False candidates cost a base-table read each on the coordinator instead of a store read on
  the node, and a page with many of them costs extra node round trips, bounded as above.

### Measured at 10M names with names up to 32 characters (AWS, 2026-09-29, run `697ea24b`)

Same corpus, plan (`aws_followup_config.yaml`) and machines as 2026-09-28, capped index
(100k rows a segment, 100 segments), `case_sensitive: 'true'` so that stage 5 takes the long
keywords -- which also makes this a different index from the 2026-09-28 one (see below).
Queries per second at 20 rows a page, ordered:

| keyword | 2026-09-28 (node verifies, case-insensitive) | 2026-09-29 (ScyllaDB verifies, case-sensitive) | store reads per query |
|---|---|---|---|
| 8 characters | 5.0k | 6.2k | 2.4 |
| 16 characters | 2.3k | 2.8k | 1.0 |
| 32 characters | 1.1k | 1.7k | 1.0 |

- The store reads left the query path: one per query is the page being resolved.
- The walk itself did not change: a rare keyword still opens all 100 segments, 0.6 ms of walk
  at 8 characters and 2.4 ms at 32. That was the next thing to fix, and stage 5 only took verification off it. The segment
  skip below does it for 16 and 32 characters.
- Short keywords got slower, not faster: 1 to 4 characters went from 9.8k-10k to 9.1k-9.9k per
  second, and the index node's walk for 1 and 2 characters from 92-119 us to 128-170 us. Just
  under the loader's fixed 10k rate is where queueing starts, so p99 went from 0.2-2.2 s to
  1.2-10.7 s on those rows. The cause is not established. The two runs differ in more than
  stage 5: this one's index is case-sensitive, and about 5.5% of the corpus's names carry
  uppercase Latin letters, so the index has more distinct grams (57.6 bytes per name against
  49.8) and the lowercase keywords match slightly fewer rows. The coordinator also builds its
  result differently for every query now. Telling the two apart needs either the stage-4 code
  on a case-sensitive index, or a latency phase at a rate below saturation.
- The deep 8-character page (window 0.5) went from 9.2k to the full 10k, with p99 from 9.4 s
  to 5 ms, because it moved from just above saturation to below it.

Latency caveat for every row above: the loader drives a fixed 10k queries per second with 64
in flight, and latte measures from the intended start time. Where a run cannot sustain 10k, the
queue grows for the whole 120 s and p50/p99 measure the queue, not a query. Only the rows at
the full 10k have latencies that describe the service time.

The second dataset, the backfill with the rewrite, did not complete: at the end of the full scan
(9.85M of 10M ingested) a status request to the index node and an SSH session to the ScyllaDB
node both timed out within the same minute, and the test stopped. Two nodes failing together
over different protocols points at the path from the laptop rather than at either node, but the
node's own logs were not collected, so the cause is not established. The rewrite at 10M remains
unmeasured.

### The segment skip (vector-store `d7c5901`)

A keyword past `max_gram` is an intersection of its `max_gram`-long grams. Before the walk opens
a segment it looks each of those grams up in the segment's term dictionary, and a missing gram
rules the segment out without opening its postings or its sort column. The gram that ruled out
the last segment moves to the front of the list, so a rare gram found once is tried first in the
segments after it. Nothing new is stored: the term dictionary is already there. Both walks, the
ordered and the unordered, do it.

The check finds each gram's term-dictionary entry, and a segment that passes has its postings
built from those entries. The first version kept only "present or not" and then opened the
segment through a `BooleanQuery` of `TermQuery`s, whose weight looked every gram up again and
opened a field-norm reader per gram that an unscored walk never reads. `benches/substring_probe.rs`
measures the two on 1M names of the 10M corpus in 16 segments of 66k (the AWS segment size), with
the query files of the AWS runs:

| keyword | segments opened | walk before | walk after | the check alone |
|---|---|---|---|---|
| 4 chars | 16 of 16 | 65 us | 62 us | 5 us |
| 8 chars | 9.2 of 16 | 38 us | 25 us | 8 us |
| 16 chars | 4.1 of 16 | 50 us | 33 us | 14 us |
| 32 chars | 1.2 of 16 | 43 us | 27 us | 16 us |

That is a third off the walk for 8 to 32 characters. At 4 characters the posting lists themselves
dominate. What is left scales with the segment count: at 152 segments the check alone would be
about 75-150 us, and the segments that hold every gram of an 8-character keyword without holding
it are opened all the same. Measured on AWS in run `7212926d` below.

### Measured at 10M names: the A/B run and the segment skip (AWS, 2026-09-29, run `4ce3f8b8`)

Plan `aws_stage5_ab_config.yaml`, same corpus (`names_10M_long`, 10 shards of 1M) and machines
(i4i.xlarge ScyllaDB, 4-core c8g.xlarge index node), 20 rows a page, ordered newest-first. Two
case-sensitive indexes on the same table, both capped at 100k rows a segment and both built while
the table loaded: `node_checks` (`verify_candidates: 'index'`) and `scylla_checks` (the default,
so ScyllaDB verifies past `max_gram`). Both caught up 407 s after the load. `node_checks` settled at
130 segments and 58.6 bytes a name, and `scylla_checks` at 152 segments and 59.0 bytes a name.
Each segment spans about 2% of the sort range.

Every query set was measured twice. The capacity run had no rate limit and 64 in flight, and its
p99 includes queueing inside the loader. The latency run had a fixed rate below capacity and 16 in
flight, so its p99 describes a query. Each run lasted 60 s. The walk column is the index node's
walk per query, and the store column is document-store reads per query (the page being resolved
counts too).

Capacity:

| keyword | scylla_checks q/s | node_checks q/s | walk us (S / N) | segments opened (S / N) | store reads (S / N) |
|---|---|---|---|---|---|
| 1 char | 9,578 | 10,100 | 206 / 179 | 2 of 152 / 2 of 130 | 20 / 20 |
| 2 chars | 9,381 | 9,406 | 158 / 137 | 2 / 2 | 20 / 20 |
| 2 chars, window 0.5 | 9,441 | 9,797 | 194 / 225 | 3 of 65 / 3 of 57 | 20 / 20 |
| 4 chars | 8,597 | 8,783 | 98 / 236 | 2 / 2 | 18.3 / 52.4 |
| 8 chars | 5,457 | 5,615 | 683 / 661 | 87 / 76 | 2.4 / 4.8 |
| 8 chars, window 0.5 | 10,275 | 10,436 | 331 / 325 | 39 of 65 / 35 of 57 | 1.6 / 3.2 |
| 16 chars | 5,874 | 5,850 | 616 / 622 | 28 / 28 | 1.0 / 2.0 |
| 32 chars | 11,913 | 11,484 | 268 / 280 | 2.3 / 3.0 | 1.0 / 2.0 |

In the capacity runs p99 was 7.6-19 ms and p50 5.3-11.7 ms.

Latency (p50 / p99 in ms):

| keyword | rate | scylla_checks | node_checks |
|---|---|---|---|
| 2 chars | 2,000/s | 1.20 / 1.81 | 1.12 / 1.66 |
| 4 chars | 2,000/s | 1.18 / 4.65 | 1.28 / 1.91 |
| 8 chars | 2,000/s | 1.67 / 3.48 | 1.61 / 3.38 |
| 16 chars | 1,000/s | 1.58 / 4.25 | 1.56 / 3.96 |
| 32 chars | 500/s | 1.49 / 2.42 | 1.55 / 2.84 |

Where verification runs does not change capacity: every row is within 4% between the two
indexes, and the indexes differ in segment count, so part of that is layout.

- **4 characters** is the only case where the choice matters. ScyllaDB verifying takes the node's
  walk from 236 to 98 us and its store reads from 52 to 18. But the pages come back short and are
  topped up, and p99 goes from 1.9 to 4.7 ms. Capacity stays the same, because at 4 characters the
  index node is not the limit: 1 to 4 characters all stop near 9-10k.
- **8 to 32 characters**: the candidates are the matches (2.4 postings for 2.4 rows at 8
  characters), so there is almost nothing to verify either way. The node only pays one extra
  store read per match.

For the dynamic choice this suggests the node should verify unless candidates are many and mostly
false, since that is where shipping them costs round trips. The static default (ScyllaDB verifies
on a case-sensitive index with `order_by`) saves node CPU but costs latency at 4 characters. It
is kept for now, and the per-query choice stays future work.

The segment skip, against run `697ea24b` the same morning (100 segments, no skip, ScyllaDB
verifying):

| keyword | 697ea24b q/s | 4ce3f8b8 q/s (scylla_checks) | walk us | segments opened |
|---|---|---|---|---|
| 8 chars | 6,190 | 5,457 | 598 -> 683 | 100 of 100 -> 87 of 152 |
| 16 chars | 2,829 | 5,874 | 1,373 -> 616 | 100 of 100 -> 28 of 152 |
| 32 chars | 1,668 | 11,913 | 2,353 -> 268 | 100 of 100 -> 2.3 of 152 |

- 32 characters went up 7x and 16 characters 2x: one of their grams is missing from almost
  every segment.
- 8 characters did not gain. Its grams are common enough that 87 of 152 segments hold all of
  them. It lost 12%, most likely because the index now has 152 segments instead of 100 (the load
  was 10 shards of 1M instead of 100 of 100k). This is not established.
- 8 and 16 characters over the whole range are now bounded by the index node's CPU: 620-680 us
  a query on 4 cores is about 6k/s, which is what was measured. The budget is 400 us. The
  remaining cost is the segments that do hold every gram but no match. Fewer, value-tight
  segments (the rewrite, still unmeasured at 10M) or a per-segment filter on gram pairs are the
  candidates.

Against the requirement (10k q/s, p99 under 100 ms), the requirement is met at 1-2 characters,
32 characters, and 8 characters over half the range. It is close at 4 characters (8.6-8.8k). It
is not met at 8 and 16 characters over the whole range (5.5-5.9k). The latency part holds
everywhere.

Raw results: `~/sct-results/20260929-151353-071783/` on the laptop that drove the run. The index
node's logs were collected this time (`collected_logs/`).

### Measured at 10M names: segment size (AWS, 2026-09-30, run `7212926d`)

Plan `aws_segment_size_config.yaml`, with the same corpus, machines and page of 20 as run
`4ce3f8b8`. It runs vector-store `a7fefd2`, so a segment that passes the gram check has its
postings built from the entries the check found. There are three case-sensitive indexes on one
table, all built while it loaded and all verifying on the node (`verify_candidates: 'index'`).
They differ only in the cap, 100k, 200k or 400k rows a segment, and settled at 107, 51 and 26
segments, at 58.2, 56.3 and 54.9 bytes a name. The three caught up 756 s after the load.

Capacity (unthrottled, 64 in flight), queries per second, with the index node's walk in us:

| keyword | cap 100k | cap 200k | cap 400k |
|---|---|---|---|
| 1 char | 9,900 (166) | 9,973 (308) | 7,522 (476) |
| 2 chars | 9,680 (125) | 9,731 (208) | 9,597 (309) |
| 2 chars, window 0.5 | 9,850 (197) | 9,628 (319) | 9,562 (322) |
| 4 chars | 9,258 (222) | 9,737 (186) | 9,717 (218) |
| 8 chars | 7,864 (454) | 12,094 (275) | 18,732 (169) |
| 8 chars, window 0.5 | 13,648 (241) | 23,401 (137) | 32,870 (89) |
| 16 chars | 7,538 (474) | 8,661 (404) | 11,520 (290) |
| 32 chars | 13,357 (244) | 10,282 (330) | 9,083 (380) |
| no match | 41,376 (78) | 64,669 (35) | 70,061 (15) |
| weakest | 7,538 | 8,661 | 7,522 |

At a fixed rate below capacity (16 in flight), p99 was 1.6 to 3.4 ms for every keyword length on
every index.

Reading the table:

- **The ~9.7k ceiling is ScyllaDB's row reads,** not the index. A 1- to 4-character keyword fills
  its page, so each query reads 20 base rows, which comes to about 195k rows a second on one
  i4i.xlarge. That is stage 1's ceiling, and it does not move with the cap. The query types above
  it return fewer rows because the table holds no more: about 2.4 at 8 characters, 1.6 on the
  deep 8-character page, 1 at 16 and 32 characters, and 0 for no match (the store reads per query
  are twice the rows, one to verify and one to resolve). Their pages are complete, not cut short,
  and for them the index node is the limit.
- **Larger segments help the rare long keywords.** 8 characters goes from 7.9k to 18.7k and 16
  characters from 7.5k to 11.5k, because fewer segments are checked and opened.
- **Larger segments cost the commonest keyword.** At 400k a 1-character keyword's page comes from
  two segments of 400k rows, whose postings are four times as long, and its walk reaches 476 us.
- **32 characters gets worse as segments grow** (13.4k, 10.3k, 9.1k). A larger segment is more
  likely to hold all ~30 of its grams, so more segments pass the check: 3.8 of 107 at 100k and
  12.4 of 26 at 400k.
- **200k is the flattest of the three.** Every shape is at 8.7k or above, and the weakest is 16
  characters. A cap between 200k and 400k may balance 16 characters against 1 character better.
  That is not measured.

The gain from `a7fefd2`, at the same 100k cap as run `4ce3f8b8` (`node_checks`, the same code
without the reuse): 8 characters went from 5,615 to 7,864 per second (walk 661 to 454 us), 16
characters from 5,850 to 7,538 (622 to 474 us) and 32 characters from 11,484 to 13,357 (280 to
244 us). Part of this is layout, since that index had 130 segments and this one 107. Per opened
segment an 8-character keyword went from 8.7 to 6.9 us, -21%, which is the reuse itself.

Raw results: `~/sct-results/20260930-075658-530733/` on the laptop that drove the run, with one
row per phase in `summary.csv`.

## Open questions and risks

- ~~**The sort-key encoding is written twice**~~ Closed (stage 4): a range bound travels as a
  value of the sort column (`min_sort_value` / `max_sort_value`, each a JSON value plus an
  `inclusive` flag), and the node maps it with the same `to_sort_key` it applies at ingestion.
  ScyllaDB no longer holds a copy of the encoding. What remains duplicated is the list of
  orderable types, checked at index creation on both sides; a disagreement there refuses an
  index rather than misordering one.
- **Ties are ordered by internal id, not by a CQL order.** The cursor resumes among tied rows
  exactly (stage 4), but the order among them is the node's and can change across a rebuild of
  the index; ordering ties by primary key would need the key per candidate in the walk.

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
