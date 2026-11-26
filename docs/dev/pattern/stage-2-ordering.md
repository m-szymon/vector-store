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

`poc_option_2` = the segment cap in rows. An ordered index with it merges by `merge_runs`:
segments sorted by lowest sort key, runs of neighbours merged while under the cap, a segment at
the cap left alone. The bounds come from the column cache after every reload, and the actor asks
the writer for the merges on every tick, since Tantivy evaluates merges right after a commit,
before the new segments' bounds are known; idle ticks reload the reader so merges finishing after
the writes stop become visible. The tick is the only decider: the writer's own policy is
Tantivy's `NoMergePolicy`. A `RangeMergePolicy` that chose the same runs after each commit, next to
the tick that knew only its own merges, had each run started twice (2026-11-02), and a tick that
did not know its own merges still running queued each again on every tick until a 10M-row index's
node ran out of memory (run a0e51aa9); `MergesInFlight` keeps a run waiting while one of its
segments is being merged. Write
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

### Two tiers and a continuous repair (2026-11-03; replaces the plan of P3b)

The widths above told the rows still to be placed from the ones already placed by a segment's
span, and that guess failed at the boundary: a remainder of a drained segment holding a few rows
of ranges not moved yet looked narrow, merged with placed pieces into a near-full segment no
step drained, and segments overlapped for good. A local run with a few-values column ended
overlapping two runs in eight; a 10M load in table order settled at 435 segments. The layout is
now two tiers, told apart by where the rows came from rather than by their span:

- **Ordered**: segments a repair step placed (every row carries `repaired = 1`, a fast field of
  ordered indexes), and segments that overlapped no other when first seen, which are where they
  belong whoever placed them: rows inserted in sort order once merged, a build by full scan in
  table order. Those are adopted, in memory, as is what a merge of ordered segments produces
  (the merge's future names its output); once adopted a segment stays ordered, so fresh rows
  arriving over its stretch later are rows to repair into it, not a reason to rewrite it.
- **Fresh**: everything else, the pieces commits leave.

No merge mixes them. Ordered merges take overlapping segments (a *cluster*: the pieces of one
repair step, or a segment with the rows later repaired into its stretch) whole, alone up to the
cap, with their neighbours while within three quarters of it, so that fresh rows fit beside a
segment before it has to be split; neighbours have no ordered segment between them, so the tier
stays disjoint. Fresh segments merge in runs by lowest key under the cap, full ones left out. A
segment whose deleted rows outnumber its live ones is merged on its own, which drops them.

The repair is continuous, worst first. The ordered clusters cut the key space into stretches
(each cluster's own, and the gaps between); a stretch is as bad as the fresh rows in it,
estimated from a sample of the fresh tier's keys, and a cluster past the cap counts its own
rows. A step takes the worst stretch, at most a cap of its fresh rows from its lowest keys, and,
under the writer lock as before, deletes and re-adds them as repaired: in a gap they make a
cluster of their own, in a cluster's stretch they overlap it and merge into it. Rows that would
not fit beside the cluster under the cap are moved with the cluster's own, in parts of half a cap
over disjoint key ranges, one commit each.

When a step may run depends on what brings the rows. During a full scan, which builds the index
and serves no queries, no step runs: the scan gets the node, and the repair then works from every
row at once (the scan counts as ended five ticks after its last row). After it, an idle tick
always takes a step, so an idle index converges to an empty fresh tier and a disjoint ordered
one; while the index ingests, a step waits until the fresh tier holds 400,000 rows, and each of
its holds of the writer then waits for a place among the worker pool's background tasks, behind
the searches and the live writes of every index: a burst of writes grows the fresh tier, which
searches pay for with an open per fresh segment, rather than the lag. (Until 2026-11-25 a step
ran only while indexing tasks had waited under 1 ms on average in the worker's queue since the
last tick; see "The repair takes its turn in the pool" for what replaced that and why.)

The searches say where the overlap costs them. An ordered search counts the segments it opened
that put no row on its page (`pattern_search_segments_wasted_total`), and the last 4096 that
wasted any are kept with the layout key their page ended at. A stretch's fresh rows counted up to
nine times as much in the repair's choice where those searches went, so that the stretches
searched most were repaired first. That is out for now: its benefit was never measured. The one
measurement made (10M shuffled rows into an empty index, searches kept on the middle, the newest
or the oldest end) could not show any, as there is one stretch until ordered segments exist. The
case it is for is a settled index under steady scattered writes with searches away from the
lowest keys; it is to come back as an index option unless measured never to make a case worse
(searches that move to another place, searches spread evenly).

Locally, 10M rows by full scan in table order (4 cores, `a_fast_loaded_table_order_index_settles_and_stops_when_dropped`):

| | rewrite plan (`8e5e26f`) | repair during the scan, at most half of the time | repair after the scan |
|---|---|---|---|
| load | 67 s | 124 s | **49 s** |
| settled after the load | ~310 s | ~300 s | **~200 s** |
| segments, depth | 56, not measured | 78, 1 | 65, 1 |
| rows moved | | 14.7M | 9.8M (the rest adopted where they were) |
| RSS peak, after the drop | 3.1 GB, 1.4 GB | 2.3 GB, 1.7 GB | 2.1 GB, 1.2 GB |

`/metrics` has `pattern_rewrite_steps_total`, `pattern_rewrite_docs_total` and
`pattern_rewrite_pending` (an index has settled once nothing is pending and no merge runs), and
the layout's `pattern_layout_fresh_rows` and `pattern_layout_fresh_segments` beside its overlap
depth. In-process: the few-values tests are stable (eight runs of eight); shuffled loads end
with no fresh rows and depth 1; deletes and updates during the repair are kept, an update
moving a row's sort value takes it to its new place, and most rows deleted compact to a few
segments with no deleted row left.

Reading a sorted segment from the end the page needs (`33d3fe3`): an ordered walk read every
posting of each segment it opened, since postings come in doc order, not sort order, so a page of
20 for a dense keyword scanned the whole top segment (AWS run 7a300b9d: 7.3k postings for char2 in
table order). The repair adds its rows in key order; the writer threads each take an ordered share,
so a commit's pieces are sorted runs, and an ordered merge, given its segments in key order, keeps
them. A segment is cut into its runs of non-decreasing layout key when its columns are opened (at
most 512, 1024 doc ids each on average, or read whole), and the walk reads each run from the end
its page needs, from the cursor or the range's top, in stretches of 64 doc ids doubling, until the
run can no longer enter the page. A segment with fewer than 4096 matches is read whole: each
stretch going down opens the postings again. Locally, 10M rows at the 200k cap
(`the_cost_of_a_page_read_by_runs_and_whole`, the same index read both ways):

| keyword | newest | ascending | deep |
|---|---|---|---|
| dense (every row) | 16 us (whole: 710) | 23 us (729) | 28 us (2553) |
| one in nine, verified | 434 us (1705) | 39 us (535) | 427 us (1702) |
| sparse, miss | unchanged | unchanged | unchanged |

A 1M cap cuts a miss from 22 to 5 us (17 segments instead of 73) but costs a verified keyword
3.3-4.2 ms a page, as a 1M-row segment has hundreds of runs and each stretch reopens the
intersection; the 200k default stays until a run can be walked with one open.

The walk now reads a segment's runs in passes of one postings open each (`1fd9342`): the stretches
a page needs lie in doc order, the first as wide as the page needs at the segment's density of
matches, and a second pass, four times wider, only for the runs that can still give the page a
row. With it a larger cap no longer costs a page: at 10M rows and a 1M cap (14 segments) a dense
page is 18-61 us, one in nine verified 108-171 us, a miss 3-4 us (13-17 us at the 200k cap's 72
segments). What a repair step holds the writer lock for no longer grows with the cap either
(`ababb2b`): a step moves its rows in key ranges of at most 50k, each under a hold of its own, and
the repair under writes starts at 400k fresh rows. A 10M build by full scan settles ~210 s after
the load at the 1M cap, ~280 s at the 200k one; the 1M cap peaks about 1 GB higher (2.7 GB for one
index).

#### A page over a matrix of cases, and the walk as a merge by key (2026-11-17)

Every number above is of one corpus, whose keywords are spread evenly over the order, loaded
shuffled and repaired. `the_cost_of_a_page_over_a_matrix_of_cases` (ignored; `MATRIX_STATE`,
`MATRIX_CAP`, `MATRIX_ROWS`) asks how flat a page's cost is rather than how low in one case: twelve
keywords (in every row; in one of 5, 9, 1000, 10000, spread evenly; in the top tenth of the order
only, the bottom tenth, and one in a thousand of that; past max_gram, verified; one whose grams a
third of the rows hold apart and one row in 3000 holds together; none), pages of 20 newest first,
ascending and from the middle, and of 200 and 1000, over an index settled at either cap, settled
after a tenth of the rows were deleted and a tenth written again, ordered by a column of five
values, loaded in the index's order 2000 rows a commit (`stream`), and not repaired at all.

What it found in the walk as it was (`1d89737`), at 10M rows:

- A keyword past max_gram read seven times the postings of one as frequent within it (22.7k
  against 3.3k at the 1M cap, 1 ms a page): the first stretch was sized by Tantivy's estimate of
  an intersection, which takes the grams for independent, and the grams of a pattern the values
  hold come together. It is now sized by the rarest gram's rows.
- A candidate that is not a match cost 1.5 us, 68% of it decompressing store blocks: the reader
  cached one block, and candidates taken in key order alternate between the runs the writer's
  threads leave over one key range. With 32 blocks, 0.55 us. Telling a miss from the stored bytes
  before building the document saved a tenth and cost a match two reads: not kept.
- A segment may be in the ordered tier and in no order inside. One of 56k rows, adopted for
  overlapping nothing, held 6251 runs and was read whole (1.6 ms for an ascending page, 19 us on
  the build before it). And rows that arrive in the index's order were all such: a 150k-row
  segment of the `stream` state held 142k runs (why is below: the index is declared descending,
  so the keys of rows arriving in ascending order go down). A newest-first page of a keyword in
  every row cost 0.7 ms at the 200k cap and 5.9 ms at 1M, on the most common shape there is --
  rows stamped with the time, read newest first.
- A page cost a stretch of every run of the segment, so more at a larger cap (62 us against 23 for
  a keyword in every row), and its candidates were verified as each pass gathered them, before
  the rows that would push them out were read.

The walk is now a merge of a segment's runs by key (`walk_runs`). A run is sorted (the key never
decreases over at least 256 rows, a block's worth) or coarse: the rows between sorted runs, in blocks of 256, each
block with the lowest key from it to the run's end and the highest from the run's start to it.
Neither bound ever decreases, so a coarse run is read from either end as a sorted one is, a block
at a time, and rows in no order at all make one whose bounds are all about the whole of it, read
whole. A pass has a floor: the highest of the keys the runs with the highest unread keys would
reach by reading a stretch each. Every run reads its rows from the floor up, so a run all under it
reads nothing, and after the pass what the page holds over the highest key left unread is final.
(The candidates of a pass were verified only over that key, the rest waiting for the rows that
might push them out. On one index, with that off and on by turns, no cell of the matrix differed
in time; it saved store reads on an unrepaired index only, 49 against 103 a search at most. It
is out, to be measured again where the store is not in memory.) Of a stretch only so many matches are offered as
a page or two needs, four times as many each pass that found some: going up the stretch ends with
the last one taken, going down -- the postings come from the far end -- it is halved towards the
near end until what is left holds no more. A block of a coarse run that cannot enter the page by
its own rows' keys is not read. After a pass that met no match, one pass of the postings in doc
order tells the runs that hold none, which are left out. A segment with few matches for its runs
(256, four pages, and eight for each run the walk would read together: one where the runs follow
one another, all of them where each holds rows in no order) is read whole, in one pass.

Walk time for a page of 20, median and highest over the keywords and the three page shapes but
for the miss and the one with false candidates (us; `walk_matrix.sh`, 4 cores):

| state | before | now |
|---|---|---|
| settled, 200k cap, 10M | 26, 607 | 30, 250 |
| settled, 1M cap, 10M | 120, 1035 | 19, 283 |
| five values, 1M cap, 10M | 103, 1115 | 11, 385 |
| stream, 200k cap, 2M | 373, 2917 | 39, 185 |
| stream, 1M cap, 2M | 2152, 27074 | 47, 246 |
| not repaired, 1M cap, 10M | 20949, 152454 | 2223, 181660 |

- The highest of a settled or stream index is now the sparse verified keyword: 40 store reads,
  each of a block of its own, about 5 us apiece. A keyword in every row reads 43 postings a page
  at either cap (720 and 3161 before), a page of 1000 costs 0.4-0.55 ms (0.85 and 2.5 ms), and a
  larger cap no longer costs a page anything.
- Not repaired, a first page from either end is 0.26-0.42 ms for a keyword in every row (105-132
  ms): once the page is full most blocks of shuffled rows cannot enter it. A page from the middle
  still reads every match (180 ms): every block straddles the cursor.
- False candidates are a third of what they were and no flatter: 11 ms a page of 20 at a thousand
  to a match (44-69 ms), 0.5 s a page of 1000, and 1.5 s where the matches lie at the far end of
  the order (5.9 s). The cost is the candidates the grams let through, each a store read; what
  cuts their number is in the index -- `max_gram` 4, a third more index (run `a7be7f76`), or
  positions.

Every page of five keywords, followed from its cursor both ways and within a range, is checked
against the rows a full read gives on a repaired, a stream and an unrepaired index
(`pages_hold_the_same_rows_whatever_the_runs`).

Through ScyllaDB (`local_lag_config.yaml`, 1M names, three indexes, 4 threads; walk us and
postings a query, `1d89737` against this walk):

| phase | 200k cap, before | now | 1M cap, before | now |
|---|---|---|---|---|
| table order | 26, 575 | 16-33, 55-600 | 73, 3002 | 18, 59 |
| table order, 5000 writes/s | 136, 2138 | 87-113, 162-716 | 256, 4520 | 101, 320 |
| tier order | 36, 842 | 20-24, 53-71 | 91, 2955 | 18, 37 |
| time order | 34, 289 | 39-78, 375-1532 | 226, 2712 | 30, 228 |
| time order, random writes | 89, 289 | 147-199, 377-1533 | 382, 2713 | 129, 232 |

The ranges at the 200k cap are three runs of the same plan with the same walk. In the one whose
segments were logged (debug logging gives each sizeable segment's runs) the newest segment was 7
sorted runs of 6-12k rows, written by a repair step, and a page cost 375 postings, about what it
did before. The other two were not logged; 800-1500 postings is what the `stream` state reads for
a keyword this dense, the matches of about one commit, so the likely difference is rows a writer
added in order and the index adopted as they lay rather than repaired -- which the walk before
would have read whole. If so, sorting a commit's rows before they are added, or repairing an
adopted segment in no order inside when the index is idle, would close it; neither is done.

Rows that arrive in the index's order (2026-11-18). They were read as a coarse run, a commit's
worth of matches a page, for two reasons, neither of them what the paragraph above guessed:

- An index declared `DESC` keeps the complement of the value, so rows arriving in ascending order
  lie with their keys going down, and a run was a stretch whose key never *decreases*: every row
  was a run of its own. A stretch whose key never increases is now a sorted run too, reversed: the
  walk counts a run's rows from the end its keys are lowest at, and a coarse run whose keys go
  down on the whole has its blocks counted from its last doc ids.
- Each addition and removal is a task of its own on the worker, and the workers take them at
  once, so a few in a hundred reached the writer ahead of the one before, which cuts a sorted
  stretch into pieces too short to be runs. They now take a turn when the index's actor hands
  them over and the writer applies them in that order (`tantivy::Writer::apply`). That also
  closes, in this branch, what upstream closed in `5030ae4` by holding the removal back: a row
  written again quickly is removed at its old id soon after being added at it, and the removal
  running ahead left the old copy in the index. (An update's add has a new id, so the removal of
  the old one running behind it takes nothing with it.)

With both, each of the writer's threads makes a sorted run of its share of a commit, a quarter of
it on four threads, which is a run from 1024 rows a commit. The `stream` state now commits as
production does (every second or at 10000 rows; it committed every 50 ms, shares too short to
show any of this): its segments hold 36-40 sorted runs at the 200k cap and 254-292 at 1M, and a
page of 20, in the median and at the highest over the keywords and shapes, costs

| cap | coarse (10k commits) | sorted either way |
|---|---|---|
| 200k | 101, 654 us | 16, 262 us |
| 1M | 69, 2850 us | 21, 331 us |

A keyword in every row reads 40-70 postings a page (2600-2800). Loading 10M rows takes 46-47 s
as before (44-55 s). Through ScyllaDB (`local_lag_config.yaml`, 200k cap, one run at `8214a19`)
the time order, whose newest rows a writer adds in order, costs 24 us and 47 postings a query
(34 us and 289 at `1d89737`, 39-78 us and 375-1532 in the three runs before this change), 123 us
under random writes (89; 147-199), and the table and tier orders 17 and 25 us (26 and 36). A page of 1000 of such a keyword is 1.5 ms there against 0.4-0.5 ms on a
settled index: the page is a sorted vector, and the rows of runs that lie over one another go
into its middle one at a time. Not done: gathering a pass's rows and merging them in once.

Known costs: rows that arrive out of sort order are indexed twice, once by the ingest and once
by the repair, as they were by the rewrite; and fresh rows repaired into a full stretch rewrite
the stretch's rows too, at most every quarter of a cap of inflow.

#### A budget for a request (2026-11-23)

The matrix left one row out of line: a keyword whose grams a third of the rows hold apart
(`甲乙丙之乙丙丁` for `甲乙丙丁`). The index can tell such a row from a match only by reading its
stored value, so a page costs every such row above its last one: 11 ms for a page of 20 on 2M
rows, 365 ms for a page of 1000. The cost has no bound in the size of the index, the request
holds one of the node's workers for all of it, and it runs to its end after ScyllaDB has given
up on it.

A request now has a budget of time, `VECTOR_STORE_PATTERN_SEARCH_BUDGET`, 50 ms by default, a
setting of the node (the workers it keeps free are shared by the node's indexes); `0s` is no
limit. A request reads it as it starts, so a reload of the configuration changes it for the
requests that follow. When the budget ends:

- The walk stops, and the page is cut at the highest layout key it has not examined. The rows
  over that key are final and are returned. The cursor is that key with no row, which takes
  every row at it again (the cursor of a row that is gone already worked so).
- The page may hold fewer rows than asked for, or none, and is still not the last. ScyllaDB goes
  on from the cursor: a paged query for half of the statement's timeout, then it hands the client
  the page as it is; an unpaged one until its LIMIT is met or the timeout fails it.

Scenarios:

1. `LIKE '%ab%'`, a page of 20, 50 us. The budget is never looked at twice.
2. The keyword above, a page of 1000 on a repaired index: 8 requests of at most 50.4 ms, 398 ms
   in all, against 365 ms in one request.
3. A keyword with candidates everywhere and no match: request after request returns no rows and
   a cursor further down. A paged client gets an empty page with more to come after half the
   timeout, and the scan ends over as many pages as it takes, where it timed out before.
4. An unpaged query that cannot be filled in time fails at the timeout, as before; the node no
   longer works on after it.

Where the walk looks at the budget: before a segment, between the passes over a segment's runs,
and once in 32 stored values verified. What keeps the cut right and a row of requests moving:

- **A coarse run.** Its blocks are read in doc order, so after a pass the block with the highest
  keys may be unread: the cut is that run's key, not the candidate the verification stopped at.
- **Overlapping segments.** The walk reads a segment at a time, so the segments after a cut may
  hold rows over it. Each is read over the cut only, with a share of half the budget, and a stop
  there raises the cut. Where the first segment's rows lie among another's it gets half the
  budget itself, or it reads far down while the others get hardly under the cursor and the cut
  is theirs. A budget that ends between two segments has the rest read the same way, rather than
  the page given up.
- **Progress.** The walk stops only under the key of the cursor the request came with. Rows
  sharing a layout key -- the first two slots -- are read by one request, however many.
- **Unsorted stretches.** A coarse run, or a segment with no runs, settles nothing until all of
  it is read, and the next request reads it again. So the verification is not ended before it
  has cost, since the segment was last read, what the reading did. Without this a request read
  for all of its time and verified 32 values: a page of 20 over two unrepaired segments of 1M
  rows took 311 requests and 35 s.

Measured (2M rows, cap 1M, `MATRIX_BUDGET_MS=50`, against the same build's run without it):

| index | page | requests | longest | total | one request, no budget |
|---|---|---|---|---|---|
| repaired | 200 | 3 | 56 ms | 120 ms | 112 ms |
| repaired | 1000 (666 rows) | 8 | 55 ms | 408 ms | 365 ms |
| stream | 1000 (666 rows) | 8 | 52 ms | 381 ms | 370 ms |
| two unrepaired segments | 20 | 2 | 246 ms | 281 ms | 227 ms |
| two unrepaired segments | 1000 (666 rows) | 5 | 245 ms | 792 ms | 407 ms |

The other cells of the matrix are within 5-8% of the run without the budget, which is in the
range two runs of one build differ by.

One cell was not: a pattern nothing matches (2026-11-24). The walk looked at the clock before
every segment, and such a pattern leaves each of its segments after one look in the dictionary,
which costs about what the clock does. On 10M rows in about 70 segments its walk took 22-23 us
with the budget against 17.5-19 us without (two builds each), and on AWS 58 us against 49 (run
`0818814c` against `a2b5883f`). The budget is now looked at only before a segment that has
candidates: 19-21 us.

What the budget does not bound: a request over a large unsorted segment takes that segment's
reading and as much again, as the two unrepaired segments of the table show. The next section
is what left such segments out.

#### A commit written in the order of its keys (2026-11-23)

Tantivy numbers a segment's documents in the order they were added. Rows that arrive in no
order -- a restore, a backfill by partition key, writes with sort values all over the order --
made segments in no order: one coarse run, of which nothing is settled until all is read.

A pattern index's documents are now held until the commit and written then in the order of
their sort prefixes, in groups Tantivy writes into one segment each (`IndexWriter::run`): the
rows dealt to the groups in turn, as many groups as the writer has threads, none under 1024
rows. A fresh segment is then a set of sorted runs, one per group, and after its merges still
is: about 660 runs in a segment of 1M rows. A removal takes a held row out of the held ones,
since a delete only reaches documents written before it.

With hundreds of runs lying among one another, the first stretch of a pass is divided among
the runs whose bounds overlap the highest one's; it had been a page's worth from every run.

Measured on 2M rows that arrived in no order (`MATRIX_RANDOM=1`), repair off, two segments of
1M, against `MATRIX_OFF=sorted_commits`:

| cell | as they came | in key order |
|---|---|---|
| keyword in every row, page of 20 | 0.69 ms, 27392 postings | 0.30 ms, 1220 postings |
| keyword in every row, page of 1000 | 13.4 ms | 1.5 ms |
| 1/9 of the rows, verified, page of 20 | 1.7 ms | 0.64 ms |
| 1/1000 of the rows, page of 20 | 0.07 ms | 0.37 ms |
| median of the matrix | 1 | 0.59 |
| longest request with a budget of 50 ms | 2314 ms | 69 ms |

The states the matrix loads in its own order (settled, stream) are within 5% in the median.

The row that got worse, a keyword in one row of a thousand, is the walk looking at each of
660 runs where it had read the segment whole: the threshold for reading whole counted one match
a run. Both readings by turns on that index (`MATRIX_WHOLE=1 MATRIX_SPARSE=1`, 2026-11-25), a
page of 20:

| keyword in one row of | by its runs | whole | as chosen since |
|---|---|---|---|
| 3000 | 0.34 ms | 0.02 ms | 0.02 ms |
| 1000 | 0.29 ms | 0.04 ms | 0.04 ms |
| 300 | 0.24 ms | 0.12 ms | 0.13 ms |
| 100 | 0.20 ms | 0.47 ms | 0.20 ms |
| 9 | 0.31 ms | 3.3 ms | 0.32 ms |

A look at a run costs what about eight matches read in a pass do, and only the runs read
together count: where the runs follow one another (the stream and the settled index) that is
one or a few of them, and the threshold there was right as it was. So it is 256 matches, four
pages, and eight for each run the walk would read together (`Runs::overlap`).

#### More runs a pass for a keyword in few rows (2026-11-25)

Where the runs follow one another -- rows that arrived in order, a repaired index -- a pass
read one run: the next run's rows all lie under the floor the first one sets. A keyword in few
rows needs many runs for a page, and each pass makes a reader of the postings: 271 passes for a
page of 200 of a keyword in one row of 10,000 on 2M rows, 0.6 ms of scanning for the three
grams of `用户42号` and 2.7 ms for the pattern `用户42号%`, beside 1.5 ms of verification that
reading the segments whole pays too (the walk's probes, `MATRIX_WHOLE=1`).

A run its stretch reads to the end now sets no floor, for so many runs a pass (`spread`), and
the number grows after a pass that met matches and fewer than a page: by as many times as the
page is short at that rate, four at most. A pass that met none does not count, since the sweep
has by then told the runs that hold none; nor one that met a page's worth, whose keyword is in
many rows. Unconditional growth was tried first: a keyword whose rows are all at the far end of
the order then cost 1.2 ms for 0.05, the walk arriving there sixteen runs at a time.

With and without by turns on one index (`MATRIX_WHOLE=spread`), 2M rows, walk time:

| keyword in one row of 10,000, verified | without | with | read whole |
|---|---|---|---|
| `用户42号`, page of 200, stream | 2.40 ms | 1.83 ms | 1.62 ms |
| `用户42号%`, page of 200, stream | 4.36 ms | 1.81 ms | 1.63 ms |
| `用户42号%`, ascending page of 20, stream | 0.65 ms | 0.27 ms | 0.18 ms |
| `用户42号%`, page of 200, settled | 3.74 ms | 2.22 ms | 2.07 ms |
| `用户42号`, page of 20, settled | 0.28 ms | 0.23 ms | 0.23 ms |

21 of the 285 cells of five runs are 20-60% faster, all of these two keywords; five are
6-10 us slower (1.25-1.7x), each of a keyword that meets a page's worth in its first pass and
so reads as before, and each measured just after the turn that read its segments whole. Rows
in no order are as they were: their runs are read together already.

#### Ordered neighbours are not merged across fresh pieces (2026-11-25)

Two ordered clusters were neighbours, and merged when they fit, with no ordered segment
between them, whatever fresh rows lay there. The merged segment spans those rows, and every
repair step among them splits it again. A row committed alone at one end of the order was the
case first seen (the stall of 2026-11-19); the same merge is in the log of a load without one,
between the first stretches repaired. A run of ordered clusters now ends where a fresh piece
reaches between two of them, by the pieces' bounds. During a load in no order that is every
gap, so the stretches a repair leaves are merged once the fresh tier is empty, not as they
come; a cluster still merges whole at any time, which is what takes the rows repaired into a
segment's stretch.

10M rows in no order, cap 1M, to the last merge (`the_repair_under_searches_kept_on_one_place`,
which now goes on until the merges have ended; `HEAT_MERGE_ACROSS=1` for the merges as they
were), one build, one run each:

| | merged across | not merged across |
|---|---|---|
| settled | 293 s | 211 s |
| settled, with the stray row | 278 s | 193 s |
| rows the repair moved | 15.3M, 15.1M | 10.0M, 10.0M |
| rows the merges wrote | 116M, 109M | 121M, 116M |
| segments at most | 375, 326 | 317, 298 |
| segments once settled | 15, 14 | 14, 14 |

The merges left for the end take about 6 s after the last repair step. Under a steady stream
of writes all over the order there is always a fresh piece over every gap, and neighbours do
not merge until it stops; the clusters there come from splits, half a cap each, so few are
small. Not measured: such a stream over days, with deletes thinning the ordered segments.

#### The order of partitions is an option (2026-11-24)

AWS run `0818814c` showed where table order stops: `char2` in table order or `tier` order
served 9,400 q/s with one ScyllaDB shard at 100% and the node at 49%, against 17,000 q/s in the
time orders with every shard near 90%. The profile of the node has every read of a row on shard
0, and the tablet map has the ring's first tablet there. The first page of a frequent pattern in
token order is its matches with the lowest tokens, all inside that tablet (1/64 of a 10M-row
table holds more than a page of them), so every first page of every such pattern is read from
one tablet's replicas, on any cluster and whatever the keyword. `tier` order has the same first
page: rows that tie on the column fall back to table order.

What a client needs from table order is less than the order of a scan: the rows of a partition
together and in clustering order, and one fixed order, so that pages resume exactly. The token's
slot of the sort prefix therefore holds what the `partition_order` option says:

- `scattered` (the default): the token through a fixed one-to-one mixing function
  (`order::scattered_key`, the finalizer of SplitMix64). Neighbours in the index are partitions
  from unrelated places of the ring, so a page's rows come from as many tablets as it has
  partitions. The index needs to know nothing of the topology, and a split, merge or move of
  tablets changes nothing: places taken at random on the ring are spread under any layout.
  One-to-one, so the tie-break on a shared slot is the token collision it was.
- `token`: the token, the order of a scan and of a filtered `LIKE`. For a client that needs
  that order, and it keeps what the default gives up: a full scan delivers the rows already in
  the index's order.

Nothing else differs: the cursor, the layout key, the merge policy and the repair work on the
slot's value, whichever it is. ScyllaDB validates the option and does not read it.

Page cost, 2M rows without ordering columns, loaded by token as a full scan loads them,
budget 50 ms (`MATRIX_PARTITION`, logs `walk_matrix_part_*`): the build with the repair took
58 s for `token` and 64 s for `scattered`, both to 3 disjoint segments; settled, the 60 cells
are the same within noise (page of 20: 0.02-0.07 ms both; of 1000: 0.5-1.3 ms both; the
worst keyword's page of 20, 8.8 against 11.0 ms). Before the repair a `scattered` index is the
unrepaired index of the other orders: a page of 20 costs 0.15-0.4 ms, a verified page of 1000
8-9 ms, the worst keyword's page of 200 twenty requests. The throughput on a cluster is not
measured yet.

At 10M rows the build is another matter (2026-11-25, the same load, cap 1M, logs
`walk_matrix_part10m_*`): 113 s for `token`, to 15 segments, and 374 s for `scattered`, to 14.
By token the rows arrive in the index's order and its segments are adopted as they are, with
nothing to repair; scattered they arrive in no order, and the build is the load of 10M rows in
no order measured above, about 90 s of indexing and 200 s of repair, each row written twice.
Settled, a page of 20 costs the same (median 12 and 21 us over the matrix, one build each). So
an index created on a large table pays for the spread of its reads with a build three times as
long. What would take that away is rows dealt to a commit's segments by stretch of the order,
not in turn: the keys of both orders are spread evenly over 64 bits, so the stretches are
known before any row is, and the segments of one stretch from commit after commit would merge
into ordered ones with no repair. Not started.


#### A build beside the searches of other indexes (2026-11-24)

The pool of workers has three classes since this date (`worker.rs`): searches, indexing (the
live writes, served first so that the lag holds), and background, for the writes nobody waits
for: the rows of a full scan, building an index that does not serve yet. Before, a build's rows
were indexing, ahead of the searches of every other index for as long as it lasted. Background
tasks run when neither of the other classes waits, and otherwise up to a minimum share of the
workers' time (`VECTOR_STORE_MIN_BACKGROUND_SHARE`, 10%), so that a build under a steady load
still ends. The three scheduling settings follow a reload of the configuration.

Three things the first version got wrong, each found by a test or by the benchmark
(`a_build_beside_the_searches_of_another_index`: a settled index of 1M rows searched, a second
of 1M rows built by a full scan, one pool of 4 workers):

- **One row's changes in one queue.** A row of a full scan that replaces an older value is a
  removal and then an add. The removal's guard did not say it was of the full scan, so the two
  would have waited in different queues and could have run in the other order. The guards made
  from a full-scan row's guard keep the mark. A change of the row read from CDC was safe
  already: the monitor holds it until the row's add is applied.
- **Room is told to the sender, not waited for in the channel.** Every class comes through one
  channel; a background task waiting in it for room would have held up the searches and the
  live writes sent after it. The pool queues it at once and answers its sender when the queue
  has room.
- **At most half of the workers run background tasks at once.** They all take their index's
  writer, and a commit holds it for about 30 ms: uncapped, every free worker took one in a
  moment with no search waiting and sat behind the commit.

Measured, 8 clients with 5 ms between their searches, the build beside them (single runs
except where two are given; the laptop was shared, so differences under 10-15% are noise):

| Build's rows as            | searches against no build | wait for a worker | p90     | p99      |
| -------------------------- | ------------------------- | ----------------- | ------- | -------- |
| indexing (before)          | -25%                      |                   | 15 ms   | 27 ms    |
| background, no cap         | -20%, -20%                | 1.5 ms            | 2.4 ms  | 25-27 ms |
| background, capped         | -8%, -8%                  | 0.5-0.6 ms        | 1.7 ms  | 18-21 ms |
| and the lock fix (below)   | -6%, -7%                  | 0.18 ms           | 1.6 ms  | 5.3 ms   |

Without the build: p90 1.0 ms, p99 1.6 ms. With the workers saturated (8 clients, no pause) the
searches kept 11% more of their throughput as background than as indexing, and the build took
50 s against 34 s (8 s alone); with 2 clients nothing changed, the searches 29% slower either
way.

**The lock that slept 100 ms.** The p99 left after the cap was one worker asleep: an indexing
task of 100.4 ms after nearly every commit of the build. A reload of the reader takes the
directory's meta lock to open the segments; Tantivy's removal of unused files takes it in every
commit and after every merge; and in the RAM directory the lock is a file, which Tantivy waits
for by sleeping 100 ms between tries (`directory.rs`, since its #482 of 2019; a file so that
another process can read an index being written, and on any platform). The reload starts when
the commit writes the meta, just before the removal, so the two often meet, and whoever comes
second sleeps: the reload on a worker, with two more on the build's commit and none left, or
the removal inside the commit, which holds the index's writer.

So the indexes live in a directory of our own (`tantivy/directory.rs`): the RAM directory's
files, with a meta lock that is a mutex and a condition variable and wakes whoever waits as it
is released, as Tantivy's file system directory does with an operating system lock. Counted in
the directory during the build of 1M rows beside searches (10 s, twice): with the RAM
directory's lock 89 and 85 takes of the lock waited, 8.9 s and 8.5 s in all; with this one 1
and 2 takes waited a millisecond or more, under 10 ms in all.

It is not this index's doing, as first thought. With the merges this index asks for switched
off, the reloads of a 1M-row build found the lock held more often, not less (77-80% of them
against 66-68%, and for 0.64 ms against 0.25 ms): it is the removal after each commit, and it
takes longer the more segments there are. A full-text index of 1M documents, committed as in
production, finds it held for 76% of its reloads on this branch. On upstream's master
(`8568c14`) the same build has 13 of its 134 reloads take 100 ms, in 5.1 s: one commit in four,
and a worker asleep for a quarter of the build. Upstream's unit tests do not show it (3 of 127
reloads), since they index a few documents.

The crate's unit tests take 8.5 s since, from 37.

**Each search runs longer beside a build: the cores it shares (2026-11-25).** Beside a build
each search ran 20-35% longer on its worker, and two threads that only compute, in the build's
place, cost 2-3%. The runs were pinned to CPUs 0-7 of the laptop, which are four cores of two
hyperthreads each: two searches and two computing threads have a core each, and a build, with
its two workers and Tantivy's indexing and merge threads, puts a thread beside a search on one
core. Measured with 2 clients, two runs each (`build_smt.sh`):

| CPUs                                | beside                | a search on its worker | searches |
| ----------------------------------- | --------------------- | ---------------------- | -------- |
| 0-7: four cores, two threads each   | the build             | +33%, +33%             | -29%     |
| the same                            | six computing threads | +90%, +89%             | -42%     |
| six cores, one thread of each       | the build             | +8%, +8%               | -12%, -11% |
| eight cores without hyperthreads    | the build             | +32%, +25%             | -28%, -24% |
| the same                            | two computing threads | +16%, +18%             | -14%, -15% |

On a core of its own a search loses 8% beside the build, what the two threads copying memory
cost (7-9%): the caches and the memory, not the allocator or the runtime's threads. The cores
without hyperthreads are the laptop's efficiency cores, which share a cache in fours and slow
down together: two computing threads alone cost 16-18% there. So the figure is the machine's,
and on a node whose vCPUs are whole cores (the c8g of the AWS runs) about 8% is what to
expect, with the build's threads then competing for the cores themselves when all are busy.

#### The repair takes its turn in the pool (2026-11-25)

A repair step runs on a thread of its own, because one hold of the writer lasts most of a
second and a worker is a task of the runtime. That left it outside the pool's policy, and what
held it back while the index ingested was a signal read on the tick: no step if indexing tasks
had waited more than 1 ms in the queue on average. Now each hold of the writer first takes a
place among the pool's background tasks (`worker::Place`, `repair_place`): a worker lends it
in the turn of a background task and runs nothing until the hold ends, while the work stays on
the repair's thread. So a hold waits for the searches and the live writes of every index as a
build's rows do, counts into the cap on background tasks and into their share, and the signal
is gone: a step is due on an idle tick, or once the fresh tier holds 400,000 rows.

Measured on the laptop, 4 workers on CPUs 0-3, single runs (`sched_chain.sh`; runs of one
benchmark differ by up to 20% from day to day):

| 3M rows written shuffled, searches beside | before | place in the pool | new gate, no place |
| ----------------------------------------- | ------ | ----------------- | ------------------ |
| 8 searchers: rows indexed in              | 98 s   | 63 s              | 57 s               |
| search p50 / p99                          | 6.9 / 50 ms | 2.9 / 275 ms | 2.4 / 278 ms       |
| repair steps, fresh rows left             | 9, 1.02M | 22, 0.41M       | 25, 0.38M          |
| 32 searchers: rows indexed in             | 490 s  | 384 s             | 338 s              |
| search p50 / p99                          | 14 / 52 ms | 8.2 / 42 ms   | 7.9 / 39 ms        |
| repair steps, fresh rows left             | 23, 0.39M | 22, 0.38M      | 21, 0.34M          |

What changed the figures is the gate, not the place: the old signal alone held the repair on
69 of 86 ticks with 8 searchers, and the fresh tier it left made every search and so every
write slower. The p99 with 8 searchers is the repair done inside the run instead of after it:
22 steps in a minute, each with holds that park the writes of the index, and the workers they
run on, on the writer. A 10M-row repair with a row written every 10 ms took 195 s against 197,
with the rows searchable within a second 87% against 91% (89% without the place); the repair
of 10M rows under searches 242 s against 265. Neither is outside the noise.

**A change that finds the writer held is left for its holder.** That p99 was the writes of
the index waiting for its writer on the workers that took them, for as long as a hold lasted,
until every worker waited and the searches with them. A change (an add, a removal, a commit)
now goes into a queue of its index, and whoever has the writer applies the queue in order:
the change's own worker if the writer is free, else the holder as it lets go
(`Index::change`, `WriterGuard` in `tantivy/mod.rs`). With 65,536 changes waiting, one more
waits on its worker as before, so that a reader faster than a long hold is still slowed. The
same loads, before and after (`queue_chain.sh`; 8 searchers twice):

| 3M rows written shuffled, searches beside | writes wait on workers | left for the holder |
| ----------------------------------------- | ---------------------- | ------------------- |
| 8 searchers: rows indexed in              | 62 s, 58 s             | 60 s, 57 s          |
| searches done                             | 31,700, 29,000         | 45,000, 44,100      |
| search p50 / p99 / longest                | 2.6 / 267 / 898 ms, 2.5 / 278 / 840 ms | 3.1 / 20 / 87 ms, 2.9 / 18 / 45 ms |
| 32 searchers: rows indexed in             | 367 s                  | 365 s               |
| search p50 / p99 / longest                | 8.4 / 36 / 912 ms      | 8.6 / 39 / 94 ms    |

A 10M-row repair with a writer of a row every 10 ms took 192 s against 197, and took in
17,284 of the writer's rows instead of 1,801: the writer itself used to wait out each hold.
Their time until searchable is the same (within a second 92% against 89%). A build beside
the searches of another index is unchanged (p99 3.4 ms against 3.8).

**The cap on background tasks is kept, for another reason.** It was made for the rows of a
build, which waited on their index's writer behind a commit and took every worker. They no
longer wait there, and for them the cap changes nothing (`cap_chain.sh`, three runs each, the
build beside the searches of another index):

| 8 clients          | cap on: searches, p99, build's rows sent in | cap off |
| ------------------ | ------------------------------------------- | ------- |
| 5 ms between searches | 1180-1203 a second, 3.3-3.7 ms, 7.6-7.9 s | 1186-1189 a second, 3.2-3.3 ms, 7.7 s |
| no pause           | 7671-7801 a second, 3.5 ms, 40.0-40.4 s     | 7704-8266 a second, 3.4-3.6 ms, 38.5-41.0 s |

What it bounds now is the places lent to the repairs' holds: each keeps a worker idle for
about 0.3 s here, and without the cap the repairs of several indexes take every worker at
once. Measured (2026-11-26, `repairs_chain.sh`, twice each): four indexes of 1M shuffled rows
built and then repaired at once, beside 8 clients searching a fifth, 5 ms between searches,
4 workers; the searches while the four repair:

| cap | searches a second | p50 | p90 | p99 | longest | wait for a worker | repaired in |
| --- | ----------------- | --- | --- | --- | ------- | ----------------- | ----------- |
| on  | 1044, 1057        | 0.6 ms | 2.1, 2.0 ms | 6.7, 7.8 ms | 18, 43 ms   | 0.18, 0.24 ms | 31.1, 30.6 s |
| off | 746, 731          | 0.6 ms | 6.2, 6.5 ms | 75, 70 ms   | 443, 466 ms | 3.8, 4.1 ms   | 29.1, 27.5 s |

Alone the searches do 1214 a second with a p99 of 1.2 ms. The cap costs the repairs 2-3 s of
30, and their holds wait 0.2 s for a place instead of 0.01-0.02 s.

**Searches beside four builds at once (2026-11-26).** While the four indexes above were
built, before any repair, the searches of the fifth had a p99 of 15-21 ms and a longest of
77-157 ms, against 4 ms and 10-18 ms beside one build. Three things looked at
(`builds_chain.sh`, and the benchmark's trace):

- **It is the cores, mostly.** The pool had 2.1 of its 4 workers busy on average, but its
  threads got 1.8 cores where they wanted 2.7: four writers and their merges share the same
  four CPUs. On 16 CPUs, still with 4 workers, the p99 is 4.4 ms and the searches lose
  nothing (1236 a second). A background task is not interrupted, so work started while no
  search waited is still on a core when the next search comes.
- **A commit kept its runtime thread's last wake-up waiting.** A search that took 69.5 ms had
  neither waited in the pool nor run long: it ended beside a commit of 69.0 ms. A worker is a
  task of the runtime; the request it had just answered was woken into the slot of the
  worker's own thread, which no other thread takes from, and the worker's next task was the
  commit. A commit now hands the thread's tasks over first (`off_the_runtime`,
  `block_in_place`): p99 12.0-14.3 ms against 16.5-20.6 with four builds (three runs and
  two), p90 2.8 ms against 3.5; nothing changes with one build. The longest searches stay
  at 66-91 ms, not explained.
- **The tick's commit of a build as background work: no gain.** It is indexing, so ahead of
  the searches and outside the cap. Sent as background it changed the p99 from 14.7-15.8 to
  13.3-13.6 ms, within the noise, and the repair of a single build then took 21 s instead of
  17-18. Left as it was.

**The budget for indexing stays at zero.** With `VECTOR_STORE_INDEXING_QUEUE_BUDGET` at
100 ms, searches first unless a write has waited that long, the same load with 32 searchers
took 643 s to index instead of 384, and the searches gained nothing: as many a second, p50
10.6 ms against 8.2. With 8 searchers nothing changed (58 s against 63). The indexing queue
is short, twelve tasks on four workers, so that a CDC reader waits on its sends; a write let
through after 100 ms is followed by ones that have only just been queued, and the rate of
writes under a full query load falls towards a queue's worth every 100 ms. That is the lag of
AWS run 5df052ac again, which the indexing class was made to stop.


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

### Measured at 10M names: the port onto upstream's pattern index (AWS, 2026-10-27, run `eb0023ac`)

Plan `aws_port_config.yaml`: the first run of the `pattern-index-poc` branches (scylladb
`45236f6c70`, vector-store `1b279cb`), with the corpus, phases and machines of run `7212926d`,
case-sensitive and verifying on the node. Two indexes ingested the load: `default` (no options:
the port's 200k cap, the rewrite on) and `cap_300k`. Both caught up 498 s after the load and
settled within 31 s, at 51 and 34 segments; 51 is what `7212926d`'s 200k index settled at.

Capacity (unthrottled, 64 in flight), queries per second, with the index node's walk in us, and
`7212926d`'s 200k index for comparison:

| keyword | default (200k) | 7212926d, 200k | change | cap_300k |
|---|---|---|---|---|
| 1 char | 8,651 (390) | 9,973 (308) | -13% | 9,037 (217) |
| 2 chars | 8,828 (258) | 9,731 (208) | -9% | 8,399 (150) |
| 2 chars, window 0.5 | 8,687 (380) | 9,628 (319) | -10% | 8,626 (306) |
| 4 chars | 8,800 (193) | 9,737 (186) | -10% | 8,339 (165) |
| 8 chars | 10,667 (313) | 12,094 (275) | -12% | 14,635 (221) |
| 8 chars, window 0.5 | 21,754 (147) | 23,401 (137) | -7% | 28,013 (110) |
| 16 chars | 7,237 (490) | 8,661 (404) | -16% | 8,268 (418) |
| 32 chars | 8,142 (423) | 10,282 (330) | -21% | 7,578 (462) |
| no match | 59,276 (33) | 64,669 (35) | -8% | 59,114 (20) |

At a fixed rate below capacity (16 in flight), p99 was 1.8 to 3.6 ms on both indexes, against
1.6 to 3.4 ms in `7212926d`.

- **The port measured 7-21% below the stage-5 code, and the difference is not attributed to the
  code.** Per query the two did the same work: the same 51 segments, the same segments opened,
  the same postings (21,135 against 21,022 at one character), no column opens, the same prepare
  time; only the time differs, and it differs at a low fixed rate too (32 characters: 416 against
  566 us at 500 queries a second, on an idle node), so it is not contention. The search code is
  the same in both commits (the walk, the segment skip, the query building and the window differ
  only by an accessor rename), as are the toolchain (Rust 1.97.1) and the build configuration. What
  differs is the machines: every AMI was a month newer (Ubuntu 2026-10-23 under ScyllaDB, the
  vector-store 1.14.0 AMI on that month's Ubuntu under the index node), and the rows ScyllaDB bounds
  dropped by as much as those the index node bounds. Images, instances and the day are not
  separable in one run, so the AWS test case now pins its AMIs (SCT `9447658ea`), and the next run
  carries this run's default index as a control: its difference from this run is the first measure
  of run-to-run noise. Until then, differences between runs of up to ~20% in this note -- the
  segment-size sweep against the A/B run, the gain credited to `a7fefd2` -- are not established
  either; differences within a run, and the large ones (the segment skip, the rewrite), stand.
- **The port merges an index loaded in bulk much sooner.** In a local benchmark of both commits on
  2M names with a 200k cap, through the actor, the port settled at 11 segments within 10 s of the
  last commit; the stage-5 code stood at 162 segments after four minutes. The port learns the new
  segments' bounds and asks for their merges on every commit tick, the stage-5 code only after
  each commit and on idle ticks.
- **cap_300k does not balance better than 200k here**: it is better at 1 and 8 characters and worse
  at 2, 4 and 32. Its weakest shape (32 characters, 7.6k) is about as weak as the default's
  (16 characters, 7.2k).
- **The backfill half was not measured.** Both indexes were created on the reloaded table and the
  default built by full scan in 220 s; while the cap_300k one was still building, one status request
  from the runner timed out after 30 s, together with SSH channels to the nodes, and the test ended
  (fixed in SCT `54537260a`: a poll without an answer is skipped). The rewrite at 10M was
  measured in run `e78f96a6`, below.

Raw results: `~/sct-results/20261027-090852-795237/` on the laptop that drove the run
(`latency_results.json`, `sct.log`). The index node's logs were not collected.

### Run-to-run noise, and the backfill half again (AWS, 2026-10-27, run `1c3ae034`)

Plan `aws_backfill_config.yaml`, on the images of run `eb0023ac`, now pinned in the SCT test case.
Its `control` index repeats `eb0023ac`'s `default` (same code, corpus, phases, images), so the
difference between the two is noise:

| keyword | eb0023ac | control | change |
|---|---|---|---|
| 1 char | 8,652 | 9,714 | +12% |
| 2 chars | 8,829 | 9,413 | +7% |
| 4 chars | 8,800 | 9,264 | +5% |
| 8 chars | 10,668 | 12,375 | +16% |
| 16 chars | 7,237 | 8,482 | +17% |
| 32 chars | 8,142 | 10,230 | +26% |

For the 8- to 32-character keywords the index opened the same segments as in `eb0023ac`, and the
walk took 14-22% less time; the 1- and 2-character pages came from one segment here and two there.
The control is back at the numbers of `7212926d` (9.7k against 9.97k at one character, 8.5k against
8.7k at 16, 10.2k against 10.3k at 32). So the "slowdown" of `eb0023ac` was the run, not the port,
and **two runs of the same code differ by up to ~25%**: more than most of the differences between
runs this note has read as effects.

The backfill index was built by full scan in 122 s. As its rewrite began, the node stopped answering
HTTP for more than 30 s and the test ended, the same moment and the same symptom as in `eb0023ac`.
The cause was in the pattern index, not the network: every commit tick asked each index whether it
had uncommitted documents by taking its writer lock to read, from inside the actor's task, and a step
of the rewrite holds that lock for seconds. The actor, and every request queued behind it, waited out
each step; and the steps, run on the shared worker, took the runtime's threads besides, queued one
behind another by the following ticks. A test of the rewrite of 10M rows on 4 cores reproduces it
(answers 5.9 s late). Fixed by a tick that never waits for a writer (`try_read`; a held writer means
"not idle, commit next tick") and by tick work that runs on tokio's blocking pool, one at a time per
index: the same test then answers within 10 ms. The rewrite runs at most one range per tick now, 75
ranges in about three minutes at 10M locally.

What is left: a step of the rewrite still holds the writer for seconds, and writes to that index
wait for it -- under a steady CDC stream, ingestion stalls once per step. Moving the reading part
of a step out from under the lock needs a way to keep a concurrent update from being undone by the
re-add; not done.

### The backfill half measured (AWS, 2026-10-27, run `e78f96a6`)

Plan `aws_backfill_config.yaml` again, on the same pinned images, with vector-store `a4fcd76` (the
fix above). The `control` index is the third sample of the same index:

| keyword | eb0023ac | 1c3ae034 | e78f96a6 |
|---|---|---|---|
| 1 char | 8,652 | 9,714 | 9,162 |
| 2 chars | 8,829 | 9,413 | 9,124 |
| 4 chars | 8,800 | 9,264 | 8,547 |
| 8 chars | 10,668 | 12,375 | 10,609 |
| 16 chars | 7,237 | 8,482 | 6,753 |
| 32 chars | 8,142 | 10,230 | 7,777 |
| no match | 59,276 | 61,605 | 57,294 |

The third run lands near the first: the spread is ~10% at 1-4 characters and up to ~30% at 16-32,
with no sign that either end is the "right" one. Compare within a run.

The control caught up 90 s after the load and settled at 51 segments. The backfill index, created
on the reloaded table, built by full scan and caught up 120 s after the load; its rewrite moved
all 75 ranges in about 3 minutes (one range every ~2.4 s, as locally) and it settled at 54
segments, 204 s after it caught up. The node answered every status poll and metrics scrape
throughout.

Capacity (unthrottled, 64 in flight), queries per second, with the index node's walk in us:

| keyword | control | backfill | change |
|---|---|---|---|
| 1 char | 9,162 (312) | 9,363 (339) | +2% |
| 2 chars | 9,124 (211) | 9,148 (228) | 0% |
| 2 chars, window 0.5 | 8,983 (317) | 8,970 (334) | 0% |
| 4 chars | 8,547 (194) | 8,911 (295) | +4% |
| 8 chars | 10,609 (314) | 10,056 (334) | -5% |
| 8 chars, window 0.5 | 21,572 (148) | 20,378 (158) | -6% |
| 16 chars | 6,753 (528) | 6,784 (526) | 0% |
| 32 chars | 7,777 (445) | 8,033 (429) | +3% |
| no match | 57,294 (32) | 58,184 (34) | +2% |

At a fixed rate below capacity (16 in flight), p99 was 2.0 to 3.8 ms on the control and 1.9 to
3.2 ms on the backfill index. **An index built on a loaded table and rewritten serves as well as
one that ingested the load**: the same segments opened per query (2 at 1-4 characters, 37-40 at 8,
~28 at 16, ~9.5 at 32), capacity within 6%, and the walk within ~10% except at 4 characters (295
against 194 us, at a higher rate), all inside the noise of a run.

Raw results: `~/sct-results/20261027-192653-792986/` on the laptop that drove the run.

### Rarest grams and max_gram 4 (AWS, 2026-10-28, run `a7be7f76`)

Plan `aws_rarest_config.yaml`: vector-store `dbbb491`, scylladb `6a0b52f4bf`, the pinned images,
three indexes built while the table loaded. `control` is the port's default; `rarest` intersects only
the 4 grams of a long keyword with the fewest documents in each segment (`rarest_grams: '4'`);
`gram4_rarest` does that over `max_gram: '4'`. All three caught up 788-804 s after the load and
settled at 51 segments. `control` and `rarest` are 56.3 bytes a name, `gram4_rarest` 74.8 (+33%).

Capacity (unthrottled, 64 in flight), queries per second, with the index node's walk in us and the
segments opened per query:

| keyword | control | rarest | gram4_rarest |
|---|---|---|---|
| 1 char | 9,533 (295, 2) | 9,529 (286, 2) | 9,351 (301, 2) |
| 2 chars | 9,509 (201, 2) | 9,448 (194, 2) | 9,264 (205, 2) |
| 2 chars, window 0.5 | 9,407 (322, 2) | 9,304 (321, 2) | 9,123 (321, 2) |
| 4 chars | 9,268 (194, 2) | 9,315 (190, 2) | 9,173 (66, 2) |
| 8 chars | 10,555 (316, 38) | 12,882 (255, 38) +22% | 19,620 (164, 24) +86% |
| 8 chars, window 0.5 | 21,308 (150, 19) | 23,959 (131, 19) +12% | 30,570 (94, 12) +43% |
| 16 chars | 6,891 (516, 28) | 12,202 (268, 28) +77% | 25,406 (120, 8) +269% |
| 32 chars | 7,794 (445, 10) | 15,148 (210, 10) +94% | 34,810 (74, 1) +347% |
| no match | 59,459 (31) | 61,209 (32) | 60,253 (31) |

At a fixed rate below capacity, p99 at 16 characters went from 3.0 ms (control) to 2.1 ms on both,
and at 32 from 4.1 to 2.5 and 2.1 ms.

- **Both are far outside the noise.** The control is where the three earlier controls are (16
  characters 6.9k against 6.8-8.5k), and the gains at 8-32 characters are 22% to 4.5x.
- **Rarest grams halve the walk at 16 and 32 characters** (516 to 268 us, 445 to 210), with the same
  segments opened: the cost was opening and walking the common grams' postings. It helps 8
  characters too (+22%), more than the 2M-name bench suggested.
- **max_gram 4 cuts the segments opened** (8 characters 38 to 24, 16 characters 28 to 8, 32
  characters 10 to 1): a 4-gram is rare enough that few segments hold all of a keyword's without
  holding it. A 4-character keyword becomes one exact lookup (walk 194 to 66 us), which does not move
  its capacity, since ScyllaDB's row reads are the limit there.
- **Every shape now meets 10k except 1 to 4 characters**, which stay at 9.2-9.5k, bounded by
  ScyllaDB reading 20 rows a query on one node (a dev-mode build in every AWS run so far). The index
  node is no longer the limit anywhere.
- The cost is the index, a third larger, and ingestion: not separable here, as the three indexes
  ingested together (the same 788-804 s each).

Six indexes were tried first (run `4c4b7647`, adding `max_gram` 4 alone, 5 with rarest grams, and
the control again at the end): at about 5.2M names each the index node (c8g.xlarge, 8 GB) began to
thrash -- CPU down from ~88% to ~22%, 12 GB read from its disk in five minutes, no answers -- and the
run was aborted. Three indexes at 10M is what that node carries.

`rarest_grams` 4 is the default since this run: it lost no phase and costs no index size. `'0'`
still intersects all grams. Whether `max_gram` 4 should be the default stays open.

Raw results: `~/sct-results/20261028-125049-031151/` on the laptop that drove the run.

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
