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
- The paging state carries `(sort_value, primary_key)`. Offset paging is not extended.
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

## Open questions and risks

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
- **Replacing the merge policy is load-bearing.** Getting it wrong does not fail loudly; it silently
  widens segments until pruning stops working.
