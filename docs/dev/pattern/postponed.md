# Pattern index: postponed topics

Things that were implemented in the PoC and taken out, or designed and not started. Each entry
says where the code is, why it is out, and what has to be measured or decided before it returns.

The whole PoC as it stood before any of this was removed is on the branch
`pattern-index-poc-before-rebase-20261118` (vector-store, tip `9c059d2`; ScyllaDB has a branch of
the same name).

## Implemented, then taken out

### 1. Repair order by where searches go ("heat")

- **What it did:** a stretch's fresh rows counted up to nine times as much in the repair's choice
  where the last 4096 searches had opened segments in vain, so that the stretches searched most
  were repaired first.
- **Code:** added in `20a25f3`, removed in `81a5cff`.
- **Why out:** benefit never measured. The one measurement (10M shuffled rows into an empty
  index, searches kept on the middle, the newest or the oldest end) could not show any: there is
  one stretch until ordered segments exist.
- **Before it returns:** measure on a settled index under steady scattered writes, with searches
  (a) kept on one place away from the lowest keys, (b) moving from one place to another,
  (c) spread evenly. It comes back as an index option, off by default, unless (b) and (c) are no
  worse with it.
- **Needs first:** nothing; the handover between the repair's holds (`d07c363`) is in.

### 2. Verification only above the highest unread key ("deferred verification")

- **What it did:** the candidates of a pass of the walk were verified only over the highest key
  left unread; the rest waited for the rows that might push them out of the page.
- **Code:** part of `0ef6d84`, removed in `879932a`.
- **Why out:** on one index of each state, off and on by turns, no cell of the matrix differed in
  time. It saved store reads on an unrepaired index only (49 against 103 a search at most).
- **Before it returns:** measure where the document store is not in memory, which is where a
  store read costs more than a few microseconds.

### 3. Verification on the coordinator

- **What it did:** past `max_gram` ScyllaDB sent `verify: false`, the index answered with
  candidates in order and `verified: false`, and ScyllaDB applied the pattern to the value it
  reads anyway. ScyllaDB index option `verify_candidates`.
- **Code:** vector-store: added in `673e9e2` (stage 5 of the design note), removed in `2044e34`.
  ScyllaDB: `a02f63b420` (the option), `5381acd20c` and `b9d7c4048f` (the request flag and the
  matcher), removed in `a485a1fd8d`.
- **Why out:** measured in September on a design that is gone; never measured on the current walk.
- **Before it returns:** measure store reads and query capacity for a keyword past `max_gram`,
  index verifying against ScyllaDB verifying, on the current walk.

### 4. The `unordered` index

- **What it did:** `'order_by': 'unordered'` kept no sort or token columns, never repaired, and a
  search stopped at the first `n` live matches. No cursor; the request's `offset` paged through
  the matches, stable only between writes.
- **Code:** vector-store: added in `673e9e2`, `4c245c8`, `125900f`; removed in `1c781c6`. ScyllaDB: `422602bda3`, `41da8b3405`, removed in `a485a1fd8d`.
- **Why out:** benefit never measured. It is plausible on a write-heavy table with random keys,
  where table order writes every row twice and searches pay for the fresh tier until the repair
  catches up.
- **Before it returns:** measure ingestion rate, index size and memory, and query capacity under a
  steady insert stream, unordered against table order. Decide what a query without paging means
  to a user.
- **Cost of leaving it out:**
  - A table whose partitioner is not Murmur3, or whose primary key has a column of a type table
    order does not support, gets no pattern index at all; `unordered` was its way out.
  - The request has no `offset` any more; pages come by `cursor` only.

## Designed or proposed, not started

### 5. Work limit per request

A page gets a time budget from the CQL statement timeout; past it the index returns a short page
with a cursor at the highest key not examined. Next after the rebase. Open: how much of the
timeout one page may use, and what an unpaged query does at the deadline.

### 6. Positions option

An index option storing gram positions, so that a keyword past `max_gram` is decided from the
postings and not the store. Removes the false-candidate cost (11 ms a page of 20, 1.5 s in the
worst cell of the matrix). Index size not estimated.

### 7. Sorting each commit

Rows of one commit sorted by key before they are written, so that a fresh segment is one sorted
run. Would cut the cost of the fresh tier for rows arriving out of order.

### 8. Tuning left as found

- The per-run cap and the halving cost 1.6-1.8x on dense keywords with a page of 20 on a stream
  index (20 us against 11); they save up to 5.5x elsewhere.
- The "indexing contended" signal alone holds the repair on 60-80% of ticks under mixed load.
  Whether it should hold it that much is not measured.
- A dense page of 1000 on a stream index costs about 1.5 ms in sorted-vector inserts.
- Time order under random writes: 123 us against 89 us before, one run each, not investigated.

## Changed on the move onto upstream (2026-11-20)

The series was moved from upstream `4dd84dfe6` onto `25c9b70`, which had gained its own `LIKE`
search (`gram.rs`, `like.rs`). What changed in the series because of it:

- **Every `LIKE` pattern is served.** Upstream parses the whole pattern (`%`, `_`, `\`) and checks
  a candidate with a regex. The series served only `'%kw%'`, `'kw%'` and `'%kw'` and answered any
  other pattern with 400; its own parser and the keyword/kind pair of the search message are
  gone. The ordered walk takes upstream's query: the grams give the candidates, and a candidate
  is checked only when it could enter the page.
- **A pattern without a gram is refused only for `min_gram`.** A pattern whose literal characters
  all stand in runs shorter than `min_gram` is answered with 400, as a keyword shorter than
  `min_gram` was. A pattern with no literal character (`%`, `_`) is served, by reading every row
  of the segments the walk opens, as upstream does. An empty pattern is no longer an error: it
  matches the empty value.
- **A row indexed twice is returned once**, as upstream's search did: the page refuses a second
  document with the primary id of an entry it holds.
- **A stored value is checked as text where the pattern allows, not with upstream's regex.**
  The regex is built for every query and cost a verified query of a page of 20 about 55 us
  (97 us against 42 us on a stream-built index of 2M rows, 2026-11-20). A pattern of one literal
  run (`abc`, `abc%`, `%abc`, `%abc%`) is compared as text; `_` and an inner `%` keep the regex.
- **A pattern tied to an end of the value is always checked against the stored value**, as
  upstream does: a value may hold a frame mark itself. Before, a short `ab%` or `%ab` was one
  exact term and read nothing from the store. Measured on 2M rows (2026-11-20, matrix run
  `up3`): such a pattern costs what any verified one does, one store read per row it returns,
  about 1 us each on a page of 20 and 0.4 us on a page of 1000 (`%新人`: 50 us against 29 us
  for the unchecked `新人` at 20 rows, 1257 us against 876 us at 1000, settled).
- **Dropped: `VECTOR_STORE_PATTERN_INDEXES`.** Upstream removed the full-text flag it was modelled
  on (`6654aaa`), as a gate of a feature under development.
- **Operations applied in the order they came: kept, for the layout.** Upstream holds the
  removal of a row in the monitor until its add is applied (`5030ae4`), which covers the race
  the commit was written for, and it was dropped at first. Without it the workers apply
  neighbouring rows in a shuffled order: on a stream-built index of 2M rows over 90% of the rows
  lay in coarse runs (about 100 sorted and 100 coarse runs a segment, against about 270 sorted
  and one coarse), and a page of 20 scanned 10 to 70 times the postings (2026-11-20, matrix
  runs `up` and `up2`). It was restored for that alone. Sorting each commit (7 above) would
  give the same layout for rows arriving in any order and make it unnecessary; this
  measurement is an argument for doing that.
- **Dropped: the series' validator tests**, which called `/like` with keywords. Upstream's
  validator tests of the pattern index stay. None covers the order or the paging.
- **The index size in the statistics skips a segment whose files are gone.** A merge deletes the
  files of the segments it replaced while a searcher from before the reload still lists them;
  reading their size failed the whole statistics read, in about one unit-test run in three.

Measured again on the new base: the page-cost matrix (runs `up`, `up2`, `up3`: settled,
unrepaired and stream). Not measured again: the repair runs, so the repair figures in these
notes are those of the old base. The ScyllaDB side still sends what it
sent; what it may now send (any pattern) and what it must expect (the 400 above) is its part of
the move.

## Open problems

- **A stray row in the ordered tier makes a wide cluster.** A segment that overlaps no other is
  adopted into the ordered tier; in a shuffled load an early commit of one row sometimes is such
  a segment (2 of 15 local runs of 10M rows on 2026-11-19). Neighbouring ordered clusters merge
  when they fit, so that row was merged with 287,286 rows at the other end of the key space, into
  a segment spanning 7M fresh rows. This caused the repair stall of about 930 s (2 of 18 runs on
  2026-11-18, 1 of 15 on 2026-11-19): fixed by bounding a hold to the step's range. What is left:
  each step over such a cluster splits it again, so the repair rewrote 15.0M rows for 10M and
  took 328 s against about 190 s. Not decided: whether a merge should join clusters across fresh
  rows, and whether a segment of a few rows should be adopted. To reproduce:
  `HEAT_STRAY=1` on the test `the_repair_under_searches_kept_on_one_place`
  (`HEAT_NO_HOLD_BOUND=1` switches the fix off).
- **No fast test of the hold bound.** A 40,000-row index did not form the wide cluster; only the
  10M run does.
- **Single cells of the page-cost matrix swing 3-5x between builds of one index.** Measured
  2026-11-19 at six commits from `69aeb32` to `2044e34`, one build of the index each: on an
  unrepaired index of 10M rows a keyword in one row of five, page of 1000, scanned 35k-57k
  postings at four commits and 170k and 398k at two; on a stream index of 2M rows two cells
  scanned 5x more at one commit. No commit orders them: `69aeb32`, whose three runs of
  2026-11-18 scanned 37k-59k, scanned 170k. Runs of one binary agreed with each other (three
  of three, twice), which made it read as a step. What differs between a cheap layout and a
  dear one is not known. Until it is, a comparison between builds decides nothing under about
  5x in one cell; the same index with a mechanism off and on by turns (`MATRIX_AB`) does.
