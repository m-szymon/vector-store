/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! Exact substring containment (`LIKE '%kw%'`) over short text values, on Tantivy.
//!
//! Every substring of a value between `min_gram` and `max_gram` characters long is an indexed
//! term. A query of at most `max_gram` characters is then a single term lookup and every hit is
//! an exact match. A longer query is answered by intersecting the posting lists of its
//! `max_gram`-long substrings, which admits false positives (all the pieces present, but not
//! contiguous), so each candidate is verified against the stored value before it is returned.
//!
//! # Ordered search, and what it does not do yet
//!
//! An index created with an `order_by` column carries that column's value as a `FAST` u64 and
//! answers newest-first, paging by a cursor rather than an offset. Known gaps, all of them things
//! a proof of concept can live with and a shipped feature cannot:
//!
//! * **Ties are ordered by primary id, which is not a CQL order.** Rows sharing a sort key come
//!   highest internal id first (the most recently indexed first, in practice), and the cursor
//!   names the last row's primary key so the next page resumes among them exactly. A rebuild of
//!   the index assigns new ids, so the order among tied rows can change across one; a cursor
//!   whose row is gone takes the tied rows again rather than skip any.
//! * **A short page is ambiguous.** Rows dropped because the table no longer knows them shorten a
//!   page after the walk has filled it, so a caller cannot infer "no more results" from a page
//!   shorter than the limit, and must follow the cursor instead. The converse is exact: the walk
//!   reports a cursor only when it filled the page, so no cursor does mean no more results.
//! * **Pruning depends on segment layout.** Cost is flat only where a segment's span of the sort
//!   column is narrow. Two options keep it so: `segment_max_docs` (`poc_option_2`) installs a
//!   merge policy that only joins neighbours in sort order under a cap, and
//!   `rewrite_wide_segments` (`poc_option_3`) moves the rows of segments that span far more than
//!   their share -- an index built by a full scan, a shuffled load -- into range-aligned ones,
//!   one range per idle tick. Without them, after an unordered backfill every segment spans
//!   everything and the search degrades to visiting every match: correct, but not fast. See
//!   `docs/dev/substring/stage-2-ordering.md`.

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::ops::Bound;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Duration;
use std::time::Instant;

use anyhow::anyhow;
use tantivy::DocAddress;
use tantivy::DocSet;
use tantivy::SegmentReader;
use tantivy::TERMINATED;
use tantivy::TantivyDocument;
use tantivy::Term;
use tantivy::columnar::Column;
use tantivy::index::SegmentId;
use tantivy::index::SegmentMeta;
use tantivy::indexer::MergeCandidate;
use tantivy::indexer::MergePolicy;
use tantivy::query::BooleanQuery;
use tantivy::query::EnableScoring;
use tantivy::query::Query;
use tantivy::query::TermQuery;
use tantivy::schema::FAST;
use tantivy::schema::INDEXED;
use tantivy::schema::IndexRecordOption;
use tantivy::schema::STORED;
use tantivy::schema::Schema;
use tantivy::schema::TextFieldIndexing;
use tantivy::schema::TextOptions;
use tantivy::schema::Value;
use tantivy::tokenizer::NgramTokenizer;
use tantivy::tokenizer::TextAnalyzer;
use tokio::sync::mpsc;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::CaseSensitive;
use crate::IndexKey;
use crate::IndexOptionsSubstring;
use crate::Limit;
use crate::memory::Allocate;
use crate::memory::Memory;
use crate::memory::MemoryExt;
use crate::metrics::Metrics;
use crate::perf;
use crate::table::IndexId;
use crate::table::PartitionId;
use crate::table::PrimaryId;
use crate::table::Table;
use crate::table::TableSearch;
use crate::tantivy_common::COMMIT_INTERVAL;
use crate::tantivy_common::IndexState;
use crate::tantivy_common::MAX_UNCOMMITTED_THRESHOLD;
use crate::tantivy_common::PRIMARY_ID_FIELD;
use crate::tantivy_common::QueryError;
use crate::tantivy_common::TantivyBackend;
use crate::tantivy_common::TantivyStats;
use crate::tantivy_common::can_allocate_memory;
use crate::tantivy_common::commit;
use crate::tantivy_common::find_partition_id;
use crate::tantivy_common::get_or_create_state;
use crate::tantivy_common::get_state;
use crate::tantivy_common::handle_add_document;
use crate::tantivy_common::handle_remove_document;
use crate::tantivy_common::handle_stats;
use crate::tantivy_common::primary_id_term;
use crate::tantivy_common::reload;
use crate::worker::Worker;
use crate::worker::WorkerExt;

use super::actor::Cursor;
use super::actor::MatchKind;
use super::actor::SearchWindow;
use super::actor::SortOrder;
use super::actor::SubstringIndex;
use super::actor::SubstringPage;
use super::actor::SubstringSearchR;
use super::factory::SubstringIndexConfiguration;
use super::factory::SubstringIndexFactory;

pub(crate) struct TantivySubstringIndexFactory {
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
    metrics: Arc<Metrics>,
}

impl TantivySubstringIndexFactory {
    pub(crate) fn new(
        worker: async_channel::Sender<Worker>,
        memory: mpsc::Sender<Memory>,
        metrics: Arc<Metrics>,
    ) -> Self {
        Self {
            worker,
            memory,
            metrics,
        }
    }
}

impl SubstringIndexFactory for TantivySubstringIndexFactory {
    fn create_index(
        &self,
        index: SubstringIndexConfiguration,
        table: Arc<RwLock<Table>>,
    ) -> mpsc::Sender<SubstringIndex> {
        new(
            index,
            table,
            self.worker.clone(),
            self.memory.clone(),
            COMMIT_INTERVAL,
            MAX_UNCOMMITTED_THRESHOLD,
            Some(Arc::clone(&self.metrics)),
        )
    }
}

const TEXT_FIELD: &str = "text";
const SORT_FIELD: &str = "sort_key";
const TOKENIZER_NAME: &str = "substring_ngram";
/// Every value is indexed and stored between these two marks, so that a prefix query is the
/// containment of `START + keyword`, a suffix query of `keyword + END`, and both take the very
/// same walk as a containment query: the marks are ordinary characters to the tokenizer and to
/// the verification. Two grams per value, a few percent of the index. A value holding one of the
/// marks itself would match a prefix or suffix query wrongly; they are C0 control characters,
/// which nicknames do not carry.
const VALUE_START: char = '\u{2}';
const VALUE_END: char = '\u{3}';

/// The form a value is indexed and stored in.
fn framed(normalized: &str) -> String {
    format!("{VALUE_START}{normalized}{VALUE_END}")
}

/// A stored value without its frame: what a re-indexed row is built from.
fn unframed(stored: &str) -> &str {
    stored
        .strip_prefix(VALUE_START)
        .unwrap_or(stored)
        .strip_suffix(VALUE_END)
        .unwrap_or(stored)
}

/// The text a query of `kind` for `normalized` has to find in a framed value.
fn pattern_of(kind: MatchKind, normalized: &str) -> String {
    match kind {
        MatchKind::Contains => normalized.to_string(),
        MatchKind::Prefix => format!("{VALUE_START}{normalized}"),
        MatchKind::Suffix => format!("{normalized}{VALUE_END}"),
    }
}
/// Values are short, so a single cached store block per segment covers consecutive lookups.
const STORE_CACHE_BLOCKS: usize = 1;

/// The substring flavour of a Tantivy index: the (normalized) value is stored verbatim for
/// verification and indexed as all of its `min_gram..=max_gram`-long substrings.
struct SubstringBackend {
    options: IndexOptionsSubstring,
    /// What every search since the index was created has cost, for `/metrics` to export.
    walk: WalkCounters,
    /// The columns an ordered search reads, opened once per segment rather than once per query.
    /// Opening a column costs time proportional to the segment's size -- measured at 2.7 us for
    /// 77k rows, and it was most of a query's fixed cost against 1.3M-row segments on AWS -- while
    /// the handle itself is a cheap clone. Keyed by segment id, so a merge simply leaves stale
    /// entries behind, which `prune_columns` drops once they outnumber the live segments.
    columns: RwLock<HashMap<SegmentId, SegmentColumns>>,
    /// Each live segment's sort bounds, for the merge policy, which sees only segment metas.
    /// Refreshed after every reload; shared with the policy the writer holds.
    bounds: SharedBounds,
    /// P3b: the rewrite of wide segments in progress, if any, and what it has done so far.
    plan: Mutex<Option<RewritePlan>>,
    rewrite: RewriteCounters,
}

/// A segment is wide when it spans more of the sort range than this many caps' worth of rows
/// would if they were contiguous. Measured against the cap rather than against the segment's own
/// row count so that the pieces a commit's writer threads leave -- each a fraction of the
/// commit's rows over the commit's whole key range -- count as narrow however many threads there
/// are, while a segment built by a full scan, spanning everything, is wide whatever its size.
const WIDE_SEGMENT_FACTOR: f64 = 8.0;
/// Sort keys sampled from the wide segments to place the range boundaries.
const REWRITE_SAMPLE_TARGET: u64 = 200_000;

/// P3b: which segments to empty and the sort ranges to move their rows into, one range per
/// idle tick so that ingestion and searches interleave with the rewrite.
#[derive(Debug)]
struct RewritePlan {
    /// `[low, high)` in sort-key space, ascending, covering everything. Each is moved out of
    /// whichever segments are wide when its turn comes: the merge policy retires segment ids all
    /// the time, so a plan that named them would find them gone.
    ranges: Vec<(u64, u64)>,
    next: usize,
}

#[derive(Debug, Default)]
struct RewriteCounters {
    ranges_total: AtomicU64,
    ranges_done: AtomicU64,
    docs_rewritten: AtomicU64,
    l0_docs: AtomicU64,
}

/// [`RewriteCounters`] at one moment, as `Stats` reports them.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RewriteTotals {
    pub(crate) ranges_total: u64,
    pub(crate) ranges_done: u64,
    pub(crate) docs_rewritten: u64,
    /// Rows in wide segments when the current (or last) plan was made.
    pub(crate) l0_docs: u64,
}

/// The widest span, in sort-key units, a narrow segment may have: `WIDE_SEGMENT_FACTOR` caps'
/// worth of a range holding `total_docs` rows. Everything is narrow while the index holds fewer
/// rows than that many caps.
fn narrow_span(segments: &[(u32, u64, u64, SegmentId)], total_docs: u64, cap: u32) -> u64 {
    if total_docs == 0 {
        return u64::MAX;
    }
    let low = segments.iter().map(|s| s.1).min().unwrap_or(0);
    let high = segments.iter().map(|s| s.2).max().unwrap_or(0);
    let range = (high - low).max(1) as f64;
    let span = range * WIDE_SEGMENT_FACTOR * f64::from(cap) / total_docs as f64;
    if span >= u64::MAX as f64 {
        u64::MAX
    } else {
        span as u64
    }
}

/// The segments that call for a rewrite: those spanning more than `narrow_span`, and those over
/// the cap, which no merge can ever bring under it. Input tuples are
/// `(docs, sort_min, sort_max, id)`; `min_docs` says how big a wide segment has to be to count.
fn wide_segments(
    segments: &[(u32, u64, u64, SegmentId)],
    total_docs: u64,
    min_docs: u32,
    cap: u32,
) -> Vec<SegmentId> {
    let narrow = narrow_span(segments, total_docs, cap);
    segments
        .iter()
        .filter(|(docs, min, max, _)| *docs >= min_docs && (*docs > cap || max - min > narrow))
        .map(|(_, _, _, id)| *id)
        .collect()
}

/// The segments a rewrite in progress takes rows from: everything the policy would not fold
/// into a narrow run, i.e. spanning more than a narrow run may, or over the cap. Wider than
/// [`wide_segments`] on purpose: the rewrite moves ranges in ascending key order, so what is
/// left in the segments it drains is a shrinking band of the highest keys, and once that band
/// spans under eight caps their merged remains would stop counting as wide and escape with a
/// cap's rows over several caps' width, at the top of the range where "newest first" looks.
fn spread_segments(
    segments: &[(u32, u64, u64, SegmentId)],
    total_docs: u64,
    cap: u32,
) -> Vec<SegmentId> {
    let run_span = narrow_span(segments, total_docs, cap) / NARROW_RUN_SPAN_DIVISOR;
    segments
        .iter()
        .filter(|(docs, min, max, _)| *docs > cap || max - min > run_span)
        .map(|(_, _, _, id)| *id)
        .collect()
}

/// About `count` ranges covering `low..=high` with about as many of the sampled keys in each,
/// none wider than `max_width` in key space: a quantile range over sparse rows -- the remains of
/// a drained band -- can span half the key space, and the pieces its rows land in are as wide as
/// the range, wide enough to be planned again and again. Such a range is cut into equal parts
/// of at most `max_width`, so a rewritten range always lands narrow. Fewer quantile cuts when
/// the sample has ties at a boundary.
fn quantile_ranges(
    mut sample: Vec<u64>,
    count: usize,
    low: u64,
    high: u64,
    max_width: u64,
) -> Vec<(u64, u64)> {
    sample.sort_unstable();
    let count = count.max(1);
    let max_width = max_width.max(1);
    let end = high.saturating_add(1);
    let mut bounds = vec![low];
    for k in 1..count {
        let key = sample[k * sample.len() / count];
        if key > *bounds.last().unwrap() && key < end {
            bounds.push(key);
        }
    }
    bounds.push(end);
    let mut ranges = Vec::new();
    for w in bounds.windows(2) {
        let (from, to) = (w[0], w[1]);
        let parts = (to - from).div_ceil(max_width).max(1);
        let step = (to - from).div_ceil(parts);
        let mut at = from;
        while at < to {
            let next = at.saturating_add(step).min(to);
            ranges.push((at, next));
            at = next;
        }
    }
    ranges
}

type SharedBounds = Arc<RwLock<HashMap<SegmentId, (u64, u64)>>>;

/// P3a: merges neighbours in sort order, never past a size cap.
///
/// Tantivy's default policy merges segments of similar size whatever they hold, which is how an
/// index ends up with 1.3M-row segments spanning a quarter of the range each and a tail of small
/// ones that all reach the top: a page-1 query opens every tail segment and a deep page scans
/// whichever big segment its cursor lands in. This one sorts the segments by their lowest sort
/// key and merges runs of neighbours while the run stays under `max_docs`, so a segment only ever
/// grows into the rows next to it in sort order and never past the cap. A segment at or over the
/// cap is left alone. Segments whose bounds are not known yet (committed but not reloaded) wait
/// for the next round.
///
/// Wide segments -- rows that arrived out of sort order, a build by full scan -- never join a
/// narrow run, which would only widen it: they merge among themselves, and the rewrite (P3b)
/// takes them apart. Without the rewrite they stay, a few segments every ordered query has to
/// open, which is the stage-2 state.
///
/// The price is write amplification on the tail: each commit adds a small segment next to it
/// and the run is rewritten, up to `max_docs` rows per commit. Acceptable for measuring the
/// layout; a levelled tail is the obvious refinement.
#[derive(Debug)]
struct RangeMergePolicy {
    max_docs: u32,
    bounds: SharedBounds,
}

impl MergePolicy for RangeMergePolicy {
    fn compute_merge_candidates(&self, segments: &[SegmentMeta]) -> Vec<MergeCandidate> {
        let bounds = self.bounds.read().unwrap();
        let known = segments
            .iter()
            .filter_map(|meta| {
                bounds
                    .get(&meta.id())
                    .map(|&(low, high)| (meta.num_docs(), low, high, meta.id()))
            })
            .collect::<Vec<_>>();
        let total: u64 = segments.iter().map(|meta| u64::from(meta.num_docs())).sum();
        let narrow = narrow_span(&known, total, self.max_docs);
        merge_runs(known, self.max_docs, narrow)
            .into_iter()
            .map(MergeCandidate)
            .collect()
    }
}

/// How much of `narrow_span` a merged narrow run may cover: a quarter, i.e. two caps' worth of
/// rows. The pieces of one rewritten range fit in one cap's width; letting a run reach the
/// eight of the wide limit would fold the sparse pieces of several ranges into one segment
/// holding a cap's rows over four caps' width, which nothing would take apart again.
const NARROW_RUN_SPAN_DIVISOR: u64 = 4;

/// The runs of sort-order neighbours worth merging: each at least two segments, each under
/// `max_docs` rows in total. A narrow run -- segments spanning no more than a quarter of
/// `narrow_span` each -- stops growing where its combined span would pass that quarter. Anything
/// spanning more than that never joins a narrow run, which would only widen it; those segments
/// merge among themselves with no limit on the span, so slivers grow into a segment the rewrite
/// will find wide. Input tuples are `(docs, sort_min, sort_max, id)`. Pure, so it can be tested
/// without a writer.
fn merge_runs(
    segments: Vec<(u32, u64, u64, SegmentId)>,
    max_docs: u32,
    narrow_span: u64,
) -> Vec<Vec<SegmentId>> {
    let run_span = narrow_span / NARROW_RUN_SPAN_DIVISOR;
    let (spread, narrow): (Vec<_>, Vec<_>) = segments
        .into_iter()
        .partition(|&(_, low, high, _)| high - low > run_span);
    let mut runs = runs_under(narrow, max_docs, run_span);
    runs.extend(runs_under(spread, max_docs, u64::MAX));
    runs
}

fn runs_under(
    mut segments: Vec<(u32, u64, u64, SegmentId)>,
    max_docs: u32,
    max_span: u64,
) -> Vec<Vec<SegmentId>> {
    segments.sort_by_key(|&(_, low, high, _)| (low, high));
    let mut runs = Vec::new();
    let mut run: Vec<SegmentId> = Vec::new();
    let mut run_docs = 0u32;
    let mut run_span = (u64::MAX, 0u64);
    let mut flush = |run: &mut Vec<SegmentId>, run_docs: &mut u32, run_span: &mut (u64, u64)| {
        if run.len() >= 2 {
            runs.push(std::mem::take(run));
        } else {
            run.clear();
        }
        *run_docs = 0;
        *run_span = (u64::MAX, 0);
    };
    for (docs, low, high, id) in segments {
        if docs >= max_docs {
            // Full already: it ends the run it would have joined and stands alone.
            flush(&mut run, &mut run_docs, &mut run_span);
            continue;
        }
        let joined = (run_span.0.min(low), run_span.1.max(high));
        if run_docs + docs > max_docs || joined.1 - joined.0 > max_span {
            flush(&mut run, &mut run_docs, &mut run_span);
        }
        run.push(id);
        run_docs += docs;
        run_span = (run_span.0.min(low), run_span.1.max(high));
    }
    flush(&mut run, &mut run_docs, &mut run_span);
    runs
}

/// One segment's columnar handles and the sort bounds the pruning reads from them.
#[derive(Clone)]
struct SegmentColumns {
    sort: Column<u64>,
    sort_min: u64,
    sort_max: u64,
    /// Present on every ordered index (the walk breaks ties on it) and with `primary_id_fast`.
    primary_id: Option<Column<u64>>,
}

impl SubstringBackend {
    /// The cached columns of `segment`, opened on first sight. `opened` counts a miss, so a
    /// search can report how much opening it did.
    fn columns_for(
        &self,
        segment: &SegmentReader,
        opened: &mut u64,
    ) -> anyhow::Result<SegmentColumns> {
        let id = segment.segment_id();
        if let Some(columns) = self.columns.read().unwrap().get(&id) {
            return Ok(columns.clone());
        }
        let sort = segment
            .fast_fields()
            .u64(SORT_FIELD)
            .map_err(|e| anyhow!("substring: failed to open the sort column: {e}"))?;
        let primary_id = (self.orders_results() || *self.options.primary_id_fast.as_ref())
            .then(|| segment.fast_fields().u64(PRIMARY_ID_FIELD))
            .transpose()
            .map_err(|e| anyhow!("substring: failed to open the primary id column: {e}"))?;
        let columns = SegmentColumns {
            sort_min: sort.min_value(),
            sort_max: sort.max_value(),
            sort,
            primary_id,
        };
        *opened += 1;
        self.columns.write().unwrap().insert(id, columns.clone());
        Ok(columns)
    }

    /// Every live segment as `(docs, sort_min, sort_max, id)`, from the cached columns; a
    /// segment whose sort column cannot be opened is left out, with a warning.
    fn geometry(&self, searcher: &tantivy::Searcher) -> Vec<(u32, u64, u64, SegmentId)> {
        let mut opened = 0;
        searcher
            .segment_readers()
            .iter()
            .filter_map(|segment| match self.columns_for(segment, &mut opened) {
                Ok(columns) => Some((
                    segment.num_docs(),
                    columns.sort_min,
                    columns.sort_max,
                    segment.segment_id(),
                )),
                Err(err) => {
                    warn!(
                        "substring: no bounds for segment {}: {err}",
                        segment.segment_id().short_uuid_string()
                    );
                    None
                }
            })
            .collect()
    }

    /// Drops cache entries for segments a merge has retired, once they outnumber the live ones.
    fn prune_columns(&self, live: &[SegmentReader]) {
        let mut cache = self.columns.write().unwrap();
        if cache.len() > 2 * live.len().max(1) {
            let keep: std::collections::HashSet<SegmentId> =
                live.iter().map(|segment| segment.segment_id()).collect();
            cache.retain(|id, _| keep.contains(id));
        }
    }
}

/// Running totals of what the walks did, so that a rate of queries can be explained rather than
/// guessed at: how many segments the pruning left, how many postings had to be scanned for them,
/// how many candidates made the heap, and how often the document store was opened. Each search
/// adds its own tally once, at the end, so the hot loop touches only locals.
#[derive(Debug, Default)]
struct WalkCounters {
    searches: AtomicU64,
    segments_considered: AtomicU64,
    segments_opened: AtomicU64,
    postings_scanned: AtomicU64,
    heap_entrants: AtomicU64,
    store_reads: AtomicU64,
    column_opens: AtomicU64,
    walk_nanos: AtomicU64,
    prepare_nanos: AtomicU64,
    page_resolve_nanos: AtomicU64,
}

/// One search's tally, added to [`WalkCounters`] when it finishes.
#[derive(Debug, Default, Clone, Copy)]
struct WalkTally {
    segments_considered: u64,
    segments_opened: u64,
    postings_scanned: u64,
    heap_entrants: u64,
    store_reads: u64,
    column_opens: u64,
}

/// How long the parts of one search took.
#[derive(Debug, Default, Clone, Copy)]
struct WalkTimes {
    walk: Duration,
    prepare: Duration,
    page_resolve: Duration,
}

impl WalkCounters {
    fn record(&self, tally: WalkTally, times: WalkTimes) {
        self.searches.fetch_add(1, Relaxed);
        self.segments_considered
            .fetch_add(tally.segments_considered, Relaxed);
        self.segments_opened
            .fetch_add(tally.segments_opened, Relaxed);
        self.postings_scanned
            .fetch_add(tally.postings_scanned, Relaxed);
        self.heap_entrants.fetch_add(tally.heap_entrants, Relaxed);
        self.store_reads.fetch_add(tally.store_reads, Relaxed);
        self.column_opens.fetch_add(tally.column_opens, Relaxed);
        let nanos = |d: Duration| d.as_nanos().try_into().unwrap_or(u64::MAX);
        self.walk_nanos.fetch_add(nanos(times.walk), Relaxed);
        self.prepare_nanos.fetch_add(nanos(times.prepare), Relaxed);
        self.page_resolve_nanos
            .fetch_add(nanos(times.page_resolve), Relaxed);
    }

    fn snapshot(&self) -> WalkTotals {
        WalkTotals {
            searches: self.searches.load(Relaxed),
            segments_considered: self.segments_considered.load(Relaxed),
            segments_opened: self.segments_opened.load(Relaxed),
            postings_scanned: self.postings_scanned.load(Relaxed),
            heap_entrants: self.heap_entrants.load(Relaxed),
            store_reads: self.store_reads.load(Relaxed),
            column_opens: self.column_opens.load(Relaxed),
            walk_nanos: self.walk_nanos.load(Relaxed),
            prepare_nanos: self.prepare_nanos.load(Relaxed),
            page_resolve_nanos: self.page_resolve_nanos.load(Relaxed),
        }
    }
}

/// [`WalkCounters`] at one moment, as `Stats` reports them.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WalkTotals {
    pub(crate) searches: u64,
    pub(crate) segments_considered: u64,
    pub(crate) segments_opened: u64,
    pub(crate) postings_scanned: u64,
    pub(crate) heap_entrants: u64,
    pub(crate) store_reads: u64,
    /// Columns opened, i.e. misses of the per-segment column cache.
    pub(crate) column_opens: u64,
    pub(crate) walk_nanos: u64,
    /// Time spent before the first posting: reading every segment's bounds. Part of `walk_nanos`.
    pub(crate) prepare_nanos: u64,
    /// Time spent turning the walk's page into primary ids, included in `walk_nanos`. For an
    /// index without a FAST primary id that is the store reads; it is what they cost.
    pub(crate) page_resolve_nanos: u64,
}

/// One segment as the ordered walk sees it: how many rows it holds and the span of sort keys the
/// pruning judges it by. The span is what decides whether ordering is cheap -- a segment covering
/// the whole range can never be skipped -- and it is not visible from anything else the node
/// reports. `None` bounds mean the index has no sort column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SegmentLayout {
    pub(crate) docs: u32,
    pub(crate) deleted: u32,
    pub(crate) sort_min: Option<u64>,
    pub(crate) sort_max: Option<u64>,
}

/// What `Stats` reports for a substring index: the Tantivy figures every index has, plus what
/// this one's searches have cost and how its segments are laid out.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct SubstringStats {
    pub(crate) tantivy: TantivyStats,
    pub(crate) walk: WalkTotals,
    pub(crate) rewrite: RewriteTotals,
    pub(crate) segments: Vec<SegmentLayout>,
}

pub(crate) type SubstringStatsR = anyhow::Result<SubstringStats>;

impl SubstringBackend {
    fn min_gram(&self) -> usize {
        self.options.min_gram.as_ref().get()
    }

    fn max_gram(&self) -> usize {
        self.options.max_gram.as_ref().get()
    }

    /// Whether this index was given a column to order by. Without one it answers in whatever order
    /// the walk finds matches, and carries no sort field at all.
    fn orders_results(&self) -> bool {
        self.options.order_by.as_ref().is_some()
    }
}

/// What a row contributes to a substring index: the text to index, and the value to order by if
/// the index was given a sort column.
pub(crate) struct SubstringRow<'a> {
    pub(crate) text: &'a str,
    pub(crate) sort_key: Option<u64>,
}

impl TantivyBackend for SubstringBackend {
    const NAME: &'static str = "substring";

    fn build_schema(&self) -> Schema {
        // Neither term frequencies nor positions are needed: a term is either present in a
        // value or not, and the n-gram tokenizer reports every position as 0 anyway.
        let indexing = TextFieldIndexing::default()
            .set_tokenizer(TOKENIZER_NAME)
            .set_index_option(IndexRecordOption::Basic);
        let text_options = TextOptions::default()
            .set_indexing_options(indexing)
            .set_stored();
        let mut schema_builder = Schema::builder();
        if self.orders_results() || *self.options.primary_id_fast.as_ref() {
            // Also columnar: an ordered walk reads it per candidate to break ties on the sort
            // key, and with `primary_id_fast` a page's ids come from it as well. The store keeps
            // its copy for the verified path, which reads the document anyway.
            schema_builder.add_u64_field(PRIMARY_ID_FIELD, INDEXED | STORED | FAST);
        } else {
            schema_builder.add_u64_field(PRIMARY_ID_FIELD, INDEXED | STORED);
        }
        schema_builder.add_text_field(TEXT_FIELD, text_options);
        if self.orders_results() {
            // FAST, not STORED: the walk reads it once per candidate, and a columnar read is far
            // cheaper than the document store. Its per-segment min/max is also what lets an
            // ordered search skip whole segments.
            schema_builder.add_u64_field(SORT_FIELD, FAST);
        }
        schema_builder.build()
    }

    fn register_tokenizers(&self, index: &tantivy::Index) -> anyhow::Result<()> {
        let tokenizer = NgramTokenizer::new(self.min_gram(), self.max_gram(), false)
            .map_err(|e| anyhow!("substring: failed to create the n-gram tokenizer: {e}"))?;
        index
            .tokenizers()
            .register(TOKENIZER_NAME, TextAnalyzer::builder(tokenizer).build());
        Ok(())
    }

    type Row<'a> = SubstringRow<'a>;

    fn create_doc(
        &self,
        schema: &Schema,
        primary_id: PrimaryId,
        row: SubstringRow<'_>,
    ) -> TantivyDocument {
        let primary_id_field = schema.get_field(PRIMARY_ID_FIELD).unwrap();
        let text_field = schema.get_field(TEXT_FIELD).unwrap();

        // The normalized form is both indexed and stored, so that grams and the verification
        // compare like with like even when Unicode lowercasing changes the character count.
        // The stored value is never returned to clients.
        let normalized = normalize(row.text, self.options.case_sensitive);
        let mut doc = TantivyDocument::new();
        doc.add_u64(primary_id_field, u64::from(primary_id));
        doc.add_text(text_field, framed(normalized.as_ref()));
        if self.orders_results() {
            // A row whose sort column is null still belongs in the index; it just sorts lowest,
            // which for "newest first" puts it last. Dropping it would make the index disagree
            // with an unordered LIKE about which rows exist.
            doc.add_u64(
                schema.get_field(SORT_FIELD).unwrap(),
                row.sort_key.unwrap_or(0),
            );
        }
        doc
    }

    fn merge_policy(&self) -> Option<Box<dyn MergePolicy>> {
        let max_docs = (*self.options.segment_max_docs.as_ref())?;
        if !self.orders_results() {
            // Nothing to order the segments by; the cap alone is not worth a policy.
            return None;
        }
        Some(Box::new(RangeMergePolicy {
            max_docs: max_docs.get(),
            bounds: Arc::clone(&self.bounds),
        }))
    }

    /// Learns every live segment's bounds (which also warms the column cache), drops the retired
    /// ones, and asks the writer to merge what the policy would: Tantivy evaluates merges on its
    /// own after a commit, but at that moment the bounds of the segments the commit produced are
    /// not known yet, so the runs they belong to would otherwise wait for a commit that may never
    /// come once the writes stop.
    fn after_reload(&self, state: &IndexState<Self>) {
        if !self.orders_results() {
            return;
        }
        let searcher = state.reader.searcher();
        let geometry = self.geometry(&searcher);
        self.prune_columns(searcher.segment_readers());
        *self.bounds.write().unwrap() = geometry
            .iter()
            .map(|&(_, low, high, id)| (id, (low, high)))
            .collect();

        let Some(max_docs) = *self.options.segment_max_docs.as_ref() else {
            return;
        };
        let narrow = narrow_span(&geometry, searcher.num_docs(), max_docs.get());
        let runs = merge_runs(geometry.clone(), max_docs.get(), narrow);
        if !runs.is_empty() {
            info!(
                "substring: asking for {} merges of {} segments ({} live)",
                runs.len(),
                runs.iter().map(Vec::len).sum::<usize>(),
                searcher.segment_readers().len()
            );
        }
        for run in runs {
            // The merge runs on Tantivy's own thread; the future only reports its outcome, and
            // the next reload picks the merged segment up either way.
            drop(state.writer.write().unwrap().merge(&run));
        }
    }
}

impl SubstringBackend {
    /// Starts a rewrite when wide segments hold at least a cap's worth of rows and none is in
    /// progress. The boundaries come from a sample of the wide segments' sort keys, so skewed
    /// keys still give ranges of about `cap` rows each. Called on idle ticks only: a plan made
    /// while a build is still committing samples the rows so far and cuts too few, too wide
    /// ranges, whose segments end up spanning several caps and never get revisited.
    fn maybe_plan_rewrite(&self, state: &IndexState<Self>) {
        if !*self.options.rewrite_wide_segments.as_ref() || self.plan.lock().unwrap().is_some() {
            return;
        }
        let Some(cap) = *self.options.segment_max_docs.as_ref() else {
            return;
        };
        let cap = cap.get();
        let searcher = state.reader.searcher();
        let searcher = &searcher;
        let geometry = self.geometry(searcher);
        let segments = geometry.as_slice();
        // Big wide segments trigger a plan; once there is one, everything that is not narrow is
        // in it, slivers included, since a sliver has nowhere narrow to merge.
        let wide = if wide_segments(segments, searcher.num_docs(), cap / 2, cap).is_empty() {
            Vec::new()
        } else {
            spread_segments(segments, searcher.num_docs(), cap)
        };
        let l0_docs: u64 = searcher
            .segment_readers()
            .iter()
            .filter(|segment| wide.contains(&segment.segment_id()))
            .map(|segment| u64::from(segment.num_docs()))
            .sum();
        if l0_docs < u64::from(cap) {
            return;
        }
        // Recorded per plan, so the size row of a finished rewrite still says what it moved.
        self.rewrite.l0_docs.store(l0_docs, Relaxed);
        let stride = (l0_docs / REWRITE_SAMPLE_TARGET).max(1) as u32;
        let mut sample = Vec::new();
        let mut opened = 0;
        for segment in searcher.segment_readers() {
            if !wide.contains(&segment.segment_id()) {
                continue;
            }
            let Ok(columns) = self.columns_for(segment, &mut opened) else {
                continue;
            };
            let alive = segment.alive_bitset();
            for doc_id in (0..segment.max_doc()).step_by(stride as usize) {
                if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                    sample.push(columns.sort.first(doc_id).unwrap_or(0));
                }
            }
        }
        if sample.is_empty() {
            return;
        }
        let count = l0_docs.div_ceil(u64::from(cap)) as usize;
        // Everything a step may take rows from: the spread segments, wide ones included.
        let spread = spread_segments(segments, searcher.num_docs(), cap);
        let (low, high) = segments
            .iter()
            .filter(|(_, _, _, id)| spread.contains(id))
            .fold((u64::MAX, 0u64), |(lo, hi), &(_, min, max, _)| {
                (lo.min(min), hi.max(max))
            });
        let cap_width =
            narrow_span(segments, searcher.num_docs(), cap) / WIDE_SEGMENT_FACTOR as u64;
        let ranges = quantile_ranges(sample, count, low, high, cap_width);
        info!(
            "substring: rewriting {l0_docs} rows of {} wide segments into {} ranges",
            wide.len(),
            ranges.len()
        );
        self.rewrite
            .ranges_total
            .store(ranges.len() as u64, Relaxed);
        self.rewrite.ranges_done.store(0, Relaxed);
        *self.plan.lock().unwrap() = Some(RewritePlan { ranges, next: 0 });
    }

    fn rewrite_in_progress(&self) -> bool {
        self.plan.lock().unwrap().is_some()
    }

    fn rewrite_totals(&self) -> RewriteTotals {
        RewriteTotals {
            ranges_total: self.rewrite.ranges_total.load(Relaxed),
            ranges_done: self.rewrite.ranges_done.load(Relaxed),
            docs_rewritten: self.rewrite.docs_rewritten.load(Relaxed),
            l0_docs: self.rewrite.l0_docs.load(Relaxed),
        }
    }
}

/// Moves the plan's next range: every live row of a segment that is not narrow now whose sort
/// key falls in it is deleted by primary id and re-added, and the writer is committed before and
/// after, all under the writer lock. The rows land in pieces spanning no more than the range,
/// narrow by construction, which the policy folds into one segment. The lock is what keeps a concurrent update or delete of one of those rows
/// from being undone by the re-add: it waits, and lands with a higher opstamp. Searches take the
/// reader, not this lock. Returns whether a range was processed.
fn rewrite_next_range(state: &SubstringIndexState, key: &IndexKey) -> anyhow::Result<bool> {
    let backend = &state.backend;
    let Some((low, high)) = backend
        .plan
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|plan| plan.ranges.get(plan.next).copied())
    else {
        return Ok(false);
    };
    let Some(cap) = *backend.options.segment_max_docs.as_ref() else {
        return Ok(false);
    };
    let text_field = state.schema.get_field(TEXT_FIELD).unwrap();
    let primary_id_field = state.schema.get_field(PRIMARY_ID_FIELD).unwrap();

    let mut writer = state.writer.write().unwrap();
    writer
        .commit(|| state.reader.reload())
        .map_err(|e| anyhow!("substring: failed to commit before a rewrite: {e}"))?;
    let searcher = state.reader.searcher();
    let wide = spread_segments(&backend.geometry(&searcher), searcher.num_docs(), cap.get());
    let mut moved = Vec::new();
    let mut opened = 0;
    for segment in searcher.segment_readers() {
        if !wide.contains(&segment.segment_id()) {
            continue;
        }
        let columns = backend.columns_for(segment, &mut opened)?;
        let alive = segment.alive_bitset();
        let store = segment
            .get_store_reader(STORE_CACHE_BLOCKS)
            .map_err(|e| anyhow!("substring: failed to open the document store: {e}"))?;
        for doc_id in 0..segment.max_doc() {
            if alive.is_some_and(|alive| !alive.is_alive(doc_id)) {
                continue;
            }
            let sort_key = columns.sort.first(doc_id).unwrap_or(0);
            if sort_key < low || sort_key >= high {
                continue;
            }
            let doc: TantivyDocument = store
                .get(doc_id)
                .map_err(|e| anyhow!("substring: failed to retrieve doc: {e}"))?;
            let primary_id = doc
                .get_first(primary_id_field)
                .and_then(|value| value.as_u64())
                .map(PrimaryId::from)
                .ok_or_else(|| anyhow!("substring: missing primary_id in doc"))?;
            let text = doc
                .get_first(text_field)
                .and_then(|value| value.as_str())
                .map(unframed)
                .unwrap_or_default();
            let new_doc = backend.create_doc(
                &state.schema,
                primary_id,
                SubstringRow {
                    text,
                    sort_key: Some(sort_key),
                },
            );
            moved.push((primary_id_term(&state.schema, primary_id), new_doc));
        }
    }
    let count = writer
        .rewrite_documents(moved)
        .map_err(|e| anyhow!("substring: failed to rewrite a range: {e}"))?;
    writer
        .commit(|| state.reader.reload())
        .map_err(|e| anyhow!("substring: failed to commit a rewrite: {e}"))?;
    drop(writer);

    backend
        .rewrite
        .docs_rewritten
        .fetch_add(count as u64, Relaxed);
    backend.rewrite.ranges_done.fetch_add(1, Relaxed);
    let finished = {
        let mut plan = backend.plan.lock().unwrap();
        if let Some(inner) = plan.as_mut() {
            inner.next += 1;
            if inner.next >= inner.ranges.len() {
                *plan = None;
                true
            } else {
                false
            }
        } else {
            true
        }
    };
    if finished {
        info!(
            "substring: rewrite of {key} finished, {} rows moved",
            backend.rewrite.docs_rewritten.load(Relaxed)
        );
    }
    // Learn the new segments' bounds and let the policy fold the pieces together.
    backend.after_reload(state);
    Ok(true)
}

type SubstringIndexState = IndexState<SubstringBackend>;

/// The value this row is ordered by, or `None` when the index has no sort column, the row has no
/// value for it, or the value has no ordering the index can use.
///
/// `None` is not an error: the document is still indexed, it just sorts lowest. Refusing the row
/// would make an ordered search disagree with an unordered one about which rows exist, which is a
/// worse failure than a row appearing last.
fn read_sort_key(
    table: &RwLock<impl TableSearch>,
    options: &IndexOptionsSubstring,
    partition_id: PartitionId,
    primary_id: PrimaryId,
) -> Option<u64> {
    let order_by = options.order_by.as_ref().as_ref()?;
    let value = table
        .read()
        .unwrap()
        .column_value_for(partition_id, primary_id, order_by)?;
    crate::cql_types::to_sort_key(&value)
}

/// Brings a value or a query to the form the index compares: unchanged for a case-sensitive
/// index, fully (Unicode) lowercased otherwise.
fn normalize(text: &str, case_sensitive: CaseSensitive) -> Cow<'_, str> {
    if *case_sensitive.as_ref() {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(text.to_lowercase())
    }
}

/// The distinct `gram_len`-character substrings of `text`, i.e. the terms the n-gram tokenizer
/// emits for it at that length. `text` must be at least `gram_len` characters long.
fn grams_of_length(text: &str, gram_len: usize) -> BTreeSet<String> {
    let chars: Vec<char> = text.chars().collect();
    chars
        .windows(gram_len)
        .map(|window| window.iter().collect())
        .collect()
}

/// How a normalized query is answered.
enum SubstringQuery {
    /// The query is short enough to be an indexed term itself, so every posting is an exact hit.
    Exact(Box<dyn Query>),
    /// The query is longer than any indexed term: candidates come from intersecting the
    /// postings of its `max_gram`-long substrings and must be verified against the stored value.
    /// The grams come along so that a segment lacking one of them can be ruled out without
    /// opening any posting list (`GramProbe`).
    Candidates(Box<dyn Query>, Vec<Term>),
}

/// Rules segments out for a query answered from several grams: a segment whose term dictionary
/// lacks any one of them holds no candidate, so it is skipped before the intersection opens a
/// posting list per gram -- which, for a keyword that occurs in a handful of names, is nearly
/// every segment, and what made a rare long keyword cost a full walk of every segment.
///
/// The gram last found missing is asked first (move to front): a rare keyword usually has one
/// rare gram, which is absent from nearly every segment, so after the first miss a segment costs
/// one dictionary lookup rather than one per gram plus opening their postings.
struct GramProbe {
    terms: Vec<Term>,
    field: tantivy::schema::Field,
}

impl GramProbe {
    fn new(terms: Vec<Term>, field: tantivy::schema::Field) -> Self {
        Self { terms, field }
    }

    /// Whether `segment` certainly holds no candidate.
    fn rules_out(&mut self, segment: &SegmentReader) -> anyhow::Result<bool> {
        let inverted = segment
            .inverted_index(self.field)
            .map_err(|e| anyhow!("substring: failed to open the term dictionary: {e}"))?;
        for i in 0..self.terms.len() {
            let info = inverted
                .get_term_info(&self.terms[i])
                .map_err(|e| anyhow!("substring: failed to look a gram up: {e}"))?;
            if info.is_none() {
                self.terms[..=i].rotate_right(1);
                return Ok(true);
            }
        }
        Ok(false)
    }
}

fn build_query(state: &SubstringIndexState, normalized: &str) -> anyhow::Result<SubstringQuery> {
    let text_field = state.schema.get_field(TEXT_FIELD).unwrap();
    let min_gram = state.backend.min_gram();
    let max_gram = state.backend.max_gram();
    let query_len = normalized.chars().count();

    if query_len == 0 {
        return Err(QueryError("substring: the query must not be empty".to_string()).into());
    }
    if query_len < min_gram {
        return Err(QueryError(format!(
            "substring: the query has {query_len} characters, \
            but the index only answers queries of at least min_gram={min_gram}"
        ))
        .into());
    }

    let term_query = |gram: &str| -> Box<dyn Query> {
        Box::new(TermQuery::new(
            Term::from_field_text(text_field, gram),
            IndexRecordOption::Basic,
        ))
    };

    if query_len <= max_gram {
        return Ok(SubstringQuery::Exact(term_query(normalized)));
    }

    // Only the longest grams take part: every shorter substring of the query is contained in
    // one of them, so it would add a posting list to intersect without narrowing the result.
    let grams = grams_of_length(normalized, max_gram);
    let clauses = grams.iter().map(|gram| term_query(gram)).collect();
    let terms = grams
        .iter()
        .map(|gram| Term::from_field_text(text_field, gram))
        .collect();
    Ok(SubstringQuery::Candidates(
        Box::new(BooleanQuery::intersection(clauses)),
        terms,
    ))
}

/// Walks the matching documents in index order and returns the primary ids of the first
/// `limit` verified matches after skipping `offset` of them.
///
/// The query is executed once, unscored, and the walk stops as soon as enough matches are
/// found, so a hot single-character query does not pay for its whole posting list. The stored
/// document has to be read anyway for the primary id, so verifying the containment on the way
/// costs no extra I/O.
/// The window of `(sort key, primary id)` pairs a search may return, in *walk orientation*: the
/// walk always takes the highest pair first, and an ascending search is the same walk over the
/// bitwise complement of every key, which reverses the order of both members. The range from the
/// query's `WHERE` and the cursor from the previous page constrain the same value, so they are
/// resolved into one pair of bounds once rather than checked separately on every candidate.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SortWindow {
    asc: bool,
    lower: Option<(u64, u64)>,
    upper: Option<Bound<(u64, u64)>>,
}

/// The primary id a cursor resumes below when its row's id is not known (the row was deleted,
/// or the cursor never named one): the id every tied row's oriented id is below, so all of them
/// are taken again rather than any skipped.
const TIE_UNKNOWN: u64 = u64::MAX;

impl SortWindow {
    /// `cursor` is the previous page's last `(sort key, primary id)` in the sort column's own
    /// space; the page resumes strictly below it in walk orientation.
    pub(crate) fn new(
        order: SortOrder,
        cursor: Option<(u64, u64)>,
        min_sort_key: Option<u64>,
        max_sort_key: Option<u64>,
    ) -> Self {
        let asc = order == SortOrder::Asc;
        let flip = |key: u64| if asc { !key } else { key };
        // In walk orientation the range's lower bound is the real minimum for a descending walk
        // and the complemented real maximum for an ascending one, and the upper bound the other
        // way round; a bound includes the key it names, the cursor excludes it.
        let lower = if asc { max_sort_key } else { min_sort_key }.map(|key| (flip(key), 0));
        let range_upper =
            if asc { min_sort_key } else { max_sort_key }.map(|key| (flip(key), u64::MAX));
        let cursor = cursor.map(|(sort_key, primary_id)| (flip(sort_key), flip(primary_id)));
        let upper = match (cursor, range_upper) {
            (Some(cursor), Some(range)) if range < cursor => Some(Bound::Included(range)),
            (Some(cursor), _) => Some(Bound::Excluded(cursor)),
            (None, Some(range)) => Some(Bound::Included(range)),
            (None, None) => None,
        };
        Self { asc, lower, upper }
    }

    /// A key from the column, in walk orientation.
    fn orient(&self, key: u64) -> u64 {
        if self.asc { !key } else { key }
    }

    /// A segment's real bounds, in walk orientation.
    fn orient_bounds(&self, min: u64, max: u64) -> (u64, u64) {
        if self.asc { (!max, !min) } else { (min, max) }
    }

    fn contains(&self, entry: (u64, u64)) -> bool {
        let above_lower = self.lower.is_none_or(|lower| entry >= lower);
        let below_upper = match self.upper {
            None => true,
            Some(Bound::Included(upper)) => entry <= upper,
            Some(Bound::Excluded(upper)) => entry < upper,
            Some(Bound::Unbounded) => true,
        };
        above_lower && below_upper
    }

    /// Whether a segment spanning `[min, max]` (oriented) can hold anything in the window.
    /// Answered from the segment's bounds alone, so a segment ruled out here is never opened. A
    /// segment ending exactly at an excluded upper key may still hold rows below it on the tie,
    /// so it stays in.
    fn overlaps(&self, min: u64, max: u64) -> bool {
        let above = self.lower.is_none_or(|(lower, _)| max >= lower);
        let below = match self.upper {
            None => true,
            Some(Bound::Included((upper, _))) | Some(Bound::Excluded((upper, _))) => min <= upper,
            Some(Bound::Unbounded) => true,
        };
        above && below
    }
}

/// One page of an ordered search: the `limit` highest `(sort key, primary id)` pairs in the
/// window, highest first, with where the next page resumes. (For an ascending search, read
/// "highest" in walk orientation; see [`SortWindow`].)
///
/// Two things keep this off the O(matches) path the obvious implementation lands on. Segments are
/// visited by descending upper bound and the walk stops once the next segment's bound cannot beat
/// the page's weakest entry, which skips whole segments unopened. And within a segment the sort key
/// -- a columnar read -- is checked before the document store is touched, so a candidate that
/// cannot make the page costs almost nothing. Measured, the second is worth 4-13x on its own and
/// does not depend on how the segments are laid out.
///
/// Returns the page's primary ids and, when the page filled, the real `(sort key, primary id)`
/// of its last row for the caller to turn into a cursor.
/// The page's primary ids and, when the page filled, the real `(sort key, primary id)` of its
/// last row.
/// The rows of one ordered page, where the next page resumes, and whether the rows are verified
/// matches rather than candidates.
type OrderedPage = (Vec<PrimaryId>, Option<(u64, PrimaryId)>, bool);

fn collect_matches_ordered(
    state: &SubstringIndexState,
    normalized: &str,
    limit: usize,
    window: SortWindow,
    verify: bool,
) -> anyhow::Result<OrderedPage> {
    let text_field = state.schema.get_field(TEXT_FIELD).unwrap();
    let primary_id_field = state.schema.get_field(PRIMARY_ID_FIELD).unwrap();

    let (query, has_candidates, mut probe) = match build_query(state, normalized)? {
        SubstringQuery::Exact(query) => (query, false, None),
        SubstringQuery::Candidates(query, terms) => {
            (query, true, Some(GramProbe::new(terms, text_field)))
        }
    };
    // A caller that declines verification takes the candidates as they are: the walk then
    // treats them as exact matches, which is the cheap branch below, and the page says so.
    let needs_verification = has_candidates && verify;
    let verified = !has_candidates || verify;

    let searcher = state.reader.searcher();
    let weight = query
        .weight(EnableScoring::disabled_from_searcher(&searcher))
        .map_err(|e| anyhow!("substring: failed to build the query: {e}"))?;

    let started = Instant::now();
    let mut tally = WalkTally::default();
    let mut segments = Vec::with_capacity(searcher.segment_readers().len());
    for (segment_ord, segment) in searcher.segment_readers().iter().enumerate() {
        let columns = state
            .backend
            .columns_for(segment, &mut tally.column_opens)?;
        let (lower_bound, upper_bound) = window.orient_bounds(columns.sort_min, columns.sort_max);
        // Ruled out from the bounds alone, so the segment is never opened.
        if window.overlaps(lower_bound, upper_bound) {
            let primary_id = columns.primary_id.clone().ok_or_else(|| {
                anyhow!("substring: an ordered index without a primary id column")
            })?;
            segments.push((
                segment_ord as u32,
                segment,
                columns.sort,
                primary_id,
                upper_bound,
            ));
        }
    }
    state.backend.prune_columns(searcher.segment_readers());
    segments.sort_by_key(|(_, _, _, _, upper_bound)| Reverse(*upper_bound));
    tally.segments_considered = segments.len() as u64;
    let prepare = started.elapsed();

    // Min-heap of the best `limit` so far, so the root is the entry to beat. An entry is the
    // oriented (sort key, primary id) pair -- the id breaks ties on the sort key -- and the
    // document's address; the address is what the page is resolved from at the end, once, since
    // reading the store for every entrant is what dominated the first walk (9-27x the walk
    // itself, benches/substring_order.rs).
    type Entry = (u64, u64, DocAddress);
    let mut best: BinaryHeap<Reverse<Entry>> = BinaryHeap::with_capacity(limit + 1);
    // Whether a pair can still enter the page: always while it is not full, and above the
    // weakest entry once it is.
    let beats_page = |best: &BinaryHeap<Reverse<Entry>>, pair: (u64, u64)| {
        best.len() < limit
            || best
                .peek()
                .is_none_or(|Reverse((sort_key, primary_id, _))| pair > (*sort_key, *primary_id))
    };
    let mut candidates: Vec<(u64, u64, u32)> = Vec::new();
    for (segment_ord, segment, sort_column, id_column, upper_bound) in segments {
        if best.len() == limit
            && let Some(Reverse((weakest, _, _))) = best.peek()
            && upper_bound < *weakest
        {
            // Nothing in this segment, nor in any later one, can enter the page. (A segment
            // ending exactly at the weakest key may still hold a higher id on the tie.)
            break;
        }
        let pair_of = |doc_id: u32| {
            (
                window.orient(sort_column.first(doc_id).unwrap_or(0)),
                window.orient(id_column.first(doc_id).unwrap_or(0)),
            )
        };

        if let Some(probe) = probe.as_mut()
            && probe.rules_out(segment)?
        {
            continue;
        }
        tally.segments_opened += 1;
        let mut scorer = weight
            .scorer(segment, 1.0)
            .map_err(|e| anyhow!("substring: failed to run the query: {e}"))?;
        let alive = segment.alive_bitset();

        if !needs_verification {
            // Every match is exact, so it enters the page straight from the column.
            let mut doc_id = scorer.doc();
            while doc_id != TERMINATED {
                tally.postings_scanned += 1;
                if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                    let pair = pair_of(doc_id);
                    if window.contains(pair) && beats_page(&best, pair) {
                        tally.heap_entrants += 1;
                        best.push(Reverse((
                            pair.0,
                            pair.1,
                            DocAddress::new(segment_ord, doc_id),
                        )));
                        if best.len() > limit {
                            best.pop();
                        }
                    }
                }
                doc_id = scorer.advance();
            }
            continue;
        }

        // Past max_gram the grams only nominate candidates, and the text that decides is in the
        // document store. Postings arrive in doc order, which is roughly ascending sort order
        // within a segment, so verifying as they come reads nearly every candidate: each later
        // one beats the page the earlier ones built. Two passes instead: gather the candidates'
        // sort keys from the column, then verify from the highest down and stop at the first
        // that can no longer enter the page. The store is read for the page's rows plus the
        // false positives above them, whatever the segment holds.
        let store = segment
            .get_store_reader(STORE_CACHE_BLOCKS)
            .map_err(|e| anyhow!("substring: failed to open the document store: {e}"))?;
        if *state.backend.options.verify_in_order.as_ref() {
            // The first stage-2 build's walk, kept to measure the two passes against.
            let mut doc_id = scorer.doc();
            while doc_id != TERMINATED {
                tally.postings_scanned += 1;
                if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                    let pair = pair_of(doc_id);
                    if window.contains(pair) && beats_page(&best, pair) {
                        tally.store_reads += 1;
                        let verified = store
                            .get::<TantivyDocument>(doc_id)
                            .map_err(|e| anyhow!("substring: failed to retrieve doc: {e}"))?
                            .get_first(text_field)
                            .and_then(|value| value.as_str())
                            .is_some_and(|text| text.contains(normalized));
                        if verified {
                            tally.heap_entrants += 1;
                            best.push(Reverse((
                                pair.0,
                                pair.1,
                                DocAddress::new(segment_ord, doc_id),
                            )));
                            if best.len() > limit {
                                best.pop();
                            }
                        }
                    }
                }
                doc_id = scorer.advance();
            }
            continue;
        }
        candidates.clear();
        let mut doc_id = scorer.doc();
        while doc_id != TERMINATED {
            tally.postings_scanned += 1;
            if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                let pair = pair_of(doc_id);
                if window.contains(pair) && beats_page(&best, pair) {
                    candidates.push((pair.0, pair.1, doc_id));
                }
            }
            doc_id = scorer.advance();
        }
        candidates.sort_unstable_by(|a, b| b.cmp(a));
        for &(sort_key, primary_id, doc_id) in &candidates {
            if !beats_page(&best, (sort_key, primary_id)) {
                break;
            }
            tally.store_reads += 1;
            let verified = store
                .get::<TantivyDocument>(doc_id)
                .map_err(|e| anyhow!("substring: failed to retrieve doc: {e}"))?
                .get_first(text_field)
                .and_then(|value| value.as_str())
                .is_some_and(|text| text.contains(normalized));
            if verified {
                tally.heap_entrants += 1;
                best.push(Reverse((
                    sort_key,
                    primary_id,
                    DocAddress::new(segment_ord, doc_id),
                )));
                if best.len() > limit {
                    best.pop();
                }
            }
        }
    }

    // A cursor says there may be more below this point. That is only true if the page filled: a
    // walk that ended with fewer than `limit` matches visited every segment overlapping the window,
    // so there is nothing left to resume from and a cursor would only cost the caller an empty
    // round trip. (Rows dropped later, in `handle_search`, shorten the page after this point and
    // must not suppress the cursor -- which is why this asks the heap, not the returned page.)
    let resume_at = (best.len() == limit)
        .then(|| {
            best.peek().map(|Reverse((sort_key, primary_id, _))| {
                // Back from walk orientation to the column's own values.
                (
                    window.orient(*sort_key),
                    PrimaryId::from(window.orient(*primary_id)),
                )
            })
        })
        .flatten();
    let mut page: Vec<Entry> = best.into_iter().map(|Reverse(entry)| entry).collect();
    page.sort_unstable_by_key(|(sort_key, primary_id, _)| Reverse((*sort_key, *primary_id)));
    let resolving = Instant::now();
    let ids = if *state.backend.options.primary_id_fast.as_ref() {
        page.into_iter()
            .map(|(_, _, address)| {
                let segment = &searcher.segment_readers()[address.segment_ord as usize];
                state
                    .backend
                    .columns_for(segment, &mut tally.column_opens)?
                    .primary_id
                    .as_ref()
                    .and_then(|column| column.first(address.doc_id))
                    .map(PrimaryId::from)
                    .ok_or_else(|| anyhow!("substring: missing primary_id in the column"))
            })
            .collect::<anyhow::Result<_>>()?
    } else {
        tally.store_reads += page.len() as u64;
        page.into_iter()
            .map(|(_, _, address)| {
                let doc: TantivyDocument = searcher
                    .doc(address)
                    .map_err(|e| anyhow!("substring: failed to retrieve doc: {e}"))?;
                doc.get_first(primary_id_field)
                    .and_then(|value| value.as_u64())
                    .map(PrimaryId::from)
                    .ok_or_else(|| anyhow!("substring: missing primary_id in doc"))
            })
            .collect::<anyhow::Result<_>>()?
    };
    let page_resolve = resolving.elapsed();
    state.backend.walk.record(
        tally,
        WalkTimes {
            walk: started.elapsed(),
            prepare,
            page_resolve,
        },
    );
    Ok((ids, resume_at, verified))
}

fn collect_matches(
    state: &SubstringIndexState,
    normalized: &str,
    limit: usize,
    offset: usize,
    verify: bool,
) -> anyhow::Result<(Vec<PrimaryId>, bool)> {
    let text_field = state.schema.get_field(TEXT_FIELD).unwrap();
    let primary_id_field = state.schema.get_field(PRIMARY_ID_FIELD).unwrap();

    let (query, has_candidates, mut probe) = match build_query(state, normalized)? {
        SubstringQuery::Exact(query) => (query, false, None),
        SubstringQuery::Candidates(query, terms) => {
            (query, true, Some(GramProbe::new(terms, text_field)))
        }
    };
    // The store is read here anyway, for the primary id, so declining verification saves the
    // text comparison only; it exists so that both walks answer the request the same way.
    let needs_verification = has_candidates && verify;
    let verified = !has_candidates || verify;

    let searcher = state.reader.searcher();
    let weight = query
        .weight(EnableScoring::disabled_from_searcher(&searcher))
        .map_err(|e| anyhow!("substring: failed to build the query: {e}"))?;

    let started = Instant::now();
    let mut tally = WalkTally {
        segments_considered: searcher.segment_readers().len() as u64,
        ..WalkTally::default()
    };
    let mut to_skip = offset;
    let mut matches = Vec::with_capacity(limit);
    'segments: for segment in searcher.segment_readers() {
        if let Some(probe) = probe.as_mut()
            && probe.rules_out(segment)?
        {
            continue;
        }
        tally.segments_opened += 1;
        let mut scorer = weight
            .scorer(segment, 1.0)
            .map_err(|e| anyhow!("substring: failed to run the query: {e}"))?;
        let alive = segment.alive_bitset();
        let store = segment
            .get_store_reader(STORE_CACHE_BLOCKS)
            .map_err(|e| anyhow!("substring: failed to open the document store: {e}"))?;

        let mut doc_id = scorer.doc();
        while doc_id != TERMINATED {
            tally.postings_scanned += 1;
            if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                tally.store_reads += 1;
                let doc: TantivyDocument = store
                    .get(doc_id)
                    .map_err(|e| anyhow!("substring: failed to retrieve doc: {e}"))?;
                let verified = !needs_verification
                    || doc
                        .get_first(text_field)
                        .and_then(|value| value.as_str())
                        .is_some_and(|text| text.contains(normalized));
                if verified {
                    tally.heap_entrants += 1;
                    if to_skip > 0 {
                        to_skip -= 1;
                    } else {
                        let raw_id = doc
                            .get_first(primary_id_field)
                            .and_then(|value| value.as_u64())
                            .ok_or_else(|| anyhow!("substring: missing primary_id in doc"))?;
                        matches.push(PrimaryId::from(raw_id));
                        if matches.len() == limit {
                            break 'segments;
                        }
                    }
                }
            }
            doc_id = scorer.advance();
        }
    }
    state.backend.walk.record(
        tally,
        WalkTimes {
            walk: started.elapsed(),
            ..WalkTimes::default()
        },
    );
    Ok((matches, verified))
}

fn handle_substring_stats(state: &SubstringIndexState) -> SubstringStatsR {
    let tantivy = handle_stats(state)?;
    let searcher = state.reader.searcher();
    let segments = searcher
        .segment_readers()
        .iter()
        .map(|segment| {
            let mut opened = 0;
            let bounds = state
                .backend
                .orders_results()
                .then(|| state.backend.columns_for(segment, &mut opened))
                .transpose()?
                .map(|columns| (columns.sort_min, columns.sort_max));
            Ok(SegmentLayout {
                docs: segment.num_docs(),
                deleted: segment.num_deleted_docs(),
                sort_min: bounds.map(|(min, _)| min),
                sort_max: bounds.map(|(_, max)| max),
            })
        })
        .collect::<anyhow::Result<_>>()?;
    Ok(SubstringStats {
        tantivy,
        walk: state.backend.walk.snapshot(),
        rewrite: state.backend.rewrite_totals(),
        segments,
    })
}

// One argument per thing the request says; bundling them would only move the count elsewhere.
#[allow(clippy::too_many_arguments)]
fn handle_search(
    state: &SubstringIndexState,
    table: &RwLock<impl TableSearch>,
    index_key: &IndexKey,
    query: &str,
    kind: MatchKind,
    limit: Limit,
    offset: usize,
    window: SearchWindow,
    verify: bool,
) -> SubstringSearchR {
    // A prefix or suffix query is a containment query for the keyword with the value's frame
    // mark on the anchored side, so from here on nothing knows the kind.
    let normalized = pattern_of(
        kind,
        &normalize(query, state.backend.options.case_sensitive),
    );
    let limit: usize = (*limit.as_ref()).into();
    let (primary_ids, resume_at, verified) = if state.backend.orders_results() {
        // The cursor names its row by primary key; the walk breaks ties by primary id, so the
        // key is looked up first. A key the table no longer knows (the row was deleted since)
        // resumes at TIE_UNKNOWN, which takes the rows tied with it again rather than skip any.
        let cursor = window.cursor.as_ref().map(|cursor| {
            let table = table.read().unwrap();
            let primary_id = cursor
                .primary_key
                .as_ref()
                .and_then(|key| table.primary_id(key))
                .map_or(TIE_UNKNOWN, u64::from);
            // TIE_UNKNOWN is "above every id" in walk orientation; in the column's own space
            // that is the complement for an ascending walk.
            let primary_id = if primary_id == TIE_UNKNOWN && window.order == SortOrder::Asc {
                !TIE_UNKNOWN
            } else {
                primary_id
            };
            (cursor.sort_key, primary_id)
        });
        let sort_window = SortWindow::new(
            window.order,
            cursor,
            window.min_sort_key,
            window.max_sort_key,
        );
        collect_matches_ordered(state, &normalized, limit, sort_window, verify)?
    } else {
        let (ids, verified) = collect_matches(state, &normalized, limit, offset, verify)?;
        (ids, None, verified)
    };

    let table = table.read().unwrap();
    let partition_id = find_partition_id::<SubstringBackend>(table.deref(), index_key)?;
    // A row the table cache no longer knows was deleted after the index snapshot was taken;
    // it is simply left out, as the full-text search does. Note this can shorten a page below
    // `limit` without the search being exhausted, which is why the cursor is reported separately
    // rather than inferred from the page being short.
    Ok(SubstringPage {
        primary_keys: primary_ids
            .into_iter()
            .filter_map(|primary_id| table.primary_key(partition_id, primary_id))
            .collect(),
        next_cursor: resume_at.map(|(sort_key, primary_id)| Cursor {
            sort_key,
            primary_key: table.primary_key(partition_id, primary_id),
        }),
        verified,
    })
}

pub(crate) fn new(
    index: SubstringIndexConfiguration,
    table: Arc<RwLock<impl TableSearch + Send + Sync + 'static>>,
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
    commit_interval: Duration,
    commit_threshold: usize,
    metrics: Option<Arc<Metrics>>,
) -> mpsc::Sender<SubstringIndex> {
    let (tx, mut rx) = mpsc::channel::<SubstringIndex>(perf::channel_size().into());
    tokio::spawn(async move {
        let key = index.key.clone();
        // The layout changes on the ticks -- merges after the writes stop, a rewrite in progress
        // -- with no write or search to mark the index for a refresh, so the ticks mark it.
        let mark_dirty = |key: &IndexKey| {
            if let Some(metrics) = &metrics {
                metrics.mark_dirty(key.keyspace().as_ref(), key.index().as_ref());
            }
        };
        debug!("substring index actor starting for {key}");
        let mut states: BTreeMap<IndexId, Arc<SubstringIndexState>> = BTreeMap::new();
        let make_backend = || SubstringBackend {
            walk: WalkCounters::default(),
            columns: RwLock::new(HashMap::new()),
            bounds: Arc::new(RwLock::new(HashMap::new())),
            plan: Mutex::new(None),
            rewrite: RewriteCounters::default(),
            options: index.options.clone(),
        };

        let mut allocate_prev = Allocate::Can;
        let allocate_rx = memory.subscribe_allocate().await;

        let mut interval = tokio::time::interval(commit_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                msg = rx.recv() => {
                    let Some(msg) = msg else {
                        break;
                    };
                    match msg {
                        SubstringIndex::AddDocument {
                            partition_id,
                            primary_id,
                            document,
                            in_progress,
                        } => {
                            // Read before the index state is touched: Table::upsert writes the
                            // column before it emits the operation that got us here, so the value
                            // is already in place.
                            let sort_key = read_sort_key(
                                table.as_ref(),
                                &index.options,
                                partition_id,
                                primary_id,
                            );
                            let Some(state) = get_or_create_state(
                                &mut states,
                                table.as_ref(),
                                &key,
                                make_backend,
                            ) else {
                                continue;
                            };
                            if !can_allocate_memory(&allocate_rx, &mut allocate_prev, &key) {
                                continue;
                            }
                            let key = key.clone();
                            worker
                                .spawn_blocking(move || {
                                    let pending = handle_add_document(
                                        &state,
                                        primary_id,
                                        SubstringRow {
                                            text: &document,
                                            sort_key,
                                        },
                                        in_progress,
                                    );
                                    if pending >= commit_threshold {
                                        commit(&state, &key);
                                    }
                                })
                                .await;
                        }
                        SubstringIndex::RemoveDocument {
                            primary_id,
                            in_progress,
                        } => {
                            let Some(state) = get_or_create_state(
                                &mut states,
                                table.as_ref(),
                                &key,
                                make_backend,
                            ) else {
                                continue;
                            };
                            let key = key.clone();
                            worker
                                .spawn_blocking(move || {
                                    let pending =
                                        handle_remove_document(&state, primary_id, in_progress);
                                    if pending >= commit_threshold {
                                        commit(&state, &key);
                                    }
                                })
                                .await;
                        }
                        SubstringIndex::Count { tx, index_key, .. } => {
                            let result = get_state(&states, table.as_ref(), &index_key)
                                .map(|s| s.reader.searcher().num_docs() as usize)
                                .unwrap_or(0);
                            _ = tx.send(Ok(result));
                        }
                        SubstringIndex::Search {
                            index_key,
                            query,
                            kind,
                            limit,
                            offset,
                            window,
                            verify,
                            tx,
                        } => {
                            let Some(state) = get_state(&states, table.as_ref(), &index_key) else {
                                _ = tx.send(Ok(SubstringPage {
                                    primary_keys: vec![],
                                    next_cursor: None,
                                    verified: true,
                                }));
                                continue;
                            };
                            let table = Arc::clone(&table);
                            worker
                                .spawn_blocking(move || {
                                    let result = handle_search(
                                        &state,
                                        table.as_ref(),
                                        &index_key,
                                        &query,
                                        kind,
                                        limit,
                                        offset,
                                        window,
                                        verify,
                                    );
                                    _ = tx.send(result);
                                })
                                .await;
                        }
                        SubstringIndex::Stats { index_key, tx } => {
                            let Some(state) = get_state(&states, table.as_ref(), &index_key)
                            else {
                                _ = tx.send(Ok(Default::default()));
                                continue;
                            };
                            worker
                                .spawn_blocking(move || {
                                    let result = handle_substring_stats(&state);
                                    _ = tx.send(result);
                                })
                                .await;
                        }
                    }
                }
                _ = interval.tick() => {
                    for state in states.values() {
                        let pending = state.writer.read().unwrap().has_uncommitted_docs();
                        let state = Arc::clone(state);
                        let key = key.clone();
                        if pending {
                            worker.spawn_blocking(move || commit(&state, &key)).await;
                            mark_dirty(&index.key);
                        } else {
                            // Merges finish after the writes stop; let the reader see them. Then
                            // one range of a rewrite in progress, so the rewrite paces itself to
                            // the ticks and never starves ingestion or searches.
                            worker
                                .spawn_blocking(move || {
                                    reload(&state, &key);
                                    if !state.backend.rewrite_in_progress() {
                                        state.backend.maybe_plan_rewrite(&state);
                                    }
                                    if state.backend.rewrite_in_progress()
                                        && let Err(err) = rewrite_next_range(&state, &key)
                                    {
                                        error!("substring: rewrite of {key} failed: {err}");
                                    }
                                })
                                .await;
                            mark_dirty(&index.key);
                        }
                    }
                }
            }
        }
        debug!("substring index actor finished for {key}");
    });
    tx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AsyncInProgress;
    use crate::IndexKey;
    use crate::MaxGram;
    use crate::MinGram;
    use crate::OrderBy;
    use crate::PrimaryIdFast;
    use crate::PrimaryKey;
    use crate::RewriteWideSegments;
    use crate::SegmentMaxDocs;
    use crate::VerifyInOrder;
    use crate::table::IndexIdGenerator;
    use crate::table::MockTableSearch;
    use crate::table::PartitionId;
    use crate::worker;
    use rstest::rstest;
    use scylla::value::CqlValue;
    use std::num::NonZeroUsize;
    use tokio::sync::watch;

    use super::super::actor::SubstringIndexExt;

    /// The partition every test document lives in: these tests use a global index, so there is
    /// only one, and it has to match what `make_table_with_keys` hands back.
    fn test_partition_id() -> PartitionId {
        PartitionId::global(IndexIdGenerator::new().next(true).unwrap())
    }

    fn make_table_with_keys() -> Arc<RwLock<MockTableSearch>> {
        let index_id = IndexIdGenerator::new().next(true).unwrap();
        let partition_id = PartitionId::global(index_id);
        let mut mock = MockTableSearch::new();
        mock.expect_index_id()
            .returning(move |_index_key| Some(index_id));
        mock.expect_partition_id()
            .returning(move |_index_key, _restrictions| Some((partition_id, None)));
        mock.expect_primary_key()
            .returning(|_partition_id, primary_id| {
                let id_val = u64::from(primary_id);
                Some(PrimaryKey::from(vec![CqlValue::BigInt(id_val as i64)]))
            });
        mock.expect_primary_id()
            .returning(|primary_key| match primary_key.get(0) {
                Some(CqlValue::BigInt(id)) => Some(PrimaryId::from(id as u64)),
                _ => None,
            });
        // The sort column mirrors the primary id, so "newest first" is "highest id first" and a
        // test can assert on plain integers.
        mock.expect_column_value_for()
            .returning(|_partition_id, primary_id, _column| {
                Some(CqlValue::BigInt(u64::from(primary_id) as i64))
            });
        Arc::new(RwLock::new(mock))
    }

    fn make_index_key() -> IndexKey {
        IndexKey::new(&"ks".into(), &"idx".into())
    }

    fn make_memory_actor() -> mpsc::Sender<Memory> {
        let (tx, mut rx) = mpsc::channel::<Memory>(1);
        tokio::spawn(async move {
            let (watch_tx, _) = watch::channel(Allocate::Can);
            while let Some(msg) = rx.recv().await {
                match msg {
                    Memory::SubscribeAllocate { tx } => {
                        let _ = tx.send(watch_tx.subscribe());
                    }
                }
            }
        });
        tx
    }

    const TEST_COMMIT_INTERVAL: Duration = Duration::from_millis(50);
    const TEST_COMMIT_THRESHOLD: usize = 100;

    fn options(min_gram: usize, max_gram: usize, case_sensitive: bool) -> IndexOptionsSubstring {
        IndexOptionsSubstring {
            min_gram: MinGram::from(NonZeroUsize::new(min_gram).unwrap()),
            max_gram: MaxGram::from(NonZeroUsize::new(max_gram).unwrap()),
            case_sensitive: CaseSensitive::from(case_sensitive),
            order_by: OrderBy::default(),
            primary_id_fast: PrimaryIdFast::default(),
            segment_max_docs: SegmentMaxDocs::default(),
            verify_in_order: VerifyInOrder::default(),
            rewrite_wide_segments: RewriteWideSegments::default(),
        }
    }

    /// Options with a sort column. The mock table answers `column_value_for` with the primary id,
    /// whatever the column is called.
    /// The sort key the index stores for a row whose sort column holds `value`.
    ///
    /// Not the value itself: signed types are biased so that negatives sort below positives, and
    /// the window bounds are in that space. A caller building a window from a CQL value has to
    /// apply the same conversion -- see the note on `PostIndexContainsRequest`.
    fn sort_key(value: i64) -> u64 {
        crate::cql_types::to_sort_key(&CqlValue::BigInt(value)).unwrap()
    }

    fn ordered_options() -> IndexOptionsSubstring {
        IndexOptionsSubstring {
            order_by: "sort_col".parse().unwrap(),
            ..IndexOptionsSubstring::default()
        }
    }

    fn make_sender_with_options(options: IndexOptionsSubstring) -> mpsc::Sender<SubstringIndex> {
        make_sender_with_threshold(options, TEST_COMMIT_THRESHOLD)
    }

    fn make_sender_with_threshold(
        options: IndexOptionsSubstring,
        commit_threshold: usize,
    ) -> mpsc::Sender<SubstringIndex> {
        new(
            SubstringIndexConfiguration {
                key: make_index_key(),
                options,
            },
            make_table_with_keys(),
            worker::new(),
            make_memory_actor(),
            TEST_COMMIT_INTERVAL,
            commit_threshold,
            None,
        )
    }

    /// The defaults: every substring of 1 to 3 characters, case-sensitive.
    fn make_sender() -> mpsc::Sender<SubstringIndex> {
        make_sender_with_options(IndexOptionsSubstring::default())
    }

    async fn add_doc(sender: &mpsc::Sender<SubstringIndex>, primary: u64, content: &str) {
        let (tx, mut rx) = mpsc::channel(1);
        sender
            .add_document(
                test_partition_id(),
                primary.into(),
                content.into(),
                AsyncInProgress::Fullscan(tx),
            )
            .await
            .unwrap();
        rx.recv().await;
    }

    async fn rm_doc_no_wait(sender: &mpsc::Sender<SubstringIndex>, primary: u64) {
        let (tx, _rx) = mpsc::channel(1);
        sender
            .remove_document(primary.into(), AsyncInProgress::Fullscan(tx))
            .await
            .unwrap();
    }

    async fn rm_doc(sender: &mpsc::Sender<SubstringIndex>, primary: u64) {
        let (tx, mut rx) = mpsc::channel(1);
        sender
            .remove_document(primary.into(), AsyncInProgress::Fullscan(tx))
            .await
            .unwrap();
        rx.recv().await;
    }

    async fn add_docs(sender: &mpsc::Sender<SubstringIndex>, docs: &[(u64, &str)]) {
        for (primary, content) in docs {
            add_doc(sender, *primary, content).await;
        }
    }

    fn limit(n: usize) -> Limit {
        Limit::from(NonZeroUsize::new(n).unwrap())
    }

    /// The primary ids matching `query`, sorted, so tests are independent of index order.
    async fn search(sender: &mpsc::Sender<SubstringIndex>, query: &str) -> Vec<i64> {
        search_page(sender, query, 100, 0).await
    }

    /// Like `search_page`, but keeps the order the index returned -- the sorting helpers above
    /// cannot tell a correctly ordered answer from a scrambled one.
    async fn search_ordered(
        sender: &mpsc::Sender<SubstringIndex>,
        query: &str,
        limit_n: usize,
        window: SearchWindow,
    ) -> (Vec<i64>, Option<Cursor>) {
        let page = sender
            .search(
                make_index_key(),
                query.into(),
                MatchKind::Contains,
                limit(limit_n),
                0,
                window,
                true,
            )
            .await
            .unwrap();
        let ids = page
            .primary_keys
            .iter()
            .map(|pk| match pk.get(0).unwrap() {
                CqlValue::BigInt(id) => id,
                other => panic!("unexpected primary key value {other:?}"),
            })
            .collect();
        (ids, page.next_cursor)
    }

    /// A window whose cursor is the row that sorts at `cursor`: in these fixtures the sort column
    /// mirrors the primary id, so the row is named from the key's value.
    fn window(
        cursor: Option<u64>,
        min_sort_key: Option<u64>,
        max_sort_key: Option<u64>,
    ) -> SearchWindow {
        SearchWindow {
            order: SortOrder::Desc,
            cursor: cursor.map(|sort_key| Cursor {
                sort_key,
                primary_key: Some(PrimaryKey::from(vec![CqlValue::BigInt(
                    (sort_key ^ (1 << 63)) as i64,
                )])),
            }),
            min_sort_key,
            max_sort_key,
        }
    }

    /// The window that resumes after a page, in the given direction.
    fn resume(order: SortOrder, cursor: Cursor) -> SearchWindow {
        SearchWindow {
            order,
            cursor: Some(cursor),
            ..SearchWindow::default()
        }
    }

    fn ascending() -> SearchWindow {
        SearchWindow {
            order: SortOrder::Asc,
            ..SearchWindow::default()
        }
    }

    async fn search_page(
        sender: &mpsc::Sender<SubstringIndex>,
        query: &str,
        limit_n: usize,
        offset: usize,
    ) -> Vec<i64> {
        let mut ids: Vec<i64> = sender
            .search(
                make_index_key(),
                query.into(),
                MatchKind::Contains,
                limit(limit_n),
                offset,
                SearchWindow::default(),
                true,
            )
            .await
            .unwrap()
            .primary_keys
            .into_iter()
            .map(|pk| match pk.get(0).unwrap() {
                CqlValue::BigInt(id) => id,
                other => panic!("unexpected primary key value {other:?}"),
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    const NICKNAMES: &[(u64, &str)] = &[
        (1, "宇将军"),
        (2, "李将军"),
        (3, "将军来了"),
        (4, "将领"),
        (5, "元帅"),
        (6, "南宫月"),
        (7, "小南宫粉丝团"),
        (8, "宫南"),
    ];

    const USERNAMES: &[(u64, &str)] = &[
        (1, "NGgamer"),
        (2, "kingNG"),
        (3, "gn"),
        (4, "925555"),
        (5, "9255551"),
        (6, "925"),
    ];

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn two_char_query_matches_infix_but_not_a_different_word() {
        let sender = make_sender();
        add_docs(&sender, NICKNAMES).await;

        // 将领 shares the first character only; 元帅 is a synonym, which is irrelevant here.
        assert_eq!(search(&sender, "将军").await, vec![1, 2, 3]);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn query_is_order_sensitive() {
        let sender = make_sender();
        add_docs(&sender, NICKNAMES).await;

        // 宫南 has the same two characters in the other order.
        assert_eq!(search(&sender, "南宫").await, vec![6, 7]);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn single_char_query_is_served() {
        let sender = make_sender();
        add_docs(&sender, NICKNAMES).await;

        assert_eq!(search(&sender, "宫").await, vec![6, 7, 8]);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn query_longer_than_max_gram_is_exact() {
        let sender = make_sender();
        add_docs(&sender, NICKNAMES).await;

        // Four characters against a max_gram of three takes the intersect-and-verify path.
        assert_eq!(search(&sender, "将军来了").await, vec![3]);
    }

    #[rstest]
    #[case::cjk("将军来 军来了", "将军来了")]
    #[case::latin("abcd cde", "abcde")]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn all_grams_present_but_not_contiguous_is_not_a_match(
        #[case] decoy: &str,
        #[case] query: &str,
    ) {
        let sender = make_sender();
        // The decoy contains every 3-gram of the query, but not the query itself.
        add_docs(&sender, &[(1, decoy), (2, query)]).await;

        assert_eq!(search(&sender, query).await, vec![2]);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn case_insensitive_index_matches_latin_regardless_of_case() {
        let sender = make_sender_with_options(options(1, 3, false));
        add_docs(&sender, USERNAMES).await;

        // "gn" is the reverse, not a match.
        assert_eq!(search(&sender, "ng").await, vec![1, 2]);
        assert_eq!(search(&sender, "NG").await, vec![1, 2]);
        assert_eq!(search(&sender, "Ng").await, vec![1, 2]);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn case_sensitive_index_matches_exact_case_only() {
        let sender = make_sender();
        add_docs(&sender, USERNAMES).await;

        // Only "kingNG" contains a lowercase "ng" (inside "king").
        assert_eq!(search(&sender, "ng").await, vec![2]);
        assert_eq!(search(&sender, "NG").await, vec![1, 2]);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn digit_queries_match_by_containment() {
        let sender = make_sender();
        add_docs(&sender, USERNAMES).await;

        assert_eq!(search(&sender, "925555").await, vec![4, 5]);
        assert_eq!(search(&sender, "925").await, vec![4, 5, 6]);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn search_respects_limit() {
        let sender = make_sender();
        add_docs(&sender, NICKNAMES).await;

        assert_eq!(search_page(&sender, "宫", 2, 0).await.len(), 2);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn offset_pages_are_disjoint_and_cover_all_matches() {
        let sender = make_sender();
        add_docs(&sender, NICKNAMES).await;

        let mut seen = Vec::new();
        for offset in 0..3 {
            let page = search_page(&sender, "宫", 1, offset).await;
            assert_eq!(page.len(), 1, "page at offset {offset}");
            seen.extend(page);
        }
        seen.sort_unstable();
        assert_eq!(seen, vec![6, 7, 8]);
        assert!(search_page(&sender, "宫", 1, 3).await.is_empty());
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn walk_continues_past_false_positives_until_limit() {
        let sender = make_sender();
        // Many decoys with all the grams of the query, then the one true match.
        let decoys: Vec<(u64, String)> = (1..=20).map(|id| (id, format!("abcd{id}cde"))).collect();
        for (id, decoy) in &decoys {
            add_doc(&sender, *id, decoy).await;
        }
        add_doc(&sender, 21, "xxabcdexx").await;

        assert_eq!(search_page(&sender, "abcde", 1, 0).await, vec![21]);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn empty_query_is_a_query_error() {
        let sender = make_sender();
        add_docs(&sender, NICKNAMES).await;

        let err = sender
            .search(
                make_index_key(),
                "".into(),
                MatchKind::Contains,
                limit(10),
                0,
                SearchWindow::default(),
                true,
            )
            .await
            .expect_err("an empty query cannot be answered");

        assert!(err.downcast_ref::<QueryError>().is_some(), "got: {err}");
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn query_shorter_than_min_gram_is_a_query_error() {
        let sender = make_sender_with_options(options(2, 3, true));
        add_docs(&sender, NICKNAMES).await;

        let err = sender
            .search(
                make_index_key(),
                "宫".into(),
                MatchKind::Contains,
                limit(10),
                0,
                SearchWindow::default(),
                true,
            )
            .await
            .expect_err("a one-character query cannot be answered by a min_gram=2 index");

        assert!(err.downcast_ref::<QueryError>().is_some(), "got: {err}");
        // A query at min_gram is still served.
        assert_eq!(search(&sender, "南宫").await, vec![6, 7]);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn remove_then_search_excludes_removed() {
        let sender = make_sender();
        add_docs(&sender, NICKNAMES).await;

        rm_doc(&sender, 6).await;

        assert_eq!(search(&sender, "宫").await, vec![7, 8]);
        assert_eq!(sender.count(make_index_key()).await.unwrap(), 7);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn add_document_increments_count() {
        let sender = make_sender();
        add_docs(&sender, USERNAMES).await;

        assert_eq!(sender.count(make_index_key()).await.unwrap(), 6);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn stats_reflect_doc_count_and_segments() {
        let sender = make_sender();
        add_docs(&sender, USERNAMES).await;

        let stats = sender.stats(make_index_key()).await.unwrap();

        assert_eq!(stats.tantivy.num_docs, 6);
        assert!(stats.tantivy.segment_count > 0);
        assert!(stats.tantivy.size_bytes > 0);
        assert_eq!(stats.segments.len(), stats.tantivy.segment_count);
        assert_eq!(stats.segments.iter().map(|s| s.docs).sum::<u32>(), 6);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn stats_for_unknown_index_returns_default() {
        let sender = make_sender();

        let stats = sender.stats(make_index_key()).await.unwrap();

        assert_eq!(stats, Default::default());
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn search_before_any_document_returns_nothing() {
        let sender = make_sender();

        assert!(search(&sender, "宫").await.is_empty());
    }

    #[test]
    fn grams_of_length_keeps_only_that_length_and_dedupes() {
        assert_eq!(
            grams_of_length("将军来了", 3),
            BTreeSet::from(["将军来".to_string(), "军来了".to_string()])
        );
        assert_eq!(
            grams_of_length("aaaa", 3),
            BTreeSet::from(["aaa".to_string()])
        );
    }

    #[test]
    fn grams_of_length_matches_the_ngram_tokenizer() {
        let text = "a b 将军";
        let mut analyzer = TextAnalyzer::builder(NgramTokenizer::new(2, 2, false).unwrap()).build();
        let mut stream = analyzer.token_stream(text);
        let mut tokens = BTreeSet::new();
        while stream.advance() {
            tokens.insert(stream.token().text.clone());
        }
        assert_eq!(grams_of_length(text, 2), tokens);
        // Whitespace is an ordinary character: `LIKE '%a b%'` must match "a b".
        assert!(tokens.contains("a "));
        assert!(tokens.contains(" b"));
    }

    #[test]
    fn normalize_lowercases_only_when_case_insensitive() {
        assert_eq!(normalize("NGgamer", CaseSensitive::from(true)), "NGgamer");
        assert_eq!(normalize("NGgamer", CaseSensitive::from(false)), "nggamer");
        assert_eq!(normalize("将军", CaseSensitive::from(false)), "将军");
    }

    /// An index with a sort column returns the highest sort keys first. The unordered helpers sort
    /// their results, so only `search_ordered` can see this.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn an_ordered_index_returns_the_highest_sort_keys_first() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        let (ids, _) = search_ordered(&sender, "将军", 10, SearchWindow::default()).await;
        let mut descending = ids.clone();
        descending.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(ids, descending, "not ordered by the sort column");
        assert_eq!(ids, vec![3, 2, 1]);
    }

    /// `Stats` prices the searches served so far and describes the segments the pruning sees: an
    /// ordered index reports each segment's sort bounds, and the walk's tally grows per search.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn stats_count_the_walk_and_describe_the_segments() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        let before = sender.stats(make_index_key()).await.unwrap();
        assert_eq!(before.walk.searches, 0);
        assert_eq!(before.segments.len(), before.tantivy.segment_count);
        let docs: u32 = before.segments.iter().map(|s| s.docs).sum();
        assert_eq!(u64::from(docs), before.tantivy.num_docs);
        for segment in &before.segments {
            let (min, max) = (segment.sort_min.unwrap(), segment.sort_max.unwrap());
            assert!(min <= max, "segment bounds inverted: {segment:?}");
        }

        search_ordered(&sender, "将军", 2, SearchWindow::default()).await;

        let after = sender.stats(make_index_key()).await.unwrap();
        let walk = after.walk;
        assert_eq!(walk.searches, 1);
        assert!(walk.segments_opened >= 1 && walk.segments_opened <= walk.segments_considered);
        // Two of the three matches fill the page; the third enters only if its segment was not
        // pruned by then, which depends on how the documents were merged.
        assert!(walk.postings_scanned >= 2, "{walk:?}");
        assert!((2..=3).contains(&walk.heap_entrants), "{walk:?}");
        // Only the page is read from the store: two rows, not every entrant.
        assert_eq!(walk.store_reads, 2, "{walk:?}");
    }

    /// With the primary id kept as a column, an ordered page is resolved without the store:
    /// same rows, same order, zero store reads for a keyword the grams answer exactly.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn a_fast_primary_id_resolves_the_page_without_the_store() {
        let sender = make_sender_with_options(IndexOptionsSubstring {
            primary_id_fast: PrimaryIdFast::from(true),
            ..ordered_options()
        });
        add_docs(&sender, NICKNAMES).await;

        let (ids, cursor) = search_ordered(&sender, "将军", 2, SearchWindow::default()).await;
        assert_eq!(ids, vec![3, 2]);
        // The cursor's sort key carries the sign bias, not the value; its primary key is the
        // last row's.
        let cursor = cursor.expect("a full page leaves a cursor");
        assert_eq!(cursor.sort_key, 2 ^ (1 << 63));
        assert_eq!(
            cursor.primary_key,
            Some(PrimaryKey::from(vec![CqlValue::BigInt(2)]))
        );

        let walk = sender.stats(make_index_key()).await.unwrap().walk;
        assert_eq!(walk.searches, 1);
        assert_eq!(walk.store_reads, 0, "{walk:?}");
        assert!(walk.page_resolve_nanos <= walk.walk_nanos);
    }

    fn id(n: u8) -> SegmentId {
        SegmentId::from_uuid_string(&format!("00000000-0000-0000-0000-0000000000{n:02}")).unwrap()
    }

    /// Neighbours in sort order merge while under the cap; a full segment stands alone and ends
    /// the run; a lone segment is left as it is.
    #[test]
    fn merge_runs_follow_sort_order_under_the_cap() {
        // (docs, low, high, id), deliberately out of order: the tail (7, 8, 9) all reach the top.
        let segments = vec![
            (300, 0, 100, id(1)),   // full: stands alone
            (120, 100, 200, id(2)), // with 3 -> 220 <= 250
            (100, 200, 300, id(3)),
            (200, 300, 400, id(4)), // 220 + 200 > 250: new run; with 5 -> 250
            (50, 400, 500, id(5)),
            (300, 500, 600, id(6)), // full again
            (10, 990, 1000, id(7)), // the tail: three small overlapping segments
            (20, 985, 1000, id(8)),
            (5, 995, 1000, id(9)),
        ];
        // A narrow limit of 1600 lets a run span 400: room for every run here.
        let runs = merge_runs(segments, 250, 1600);
        assert_eq!(
            runs,
            vec![
                vec![id(2), id(3)],
                vec![id(4), id(5)],
                vec![id(8), id(7), id(9)],
            ]
        );
        assert!(merge_runs(vec![(10, 0, 1, id(1))], 250, 1600).is_empty());
        assert!(merge_runs(vec![], 250, 1600).is_empty());
    }

    /// A sliver never joins the narrow run it sorts into; slivers merge among themselves, and
    /// a narrow run stops where its span would pass a quarter of the limit.
    #[test]
    fn merge_runs_keep_spread_segments_out_of_narrow_runs() {
        // A limit of 400: a narrow run spans at most 100.
        let segments = vec![
            (50, 0, 49, id(1)), // narrow, with 2 -> 100 rows over 0..99
            (50, 50, 99, id(2)),
            (30, 5, 990, id(3)),   // a sliver sorting between 1 and 2
            (50, 100, 149, id(4)), // 1 + 2 + 4 would be 150 rows under the cap, but span 0..149
            (50, 800, 849, id(5)), // narrow, nothing within a run's span of it
            (20, 300, 950, id(6)), // a sliver: merges with 3, whatever the two span together
        ];
        let runs = merge_runs(segments, 250, 400);
        assert_eq!(runs, vec![vec![id(1), id(2)], vec![id(3), id(6)]]);
    }

    /// With the cap set, an index built from many small commits ends with segments that hold
    /// contiguous slices of the sort range, none past the cap, and still answers correctly.
    #[rstest]
    #[timeout(Duration::from_secs(30))]
    #[tokio::test]
    async fn the_range_policy_keeps_segments_narrow_and_capped() {
        // A commit of TEST_COMMIT_THRESHOLD rows lands as one segment per writer thread; the cap
        // lets the pieces of a commit merge and stops a run growing past two commits' worth.
        let cap = 120u32;
        let sender = make_sender_with_options(IndexOptionsSubstring {
            segment_max_docs: cap.to_string().parse().unwrap(),
            ..ordered_options()
        });
        // 400 rows in sort order, sort key = primary id, sent as one batch: `add_doc` waits for
        // the commit that carries its row, and one wait per row would be 400 commit ticks.
        let mut acks = Vec::with_capacity(400);
        for primary in 1..=400u64 {
            let (tx, rx) = mpsc::channel(1);
            sender
                .add_document(
                    test_partition_id(),
                    primary.into(),
                    format!("user{primary}将军"),
                    AsyncInProgress::Fullscan(tx),
                )
                .await
                .unwrap();
            acks.push(rx);
        }
        for mut ack in acks {
            ack.recv().await;
        }
        // Merges finish on their own thread and become visible on the actor's idle reloads.
        let deadline = Instant::now() + Duration::from_secs(20);
        let stats = loop {
            let stats = sender.stats(make_index_key()).await.unwrap();
            if stats.tantivy.segment_count <= 6 || Instant::now() > deadline {
                break stats;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        assert!(
            stats.segments.iter().all(|s| s.docs <= cap),
            "a segment grew past the cap: {:?}",
            stats.segments
        );
        let mut spans: Vec<(u64, u64)> = stats
            .segments
            .iter()
            .map(|s| (s.sort_min.unwrap(), s.sort_max.unwrap()))
            .collect();
        spans.sort_unstable();
        assert!(
            stats.tantivy.segment_count <= 6,
            "{} segments: {spans:?}",
            stats.tantivy.segment_count
        );
        // Sorted by low bound, each segment ends before the next one begins: contiguous slices.
        for pair in spans.windows(2) {
            assert!(pair[0].1 < pair[1].0, "segments overlap: {spans:?}");
        }

        let (ids, _) = search_ordered(&sender, "将军", 5, SearchWindow::default()).await;
        assert_eq!(ids, vec![400, 399, 398, 397, 396]);
    }

    /// A keyword past max_gram reads the store only for the page and the false positives above
    /// it, not for every candidate: 30 rows all matching, a page of 5, and 5 verification reads
    /// plus the 5 that resolve the page.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn the_range_policy_folds_the_tail_a_stream_of_updates_leaves() {
        // What the docker smoke saw (2026-09-26): 3000 shuffled rows in one build, then every row
        // delivered again as an update in small commits, as the CDC reader replays the load. The
        // updates land as small segments spanning the whole range; once the writes stop, the
        // idle reloads must fold them together under the cap rather than leave two dozen slivers.
        let cap = 500u32;
        let sender = make_sender_with_threshold(
            IndexOptionsSubstring {
                segment_max_docs: cap.to_string().parse().unwrap(),
                ..ordered_options()
            },
            3000,
        );
        let ids: Vec<u64> = (0..3000u64).map(|i| (i * 1103 + 977) % 3000 + 1).collect();
        let send = |primary: u64| {
            let sender = sender.clone();
            async move {
                let (tx, rx) = mpsc::channel(1);
                sender
                    .add_document(
                        test_partition_id(),
                        primary.into(),
                        format!("user{primary}将军"),
                        AsyncInProgress::Fullscan(tx),
                    )
                    .await
                    .unwrap();
                rx
            }
        };
        let mut acks = Vec::with_capacity(3000);
        for &primary in &ids {
            acks.push(send(primary).await);
        }
        for mut ack in acks {
            ack.recv().await;
        }
        // The replay: rows come back in the same order, a commit interval apart per hundred.
        for chunk in ids.chunks(100) {
            let mut acks = Vec::with_capacity(chunk.len());
            for &primary in chunk {
                rm_doc_no_wait(&sender, primary).await;
                acks.push(send(primary).await);
            }
            for mut ack in acks {
                ack.recv().await;
            }
        }
        let deadline = Instant::now() + Duration::from_secs(20);
        let stats = loop {
            let stats = sender.stats(make_index_key()).await.unwrap();
            if stats.tantivy.segment_count <= 8 || Instant::now() > deadline {
                break stats;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        assert_eq!(stats.tantivy.num_docs, 3000);
        assert!(
            stats.tantivy.segment_count <= 8,
            "{} segments: {:?}",
            stats.tantivy.segment_count,
            stats.segments
        );
        assert!(stats.segments.iter().all(|s| s.docs <= cap));
    }

    /// The docker smoke's case: thirty caps' worth of shuffled rows, built in one commit or in
    /// commits of a few caps as a full scan would, must end narrow and capped after the rewrite,
    /// with every row exactly once.
    #[rstest]
    #[case::one_commit(3000)]
    #[case::scan_commits(500)]
    #[timeout(Duration::from_secs(90))]
    #[tokio::test]
    async fn many_caps_of_shuffled_rows_end_narrow_and_capped(#[case] threshold: usize) {
        let rows = 3000u64;
        let cap = 100u32;
        let sender = make_sender_with_threshold(
            IndexOptionsSubstring {
                segment_max_docs: cap.to_string().parse().unwrap(),
                rewrite_wide_segments: RewriteWideSegments::from(true),
                ..ordered_options()
            },
            threshold,
        );
        let mut acks = Vec::new();
        for i in 0..rows {
            let primary = (i * 1103 + 977) % rows + 1;
            let (tx, rx) = mpsc::channel(1);
            sender
                .add_document(
                    test_partition_id(),
                    primary.into(),
                    format!("user{primary}将军"),
                    AsyncInProgress::Fullscan(tx),
                )
                .await
                .unwrap();
            acks.push(rx);
        }
        for mut ack in acks {
            ack.recv().await;
        }
        let deadline = Instant::now() + Duration::from_secs(60);
        let stats = loop {
            let stats = sender.stats(make_index_key()).await.unwrap();
            let idle = stats.rewrite.ranges_done == stats.rewrite.ranges_total;
            let capped = stats.segments.iter().all(|s| s.docs <= cap);
            let narrow = stats
                .segments
                .iter()
                .all(|s| s.sort_max.unwrap() - s.sort_min.unwrap() <= 2 * u64::from(cap));
            if (idle && capped && narrow && stats.rewrite.docs_rewritten > 0)
                || Instant::now() > deadline
            {
                break stats;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        let layout: Vec<(u32, u64, u64)> = stats
            .segments
            .iter()
            .map(|s| (s.docs, s.sort_min.unwrap(), s.sort_max.unwrap()))
            .collect();
        assert_eq!(stats.tantivy.num_docs, rows, "{layout:?}");
        assert_eq!(
            stats.rewrite.ranges_done, stats.rewrite.ranges_total,
            "{:?}",
            stats.rewrite
        );
        assert!(
            layout
                .iter()
                .all(|&(docs, low, high)| docs <= cap && high - low <= 2 * u64::from(cap)),
            "{} segments, rewrite {:?}: {layout:?}",
            layout.len(),
            stats.rewrite
        );
        let (ids, _) = search_ordered(&sender, "将军", 3, SearchWindow::default()).await;
        assert_eq!(ids, vec![3000, 2999, 2998]);
    }

    #[tokio::test]
    async fn a_verified_search_reads_the_store_for_the_page_only() {
        let sender = make_sender_with_options(ordered_options());
        // Sort key = primary id; "abcd" is past max_gram (3) and every row contains it.
        // Sent as one batch so they share a commit and land in a few segments together; one
        // segment per row would make every segment a single candidate and prove nothing.
        let mut acks = Vec::new();
        for n in 1..=30u64 {
            let (tx, rx) = mpsc::channel(1);
            sender
                .add_document(
                    test_partition_id(),
                    n.into(),
                    format!("x{n}abcdx"),
                    AsyncInProgress::Fullscan(tx),
                )
                .await
                .unwrap();
            acks.push(rx);
        }
        for mut ack in acks {
            ack.recv().await;
        }
        let segments = sender
            .stats(make_index_key())
            .await
            .unwrap()
            .tantivy
            .segment_count;
        assert!(
            segments < 30,
            "expected the rows to share segments, got {segments}"
        );

        let (ids, _) = search_ordered(&sender, "abcd", 5, SearchWindow::default()).await;
        assert_eq!(ids, vec![30, 29, 28, 27, 26]);

        // The candidates the walk scanned (the top segment's, at least; a lower segment may be
        // pruned by its bounds) versus the 5 it read to verify and the 5 it read for the page.
        let walk = sender.stats(make_index_key()).await.unwrap().walk;
        assert!(walk.postings_scanned >= 5, "{walk:?}");
        assert_eq!(walk.heap_entrants, 5, "{walk:?}");
        assert_eq!(walk.store_reads, 10, "{walk:?}");
    }

    /// A segment is wide when it covers far more of the range than its share of the rows; small
    /// fresh segments are ignored whatever they span.
    #[test]
    fn wide_segments_are_those_spanning_more_than_eight_caps_worth() {
        // 10,000 rows at a cap of 100: a narrow segment spans at most 8% of the range.
        let total = 10_000;
        let segments = vec![
            (100, 0, 99, id(1)),      // one cap's worth, one cap's width: aligned
            (100, 100, 899, id(2)),   // the same rows over 8% of the range: still narrow
            (100, 0, 9999, id(3)),    // over all of the range: wide
            (4, 0, 9999, id(4)),      // wide but too small to bother with, unless min_docs is 1
            (120, 9000, 9119, id(5)), // narrow but over the cap: a rewrite target all the same
            (5, 9800, 9804, id(6)),   // a piece a commit's writer thread left: narrow
        ];
        assert_eq!(wide_segments(&segments, total, 50, 100), vec![id(3), id(5)]);
        assert_eq!(
            wide_segments(&segments, total, 1, 100),
            vec![id(3), id(4), id(5)]
        );
        assert_eq!(narrow_span(&segments, total, 100), 799);
        // Fewer rows than eight caps: nothing spans more than eight caps' worth, and only the
        // segment over the cap is left to take apart.
        assert_eq!(wide_segments(&segments, 700, 1, 100), vec![id(5)]);
        assert!(wide_segments(&[], 0, 50, 100).is_empty());
        assert!(wide_segments(&[(10, 7, 7, id(1))], 10, 1, 100).is_empty());
        assert_eq!(wide_segments(&[(10, 7, 7, id(1))], 10, 1, 5), vec![id(1)]);
    }

    /// Ranges follow the sample's quantiles, cover the bounds given, collapse on ties, and a
    /// range wider than the limit is cut into equal parts.
    #[test]
    fn quantile_ranges_cover_the_key_space_in_order() {
        let sample: Vec<u64> = (0..100).map(|n| n * 10).collect();
        let ranges = quantile_ranges(sample, 4, 0, 990, u64::MAX);
        assert_eq!(ranges, vec![(0, 250), (250, 500), (500, 750), (750, 991)]);
        assert_eq!(
            quantile_ranges(vec![5, 5, 5, 5], 3, 0, 5, u64::MAX),
            vec![(0, 5), (5, 6)]
        );
        assert_eq!(quantile_ranges(vec![1, 2], 1, 0, 2, u64::MAX), vec![(0, 3)]);
        // Sparse rows: one range over 0..1000 would land as wide as it is; cut to 300 apiece.
        assert_eq!(
            quantile_ranges(vec![0, 1000], 1, 0, 1000, 300),
            vec![(0, 251), (251, 502), (502, 753), (753, 1001)]
        );
        // The cut applies per quantile range: 10..500 and 500..901 are both over 400.
        assert_eq!(
            quantile_ranges(vec![10, 20, 500, 900], 2, 10, 900, 400),
            vec![(10, 255), (255, 500), (500, 701), (701, 901)]
        );
    }

    /// The whole of P3b: rows arriving in shuffled sort order make wide segments; with the
    /// rewrite on they end up in narrow, contiguous, capped segments, and the search still
    /// answers correctly throughout.
    #[rstest]
    #[timeout(Duration::from_secs(60))]
    #[tokio::test]
    async fn wide_segments_are_rewritten_into_narrow_ones() {
        let cap = 200u32;
        // One commit for the whole batch, so the segments it produces are big and span nearly
        // the whole range, like a full-scan build's. (Committed in small chunks instead, the
        // pieces are too small to count as wide until the policy has merged them into capped
        // segments that are, and the rewrite then proceeds in several smaller plans; the end
        // state is the same.)
        let sender = make_sender_with_threshold(
            IndexOptionsSubstring {
                segment_max_docs: cap.to_string().parse().unwrap(),
                rewrite_wide_segments: RewriteWideSegments::from(true),
                ..ordered_options()
            },
            2000,
        );
        // 2000 rows in a fixed shuffled order: i -> i * 1103 + 977 (mod 2000) is a permutation,
        // since 1103 and 2000 are coprime.
        let mut acks = Vec::new();
        for i in 0..2000u64 {
            let primary = (i * 1103 + 977) % 2000 + 1;
            let (tx, rx) = mpsc::channel(1);
            sender
                .add_document(
                    test_partition_id(),
                    primary.into(),
                    format!("user{primary}将军"),
                    AsyncInProgress::Fullscan(tx),
                )
                .await
                .unwrap();
            acks.push(rx);
        }
        for mut ack in acks {
            ack.recv().await;
        }

        let widest_of = |stats: &SubstringStats| {
            stats
                .segments
                .iter()
                .map(|s| s.sort_max.unwrap() - s.sort_min.unwrap())
                .max()
                .unwrap_or(0)
        };
        // Until every segment is capped and no wider than two ranges, with no rewrite pending.
        let deadline = Instant::now() + Duration::from_secs(45);
        let stats = loop {
            let stats = sender.stats(make_index_key()).await.unwrap();
            let idle = stats.rewrite.ranges_done == stats.rewrite.ranges_total;
            let capped = stats.segments.iter().all(|s| s.docs <= cap);
            let narrow = widest_of(&stats) <= 2 * u64::from(cap);
            if (idle && capped && narrow && stats.rewrite.docs_rewritten > 0)
                || Instant::now() > deadline
            {
                break stats;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        assert_eq!(
            stats.rewrite.ranges_done, stats.rewrite.ranges_total,
            "{:?}",
            stats.rewrite
        );
        assert!(stats.rewrite.docs_rewritten >= 1000, "{:?}", stats.rewrite);
        assert_eq!(stats.tantivy.num_docs, 2000);
        assert!(
            stats.segments.iter().all(|s| s.docs <= cap),
            "a segment is past the cap: {:?}",
            stats.segments
        );
        let mut spans: Vec<(u64, u64, u32)> = stats
            .segments
            .iter()
            .map(|s| (s.sort_min.unwrap(), s.sort_max.unwrap(), s.docs))
            .collect();
        spans.sort_unstable();
        assert!(
            widest_of(&stats) <= 2 * u64::from(cap),
            "a segment is still wide: {spans:?}"
        );

        let (ids, _) = search_ordered(&sender, "将军", 3, SearchWindow::default()).await;
        assert_eq!(ids, vec![2000, 1999, 1998]);
    }

    /// With `verify_in_order` the old walk is back: candidates are read as the postings come, so
    /// a segment scanned in ascending sort order reads nearly all of them. Kept for measurement.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn verifying_in_order_reads_every_candidate_that_beats_the_page() {
        // The cap folds the pieces a commit's writer threads leave into one segment, which is
        // what the count below assumes: over several segments the walk prunes most of them.
        let sender = make_sender_with_options(IndexOptionsSubstring {
            verify_in_order: VerifyInOrder::from(true),
            segment_max_docs: "1000".parse().unwrap(),
            ..ordered_options()
        });
        let mut acks = Vec::new();
        for n in 1..=30u64 {
            let (tx, rx) = mpsc::channel(1);
            sender
                .add_document(
                    test_partition_id(),
                    n.into(),
                    format!("x{n}abcdx"),
                    AsyncInProgress::Fullscan(tx),
                )
                .await
                .unwrap();
            acks.push(rx);
        }
        for mut ack in acks {
            ack.recv().await;
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        while sender
            .stats(make_index_key())
            .await
            .unwrap()
            .tantivy
            .segment_count
            > 1
            && Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let (ids, _) = search_ordered(&sender, "abcd", 5, SearchWindow::default()).await;
        assert_eq!(ids, vec![30, 29, 28, 27, 26]);
        let walk = sender.stats(make_index_key()).await.unwrap().walk;
        // Every candidate that beat the page at its moment was read, plus the page itself; with
        // rows in ascending order that is well over the 5 + 5 the two-pass walk needs.
        assert!(walk.store_reads > 10, "{walk:?}");
    }

    /// The columns are opened once per segment, by the reload that follows a commit, so a search
    /// opens nothing -- and the time before the first posting is part of the walk time.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn columns_are_opened_once_per_segment() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        search_ordered(&sender, "将军", 10, SearchWindow::default()).await;
        let first = sender.stats(make_index_key()).await.unwrap();
        assert_eq!(first.walk.column_opens, 0, "{:?}", first.walk);
        assert!(first.walk.prepare_nanos <= first.walk.walk_nanos);

        search_ordered(&sender, "将军", 10, SearchWindow::default()).await;
        let second = sender.stats(make_index_key()).await.unwrap();
        assert_eq!(second.walk.column_opens, 0, "{:?}", second.walk);
        assert_eq!(second.walk.searches, 2);
    }

    /// An unordered index has no sort column and says so, rather than inventing bounds.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn an_unordered_index_reports_segments_without_bounds() {
        let sender = make_sender();
        add_docs(&sender, NICKNAMES).await;
        search_ordered(&sender, "将军", 10, SearchWindow::default()).await;

        let stats = sender.stats(make_index_key()).await.unwrap();
        assert!(!stats.segments.is_empty());
        assert!(
            stats
                .segments
                .iter()
                .all(|s| s.sort_min.is_none() && s.sort_max.is_none())
        );
        assert_eq!(stats.walk.searches, 1);
        // The unordered walk reads every match it keeps from the store.
        assert_eq!(stats.walk.store_reads, stats.walk.heap_entrants);
    }

    /// Without a sort column nothing changes: the index answers as it did before, and reports no
    /// cursor for the caller to page with.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn an_unordered_index_reports_no_cursor() {
        let sender = make_sender();
        add_docs(&sender, NICKNAMES).await;

        let (ids, cursor) = search_ordered(&sender, "将军", 10, SearchWindow::default()).await;
        assert_eq!(cursor, None);
        assert_eq!(ids.len(), 3);
    }

    /// Paging by cursor: the pages are disjoint, in order, and together cover every match.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn cursor_pages_are_disjoint_and_cover_every_match() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        let (first, cursor) = search_ordered(&sender, "将军", 2, SearchWindow::default()).await;
        assert_eq!(first, vec![3, 2]);
        let cursor = cursor.expect("a full page leaves a cursor");

        let (second, cursor) =
            search_ordered(&sender, "将军", 2, resume(SortOrder::Desc, cursor)).await;
        assert_eq!(second, vec![1]);
        assert_eq!(
            cursor, None,
            "a page the walk could not fill is the last one"
        );

        let seen: Vec<i64> = first.into_iter().chain(second).collect();
        assert_eq!(seen, vec![3, 2, 1]);
    }

    /// A cursor is an invitation to ask again, so it is only offered when the walk filled the page.
    /// Offering one after an exhausted walk costs the caller a round trip that returns nothing, and
    /// tells a client there are more pages when there are not.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn an_unfilled_page_offers_no_cursor() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        // Three matches, asked for ten.
        let (ids, cursor) = search_ordered(&sender, "将军", 10, SearchWindow::default()).await;
        assert_eq!(ids, vec![3, 2, 1]);
        assert_eq!(
            cursor, None,
            "the walk ran out, so there is nothing to resume from"
        );

        // Exactly as many as there are: the walk stopped because the page was full, not because it
        // ran out, so it cannot tell that the next page would be empty and says so with a cursor.
        let (ids, cursor) = search_ordered(&sender, "将军", 3, SearchWindow::default()).await;
        assert_eq!(ids, vec![3, 2, 1]);
        assert!(cursor.is_some(), "a filled page leaves a cursor");
    }

    /// No match at all is not a page to resume either.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn an_empty_page_offers_no_cursor() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        let (ids, cursor) = search_ordered(&sender, "没有人", 10, SearchWindow::default()).await;
        assert!(ids.is_empty(), "got {ids:?}");
        assert_eq!(cursor, None);
    }

    /// A page the caller asked not to verify: the walk hands back every candidate that beats the
    /// page, flagged as unverified, and reads the store only to resolve the page's ids.
    async fn search_unverified(
        sender: &mpsc::Sender<SubstringIndex>,
        query: &str,
        limit_n: usize,
        window: SearchWindow,
    ) -> SubstringPage {
        sender
            .search(
                make_index_key(),
                query.into(),
                MatchKind::Contains,
                limit(limit_n),
                0,
                window,
                false,
            )
            .await
            .unwrap()
    }

    fn ids_of(page: &SubstringPage) -> Vec<i64> {
        page.primary_keys
            .iter()
            .map(|key| match key.get(0).unwrap() {
                CqlValue::BigInt(id) => id,
                other => panic!("unexpected primary key {other:?}"),
            })
            .collect()
    }

    /// Declining verification on a keyword past `max_gram` returns the candidates: the rows
    /// holding every gram, decoys included, in sort order, and the page says it is unverified.
    /// The store is read for the page's ids only, not to check any text.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn an_unverified_page_returns_the_candidates_in_order_and_says_so() {
        let sender = make_sender_with_options(IndexOptionsSubstring {
            order_by: "sort_col".parse().unwrap(),
            ..options(1, 3, true)
        });
        // Decoys hold every gram of "abcde" without containing it; row 21 is the one match.
        for id in 1..=20u64 {
            add_doc(&sender, id, &format!("abcd{id}cde")).await;
        }
        add_doc(&sender, 21, "xxabcdexx").await;

        let page = search_unverified(&sender, "abcde", 5, SearchWindow::default()).await;
        assert!(!page.verified);
        assert_eq!(ids_of(&page), vec![21, 20, 19, 18, 17]);
        assert!(page.next_cursor.is_some(), "a full page leaves a cursor");
        let walk = sender.stats(make_index_key()).await.unwrap().walk;
        assert_eq!(walk.store_reads, 5, "only the page is resolved: {walk:?}");

        // The same request verified returns the one match, and says so.
        let (ids, _) = search_ordered(&sender, "abcde", 5, SearchWindow::default()).await;
        assert_eq!(ids, vec![21]);
    }

    /// A keyword within `max_gram` is exact whatever the caller asked, so the page is verified.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn an_exact_keyword_is_verified_even_when_not_asked_to() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        let page = search_unverified(&sender, "将军", 10, SearchWindow::default()).await;
        assert!(page.verified);
        assert_eq!(ids_of(&page), vec![3, 2, 1]);
    }

    /// The unordered walk answers the flag the same way: candidates, flagged.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn an_unordered_unverified_page_is_flagged_too() {
        let sender = make_sender();
        for id in 1..=3u64 {
            add_doc(&sender, id, &format!("abcd{id}cde")).await;
        }
        add_doc(&sender, 4, "xxabcdexx").await;

        let page = search_unverified(&sender, "abcde", 10, SearchWindow::default()).await;
        assert!(!page.verified);
        let mut ids = ids_of(&page);
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 2, 3, 4]);
        let verified = search_unverified(&sender, "abc", 10, SearchWindow::default()).await;
        assert!(verified.verified);
    }

    /// A long keyword that occurs in one name opens only the segment holding it: every other
    /// segment lacks one of its grams, which one term-dictionary lookup shows, and is skipped
    /// before any posting list is opened. The answer is what the full walk gave.
    #[rstest]
    #[timeout(Duration::from_secs(30))]
    #[tokio::test]
    async fn a_rare_long_keyword_opens_only_the_segments_that_hold_its_grams() {
        let cap = 120u32;
        let sender = make_sender_with_options(IndexOptionsSubstring {
            segment_max_docs: cap.to_string().parse().unwrap(),
            ..ordered_options()
        });
        let mut acks = Vec::with_capacity(400);
        for primary in 1..=400u64 {
            let name = if primary == 250 {
                "zqxwvkj将军".to_string()
            } else {
                format!("user{primary}将军")
            };
            let (tx, rx) = mpsc::channel(1);
            sender
                .add_document(
                    test_partition_id(),
                    primary.into(),
                    name,
                    AsyncInProgress::Fullscan(tx),
                )
                .await
                .unwrap();
            acks.push(rx);
        }
        for mut ack in acks {
            ack.recv().await;
        }
        let deadline = Instant::now() + Duration::from_secs(20);
        let segments = loop {
            let stats = sender.stats(make_index_key()).await.unwrap();
            if stats.tantivy.segment_count <= 6 || Instant::now() > deadline {
                break stats.tantivy.segment_count;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        assert!(
            segments >= 3,
            "the test needs several segments, got {segments}"
        );

        let before = sender.stats(make_index_key()).await.unwrap().walk;
        let (ids, _) = search_ordered(&sender, "qxwvkj将", 20, SearchWindow::default()).await;
        let after = sender.stats(make_index_key()).await.unwrap().walk;
        assert_eq!(ids, vec![250]);
        assert_eq!(
            after.segments_opened - before.segments_opened,
            1,
            "{segments} segments considered, only the one holding the name should open"
        );

        // A keyword every segment holds the grams of still opens them all and answers in full.
        let before = sender.stats(make_index_key()).await.unwrap().walk;
        let (ids, _) = search_ordered(&sender, "er1将军", 400, SearchWindow::default()).await;
        let after = sender.stats(make_index_key()).await.unwrap().walk;
        assert_eq!(ids, vec![1]);
        assert!(after.segments_opened - before.segments_opened >= 1);
    }

    /// The unordered walk skips the same way and answers the same.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn the_unordered_walk_skips_segments_without_the_grams_too() {
        let sender = make_sender();
        add_docs(&sender, NICKNAMES).await;
        add_doc(&sender, 9, "zqxwvkj").await;
        assert_eq!(search(&sender, "qxwvkj").await, vec![9]);
        assert_eq!(search(&sender, "将军来了").await, vec![3]);
        assert!(search(&sender, "qxwvkz").await.is_empty());
    }

    /// A keyword longer than `max_gram` takes the verification path, where the sort key is read
    /// before the document store rather than after. The answer must be the same.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn ordering_holds_for_a_keyword_past_max_gram() {
        let sender = make_sender_with_options(IndexOptionsSubstring {
            order_by: "sort_col".parse().unwrap(),
            ..options(1, 3, true)
        });
        add_docs(&sender, NICKNAMES).await;

        let (ids, _) = search_ordered(&sender, "将军来了", 10, SearchWindow::default()).await;
        assert_eq!(ids, vec![3]);
    }

    /// Reading past the end yields an empty page rather than an error.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn paging_past_the_last_match_is_empty() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        let (ids, _) =
            search_ordered(&sender, "将军", 10, window(Some(sort_key(1)), None, None)).await;
        assert!(ids.is_empty(), "got {ids:?}");
    }

    /// A range restriction narrows the answer without changing its order.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn a_range_restricts_which_rows_are_returned() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        // The mock table answers with the primary id as the sort key, so this is ids 2..=3.
        let window = window(None, Some(sort_key(2)), Some(sort_key(3)));
        let (ids, _) = search_ordered(&sender, "将军", 10, window).await;
        assert_eq!(ids, vec![3, 2]);
    }

    /// The bounds are inclusive, unlike the cursor, which excludes the key it names.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn range_bounds_are_inclusive_and_the_cursor_is_not() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        let (inclusive, _) = search_ordered(
            &sender,
            "将军",
            10,
            window(None, Some(sort_key(3)), Some(sort_key(3))),
        )
        .await;
        assert_eq!(inclusive, vec![3]);

        let (exclusive, _) = search_ordered(
            &sender,
            "将军",
            10,
            window(Some(sort_key(3)), Some(sort_key(3)), None),
        )
        .await;
        assert!(exclusive.is_empty(), "got {exclusive:?}");
    }

    /// A range and a cursor constrain the same value, so the tighter upper bound wins rather than
    /// one quietly overriding the other.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn a_cursor_and_a_range_both_apply() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        // The range allows 1..=3, the cursor excludes 3 and above: 2 and 1 remain.
        let window = window(Some(sort_key(3)), Some(sort_key(1)), Some(sort_key(3)));
        let (ids, _) = search_ordered(&sender, "将军", 10, window).await;
        assert_eq!(ids, vec![2, 1]);
    }

    /// An empty window is not an error, just an empty answer.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn a_window_matching_nothing_returns_nothing() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        let (ids, cursor) = search_ordered(
            &sender,
            "将军",
            10,
            window(None, Some(sort_key(900)), Some(sort_key(999))),
        )
        .await;
        assert!(ids.is_empty(), "got {ids:?}");
        assert_eq!(cursor, None);
    }

    /// A table whose sort column has ties: rows 1..=n share a sort key in pairs (1 and 2 sort
    /// as 1, 3 and 4 as 2, ...), so a page boundary can fall between two rows with one key.
    fn make_table_with_ties() -> Arc<RwLock<MockTableSearch>> {
        let index_id = IndexIdGenerator::new().next(true).unwrap();
        let partition_id = PartitionId::global(index_id);
        let mut mock = MockTableSearch::new();
        mock.expect_index_id()
            .returning(move |_index_key| Some(index_id));
        mock.expect_partition_id()
            .returning(move |_index_key, _restrictions| Some((partition_id, None)));
        mock.expect_primary_key()
            .returning(|_partition_id, primary_id| {
                Some(PrimaryKey::from(vec![CqlValue::BigInt(
                    u64::from(primary_id) as i64,
                )]))
            });
        mock.expect_primary_id()
            .returning(|primary_key| match primary_key.get(0) {
                Some(CqlValue::BigInt(id)) => Some(PrimaryId::from(id as u64)),
                _ => None,
            });
        mock.expect_column_value_for()
            .returning(|_partition_id, primary_id, _column| {
                Some(CqlValue::BigInt((u64::from(primary_id) as i64 + 1) / 2))
            });
        mock.expect_is_valid_for().returning(|_, _, _| true);
        mock.expect_target_column().returning(|_| None);
        Arc::new(RwLock::new(mock))
    }

    fn make_sender_with_ties(options: IndexOptionsSubstring) -> mpsc::Sender<SubstringIndex> {
        new(
            SubstringIndexConfiguration {
                key: make_index_key(),
                options,
            },
            make_table_with_ties(),
            worker::new(),
            make_memory_actor(),
            TEST_COMMIT_INTERVAL,
            TEST_COMMIT_THRESHOLD,
            None,
        )
    }

    /// Every page is followed from its cursor until none is offered; the pages concatenated.
    async fn all_pages(
        sender: &mpsc::Sender<SubstringIndex>,
        query: &str,
        page: usize,
        order: SortOrder,
    ) -> (Vec<i64>, usize) {
        let mut seen = Vec::new();
        let mut pages = 0;
        let mut window = SearchWindow {
            order,
            ..SearchWindow::default()
        };
        loop {
            let (ids, cursor) = search_ordered(sender, query, page, window).await;
            pages += 1;
            seen.extend(ids);
            match cursor {
                Some(cursor) => window = resume(order, cursor),
                None => return (seen, pages),
            }
        }
    }

    /// Rows sharing a sort key are neither skipped nor repeated across a page boundary, whatever
    /// the page size: the cursor names the last row, and the tie is broken on it.
    #[rstest]
    #[case::pages_split_every_tie(3)]
    #[case::pages_of_one(1)]
    #[case::pages_align_with_the_ties(2)]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn ties_are_paged_without_gaps_or_repeats(#[case] page: usize) {
        let sender = make_sender_with_ties(ordered_options());
        let docs: Vec<(u64, String)> = (1..=7).map(|i| (i, format!("user{i}将军"))).collect();
        let docs: Vec<(u64, &str)> = docs.iter().map(|(i, s)| (*i, s.as_str())).collect();
        add_docs(&sender, &docs).await;

        let (seen, _) = all_pages(&sender, "将军", page, SortOrder::Desc).await;
        // Sort keys 4,4,3,3,2,2,1 for ids 7,8.. -- highest key first, and within a key the
        // higher id first.
        assert_eq!(seen, vec![7, 6, 5, 4, 3, 2, 1]);

        let (seen, _) = all_pages(&sender, "将军", page, SortOrder::Asc).await;
        assert_eq!(seen, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    /// A cursor that names no row (or a row the table has forgotten) resumes at its sort key
    /// inclusive: the rows tied there come again, which a client can dedupe, rather than being
    /// lost, which it cannot repair.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn a_cursor_without_a_row_takes_the_tied_rows_again() {
        let sender = make_sender_with_ties(ordered_options());
        let docs: Vec<(u64, String)> = (1..=6).map(|i| (i, format!("user{i}将军"))).collect();
        let docs: Vec<(u64, &str)> = docs.iter().map(|(i, s)| (*i, s.as_str())).collect();
        add_docs(&sender, &docs).await;

        let (first, cursor) = search_ordered(&sender, "将军", 3, SearchWindow::default()).await;
        assert_eq!(first, vec![6, 5, 4]);
        let cursor = cursor.unwrap();
        assert_eq!(
            cursor.primary_key,
            Some(PrimaryKey::from(vec![CqlValue::BigInt(4)]))
        );

        // With the row: strictly after it.
        let (second, _) =
            search_ordered(&sender, "将军", 3, resume(SortOrder::Desc, cursor.clone())).await;
        assert_eq!(second, vec![3, 2, 1]);
        // Without it: the row tied with 4 (id 3 shares its sort key) comes again, nothing is lost.
        let unknown = Cursor {
            sort_key: cursor.sort_key,
            primary_key: None,
        };
        let (second, _) =
            search_ordered(&sender, "将军", 3, resume(SortOrder::Desc, unknown)).await;
        assert_eq!(second, vec![4, 3, 2]);
        let forgotten = Cursor {
            sort_key: cursor.sort_key,
            primary_key: Some(PrimaryKey::from(vec![CqlValue::Text("gone".into())])),
        };
        let (second, _) =
            search_ordered(&sender, "将军", 3, resume(SortOrder::Desc, forgotten)).await;
        assert_eq!(second, vec![4, 3, 2]);
    }

    /// Ascending is the same walk the other way: lowest sort key first, a range honoured, the
    /// cursor resuming above the last row, and no cursor once the walk ran out.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn ascending_order_walks_the_other_way() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        let (ids, cursor) = search_ordered(&sender, "将军", 2, ascending()).await;
        assert_eq!(ids, vec![1, 2]);
        let cursor = cursor.expect("a full page leaves a cursor");
        assert_eq!(cursor.sort_key, sort_key(2));

        let (ids, cursor) =
            search_ordered(&sender, "将军", 2, resume(SortOrder::Asc, cursor)).await;
        assert_eq!(ids, vec![3]);
        assert_eq!(cursor, None);

        // A range applies the same way round: keys 2..=3, lowest first.
        let (ids, _) = search_ordered(
            &sender,
            "将军",
            10,
            SearchWindow {
                order: SortOrder::Asc,
                cursor: None,
                min_sort_key: Some(sort_key(2)),
                max_sort_key: Some(sort_key(3)),
            },
        )
        .await;
        assert_eq!(ids, vec![2, 3]);

        // Past max_gram too: the verified path orders the same way.
        let (ids, _) = search_ordered(&sender, "将军来了", 10, ascending()).await;
        assert!(ids.windows(2).all(|w| w[0] < w[1]), "{ids:?}");
    }

    async fn search_kind(
        sender: &mpsc::Sender<SubstringIndex>,
        query: &str,
        kind: MatchKind,
    ) -> Vec<i64> {
        let mut ids: Vec<i64> = sender
            .search(
                make_index_key(),
                query.into(),
                kind,
                limit(100),
                0,
                SearchWindow::default(),
                true,
            )
            .await
            .unwrap()
            .primary_keys
            .into_iter()
            .map(|pk| match pk.get(0).unwrap() {
                CqlValue::BigInt(id) => id,
                other => panic!("unexpected primary key value {other:?}"),
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    /// A prefix or a suffix query is answered by the same index: the keyword anchored by the
    /// value's frame. Containment is unchanged, one-character anchors work, and a keyword that is
    /// the whole value matches on both sides.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn prefix_and_suffix_queries_anchor_the_keyword() {
        let sender = make_sender();
        add_docs(&sender, NICKNAMES).await;

        assert_eq!(
            search_kind(&sender, "将军", MatchKind::Contains).await,
            vec![1, 2, 3]
        );
        assert_eq!(
            search_kind(&sender, "将军", MatchKind::Prefix).await,
            vec![3]
        );
        assert_eq!(
            search_kind(&sender, "将军", MatchKind::Suffix).await,
            vec![1, 2]
        );
        // Single characters, which the index answers from a two-character gram with the mark.
        assert_eq!(search_kind(&sender, "南", MatchKind::Prefix).await, vec![6]);
        assert_eq!(search_kind(&sender, "南", MatchKind::Suffix).await, vec![8]);
        assert_eq!(
            search_kind(&sender, "军", MatchKind::Prefix).await,
            Vec::<i64>::new()
        );
        // The whole value is both its prefix and its suffix.
        assert_eq!(
            search_kind(&sender, "元帅", MatchKind::Prefix).await,
            vec![5]
        );
        assert_eq!(
            search_kind(&sender, "元帅", MatchKind::Suffix).await,
            vec![5]
        );
        // Past max_gram the verification anchors as well.
        assert_eq!(
            search_kind(&sender, "将军来了", MatchKind::Prefix).await,
            vec![3]
        );
        assert_eq!(
            search_kind(&sender, "军来了", MatchKind::Prefix).await,
            Vec::<i64>::new()
        );
        assert_eq!(
            search_kind(&sender, "南宫粉丝团", MatchKind::Suffix).await,
            vec![7]
        );
    }

    /// An ordered index anchors the same way, and pages through the anchored matches.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn anchored_queries_are_ordered_and_paged_too() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        let page = sender
            .search(
                make_index_key(),
                "将军".into(),
                MatchKind::Suffix,
                limit(1),
                0,
                SearchWindow::default(),
                true,
            )
            .await
            .unwrap();
        let ids: Vec<i64> = page
            .primary_keys
            .iter()
            .map(|pk| match pk.get(0).unwrap() {
                CqlValue::BigInt(id) => id,
                other => panic!("unexpected primary key value {other:?}"),
            })
            .collect();
        assert_eq!(ids, vec![2]);
        let cursor = page.next_cursor.expect("a full page leaves a cursor");
        let page = sender
            .search(
                make_index_key(),
                "将军".into(),
                MatchKind::Suffix,
                limit(1),
                0,
                resume(SortOrder::Desc, cursor),
                true,
            )
            .await
            .unwrap();
        let ids: Vec<i64> = page
            .primary_keys
            .iter()
            .map(|pk| match pk.get(0).unwrap() {
                CqlValue::BigInt(id) => id,
                other => panic!("unexpected primary key value {other:?}"),
            })
            .collect();
        assert_eq!(ids, vec![1]);
        // A filled page always leaves a cursor; the page after it is empty and leaves none.
        let cursor = page.next_cursor.expect("a full page leaves a cursor");
        let page = sender
            .search(
                make_index_key(),
                "将军".into(),
                MatchKind::Suffix,
                limit(1),
                0,
                resume(SortOrder::Desc, cursor),
                true,
            )
            .await
            .unwrap();
        assert!(page.primary_keys.is_empty());
        assert_eq!(page.next_cursor, None);
    }

    /// The frame is stripped before a row is re-indexed by the rewrite; otherwise every pass
    /// would add another pair of marks.
    #[test]
    fn a_stored_value_is_unframed_once() {
        assert_eq!(framed("abc"), "\u{2}abc\u{3}");
        assert_eq!(unframed("\u{2}abc\u{3}"), "abc");
        assert_eq!(unframed("abc"), "abc");
        assert_eq!(pattern_of(MatchKind::Prefix, "ab"), "\u{2}ab");
        assert_eq!(pattern_of(MatchKind::Suffix, "ab"), "ab\u{3}");
        assert_eq!(pattern_of(MatchKind::Contains, "ab"), "ab");
    }
}
