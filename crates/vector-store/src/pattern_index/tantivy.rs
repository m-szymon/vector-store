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
//!   column is narrow. Two options keep it so: `segment_rows` installs a
//!   merge policy that only joins neighbours in sort order under a cap, and
//!   `rewrite` moves the rows of segments that span far more than
//!   their share -- an index built by a full scan, a shuffled load -- into range-aligned ones,
//!   one range per idle tick. Without them, after an unordered backfill every segment spans
//!   everything and the search degrades to visiting every match: correct, but not fast. See
//!   `docs/dev/pattern/stage-2-ordering.md`.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::AcqRel;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::Ordering::Release;
use std::time::Duration;
use std::time::Instant;

use anyhow::anyhow;
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
use tantivy::postings::TermInfo;
use tantivy::query::ConstScorer;
use tantivy::query::EnableScoring;
use tantivy::query::Query;
use tantivy::query::Scorer;
use tantivy::query::TermQuery;
use tantivy::query::Weight;
use tantivy::query::intersect_scorers;
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
use crate::IndexOptionsPattern;
use crate::Limit;
use crate::PrimaryKey;
use crate::QueryError;
use crate::memory::Memory;
use crate::metrics::Metrics;
use crate::perf;
use crate::table::PartitionId;
use crate::table::PrimaryId;
use crate::table::Table;
use crate::table::TableSearch;
use crate::tantivy::COMMIT_INTERVAL;
use crate::tantivy::Index;
use crate::tantivy::Indexes;
use crate::tantivy::MAX_UNCOMMITTED_THRESHOLD;
use crate::worker::Worker;
use crate::worker::WorkerExt;

use super::actor::Cursor;
use super::actor::MatchKind;
use super::actor::PatternIndex;
use super::actor::PatternPage;
use super::actor::PatternSearchR;
use super::actor::SearchWindow;
use super::actor::SortOrder;
use super::factory::PatternIndexConfiguration;
use super::factory::PatternIndexFactory;
use super::order::KeyLayout;
use super::order::PREFIX_SLOTS;
use super::order::SortPrefix;

pub(crate) struct TantivyPatternIndexFactory {
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
    metrics: Arc<Metrics>,
}

impl TantivyPatternIndexFactory {
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

impl PatternIndexFactory for TantivyPatternIndexFactory {
    fn create_index(
        &self,
        index: PatternIndexConfiguration,
        table: Arc<RwLock<Table>>,
    ) -> mpsc::Sender<PatternIndex> {
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

/// The field the shared tantivy code deletes documents by; the pattern index also reads it back.
const PRIMARY_ID_FIELD: &str = "primary_id";
const TEXT_FIELD: &str = "text";
/// The fast field holding a row's token, the last slot of its sort prefix. The ordering columns'
/// slots come before it, as `sort_<i>`.
const TOKEN_FIELD: &str = "token";
const TOKENIZER_NAME: &str = "pattern_ngram";
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
struct PatternBackend {
    options: IndexOptionsPattern,
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
    /// The table's primary-key layout: tokens and table order.
    key_layout: KeyLayout,
    rewrite: RewriteCounters,
    /// Set while a tick's work on this index -- learning the layout, a step of the rewrite -- is
    /// running, so that the next tick does not queue more behind it. See `TickWork`.
    tick_running: AtomicBool,
}

/// Holds an index's `tick_running` for one tick's work and clears it however that work ends.
struct TickWork(Arc<PatternIndexState>);

impl TickWork {
    /// None while the previous tick's work on this index is still running.
    fn start(state: &Arc<PatternIndexState>) -> Option<Self> {
        (!state.backend().tick_running.swap(true, AcqRel)).then(|| Self(Arc::clone(state)))
    }
}

impl Drop for TickWork {
    fn drop(&mut self) {
        self.0.backend().tick_running.store(false, Release);
    }
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
    /// `[low, high)` in layout-key space, ascending, covering everything. Each is moved out of
    /// whichever segments are wide when its turn comes: the merge policy retires segment ids all
    /// the time, so a plan that named them would find them gone.
    ranges: Vec<(u128, u128)>,
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

/// The widest span, in layout-key units, a narrow segment may have: `WIDE_SEGMENT_FACTOR` caps'
/// worth of a range holding `total_docs` rows. Everything is narrow while the index holds fewer
/// rows than that many caps.
fn narrow_span(segments: &[(u32, u128, u128, SegmentId)], total_docs: u64, cap: u32) -> u128 {
    if total_docs == 0 {
        return u128::MAX;
    }
    let low = segments.iter().map(|s| s.1).min().unwrap_or(0);
    let high = segments.iter().map(|s| s.2).max().unwrap_or(0);
    let range = (high - low).max(1) as f64;
    let span = range * WIDE_SEGMENT_FACTOR * f64::from(cap) / total_docs as f64;
    if span >= u128::MAX as f64 {
        u128::MAX
    } else {
        span as u128
    }
}

/// The segments that call for a rewrite: those spanning more than `narrow_span`, and those over
/// the cap, which no merge can ever bring under it. Input tuples are
/// `(docs, sort_min, sort_max, id)`; `min_docs` says how big a wide segment has to be to count.
fn wide_segments(
    segments: &[(u32, u128, u128, SegmentId)],
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
    segments: &[(u32, u128, u128, SegmentId)],
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
    mut sample: Vec<u128>,
    count: usize,
    low: u128,
    high: u128,
    max_width: u128,
) -> Vec<(u128, u128)> {
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

type SharedBounds = Arc<RwLock<HashMap<SegmentId, (u128, u128)>>>;

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
const NARROW_RUN_SPAN_DIVISOR: u128 = 4;

/// The runs of sort-order neighbours worth merging: each at least two segments, each under
/// `max_docs` rows in total. A narrow run -- segments spanning no more than a quarter of
/// `narrow_span` each -- stops growing where its combined span would pass that quarter. Anything
/// spanning more than that never joins a narrow run, which would only widen it; those segments
/// merge among themselves with no limit on the span, so slivers grow into a segment the rewrite
/// will find wide. Input tuples are `(docs, sort_min, sort_max, id)`. Pure, so it can be tested
/// without a writer.
fn merge_runs(
    segments: Vec<(u32, u128, u128, SegmentId)>,
    max_docs: u32,
    narrow_span: u128,
) -> Vec<Vec<SegmentId>> {
    let run_span = narrow_span / NARROW_RUN_SPAN_DIVISOR;
    let (spread, narrow): (Vec<_>, Vec<_>) = segments
        .into_iter()
        .partition(|&(_, low, high, _)| high - low > run_span);
    let mut runs = runs_under(narrow, max_docs, run_span);
    // Spread segments each cover most of the range, so being neighbours by lowest key means
    // nothing: a full one between two small ones must not keep them apart, as it does in a
    // narrow run, where it stands for the rows between them.
    let spread = spread
        .into_iter()
        .filter(|&(docs, _, _, _)| docs < max_docs)
        .collect();
    runs.extend(runs_under(spread, max_docs, u128::MAX));
    runs
}

fn runs_under(
    mut segments: Vec<(u32, u128, u128, SegmentId)>,
    max_docs: u32,
    max_span: u128,
) -> Vec<Vec<SegmentId>> {
    segments.sort_by_key(|&(_, low, high, _)| (low, high));
    let mut runs = Vec::new();
    let mut run: Vec<SegmentId> = Vec::new();
    let mut run_docs = 0u32;
    let mut run_span = (u128::MAX, 0u128);
    let mut flush = |run: &mut Vec<SegmentId>, run_docs: &mut u32, run_span: &mut (u128, u128)| {
        if run.len() >= 2 {
            runs.push(std::mem::take(run));
        } else {
            run.clear();
        }
        *run_docs = 0;
        *run_span = (u128::MAX, 0);
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

/// One segment's columnar handles and the layout bounds the pruning reads from them.
#[derive(Clone)]
struct SegmentColumns {
    /// One column per slot of the sort prefix: the ordering columns, then the token.
    slots: Vec<Column<u64>>,
    /// The lowest and highest layout key of the segment's rows, exactly: the second slot at either
    /// end is its rows', not the whole of it, so that a segment holding the last rows of one value
    /// and the first of the next is as narrow as they are.
    layout_min: u128,
    layout_max: u128,
    /// The walk reads it only to tell apart rows whose whole prefix is equal.
    primary_id: Column<u64>,
}

impl SegmentColumns {
    /// The row's sort prefix, in the index's own space.
    fn prefix(&self, doc_id: u32) -> SortPrefix {
        let mut prefix = SortPrefix::default();
        for (slot, column) in self.slots.iter().enumerate() {
            prefix.0[slot] = column.first(doc_id).unwrap_or(0);
        }
        prefix
    }

    fn layout_key(&self, doc_id: u32) -> u128 {
        let first = u128::from(self.slots[0].first(doc_id).unwrap_or(0));
        let second = self
            .slots
            .get(1)
            .map_or(0, |column| u128::from(column.first(doc_id).unwrap_or(0)));
        (first << 64) | second
    }
}

/// The lowest second slot among the rows whose first slot is `first_min`, and the highest among
/// those whose first slot is `first_max`, over every row's `(first, second)`: a segment's exact
/// layout bounds when its rows hold more than one first slot. Bounds taking in the whole second
/// slot at either end, as before, made a segment holding the end of one value and the start of
/// the next span both values. With a column of few values the rewrite makes such a piece at every
/// value it crosses; each read as spread, the spread ones merged among themselves into segments
/// spanning several values, which the rewrite cut up into such pieces again, without end (a local
/// run with five values, 2026-10-30). One pass over the two columns, once per segment, as the
/// columns are cached. Deleted rows count too, as they do in the columns' own bounds.
fn second_slot_at_the_ends(
    rows: impl Iterator<Item = (u64, u64)>,
    first_min: u64,
    first_max: u64,
) -> (u64, u64) {
    let (mut low, mut high) = (None::<u64>, None::<u64>);
    for (first, second) in rows {
        if first == first_min {
            low = Some(low.map_or(second, |low| low.min(second)));
        }
        if first == first_max {
            high = Some(high.map_or(second, |high| high.max(second)));
        }
    }
    (low.unwrap_or(u64::MIN), high.unwrap_or(u64::MAX))
}

impl PatternBackend {
    /// The fast fields of the sort prefix: `sort_<i>` per ordering column, then the token.
    fn slot_fields(&self) -> Vec<String> {
        (0..self.options.order_by.columns().len())
            .map(|slot| format!("sort_{slot}"))
            .chain(std::iter::once(TOKEN_FIELD.to_string()))
            .collect()
    }

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
        let slots =
            self.slot_fields()
                .iter()
                .map(|field| {
                    segment.fast_fields().u64(field).map_err(|e| {
                        anyhow!("pattern: failed to open the sort column {field}: {e}")
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
        let primary_id = segment
            .fast_fields()
            .u64(PRIMARY_ID_FIELD)
            .map_err(|e| anyhow!("pattern: failed to open the primary id column: {e}"))?;
        let (first_min, first_max) = (slots[0].min_value(), slots[0].max_value());
        let (layout_min, layout_max) = match slots.get(1) {
            None => (u128::from(first_min) << 64, u128::from(first_max) << 64),
            Some(second) if first_min == first_max => (
                (u128::from(first_min) << 64) | u128::from(second.min_value()),
                (u128::from(first_max) << 64) | u128::from(second.max_value()),
            ),
            Some(second) => {
                let rows = (0..segment.max_doc())
                    .filter_map(|doc_id| Some((slots[0].first(doc_id)?, second.first(doc_id)?)));
                let (low, high) = second_slot_at_the_ends(rows, first_min, first_max);
                (
                    (u128::from(first_min) << 64) | u128::from(low),
                    (u128::from(first_max) << 64) | u128::from(high),
                )
            }
        };
        let columns = SegmentColumns {
            slots,
            layout_min,
            layout_max,
            primary_id,
        };
        *opened += 1;
        self.columns.write().unwrap().insert(id, columns.clone());
        Ok(columns)
    }

    /// Every live segment as `(docs, layout_min, layout_max, id)`, from the cached columns; a
    /// segment whose sort columns cannot be opened is left out, with a warning.
    fn geometry(&self, searcher: &tantivy::Searcher) -> Vec<(u32, u128, u128, SegmentId)> {
        let mut opened = 0;
        searcher
            .segment_readers()
            .iter()
            .filter_map(|segment| match self.columns_for(segment, &mut opened) {
                Ok(columns) => Some((
                    segment.num_docs(),
                    columns.layout_min,
                    columns.layout_max,
                    segment.segment_id(),
                )),
                Err(err) => {
                    warn!(
                        "pattern: no bounds for segment {}: {err}",
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

/// The figures every Tantivy index has.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TantivyStats {
    pub(crate) num_docs: u64,
    pub(crate) size_bytes: u64,
    pub(crate) segment_count: usize,
}

/// What `Stats` reports for a pattern index: the Tantivy figures every index has, plus what
/// this one's searches have cost and how its segments are laid out.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct PatternStats {
    pub(crate) tantivy: TantivyStats,
    pub(crate) walk: WalkTotals,
    pub(crate) rewrite: RewriteTotals,
    pub(crate) segments: Vec<SegmentLayout>,
}

pub(crate) type PatternStatsR = anyhow::Result<PatternStats>;

impl PatternBackend {
    fn min_gram(&self) -> usize {
        self.options.min_gram.as_ref().get()
    }

    fn max_gram(&self) -> usize {
        self.options.max_gram.as_ref().get()
    }

    fn rarest_grams(&self) -> Option<usize> {
        self.options.rarest_grams.as_ref().map(|grams| grams.get())
    }

    /// Whether this index keeps an order: its declared columns, or table order. An unordered one
    /// answers in whatever order the walk finds matches, and carries no sort fields at all.
    fn orders_results(&self) -> bool {
        self.options.order_by.is_ordered()
    }
}

/// What a row contributes to a pattern index: the text to index, and for an index that keeps an
/// order, the row's sort prefix.
pub(crate) struct PatternRow<'a> {
    pub(crate) text: &'a str,
    pub(crate) prefix: Option<SortPrefix>,
}

impl PatternBackend {
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
        if self.orders_results() {
            // Also columnar: an ordered walk reads it to tell apart rows whose whole sort prefix
            // is equal. A page's ids still come from the store, which measured no slower.
            schema_builder.add_u64_field(PRIMARY_ID_FIELD, INDEXED | STORED | FAST);
        } else {
            schema_builder.add_u64_field(PRIMARY_ID_FIELD, INDEXED | STORED);
        }
        schema_builder.add_text_field(TEXT_FIELD, text_options);
        if self.orders_results() {
            // FAST, not STORED: the walk reads them once per candidate, and a columnar read is
            // far cheaper than the document store. Their per-segment min/max is also what lets
            // an ordered search skip whole segments.
            for field in self.slot_fields() {
                schema_builder.add_u64_field(&field, FAST);
            }
        }
        schema_builder.build()
    }

    fn analyzer(&self) -> anyhow::Result<TextAnalyzer> {
        let tokenizer = NgramTokenizer::new(self.min_gram(), self.max_gram(), false)
            .map_err(|e| anyhow!("pattern: failed to create the n-gram tokenizer: {e}"))?;
        Ok(TextAnalyzer::builder(tokenizer).build())
    }

    fn create_doc(
        &self,
        schema: &Schema,
        primary_id: PrimaryId,
        row: PatternRow<'_>,
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
            // The caller leaves out a row without a prefix (a null ordering value), so one is
            // always there; a zero prefix keeps the document well-formed if not.
            let prefix = row.prefix.unwrap_or_default();
            for (slot, field) in self.slot_fields().iter().enumerate() {
                doc.add_u64(schema.get_field(field).unwrap(), prefix.0[slot]);
            }
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
    /// come once the writes stop. Run on every tick, since the reader reloads on its own.
    fn after_reload(&self, state: &PatternIndexState) {
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
                "pattern: asking for {} merges of {} segments ({} live)",
                runs.len(),
                runs.iter().map(Vec::len).sum::<usize>(),
                searcher.segment_readers().len()
            );
        }
        for run in runs {
            // The merge runs on Tantivy's own thread; the future only reports its outcome, and
            // the next reload picks the merged segment up either way.
            drop(state.lock_writer().merge(&run));
        }
    }
}

impl PatternBackend {
    /// Starts a rewrite when wide segments hold at least a cap's worth of rows and none is in
    /// progress. The boundaries come from a sample of the wide segments' sort keys, so skewed
    /// keys still give ranges of about `cap` rows each. Called on idle ticks only: a plan made
    /// while a build is still committing samples the rows so far and cuts too few, too wide
    /// ranges, whose segments end up spanning several caps and never get revisited.
    fn maybe_plan_rewrite(&self, state: &PatternIndexState) {
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
                    sample.push(columns.layout_key(doc_id));
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
            .fold((u128::MAX, 0u128), |(lo, hi), &(_, min, max, _)| {
                (lo.min(min), hi.max(max))
            });
        let cap_width =
            narrow_span(segments, searcher.num_docs(), cap) / WIDE_SEGMENT_FACTOR as u128;
        let ranges = quantile_ranges(sample, count, low, high, cap_width);
        info!(
            "pattern: rewriting {l0_docs} rows of {} wide segments into {} ranges",
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
fn rewrite_next_range(state: &PatternIndexState, key: &IndexKey) -> anyhow::Result<bool> {
    let backend = state.backend();
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

    let mut writer = state.lock_writer();
    state
        .commit_and_reload(&mut writer)
        .map_err(|e| anyhow!("pattern: failed to commit before a rewrite: {e}"))?;
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
            .map_err(|e| anyhow!("pattern: failed to open the document store: {e}"))?;
        for doc_id in 0..segment.max_doc() {
            if alive.is_some_and(|alive| !alive.is_alive(doc_id)) {
                continue;
            }
            let layout_key = columns.layout_key(doc_id);
            if layout_key < low || layout_key >= high {
                continue;
            }
            let doc: TantivyDocument = store
                .get(doc_id)
                .map_err(|e| anyhow!("pattern: failed to retrieve doc: {e}"))?;
            let primary_id = doc
                .get_first(primary_id_field)
                .and_then(|value| value.as_u64())
                .map(PrimaryId::from)
                .ok_or_else(|| anyhow!("pattern: missing primary_id in doc"))?;
            let text = doc
                .get_first(text_field)
                .and_then(|value| value.as_str())
                .map(unframed)
                .unwrap_or_default();
            let new_doc = backend.create_doc(
                &state.schema,
                primary_id,
                PatternRow {
                    text,
                    prefix: Some(columns.prefix(doc_id)),
                },
            );
            moved.push((primary_id_term(&state.schema, primary_id), new_doc));
        }
    }
    let count = writer
        .rewrite_documents(moved)
        .map_err(|e| anyhow!("pattern: failed to rewrite a range: {e}"))?;
    state
        .commit_and_reload(&mut writer)
        .map_err(|e| anyhow!("pattern: failed to commit a rewrite: {e}"))?;
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
            "pattern: rewrite of {key} finished, {} rows moved",
            backend.rewrite.docs_rewritten.load(Relaxed)
        );
    }
    // Learn the new segments' bounds and let the policy fold the pieces together.
    backend.after_reload(state);
    Ok(true)
}

type PatternIndexState = Index<PatternBackend>;

impl PatternIndexState {
    fn backend(&self) -> &PatternBackend {
        &self.state
    }
}

/// A new tantivy index for `options`, with the merge policy they ask for.
fn new_index(
    options: &IndexOptionsPattern,
    key_layout: &KeyLayout,
    key: &IndexKey,
    worker: async_channel::Sender<Worker>,
) -> anyhow::Result<PatternIndexState> {
    let backend = PatternBackend {
        key_layout: key_layout.clone(),
        walk: WalkCounters::default(),
        columns: RwLock::new(HashMap::new()),
        bounds: Arc::new(RwLock::new(HashMap::new())),
        plan: Mutex::new(None),
        rewrite: RewriteCounters::default(),
        tick_running: AtomicBool::new(false),
        options: options.clone(),
    };
    let schema = backend.build_schema();
    let analyzer = backend.analyzer()?;
    let policy = backend.merge_policy();
    let index = Index::with_state(schema, TOKENIZER_NAME, analyzer, key, worker, backend)?;
    if let Some(policy) = policy {
        index.set_merge_policy(policy);
    }
    Ok(index)
}

fn primary_id_term(schema: &Schema, primary_id: PrimaryId) -> Term {
    let primary_id_field = schema.get_field(PRIMARY_ID_FIELD).unwrap();
    Term::from_field_u64(primary_id_field, u64::from(primary_id))
}

fn find_partition_id(
    table: &impl TableSearch,
    index_key: &IndexKey,
) -> anyhow::Result<PartitionId> {
    let (partition_id, _) = table
        .partition_id(index_key, None)
        .ok_or_else(|| anyhow!("pattern: partition id not found for index key {index_key:?}"))?;
    Ok(partition_id)
}

/// The value this row is ordered by, or `None` when the index has no sort column, the row has no
/// value for it, or the value has no ordering the index can use.
///
/// `None` is not an error: the document is still indexed, it just sorts lowest. Refusing the row
/// would make an ordered search disagree with an unordered one about which rows exist, which is a
/// worse failure than a row appearing last.
/// The row's place in an ordered index: each ordering column's sort key, complemented for a
/// `DESC` column, then the token. None when an ordering column is null or empty, which leaves the
/// row out of the index, as a materialized view leaves out a row whose key column is null; or when
/// the row's key is not known.
fn read_prefix(
    table: &RwLock<impl TableSearch>,
    options: &IndexOptionsPattern,
    key_layout: &KeyLayout,
    partition_id: PartitionId,
    primary_id: PrimaryId,
) -> Option<SortPrefix> {
    if !options.order_by.is_ordered() {
        return None;
    }
    let table = table.read().unwrap();
    let mut prefix = SortPrefix::default();
    let columns = options.order_by.columns();
    for (slot, column) in columns.iter().enumerate() {
        let value = table.column_value_for(partition_id, primary_id, &column.name)?;
        let key = crate::cql_types::to_sort_key(&value)?;
        prefix.0[slot] = if column.descending { !key } else { key };
    }
    let primary_key = table.primary_key(partition_id, primary_id)?;
    prefix.0[columns.len()] = key_layout.token_key(&primary_key)?;
    Some(prefix)
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
enum PatternQuery {
    /// The query is short enough to be an indexed term itself, so every posting is an exact hit.
    Exact(Box<dyn Query>),
    /// The query is longer than any indexed term: candidates come from intersecting the
    /// postings of its `max_gram`-long substrings and must be verified against the stored value.
    Candidates(Vec<Term>),
}

/// Opens the candidates of a query answered from several grams, one segment at a time.
///
/// A segment whose term dictionary lacks any one of the grams holds no candidate, so it is
/// skipped before a single posting list is opened -- which, for a keyword that occurs in a
/// handful of names, is nearly every segment, and what made a rare long keyword cost a full walk
/// of every segment. The gram last found missing is asked first (move to front): a rare keyword
/// usually has one rare gram, which is absent from nearly every segment, so after the first miss
/// a segment costs one dictionary lookup.
///
/// A segment that holds every gram has its postings built from the entries the check just
/// found, rather than through a `TermQuery` weight per gram, which would look each gram up again
/// and open a field-norm reader an unscored walk never reads. Measured on the 10M-name corpus
/// (`benches/pattern_probe.rs`), that second pass was a third of the walk for 8 to 32
/// characters.
///
/// With `rarest`, only that many of the grams found, those with the fewest documents in the
/// segment, are intersected: the verification drops what the others would have. A long keyword's
/// grams are mostly common ones, and each costs a posting list opened and walked; a few rare ones
/// narrow the candidates nearly as far (`benches/pattern_probe.rs`: at 16 characters, 4 of the
/// keyword's 14 grams took the walk to a quarter, with the same matches).
struct GramProbe {
    terms: Vec<Term>,
    field: tantivy::schema::Field,
    found: Vec<TermInfo>,
    rarest: Option<usize>,
}

impl GramProbe {
    fn new(terms: Vec<Term>, field: tantivy::schema::Field, rarest: Option<usize>) -> Self {
        let found = Vec::with_capacity(terms.len());
        Self {
            terms,
            field,
            found,
            rarest,
        }
    }

    /// The segment's candidates, or `None` when it certainly holds none.
    fn open(&mut self, segment: &SegmentReader) -> anyhow::Result<Option<Box<dyn Scorer>>> {
        let inverted = segment
            .inverted_index(self.field)
            .map_err(|e| anyhow!("pattern: failed to open the term dictionary: {e}"))?;
        self.found.clear();
        for i in 0..self.terms.len() {
            let info = inverted
                .get_term_info(&self.terms[i])
                .map_err(|e| anyhow!("pattern: failed to look a gram up: {e}"))?;
            let Some(info) = info else {
                self.terms[..=i].rotate_right(1);
                return Ok(None);
            };
            self.found.push(info);
        }
        if let Some(rarest) = self.rarest
            && rarest < self.found.len()
        {
            self.found.sort_unstable_by_key(|info| info.doc_freq);
            self.found.truncate(rarest);
        }
        let postings = self
            .found
            .iter()
            .map(|info| {
                let postings = inverted
                    .read_postings_from_terminfo(info, IndexRecordOption::Basic)
                    .map_err(|e| anyhow!("pattern: failed to open a gram's postings: {e}"))?;
                Ok(Box::new(ConstScorer::new(postings, 1.0)) as Box<dyn Scorer>)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Some(intersect_scorers(postings, segment.max_doc())))
    }
}

/// A query's matches, opened one segment at a time.
enum QueryPostings {
    /// One indexed term: every posting is an exact hit.
    Term(Box<dyn Weight>),
    /// Several grams: every posting is a candidate.
    Grams(GramProbe),
}

impl QueryPostings {
    /// The postings for `normalized`, and whether they are candidates rather than exact hits.
    fn new(
        state: &PatternIndexState,
        searcher: &tantivy::Searcher,
        normalized: &str,
    ) -> anyhow::Result<(Self, bool)> {
        let text_field = state.schema.get_field(TEXT_FIELD).unwrap();
        Ok(match build_query(state, normalized)? {
            PatternQuery::Exact(query) => {
                let weight = query
                    .weight(EnableScoring::disabled_from_searcher(searcher))
                    .map_err(|e| anyhow!("pattern: failed to build the query: {e}"))?;
                (Self::Term(weight), false)
            }
            PatternQuery::Candidates(terms) => {
                let rarest = state.backend().rarest_grams();
                (Self::Grams(GramProbe::new(terms, text_field, rarest)), true)
            }
        })
    }

    /// The segment's matches, or `None` when it certainly holds none.
    fn open(&mut self, segment: &SegmentReader) -> anyhow::Result<Option<Box<dyn Scorer>>> {
        match self {
            Self::Term(weight) => weight
                .scorer(segment, 1.0)
                .map(Some)
                .map_err(|e| anyhow!("pattern: failed to run the query: {e}")),
            Self::Grams(probe) => probe.open(segment),
        }
    }
}

fn build_query(state: &PatternIndexState, normalized: &str) -> anyhow::Result<PatternQuery> {
    let text_field = state.schema.get_field(TEXT_FIELD).unwrap();
    let min_gram = state.backend().min_gram();
    let max_gram = state.backend().max_gram();
    let query_len = normalized.chars().count();

    if query_len == 0 {
        return Err(QueryError("pattern: the query must not be empty".to_string()).into());
    }
    if query_len < min_gram {
        return Err(QueryError(format!(
            "pattern: the query has {query_len} characters, \
            but the index only answers queries of at least min_gram={min_gram}"
        ))
        .into());
    }

    if query_len <= max_gram {
        return Ok(PatternQuery::Exact(Box::new(TermQuery::new(
            Term::from_field_text(text_field, normalized),
            IndexRecordOption::Basic,
        ))));
    }

    // Only the longest grams take part: every shorter substring of the query is contained in
    // one of them, so it would add a posting list to intersect without narrowing the result.
    Ok(PatternQuery::Candidates(
        grams_of_length(normalized, max_gram)
            .iter()
            .map(|gram| Term::from_field_text(text_field, gram))
            .collect(),
    ))
}

/// The part of the index's order a search may return, in *walk orientation*: the walk always
/// takes the highest row first, and a search in the declared order is the same walk over the
/// bitwise complement of every sort prefix, with ties between whole prefixes reversed too. The
/// range from the query's `WHERE` (on the first ordering column) and the cursor from the previous
/// page are resolved into bounds once rather than checked separately on every candidate.
#[derive(Clone, Debug, Default)]
pub(crate) struct SortWindow {
    complement: bool,
    /// Inclusive bounds from the range, oriented.
    lower: Option<SortPrefix>,
    upper: Option<SortPrefix>,
    /// The previous page's last row, oriented: the page resumes strictly below it. Without a key
    /// the rows sharing its prefix are taken again rather than any skipped.
    cursor: Option<(SortPrefix, Option<PrimaryKey>)>,
}

impl SortWindow {
    /// `forward` walks the index's declared order from its start. `cursor` and the bounds on the
    /// first slot are in the index's own space.
    pub(crate) fn new(
        forward: bool,
        cursor: Option<Cursor>,
        min_first: Option<u64>,
        max_first: Option<u64>,
    ) -> Self {
        let orient = |prefix: SortPrefix| {
            if forward { prefix.complement() } else { prefix }
        };
        let with_first = |first: u64, rest: u64| {
            let mut prefix = SortPrefix([rest; PREFIX_SLOTS]);
            prefix.0[0] = first;
            prefix
        };
        let low = min_first.map(|key| with_first(key, 0));
        let high = max_first.map(|key| with_first(key, u64::MAX));
        let (lower, upper) = if forward {
            (high.map(orient), low.map(orient))
        } else {
            (low, high)
        };
        Self {
            complement: forward,
            lower,
            upper,
            cursor: cursor.map(|cursor| (orient(cursor.prefix), cursor.primary_key)),
        }
    }

    /// A prefix from the columns, in walk orientation.
    fn orient(&self, prefix: SortPrefix) -> SortPrefix {
        if self.complement {
            prefix.complement()
        } else {
            prefix
        }
    }

    /// The other way: a prefix in walk orientation, back in the index's own space.
    fn unorient(&self, prefix: SortPrefix) -> SortPrefix {
        self.orient(prefix)
    }

    /// An order of two rows in the index's own space, in walk orientation.
    fn orient_order(&self, order: Ordering) -> Ordering {
        if self.complement {
            order.reverse()
        } else {
            order
        }
    }

    /// A segment's layout bounds, in walk orientation.
    fn orient_bounds(&self, min: u128, max: u128) -> (u128, u128) {
        if self.complement {
            (!max, !min)
        } else {
            (min, max)
        }
    }

    /// Whether an oriented prefix is inside the range, and not above the cursor's. A prefix equal
    /// to the cursor's is decided by the keys, in [`SortWindow::below_cursor`].
    fn admits(&self, prefix: SortPrefix) -> bool {
        self.lower.is_none_or(|lower| prefix >= lower)
            && self.upper.is_none_or(|upper| prefix <= upper)
            && self
                .cursor
                .as_ref()
                .is_none_or(|(cursor, _)| prefix <= *cursor)
    }

    /// For an admitted row: whether it comes after the cursor in the walk.
    fn below_cursor(
        &self,
        prefix: SortPrefix,
        key_of_row: impl FnOnce() -> Option<PrimaryKey>,
        ties: &Ties,
    ) -> bool {
        match &self.cursor {
            Some((cursor, Some(cursor_key))) if prefix == *cursor => {
                let order = match key_of_row() {
                    Some(key) => ties.layout.compare_same_token(&key, cursor_key),
                    None => Ordering::Less,
                };
                self.orient_order(order) == Ordering::Less
            }
            _ => true,
        }
    }

    /// Whether a segment spanning `[min, max]` (oriented layout keys) can hold anything in the
    /// window. Answered from the bounds alone, so a segment ruled out here is never opened. The
    /// layout key is only the first two slots, so a bound equal to it may still hold rows on
    /// either side of it and stays in.
    fn overlaps(&self, min: u128, max: u128) -> bool {
        let above = self.lower.is_none_or(|lower| max >= lower.layout_key());
        let below = self.upper.is_none_or(|upper| min <= upper.layout_key())
            && self
                .cursor
                .as_ref()
                .is_none_or(|(cursor, _)| min <= cursor.layout_key());
        above && below
    }
}

/// Orders rows whose whole sort prefix is equal -- same ordering values, same token -- by their
/// primary keys in table order. Looking a key up takes the table, so it is asked only for such
/// exact ties: rows of one partition with equal ordering values, or a token collision.
struct Ties<'a> {
    key_of: &'a dyn Fn(PrimaryId) -> Option<PrimaryKey>,
    layout: &'a KeyLayout,
}

impl Ties<'_> {
    /// In the index's own space. A row the table no longer knows sorts below the rest, and the
    /// primary id keeps the order total.
    fn compare(&self, a: u64, b: u64) -> Ordering {
        if a == b {
            return Ordering::Equal;
        }
        let keys = ((self.key_of)(a.into()), (self.key_of)(b.into()));
        let order = match &keys {
            (Some(a), Some(b)) => self.layout.compare_same_token(a, b),
            (a, b) => a.is_some().cmp(&b.is_some()),
        };
        order.then(a.cmp(&b))
    }
}

/// A row on the page: its oriented sort prefix, its primary id and where its document is.
#[derive(Clone, Copy, Debug)]
struct PageEntry {
    prefix: SortPrefix,
    primary_id: u64,
}

/// The best `limit` rows so far, best first, in walk orientation with exact ties broken by key.
struct Page<'a> {
    limit: usize,
    entries: Vec<PageEntry>,
    window: &'a SortWindow,
    ties: &'a Ties<'a>,
}

impl Page<'_> {
    fn is_full(&self) -> bool {
        self.entries.len() >= self.limit
    }

    fn weakest(&self) -> Option<&PageEntry> {
        self.is_full().then(|| self.entries.last()).flatten()
    }

    fn compare(&self, a: &PageEntry, b: &PageEntry) -> Ordering {
        a.prefix.cmp(&b.prefix).then_with(|| {
            self.window
                .orient_order(self.ties.compare(a.primary_id, b.primary_id))
        })
    }

    /// Whether a row with this prefix might enter the page, before its id is read: always while
    /// the page is not full, and at or above the weakest entry's prefix once it is.
    fn may_take(&self, prefix: SortPrefix) -> bool {
        self.weakest()
            .is_none_or(|weakest| prefix >= weakest.prefix)
    }

    /// Whether the row enters the page.
    fn takes(&self, entry: &PageEntry) -> bool {
        self.weakest()
            .is_none_or(|weakest| self.compare(entry, weakest) == Ordering::Greater)
    }

    fn insert(&mut self, entry: PageEntry) {
        let at = self
            .entries
            .partition_point(|held| self.compare(held, &entry) == Ordering::Greater);
        self.entries.insert(at, entry);
        self.entries.truncate(self.limit);
    }
}

/// One page of an ordered search: the `limit` highest rows in the window (in walk orientation),
/// highest first, with where the next page resumes.
///
/// Two things keep this off the O(matches) path the obvious implementation lands on. Segments are
/// visited by descending upper bound and the walk stops once the next segment's bound cannot beat
/// the page's weakest entry, which skips whole segments unopened. And within a segment the sort
/// prefix -- columnar reads -- is checked before anything else is touched, so a candidate that
/// cannot make the page costs almost nothing. Measured, the second is worth 4-13x on its own and
/// does not depend on how the segments are laid out.
///
/// Returns the page's primary ids, the cursor of its last row when the page filled, and whether
/// the rows are verified matches rather than candidates.
type OrderedPage = (Vec<PrimaryId>, Option<Cursor>, bool);

fn collect_matches_ordered(
    state: &PatternIndexState,
    normalized: &str,
    limit: usize,
    window: SortWindow,
    verify: bool,
    key_of: &dyn Fn(PrimaryId) -> Option<PrimaryKey>,
) -> anyhow::Result<OrderedPage> {
    let text_field = state.schema.get_field(TEXT_FIELD).unwrap();

    let searcher = state.reader.searcher();
    let (mut postings, has_candidates) = QueryPostings::new(state, &searcher, normalized)?;
    // A caller that declines verification takes the candidates as they are: the walk then
    // treats them as exact matches, which is the cheap branch below, and the page says so.
    let needs_verification = has_candidates && verify;
    let verified = !has_candidates || verify;

    let started = Instant::now();
    let mut tally = WalkTally::default();
    let mut segments = Vec::with_capacity(searcher.segment_readers().len());
    for (segment_ord, segment) in searcher.segment_readers().iter().enumerate() {
        let columns = state
            .backend()
            .columns_for(segment, &mut tally.column_opens)?;
        let (lower_bound, upper_bound) =
            window.orient_bounds(columns.layout_min, columns.layout_max);
        // Ruled out from the bounds alone, so the segment is never opened.
        if window.overlaps(lower_bound, upper_bound) {
            segments.push((segment_ord as u32, segment, columns, upper_bound));
        }
    }
    state.backend().prune_columns(searcher.segment_readers());
    segments.sort_by_key(|(_, _, _, upper_bound)| Reverse(*upper_bound));
    tally.segments_considered = segments.len() as u64;
    let prepare = started.elapsed();

    let ties = Ties {
        key_of,
        layout: &state.backend().key_layout,
    };
    let mut page = Page {
        limit,
        entries: Vec::with_capacity(limit + 1),
        window: &window,
        ties: &ties,
    };
    let mut candidates: Vec<(SortPrefix, u32)> = Vec::new();
    for (_segment_ord, segment, columns, upper_bound) in segments {
        if let Some(weakest) = page.weakest()
            && upper_bound < weakest.prefix.layout_key()
        {
            // Nothing in this segment, nor in any later one, can enter the page. (A segment
            // ending exactly at the weakest layout key may still hold a row above it.)
            break;
        }
        let Some(mut scorer) = postings.open(segment)? else {
            continue;
        };
        tally.segments_opened += 1;
        let alive = segment.alive_bitset();
        // The row's entry if it is in the window and enters the page; its id is read only then.
        let entry_of = |page: &Page, prefix: SortPrefix, doc_id: u32| -> Option<PageEntry> {
            let primary_id = columns.primary_id.first(doc_id).unwrap_or(0);
            let entry = PageEntry { prefix, primary_id };
            (window.below_cursor(prefix, || key_of(primary_id.into()), &ties) && page.takes(&entry))
                .then_some(entry)
        };

        if !needs_verification {
            // Every match is exact, so it enters the page straight from the columns.
            let mut doc_id = scorer.doc();
            while doc_id != TERMINATED {
                tally.postings_scanned += 1;
                if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                    let prefix = window.orient(columns.prefix(doc_id));
                    if window.admits(prefix)
                        && page.may_take(prefix)
                        && let Some(entry) = entry_of(&page, prefix, doc_id)
                    {
                        tally.heap_entrants += 1;
                        page.insert(entry);
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
        // prefixes from the columns, then verify from the highest down and stop at the first
        // whose prefix can no longer enter the page. The store is read for the page's rows plus
        // the false positives above them, whatever the segment holds.
        let store = segment
            .get_store_reader(STORE_CACHE_BLOCKS)
            .map_err(|e| anyhow!("pattern: failed to open the document store: {e}"))?;
        candidates.clear();
        let mut doc_id = scorer.doc();
        while doc_id != TERMINATED {
            tally.postings_scanned += 1;
            if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                let prefix = window.orient(columns.prefix(doc_id));
                if window.admits(prefix) && page.may_take(prefix) {
                    candidates.push((prefix, doc_id));
                }
            }
            doc_id = scorer.advance();
        }
        candidates.sort_unstable_by(|a, b| b.cmp(a));
        for &(prefix, doc_id) in &candidates {
            if !page.may_take(prefix) {
                break;
            }
            let Some(entry) = entry_of(&page, prefix, doc_id) else {
                continue;
            };
            tally.store_reads += 1;
            let verified = store
                .get::<TantivyDocument>(doc_id)
                .map_err(|e| anyhow!("pattern: failed to retrieve doc: {e}"))?
                .get_first(text_field)
                .and_then(|value| value.as_str())
                .is_some_and(|text| text.contains(normalized));
            if verified {
                tally.heap_entrants += 1;
                page.insert(entry);
            }
        }
    }

    // A cursor says there may be more below this point. That is only true if the page filled: a
    // walk that ended with fewer than `limit` matches visited every segment overlapping the window,
    // so there is nothing left to resume from and a cursor would only cost the caller an empty
    // round trip. (Rows dropped later, in `handle_search`, shorten the page after this point and
    // must not suppress the cursor -- which is why this asks the page, not the returned rows.)
    let resolving = Instant::now();
    let resume_at = page.weakest().map(|weakest| Cursor {
        prefix: window.unorient(weakest.prefix),
        primary_key: key_of(weakest.primary_id.into()),
    });
    let ids = page
        .entries
        .iter()
        .map(|entry| PrimaryId::from(entry.primary_id))
        .collect();
    let page_resolve = resolving.elapsed();
    state.backend().walk.record(
        tally,
        WalkTimes {
            walk: started.elapsed(),
            prepare,
            page_resolve,
        },
    );
    Ok((ids, resume_at, verified))
}

/// Walks the matching documents in index order and returns the primary ids of the first
/// `limit` verified matches after skipping `offset` of them.
///
/// The query is executed once, unscored, and the walk stops as soon as enough matches are
/// found, so a hot single-character query does not pay for its whole posting list. The stored
/// document has to be read anyway for the primary id, so verifying the containment on the way
/// costs no extra I/O.
fn collect_matches(
    state: &PatternIndexState,
    normalized: &str,
    limit: usize,
    offset: usize,
    verify: bool,
) -> anyhow::Result<(Vec<PrimaryId>, bool)> {
    let text_field = state.schema.get_field(TEXT_FIELD).unwrap();
    let primary_id_field = state.schema.get_field(PRIMARY_ID_FIELD).unwrap();

    let searcher = state.reader.searcher();
    let (mut postings, has_candidates) = QueryPostings::new(state, &searcher, normalized)?;
    // The store is read here anyway, for the primary id, so declining verification saves the
    // text comparison only; it exists so that both walks answer the request the same way.
    let needs_verification = has_candidates && verify;
    let verified = !has_candidates || verify;

    let started = Instant::now();
    let mut tally = WalkTally {
        segments_considered: searcher.segment_readers().len() as u64,
        ..WalkTally::default()
    };
    let mut to_skip = offset;
    let mut matches = Vec::with_capacity(limit);
    'segments: for segment in searcher.segment_readers() {
        let Some(mut scorer) = postings.open(segment)? else {
            continue;
        };
        tally.segments_opened += 1;
        let alive = segment.alive_bitset();
        let store = segment
            .get_store_reader(STORE_CACHE_BLOCKS)
            .map_err(|e| anyhow!("pattern: failed to open the document store: {e}"))?;

        let mut doc_id = scorer.doc();
        while doc_id != TERMINATED {
            tally.postings_scanned += 1;
            if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                tally.store_reads += 1;
                let doc: TantivyDocument = store
                    .get(doc_id)
                    .map_err(|e| anyhow!("pattern: failed to retrieve doc: {e}"))?;
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
                            .ok_or_else(|| anyhow!("pattern: missing primary_id in doc"))?;
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
    state.backend().walk.record(
        tally,
        WalkTimes {
            walk: started.elapsed(),
            ..WalkTimes::default()
        },
    );
    Ok((matches, verified))
}

fn handle_pattern_stats(state: &PatternIndexState) -> PatternStatsR {
    let searcher = state.reader.searcher();
    let tantivy = TantivyStats {
        num_docs: searcher.num_docs(),
        size_bytes: searcher
            .space_usage()
            .map_err(|e| anyhow!("pattern: failed to compute space usage: {e}"))?
            .total()
            .get_bytes(),
        segment_count: searcher.segment_readers().len(),
    };
    let segments = searcher
        .segment_readers()
        .iter()
        .map(|segment| {
            let mut opened = 0;
            let bounds = state
                .backend()
                .orders_results()
                .then(|| state.backend().columns_for(segment, &mut opened))
                .transpose()?
                // The first slot: the first ordering column's sort key, or the token.
                .map(|columns| {
                    (
                        (columns.layout_min >> 64) as u64,
                        (columns.layout_max >> 64) as u64,
                    )
                });
            Ok(SegmentLayout {
                docs: segment.num_docs(),
                deleted: segment.num_deleted_docs(),
                sort_min: bounds.map(|(min, _)| min),
                sort_max: bounds.map(|(_, max)| max),
            })
        })
        .collect::<anyhow::Result<_>>()?;
    Ok(PatternStats {
        tantivy,
        walk: state.backend().walk.snapshot(),
        rewrite: state.backend().rewrite_totals(),
        segments,
    })
}

// One argument per thing the request says; bundling them would only move the count elsewhere.
#[allow(clippy::too_many_arguments)]
fn handle_search(
    state: &PatternIndexState,
    table: &RwLock<impl TableSearch>,
    index_key: &IndexKey,
    query: &str,
    kind: MatchKind,
    limit: Limit,
    offset: usize,
    window: SearchWindow,
    verify: bool,
) -> PatternSearchR {
    // A prefix or suffix query is a containment query for the keyword with the value's frame
    // mark on the anchored side, so from here on nothing knows the kind.
    let normalized = pattern_of(
        kind,
        &normalize(query, state.backend().options.case_sensitive),
    );
    let limit: usize = (*limit.as_ref()).into();
    let partition_id = find_partition_id(table.read().unwrap().deref(), index_key)?;
    let (primary_ids, next_cursor, verified) = if state.backend().orders_results() {
        let columns = state.backend().options.order_by.columns();
        // The request names a direction for the first ordering column; the walk needs to know
        // whether that is the declared order or its reverse. None is the declared order, and an
        // index in table order declares ascending.
        let first_descending = columns.first().is_some_and(|column| column.descending);
        let forward = match window.order {
            None => true,
            Some(SortOrder::Asc) => !first_descending,
            Some(SortOrder::Desc) => first_descending,
        };
        // A range is on the first ordering column's values; a DESC column is stored complemented,
        // which swaps the ends.
        let (min_first, max_first) = if columns.is_empty() {
            (None, None)
        } else if first_descending {
            (
                window.max_sort_key.map(|key| !key),
                window.min_sort_key.map(|key| !key),
            )
        } else {
            (window.min_sort_key, window.max_sort_key)
        };
        let key_of =
            |primary_id: PrimaryId| table.read().unwrap().primary_key(partition_id, primary_id);
        let sort_window = SortWindow::new(forward, window.cursor, min_first, max_first);
        collect_matches_ordered(state, &normalized, limit, sort_window, verify, &key_of)?
    } else {
        let (ids, verified) = collect_matches(state, &normalized, limit, offset, verify)?;
        (ids, None, verified)
    };

    let table = table.read().unwrap();
    Ok(PatternPage {
        primary_keys: primary_ids
            .into_iter()
            .filter_map(|primary_id| table.primary_key(partition_id, primary_id))
            .collect(),
        next_cursor,
        verified,
    })
}

pub(crate) fn new(
    index: PatternIndexConfiguration,
    table: Arc<RwLock<impl TableSearch + Send + Sync + 'static>>,
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
    commit_interval: Duration,
    commit_threshold: usize,
    metrics: Option<Arc<Metrics>>,
) -> mpsc::Sender<PatternIndex> {
    let (tx, mut rx) = mpsc::channel::<PatternIndex>(perf::channel_size().into());
    tokio::spawn(async move {
        let key = index.key.clone();
        // The layout changes on the ticks -- merges after the writes stop, a rewrite in progress
        // -- with no write or search to mark the index for a refresh, so the ticks mark it.
        let mark_dirty = |key: &IndexKey| {
            if let Some(metrics) = &metrics {
                metrics.mark_dirty(key.keyspace().as_ref(), key.index().as_ref());
            }
        };
        debug!("pattern index actor starting for {key}");
        let options = index.options.clone();
        let key_layout = index.key_layout.clone();
        let mut indexes = Indexes::new(
            key.clone(),
            Arc::clone(&table),
            worker.clone(),
            memory,
            commit_interval,
            commit_threshold,
            move |key: &IndexKey, worker| new_index(&options, &key_layout, key, worker),
        )
        .await;

        loop {
            tokio::select! {
                msg = rx.recv() => {
                    let Some(msg) = msg else {
                        break;
                    };
                    match msg {
                        PatternIndex::AddDocument {
                            partition_id,
                            primary_id,
                            document,
                            in_progress,
                        } => {
                            // Table::upsert writes the column before it emits the operation that
                            // got us here, so the value is already in place.
                            let prefix = read_prefix(
                                table.as_ref(),
                                &index.options,
                                &index.key_layout,
                                partition_id,
                                primary_id,
                            );
                            if index.options.order_by.is_ordered() && prefix.is_none() {
                                // A null ordering value leaves the row out, as a view leaves out
                                // a row with a null key column. An update that set it to null
                                // already removed the old document (RemoveBeforeAddValue).
                                debug!(
                                    "pattern: {key}: row {primary_id:?} has no sort prefix, not indexed"
                                );
                                continue;
                            }
                            indexes
                                .add_document_with(primary_id, in_progress, move |index| {
                                    index.backend().create_doc(
                                        &index.schema,
                                        primary_id,
                                        PatternRow {
                                            text: &document,
                                            prefix,
                                        },
                                    )
                                })
                                .await;
                        }
                        PatternIndex::RemoveDocument {
                            primary_id,
                            in_progress,
                        } => {
                            indexes.remove_document(primary_id, in_progress).await;
                        }
                        PatternIndex::Count { tx, index_key, .. } => {
                            _ = tx.send(Ok(indexes.count(&index_key)));
                        }
                        PatternIndex::Search {
                            index_key,
                            query,
                            kind,
                            limit,
                            offset,
                            window,
                            verify,
                            tx,
                        } => {
                            let Some(state) = indexes.get(&index_key) else {
                                _ = tx.send(Ok(PatternPage {
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
                        PatternIndex::Stats { index_key, tx } => {
                            let Some(state) = indexes.get(&index_key)
                            else {
                                _ = tx.send(Ok(Default::default()));
                                continue;
                            };
                            worker
                                .spawn_blocking(move || {
                                    let result = handle_pattern_stats(&state);
                                    _ = tx.send(result);
                                })
                                .await;
                        }
                    }
                }
                _ = indexes.commit_interval_tick() => {
                    // An index with nothing to commit is idle: the rewrite runs then, one range
                    // per tick at most, so it paces itself to the ticks and never starves
                    // ingestion or searches. One whose writer is held right now is not idle.
                    let idle: Vec<bool> = indexes
                        .iter()
                        .map(|index| index.has_uncommitted_docs() == Some(false))
                        .collect();
                    indexes.commit_or_reload().await;
                    for (state, idle) in indexes.iter().zip(idle) {
                        // A step of the rewrite at 10M rows takes seconds under the writer lock.
                        // Run on the worker, it took a runtime thread, and the steps the next ticks
                        // queued behind it took the others, waiting for the lock, until the node
                        // stopped answering HTTP (AWS runs eb0023ac and 1c3ae034). So the work runs
                        // on tokio's blocking pool, and a tick adds none while an index's previous
                        // work still runs.
                        let Some(work) = TickWork::start(state) else {
                            continue;
                        };
                        let key = key.clone();
                        tokio::task::spawn_blocking(move || {
                            let state = &work.0;
                            let backend = state.backend();
                            backend.after_reload(state);
                            if !idle {
                                return;
                            }
                            if !backend.rewrite_in_progress() {
                                backend.maybe_plan_rewrite(state);
                            }
                            if backend.rewrite_in_progress()
                                && let Err(err) = rewrite_next_range(state, &key)
                            {
                                error!("pattern: rewrite of {key} failed: {err}");
                            }
                        });
                    }
                    mark_dirty(&index.key);
                }
            }
        }
        debug!("pattern index actor finished for {key}");
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
    use crate::PrimaryKey;
    use crate::RarestGrams;
    use crate::RewriteWideSegments;
    use crate::SegmentMaxDocs;
    use crate::memory::Allocate;
    use crate::table::IndexIdGenerator;
    use crate::table::MockTableSearch;
    use crate::table::PartitionId;
    use crate::worker;
    use rstest::rstest;
    use scylla::value::CqlValue;
    use std::cmp::Reverse;
    use std::num::NonZeroUsize;
    use tokio::sync::watch;

    use super::super::actor::PatternIndexExt;

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

    fn options(min_gram: usize, max_gram: usize, case_sensitive: bool) -> IndexOptionsPattern {
        IndexOptionsPattern {
            min_gram: MinGram::from(NonZeroUsize::new(min_gram).unwrap()),
            max_gram: MaxGram::from(NonZeroUsize::new(max_gram).unwrap()),
            case_sensitive: CaseSensitive::from(case_sensitive),
            // The matching tests run on an unordered index, which returns matches as it finds
            // them; the ordered and table-order walks have tests of their own.
            order_by: OrderBy::Unordered,
            segment_max_docs: SegmentMaxDocs::default(),
            rewrite_wide_segments: RewriteWideSegments::default(),
            rarest_grams: RarestGrams::default(),
        }
    }

    /// Options with a sort column. The mock table answers `column_value_for` with the primary id,
    /// whatever the column is called.
    /// The sort key the index stores for a row whose sort column holds `value`.
    ///
    /// Not the value itself: signed types are biased so that negatives sort below positives, and
    /// the window bounds are in that space. A caller building a window from a CQL value has to
    /// apply the same conversion -- see the note on `PostIndexLikeRequest`.
    fn sort_key(value: i64) -> u64 {
        crate::cql_types::to_sort_key(&CqlValue::BigInt(value)).unwrap()
    }

    /// Declared `DESC`, so a search that names no direction gets the highest first, as the
    /// newest-first search box these tests stand for does.
    fn ordered_options() -> IndexOptionsPattern {
        IndexOptionsPattern {
            order_by: "sort_col DESC".parse().unwrap(),
            ..IndexOptionsPattern::default()
        }
    }

    fn unordered_options() -> IndexOptionsPattern {
        IndexOptionsPattern {
            order_by: OrderBy::Unordered,
            ..IndexOptionsPattern::default()
        }
    }

    /// The fixtures' tables: a single bigint partition key, no clustering columns.
    fn test_layout() -> KeyLayout {
        KeyLayout {
            partition_key_count: 1,
            clustering_descending: Arc::new([]),
        }
    }

    /// The token of the row whose key is `id`, as the index keeps it.
    fn token_of(id: u64) -> u64 {
        test_layout()
            .token_key(&PrimaryKey::from(vec![CqlValue::BigInt(id as i64)]))
            .unwrap()
    }

    fn make_sender_with_options(options: IndexOptionsPattern) -> mpsc::Sender<PatternIndex> {
        make_sender_with_threshold(options, TEST_COMMIT_THRESHOLD)
    }

    fn make_sender_with_threshold(
        options: IndexOptionsPattern,
        commit_threshold: usize,
    ) -> mpsc::Sender<PatternIndex> {
        new(
            PatternIndexConfiguration {
                key: make_index_key(),
                options,
                key_layout: test_layout(),
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
    fn make_sender() -> mpsc::Sender<PatternIndex> {
        make_sender_with_options(unordered_options())
    }

    async fn add_doc(sender: &mpsc::Sender<PatternIndex>, primary: u64, content: &str) {
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

    async fn rm_doc_no_wait(sender: &mpsc::Sender<PatternIndex>, primary: u64) {
        let (tx, _rx) = mpsc::channel(1);
        sender
            .remove_document(primary.into(), AsyncInProgress::Fullscan(tx))
            .await
            .unwrap();
    }

    async fn rm_doc(sender: &mpsc::Sender<PatternIndex>, primary: u64) {
        let (tx, mut rx) = mpsc::channel(1);
        sender
            .remove_document(primary.into(), AsyncInProgress::Fullscan(tx))
            .await
            .unwrap();
        rx.recv().await;
    }

    async fn add_docs(sender: &mpsc::Sender<PatternIndex>, docs: &[(u64, &str)]) {
        for (primary, content) in docs {
            add_doc(sender, *primary, content).await;
        }
    }

    fn limit(n: usize) -> Limit {
        Limit::from(NonZeroUsize::new(n).unwrap())
    }

    /// The primary ids matching `query`, sorted, so tests are independent of index order.
    async fn search(sender: &mpsc::Sender<PatternIndex>, query: &str) -> Vec<i64> {
        search_page(sender, query, 100, 0).await
    }

    /// Like `search_page`, but keeps the order the index returned -- the sorting helpers above
    /// cannot tell a correctly ordered answer from a scrambled one.
    async fn search_ordered(
        sender: &mpsc::Sender<PatternIndex>,
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
            order: Some(SortOrder::Desc),
            cursor: cursor.map(|sort_key| {
                let id = (sort_key ^ (1 << 63)) as i64;
                // Declared DESC, so the column's key is stored complemented.
                let mut prefix = SortPrefix::default();
                prefix.0[0] = !sort_key;
                prefix.0[1] = token_of(id as u64);
                Cursor {
                    prefix,
                    primary_key: Some(PrimaryKey::from(vec![CqlValue::BigInt(id)])),
                }
            }),
            min_sort_key,
            max_sort_key,
        }
    }

    /// The window that resumes after a page, in the given direction.
    fn resume(order: SortOrder, cursor: Cursor) -> SearchWindow {
        SearchWindow {
            order: Some(order),
            cursor: Some(cursor),
            ..SearchWindow::default()
        }
    }

    fn ascending() -> SearchWindow {
        SearchWindow {
            order: Some(SortOrder::Asc),
            ..SearchWindow::default()
        }
    }

    async fn search_page(
        sender: &mpsc::Sender<PatternIndex>,
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
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn a_long_query_with_a_single_distinct_gram_is_exact() {
        let sender = make_sender();
        add_docs(
            &sender,
            &[(1, "aaa"), (2, "aaaa"), (3, "baaaab"), (4, "aa")],
        )
        .await;

        // "aaaa" has one distinct gram, so its intersection has one member.
        assert_eq!(search(&sender, "aaaa").await, vec![2, 3]);
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

    /// Intersecting only the rarest grams lets more candidates through, and the verification drops
    /// them again: the matches are those of the whole intersection, ordered or not.
    #[rstest]
    #[case::one_unordered(1, false)]
    #[case::two_unordered(2, false)]
    #[case::one_ordered(1, true)]
    #[case::two_ordered(2, true)]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn rarest_grams_return_the_matches_of_all_grams(
        #[case] rarest: usize,
        #[case] ordered: bool,
    ) {
        let sender = make_sender_with_options(IndexOptionsPattern {
            rarest_grams: rarest.to_string().parse().unwrap(),
            order_by: if ordered {
                "sort_col".parse().unwrap()
            } else {
                OrderBy::default()
            },
            ..options(1, 3, true)
        });
        // Every decoy holds some of the grams of "abcde", the first one all of them.
        let docs = [
            (1, "abcd cde"),
            (2, "abcde"),
            (3, "xbcdex"),
            (4, "zzabcdezz"),
            (5, "cdeab"),
            (6, "abc bcd"),
        ];
        add_docs(&sender, &docs).await;

        let mut ids = if ordered {
            search_ordered(&sender, "abcde", 10, SearchWindow::default())
                .await
                .0
        } else {
            search(&sender, "abcde").await
        };
        ids.sort_unstable();
        assert_eq!(ids, vec![2, 4]);
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
        // Nothing is read from the store: the matches are exact, and the page's ids come from
        // the id column.
        assert_eq!(walk.store_reads, 0, "{walk:?}");
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
        let sender = make_sender_with_options(IndexOptionsPattern {
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
            IndexOptionsPattern {
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
            IndexOptionsPattern {
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
        // pruned by its bounds) versus the 5 it read to verify; the page's ids come from the id
        // column the walk reads anyway.
        let walk = sender.stats(make_index_key()).await.unwrap().walk;
        assert!(walk.postings_scanned >= 5, "{walk:?}");
        assert_eq!(walk.heap_entrants, 5, "{walk:?}");
        assert_eq!(walk.store_reads, 5, "{walk:?}");
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
        let sample: Vec<u128> = (0..100).map(|n| n * 10).collect();
        let ranges = quantile_ranges(sample, 4, 0, 990, u128::MAX);
        assert_eq!(ranges, vec![(0, 250), (250, 500), (500, 750), (750, 991)]);
        assert_eq!(
            quantile_ranges(vec![5, 5, 5, 5], 3, 0, 5, u128::MAX),
            vec![(0, 5), (5, 6)]
        );
        assert_eq!(
            quantile_ranges(vec![1, 2], 1, 0, 2, u128::MAX),
            vec![(0, 3)]
        );
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
            IndexOptionsPattern {
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

        let widest_of = |stats: &PatternStats| {
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

    /// The columns are opened once per segment: by the commit tick after the reload that made the
    /// segment searchable, or by a search that comes first, and never again -- and the time
    /// before the first posting is part of the walk time.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn columns_are_opened_once_per_segment() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        search_ordered(&sender, "将军", 10, SearchWindow::default()).await;
        let first = sender.stats(make_index_key()).await.unwrap();
        assert!(
            first.walk.column_opens <= first.segments.len() as u64,
            "{:?}",
            first.walk
        );
        assert!(first.walk.prepare_nanos <= first.walk.walk_nanos);

        search_ordered(&sender, "将军", 10, SearchWindow::default()).await;
        let second = sender.stats(make_index_key()).await.unwrap();
        assert_eq!(
            second.walk.column_opens, first.walk.column_opens,
            "{:?}",
            second.walk
        );
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
        sender: &mpsc::Sender<PatternIndex>,
        query: &str,
        limit_n: usize,
        window: SearchWindow,
    ) -> PatternPage {
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

    fn ids_of(page: &PatternPage) -> Vec<i64> {
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
        let sender = make_sender_with_options(IndexOptionsPattern {
            order_by: "sort_col DESC".parse().unwrap(),
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
        assert_eq!(walk.store_reads, 0, "candidates are not read: {walk:?}");

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
        let sender = make_sender_with_options(IndexOptionsPattern {
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
        let sender = make_sender_with_options(IndexOptionsPattern {
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
        mock.expect_column_value_for()
            .returning(|_partition_id, primary_id, _column| {
                Some(CqlValue::BigInt((u64::from(primary_id) as i64 + 1) / 2))
            });
        mock.expect_is_valid_for().returning(|_, _, _| true);
        mock.expect_target_column().returning(|_| None);
        Arc::new(RwLock::new(mock))
    }

    fn make_sender_with_ties(options: IndexOptionsPattern) -> mpsc::Sender<PatternIndex> {
        new(
            PatternIndexConfiguration {
                key: make_index_key(),
                options,
                key_layout: test_layout(),
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
        sender: &mpsc::Sender<PatternIndex>,
        query: &str,
        page: usize,
        order: SortOrder,
    ) -> (Vec<i64>, usize) {
        let mut seen = Vec::new();
        let mut pages = 0;
        let mut window = SearchWindow {
            order: Some(order),
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

        // Sort keys 1,1,2,2,3,3,4 for ids 1..=7. Declared DESC: highest key first, and rows
        // sharing a key in table order, i.e. by token. Ascending is the exact reverse.
        let mut expected: Vec<i64> = (1..=7).collect();
        expected.sort_by_key(|id| (Reverse((id + 1) / 2), token_of(*id as u64)));
        let (seen, _) = all_pages(&sender, "将军", page, SortOrder::Desc).await;
        assert_eq!(seen, expected);

        let (seen, _) = all_pages(&sender, "将军", page, SortOrder::Asc).await;
        expected.reverse();
        assert_eq!(seen, expected);
    }

    /// A cursor resumes strictly after its row, found by key, so a row deleted since resumes
    /// exactly too. One that names no row resumes at its prefix inclusive: the rows sharing it come
    /// again, which a client can dedupe, rather than being lost, which it cannot repair.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn a_cursor_resumes_after_its_row_and_without_one_takes_its_prefix_again() {
        let sender = make_sender_with_ties(ordered_options());
        let docs: Vec<(u64, String)> = (1..=6).map(|i| (i, format!("user{i}将军"))).collect();
        let docs: Vec<(u64, &str)> = docs.iter().map(|(i, s)| (*i, s.as_str())).collect();
        add_docs(&sender, &docs).await;
        let mut expected: Vec<i64> = (1..=6).collect();
        expected.sort_by_key(|id| (Reverse((id + 1) / 2), token_of(*id as u64)));

        let (first, cursor) = search_ordered(&sender, "将军", 3, SearchWindow::default()).await;
        assert_eq!(first, expected[..3]);
        let cursor = cursor.unwrap();
        assert_eq!(
            cursor.primary_key,
            Some(PrimaryKey::from(vec![CqlValue::BigInt(expected[2])]))
        );

        // With the row: strictly after it.
        let (second, _) =
            search_ordered(&sender, "将军", 3, resume(SortOrder::Desc, cursor.clone())).await;
        assert_eq!(second, expected[3..]);
        // The row deleted since: still exactly after it.
        rm_doc(&sender, expected[2] as u64).await;
        let (second, _) =
            search_ordered(&sender, "将军", 3, resume(SortOrder::Desc, cursor.clone())).await;
        assert_eq!(second, expected[3..]);
        add_doc(
            &sender,
            expected[2] as u64,
            &format!("user{}将军", expected[2]),
        )
        .await;
        // Without a key: the row with the cursor's prefix comes again, nothing is lost.
        let unknown = Cursor {
            prefix: cursor.prefix,
            primary_key: None,
        };
        let (second, _) =
            search_ordered(&sender, "将军", 3, resume(SortOrder::Desc, unknown)).await;
        assert_eq!(second, expected[2..5]);
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
        // The page ended on the row with key 2; declared DESC, the column is stored complemented.
        assert_eq!(cursor.prefix.0[0], !sort_key(2));

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
                order: Some(SortOrder::Asc),
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
        sender: &mpsc::Sender<PatternIndex>,
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

    /// The rewrite of a large index must not stop the node answering. Its steps hold the writer
    /// lock for seconds. The actor's tick asked every index for uncommitted documents through that
    /// lock, so the actor, and every request behind it, waited out each step; and steps run on the
    /// worker took runtime threads. On AWS the node stopped answering HTTP once a 10M-row build
    /// began its rewrite; at 10M rows on 4 cores this test saw answers 5.9 s late. Too slow for
    /// the suite: run with `--ignored` (STALL_ROWS sets the size).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn a_rewrite_in_progress_leaves_the_actor_answering() {
        // STALL_ROWS=10000000, on 4 cores (taskset -c 0-3), is the size and shape of the AWS run.
        let rows: u64 = std::env::var("STALL_ROWS").map_or(2_000_000, |rows| rows.parse().unwrap());
        // Shuffled sort keys, as a build by full scan leaves them: every segment spans the range.
        let mut mock = MockTableSearch::new();
        let index_id = IndexIdGenerator::new().next(true).unwrap();
        let partition_id = PartitionId::global(index_id);
        mock.expect_index_id().returning(move |_| Some(index_id));
        mock.expect_partition_id()
            .returning(move |_, _| Some((partition_id, None)));
        mock.expect_primary_key().returning(|_, primary_id| {
            Some(PrimaryKey::from(vec![CqlValue::BigInt(
                u64::from(primary_id) as i64,
            )]))
        });
        mock.expect_column_value_for()
            .returning(move |_, primary_id, _| {
                let shuffled = u64::from(primary_id).wrapping_mul(2_654_435_761) % rows;
                Some(CqlValue::BigInt(shuffled as i64))
            });
        let sender = new(
            PatternIndexConfiguration {
                key: make_index_key(),
                options: IndexOptionsPattern {
                    order_by: "sort_col".parse().unwrap(),
                    segment_max_docs: "200000".parse().unwrap(),
                    rewrite_wide_segments: "true".parse().unwrap(),
                    ..IndexOptionsPattern::default()
                },
                key_layout: test_layout(),
            },
            Arc::new(RwLock::new(mock)),
            worker::new(),
            make_memory_actor(),
            COMMIT_INTERVAL,
            MAX_UNCOMMITTED_THRESHOLD,
            None,
        );
        let (tx, mut rx) = mpsc::channel(1);
        for n in 1..=rows {
            let name = format!("用户{}号玩家{}", n % 9973, n);
            sender
                .add_document(
                    test_partition_id(),
                    n.into(),
                    name,
                    AsyncInProgress::Fullscan(tx.clone()),
                )
                .await
                .unwrap();
        }
        drop(tx);
        rx.recv().await;

        // Ask while the rewrite runs, until it has finished or two minutes pass.
        let started = Instant::now();
        let mut slowest = Duration::ZERO;
        let mut slowest_count = Duration::ZERO;
        let mut ranges = (0, 0);
        while started.elapsed() < Duration::from_secs(120) {
            let asked = Instant::now();
            sender.count(make_index_key()).await.unwrap();
            slowest_count = slowest_count.max(asked.elapsed());
            let asked = Instant::now();
            let stats = sender.stats(make_index_key()).await.unwrap();
            slowest = slowest.max(asked.elapsed());
            ranges = (stats.rewrite.ranges_done, stats.rewrite.ranges_total);
            if ranges.1 > 0 && ranges.0 == ranges.1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        println!(
            "STALL rewrite {}/{} ranges after {:.0}s, slowest stats {:?}, slowest count {:?}",
            ranges.0,
            ranges.1,
            started.elapsed().as_secs_f64(),
            slowest,
            slowest_count
        );
        assert!(ranges.1 > 0, "no rewrite was planned");
        assert!(
            slowest_count < Duration::from_secs(1),
            "the actor stalled for {slowest_count:?}"
        );
        assert!(slowest < Duration::from_secs(1), "stats took {slowest:?}");
    }

    /// A table whose rows are named by id: `key_of` gives a row's primary key and `value_of` the
    /// value of an ordering column, `None` for a null.
    fn make_table(
        key_of: fn(u64) -> PrimaryKey,
        value_of: fn(u64, &str) -> Option<CqlValue>,
    ) -> Arc<RwLock<MockTableSearch>> {
        let index_id = IndexIdGenerator::new().next(true).unwrap();
        let partition_id = PartitionId::global(index_id);
        let mut mock = MockTableSearch::new();
        mock.expect_index_id()
            .returning(move |_index_key| Some(index_id));
        mock.expect_partition_id()
            .returning(move |_index_key, _restrictions| Some((partition_id, None)));
        mock.expect_primary_key()
            .returning(move |_partition_id, primary_id| Some(key_of(u64::from(primary_id))));
        mock.expect_column_value_for()
            .returning(move |_partition_id, primary_id, column| {
                value_of(u64::from(primary_id), column.as_ref())
            });
        Arc::new(RwLock::new(mock))
    }

    fn make_sender_for(
        options: IndexOptionsPattern,
        table: Arc<RwLock<MockTableSearch>>,
        key_layout: KeyLayout,
    ) -> mpsc::Sender<PatternIndex> {
        new(
            PatternIndexConfiguration {
                key: make_index_key(),
                options,
                key_layout,
            },
            table,
            worker::new(),
            make_memory_actor(),
            TEST_COMMIT_INTERVAL,
            TEST_COMMIT_THRESHOLD,
            None,
        )
    }

    fn bigint_key(id: u64) -> PrimaryKey {
        PrimaryKey::from(vec![CqlValue::BigInt(id as i64)])
    }

    /// Every page of a search, followed from its cursor; the pages concatenated.
    async fn every_page(
        sender: &mpsc::Sender<PatternIndex>,
        query: &str,
        page: usize,
        order: Option<SortOrder>,
    ) -> Vec<i64> {
        let mut seen = Vec::new();
        let mut window = SearchWindow {
            order,
            ..SearchWindow::default()
        };
        loop {
            let (ids, cursor) = search_ordered(sender, query, page, window).await;
            seen.extend(ids);
            match cursor {
                Some(cursor) => {
                    window = SearchWindow {
                        order,
                        cursor: Some(cursor),
                        ..SearchWindow::default()
                    }
                }
                None => return seen,
            }
        }
    }

    /// An index without `order_by` answers in table order -- by token for a table without
    /// clustering columns -- pages through it with a cursor, and a smaller limit is a prefix of a
    /// larger one.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn an_index_without_order_by_answers_in_table_order() {
        let sender = make_sender_for(
            IndexOptionsPattern::default(),
            make_table(bigint_key, |_, _| None),
            test_layout(),
        );
        let docs: Vec<(u64, String)> = (1..=12).map(|i| (i, format!("user{i}将军"))).collect();
        let docs: Vec<(u64, &str)> = docs.iter().map(|(i, s)| (*i, s.as_str())).collect();
        add_docs(&sender, &docs).await;
        let mut expected: Vec<i64> = (1..=12).collect();
        expected.sort_by_key(|id| token_of(*id as u64));

        let (five, cursor) = search_ordered(&sender, "将军", 5, SearchWindow::default()).await;
        assert_eq!(five, expected[..5]);
        assert!(cursor.is_some());
        let (ten, _) = search_ordered(&sender, "将军", 10, SearchWindow::default()).await;
        assert_eq!(ten, expected[..10]);
        assert_eq!(every_page(&sender, "将军", 5, None).await, expected);
        // Asked for descending, it is the exact reverse.
        let mut reversed = expected.clone();
        reversed.reverse();
        assert_eq!(
            every_page(&sender, "将军", 5, Some(SortOrder::Desc)).await,
            reversed
        );
    }

    /// Rows of one partition come in clustering order, honouring a `DESC` clustering column, and
    /// partitions in token order; a page boundary inside a partition resumes exactly.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn rows_of_a_partition_follow_the_clustering_order() {
        // id = 10 * partition + clustering, three partitions of four rows.
        let key_of = |id: u64| {
            PrimaryKey::from(vec![
                CqlValue::BigInt((id / 10) as i64),
                CqlValue::Int((id % 10) as i32),
            ])
        };
        let layout = KeyLayout {
            partition_key_count: 1,
            clustering_descending: Arc::new([true]),
        };
        let sender = make_sender_for(
            IndexOptionsPattern::default(),
            make_table(key_of, |_, _| None),
            layout.clone(),
        );
        let ids: Vec<u64> = (1..=3)
            .flat_map(|p| (1..=4).map(move |c| 10 * p + c))
            .collect();
        for id in &ids {
            add_doc(&sender, *id, &format!("user{id}将军")).await;
        }
        let token = |id: u64| layout.token_key(&key_of(id)).unwrap();
        let mut expected = ids.clone();
        // By token, then clustering descending.
        expected.sort_by_key(|id| (token(*id), Reverse(id % 10)));
        let expected: Vec<PrimaryKey> = expected.into_iter().map(key_of).collect();

        for page in [1, 3, 5] {
            let mut seen = Vec::new();
            let mut window = SearchWindow::default();
            loop {
                let found = sender
                    .search(
                        make_index_key(),
                        "将军".into(),
                        MatchKind::Contains,
                        limit(page),
                        0,
                        window,
                        true,
                    )
                    .await
                    .unwrap();
                seen.extend(found.primary_keys);
                match found.next_cursor {
                    Some(cursor) => {
                        window = SearchWindow {
                            cursor: Some(cursor),
                            ..SearchWindow::default()
                        }
                    }
                    None => break,
                }
            }
            assert_eq!(seen, expected, "page {page}");
        }
    }

    /// Several ordering columns with their own directions: the first decides, the second breaks
    /// its ties, and table order breaks theirs. Naming the first column's other direction walks
    /// the exact reverse.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn several_ordering_columns_order_with_their_own_directions() {
        // a = id % 3 (few values, many ties), b = id % 5.
        let value_of = |id: u64, column: &str| match column {
            "a" => Some(CqlValue::Int((id % 3) as i32)),
            "b" => Some(CqlValue::Int((id % 5) as i32)),
            _ => None,
        };
        let sender = make_sender_for(
            IndexOptionsPattern {
                order_by: "a DESC, b ASC".parse().unwrap(),
                ..IndexOptionsPattern::default()
            },
            make_table(bigint_key, value_of),
            test_layout(),
        );
        let docs: Vec<(u64, String)> = (1..=30).map(|i| (i, format!("user{i}将军"))).collect();
        let docs: Vec<(u64, &str)> = docs.iter().map(|(i, s)| (*i, s.as_str())).collect();
        add_docs(&sender, &docs).await;
        let mut expected: Vec<i64> = (1..=30).collect();
        expected.sort_by_key(|id| (Reverse(id % 3), id % 5, token_of(*id as u64)));

        assert_eq!(
            every_page(&sender, "将军", 4, Some(SortOrder::Desc)).await,
            expected
        );
        assert_eq!(every_page(&sender, "将军", 7, None).await, expected);
        expected.reverse();
        assert_eq!(
            every_page(&sender, "将军", 4, Some(SortOrder::Asc)).await,
            expected
        );
    }

    /// A row whose ordering column is null or empty is not in an ordered index -- not in the
    /// order, and not in a range below every value, where a null used to land as sort key 0 --
    /// and a later value brings it in.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn a_row_without_an_ordering_value_is_left_out() {
        // Ids up to 3 have no value (1 null, 2 empty); the rest sort by their id.
        let value_of = |id: u64, _: &str| match id {
            1 => None,
            2 | 3 => Some(CqlValue::Empty),
            _ => Some(CqlValue::BigInt(id as i64)),
        };
        let sender = make_sender_for(
            ordered_options(),
            make_table(bigint_key, value_of),
            test_layout(),
        );
        add_docs(
            &sender,
            &[
                (1, "a将军"),
                (2, "b将军"),
                (3, "c将军"),
                (4, "d将军"),
                (5, "e将军"),
            ],
        )
        .await;

        let (ids, _) = search_ordered(&sender, "将军", 10, SearchWindow::default()).await;
        assert_eq!(ids, vec![5, 4]);
        let (ids, _) =
            search_ordered(&sender, "将军", 10, window(None, None, Some(sort_key(4)))).await;
        assert_eq!(ids, vec![4]);
        let (ids, _) = search_ordered(&sender, "将军", 10, ascending()).await;
        assert_eq!(ids, vec![4, 5]);
        assert_eq!(sender.count(make_index_key()).await.unwrap(), 2);
    }

    /// Whether a segment of a few-values index is narrow: one value, or the end of one and the
    /// start of the next, which is what a rewritten range crossing a value leaves.
    fn at_most_two_values(segment: &SegmentLayout) -> bool {
        segment.sort_max.unwrap() - segment.sort_min.unwrap() <= 1
    }

    /// An ordered index on a column with few values: rows sharing one keep table order, and the
    /// rewrite splits a run of one value by token, so segments end narrow even when every row has
    /// one of three values.
    #[rstest]
    #[timeout(Duration::from_secs(60))]
    #[tokio::test]
    async fn a_column_with_few_values_still_ends_in_narrow_segments() {
        let rows = 1200u64;
        let cap = 100u32;
        let sender = make_sender_for(
            IndexOptionsPattern {
                order_by: "rating".parse().unwrap(),
                segment_max_docs: cap.to_string().parse().unwrap(),
                rewrite_wide_segments: "true".parse().unwrap(),
                ..IndexOptionsPattern::default()
            },
            make_table(bigint_key, |id, _| Some(CqlValue::Int((id % 3) as i32))),
            test_layout(),
        );
        let mut acks = Vec::with_capacity(rows as usize);
        for id in 1..=rows {
            let (tx, rx) = mpsc::channel(1);
            sender
                .add_document(
                    test_partition_id(),
                    id.into(),
                    format!("user{id}将军"),
                    AsyncInProgress::Fullscan(tx),
                )
                .await
                .unwrap();
            acks.push(rx);
        }
        for mut ack in acks {
            ack.recv().await;
        }
        let deadline = Instant::now() + Duration::from_secs(45);
        let stats = loop {
            let stats = sender.stats(make_index_key()).await.unwrap();
            let settled = stats.rewrite.ranges_total > 0
                && stats.rewrite.ranges_done == stats.rewrite.ranges_total
                && stats
                    .segments
                    .iter()
                    .all(|s| s.docs <= cap && at_most_two_values(s));
            if settled || Instant::now() > deadline {
                break stats;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        assert_eq!(stats.tantivy.num_docs, rows);
        assert!(
            stats
                .segments
                .iter()
                .all(|s| s.docs <= cap && at_most_two_values(s)),
            "{:?}",
            stats.segments
        );
        // Pruning works again: the first page of the highest value opens a segment or two of it.
        let mut expected: Vec<i64> = (1..=rows as i64).collect();
        expected.sort_by_key(|id| (Reverse(id % 3), Reverse(token_of(*id as u64))));
        let before = sender.stats(make_index_key()).await.unwrap().walk;
        let (ids, _) = search_ordered(
            &sender,
            "将军",
            10,
            SearchWindow {
                order: Some(SortOrder::Desc),
                ..SearchWindow::default()
            },
        )
        .await;
        assert_eq!(ids, expected[..10]);
        let after = sender.stats(make_index_key()).await.unwrap().walk;
        assert!(
            after.segments_opened - before.segments_opened <= 2,
            "{before:?} -> {after:?}"
        );
    }

    /// The rewrite of a column with few values comes to an end, and leaves no segment spanning
    /// more than the end of one value and the start of the next. The endless rewrite of a local run with five values (2026-10-30) needed
    /// merges this size does not reproduce; `a_segment_across_two_values_is_as_narrow_as_its_rows`
    /// covers its cause.
    #[rstest]
    #[timeout(Duration::from_secs(60))]
    #[tokio::test]
    async fn the_rewrite_of_a_column_with_few_values_comes_to_an_end() {
        // Enough caps that a segment holding the whole of two values is wide: the narrow span is
        // eight caps' worth of the five values' range, half of one value at 80 caps.
        let rows = 4000u64;
        let cap = 50u32;
        let sender = make_sender_for(
            IndexOptionsPattern {
                order_by: "rating DESC".parse().unwrap(),
                segment_max_docs: cap.to_string().parse().unwrap(),
                rewrite_wide_segments: "true".parse().unwrap(),
                ..IndexOptionsPattern::default()
            },
            make_table(bigint_key, |id, _| Some(CqlValue::Int((id % 5) as i32))),
            test_layout(),
        );
        let mut acks = Vec::with_capacity(rows as usize);
        for id in 1..=rows {
            let (tx, rx) = mpsc::channel(1);
            sender
                .add_document(
                    test_partition_id(),
                    id.into(),
                    format!("user{id}将军"),
                    AsyncInProgress::Fullscan(tx),
                )
                .await
                .unwrap();
            acks.push(rx);
        }
        for mut ack in acks {
            ack.recv().await;
        }
        // Settled: a rewrite ran to its end, and nothing has been moved since for a while.
        let deadline = Instant::now() + Duration::from_secs(45);
        let mut quiet_since = (0, Instant::now());
        let stats = loop {
            let stats = sender.stats(make_index_key()).await.unwrap();
            let rewrite = &stats.rewrite;
            let finished = rewrite.ranges_total > 0 && rewrite.ranges_done == rewrite.ranges_total;
            if !finished || rewrite.docs_rewritten != quiet_since.0 {
                quiet_since = (rewrite.docs_rewritten, Instant::now());
            } else if quiet_since.1.elapsed() > Duration::from_secs(3) {
                break stats;
            }
            assert!(
                Instant::now() < deadline,
                "the rewrite does not settle: {:?}, {:?}",
                stats.rewrite,
                stats.segments
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        assert_eq!(stats.tantivy.num_docs, rows);
        assert!(
            stats.rewrite.docs_rewritten < 3 * rows,
            "{:?}",
            stats.rewrite
        );
        assert!(
            stats.segments.iter().all(at_most_two_values),
            "{:?}",
            stats.segments
        );
    }

    /// A segment holding the last tokens of one value and the first of the next spans what its
    /// rows do, not both values whole: narrow, so it neither counts as wide nor joins the spread
    /// segments that merge whatever they span.
    #[test]
    fn a_segment_across_two_values_is_as_narrow_as_its_rows() {
        let rows = [
            (3, u64::MAX - 40),
            (3, u64::MAX - 7),
            (4, 5),
            (4, 90),
            (3, u64::MAX - 20),
        ];
        let (low, high) = second_slot_at_the_ends(rows.into_iter(), 3, 4);
        assert_eq!((low, high), (u64::MAX - 40, 90));
        let layout = |first: u64, second: u64| (u128::from(first) << 64) | u128::from(second);
        let (min, max) = (layout(3, low), layout(4, high));
        assert_eq!(max - min, 131);
        // Among segments spanning five values of a 1M-row index capped at 20k rows a segment.
        let tenth = u64::MAX / 10;
        let segments = [
            (20_000, layout(0, 0), layout(0, tenth), id(1)),
            (
                20_000,
                layout(4, u64::MAX - tenth),
                layout(4, u64::MAX),
                id(2),
            ),
            (500, min, max, id(3)),
        ];
        let total = 1_000_000;
        assert!(wide_segments(&segments, total, 0, 20_000).is_empty());
        assert!(!spread_segments(&segments, total, 20_000).contains(&id(3)));
        // The bounds as they were: both values whole, which the rewrite plans again and again.
        let widened = [(500, layout(3, 0), layout(4, u64::MAX), id(3))];
        let segments = [segments[0], segments[1], widened[0]];
        assert!(spread_segments(&segments, total, 20_000).contains(&id(3)));
    }
}
