/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! `LIKE` over short text values, on Tantivy.
//!
//! Every short piece of a value (a gram, see `gram.rs`) is an indexed term. A pattern is answered
//! by intersecting the posting lists of the grams every matching value must hold. That admits
//! false positives (all the pieces present, but not where the pattern wants them), so unless the
//! grams prove the match, each candidate is verified against the stored value before it is
//! returned (`like.rs`).
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
//! * **Pruning depends on segment layout.** Cost is flat only where segments are disjoint in sort
//!   order. Two options keep them so: `segment_rows` installs a merge policy that keeps the
//!   segments placed in sort order (the ordered tier) apart from the pieces commits leave (the
//!   fresh tier) and only joins ordered neighbours under a cap, and `rewrite` repairs the fresh
//!   rows into the ordered tier, worst stretch first, continuously when idle and holding back
//!   while the index ingests. Without them, after a backfill out of order every segment spans
//!   everything and the search degrades to visiting every match: correct, but not fast. See
//!   `docs/dev/pattern/stage-2-ordering.md`.

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::AcqRel;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::Ordering::Release;
use std::time::Duration;
use std::time::Instant;

use anyhow::anyhow;
use async_channel::Sender;
use tantivy::DocSet;
use tantivy::SegmentReader;
use tantivy::TERMINATED;
use tantivy::TantivyDocument;
use tantivy::Term;
use tantivy::columnar::Column;
use tantivy::index::SegmentId;
use tantivy::index::SegmentMeta;
use tantivy::indexer::MergePolicy;
use tantivy::query::Scorer;
use tantivy::schema::FAST;
use tantivy::schema::INDEXED;
use tantivy::schema::IndexRecordOption;
use tantivy::schema::STORED;
use tantivy::schema::Schema;
use tantivy::schema::TextFieldIndexing;
use tantivy::schema::TextOptions;
use tantivy::schema::Value;
use tokio::sync::mpsc;
use tokio::sync::watch;
use tracing::debug;
use tracing::error;
use tracing::info;
use tracing::warn;

use crate::Config;
use crate::IndexKey;
use crate::IndexOptionsPattern;
use crate::Limit;
use crate::PrimaryKey;
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
use crate::worker;
use crate::worker::Worker;
use crate::worker::WorkerExt;

use super::actor::Cursor;
use super::actor::PatternIndex;
use super::actor::PatternLikeR;
use super::actor::PatternPage;
use super::actor::SearchWindow;
use super::actor::SortOrder;
use super::factory::PatternIndexConfiguration;
use super::factory::PatternIndexFactory;
use super::gram;
use super::like::LikeQuery;
use super::order::KeyLayout;
use super::order::PREFIX_SLOTS;
use super::order::SortPrefix;

pub(crate) struct Factory {
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
    metrics: Arc<Metrics>,
    config: watch::Receiver<Arc<Config>>,
}

/// What one search request may spend before it returns what it has settled, with a cursor the
/// next request goes on from. Without one, a keyword whose grams most rows hold costs a request
/// every such row above the page's last: time without a bound, on a worker the other searches
/// wait for, spent to the end even when the caller has given up.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct SearchBudget {
    /// Wall time, from when the search starts on its worker.
    pub(crate) time: Option<Duration>,
    /// Postings read and stored values checked. For tests, which need the cut at the same place
    /// on every run.
    pub(crate) work: Option<u64>,
}

/// The time a search request may take when the configuration does not say
/// (`VECTOR_STORE_PATTERN_SEARCH_BUDGET`): a hundredth of ScyllaDB's
/// default read timeout, so that a statement has many requests in it, and short enough for a
/// worker not to be missed.
pub(crate) const DEFAULT_SEARCH_BUDGET: Duration = Duration::from_millis(50);

/// Where a search request reads its budget as it starts: the node's configuration, which may be
/// reloaded while the index serves.
pub(crate) struct SearchBudgetSource(Box<dyn Fn() -> SearchBudget + Send + Sync>);

impl From<SearchBudget> for SearchBudgetSource {
    fn from(fixed: SearchBudget) -> Self {
        Self(Box::new(move || fixed))
    }
}

impl From<watch::Receiver<Arc<Config>>> for SearchBudgetSource {
    fn from(config: watch::Receiver<Arc<Config>>) -> Self {
        Self(Box::new(move || SearchBudget::of(&config.borrow())))
    }
}

impl SearchBudget {
    /// The budget a configuration asks for. A time of zero is no limit.
    fn of(config: &Config) -> Self {
        let time = config
            .pattern_search_budget
            .unwrap_or(DEFAULT_SEARCH_BUDGET);
        Self {
            time: (!time.is_zero()).then_some(time),
            work: None,
        }
    }
}

impl PatternIndexFactory for Factory {
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
            self.config.clone(),
        )
    }
}

pub(super) fn new_factory(
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
    metrics: Arc<Metrics>,
    config: watch::Receiver<Arc<Config>>,
) -> Factory {
    Factory {
        worker,
        memory,
        metrics,
        config,
    }
}

/// The field the shared tantivy code deletes documents by; the pattern index also reads it back.
const PRIMARY_ID_FIELD: &str = "primary_id";
const TEXT_FIELD: &str = "text";
/// The fast field holding a row's token, the last slot of its sort prefix. The ordering columns'
/// slots come before it, as `sort_<i>`.
const TOKEN_FIELD: &str = "token";
/// The fast field saying whether a repair step placed the row (1) or a commit of ingested rows
/// did (0): a segment whose every row says 1 is in the ordered tier however it was merged since.
/// Ordered indexes only.
const REPAIRED_FIELD: &str = "repaired";
const TOKENIZER_NAME: &str = "pattern_ngram";
/// Values are short, so a single cached store block per segment covers consecutive lookups.
const STORE_CACHE_BLOCKS: usize = 32;

/// What a pattern index keeps beside its Tantivy index.
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
    /// The table's primary-key layout: tokens and table order.
    key_layout: KeyLayout,
    /// What the repair of the layout has done so far: see `repair_step`.
    repair: RepairCounters,
    /// Segments in the ordered tier that no row's `REPAIRED_FIELD` says are: adopted where they
    /// were found (see `ordered_tier`), or merged from ordered segments. In memory only, as the
    /// index is; a segment a merge retired is dropped from it by `prune_columns`.
    adopted: Mutex<HashSet<SegmentId>>,
    /// Set when a row of a full scan is added, and taken by the next tick: the repair waits for
    /// the scan to end (see `maybe_repair`).
    scan_rows: AtomicBool,
    /// Ticks since one last took `scan_rows` set.
    ticks_since_scan: AtomicU32,
    /// The lag of the CDC changes taken in since the last tick, summed in milliseconds, and how
    /// many: see `maybe_repair`.
    cdc_lag_millis: AtomicU64,
    cdc_changes: AtomicU64,
    /// Ticks since one last found the CDC changes lagging; back at `LAG_QUIET_TICKS` on a tick
    /// whose changes kept up.
    ticks_since_lag: AtomicU32,
    /// Set while a tick's work on this index -- learning the layout, a step of the rewrite -- is
    /// running, so that the next tick does not queue more behind it. See `TickWork`.
    tick_running: AtomicBool,
    /// The merges `after_reload` asked for that are still running; their segments are left out
    /// of the next ones it asks for, and the merges of ordered segments say what they produced.
    merging: Mutex<MergesInFlight<tantivy::FutureResult<Option<SegmentMeta>>>>,
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

/// Fresh-tier rows sampled to place a repair step's range.
const REPAIR_SAMPLE_TARGET: u64 = 200_000;
/// While an index ingests, the fresh rows it holds before a repair step competes with the writes
/// for the node. Under that, the fresh tier waits for an idle tick. In rows rather than caps, so
/// that a larger cap does not let more rows wait unrepaired.
const LOAD_FRESH_ROWS: u64 = 400_000;
/// The most rows a repair step moves under one hold of the writer lock: a step is carried out in
/// key ranges of at most this many, each committed on its own, so that the writes waiting for the
/// lock wait for one range at a time whatever the cap. At a 1M cap a split moved up to two caps
/// of rows in one hold.
const REPAIR_LOCK_ROWS: usize = 50_000;
/// How long a repair step lets the writes that waited through one of its holds go on before it
/// takes the next: until none waits, or this long under a stream of them.
const REPAIR_HANDOVER_LIMIT: Duration = Duration::from_millis(100);
/// Ticks without a row of a full scan before the scan counts as ended.
const SCAN_QUIET_TICKS: u32 = 5;
/// Ticks without a CDC change before a stream of lagging ones counts as ended. A stream that
/// lags comes in bursts with gaps between them, and one tick without a change is not its end.
const LAG_QUIET_TICKS: u32 = 5;
/// The mean lag of the CDC changes taken in since the last tick -- from the write to the table to
/// the index taking the change in -- under which a repair step may run while the index ingests.
/// An index keeping up lags about a second (AWS run 5df052ac, 1.3 s with a writer at 2000 rows a
/// second), one falling behind tens of seconds.
const REPAIR_LAG_LIMIT: Duration = Duration::from_secs(3);

/// How full a merge of ordered neighbours may make a segment: three quarters of the cap, so that
/// fresh rows repaired into its stretch later fit beside it, and merge into it, before it has to
/// be split.
fn ordered_fill(cap: u32) -> u64 {
    u64::from(cap - cap / 4)
}

/// How many rows each part of a split holds at most: half the cap, the other half being the
/// room it leaves for fresh rows and for its neighbour.
fn split_rows(cap: u32) -> usize {
    (cap / 2).max(1) as usize
}

/// How many rows a key range of a repair step holds at most: a part of a split, and never more
/// than `REPAIR_LOCK_ROWS`. The ranges are disjoint, so ordered merges fold small ones back
/// together.
fn repair_range_rows(cap: u32) -> usize {
    if walk_off(OFF_LOCK_CHUNKS) {
        return split_rows(cap);
    }
    split_rows(cap).min(REPAIR_LOCK_ROWS)
}

/// A live segment as the layout policy sees it: its live and deleted rows, its layout-key bounds
/// (inclusive; deleted rows count, as they do in the columns' own bounds), and whether it is
/// known to be in the ordered tier.
///
/// The ordered tier holds the segments that are where they belong in sort order: disjoint from
/// one another, but for the pieces of one repair step, which a merge folds together. The fresh
/// tier is everything else -- the pieces commits leave, each spanning as much of the range as the
/// rows it took in -- and a repair step moves its rows into the ordered tier range by range. A
/// fresh row never reaches an ordered segment by a merge: a merge takes segments of one tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Piece {
    docs: u32,
    deleted: u32,
    min: u128,
    max: u128,
    id: SegmentId,
    /// Placed by a repair step (every row carries `REPAIRED_FIELD`), or adopted: see
    /// `ordered_tier`.
    ordered: bool,
}

/// Which of `pieces` are in the ordered tier: the ones known to be, and the ones that overlap no
/// other piece. Those are where they belong whoever placed them -- the pieces rows inserted in
/// sort order leave once merged, a build by full scan in table order -- and the caller adopts
/// them: once in the ordered tier, a segment stays there, as fresh pieces arriving over its
/// stretch later are rows to repair into it, not a reason to rewrite it. Empty pieces are in
/// neither tier; Tantivy drops them at the next commit.
fn ordered_tier(pieces: &[Piece]) -> Vec<bool> {
    let mut order: Vec<usize> = (0..pieces.len()).filter(|&i| pieces[i].docs > 0).collect();
    order.sort_by_key(|&i| (pieces[i].min, pieces[i].max));
    let mut ordered = vec![false; pieces.len()];
    let mut reach: Option<u128> = None;
    for (k, &i) in order.iter().enumerate() {
        let piece = &pieces[i];
        let left = reach.is_some_and(|reach| reach >= piece.min);
        let right = order
            .get(k + 1)
            .is_some_and(|&next| pieces[next].min <= piece.max);
        ordered[i] = piece.ordered || (!left && !right);
        reach = Some(reach.map_or(piece.max, |reach| reach.max(piece.max)));
    }
    ordered
}

/// Ordered pieces that overlap, transitively: the pieces of one repair step before the merge
/// that folds them, or a segment with the rows later repaired into its stretch. A merge takes a
/// cluster whole or not at all -- merging part of one would leave a segment overlapping the rest
/// for good -- and a cluster past the cap is split by a repair step.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Cluster {
    min: u128,
    max: u128,
    docs: u64,
    deleted: u64,
    ids: Vec<SegmentId>,
}

/// The clusters of the ordered pieces, ascending and disjoint.
fn clusters(pieces: &[Piece], ordered: &[bool]) -> Vec<Cluster> {
    let mut members: Vec<&Piece> = pieces
        .iter()
        .zip(ordered)
        .filter(|&(_, &ordered)| ordered)
        .map(|(piece, _)| piece)
        .collect();
    members.sort_by_key(|piece| (piece.min, piece.max));
    let mut clusters: Vec<Cluster> = Vec::new();
    for piece in members {
        match clusters.last_mut() {
            Some(cluster) if piece.min <= cluster.max => {
                cluster.max = cluster.max.max(piece.max);
                cluster.docs += u64::from(piece.docs);
                cluster.deleted += u64::from(piece.deleted);
                cluster.ids.push(piece.id);
            }
            _ => clusters.push(Cluster {
                min: piece.min,
                max: piece.max,
                docs: u64::from(piece.docs),
                deleted: u64::from(piece.deleted),
                ids: vec![piece.id],
            }),
        }
    }
    clusters
}

/// What one tier's run is worth merging for: two segments or more, or one whose deleted rows
/// outnumber its live ones, which a merge of it alone drops.
fn worth_merging(segments: usize, docs: u64, deleted: u64) -> bool {
    segments >= 2 || (segments == 1 && deleted >= docs.max(1))
}

/// The merges worth asking for, each a run of one tier's segments. Pure, so it can be tested
/// without a writer.
///
/// - Ordered: runs of neighbouring clusters, each taken whole, while the run stays within
///   `ordered_fill`; a cluster alone may reach the cap, which is what lets the rows a repair step
///   adds to a segment's stretch merge into it. Neighbours have no ordered segment between them,
///   so the merged segment overlaps none, and no fresh piece reaching between them: the merged
///   segment would span the fresh rows there, and every repair step among them would split it
///   again. A cluster past the cap is left to the repair.
/// - Fresh: runs of neighbours by lowest key under the cap, whatever they span, a full segment
///   left out rather than ending a run. Their rows go to the ordered tier by repair anyway; until
///   then fewer segments are fewer for a search to open, and rows inserted in sort order merge
///   into contiguous segments the tier adopts as they are.
fn merge_runs(pieces: &[Piece], cap: u32) -> Vec<(Vec<SegmentId>, bool)> {
    let ordered = ordered_tier(pieces);
    let cap = u64::from(cap);
    let fill = ordered_fill(cap as u32);
    let mut runs = Vec::new();

    let mut run: Vec<SegmentId> = Vec::new();
    let (mut run_docs, mut run_deleted) = (0u64, 0u64);
    let mut flush = |run: &mut Vec<SegmentId>, docs: &mut u64, deleted: &mut u64, tier: bool| {
        if worth_merging(run.len(), *docs, *deleted) {
            runs.push((std::mem::take(run), tier));
        } else {
            run.clear();
        }
        (*docs, *deleted) = (0, 0);
    };
    // Where the last cluster of the run ends: a fresh piece reaching between it and the next
    // may hold rows there, and a merge of the two would span them.
    let mut run_max = 0u128;
    let fresh_between = |low: u128, high: u128| {
        pieces.iter().zip(&ordered).any(|(piece, &ordered)| {
            !ordered && piece.docs > 0 && piece.max > low && piece.min < high
        })
    };
    for cluster in clusters(pieces, &ordered) {
        if cluster.docs > cap {
            flush(&mut run, &mut run_docs, &mut run_deleted, true);
            continue;
        }
        if !run.is_empty()
            && (run_docs + cluster.docs > fill
                || (!walk_off(OFF_FRESH_BETWEEN) && fresh_between(run_max, cluster.min)))
        {
            flush(&mut run, &mut run_docs, &mut run_deleted, true);
        }
        run_max = cluster.max;
        run.extend(cluster.ids);
        run_docs += cluster.docs;
        run_deleted += cluster.deleted;
    }
    flush(&mut run, &mut run_docs, &mut run_deleted, true);

    let mut fresh: Vec<&Piece> = pieces
        .iter()
        .zip(&ordered)
        .filter(|&(piece, &ordered)| !ordered && piece.docs > 0)
        .map(|(piece, _)| piece)
        .collect();
    fresh.sort_by_key(|piece| (piece.min, piece.max));
    let mut alone = Vec::new();
    for piece in fresh {
        let docs = u64::from(piece.docs);
        if docs >= cap {
            // Full already, with nothing to merge with. Unlike an ordered one it does not keep
            // its neighbours apart: fresh segments overlap whatever lies between them anyway.
            if worth_merging(1, docs, u64::from(piece.deleted)) {
                alone.push((vec![piece.id], false));
            }
            continue;
        }
        if run_docs + docs > cap {
            flush(&mut run, &mut run_docs, &mut run_deleted, false);
        }
        run.push(piece.id);
        run_docs += docs;
        run_deleted += u64::from(piece.deleted);
    }
    flush(&mut run, &mut run_docs, &mut run_deleted, false);
    runs.extend(alone);
    runs
}

/// A repair step's work: the fresh rows with layout keys in `low..=high`, and with them the rows
/// of `cluster` -- the ordered segments of the stretch the range is in, if any -- when those would
/// not fit beside the cluster under the cap.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RepairRange {
    low: u128,
    high: u128,
    cluster: Option<Cluster>,
}

/// The next repair step, worst first, or None when there is nothing to repair.
///
/// The ordered tier's clusters cut the key space into stretches: a cluster's own, and the gaps
/// between clusters. A stretch is as bad as the fresh rows in it -- every search through it opens
/// the fresh segments as well as the ordered one -- estimated from `fresh_keys`, a sorted sample of
/// the fresh tier's keys each standing for `weight` rows; a cluster past the cap counts its own
/// rows too, as it overlaps itself until split. The worst stretch is repaired, at most a cap of its fresh rows
/// at a time, lowest keys first. Pure, so it can be tested without an index.
fn plan_repair(
    pieces: &[Piece],
    fresh_keys: &[u128],
    weight: u64,
    cap: u32,
) -> Option<RepairRange> {
    let ordered = ordered_tier(pieces);
    let clusters = clusters(pieces, &ordered);
    let weight = weight.max(1);
    // Stretch 2j is the gap below cluster j (the last, j = clusters.len(), the one above them
    // all); 2j + 1 is cluster j. The sample is sorted, so each stretch's keys are one run of it.
    let stretch_of = |key: u128| {
        let j = clusters.partition_point(|cluster| cluster.max < key);
        if clusters.get(j).is_some_and(|cluster| cluster.min <= key) {
            2 * j + 1
        } else {
            2 * j
        }
    };
    let mut keys = vec![(0usize, 0usize); 2 * clusters.len() + 1];
    let mut from = 0;
    while from < fresh_keys.len() {
        let stretch = stretch_of(fresh_keys[from]);
        let to = from + fresh_keys[from..].partition_point(|&key| stretch_of(key) == stretch);
        keys[stretch] = (from, to);
        from = to;
    }
    let score = |stretch: usize| {
        let (from, to) = keys[stretch];
        let fresh = (to - from) as u64 * weight;
        let overfull = (stretch % 2 == 1)
            .then(|| &clusters[stretch / 2])
            .filter(|cluster| cluster.docs > u64::from(cap))
            .map_or(0, |cluster| cluster.docs);
        fresh + overfull
    };
    // The first of the worst, so that equal stretches go in key order.
    let worst = (0..keys.len())
        .rev()
        .max_by_key(|&stretch| score(stretch))?;
    if score(worst) == 0 {
        return None;
    }
    let j = worst / 2;
    let (low, mut high, cluster) = if worst % 2 == 1 {
        let cluster = clusters[j].clone();
        (cluster.min, cluster.max, Some(cluster))
    } else {
        (
            j.checked_sub(1).map_or(0, |below| clusters[below].max + 1),
            clusters.get(j).map_or(u128::MAX, |above| above.min - 1),
            None,
        )
    };
    let (from, to) = keys[worst];
    let step = (u64::from(cap) / weight).max(1) as usize;
    if to - from > step {
        // The lowest cap's worth: the range starts where the stretch does, so nothing below the
        // cut is left behind in it.
        high = fresh_keys[from + step - 1];
    }
    Some(RepairRange { low, high, cluster })
}

/// The merges `after_reload` has asked the writer for and that have not finished, with the
/// segments each one reads and the tier it merges. Tantivy keeps a merge's segments out of the
/// merges its own policy picks, but not out of an explicit `IndexWriter::merge`: asked again for
/// segments already being merged, it queues another full merge of them. `after_reload` asks on
/// every tick, so a merge that outlasted a tick was queued again on the next one, and the
/// duplicates slowed every merge down until each was queued many times over. On AWS (run
/// a0e51aa9, a 10M-row table-order index) the same merge was started 60 times a minute, 24,832
/// merges were started against 1,703 finished, the queued ones held their segments in memory, and
/// they went on running after the index was dropped, until the node ran out of memory.
///
/// Generic over the merge's handle so that a test can stand in for a merge it holds back.
struct MergesInFlight<F> {
    merges: Vec<(Vec<SegmentId>, bool, F)>,
    /// Merges asked for, and merges seen to have ended, since the index was created.
    started: u64,
    ended: u64,
}

impl<F> Default for MergesInFlight<F> {
    fn default() -> Self {
        Self {
            merges: Vec::new(),
            started: 0,
            ended: 0,
        }
    }
}

/// The segment a finished merge produced, if it produced one.
trait MergeOutput {
    fn segment(&self) -> Option<SegmentId>;
}

impl MergeOutput for tantivy::Result<Option<SegmentMeta>> {
    fn segment(&self) -> Option<SegmentId> {
        self.as_ref().ok()?.as_ref().map(SegmentMeta::id)
    }
}

impl<F> MergesInFlight<F>
where
    F: std::future::Future + Unpin,
    F::Output: MergeOutput,
{
    /// The segments of the merges still running. Forgets the ones that ended, whether they
    /// succeeded or not, so that the segments of a failed merge can be asked for again, and adds
    /// what a merge of ordered segments produced to `adopted`: a merged segment is a new one,
    /// and nothing else says which tier it is in.
    fn busy(&mut self, adopted: &mut HashSet<SegmentId>) -> HashSet<SegmentId> {
        let before = self.merges.len();
        self.merges.retain_mut(|(_, ordered, merge)| {
            match futures::FutureExt::now_or_never(&mut *merge) {
                None => true,
                Some(output) => {
                    if *ordered && let Some(segment) = output.segment() {
                        adopted.insert(segment);
                    }
                    false
                }
            }
        });
        self.ended += (before - self.merges.len()) as u64;
        self.merges
            .iter()
            .flat_map(|(segments, _, _)| segments.iter().copied())
            .collect()
    }

    /// Whether a merge still running has replaced its segments in `live` already: its output is
    /// there, and not yet known to be ordered if it is, so the layout cannot be judged until it
    /// reports its end.
    fn unsettled(&self, live: &HashSet<SegmentId>) -> bool {
        self.merges
            .iter()
            .any(|(segments, _, _)| segments.iter().any(|id| !live.contains(id)))
    }

    fn started(&mut self, segments: Vec<SegmentId>, ordered: bool, merge: F) {
        self.merges.push((segments, ordered, merge));
        self.started += 1;
    }

    /// As of the last tick: a merge that ended since is still counted as running.
    fn totals(&self) -> MergeTotals {
        MergeTotals {
            started: self.started,
            ended: self.ended,
            running: self.merges.len() as u64,
        }
    }
}

/// The merges a pattern index's ticks asked for, as `Stats` reports them. Only these: the merges
/// Tantivy's own policy starts after a commit are not counted.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MergeTotals {
    pub(crate) started: u64,
    pub(crate) ended: u64,
    pub(crate) running: u64,
}

#[derive(Debug, Default)]
struct RepairCounters {
    steps: AtomicU64,
    docs_rewritten: AtomicU64,
    /// Whether the last tick found rows to repair.
    pending: AtomicBool,
}

/// [`RepairCounters`] at one moment, as `Stats` reports them.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RewriteTotals {
    /// Repair steps since the index was created.
    pub(crate) steps: u64,
    /// Rows those steps moved.
    pub(crate) docs_rewritten: u64,
    /// Whether the last tick found fresh rows, or an ordered stretch past the cap, to repair. An
    /// index has settled once it has not and no merge runs.
    pub(crate) pending: bool,
}

/// The most runs a segment may be cut into for the walk to take it run by run: what they take
/// in memory, a few dozen bytes each, is bounded by it.
const MAX_RUNS: usize = 65_536;

/// The doc ids of one block of a coarse run.
const BLOCK_DOCS: u32 = 256;

/// The fewest doc ids over which the layout key never decreases for the walk to take them as a
/// sorted run, a row at a time from either end: a block's worth. Shorter ones are taken in
/// blocks (see [`Run`]), which are as many to look at and coarser to read.
const SORTED_RUN_DOCS: u32 = BLOCK_DOCS;

/// A segment with no more matches than this, and this many pages of them, is read whole, runs or
/// not: a block's worth of matches costs about what a pass of the postings does.
const WHOLE_READ_MATCHES: usize = BLOCK_DOCS as usize;
const WHOLE_READ_PAGES: usize = 4;
/// And this many more for each run the walk would read together (see [`Runs::overlap`]): rows
/// that arrived in no order leave hundreds of runs lying among one another, a look at each of
/// which costs about what eight matches read in a pass do.
const WHOLE_READ_PER_RUN: usize = 8;

/// A stretch of a segment's doc ids in the order of the layout key, for the walk to read from
/// the end a page needs and no further than the page needs. Of two kinds:
///
/// - sorted: the key never decreases from a row to the next, or never increases. What a
///   repair step writes, what an ordered merge of such pieces keeps, and what each of the
///   writer's threads makes of rows that arrive in the index's order.
/// - coarse: the rows between sorted runs, in blocks of `BLOCK_DOCS`, each with two bounds: the
///   lowest key from it to the end of the run, and the highest from the start of the run to
///   it (start and end by position: the run is reversed when its keys go down on the whole).
///   Neither ever decreases from a block to the next, so the walk reads the blocks from
///   either end as it does the rows of a sorted run, a block whole or not at all, and stops
///   where the bound says the rest cannot enter the page. Rows that arrive in the index's
///   order in commits too small for a thread's share to make a sorted run come out in
///   blocks whose bounds follow one another closely, and a page reads the few at the end it
///   is read from.
///
/// Rows in no order at all make a coarse run whose every block's bounds are about the run's
/// own: the walk reads all of it, as it would the segment whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Run {
    start: u32,
    end: u32,
    /// The lowest and the highest key of its rows.
    min: u128,
    max: u128,
    /// For a coarse run, where its blocks start among the segment's.
    blocks: Option<u32>,
    /// Whether the key goes down as the doc ids go up: rows that arrived in ascending order
    /// in an index declared descending, whose keys are the complement of the values. The walk
    /// counts a run's rows from the end its keys are lowest at, its *positions*, so a reversed
    /// run reads like any other, from the other end of its doc ids.
    reversed: bool,
}

impl Run {
    fn len(&self) -> u32 {
        self.end - self.start
    }

    /// The doc id of the row at `position`.
    fn doc_at(&self, position: u32) -> u32 {
        if self.reversed {
            self.end - 1 - position
        } else {
            self.start + position
        }
    }

    /// The doc ids of the rows at positions `from..to`.
    fn docs(&self, from: u32, to: u32) -> (u32, u32) {
        if self.reversed {
            (self.end - to, self.end - from)
        } else {
            (self.start + from, self.start + to)
        }
    }

    /// The positions of the rows with doc ids `from..to`.
    fn positions(&self, from: u32, to: u32) -> (u32, u32) {
        if self.reversed {
            (self.end - to, self.end - from)
        } else {
            (from - self.start, to - self.start)
        }
    }

    /// The block of a coarse run that holds `doc_id`, counted by position.
    fn block_of(&self, doc_id: u32) -> u32 {
        self.positions(doc_id, doc_id + 1).0 / BLOCK_DOCS
    }

    /// The lowest doc id at or over `doc_id` at which a block starts or ends; `doc_id` for a
    /// sorted run, which is read by the row.
    fn edge_up(&self, doc_id: u32) -> u32 {
        match (self.blocks, self.reversed) {
            (None, _) => doc_id,
            (Some(_), false) => {
                (self.start + (doc_id - self.start).div_ceil(BLOCK_DOCS) * BLOCK_DOCS).min(self.end)
            }
            (Some(_), true) => self.end - (self.end - doc_id) / BLOCK_DOCS * BLOCK_DOCS,
        }
    }
}

/// Doc ids `from..to` of a run for a pass of the walk to read.
#[derive(Debug, Clone, Copy)]
struct Region {
    from: u32,
    to: u32,
    /// Which of the segment's runs; None for a segment read whole.
    run: Option<u32>,
}

/// The runs of a segment and the bounds of the blocks of its coarse ones.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Runs {
    runs: Vec<Run>,
    /// Per block: the lowest key from it on, and the highest up to it, in its run.
    blocks: Vec<(u128, u128)>,
    /// Per block: the lowest and the highest key of its own rows. A block the walk comes to
    /// that cannot enter the page by these is not read: of rows in no order, most blocks hold
    /// none of the highest few.
    own: Vec<(u128, u128)>,
}

impl Runs {
    /// How many runs lie among the rows of the run at either end of the order, the more of the
    /// two: what a walk from that end reads a stretch of in its first pass. One where the runs
    /// follow one another, every run where each holds a commit of rows in no order.
    fn overlap(&self) -> usize {
        let top = self.runs.iter().max_by_key(|run| run.max);
        let bottom = self.runs.iter().min_by_key(|run| run.min);
        let among_top = top.map_or(0, |top| {
            self.runs.iter().filter(|run| run.max >= top.min).count()
        });
        let among_bottom = bottom.map_or(0, |bottom| {
            self.runs.iter().filter(|run| run.min <= bottom.max).count()
        });
        among_top.max(among_bottom)
    }
}

/// The runs of the rows with layout keys `keys`, by doc id: the stretches where the key never
/// decreases, or never increases, as sorted runs when at least `SORTED_RUN_DOCS` long, and what
/// lies between them as coarse runs. None for no rows, or past `MAX_RUNS`.
fn layout_runs(keys: &[u128]) -> Option<Runs> {
    let mut found = Runs::default();
    // Makes `from..to`, rows of sorted stretches too short to walk, a coarse run.
    let coarse = |found: &mut Runs, from: usize, to: usize| {
        if from == to {
            return;
        }
        let mut run = Run {
            start: from as u32,
            end: to as u32,
            min: 0,
            max: 0,
            blocks: Some(found.blocks.len() as u32),
            reversed: keys[to - 1] < keys[from],
        };
        let first = found.blocks.len();
        found
            .blocks
            .extend((0..run.len()).step_by(BLOCK_DOCS as usize).map(|position| {
                let (from, to) = run.docs(position, run.len().min(position + BLOCK_DOCS));
                let rows = &keys[from as usize..to as usize];
                (
                    rows.iter().copied().min().unwrap(),
                    rows.iter().copied().max().unwrap(),
                )
            }));
        found.own.extend_from_slice(&found.blocks[first..]);
        let blocks = &mut found.blocks[first..];
        // The bounds the walk reads by: no row from a block on is under its first, no row up
        // to a block is over its second.
        for block in 1..blocks.len() {
            blocks[block].1 = blocks[block].1.max(blocks[block - 1].1);
        }
        for block in (1..blocks.len()).rev() {
            blocks[block - 1].0 = blocks[block - 1].0.min(blocks[block].0);
        }
        run.min = blocks[0].0;
        run.max = blocks[blocks.len() - 1].1;
        found.runs.push(run);
    };
    // Where the rows not yet in a run start, the sorted stretch being read, and which way its
    // keys go once two of them differ.
    let mut pending = 0;
    let mut start = 0;
    let mut direction = std::cmp::Ordering::Equal;
    for doc_id in 1..=keys.len() {
        if doc_id < keys.len() {
            let step = keys[doc_id].cmp(&keys[doc_id - 1]);
            if step.is_eq() || step == direction {
                continue;
            }
            if direction.is_eq() {
                direction = step;
                continue;
            }
        }
        if doc_id - start >= SORTED_RUN_DOCS as usize {
            coarse(&mut found, pending, start);
            let (first, last) = (keys[start], keys[doc_id - 1]);
            found.runs.push(Run {
                start: start as u32,
                end: doc_id as u32,
                min: first.min(last),
                max: first.max(last),
                blocks: None,
                reversed: direction.is_lt(),
            });
            pending = doc_id;
        }
        start = doc_id;
        direction = std::cmp::Ordering::Equal;
        if found.runs.len() > MAX_RUNS {
            return None;
        }
    }
    coarse(&mut found, pending, keys.len());
    (!found.runs.is_empty() && found.runs.len() <= MAX_RUNS).then_some(found)
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
    /// Whether a repair step placed every row: see `REPAIRED_FIELD`.
    repaired: bool,
    /// The segment's doc ids cut into runs in the order of the layout key (see [`Run`]), so
    /// that the walk can start a run at the end its page needs and stop once the run can no
    /// longer enter it, rather than read every posting of the segment. None for a segment of
    /// too many, which the walk reads whole.
    runs: Option<Arc<Runs>>,
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
        // A segment from before the field existed has no such column: ingested, then.
        let repaired = segment
            .fast_fields()
            .u64(REPAIRED_FIELD)
            .is_ok_and(|column| segment.max_doc() > 0 && column.min_value() == 1);
        let mut columns = SegmentColumns {
            slots,
            layout_min,
            layout_max,
            primary_id,
            repaired,
            runs: None,
        };
        // Deleted rows count: their keys sit where they sat.
        let keys: Vec<u128> = (0..segment.max_doc())
            .map(|doc_id| columns.layout_key(doc_id))
            .collect();
        columns.runs = layout_runs(&keys).map(Arc::new);
        // What a page costs in a segment follows from its runs: a few long sorted ones where a
        // repair wrote it, one coarse one where rows arrived in order and were adopted as they
        // lay.
        if tracing::enabled!(tracing::Level::DEBUG)
            && segment.max_doc() >= 20_000
            && let Some(runs) = &columns.runs
        {
            let mut sorted: Vec<u32> = Vec::new();
            let mut coarse: Vec<u32> = Vec::new();
            for run in &runs.runs {
                if run.blocks.is_some() {
                    coarse.push(run.end - run.start);
                } else {
                    sorted.push(run.end - run.start);
                }
            }
            sorted.sort_unstable();
            coarse.sort_unstable();
            let shape = |lengths: &[u32]| {
                format!(
                    "{} of {} rows (shortest {}, median {}, longest {})",
                    lengths.len(),
                    lengths.iter().sum::<u32>(),
                    lengths.first().copied().unwrap_or(0),
                    lengths.get(lengths.len() / 2).copied().unwrap_or(0),
                    lengths.last().copied().unwrap_or(0)
                )
            };
            debug!(
                "pattern: segment of {} rows, repaired {}, keys {:x}..{:x}: sorted runs {}, coarse {}",
                segment.max_doc(),
                columns.repaired,
                columns.layout_min >> 64,
                columns.layout_max >> 64,
                shape(&sorted),
                shape(&coarse)
            );
        }
        #[cfg(test)]
        if RECORD_RUN_SHAPES.load(Relaxed) {
            // (sorted runs, coarse runs, the rows in the coarse ones).
            let mut shape = (0usize, 0usize, 0u32);
            for run in columns.runs.iter().flat_map(|runs| runs.runs.iter()) {
                if run.blocks.is_some() {
                    shape.1 += 1;
                    shape.2 += run.end - run.start;
                } else {
                    shape.0 += 1;
                }
            }
            RUN_SHAPES
                .lock()
                .unwrap()
                .insert((segment.max_doc(), columns.layout_min), shape);
        }
        *opened += 1;
        self.columns.write().unwrap().insert(id, columns.clone());
        Ok(columns)
    }

    /// Every live segment as a [`Piece`], from the cached columns; a segment whose sort columns
    /// cannot be opened is left out, with a warning.
    fn geometry(&self, searcher: &tantivy::Searcher) -> Vec<Piece> {
        let mut opened = 0;
        // A copy: `columns_for` takes the column cache's lock, which `prune_columns` holds while
        // it would take this one.
        let adopted = self.adopted.lock().unwrap().clone();
        searcher
            .segment_readers()
            .iter()
            .filter_map(|segment| match self.columns_for(segment, &mut opened) {
                Ok(columns) => Some(Piece {
                    docs: segment.num_docs(),
                    deleted: segment.num_deleted_docs(),
                    min: columns.layout_min,
                    max: columns.layout_max,
                    id: segment.segment_id(),
                    ordered: columns.repaired || adopted.contains(&segment.segment_id()),
                }),
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

    /// Drops cache entries, and adoptions, for segments a merge has retired, once they
    /// outnumber the live ones.
    fn prune_columns(&self, live: &[SegmentReader]) {
        let keep =
            || -> HashSet<SegmentId> { live.iter().map(|segment| segment.segment_id()).collect() };
        {
            let mut cache = self.columns.write().unwrap();
            if cache.len() > 2 * live.len().max(1) {
                let keep = keep();
                cache.retain(|id, _| keep.contains(id));
            }
        }
        let mut adopted = self.adopted.lock().unwrap();
        if adopted.len() > 2 * live.len().max(1) {
            let keep = keep();
            adopted.retain(|id| keep.contains(id));
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
    segments_wasted: AtomicU64,
    cut_short: AtomicU64,
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
    /// Segments opened that put no row on the page.
    segments_wasted: u64,
    /// 1 when the search's budget ended it before its page was settled.
    cut_short: u64,
}

/// One search's budget as its walk spends it.
struct Spending {
    budget: SearchBudget,
    deadline: Option<Instant>,
    work: Option<u64>,
    /// The oriented layout key of the request's cursor: the walk stops only under it, so that
    /// the cursor it leaves is past the one it came with and a row of requests gets somewhere.
    /// Rows sharing a layout key are read by one request, however many.
    progress: Option<u128>,
}

impl Spending {
    /// One of `parts` equal parts of the budget, from now.
    fn new(budget: SearchBudget, window: &SortWindow, parts: u32) -> Self {
        Self {
            budget,
            deadline: budget.time.map(|time| Instant::now() + time / parts),
            work: budget.work.map(|work| work / u64::from(parts)),
            progress: window
                .cursor
                .as_ref()
                .map(|(cursor, _)| cursor.layout_key()),
        }
    }

    /// One of `parts` equal shares of half the budget, from now: for a segment read after the
    /// budget ended the walk in another, see `collect_matches_ordered`.
    fn share(&self, parts: usize, tally: &WalkTally) -> Self {
        let parts = 2 * parts.max(1);
        Self {
            budget: self.budget,
            deadline: self
                .budget
                .time
                .map(|time| Instant::now() + time / parts as u32),
            work: self
                .budget
                .work
                .map(|work| tally.postings_scanned + tally.store_reads + work / parts as u64),
            progress: self.progress,
        }
    }

    /// Whether the walk stops here, with `frontier` the highest oriented layout key it has not
    /// examined.
    fn stops_at(&self, frontier: u128, tally: &WalkTally) -> bool {
        self.progress.is_none_or(|progress| frontier < progress) && self.is_spent(tally)
    }

    fn is_spent(&self, tally: &WalkTally) -> bool {
        self.work
            .is_some_and(|work| tally.postings_scanned + tally.store_reads >= work)
            || self
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
    }
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
        self.segments_wasted
            .fetch_add(tally.segments_wasted, Relaxed);
        self.cut_short.fetch_add(tally.cut_short, Relaxed);
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
            segments_wasted: self.segments_wasted.load(Relaxed),
            cut_short: self.cut_short.load(Relaxed),
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
    /// Segments an ordered search opened that put no row on its page: what the overlap of the
    /// layout costs the searches, where they go.
    pub(crate) segments_wasted: u64,
    /// Searches their budget ended before the page was settled: they returned what was, and a
    /// cursor to go on from.
    pub(crate) cut_short: u64,
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
    pub(crate) merges: MergeTotals,
    pub(crate) overlap: LayoutOverlap,
    pub(crate) segments: Vec<SegmentLayout>,
}

/// How many segments may cover one key before the rows there count as misplaced: a value that
/// fills more than a segment, or a boundary two neighbours share, puts a key in two segments
/// even when the layout is as good as it gets.
const OVERLAP_TOLERANCE: u32 = 2;

/// How far an ordered index's segments are from disjoint, by their layout-key bounds. A
/// segment's depth is the most segments covering any one key within its bounds: what an ordered
/// walk through that stretch has to open where one would do. 1 everywhere is a disjoint layout;
/// the wide pieces a commit leaves over the whole range give every segment the depth of their
/// count.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LayoutOverlap {
    /// The deepest any key is covered.
    pub(crate) depth_max: u32,
    /// Live rows summed over segments, each segment's weighted by its depth: over `rows`, the
    /// depth a row sits at on average.
    pub(crate) depth_rows: u64,
    /// Live rows in the segments with bounds.
    pub(crate) rows: u64,
    /// Live rows in segments deeper than `OVERLAP_TOLERANCE`.
    pub(crate) rows_over_tolerance: u64,
    /// Deleted rows not yet merged away: a search still reads their postings.
    pub(crate) rows_deleted: u64,
    /// Live rows and segments in the fresh tier, which a repair has yet to place (see
    /// [`Piece`]).
    pub(crate) fresh_rows: u64,
    pub(crate) fresh_segments: u64,
}

/// [`LayoutOverlap`] of `pieces`, but for the deleted rows. Empty segments are left out.
fn layout_overlap(pieces: &[Piece]) -> LayoutOverlap {
    let mut overlap = LayoutOverlap::default();
    for (piece, ordered) in pieces.iter().zip(ordered_tier(pieces)) {
        if !ordered && piece.docs > 0 {
            overlap.fresh_rows += u64::from(piece.docs);
            overlap.fresh_segments += 1;
        }
    }
    let segments: Vec<_> = pieces
        .iter()
        .filter(|piece| piece.docs > 0)
        .map(|piece| (piece.docs, piece.min, piece.max))
        .collect();
    let mut starts: Vec<u128> = segments.iter().map(|s| s.1).collect();
    let mut ends: Vec<u128> = segments.iter().map(|s| s.2).collect();
    starts.sort_unstable();
    ends.sort_unstable();
    // The depth at a key is the segments starting at or before it less those ending before it.
    // Within any span it peaks at some segment's start, so the starts are the keys to look at.
    let depths: Vec<u32> = starts
        .iter()
        .map(|&key| {
            (starts.partition_point(|&start| start <= key) - ends.partition_point(|&end| end < key))
                as u32
        })
        .collect();
    for &(docs, min, max) in &segments {
        let from = starts.partition_point(|&start| start < min);
        let to = starts.partition_point(|&start| start <= max);
        let depth = depths[from..to].iter().copied().max().unwrap_or(1);
        let docs = u64::from(docs);
        overlap.depth_max = overlap.depth_max.max(depth);
        overlap.depth_rows += docs * u64::from(depth);
        overlap.rows += docs;
        if depth > OVERLAP_TOLERANCE {
            overlap.rows_over_tolerance += docs;
        }
    }
    overlap
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
}

/// What a row contributes to a pattern index: the text to index, and for an index that keeps an
/// order, the row's sort prefix.
pub(crate) struct PatternRow<'a> {
    pub(crate) text: &'a str,
    pub(crate) prefix: Option<SortPrefix>,
    /// Placed by a repair step rather than ingested: see `REPAIRED_FIELD`.
    pub(crate) repaired: bool,
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
        // Also columnar: the walk reads it to tell apart rows whose whole sort prefix is equal. A
        // page's ids still come from the store, which measured no slower.
        schema_builder.add_u64_field(PRIMARY_ID_FIELD, INDEXED | STORED | FAST);
        schema_builder.add_text_field(TEXT_FIELD, text_options);
        // FAST, not STORED: the walk reads them once per candidate, and a columnar read is far
        // cheaper than the document store. Their per-segment min/max is also what lets a search
        // skip whole segments.
        for field in self.slot_fields() {
            schema_builder.add_u64_field(&field, FAST);
        }
        schema_builder.add_u64_field(REPAIRED_FIELD, FAST);
        schema_builder.build()
    }

    fn create_doc(
        &self,
        schema: &Schema,
        primary_id: PrimaryId,
        row: PatternRow<'_>,
    ) -> TantivyDocument {
        let primary_id_field = schema.get_field(PRIMARY_ID_FIELD).unwrap();
        let text_field = schema.get_field(TEXT_FIELD).unwrap();

        // The value is indexed and stored as the patterns are compared with it: letter case
        // folded unless the index is case-sensitive, between the frame marks. The stored value
        // is never returned to clients.
        let folded = self.options.case_sensitive.fold(row.text.to_string());
        let mut doc = TantivyDocument::new();
        doc.add_u64(primary_id_field, u64::from(primary_id));
        doc.add_text(text_field, gram::framed(&folded));
        // The caller leaves out a row without a prefix (a null ordering value), so one is always
        // there; a zero prefix keeps the document well-formed if not.
        let prefix = row.prefix.unwrap_or_default();
        for (slot, field) in self.slot_fields().iter().enumerate() {
            doc.add_u64(schema.get_field(field).unwrap(), prefix.0[slot]);
        }
        doc.add_u64(
            schema.get_field(REPAIRED_FIELD).unwrap(),
            u64::from(row.repaired),
        );
        doc
    }

    /// The layout policy of an ordered index with a segment cap: two tiers, merged apart (see
    /// [`Piece`] and `merge_runs`), and the rows of the fresh one repaired into the ordered one
    /// (see `repair_step`).
    ///
    /// Tantivy's default policy merges segments of similar size whatever they hold, which is how an
    /// index ends up with 1.3M-row segments spanning a quarter of the range each and a tail of small
    /// ones that all reach the top: a page-1 query opens every tail segment and a deep page scans
    /// whichever big segment its cursor lands in. Here no merge mixes the tiers, and an ordered
    /// merge only takes neighbours in sort order, never past the cap, so the ordered tier stays
    /// disjoint. Segments whose bounds are not known yet (committed but not reloaded) wait for
    /// the next tick.
    ///
    /// All of it is done by `after_reload`, on every tick; the policy the writer holds merges
    /// nothing, beyond keeping Tantivy's default one out. Two deciders picked the same runs: Tantivy
    /// hands its policy only the segments no merge is reading, but `after_reload` cut its runs from
    /// every live segment and knew of its own merges only, so it asked again for each run the policy
    /// had just started -- in a local run of local_orders_config.yaml, 3,772 merges of 10,810, each
    /// ending in "couldn't find segment in SegmentManager". With one decider, `MergesInFlight` sees
    /// every merge. A merge starts up to a tick after the commit that made it possible.
    fn merge_policy(&self) -> Option<Box<dyn MergePolicy>> {
        self.options.segment_max_docs.as_ref().as_ref()?;
        Some(Box::new(tantivy::indexer::NoMergePolicy))
    }

    /// Learns every live segment's bounds (which also warms the column cache), drops the retired
    /// ones, adopts the segments found where they belong, and asks the writer to merge the runs
    /// `merge_runs` cuts (see `merge_policy`). Not left to Tantivy, which evaluates merges after a
    /// commit: at that moment the bounds of the segments the commit produced are not known yet,
    /// so the runs they belong to would wait for a commit that may never come once the writes
    /// stop. Run on every tick, since the reader reloads on its own.
    fn after_reload(&self, state: &PatternIndexState) {
        let searcher = state.reader.searcher();
        let mut pieces = self.geometry(&searcher);
        self.prune_columns(searcher.segment_readers());

        let Some(max_docs) = *self.options.segment_max_docs.as_ref() else {
            return;
        };
        let mut merging = self.merging.lock().unwrap();
        let busy = merging.busy(&mut self.adopted.lock().unwrap());
        if merging.unsettled(&live_segments(&searcher)) {
            return;
        }
        // A merge that ended since `geometry` read the adoptions may have adopted its output.
        let adopted = self.adopted.lock().unwrap().clone();
        for piece in &mut pieces {
            piece.ordered |= adopted.contains(&piece.id);
        }
        self.adopt(&mut pieces);
        // A run with a segment already being merged waits for that merge: see `MergesInFlight`.
        // The runs are still cut from every live segment, not from the idle ones alone, which
        // would make a busy segment's neighbours look adjacent and merge them across it into a
        // segment overlapping it.
        let runs: Vec<(Vec<SegmentId>, bool)> = merge_runs(&pieces, max_docs.get())
            .into_iter()
            .filter(|(run, _)| !run.iter().any(|id| busy.contains(id)))
            .collect();
        if !runs.is_empty() {
            info!(
                "pattern: asking for {} merges of {} segments ({} live, {} being merged)",
                runs.len(),
                runs.iter().map(|(run, _)| run.len()).sum::<usize>(),
                searcher.segment_readers().len(),
                busy.len()
            );
        }
        #[cfg(test)]
        if walk_off(OFF_TRACE_REPAIR) {
            let docs: HashMap<SegmentId, u32> =
                pieces.iter().map(|piece| (piece.id, piece.docs)).collect();
            for (run, ordered) in &runs {
                println!(
                    "MERGE asked: {} segments, {} rows, ordered {ordered}; {} live, {} busy",
                    run.len(),
                    run.iter().map(|id| u64::from(docs[id])).sum::<u64>(),
                    pieces.len(),
                    busy.len()
                );
            }
        }
        for (run, ordered) in runs {
            // The merge runs on Tantivy's own thread; the handle only reports its end, and the
            // next reload picks the merged segment up either way.
            let merge = state.lock_writer().merge(&run);
            merging.started(run, ordered, merge);
        }
    }

    /// Adopts into the ordered tier the pieces `ordered_tier` finds there that no tag says are.
    fn adopt(&self, pieces: &mut [Piece]) {
        let tiers = ordered_tier(pieces);
        let mut adopted = self.adopted.lock().unwrap();
        for (piece, ordered) in pieces.iter_mut().zip(tiers) {
            if ordered && !piece.ordered {
                adopted.insert(piece.id);
                piece.ordered = true;
            }
        }
    }
}

/// The ids of the searcher's segments.
fn live_segments(searcher: &tantivy::Searcher) -> HashSet<SegmentId> {
    searcher
        .segment_readers()
        .iter()
        .map(SegmentReader::segment_id)
        .collect()
}

impl PatternBackend {
    /// Whether the layout has anything to repair, recorded for `Stats`, and a repair step when
    /// it does and the moment allows (see `repair_due`). Two policies, by what brings the rows:
    ///
    /// - A full scan, building the index: no step until it ends, `SCAN_QUIET_TICKS` ticks after
    ///   its last row. The scan goes at the node's pace, and the repair then works from every row
    ///   at once rather than placing the first ones only to split their stretches again as the
    ///   rest arrive: a 10M-row build repaired during its scan took 124 s instead of 67, and
    ///   moved 1.5 rows for each one.
    /// - CDC changes lagging behind the table (those since the last tick, over
    ///   `REPAIR_LAG_LIMIT` on average): no step either, idle tick or not, until changes that
    ///   keep up arrive or `LAG_QUIET_TICKS` ticks pass without any, so that a bulk load
    ///   through CDC is repaired once the index has caught up, as a full scan is.
    /// - Writes the index keeps up with: on an idle tick always, so that an idle index converges
    ///   to an empty fresh tier and a disjoint ordered one; while the index ingests, once the
    ///   fresh tier holds `LOAD_FRESH_ROWS` rows. Each hold of the writer by a step then waits
    ///   for a place among the worker pool's background tasks (`repair_place`), so the searches
    ///   and the live writes of every index go first, and a burst of writes grows the fresh
    ///   tier, which searches pay for with an open per fresh segment, rather than the lag.
    fn maybe_repair(
        &self,
        state: &PatternIndexState,
        key: &IndexKey,
        idle: bool,
        worker: &Sender<Worker>,
    ) {
        if !*self.options.rewrite_wide_segments.as_ref() {
            return;
        }
        let Some(cap) = *self.options.segment_max_docs.as_ref() else {
            return;
        };
        let cap = u64::from(cap.get());
        let since_scan = if self.scan_rows.swap(false, Relaxed) {
            0
        } else {
            self.ticks_since_scan.load(Relaxed).saturating_add(1)
        };
        self.ticks_since_scan.store(since_scan, Relaxed);
        let scanning = since_scan < SCAN_QUIET_TICKS;
        let changes = self.cdc_changes.swap(0, Relaxed);
        let lag_millis = self.cdc_lag_millis.swap(0, Relaxed);
        let since_lag = match lag_millis.checked_div(changes) {
            Some(mean) if mean > REPAIR_LAG_LIMIT.as_millis() as u64 => 0,
            // Changes that keep up: the index has caught up.
            Some(_) => LAG_QUIET_TICKS,
            None => self.ticks_since_lag.load(Relaxed).saturating_add(1),
        };
        self.ticks_since_lag.store(since_lag, Relaxed);
        let lagging = since_lag < LAG_QUIET_TICKS;
        let pieces = self.geometry(&state.reader.searcher());
        let ordered = ordered_tier(&pieces);
        let fresh_rows: u64 = pieces
            .iter()
            .zip(&ordered)
            .filter(|&(_, &ordered)| !ordered)
            .map(|(piece, _)| u64::from(piece.docs))
            .sum();
        let overfull = clusters(&pieces, &ordered)
            .iter()
            .any(|cluster| cluster.docs > cap);
        let pending = fresh_rows > 0 || overfull;
        self.repair.pending.store(pending, Relaxed);
        if !pending {
            return;
        }
        // Lagging holds the repair on an idle tick too: a commit at the row threshold just before
        // the tick leaves nothing to commit in the middle of a stream of changes.
        #[cfg(test)]
        {
            GATE_TICKS[0].fetch_add(1, Relaxed);
            GATE_TICKS[1].fetch_add(u64::from(lagging), Relaxed);
        }
        if !repair_due(idle, scanning || lagging, fresh_rows, overfull) {
            return;
        }
        if let Err(err) = repair_step(state, worker) {
            error!("pattern: repair of {key} failed: {err}");
        }
    }

    /// Counts a CDC change taken in, `lag` after its write to the table.
    fn note_cdc_lag(&self, lag: Duration) {
        self.cdc_lag_millis
            .fetch_add(lag.as_millis().try_into().unwrap_or(u64::MAX), Relaxed);
        self.cdc_changes.fetch_add(1, Relaxed);
    }

    fn rewrite_totals(&self) -> RewriteTotals {
        RewriteTotals {
            steps: self.repair.steps.load(Relaxed),
            docs_rewritten: self.repair.docs_rewritten.load(Relaxed),
            pending: self.repair.pending.load(Relaxed),
        }
    }
}

/// Whether a repair step may run now, given what there is to repair: never while the index is
/// `catching_up` -- a full scan going on, or CDC changes lagging behind the table; on an idle
/// tick always; while the index ingests, once the fresh tier holds
/// `LOAD_FRESH_ROWS` rows or an ordered stretch is past the cap. When the step's work then runs
/// is the worker pool's to say: see `repair_place`. Pure, so it can be tested without an index.
fn repair_due(idle: bool, catching_up: bool, fresh_rows: u64, overfull: bool) -> bool {
    if (fresh_rows == 0 && !overfull) || catching_up {
        return false;
    }
    idle || fresh_rows >= LOAD_FRESH_ROWS || overfull
}

/// A place among the pool's background tasks for one hold of the writer by a repair step, taken
/// before the writer and given back after it. The step runs on a thread of its own, a hold
/// being too long for a worker, and with a place it still waits for the searches and the live
/// writes of every index as the writes of a build do, runs beside at most as many background
/// tasks as they may, and counts into their share of the workers' time.
fn repair_place(worker: &Sender<Worker>) -> Option<worker::Place> {
    if walk_off(OFF_POOL_PLACE) {
        return None;
    }
    worker.take_background_place()
}

/// A row a repair step moves: its layout key, the term that deletes its current copy, and the
/// document added in its place.
type MovedRow = (u128, (Term, TantivyDocument));

/// Appends to `rows` the live rows of `segment` whose layout key `wanted` takes, as repaired.
fn read_rows(
    state: &PatternIndexState,
    segment: &SegmentReader,
    wanted: impl Fn(u128) -> bool,
    rows: &mut Vec<MovedRow>,
) -> anyhow::Result<()> {
    let backend = state.backend();
    let text_field = state.schema.get_field(TEXT_FIELD).unwrap();
    let primary_id_field = state.schema.get_field(PRIMARY_ID_FIELD).unwrap();
    let mut opened = 0;
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
        if !wanted(layout_key) {
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
            .map(|stored| gram::unframed(stored).unwrap_or(stored))
            .unwrap_or_default();
        let new_doc = backend.create_doc(
            &state.schema,
            primary_id,
            PatternRow {
                text,
                prefix: Some(columns.prefix(doc_id)),
                repaired: true,
            },
        );
        rows.push((
            layout_key,
            (primary_id_term(&state.schema, primary_id), new_doc),
        ));
    }
    Ok(())
}

/// Appends to `keys` the layout keys of the live rows of `segment` that `wanted` takes.
fn read_keys(
    backend: &PatternBackend,
    segment: &SegmentReader,
    wanted: impl Fn(u128) -> bool,
    keys: &mut Vec<u128>,
) -> anyhow::Result<()> {
    let mut opened = 0;
    let columns = backend.columns_for(segment, &mut opened)?;
    let alive = segment.alive_bitset();
    keys.extend(
        (0..segment.max_doc())
            .filter(|&doc_id| alive.is_none_or(|alive| alive.is_alive(doc_id)))
            .map(|doc_id| columns.layout_key(doc_id))
            .filter(|&key| wanted(key)),
    );
    Ok(())
}

/// `rows`, sorted by layout key, cut into parts of about equal size and at most `size` rows,
/// each a range of keys: a key is never cut, so the parts are disjoint, and one holding more rows
/// than `size` is a part of its own.
fn split_parts<T>(rows: Vec<(u128, T)>, size: usize) -> Vec<Vec<(u128, T)>> {
    let size = size.max(1);
    let parts = rows.len().div_ceil(size).max(1);
    let size = rows.len().div_ceil(parts).max(1);
    let mut split: Vec<Vec<(u128, T)>> = Vec::with_capacity(parts);
    for row in rows {
        match split.last_mut() {
            Some(part) if part.len() < size || part.last().is_some_and(|last| last.0 == row.0) => {
                part.push(row);
            }
            _ => split.push(vec![row]),
        }
    }
    split
}

/// One repair step (see `plan_repair`): the fresh rows of the worst stretch's range are deleted
/// by primary id and added again as repaired. A commit's rows land in pieces spanning no more than
/// the keys they hold, so rows repaired into a cluster's stretch overlap the cluster and merge
/// into it, and rows of a gap make a cluster of their own. Rows that would not fit beside the
/// cluster under the cap are moved with the cluster's own, in parts of disjoint key ranges.
///
/// What to move is decided under the writer lock; the moving is done in key ranges of at most
/// `repair_range_rows`, each under a hold of the lock of its own, committed before and after, its
/// rows read again from what is committed then. The lock is what keeps a concurrent update or
/// delete of one of those rows from being undone by the re-add: it waits, and lands with a higher
/// opstamp; a change between two ranges is committed before the next range reads its rows.
/// Searches take the reader, not this lock. Returns whether rows were moved.
fn repair_step(state: &PatternIndexState, worker: &Sender<Worker>) -> anyhow::Result<bool> {
    let backend = state.backend();
    let Some(cap) = *backend.options.segment_max_docs.as_ref() else {
        return Ok(false);
    };
    let cap = cap.get();
    {
        // A merge whose output is not known to be ordered yet would read as fresh.
        let mut merging = backend.merging.lock().unwrap();
        merging.busy(&mut backend.adopted.lock().unwrap());
        if merging.unsettled(&live_segments(&state.reader.searcher())) {
            repair_phase(1);
            return Ok(false);
        }
    }

    repair_phase(10);
    let waiting = Instant::now();
    let place = repair_place(worker);
    let mut writer = state.lock_writer();
    let held = Instant::now();
    repair_phase(11);
    state
        .commit_and_reload(&mut writer)
        .map_err(|e| anyhow!("pattern: failed to commit before a repair: {e}"))?;
    let committed = held.elapsed();
    repair_phase(12);
    let searcher = state.reader.searcher();
    let pieces = backend.geometry(&searcher);
    let ordered = ordered_tier(&pieces);
    let fresh: HashMap<SegmentId, Piece> = pieces
        .iter()
        .zip(&ordered)
        .filter(|&(piece, &ordered)| !ordered && piece.docs > 0)
        .map(|(piece, _)| (piece.id, *piece))
        .collect();
    let fresh_rows: u64 = fresh.values().map(|piece| u64::from(piece.docs)).sum();
    // Every stride-th live row, the first of each segment included, so that a segment with a
    // few rows left is sampled too and the repair does not stop short of it.
    let stride = (fresh_rows / REPAIR_SAMPLE_TARGET).max(1);
    let mut sample = Vec::new();
    let mut opened = 0;
    for segment in searcher.segment_readers() {
        if !fresh.contains_key(&segment.segment_id()) {
            continue;
        }
        let columns = backend.columns_for(segment, &mut opened)?;
        let alive = segment.alive_bitset();
        let mut seen = 0u64;
        for doc_id in 0..segment.max_doc() {
            if alive.is_some_and(|alive| !alive.is_alive(doc_id)) {
                continue;
            }
            if seen.is_multiple_of(stride) {
                sample.push(columns.layout_key(doc_id));
            }
            seen += 1;
        }
    }
    sample.sort_unstable();
    let sampled = held.elapsed();
    let Some(range) = plan_repair(&pieces, &sample, stride, cap) else {
        return Ok(false);
    };

    // What the step moves, by key: the fresh rows of the range, and with them the cluster's own
    // when they would not fit beside it under the cap.
    let in_range = |key: u128| range.low <= key && key <= range.high;
    let mut keys = Vec::new();
    for segment in searcher.segment_readers() {
        let Some(piece) = fresh.get(&segment.segment_id()) else {
            continue;
        };
        if piece.max < range.low || piece.min > range.high {
            continue;
        }
        read_keys(backend, segment, in_range, &mut keys)?;
    }
    let split = match &range.cluster {
        Some(cluster) => cluster.docs + keys.len() as u64 > u64::from(cap),
        None => keys.len() > cap as usize,
    };
    if split && let Some(cluster) = &range.cluster {
        for segment in searcher.segment_readers() {
            if cluster.ids.contains(&segment.segment_id()) {
                read_keys(backend, segment, |_| true, &mut keys)?;
            }
        }
    }
    #[cfg(not(test))]
    let _ = (waiting, committed, sampled);
    #[cfg(test)]
    if walk_off(OFF_TRACE_REPAIR) {
        println!(
            "REPAIR plan hold: waited {:.2}s for the writer, held {:.2}s: commit {:.2}s, sample to {:.2}s",
            (held - waiting).as_secs_f64(),
            held.elapsed().as_secs_f64(),
            committed.as_secs_f64(),
            sampled.as_secs_f64(),
        );
        println!(
            "REPAIR plan: {} fresh rows in {} segments, stride {stride}, sample {}; range {:x}..{:x}, cluster {:?}, split {split}, {} keys",
            fresh_rows,
            fresh.len(),
            sample.len(),
            range.low >> 64,
            range.high >> 64,
            range
                .cluster
                .as_ref()
                .map(|cluster| (cluster.docs, cluster.ids.len())),
            keys.len(),
        );
    }
    drop(searcher);
    drop(writer);
    drop(place);
    if keys.is_empty() {
        return Ok(false);
    }
    keys.sort_unstable();
    let ranges: Vec<(u128, u128, usize)> = split_parts(
        keys.into_iter().map(|key| (key, ())).collect(),
        repair_range_rows(cap),
    )
    .iter()
    .map(|part| (part[0].0, part[part.len() - 1].0, part.len()))
    .collect();

    // Each key range under a hold of the lock of its own, its rows read again from what is
    // committed now: a change to one of them since is already in, and a change after lands on
    // the copy added here. The writes waiting for the lock go in between ranges.
    let mut count = 0;
    for &(low, high, planned) in &ranges {
        repair_phase(20);
        let waiting = Instant::now();
        let place = repair_place(worker);
        let mut writer = state.lock_writer();
        let held = Instant::now();
        repair_phase(21);
        state
            .commit_and_reload(&mut writer)
            .map_err(|e| anyhow!("pattern: failed to commit before a repair: {e}"))?;
        let committed = held.elapsed();
        repair_phase(22);
        let searcher = state.reader.searcher();
        let pieces = backend.geometry(&searcher);
        let ordered = ordered_tier(&pieces);
        let by_id: HashMap<SegmentId, (Piece, bool)> = pieces
            .iter()
            .zip(ordered)
            .map(|(piece, ordered)| (piece.id, (*piece, ordered)))
            .collect();
        let mut moved = Vec::new();
        for segment in searcher.segment_readers() {
            let Some(&(piece, ordered)) = by_id.get(&segment.segment_id()) else {
                continue;
            };
            // The range lies in the cluster's stretch when it splits it, and the ordered pieces
            // over it are the cluster's: those of the ranges already moved lie below it.
            if piece.docs == 0 || piece.max < low || piece.min > high || (ordered && !split) {
                continue;
            }
            // A fresh row only where the plan counted them: a cluster's keys may reach past the
            // step's range, and a part that does spans fresh rows the plan left for later steps,
            // any number of them.
            read_rows(
                state,
                segment,
                |key| {
                    low <= key
                        && key <= high
                        && (ordered || in_range(key) || walk_off(OFF_HOLD_BOUND))
                },
                &mut moved,
            )?;
        }
        // In key order: the writer's threads each take an ordered share of what is added, so
        // the pieces are sorted runs, which the walk takes from the end it needs (see
        // `SegmentColumns::runs`), and an ordered merge, given its segments in key order, keeps
        // them so.
        moved.sort_by_key(|row| row.0);
        count += moved.len();
        let rows = moved.len();
        let read = held.elapsed();
        repair_phase(23);
        writer
            .rewrite_documents(moved.into_iter().map(|(_, row)| row))
            .map_err(|e| anyhow!("pattern: failed to repair a range: {e}"))?;
        let rewritten = held.elapsed();
        repair_phase(24);
        state
            .commit_and_reload(&mut writer)
            .map_err(|e| anyhow!("pattern: failed to commit a repair: {e}"))?;
        #[cfg(test)]
        if walk_off(OFF_TRACE_REPAIR) {
            println!(
                "REPAIR range hold: {rows} rows of {planned} planned, waited {:.2}s, held {:.2}s: commit {:.2}s, read to {:.2}s, rewrite to {:.2}s",
                (held - waiting).as_secs_f64(),
                held.elapsed().as_secs_f64(),
                committed.as_secs_f64(),
                read.as_secs_f64(),
                rewritten.as_secs_f64(),
            );
        }
        #[cfg(not(test))]
        let _ = (waiting, committed, rows, planned, read, rewritten);
        // The writes that waited through this hold go before the next one.
        drop(searcher);
        drop(writer);
        drop(place);
        repair_phase(25);
        if !walk_off(OFF_HANDOVER) {
            state.let_writes_in(REPAIR_HANDOVER_LIMIT);
        }
    }
    let part_count = ranges.len();

    backend.repair.steps.fetch_add(1, Relaxed);
    backend
        .repair
        .docs_rewritten
        .fetch_add(count as u64, Relaxed);
    debug!(
        "pattern: repaired {count} rows into {part_count} parts{}, {fresh_rows} fresh rows before",
        if range.cluster.is_some() {
            " of an ordered stretch"
        } else {
            ""
        }
    );
    // Learn the new segments' bounds and let the policy fold the pieces together.
    repair_phase(30);
    backend.after_reload(state);
    repair_phase(0);
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
        repair: RepairCounters::default(),
        adopted: Mutex::new(HashSet::new()),
        scan_rows: AtomicBool::new(false),
        ticks_since_scan: AtomicU32::new(SCAN_QUIET_TICKS),
        cdc_lag_millis: AtomicU64::new(0),
        cdc_changes: AtomicU64::new(0),
        ticks_since_lag: AtomicU32::new(LAG_QUIET_TICKS),
        tick_running: AtomicBool::new(false),
        merging: Mutex::new(MergesInFlight::default()),
        options: options.clone(),
    };
    let schema = backend.build_schema();
    let analyzer = gram::build_ngram_analyzer(backend.min_gram(), backend.max_gram())?;
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
    let table = table.read().unwrap();
    let mut prefix = SortPrefix::default();
    let columns = options.order_by.columns();
    for (slot, column) in columns.iter().enumerate() {
        let value = table.column_value_for(partition_id, primary_id, &column.name)?;
        let key = crate::cql_types::to_sort_key(&value)?;
        prefix.0[slot] = if column.descending { !key } else { key };
    }
    let primary_key = table.primary_key(partition_id, primary_id)?;
    prefix.0[columns.len()] =
        key_layout.partition_order_key(&primary_key, options.partition_order)?;
    Some(prefix)
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

    /// The window's part over an oriented layout key: the rows whose own lies above `key`.
    /// None when no key does.
    fn above(&self, key: u128) -> Option<Self> {
        let key = key.checked_add(1)?;
        let mut floor = SortPrefix::default();
        floor.0[0] = (key >> 64) as u64;
        floor.0[1] = key as u64;
        Some(Self {
            lower: Some(self.lower.map_or(floor, |lower| lower.max(floor))),
            ..self.clone()
        })
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

    /// The highest oriented layout key the window lets in: the lower of the range's top and the
    /// cursor's. None when neither bounds it.
    fn top_layout_key(&self) -> Option<u128> {
        let upper = self.upper.map(|upper| upper.layout_key());
        let cursor = self.cursor.as_ref().map(|(cursor, _)| cursor.layout_key());
        match (upper, cursor) {
            (Some(upper), Some(cursor)) => Some(upper.min(cursor)),
            (upper, cursor) => upper.or(cursor),
        }
    }

    /// Whether rows at an oriented layout key of `key` or below may still be in the range.
    fn reaches(&self, key: u128) -> bool {
        self.lower.is_none_or(|lower| key >= lower.layout_key())
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
    /// The ordinal of the segment the row is in, to tell the segments the page needed from the
    /// ones it opened in vain.
    segment: u32,
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
        // A row indexed twice has two documents under one primary id, and is on the page once.
        if self
            .entries
            .get(at)
            .is_some_and(|held| held.primary_id == entry.primary_id)
        {
            return;
        }
        self.entries.insert(at, entry);
        self.entries.truncate(self.limit);
    }
}

/// Set by a benchmark test to record the shape of every segment's runs in `RUN_SHAPES`, by the
/// segment's size and lowest key.
#[cfg(test)]
static RECORD_RUN_SHAPES: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static RUN_SHAPES: Mutex<RunShapes> = Mutex::new(RunShapes::new());
#[cfg(test)]
type RunShapes = std::collections::BTreeMap<(u32, u128), (usize, usize, u32)>;

/// Mechanisms of the walk a benchmark test switches off, to see what each is worth: see
/// `walk_off`.
#[cfg(test)]
static WALK_OFF: AtomicU32 = AtomicU32::new(0);
const OFF_TAKE: u32 = 2;
const OFF_HALVING: u32 = 4;
const OFF_SWEEP: u32 = 8;
const OFF_WHOLE_READ: u32 = 16;
const OFF_SPREAD: u32 = 4096;
/// What the walk of segments did, for a benchmark test to report: passes over a segment's runs,
/// regions scanned, scorers made, and the nanoseconds of the scans, the verifications and the
/// sweeps.
#[cfg(test)]
static WALK_PROBE: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Adds to `WALK_PROBE[what]`: `amount`, or with a clock the time since it.
fn probe(what: usize, amount: u64, since: Option<Instant>) {
    #[cfg(test)]
    WALK_PROBE[what].fetch_add(
        since.map_or(amount, |since| since.elapsed().as_nanos() as u64),
        Relaxed,
    );
    #[cfg(not(test))]
    let _ = (what, amount, since);
}

/// Not off but always on: every segment read whole, to find where the two readings cross.
const FORCE_WHOLE_READ: u32 = 32;
const OFF_HANDOVER: u32 = 256;
const OFF_HOLD_BOUND: u32 = 512;
const OFF_SORTED_COMMITS: u32 = 1024;
const OFF_FRESH_BETWEEN: u32 = 2048;
/// The repair's holds of the writer without a place in the worker pool.
const OFF_POOL_PLACE: u32 = 8192;
/// Ticks with a repair pending, and of those with changes lagging. For a benchmark test.
#[cfg(test)]
static GATE_TICKS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
const OFF_LOCK_CHUNKS: u32 = 128;
/// Where a repair step is, for a benchmark test that watches one stall: see `repair_phase`.
#[cfg(test)]
static REPAIR_PHASE: AtomicU32 = AtomicU32::new(0);

/// Notes where a repair step is: 1 merges unsettled; 10 waiting for the writer to plan, 11
/// committing, 12 sampling and reading keys; 20 waiting for the writer for a range, 21
/// committing, 22 reading its rows, 23 rewriting, 24 committing, 25 letting writes in; 30 the
/// merges after the step; 0 done.
fn repair_phase(phase: u32) {
    #[cfg(test)]
    REPAIR_PHASE.store(phase, Relaxed);
    #[cfg(not(test))]
    let _ = phase;
}
#[cfg(test)]
const OFF_TRACE_REPAIR: u32 = 64;

/// Whether a benchmark test has switched the mechanism off; never outside tests.
fn walk_off(mechanism: u32) -> bool {
    #[cfg(test)]
    return WALK_OFF.load(Relaxed) & mechanism != 0;
    #[cfg(not(test))]
    {
        let _ = mechanism;
        false
    }
}

/// Segments an ordered walk read whole for want of sorted runs, for a benchmark test to report.
#[cfg(test)]
static SEGMENTS_READ_WHOLE: AtomicU64 = AtomicU64::new(0);

/// How many candidates are verified between two looks at a search's budget.
const BUDGET_CHECK_EVERY: u32 = 32;
/// What checking a stored value costs, in postings read: see `RegionWalk::verified_its_reading`.
const STORED_VALUE_COST: u64 = 4;

/// One segment as an ordered walk reads it: the rows of a range of doc ids at a time, either the
/// whole segment, or stretches of a sorted run from the end the page needs (see
/// `SegmentColumns::runs`).
struct RegionWalk<'a> {
    segment: &'a SegmentReader,
    segment_ord: u32,
    columns: &'a SegmentColumns,
    window: &'a SortWindow,
    ties: &'a Ties<'a>,
    key_of: &'a dyn Fn(PrimaryId) -> Option<PrimaryKey>,
    like: &'a mut LikeQuery,
    /// Opened for the first candidate to verify.
    store: Option<tantivy::store::StoreReader>,
    /// The postings, kept while the next range starts at or after where they stand: a scorer
    /// only seeks forward.
    scorer: Option<Box<dyn Scorer>>,
    /// The search's postings read when it came to this segment, and its stored values checked
    /// when this segment's postings were last read.
    before: (u64, u64),
}

impl RegionWalk<'_> {
    /// The row's entry if it is in the window and enters the page; its id is read only then.
    fn entry(&self, page: &Page, prefix: SortPrefix, doc_id: u32) -> Option<PageEntry> {
        let primary_id = self.columns.primary_id.first(doc_id).unwrap_or(0);
        let entry = PageEntry {
            prefix,
            primary_id,
            segment: self.segment_ord,
        };
        (self
            .window
            .below_cursor(prefix, || (self.key_of)(primary_id.into()), self.ties)
            && page.takes(&entry))
        .then_some(entry)
    }

    /// Offers the page a matching row: an exact match enters the page straight from the columns,
    /// a row the grams only nominate is added to `candidates`, for `verify`.
    fn offer(
        &self,
        page: &mut Page,
        tally: &mut WalkTally,
        candidates: &mut Vec<(SortPrefix, u32)>,
        doc_id: u32,
    ) {
        let alive = self.segment.alive_bitset();
        if alive.is_some_and(|alive| !alive.is_alive(doc_id)) {
            return;
        }
        let prefix = self.window.orient(self.columns.prefix(doc_id));
        if self.window.admits(prefix) && page.may_take(prefix) {
            if self.like.needs_verification() {
                candidates.push((prefix, doc_id));
            } else if let Some(entry) = self.entry(page, prefix, doc_id) {
                tally.heap_entrants += 1;
                page.insert(entry);
            }
        }
    }

    /// Offers the page the matching rows with doc ids in `regions`, which are disjoint and in
    /// doc order, with one pass of the postings: opened once, or the ones left from the last
    /// call if they stand before the first region, as postings only seek forward.
    ///
    /// Of a region that is a stretch of a run, only the `take` matches at the end
    /// the walk reads from are offered, and the region is cut to what was read: the matches may
    /// lie anywhere in it, and many more of them than a page needs once they start. Where the
    /// walk starts at the region's first doc ids, the region ends with the last match taken.
    /// Where it starts at the last, the postings come from the far end, so the region is halved
    /// towards the near one until what is left holds `take` matches or fewer. A coarse run is
    /// cut at a block's edge.
    fn scan(
        &mut self,
        page: &mut Page,
        tally: &mut WalkTally,
        candidates: &mut Vec<(SortPrefix, u32)>,
        regions: &mut [Region],
        take: usize,
    ) -> anyhow::Result<()> {
        let Some(first) = regions.first().map(|region| region.from) else {
            return Ok(());
        };
        self.before.1 = tally.store_reads;
        probe(1, regions.len() as u64, None);
        let mut scorer = match self.scorer.take() {
            Some(scorer) if scorer.doc() <= first => scorer,
            _ => {
                probe(2, 1, None);
                match self.like.candidates(self.segment)? {
                    Some(scorer) => scorer,
                    None => return Ok(()),
                }
            }
        };
        let forward = self.window.complement;
        let mut last: Vec<u32> = Vec::new();
        for region in regions {
            let runs = region.run.and(self.columns.runs.as_deref());
            let run = runs
                .zip(region.run)
                .map(|(runs, run)| &runs.runs[run as usize]);
            let mut doc_id = if scorer.doc() < region.from {
                scorer.seek(region.from)
            } else {
                scorer.doc()
            };
            // The posting to read at or after `doc_id`: past the blocks of a coarse run that
            // cannot enter the page or the window by the keys of their own rows.
            let mut block_end = 0;
            let window = self.window;
            let end = region.to;
            let mut enter = |doc_id: u32, scorer: &mut Box<dyn Scorer>, page: &Page| {
                let (Some(runs), Some(run)) = (runs, run) else {
                    return doc_id;
                };
                let Some(first) = run.blocks else {
                    return doc_id;
                };
                let mut doc_id = doc_id;
                while doc_id >= block_end && doc_id < end {
                    block_end = run.edge_up(doc_id + 1);
                    let (min, max) = runs.own[(first + run.block_of(doc_id)) as usize];
                    let best = if forward { !min } else { max };
                    if !window.reaches(best)
                        || page
                            .weakest()
                            .is_some_and(|weakest| best < weakest.prefix.layout_key())
                    {
                        doc_id = scorer.seek(block_end);
                    }
                }
                doc_id
            };
            doc_id = enter(doc_id, &mut scorer, page);
            match run {
                // The matches the walk is after lie at the far end of the region's doc ids:
                // going down a run whose keys go up with them, or up one whose keys go down.
                Some(run) if forward == run.reversed => {
                    // The postings come from the far end of the region. Its nearer half is
                    // tried over and over, until it holds no more than `take` matches: each try
                    // reads that many and no more, and seeks on.
                    let mut from = region.from;
                    loop {
                        last.clear();
                        while doc_id < region.to && last.len() <= take {
                            tally.postings_scanned += 1;
                            last.push(doc_id);
                            doc_id = scorer.advance();
                            doc_id = enter(doc_id, &mut scorer, page);
                        }
                        if last.len() <= take || doc_id >= region.to {
                            break;
                        }
                        if walk_off(OFF_HALVING) {
                            // All of the region, as before the halving: its matches offered
                            // whatever their number.
                            while doc_id < region.to {
                                tally.postings_scanned += 1;
                                last.push(doc_id);
                                doc_id = scorer.advance();
                                doc_id = enter(doc_id, &mut scorer, page);
                            }
                            break;
                        }
                        // From a block's edge in a coarse run, whose blocks are read whole.
                        let half = run.edge_up((from + (region.to - from) / 2).max(doc_id));
                        if half >= region.to {
                            // The last block alone holds more: all of what is left, then.
                            while doc_id < region.to {
                                tally.postings_scanned += 1;
                                last.push(doc_id);
                                doc_id = scorer.advance();
                                doc_id = enter(doc_id, &mut scorer, page);
                            }
                            break;
                        }
                        from = half;
                        if doc_id < from {
                            doc_id = scorer.seek(from);
                        }
                        doc_id = enter(doc_id, &mut scorer, page);
                    }
                    region.from = from;
                    for &doc_id in &last {
                        self.offer(page, tally, candidates, doc_id);
                    }
                }
                _ => {
                    let mut taken = 0;
                    while doc_id < region.to {
                        tally.postings_scanned += 1;
                        self.offer(page, tally, candidates, doc_id);
                        taken += 1;
                        if let Some(run) = run
                            && taken == take
                        {
                            // To the end of the block for a coarse run, whose rows are in no
                            // order within it.
                            region.to = run.edge_up(doc_id + 1).min(region.to);
                        }
                        doc_id = scorer.advance();
                        doc_id = enter(doc_id, &mut scorer, page);
                    }
                }
            }
        }
        self.scorer = Some(scorer);
        Ok(())
    }

    /// Whether the verification since this segment's postings were last read has cost at least
    /// what reading them did. Until then the budget does not end it. Where the rows are in
    /// sorted runs a pass reads little, and this holds almost at once. An unsorted stretch -- a
    /// coarse run, a segment with no runs -- settles nothing until all of it is read, and the
    /// next request reads it all again: a request that stopped as its budget ended would have
    /// read for all of its time and verified next to nothing, and a page would take requests
    /// without number. What was verified between two reads does not count: the later read
    /// brings candidates over it.
    fn verified_its_reading(&self, tally: &WalkTally) -> bool {
        let (postings, stored) = self.before;
        (tally.store_reads - stored) * STORED_VALUE_COST >= tally.postings_scanned - postings
    }

    /// Verifies `candidates` against the stored text, from the highest prefix down, and offers
    /// the page the ones that match, stopping at the first that can no longer enter it. The grams
    /// only nominate candidates, and the text that decides is in the document store. Postings arrive in doc order, so verifying as they come reads nearly every
    /// candidate when doc order is roughly ascending sort order: each later one beats the page
    /// the earlier ones built. From the highest down, the store is read for the page's rows plus
    /// the false positives above them.
    ///
    /// The budget does not end the verification before it has cost what the reading of this
    /// segment did (`verified_its_reading`).
    ///
    /// `unread` is the highest layout key the segment has left unread, 0 for none. When the
    /// budget ends the verification, returns the highest layout key not examined: the highest
    /// candidate left unverified, or `unread` if that is higher -- a coarse run's blocks are read
    /// in doc order, and the one with its highest keys may be among those a pass left.
    fn verify(
        &mut self,
        page: &mut Page,
        tally: &mut WalkTally,
        candidates: &mut Vec<(SortPrefix, u32)>,
        budget: &Spending,
        unread: u128,
    ) -> anyhow::Result<Option<u128>> {
        // A heap, not a sort: of the many a wide pass gathers, few are looked at.
        let mut highest = BinaryHeap::from(std::mem::take(candidates));
        let mut verified = 0u32;
        let mut stopped_at = None;
        while let Some(&(prefix, doc_id)) = highest.peek() {
            if !page.may_take(prefix) {
                // Nor can any after it, now or once the page holds more.
                highest.clear();
                break;
            }
            // The clock is read once in so many, not for every stored value.
            let frontier = prefix.layout_key().max(unread);
            if verified.is_multiple_of(BUDGET_CHECK_EVERY)
                && budget.stops_at(frontier, tally)
                && self.verified_its_reading(tally)
            {
                stopped_at = Some(frontier);
                break;
            }
            verified += 1;
            highest.pop();
            let Some(entry) = self.entry(page, prefix, doc_id) else {
                continue;
            };
            if self.store.is_none() {
                self.store = Some(
                    self.segment
                        .get_store_reader(STORE_CACHE_BLOCKS)
                        .map_err(|e| anyhow!("pattern: failed to open the document store: {e}"))?,
                );
            }
            tally.store_reads += 1;
            let store = self.store.as_ref().unwrap();
            if self.like.value_matches(store, doc_id)? {
                tally.heap_entrants += 1;
                page.insert(entry);
            }
        }
        *candidates = highest.into_vec();
        Ok(stopped_at)
    }

    /// Walks the segment's `runs` from the end the page needs -- the highest layout keys, or,
    /// for a walk over the complement, the lowest -- as a merge of them by key, a pass at a time,
    /// each pass one pass of the postings.
    ///
    /// A pass has a floor, a key: every run reads its rows from the floor up and no further. The
    /// floor is the highest of the keys the runs would reach by reading a stretch of `width` doc
    /// ids each, so no run reads more than that, the run that sets the floor reads all of it, and
    /// a run whose rows all lie under the floor reads nothing. After the pass every row over the
    /// highest key left unread has been read, whichever run holds it: what the page then holds
    /// over that key is final. So a page costs the runs that overlap at the end it is read from
    /// a stretch each, not every run of the segment. The candidates of a pass are verified after
    /// it, from the highest down.
    ///
    /// A coarse run is read by the block: its stretch ends at a block's edge, and a block is
    /// read when its bounds reach the floor.
    ///
    /// The first stretch is as wide as the page needs at the segment's density of matches
    /// (`matches` of `max_doc`, the postings' own bound), twice over, divided among the runs
    /// whose rows lie among one another at the end the page is read from, so that a page
    /// usually fills in one pass whatever the keyword and however many runs it is read from;
    /// each later one is four times wider. The walk ends
    /// when the highest key left unread can no longer enter the page or the window.
    ///
    /// Or when the budget ends it, between two passes or in the verification after one: it then
    /// returns the highest layout key it has not examined, a run's next row or a candidate left
    /// unverified. What the page holds over that key is final as far as this segment goes.
    fn walk_runs(
        &mut self,
        page: &mut Page,
        tally: &mut WalkTally,
        candidates: &mut Vec<(SortPrefix, u32)>,
        runs: &Runs,
        matches: u32,
        budget: &Spending,
    ) -> anyhow::Result<Option<u128>> {
        let columns = self.columns;
        let window = self.window;
        let forward = window.complement;
        let blocks = &runs.blocks;
        let runs = &runs.runs;
        // The key in walk orientation: the walk always takes the highest first.
        let oriented = |key: u128| if forward { !key } else { key };
        // A run's rows are counted from the end its keys are lowest at: positions, 0 to its
        // length. The first position in `from..to` whose key fails `before`, keys being
        // non-decreasing by position.
        let first_not = |run: &Run, from: u32, to: u32, before: &dyn Fn(u128) -> bool| {
            let (mut low, mut high) = (from, to);
            while low < high {
                let mid = low + (high - low) / 2;
                if before(columns.layout_key(run.doc_at(mid))) {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            }
            low
        };
        // The blocks of a coarse run that hold its positions `from..to`.
        let blocks_of = |first: u32, from: u32, to: u32| {
            &blocks
                [(first + from / BLOCK_DOCS) as usize..(first + to.div_ceil(BLOCK_DOCS)) as usize]
        };
        // The oriented key of the next row of a run whose unread part ends (going down) or
        // starts (going up) at position `next`: for a coarse run, the bound of its next block.
        let best = |run: &Run, next: u32| match (run.blocks, forward) {
            (None, true) => !columns.layout_key(run.doc_at(next)),
            (None, false) => columns.layout_key(run.doc_at(next - 1)),
            (Some(first), true) => !blocks[(first + next / BLOCK_DOCS) as usize].0,
            (Some(first), false) => blocks[(first + (next - 1) / BLOCK_DOCS) as usize].1,
        };
        // Whether the run has rows left to read.
        let unread_at = |run: &Run, next: u32| {
            if forward { next < run.len() } else { next > 0 }
        };
        let top = window.top_layout_key();
        // Per run, the position its unread part ends at (going down) or starts at (going up),
        // and the runs with something unread by the oriented key of their next row. The bounds
        // of a run tell whether the window's top cuts it: only then is it searched.
        let mut next: Vec<u32> = Vec::with_capacity(runs.len());
        let mut unread: Vec<(u128, u32)> = Vec::with_capacity(runs.len());
        for (index, run) in runs.iter().enumerate() {
            let at = match (top, forward) {
                (None, false) => run.len(),
                (None, true) => 0,
                (Some(top), false) if run.max <= top => run.len(),
                (Some(top), false) if run.min > top => 0,
                (Some(top), true) if run.min >= !top => 0,
                (Some(top), true) if run.max < !top => run.len(),
                (Some(top), forward) => match (run.blocks, forward) {
                    (None, false) => first_not(run, 0, run.len(), &|key| key <= top),
                    (None, true) => first_not(run, 0, run.len(), &|key| key < !top),
                    // Up to the first block all over the top, going down; from the first block
                    // reaching it, going up. The rows of a block past the top are left out one
                    // by one, by the window.
                    (Some(first), false) => {
                        let held = blocks_of(first, 0, run.len());
                        let over = held.partition_point(|&(min, _)| min <= top) as u32;
                        (over * BLOCK_DOCS).min(run.len())
                    }
                    (Some(first), true) => {
                        let held = blocks_of(first, 0, run.len());
                        let under = held.partition_point(|&(_, max)| max < !top) as u32;
                        (under * BLOCK_DOCS).min(run.len())
                    }
                },
            };
            next.push(at);
            if unread_at(run, at) {
                unread.push((best(run, at), index as u32));
            }
        }
        // The runs whose rows lie among those of the run with the highest key: the first pass
        // reads a stretch of each. One, where the runs follow one another, as a repair step or
        // rows arriving in the index's order leave them; every run, where each holds a commit
        // of rows that arrived in no order.
        let among = unread
            .iter()
            .max()
            .map(|&(_, index)| {
                let run = &runs[index as usize];
                oriented(if forward { run.max } else { run.min })
            })
            .map_or(1, |lowest| {
                unread.iter().filter(|&&(key, _)| key >= lowest).count()
            })
            .max(1);
        let mut unread = BinaryHeap::from(unread);
        let density = f64::from(matches.max(1)) / f64::from(self.segment.max_doc().max(1));
        // The first stretch: what the page needs at that density, twice over, in all the runs
        // it is read from together.
        let mut width = ((2 * page.limit) as f64 / density.min(1.0) / among as f64)
            .ceil()
            .clamp(1.0, f64::from(u32::MAX)) as u32;
        // The matches a run offers of its stretch, which grow with it.
        let mut take = if walk_off(OFF_TAKE) {
            usize::MAX
        } else {
            2 * page.limit
        };
        // The runs a pass reads to their ends before one of them sets the floor.
        let mut spread = 1usize;
        // Whether the runs holding no match have been told apart.
        let mut swept = walk_off(OFF_SWEEP);
        let mut regions: Vec<Region> = Vec::new();
        let mut reading: Vec<u32> = Vec::new();
        // What a run reads of a stretch of `width`: positions `from..to`, going down or up
        // from `next`, to a block's edge in a coarse run.
        let stretch = |run: &Run, next: u32, width: u32| {
            let edge = |position: u32, up: bool| match run.blocks {
                None => position,
                Some(_) if up => (position.div_ceil(BLOCK_DOCS) * BLOCK_DOCS).min(run.len()),
                Some(_) => position / BLOCK_DOCS * BLOCK_DOCS,
            };
            if forward {
                (next, edge(next.saturating_add(width).min(run.len()), true))
            } else {
                (edge(next.saturating_sub(width), false), next)
            }
        };
        while let Some(&(highest, _)) = unread.peek() {
            // The rows left start at this key: worth a pass only if it can still enter the page
            // and the window.
            if !window.reaches(highest)
                || page
                    .weakest()
                    .is_some_and(|weakest| highest < weakest.prefix.layout_key())
            {
                break;
            }
            if budget.stops_at(highest, tally) {
                return Ok(Some(highest));
            }
            // The runs with rows at the floor or over it, and the floor: the highest key a run
            // among them has left, or ends at, once its stretch is read. A run its stretch
            // reads to the end sets no floor, for `spread` runs a pass less one: so many runs
            // that follow one another are read in one pass.
            regions.clear();
            {
                reading.clear();
                let mut floor = 0u128;
                let mut ended = 1;
                while let Some(&(key, index)) = unread.peek() {
                    if key < floor {
                        break;
                    }
                    unread.pop();
                    let run = &runs[index as usize];
                    let (from, to) = stretch(run, next[index as usize], width);
                    let ends = if forward { to == run.len() } else { from == 0 };
                    let reached = match run.blocks {
                        _ if ends && ended < spread => {
                            ended += 1;
                            0
                        }
                        None => oriented(columns.layout_key(run.doc_at(if forward {
                            to - 1
                        } else {
                            from
                        }))),
                        Some(_) if unread_at(run, if forward { to } else { from }) => {
                            best(run, if forward { to } else { from })
                        }
                        Some(_) => oriented(if forward { run.max } else { run.min }),
                    };
                    floor = floor.max(reached);
                    reading.push(index);
                }
                // Back in the heap once the pass has read.
                for &index in &reading {
                    let run = &runs[index as usize];
                    let (from, to) = stretch(run, next[index as usize], width);
                    // Within the stretch, the rows at the floor or over it, or the blocks
                    // reaching it.
                    let (from, to) = match (run.blocks, forward) {
                        (None, true) => (from, first_not(run, from, to, &|key| !key >= floor)),
                        (None, false) => (first_not(run, from, to, &|key| key < floor), to),
                        (Some(first), true) => {
                            let reach = blocks_of(first, from, to)
                                .partition_point(|&(min, _)| !min >= floor)
                                as u32;
                            (from, (from + reach * BLOCK_DOCS).min(to))
                        }
                        (Some(first), false) => {
                            let held = blocks_of(first, from, to);
                            let under = held.partition_point(|&(_, max)| max < floor) as u32;
                            ((from + under * BLOCK_DOCS).min(to), to)
                        }
                    };
                    let (from, to) = run.docs(from, to);
                    regions.push(Region {
                        from,
                        to,
                        run: Some(index),
                    });
                }
            }
            regions.sort_unstable_by_key(|region| region.from);
            let scanned = tally.postings_scanned;
            probe(0, 1, None);
            let clock = cfg!(test).then(Instant::now);
            self.scan(page, tally, candidates, &mut regions, take)?;
            probe(3, 0, clock);
            // A region comes back cut to what was read of it; one that reached no floor, as it
            // was.
            for region in &regions {
                let index = region.run.unwrap();
                let run = &runs[index as usize];
                let (from, to) = run.positions(region.from, region.to);
                let at = if forward { to } else { from };
                next[index as usize] = at;
                if unread_at(run, at) {
                    unread.push((best(run, at), index));
                }
            }
            // A pass that met no match: the matches lie apart from the end the page is read
            // from, and the runs between, read a stretch a pass, are as many passes. One pass of
            // the postings over the runs left, in doc order, tells the ones that hold a match
            // from the ones that hold none, which are left out from here on.
            let clock = cfg!(test).then(Instant::now);
            if !swept && tally.postings_scanned == scanned && !unread.is_empty() {
                swept = true;
                self.scorer = None;
                let mut left: Vec<u32> = unread.drain().map(|(_, index)| index).collect();
                left.sort_unstable();
                if let Some(mut scorer) = self.like.candidates(self.segment)? {
                    for index in left {
                        let run = &runs[index as usize];
                        let at = next[index as usize];
                        let (from, to) = if forward {
                            run.docs(at, run.len())
                        } else {
                            run.docs(0, at)
                        };
                        let doc_id = if scorer.doc() < from {
                            scorer.seek(from)
                        } else {
                            scorer.doc()
                        };
                        if doc_id < to {
                            unread.push((best(run, at), index));
                        }
                    }
                }
            }
            probe(5, 0, clock);
            if unread.is_empty() {
                break;
            }
            let left = unread.peek().map_or(0, |&(key, _)| key);
            let clock = cfg!(test).then(Instant::now);
            let stopped = self.verify(page, tally, candidates, budget, left)?;
            probe(4, 0, clock);
            if let Some(unexamined) = stopped {
                return Ok(Some(unexamined));
            }
            width = width.saturating_mul(4);
            // A keyword in few rows needs many runs for a page, and a pass costs a reader of
            // the postings: more runs after a pass that met matches, and fewer than a page
            // of them. One that met none says nothing of the runs to come, which the
            // sweep has told apart; one that met a page's worth is of a keyword in many rows,
            // whose runs are worth a pass each.
            // As many times the runs as the page is short by at that rate, four at most.
            let found = (tally.postings_scanned - scanned) as usize;
            if !walk_off(OFF_SPREAD) && found > 0 {
                let short = page.limit.saturating_sub(found) / found;
                spread = spread.saturating_mul(short.clamp(1, 4));
            }
            // More of each run only once the matches of this many were not enough.
            if tally.postings_scanned > scanned {
                take = take.saturating_mul(4);
            }
        }
        let left = unread.peek().map_or(0, |&(key, _)| key);
        self.verify(page, tally, candidates, budget, left)
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
/// A search has a budget (`SearchBudget`). When it ends the walk in a segment, the page is cut at
/// the highest layout key that segment has not examined -- a run's next row, or a candidate left
/// unverified. The walk reads a segment at a time, so the segments after it may hold rows over
/// that key too, where segments overlap: each is then read over the cut only, with a share of
/// half the budget, and a stop there raises the cut. When it ends between two segments, the ones
/// left are read the same way, and the page is cut only if one of them stops. The rows over the cut are final, and the cursor
/// is the cut, a key with no row, which takes every row at it again. Such a page may hold fewer
/// rows than `limit`, or none, and still not be the last.
///
/// Returns the page's primary ids, and the cursor of its last row when the page filled, or the
/// key it was cut at.
type OrderedPage = (Vec<PrimaryId>, Option<Cursor>);

fn collect_matches_ordered(
    state: &PatternIndexState,
    like: &mut LikeQuery,
    limit: usize,
    window: SortWindow,
    key_of: &dyn Fn(PrimaryId) -> Option<PrimaryKey>,
    budget: SearchBudget,
) -> anyhow::Result<OrderedPage> {
    let searcher = state.reader.searcher();

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
            segments.push((
                segment_ord as u32,
                segment,
                columns,
                upper_bound,
                lower_bound,
            ));
        }
    }
    state.backend().prune_columns(searcher.segment_readers());
    segments.sort_by_key(|(_, _, _, upper_bound, _)| Reverse(*upper_bound));
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
    let bounds: Vec<u128> = segments.iter().map(|&(_, _, _, bound, _)| bound).collect();
    // Where the first segment's rows lie among another's, the walk may have to cut in more than
    // one: the first then has half the budget, and the others the other half between them.
    // Left to spend it all, the first would read far down, and the others, with what is left,
    // hardly under the cursor: the cut would be theirs, and the first's reading lost.
    let overlapping = segments
        .first()
        .zip(bounds.get(1))
        .is_some_and(|(&(_, _, _, _, lower), &second)| second >= lower);
    let whole = Spending::new(budget, &window, if overlapping { 2 } else { 1 });
    // Once the budget has ended the walk, the segments left are read with a share of it each,
    // and over the cut only: the highest oriented layout key the walk has not examined, when
    // the budget ended it inside a segment.
    let mut spent = false;
    let mut cut: Option<u128> = None;
    let mut share = 1;
    for (at, (segment_ord, segment, columns, upper_bound, _)) in segments.into_iter().enumerate() {
        if let Some(weakest) = page.weakest()
            && upper_bound < weakest.prefix.layout_key()
        {
            // Nothing in this segment, nor in any later one, can enter the page. (A segment
            // ending exactly at the weakest layout key may still hold a row above it.)
            break;
        }
        // Before the budget is looked at: a segment with no candidate costs a look in its
        // dictionary, and the clock costs as much. A pattern nothing matches met the clock at
        // each of its segments, a fifth of its walk over 70 of them.
        let Some(scorer) = like.candidates(segment)? else {
            continue;
        };
        tally.segments_opened += 1;
        if scorer.doc() == TERMINATED {
            // Nothing matches here: no run to look for an end in.
            continue;
        }
        if !spent && whole.is_spent(&tally) {
            // Between two segments. What the page holds is not final while a segment left may
            // hold a row over it, and to cut over them all would give up the page.
            spent = true;
            share = bounds.len() - at;
        }
        let over_cut;
        let shared;
        let (window, budget) = if spent {
            let window = match cut {
                // The segments come by descending upper bound: none after this one reaches
                // over the cut either.
                Some(cut) if upper_bound <= cut => break,
                Some(cut) => match window.above(cut) {
                    Some(above) => {
                        over_cut = above;
                        &over_cut
                    }
                    None => break,
                },
                None => &window,
            };
            shared = whole.share(share, &tally);
            (window, &shared)
        } else {
            (&window, &whole)
        };
        let matches_bound = like.matches_bound(scorer.as_ref());
        candidates.clear();
        let mut walk = RegionWalk {
            segment,
            segment_ord,
            columns: &columns,
            window,
            ties: &ties,
            key_of,
            like: &mut *like,
            store: None,
            scorer: Some(scorer),
            before: (tally.postings_scanned, tally.store_reads),
        };
        // The runs cost the walk a look at each it reads together and a pass or more of the
        // postings; the segment read whole costs it every match, in one pass. So few matches
        // are read whole.
        let runs = columns.runs.clone().filter(|runs| {
            !walk_off(FORCE_WHOLE_READ)
                && (walk_off(OFF_WHOLE_READ)
                    || matches_bound as usize
                        > WHOLE_READ_MATCHES
                            + WHOLE_READ_PAGES * page.limit
                            + WHOLE_READ_PER_RUN * runs.overlap())
        });
        let left = match runs {
            Some(runs) => walk.walk_runs(
                &mut page,
                &mut tally,
                &mut candidates,
                &runs,
                matches_bound,
                budget,
            )?,
            None => {
                #[cfg(test)]
                if columns.runs.is_none() {
                    SEGMENTS_READ_WHOLE.fetch_add(1, Relaxed);
                }
                walk.scan(
                    &mut page,
                    &mut tally,
                    &mut candidates,
                    &mut [Region {
                        from: 0,
                        to: segment.max_doc(),
                        run: None,
                    }],
                    usize::MAX,
                )?;
                walk.verify(&mut page, &mut tally, &mut candidates, budget, 0)?
            }
        };
        if let Some(left) = left {
            if !spent {
                spent = true;
                share = bounds[at + 1..]
                    .iter()
                    .filter(|&&bound| bound > left)
                    .count();
            }
            // Read over the cut, a segment leaves a key over it.
            cut = Some(cut.map_or(left, |cut| cut.max(left)));
        }
    }
    let unexamined = cut;
    // A full page whose last row is over everything unexamined is settled all the same.
    let cut_at = unexamined.filter(|&unexamined| {
        page.weakest()
            .is_none_or(|weakest| unexamined >= weakest.prefix.layout_key())
    });
    if let Some(cut_at) = cut_at {
        page.entries
            .retain(|entry| entry.prefix.layout_key() > cut_at);
        tally.cut_short = 1;
    }

    // A cursor says there may be more below this point. That is only true if the page filled: a
    // walk that ended with fewer than `limit` matches visited every segment overlapping the window,
    // so there is nothing left to resume from and a cursor would only cost the caller an empty
    // round trip. (Rows dropped later, in `handle_search`, shorten the page after this point and
    // must not suppress the cursor -- which is why this asks the page, not the returned rows.)
    let needed: HashSet<u32> = page.entries.iter().map(|entry| entry.segment).collect();
    tally.segments_wasted = tally.segments_opened - needed.len() as u64;
    let resolving = Instant::now();
    let resume_at = match cut_at {
        Some(cut_at) => {
            // Every row at the key or under it, whatever its later slots.
            let mut prefix = SortPrefix([u64::MAX; PREFIX_SLOTS]);
            prefix.0[0] = (cut_at >> 64) as u64;
            prefix.0[1] = cut_at as u64;
            Some(Cursor {
                prefix: window.unorient(prefix),
                primary_key: None,
            })
        }
        None => page.weakest().map(|weakest| Cursor {
            prefix: window.unorient(weakest.prefix),
            primary_key: key_of(weakest.primary_id.into()),
        }),
    };
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
    Ok((ids, resume_at))
}

fn handle_pattern_stats(state: &PatternIndexState) -> PatternStatsR {
    let searcher = state.reader.searcher();
    // Segment by segment, as a merge deletes the files of the segments it replaced while a
    // searcher from before the reload still lists them, and their size can no longer be read.
    // Such a segment counts for nothing until the reload brings the one that replaced it.
    let size_bytes = searcher
        .segment_readers()
        .iter()
        .filter_map(|segment| segment.space_usage().ok())
        .map(|usage| usage.total().get_bytes())
        .sum();
    let tantivy = TantivyStats {
        num_docs: searcher.num_docs(),
        size_bytes,
        segment_count: searcher.segment_readers().len(),
    };
    let segments = searcher
        .segment_readers()
        .iter()
        .map(|segment| {
            let mut opened = 0;
            let columns = state.backend().columns_for(segment, &mut opened)?;
            // The first slot: the first ordering column's sort key, or the token.
            let bounds = Some((
                (columns.layout_min >> 64) as u64,
                (columns.layout_max >> 64) as u64,
            ));
            Ok(SegmentLayout {
                docs: segment.num_docs(),
                deleted: segment.num_deleted_docs(),
                sort_min: bounds.map(|(min, _)| min),
                sort_max: bounds.map(|(_, max)| max),
            })
        })
        .collect::<anyhow::Result<_>>()?;
    let mut overlap = layout_overlap(&state.backend().geometry(&searcher));
    overlap.rows_deleted = searcher
        .segment_readers()
        .iter()
        .map(|segment| u64::from(segment.num_deleted_docs()))
        .sum();
    Ok(PatternStats {
        tantivy,
        walk: state.backend().walk.snapshot(),
        rewrite: state.backend().rewrite_totals(),
        merges: state.backend().merging.lock().unwrap().totals(),
        overlap,
        segments,
    })
}

/// Returns a page of the keys of the rows whose value matches `pattern`, in table order.
fn handle_like(
    state: &PatternIndexState,
    table: &RwLock<impl TableSearch>,
    index_key: &IndexKey,
    pattern: String,
    limit: Limit,
    window: SearchWindow,
    budget: SearchBudget,
) -> PatternLikeR {
    let text_field = state.schema.get_field(TEXT_FIELD).unwrap();
    let pattern = state.backend().options.case_sensitive.fold(pattern);
    let backend = state.backend();
    let mut like = LikeQuery::new(
        &state.schema,
        text_field,
        &pattern,
        backend.min_gram(),
        backend.max_gram(),
        backend.rarest_grams(),
    )?;
    let limit: usize = (*limit.as_ref()).into();
    let partition_id = find_partition_id(table.read().unwrap().deref(), index_key)?;
    let (primary_ids, next_cursor) = {
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
        collect_matches_ordered(state, &mut like, limit, sort_window, &key_of, budget)?
    };

    let table = table.read().unwrap();
    Ok(PatternPage {
        primary_keys: primary_ids
            .into_iter()
            .filter_map(|primary_id| table.primary_key(partition_id, primary_id))
            .collect(),
        next_cursor,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn new(
    index: PatternIndexConfiguration,
    table: Arc<RwLock<impl TableSearch + Send + Sync + 'static>>,
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
    commit_interval: Duration,
    commit_threshold: usize,
    metrics: Option<Arc<Metrics>>,
    search_budget: impl Into<SearchBudgetSource>,
) -> mpsc::Sender<PatternIndex> {
    let search_budget = search_budget.into();
    let (tx, mut rx) = mpsc::channel::<PatternIndex>(perf::channel_size().into());
    tokio::spawn(async move {
        let key = index.key.clone();
        // The layout changes on the ticks -- merges after the writes stop, a rewrite in progress
        // -- with no write or search to mark the index for a refresh, so the ticks mark it.
        let mark_dirty = |key: &IndexKey| {
            if let Some(metrics) = &metrics {
                metrics.mark_dirty(key);
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
                            if prefix.is_none() {
                                // A null ordering value leaves the row out, as a view leaves out
                                // a row with a null key column. An update that set it to null
                                // already removed the old document (RemoveBeforeAddValue).
                                debug!(
                                    "pattern: {key}: row {primary_id:?} has no sort prefix, not indexed"
                                );
                                continue;
                            }
                            let scanned = in_progress.is_fullscan();
                            let lag = in_progress.lag();
                            indexes
                                .add_document_with(
                                    primary_id,
                                    in_progress,
                                    prefix
                                        .filter(|_| !walk_off(OFF_SORTED_COMMITS))
                                        .map(|prefix| Box::from(prefix.0)),
                                    move |index| {
                                    if scanned {
                                        index.backend().scan_rows.store(true, Relaxed);
                                    }
                                    if let Some(lag) = lag {
                                        index.backend().note_cdc_lag(lag);
                                    }
                                    index.backend().create_doc(
                                        &index.schema,
                                        primary_id,
                                        PatternRow {
                                            text: &document,
                                            prefix,
                                            repaired: false,
                                        },
                                    )
                                    },
                                )
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
                        PatternIndex::Like {
                            index_key,
                            pattern,
                            limit,
                            window,
                            tx,
                        } => {
                            let Some(state) = indexes.get(&index_key) else {
                                _ = tx.send(Ok(PatternPage {
                                    primary_keys: vec![],
                                    next_cursor: None,
                                }));
                                continue;
                            };
                            let table = Arc::clone(&table);
                            let search_budget = (search_budget.0)();
                            worker
                                .spawn_blocking(move || {
                                    let result = handle_like(
                                        &state,
                                        table.as_ref(),
                                        &index_key,
                                        pattern,
                                        limit,
                                        window,
                                        search_budget,
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
                    // An index with nothing to commit is idle: the repair runs freely then, a step
                    // per tick at most, so it paces itself to the ticks. While it ingests the
                    // repair holds back (see `maybe_repair`). One whose writer is held right now
                    // is not idle.
                    let idle: Vec<bool> = indexes
                        .iter()
                        .map(|index| index.has_uncommitted_docs() == Some(false))
                        .collect();
                    indexes.commit_or_reload().await;
                    for (state, idle) in indexes.iter().zip(idle) {
                        // A repair step at 10M rows takes seconds under the writer lock.
                        // Run on the worker, it took a runtime thread, and the steps the next ticks
                        // queued behind it took the others, waiting for the lock, until the node
                        // stopped answering HTTP (AWS runs eb0023ac and 1c3ae034). So the work runs
                        // on tokio's blocking pool, and a tick adds none while an index's previous
                        // work still runs.
                        let Some(work) = TickWork::start(state) else {
                            continue;
                        };
                        let key = key.clone();
                        let worker = worker.clone();
                        tokio::task::spawn_blocking(move || {
                            let state = &work.0;
                            let backend = state.backend();
                            backend.after_reload(state);
                            backend.maybe_repair(state, &key, idle, &worker);
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
    use crate::CaseSensitive;
    use crate::DEFAULT_SEGMENT_ROWS;
    use crate::MaxGram;
    use crate::MinGram;
    use crate::OrderBy;
    use crate::PrimaryKey;
    use crate::QueryError;
    use crate::RarestGrams;
    use crate::RewriteWideSegments;
    use crate::SegmentMaxDocs;
    use crate::memory::Allocate;
    use crate::table::IndexIdGenerator;
    use crate::table::MockTableSearch;
    use crate::table::PartitionId;
    use crate::tantivy::test_support::make_index_key;
    use crate::tantivy::test_support::make_memory_actor;
    use crate::worker;
    use rstest::rstest;
    use scylla::value::CqlValue;
    use std::cmp::Reverse;
    use std::num::NonZeroUsize;

    use super::super::actor::PatternIndexExt;

    /// The partition every test document lives in: these tests use a global index, so there is
    /// only one, and it has to match what `make_table_with_keys` hands back.
    fn test_partition_id() -> PartitionId {
        PartitionId::global(IndexIdGenerator::new().next(true).unwrap())
    }

    /// The shared fixture's table, with a value for every column: an ordering column mirrors the
    /// primary id, so a test can assert on plain integers.
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
        mock.expect_column_value_for()
            .returning(|_partition_id, primary_id, _column| {
                Some(CqlValue::BigInt(u64::from(primary_id) as i64))
            });
        Arc::new(RwLock::new(mock))
    }

    const TEST_COMMIT_INTERVAL: Duration = Duration::from_millis(50);
    const TEST_COMMIT_THRESHOLD: usize = 100;

    fn options(min_gram: usize, max_gram: usize, case_sensitive: bool) -> IndexOptionsPattern {
        IndexOptionsPattern {
            min_gram: MinGram::from(NonZeroUsize::new(min_gram).unwrap()),
            max_gram: MaxGram::from(NonZeroUsize::new(max_gram).unwrap()),
            case_sensitive: CaseSensitive::from(case_sensitive),
            // The matching tests run in table order and sort what they get; the order has tests
            // of its own.
            order_by: OrderBy::Table,
            partition_order: crate::PartitionOrder::default(),
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

    /// The fixtures' tables: a single bigint partition key, no clustering columns.
    fn test_layout() -> KeyLayout {
        KeyLayout {
            partition_key_count: 1,
            clustering_descending: Arc::new([]),
        }
    }

    /// The place of the partition of the row whose key is `id`, as an index with the default
    /// `partition_order` keeps it.
    fn token_of(id: u64) -> u64 {
        partition_place(id, crate::PartitionOrder::default())
    }

    /// The same for the given `partition_order`.
    fn partition_place(id: u64, order: crate::PartitionOrder) -> u64 {
        test_layout()
            .partition_order_key(&PrimaryKey::from(vec![CqlValue::BigInt(id as i64)]), order)
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
            make_memory_actor(Allocate::Can),
            TEST_COMMIT_INTERVAL,
            commit_threshold,
            None,
            SearchBudget::default(),
        )
    }

    fn make_sender_with_memory(allocate: Allocate) -> mpsc::Sender<PatternIndex> {
        new(
            PatternIndexConfiguration {
                key: make_index_key(),
                options: IndexOptionsPattern::default(),
                key_layout: test_layout(),
            },
            make_table_with_keys(),
            worker::new(),
            make_memory_actor(allocate),
            TEST_COMMIT_INTERVAL,
            TEST_COMMIT_THRESHOLD,
            None,
            SearchBudget::default(),
        )
    }

    /// The defaults: case-sensitive.
    fn make_sender() -> mpsc::Sender<PatternIndex> {
        make_sender_with_options(IndexOptionsPattern::default())
    }

    fn make_case_insensitive_sender() -> mpsc::Sender<PatternIndex> {
        make_sender_with_options(options(1, 3, false))
    }

    async fn add_doc(sender: &mpsc::Sender<PatternIndex>, primary: u64, content: &str) {
        let (tx, mut rx) = mpsc::channel(1);
        sender
            .add_document(
                test_partition_id(),
                primary.into(),
                content.into(),
                AsyncInProgress::fullscan(tx),
            )
            .await
            .unwrap();
        rx.recv().await;
    }

    async fn rm_doc_no_wait(sender: &mpsc::Sender<PatternIndex>, primary: u64) {
        let (tx, _rx) = mpsc::channel(1);
        sender
            .remove_document(primary.into(), AsyncInProgress::fullscan(tx))
            .await
            .unwrap();
    }

    async fn rm_doc(sender: &mpsc::Sender<PatternIndex>, primary: u64) {
        let (tx, mut rx) = mpsc::channel(1);
        sender
            .remove_document(primary.into(), AsyncInProgress::fullscan(tx))
            .await
            .unwrap();
        rx.recv().await;
    }

    async fn count(sender: &mpsc::Sender<PatternIndex>) -> usize {
        sender.count(make_index_key()).await.unwrap()
    }

    /// Adds `docs` without waiting between them, then waits until all of them are searchable.
    async fn add_docs(sender: &mpsc::Sender<PatternIndex>, docs: &[(u64, &str)]) {
        let mut receivers = Vec::new();
        for &(primary, content) in docs {
            let (tx, rx) = mpsc::channel(1);
            sender
                .add_document(
                    test_partition_id(),
                    primary.into(),
                    content.into(),
                    AsyncInProgress::fullscan(tx),
                )
                .await
                .unwrap();
            receivers.push(rx);
        }
        for mut rx in receivers {
            rx.recv().await;
        }
    }

    fn limit(n: usize) -> Limit {
        Limit::from(NonZeroUsize::new(n).unwrap())
    }

    /// The pattern of the values that contain `keyword`.
    fn containing(keyword: &str) -> String {
        let escaped: String = keyword
            .chars()
            .flat_map(|c| {
                matches!(c, '\\' | '%' | '_')
                    .then_some('\\')
                    .into_iter()
                    .chain([c])
            })
            .collect();
        format!("%{escaped}%")
    }

    fn ids_of(primary_keys: &[PrimaryKey]) -> Vec<i64> {
        primary_keys
            .iter()
            .map(|key| match key.get(0).unwrap() {
                CqlValue::BigInt(id) => id,
                other => panic!("unexpected primary key value {other:?}"),
            })
            .collect()
    }

    /// One page of the rows matching `pattern`, with the cursor to the next.
    async fn like(
        sender: &mpsc::Sender<PatternIndex>,
        pattern: &str,
        limit_n: usize,
        window: SearchWindow,
    ) -> PatternPage {
        sender
            .like(make_index_key(), pattern.into(), limit(limit_n), window)
            .await
            .unwrap()
    }

    /// The sorted ids of the rows matching `pattern`, for tests that do not look at the order.
    async fn like_ids(sender: &mpsc::Sender<PatternIndex>, pattern: &str) -> Vec<i64> {
        let page = like(sender, pattern, 100, SearchWindow::default()).await;
        let mut ids = ids_of(&page.primary_keys);
        ids.sort_unstable();
        ids
    }

    /// The primary ids containing `query`, sorted, so tests are independent of index order.
    async fn search(sender: &mpsc::Sender<PatternIndex>, query: &str) -> Vec<i64> {
        search_page(sender, query, 100).await
    }

    /// Like `search_page`, but keeps the order the index returned -- the sorting helpers above
    /// cannot tell a correctly ordered answer from a scrambled one.
    async fn search_ordered(
        sender: &mpsc::Sender<PatternIndex>,
        query: &str,
        limit_n: usize,
        window: SearchWindow,
    ) -> (Vec<i64>, Option<Cursor>) {
        let page = like(sender, &containing(query), limit_n, window).await;
        (ids_of(&page.primary_keys), page.next_cursor)
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
    ) -> Vec<i64> {
        let page = like(sender, &containing(query), limit_n, SearchWindow::default()).await;
        let mut ids = ids_of(&page.primary_keys);
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

    /// Intersecting only the rarest grams lets more candidates through, and the verification drops
    /// them again: the matches are those of the whole intersection, in table order or by a column.
    #[rstest]
    #[case::one_table(1, false)]
    #[case::two_table(2, false)]
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
    async fn digit_queries_match_by_containment() {
        let sender = make_sender();
        add_docs(&sender, USERNAMES).await;

        assert_eq!(search(&sender, "925555").await, vec![4, 5]);
        assert_eq!(search(&sender, "925").await, vec![4, 5, 6]);
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

        assert_eq!(search_page(&sender, "abcde", 1).await, vec![21]);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn query_shorter_than_min_gram_is_a_query_error() {
        let sender = make_sender_with_options(options(2, 3, true));
        add_docs(&sender, NICKNAMES).await;

        let err = sender
            .like(
                make_index_key(),
                "%宫%".into(),
                limit(10),
                SearchWindow::default(),
            )
            .await
            .expect_err("a one-character query cannot be answered by a min_gram=2 index");

        assert!(err.downcast_ref::<QueryError>().is_some(), "got: {err}");
        // A query at min_gram is still served, and so is a shorter one tied to an end of the
        // value, whose mark makes the gram long enough.
        assert_eq!(search(&sender, "南宫").await, vec![6, 7]);
        assert_eq!(like_ids(&sender, "宫%").await, vec![8]);
        // A run too short is no hindrance beside one long enough.
        assert_eq!(like_ids(&sender, "%南宫%团").await, vec![7]);
        // A pattern without a literal character asks for no gram.
        assert_eq!(like_ids(&sender, "__").await, vec![4, 5, 8]);
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

    /// An index with a sort column returns the highest sort keys first. The matching helpers sort
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
        let overlap = before.overlap;
        assert_eq!(overlap.rows, before.tantivy.num_docs);
        assert!(overlap.depth_max >= 1, "{overlap:?}");
        assert!(overlap.depth_rows >= overlap.rows, "{overlap:?}");

        search_ordered(&sender, "将军", 2, SearchWindow::default()).await;

        let after = sender.stats(make_index_key()).await.unwrap();
        let walk = after.walk;
        assert_eq!(walk.searches, 1);
        assert!(walk.segments_opened >= 1 && walk.segments_opened <= walk.segments_considered);
        // The page is not empty, so at least one segment opened was needed.
        assert!(walk.segments_wasted < walk.segments_opened, "{walk:?}");
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

    /// A merge held back by a test: it ends, producing `Some(segment)` or nothing, when the
    /// sender sends, and fails when the sender is dropped.
    impl MergeOutput for Result<Option<SegmentId>, futures::channel::oneshot::Canceled> {
        fn segment(&self) -> Option<SegmentId> {
            *self.as_ref().ok()?
        }
    }

    /// A merge that outlasts a tick keeps its segments out of the next tick's merges: asked for
    /// them again, Tantivy would queue a second full merge (run a0e51aa9 queued each one up to
    /// 60 times a minute).
    #[test]
    fn segments_being_merged_are_not_merged_again() {
        let mut merging = MergesInFlight::default();
        let mut adopted = HashSet::new();
        let (finish, merge) = futures::channel::oneshot::channel::<Option<SegmentId>>();
        merging.started(vec![id(1), id(2)], false, merge);
        let (_still_running, other) = futures::channel::oneshot::channel();
        merging.started(vec![id(3)], false, other);
        assert_eq!(merging.busy(&mut adopted), [id(1), id(2), id(3)].into());
        // Still running at the next tick: still busy.
        assert_eq!(merging.busy(&mut adopted), [id(1), id(2), id(3)].into());
        finish.send(Some(id(4))).unwrap();
        assert_eq!(merging.busy(&mut adopted), [id(3)].into());
        assert_eq!(
            merging.totals(),
            MergeTotals {
                started: 2,
                ended: 1,
                running: 1
            }
        );
        // A merge of fresh segments makes a fresh one.
        assert!(adopted.is_empty());
    }

    /// A merge that failed frees its segments as one that succeeded does, so that they can be
    /// merged on a later tick rather than never again.
    #[test]
    fn the_segments_of_a_failed_merge_can_be_merged_again() {
        let mut merging = MergesInFlight::default();
        let (failed, merge) = futures::channel::oneshot::channel::<Option<SegmentId>>();
        merging.started(vec![id(1), id(2)], true, merge);
        drop(failed);
        let mut adopted = HashSet::new();
        assert!(merging.busy(&mut adopted).is_empty());
        assert!(adopted.is_empty());
    }

    /// What a merge of ordered segments produced is adopted when it reports its end; until then,
    /// a searcher that no longer holds the merged segments holds its output, which the layout
    /// cannot place yet.
    #[test]
    fn a_merge_of_ordered_segments_is_ordered_once_it_ends() {
        let mut merging = MergesInFlight::default();
        let mut adopted = HashSet::new();
        let (finish, merge) = futures::channel::oneshot::channel::<Option<SegmentId>>();
        merging.started(vec![id(1), id(2)], true, merge);
        assert!(!merging.unsettled(&[id(1), id(2), id(5)].into()));
        assert!(merging.unsettled(&[id(3), id(5)].into()));
        finish.send(Some(id(3))).unwrap();
        merging.busy(&mut adopted);
        assert_eq!(adopted, [id(3)].into());
        assert!(!merging.unsettled(&[id(3), id(5)].into()));
    }

    /// A fresh segment of `docs` rows over `min..=max`: ingested, and adopted by nobody yet.
    fn piece(docs: u32, min: u128, max: u128, n: u8) -> Piece {
        Piece {
            docs,
            deleted: 0,
            min,
            max,
            id: id(n),
            ordered: false,
        }
    }

    /// `piece`, placed by a repair or adopted.
    fn ordered(docs: u32, min: u128, max: u128, n: u8) -> Piece {
        Piece {
            ordered: true,
            ..piece(docs, min, max, n)
        }
    }

    /// Disjoint segments sit at depth 1, and neighbours sharing a boundary key at 2, within the
    /// tolerance; the depth counts the segments over the deepest key of a segment's span, so a
    /// wide one over narrow ones deepens them all, and a second wide one takes them over it.
    #[test]
    fn layout_overlap_counts_the_segments_over_each_key() {
        let disjoint = [
            piece(100, 0, 9, 1),
            piece(100, 10, 19, 2),
            piece(100, 20, 29, 3),
        ];
        assert_eq!(
            layout_overlap(&disjoint),
            LayoutOverlap {
                depth_max: 1,
                depth_rows: 300,
                rows: 300,
                ..LayoutOverlap::default()
            }
        );

        // A layout key ties the sort value with the second slot, so one value filling several
        // segments still leaves them disjoint; three meeting at one key is past the tolerance.
        let shared = [
            piece(100, 0, 10, 1),
            piece(100, 10, 10, 2),
            piece(100, 10, 29, 3),
        ];
        let overlap = layout_overlap(&shared);
        assert_eq!((overlap.depth_max, overlap.rows_over_tolerance), (3, 300));
        let neighbours = [piece(100, 0, 10, 1), piece(100, 10, 29, 2)];
        let overlap = layout_overlap(&neighbours);
        assert_eq!((overlap.depth_max, overlap.rows_over_tolerance), (2, 0));

        let mut covered = disjoint.to_vec();
        covered.push(piece(50, 0, 29, 4));
        let overlap = layout_overlap(&covered);
        assert_eq!(
            (
                overlap.depth_max,
                overlap.depth_rows,
                overlap.rows_over_tolerance
            ),
            (2, 2 * 350, 0)
        );
        // Each overlapping one, none of them known to be ordered: all fresh.
        assert_eq!((overlap.fresh_rows, overlap.fresh_segments), (350, 4));
        covered.push(piece(50, 5, 25, 5));
        let overlap = layout_overlap(&covered);
        assert_eq!(
            (
                overlap.depth_max,
                overlap.depth_rows,
                overlap.rows_over_tolerance
            ),
            (3, 3 * 400, 400)
        );

        // An empty segment covers nothing.
        let emptied = [piece(100, 0, 9, 1), piece(0, 0, 9, 2)];
        assert_eq!(layout_overlap(&emptied).depth_max, 1);
        assert_eq!(layout_overlap(&[]), LayoutOverlap::default());
    }

    /// The ordered tier: what a tag says is, and what overlaps nothing; an empty segment is in
    /// neither tier and keeps nothing out of the ordered one.
    #[test]
    fn a_segment_overlapping_nothing_is_ordered() {
        let pieces = [
            piece(10, 0, 9, 1),     // alone: ordered
            piece(0, 5, 50, 2),     // empty: overlaps nothing
            piece(10, 20, 40, 3),   // overlaps 4: fresh
            piece(10, 30, 60, 4),   // overlaps 3: fresh
            ordered(10, 35, 45, 5), // overlaps both, but tagged
            piece(10, 61, 70, 6),   // touches nothing: ordered
            piece(10, 70, 80, 7),   // shares key 70 with 6...
        ];
        assert_eq!(
            ordered_tier(&pieces),
            [true, false, false, false, true, false, false]
        );
        assert_eq!(
            ordered_tier(&pieces[..6]),
            [true, false, false, false, true, true]
        );
    }

    /// Ordered clusters merge whole, with their neighbours while the run stays within three
    /// quarters of the cap, and alone up to the cap; one past the cap is left to the repair.
    /// Fresh segments merge among themselves under the cap, whatever they span; the tiers never
    /// meet in a merge.
    #[test]
    fn merges_keep_to_one_tier_and_take_clusters_whole() {
        let cap = 100;
        let pieces = [
            ordered(30, 0, 9, 1),
            ordered(30, 10, 19, 2),
            ordered(30, 20, 29, 3), // 1 + 2 = 60, + 3 = 90 > 75
            ordered(40, 30, 39, 4), // with 3: 70
            ordered(60, 40, 49, 5), // a cluster with 6: 90, alone
            ordered(30, 45, 49, 6),
            ordered(80, 50, 59, 7), // a cluster past the cap with 8
            ordered(30, 52, 55, 8),
            ordered(40, 60, 69, 9),       // alone, and no neighbour fits
            piece(40, 200, 300, 10),      // fresh, past all of the above
            piece(50, 205, 300, 11),      // + 10 = 90
            piece(30, 210, 300, 12),      // 90 + 30 > 100: a run of its own
            piece(cap + 5, 220, 300, 13), // full: left out
            piece(5, 230, 300, 14),       // with 12: 35
        ];
        assert_eq!(
            merge_runs(&pieces, cap),
            vec![
                (vec![id(1), id(2)], true),
                (vec![id(3), id(4)], true),
                (vec![id(5), id(6)], true),
                (vec![id(10), id(11)], false),
                (vec![id(12), id(14)], false),
            ]
        );
        assert!(merge_runs(&[], cap).is_empty());
    }

    /// Ordered neighbours are not merged while a fresh piece reaches between them: the merged
    /// segment would span the fresh rows there, and each repair step among them would split it
    /// again. A row committed alone at one end of the order and the first stretch repaired at the
    /// other were merged so in a 10M-row load, which then moved 15M rows for 10M. A cluster
    /// still merges whole under fresh pieces, and neighbours with none between them do.
    #[test]
    fn ordered_neighbours_are_not_merged_across_fresh_pieces() {
        let far_apart = [
            ordered(30, 0, 9, 1),
            ordered(1, 90, 90, 2),
            piece(40, 5, 60, 3),
            piece(40, 8, 70, 4),
        ];
        assert_eq!(
            merge_runs(&far_apart, 100),
            vec![(vec![id(3), id(4)], false)]
        );
        // The two pieces of a repaired stretch under the same fresh pieces, and a neighbour
        // the fresh pieces end before.
        let together = [
            ordered(30, 0, 9, 1),
            ordered(20, 5, 9, 2),
            ordered(10, 80, 89, 3),
            ordered(10, 90, 99, 4),
            piece(40, 5, 60, 5),
            piece(40, 8, 70, 6),
        ];
        assert_eq!(
            merge_runs(&together, 100),
            vec![
                (vec![id(1), id(2)], true),
                (vec![id(3), id(4)], true),
                (vec![id(5), id(6)], false),
            ]
        );
    }

    /// A segment whose deleted rows outnumber its live ones is merged on its own, which drops
    /// them; one with fewer is left.
    #[test]
    fn a_segment_mostly_deleted_is_merged_alone() {
        let hollow = Piece {
            deleted: 60,
            ..ordered(40, 0, 9, 1)
        };
        let worn = Piece {
            deleted: 10,
            ..ordered(70, 20, 29, 2)
        };
        // 40 + 70 > 75: no run with both.
        assert_eq!(merge_runs(&[hollow, worn], 100), vec![(vec![id(1)], true)]);
    }

    /// The repair takes the stretch with the most fresh rows first, a cap's worth at a time from
    /// its lowest keys; with the ordered tier empty, the whole range is one gap.
    #[test]
    fn the_repair_takes_the_worst_stretch_first() {
        let cap = 100;
        assert_eq!(plan_repair(&[], &[], 1, cap), None);
        // Nothing fresh, nothing past the cap: nothing to do.
        let settled = [ordered(50, 0, 99, 1), ordered(50, 100, 199, 2)];
        assert_eq!(plan_repair(&settled, &[], 1, cap), None);

        // Everything fresh: one gap; 300 sampled keys a row each, so the first 100.
        let fresh = [piece(150, 0, 299, 1), piece(150, 0, 299, 2)];
        let keys: Vec<u128> = (0..300).collect();
        assert_eq!(
            plan_repair(&fresh, &keys, 1, cap),
            Some(RepairRange {
                low: 0,
                high: 99,
                cluster: None,
            })
        );
        // Each sampled key standing for 10 rows: 10 keys make a cap.
        assert_eq!(
            plan_repair(&fresh, &keys, 10, cap).map(|range| (range.low, range.high)),
            Some((0, 9))
        );

        // Ordered clusters at 100..=199 and 300..=399; most fresh rows over the second.
        let pieces = [
            ordered(50, 100, 199, 1),
            ordered(50, 300, 399, 2),
            piece(40, 0, 500, 3),
        ];
        let keys = [10, 150, 310, 320, 330, 450];
        let range = plan_repair(&pieces, &keys, 1, cap).unwrap();
        assert_eq!((range.low, range.high), (300, 399));
        assert_eq!(range.cluster.map(|cluster| cluster.ids), Some(vec![id(2)]));
        // Most of them in the gap between: the range is the gap's.
        let keys = [150, 210, 220, 230, 310];
        let range = plan_repair(&pieces, &keys, 1, cap).unwrap();
        assert_eq!((range.low, range.high, range.cluster), (200, 299, None));
        // Above everything: up to the top of the key space.
        let keys = [410, 420];
        let range = plan_repair(&pieces, &keys, 1, cap).unwrap();
        assert_eq!((range.low, range.high), (400, u128::MAX));
    }

    /// An ordered cluster past the cap is repaired even with no fresh rows over it.
    #[test]
    fn an_ordered_stretch_past_the_cap_is_repaired() {
        let pieces = [ordered(80, 0, 99, 1), ordered(60, 50, 150, 2)];
        let range = plan_repair(&pieces, &[], 1, 100).unwrap();
        assert_eq!((range.low, range.high), (0, 150));
        assert_eq!(range.cluster.unwrap().docs, 140);
    }

    /// Runs of keys that never decrease; too many, or too short on average, and the segment is
    /// read whole.
    #[test]
    fn runs_are_sorted_where_long_and_coarse_between() {
        let keys = |runs: &[std::ops::Range<u128>]| -> Vec<u128> {
            runs.iter().flat_map(|run| run.clone()).collect()
        };
        let sorted = |start, end, min, max| Run {
            start,
            end,
            min,
            max,
            blocks: None,
            reversed: false,
        };
        // Two long stretches in order: a sorted run each.
        let two = layout_runs(&keys(&[0..1500, 10..1200])).unwrap();
        assert_eq!(
            two.runs,
            vec![sorted(0, 1500, 0, 1499), sorted(1500, 2690, 10, 1199)]
        );
        assert!(two.blocks.is_empty());
        // Equal keys stay in one run.
        assert_eq!(
            layout_runs(&[7; 2000]).unwrap().runs,
            vec![sorted(0, 2000, 7, 7)]
        );
        // Rows in order but for their neighbours, as a stream leaves them: one coarse run after
        // the sorted one, its blocks in order.
        let mut stream: Vec<u128> = (0..2000).collect();
        stream.extend((2000..3000u128).map(|key| key ^ 1));
        let runs = layout_runs(&stream).unwrap();
        assert_eq!(
            runs.runs,
            vec![
                // The first row of the pair after it is still in order.
                sorted(0, 2001, 0, 2001),
                Run {
                    start: 2001,
                    end: 3000,
                    min: 2000,
                    max: 2999,
                    blocks: Some(0),
                    reversed: false,
                }
            ]
        );
        assert_eq!(runs.blocks.len(), 4);
        assert_eq!(runs.blocks[1], (2256, 2513));
        // Rows in no order: one coarse run, every block bounded by about the whole of it.
        let shuffled: Vec<u128> = (0..1024u128).map(|i| (i * 389) % 1024).collect();
        let runs = layout_runs(&shuffled).unwrap();
        assert_eq!(runs.runs.len(), 1, "{runs:?}");
        assert_eq!(runs.runs[0].blocks, Some(0));
        assert_eq!((runs.runs[0].min, runs.runs[0].max), (0, 1023));
        assert_eq!(runs.blocks.len(), 4);
        assert!(runs.blocks.is_sorted_by_key(|&(min, _)| min));
        assert!(runs.blocks.is_sorted_by_key(|&(_, max)| max));
        assert!(runs.blocks[3].0 < 256 && runs.blocks[0].1 > 768, "{runs:?}");
        // Keys going down: a sorted run read from its other end, and a coarse one whose
        // blocks are counted from its last doc ids.
        let mut down: Vec<u128> = (0..2000u128).rev().map(|key| key + 5000).collect();
        down.extend((0..1000u128).rev().map(|key| (key ^ 1) + 10_000));
        let runs = layout_runs(&down).unwrap();
        assert_eq!(runs.runs.len(), 2, "{runs:?}");
        assert_eq!(
            runs.runs[0],
            Run {
                reversed: true,
                ..sorted(0, 2000, 5000, 6999)
            }
        );
        assert_eq!(runs.runs[0].doc_at(0), 1999);
        assert_eq!(runs.runs[0].docs(0, 10), (1990, 2000));
        assert_eq!(runs.runs[0].positions(1990, 2000), (0, 10));
        let coarse = runs.runs[1];
        assert!(coarse.reversed && coarse.blocks == Some(0), "{coarse:?}");
        assert_eq!((coarse.min, coarse.max), (10_000, 10_999));
        assert_eq!(runs.blocks[0].0, 10_000);
        assert!(runs.blocks.is_sorted_by_key(|&(min, _)| min));
        assert!(runs.blocks.is_sorted_by_key(|&(_, max)| max));
        // Its first block by position is its last 256 doc ids.
        assert_eq!(coarse.block_of(2999), 0);
        assert_eq!(coarse.edge_up(2900), 3000);
        assert_eq!(coarse.edge_up(2001), 3000 - 3 * 256);
        assert_eq!(layout_runs(&[]), None);
    }

    /// A repair step's key ranges hold half a cap, as a split's parts do, and never more than
    /// one hold of the lock may move, whatever the cap.
    #[test]
    fn a_repair_range_is_bounded_whatever_the_cap() {
        assert_eq!(repair_range_rows(100), 50);
        assert_eq!(repair_range_rows(200_000), REPAIR_LOCK_ROWS);
        assert_eq!(repair_range_rows(1_000_000), REPAIR_LOCK_ROWS);
    }

    /// Parts of about equal size, never cutting a key: the parts are disjoint ranges.
    #[test]
    fn split_parts_are_even_and_never_cut_a_key() {
        let rows: Vec<(u128, ())> = (0..130).map(|key| (key, ())).collect();
        let sizes: Vec<usize> = split_parts(rows, 50).iter().map(Vec::len).collect();
        assert_eq!(sizes, [44, 44, 42]);

        let tied: Vec<(u128, ())> = [1, 2, 2, 2, 3, 4]
            .into_iter()
            .map(|key| (key, ()))
            .collect();
        let parts: Vec<Vec<u128>> = split_parts(tied, 2)
            .into_iter()
            .map(|part| part.into_iter().map(|(key, _)| key).collect())
            .collect();
        assert_eq!(parts, [vec![1, 2, 2, 2], vec![3, 4]]);
        assert!(split_parts(Vec::<(u128, ())>::new(), 50).is_empty());
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
                    AsyncInProgress::fullscan(tx),
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
                        AsyncInProgress::fullscan(tx),
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
                    AsyncInProgress::fullscan(tx),
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
            let idle = !stats.rewrite.pending && stats.merges.running == 0;
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
        assert!(!stats.rewrite.pending, "{:?}", stats.rewrite);
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
                    AsyncInProgress::fullscan(tx),
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
                    AsyncInProgress::fullscan(tx),
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
            let idle = !stats.rewrite.pending && stats.merges.running == 0;
            let capped = stats.segments.iter().all(|s| s.docs <= cap);
            let narrow = widest_of(&stats) <= 2 * u64::from(cap);
            if (idle && capped && narrow && stats.rewrite.docs_rewritten > 0)
                || Instant::now() > deadline
            {
                break stats;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        };
        assert!(!stats.rewrite.pending, "{:?}", stats.rewrite);
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

        // A merge makes a new segment, whose columns are opened once too, so only readings
        // with the same segments before and after say that nothing was opened twice.
        loop {
            let before = sender.stats(make_index_key()).await.unwrap();
            search_ordered(&sender, "将军", 10, SearchWindow::default()).await;
            let first = sender.stats(make_index_key()).await.unwrap();
            search_ordered(&sender, "将军", 10, SearchWindow::default()).await;
            let second = sender.stats(make_index_key()).await.unwrap();
            if before.segments != second.segments {
                continue;
            }
            assert!(
                first.walk.column_opens <= first.segments.len() as u64,
                "{:?}",
                first.walk
            );
            assert!(first.walk.prepare_nanos <= first.walk.walk_nanos);
            assert_eq!(
                second.walk.column_opens, first.walk.column_opens,
                "{:?}",
                second.walk
            );
            assert_eq!(second.walk.searches, before.walk.searches + 2);
            break;
        }
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

    /// A segment that lacks one of the pattern's grams is not opened.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn a_segment_without_one_of_the_grams_is_not_opened() {
        let sender = make_sender();
        add_docs(&sender, NICKNAMES).await;

        // Every value holds 将 or 军 or 宫, none holds the three-character gram.
        assert!(like_ids(&sender, "%将宫军%").await.is_empty());

        let walk = sender.stats(make_index_key()).await.unwrap().walk;
        assert!(walk.segments_considered >= 1, "{walk:?}");
        assert_eq!(walk.segments_opened, 0, "{walk:?}");
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
                    AsyncInProgress::fullscan(tx),
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
            make_memory_actor(Allocate::Can),
            TEST_COMMIT_INTERVAL,
            TEST_COMMIT_THRESHOLD,
            None,
            SearchBudget::default(),
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

    /// A pattern tied to the start or the end of the value takes the same walk: it is ordered and
    /// paged as any other.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn anchored_patterns_are_ordered_and_paged_too() {
        let sender = make_sender();
        add_docs(&sender, NICKNAMES).await;
        let mut expected = [1, 2];
        expected.sort_by_key(|id| token_of(*id as u64));

        let page = like(&sender, "%将军", 1, SearchWindow::default()).await;
        assert_eq!(ids_of(&page.primary_keys), expected[..1]);
        let cursor = page.next_cursor.expect("a full page leaves a cursor");
        let page = like(&sender, "%将军", 1, resume(SortOrder::Asc, cursor)).await;
        assert_eq!(ids_of(&page.primary_keys), expected[1..]);
        // A filled page always leaves a cursor; the page after it is empty and leaves none.
        let cursor = page.next_cursor.expect("a full page leaves a cursor");
        let page = like(&sender, "%将军", 1, resume(SortOrder::Asc, cursor)).await;
        assert!(page.primary_keys.is_empty());
        assert_eq!(page.next_cursor, None);
    }

    /// The frame is stripped before a row is re-indexed by the rewrite; otherwise every pass
    /// would add another pair of marks.
    #[test]
    fn a_stored_value_is_unframed_once() {
        assert_eq!(gram::framed("abc"), "\u{2}abc\u{3}");
        assert_eq!(gram::unframed("\u{2}abc\u{3}"), Some("abc"));
        assert_eq!(gram::unframed("abc"), None);
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
            make_memory_actor(Allocate::Can),
            COMMIT_INTERVAL,
            MAX_UNCOMMITTED_THRESHOLD,
            None,
            SearchBudget::default(),
        );
        let (tx, mut rx) = mpsc::channel(1);
        for n in 1..=rows {
            let name = format!("用户{}号玩家{}", n % 9973, n);
            sender
                .add_document(
                    test_partition_id(),
                    n.into(),
                    name,
                    AsyncInProgress::fullscan(tx.clone()),
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
        let mut repair = RewriteTotals::default();
        while started.elapsed() < Duration::from_secs(120) {
            let asked = Instant::now();
            sender.count(make_index_key()).await.unwrap();
            slowest_count = slowest_count.max(asked.elapsed());
            let asked = Instant::now();
            let stats = sender.stats(make_index_key()).await.unwrap();
            slowest = slowest.max(asked.elapsed());
            repair = stats.rewrite;
            if repair.steps > 0 && !repair.pending {
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        println!(
            "STALL repair {} steps, pending {}, after {:.0}s, slowest stats {:?}, slowest count {:?}",
            repair.steps,
            repair.pending,
            started.elapsed().as_secs_f64(),
            slowest,
            slowest_count
        );
        assert!(repair.steps > 0, "nothing was repaired");
        assert!(
            slowest_count < Duration::from_secs(1),
            "the actor stalled for {slowest_count:?}"
        );
        assert!(slowest < Duration::from_secs(1), "stats took {slowest:?}");
    }

    fn rss_mb() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| {
                status
                    .lines()
                    .find_map(|line| line.strip_prefix("VmRSS:"))
                    .and_then(|kb| kb.trim().trim_end_matches(" kB").parse::<u64>().ok())
            })
            .map_or(0, |kb| kb / 1024)
    }

    /// A table-order index with default options, loaded fast, must settle, and once dropped must
    /// stop working and give its memory back. On AWS (run a0e51aa9, 10M rows ingested in 136 s next
    /// to an unordered index) it never settled -- about 2000 merges a minute held it at 500-780
    /// segments -- and after the drop its merges went on, with the node's memory going from 8 to
    /// 29.5 GB in four minutes. Diagnostic, not a pass/fail check yet: run with `--ignored`
    /// (CHURN_ROWS sets the size, CHURN_SECS how long to watch before the drop, CHURN_CAP the cap).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn a_fast_loaded_table_order_index_settles_and_stops_when_dropped() {
        let rows: u64 = std::env::var("CHURN_ROWS").map_or(2_000_000, |rows| rows.parse().unwrap());
        let watch = std::env::var("CHURN_SECS").map_or(900, |secs| secs.parse().unwrap());
        let cap =
            std::env::var("CHURN_CAP").map_or(DEFAULT_SEGMENT_ROWS, |cap| cap.parse().unwrap());
        // RUST_LOG=info shows the merges ('Starting merge', 'pattern: asking for ... merges').
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init();
        let sender = new(
            PatternIndexConfiguration {
                key: make_index_key(),
                // What db.rs gives a table-order index created with no options.
                options: IndexOptionsPattern {
                    segment_max_docs: SegmentMaxDocs::from(Some(cap)),
                    rewrite_wide_segments: RewriteWideSegments::from(true),
                    ..IndexOptionsPattern::default()
                },
                key_layout: test_layout(),
            },
            make_table(bigint_key, |_, _| None),
            worker::new(),
            make_memory_actor(Allocate::Can),
            COMMIT_INTERVAL,
            MAX_UNCOMMITTED_THRESHOLD,
            None,
            SearchBudget::default(),
        );
        let started = Instant::now();
        let (tx, mut rx) = mpsc::channel(1);
        for n in 1..=rows {
            let name = format!("用户{}号玩家{}", n % 9973, n);
            sender
                .add_document(
                    test_partition_id(),
                    n.into(),
                    name,
                    AsyncInProgress::fullscan(tx.clone()),
                )
                .await
                .unwrap();
        }
        drop(tx);
        rx.recv().await;
        println!(
            "CHURN loaded {rows} rows in {:.0}s, rss {} MB",
            started.elapsed().as_secs_f64(),
            rss_mb()
        );

        let loaded = Instant::now();
        let mut counts = Vec::new();
        while loaded.elapsed() < Duration::from_secs(watch) {
            tokio::time::sleep(Duration::from_secs(10)).await;
            let stats = sender.stats(make_index_key()).await.unwrap();
            let segments = stats.tantivy.segment_count;
            println!(
                "CHURN t={:.0}s segments={segments} repair={} steps {} rows, pending {} \
                 fresh={}/{} depth={}/{:.2} merges={}/{} rss={} MB",
                loaded.elapsed().as_secs_f64(),
                stats.rewrite.steps,
                stats.rewrite.docs_rewritten,
                stats.rewrite.pending,
                stats.overlap.fresh_rows,
                stats.overlap.fresh_segments,
                stats.overlap.depth_max,
                stats.overlap.depth_rows as f64 / stats.overlap.rows.max(1) as f64,
                stats.merges.started,
                stats.merges.ended,
                rss_mb()
            );
            counts.push(segments);
            let rewrite_done = !stats.rewrite.pending && stats.merges.running == 0;
            if rewrite_done
                && counts.len() >= 12
                && counts[counts.len() - 12..].iter().all(|&c| c == segments)
            {
                println!("CHURN settled at {segments} segments");
                break;
            }
        }

        drop(sender);
        let dropped = Instant::now();
        while dropped.elapsed() < Duration::from_secs(180) {
            tokio::time::sleep(Duration::from_secs(10)).await;
            println!(
                "CHURN dropped t={:.0}s rss={} MB",
                dropped.elapsed().as_secs_f64(),
                rss_mb()
            );
        }
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
            make_memory_actor(Allocate::Can),
            TEST_COMMIT_INTERVAL,
            TEST_COMMIT_THRESHOLD,
            None,
            SearchBudget::default(),
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

    /// An index without `order_by` answers in the order of its partitions -- by token, as a scan
    /// of the table, or by the scrambled token, as `partition_order` says -- pages through it
    /// with a cursor, and a smaller limit is a prefix of a larger one.
    #[rstest]
    #[case::scattered(crate::PartitionOrder::Scattered)]
    #[case::token(crate::PartitionOrder::Token)]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn an_index_without_order_by_answers_in_the_order_of_its_partitions(
        #[case] partition_order: crate::PartitionOrder,
    ) {
        let token_of = |id: u64| partition_place(id, partition_order);
        let sender = make_sender_for(
            IndexOptionsPattern {
                partition_order,
                ..IndexOptionsPattern::default()
            },
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

    /// Rows of one partition come together and in clustering order, honouring a `DESC` clustering
    /// column, and partitions in the index's partition order; a page boundary inside a partition resumes exactly.
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
        let token = |id: u64| {
            layout
                .partition_order_key(&key_of(id), crate::PartitionOrder::default())
                .unwrap()
        };
        let mut expected = ids.clone();
        // By the partition's place, then clustering descending.
        expected.sort_by_key(|id| (token(*id), Reverse(id % 10)));
        let expected: Vec<PrimaryKey> = expected.into_iter().map(key_of).collect();

        for page in [1, 3, 5] {
            let mut seen = Vec::new();
            let mut window = SearchWindow::default();
            loop {
                let found = like(&sender, "%将军%", page, window).await;
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
                    AsyncInProgress::fullscan(tx),
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
            let settled = stats.rewrite.docs_rewritten > 0
                && !stats.rewrite.pending
                && stats.merges.running == 0
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
                    AsyncInProgress::fullscan(tx),
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
            let finished =
                rewrite.docs_rewritten > 0 && !rewrite.pending && stats.merges.running == 0;
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

    /// Adds `ids` as rows named by `text`, all sent before any is waited for, as one batch.
    async fn add_batch(
        sender: &mpsc::Sender<PatternIndex>,
        ids: impl IntoIterator<Item = u64>,
        text: fn(u64) -> String,
    ) {
        let mut acks = Vec::new();
        for primary in ids {
            let (tx, rx) = mpsc::channel(1);
            sender
                .add_document(
                    test_partition_id(),
                    primary.into(),
                    text(primary),
                    AsyncInProgress::fullscan(tx),
                )
                .await
                .unwrap();
            acks.push(rx);
        }
        for mut ack in acks {
            ack.recv().await;
        }
    }

    /// The ids `i -> i * 1103 + 977 (mod rows)`, plus one: a fixed shuffle of `1..=rows` for a
    /// `rows` coprime with 1103.
    fn make_sender_with_budget(
        options: IndexOptionsPattern,
        commit_threshold: usize,
        search_budget: SearchBudget,
    ) -> mpsc::Sender<PatternIndex> {
        new(
            PatternIndexConfiguration {
                key: make_index_key(),
                options,
                key_layout: test_layout(),
            },
            make_table_with_keys(),
            worker::new(),
            make_memory_actor(Allocate::Can),
            TEST_COMMIT_INTERVAL,
            commit_threshold,
            None,
            search_budget,
        )
    }

    /// A row of the budget tests: a thirtieth hold the keyword 甲乙丙丁, and a third all of its
    /// grams apart, which the index can only tell from the keyword by reading the value.
    fn nominated(primary: u64) -> String {
        if primary.is_multiple_of(30) {
            format!("user{primary}甲乙丙丁")
        } else if primary.is_multiple_of(3) {
            format!("user{primary}甲乙丙之乙丙丁")
        } else {
            format!("user{primary}将军")
        }
    }

    /// Every page of `query` from the first, each from the cursor of the one before: the ids in
    /// the order they came, and how many pages came with fewer rows than `limit_n` and a cursor.
    async fn pages_by_cursor(
        sender: &mpsc::Sender<PatternIndex>,
        query: &str,
        limit_n: usize,
        order: Option<SortOrder>,
    ) -> (Vec<i64>, usize) {
        let mut cursor = None;
        let mut ids = Vec::new();
        let mut short = 0;
        for _ in 0..100_000 {
            let window = SearchWindow {
                order,
                cursor,
                ..SearchWindow::default()
            };
            let page = like(sender, &containing(query), limit_n, window).await;
            let got = ids_of(&page.primary_keys);
            if page.next_cursor.is_some() && got.len() < limit_n {
                short += 1;
            }
            ids.extend(got);
            cursor = page.next_cursor;
            if cursor.is_none() {
                return (ids, short);
            }
        }
        panic!("the cursors do not come to an end");
    }

    /// A search whose budget ends returns the rows it has settled and a cursor, and the pages
    /// that follow from the cursors hold every match once, in order: whatever the page size and
    /// the direction, over sorted runs and over segments that have none, and with no time at
    /// all, when a request gets only as far as it must to leave a cursor past the one it came
    /// with.
    #[rstest]
    #[case::sorted_runs(true, Some(300), None)]
    #[case::no_runs(false, Some(300), None)]
    #[case::no_time_at_all(true, None, Some(Duration::ZERO))]
    #[timeout(Duration::from_secs(180))]
    #[tokio::test]
    async fn a_budget_cuts_a_page_short_and_loses_no_row_and_no_order(
        #[case] repair: bool,
        #[case] work: Option<u64>,
        #[case] time: Option<Duration>,
    ) {
        let rows = 9_000u64;
        let options = IndexOptionsPattern {
            rewrite_wide_segments: RewriteWideSegments::from(repair),
            ..capped_options(4_000)
        };
        let limited = make_sender_with_budget(options.clone(), 2000, SearchBudget { time, work });
        let unlimited = make_sender_with_threshold(options, 2000);
        for sender in [&limited, &unlimited] {
            add_batch(sender, shuffled(rows), nominated).await;
            let deadline = Instant::now() + Duration::from_secs(75);
            let mut quiet = 0;
            while quiet < 3 {
                let stats = sender.stats(make_index_key()).await.unwrap();
                let still = stats.tantivy.num_docs == rows
                    && stats.merges.running == 0
                    && !stats.rewrite.pending
                    && (!repair
                        || (stats.rewrite.docs_rewritten > 0 && stats.overlap.fresh_rows == 0));
                quiet = if still { quiet + 1 } else { 0 };
                assert!(Instant::now() < deadline, "not settled: {stats:?}");
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }

        let newest_first: Vec<i64> = (1..=rows as i64).rev().filter(|id| id % 30 == 0).collect();
        let (whole, short) = pages_by_cursor(&unlimited, "甲乙丙丁", 10_000, None).await;
        assert_eq!(whole, newest_first);
        assert_eq!(short, 0);
        assert_eq!(
            unlimited
                .stats(make_index_key())
                .await
                .unwrap()
                .walk
                .cut_short,
            0
        );

        for limit_n in [7, 10_000] {
            let (ids, _) = pages_by_cursor(&limited, "甲乙丙丁", limit_n, None).await;
            assert_eq!(ids, newest_first, "pages of {limit_n}");
            let (ids, _) =
                pages_by_cursor(&limited, "甲乙丙丁", limit_n, Some(SortOrder::Asc)).await;
            let oldest_first: Vec<i64> = newest_first.iter().rev().copied().collect();
            assert_eq!(ids, oldest_first, "pages of {limit_n}, ascending");
        }
        // No row in the answer, and every candidate still read: by many requests.
        // Every gram of this one is in a third of the rows, and the keyword in none.
        let (none, short) = pages_by_cursor(&limited, "乙丙之乙丙之", 20, None).await;
        assert_eq!(none, Vec::<i64>::new());
        assert!(short > 0);
        let walk = limited.stats(make_index_key()).await.unwrap().walk;
        assert!(walk.cut_short > 0, "{walk:?}");
        assert!(walk.cut_short < walk.searches, "{walk:?}");
    }

    /// Rows that arrive in no order are written a commit at a time in the order of their keys,
    /// so a page reads the end of each commit's stretch and no more, with no repair: written as
    /// they came they would make segments in no order, of which a page reads every match.
    #[rstest]
    #[timeout(Duration::from_secs(90))]
    #[tokio::test]
    async fn rows_arriving_in_no_order_are_read_from_the_end_without_a_repair() {
        let rows = 20_000u64;
        let options = IndexOptionsPattern {
            rewrite_wide_segments: RewriteWideSegments::from(false),
            ..capped_options(10_000)
        };
        let sender = make_sender_with_threshold(options, 5_000);
        add_batch(&sender, shuffled(rows), named).await;
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut quiet = 0;
        while quiet < 3 {
            let stats = sender.stats(make_index_key()).await.unwrap();
            let still = stats.tantivy.num_docs == rows && stats.merges.running == 0;
            quiet = if still { quiet + 1 } else { 0 };
            assert!(Instant::now() < deadline, "not settled: {stats:?}");
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let stats = sender.stats(make_index_key()).await.unwrap();
        assert_eq!(stats.rewrite.docs_rewritten, 0, "{:?}", stats.rewrite);

        for (window, expected) in [
            (
                SearchWindow::default(),
                (rows as i64 - 9..=rows as i64).rev().collect::<Vec<_>>(),
            ),
            (ascending(), (1..=10).collect()),
        ] {
            let before = sender.stats(make_index_key()).await.unwrap().walk;
            let (page, _) = search_ordered(&sender, "将军", 10, window).await;
            assert_eq!(page, expected);
            let after = sender.stats(make_index_key()).await.unwrap().walk;
            let scanned = after.postings_scanned - before.postings_scanned;
            assert!(
                scanned <= rows / 4,
                "{scanned} postings for a page of 10 of {rows} rows: {:?}",
                stats.segments
            );
        }
    }

    fn shuffled(rows: u64) -> impl Iterator<Item = u64> {
        (0..rows).map(move |i| (i * 1103 + 977) % rows + 1)
    }

    fn named(primary: u64) -> String {
        format!("user{primary}将军")
    }

    /// The stats once the repair has had something to do and has nothing left, no merge runs,
    /// and `done` holds, for three polls in a row; panics at `secs`.
    async fn settled(
        sender: &mpsc::Sender<PatternIndex>,
        secs: u64,
        done: impl Fn(&PatternStats) -> bool,
    ) -> PatternStats {
        let deadline = Instant::now() + Duration::from_secs(secs);
        let mut quiet = 0;
        loop {
            let stats = sender.stats(make_index_key()).await.unwrap();
            let still = stats.rewrite.docs_rewritten > 0
                && !stats.rewrite.pending
                && stats.merges.running == 0
                && done(&stats);
            quiet = if still { quiet + 1 } else { 0 };
            if quiet >= 3 {
                return stats;
            }
            assert!(
                Instant::now() < deadline,
                "not settled: {:?} {:?} {:?}",
                stats.rewrite,
                stats.overlap,
                stats.segments
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    fn capped_options(cap: u32) -> IndexOptionsPattern {
        IndexOptionsPattern {
            segment_max_docs: cap.to_string().parse().unwrap(),
            rewrite_wide_segments: RewriteWideSegments::from(true),
            ..ordered_options()
        }
    }

    /// The ids of the first page of `query`, highest sort key first.
    async fn first_page(sender: &mpsc::Sender<PatternIndex>, query: &str, n: usize) -> Vec<i64> {
        search_ordered(sender, query, n, SearchWindow::default())
            .await
            .0
    }

    /// A hold of the writer by a repair step moves the fresh rows of the step's range and no
    /// others, however far the ordered segment it splits reaches. A commit of the lowest rows
    /// and one row above every other overlaps nothing, so it is taken as ordered, and spans
    /// every fresh row loaded after it: the part of the split that holds its last row reaches
    /// from the step's range to the top, and took every fresh row on the way in one hold. (At
    /// 10M rows that hold lasted 930 s: see the stray row in docs/dev/pattern/postponed.md.)
    #[rstest]
    #[timeout(Duration::from_secs(90))]
    #[tokio::test]
    async fn a_step_over_a_wide_segment_moves_only_its_range() {
        let cap = 100u32;
        let rows = 3000u64;
        let sender = make_sender_with_threshold(capped_options(cap), 5000);
        add_batch(&sender, (1..=60).chain([rows]), named).await;
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let stats = sender.stats(make_index_key()).await.unwrap();
            if stats.tantivy.num_docs == 61 && stats.overlap.fresh_rows == 0 {
                assert_eq!(stats.rewrite.steps, 0, "{:?}", stats.rewrite);
                break;
            }
            assert!(Instant::now() < deadline, "{:?}", stats.overlap);
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        add_batch(&sender, shuffled(rows - 61).map(|id| id + 60), named).await;
        // A step moves at most a cap of fresh rows, and with them the rows of the segments it
        // splits: here never more than three caps. Unbounded, the first step moved them all.
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut quiet = 0;
        while quiet < 3 {
            let stats = sender.stats(make_index_key()).await.unwrap();
            assert!(
                stats.rewrite.docs_rewritten <= stats.rewrite.steps * 3 * u64::from(cap),
                "{:?}",
                stats.rewrite
            );
            let still = stats.rewrite.steps > 0
                && !stats.rewrite.pending
                && stats.merges.running == 0
                && stats.tantivy.num_docs == rows
                && stats.overlap.fresh_rows == 0;
            quiet = if still { quiet + 1 } else { 0 };
            assert!(
                Instant::now() < deadline,
                "not settled: {:?} {:?}",
                stats.rewrite,
                stats.overlap
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// An idle index converges: no fresh rows left, and ordered segments that overlap nowhere.
    #[rstest]
    #[timeout(Duration::from_secs(60))]
    #[tokio::test]
    async fn an_idle_index_ends_with_no_fresh_rows_and_disjoint_segments() {
        let cap = 100;
        let sender = make_sender_with_threshold(capped_options(cap), 2000);
        add_batch(&sender, shuffled(2000), named).await;
        let stats = settled(&sender, 45, |_| true).await;
        assert_eq!(stats.tantivy.num_docs, 2000);
        let overlap = stats.overlap;
        assert_eq!(
            (overlap.fresh_rows, overlap.fresh_segments),
            (0, 0),
            "{overlap:?}"
        );
        assert_eq!(overlap.depth_max, 1, "{overlap:?} {:?}", stats.segments);
        assert!(
            stats.segments.iter().all(|s| s.docs <= cap),
            "{:?}",
            stats.segments
        );
        // A step moves a cap of fresh rows, or splits a stretch: about the rows once or twice.
        assert!(
            stats.rewrite.docs_rewritten < 3 * 2000,
            "{:?}",
            stats.rewrite
        );
        assert_eq!(first_page(&sender, "将军", 3).await, [2000, 1999, 1998]);
    }

    /// Deletes and updates arriving while the repair moves rows are kept: a row deleted is not
    /// brought back by the re-add of its old copy, and a row updated is there once, as updated.
    #[rstest]
    #[timeout(Duration::from_secs(90))]
    #[tokio::test]
    async fn deletes_and_updates_during_the_repair_are_kept() {
        let cap = 100;
        let sender = make_sender_with_threshold(capped_options(cap), 2000);
        add_batch(&sender, shuffled(2000), named).await;
        // Odd ids up to 400 deleted, even ones renamed, in batches a commit apart, while the
        // repair runs on the idle ticks between them.
        for chunk in (1..=400u64).collect::<Vec<_>>().chunks(40) {
            for &primary in chunk {
                rm_doc_no_wait(&sender, primary).await;
            }
            add_batch(
                &sender,
                chunk.iter().copied().filter(|id| id % 2 == 0),
                |primary| format!("renamed{primary}"),
            )
            .await;
        }
        let stats = settled(&sender, 75, |stats| stats.overlap.fresh_rows == 0).await;
        assert_eq!(stats.tantivy.num_docs, 2000 - 200);

        let renamed = first_page(&sender, "renamed", 1000).await;
        let expected: Vec<i64> = (2..=400).rev().step_by(2).collect();
        assert_eq!(renamed, expected);
        let named: Vec<i64> = first_page(&sender, "将军", 2000).await;
        let expected: Vec<i64> = (401..=2000).rev().collect();
        assert_eq!(named, expected);
    }

    /// The sort value of the row with id 7, for `an_update_moving_a_row_takes_it_to_its_new_place`.
    static MOVED_VALUE: AtomicU64 = AtomicU64::new(7);

    /// An update that changes a row's sort value moves it: the new copy is repaired into its new
    /// place, and the old one, deleted, is not found at the old.
    #[rstest]
    #[timeout(Duration::from_secs(60))]
    #[tokio::test]
    async fn an_update_moving_a_row_takes_it_to_its_new_place() {
        let cap = 100;
        let sender = make_sender_for(
            capped_options(cap),
            make_table(bigint_key, |id, _| {
                let value = if id == 7 {
                    MOVED_VALUE.load(Relaxed)
                } else {
                    id
                };
                Some(CqlValue::BigInt(value as i64))
            }),
            test_layout(),
        );
        add_batch(&sender, shuffled(1000), named).await;
        let before = settled(&sender, 45, |_| true).await;
        MOVED_VALUE.store(5000, Relaxed);
        rm_doc_no_wait(&sender, 7).await;
        add_batch(&sender, [7], named).await;
        // Above every other row, the new copy overlaps nothing: it is adopted where it lands,
        // with no row moved.
        let stats = settled(&sender, 45, |stats| stats.overlap.fresh_rows == 0).await;
        assert_eq!(stats.rewrite.docs_rewritten, before.rewrite.docs_rewritten);
        assert_eq!(stats.tantivy.num_docs, 1000);
        assert_eq!(stats.overlap.depth_max, 1, "{:?}", stats.overlap);
        assert_eq!(first_page(&sender, "将军", 3).await, [7, 1000, 999]);
        let all = first_page(&sender, "将军", 1000).await;
        assert_eq!(all.iter().filter(|&&id| id == 7).count(), 1);
    }

    /// After most rows are deleted, the segments full of holes are merged until none holds more
    /// deleted rows than live ones and the survivors sit in a few segments. One with fewer may
    /// stay as it is: its neighbours can be too full to merge with, and a merge of it alone is
    /// not worth its holes.
    #[rstest]
    #[timeout(Duration::from_secs(60))]
    #[tokio::test]
    async fn a_mass_delete_compacts_to_few_segments() {
        let cap = 100;
        let sender = make_sender_with_threshold(capped_options(cap), 2000);
        add_batch(&sender, shuffled(2000), named).await;
        settled(&sender, 45, |_| true).await;
        for primary in (1..=2000u64).filter(|id| id % 4 != 0) {
            rm_doc_no_wait(&sender, primary).await;
        }
        rm_doc(&sender, 1).await;
        let stats = settled(&sender, 45, |stats| {
            stats.tantivy.num_docs == 500
                && stats
                    .segments
                    .iter()
                    .all(|segment| segment.deleted < segment.docs.max(1))
        })
        .await;
        // Neighbours merge while within three quarters of the cap, so any two of them hold more
        // than that together.
        let most = 2 * 500 / ordered_fill(cap) as usize + 1;
        assert!(
            stats.tantivy.segment_count <= most,
            "{} segments: {:?}",
            stats.tantivy.segment_count,
            stats.segments
        );
        let expected: Vec<i64> = (4..=2000).rev().step_by(4).collect();
        assert_eq!(first_page(&sender, "将军", 1000).await, expected);
    }

    /// A segment a repair leaves is a few runs in key order, and a page is read from the end of
    /// each run it needs rather than the whole segment: newest first, ascending, after a cursor
    /// either way, within a range, and verified past max_gram, all as a full read would answer.
    #[rstest]
    #[timeout(Duration::from_secs(90))]
    #[tokio::test]
    async fn a_sorted_segment_is_read_from_the_end_the_page_needs() {
        let rows = 20_000u64;
        let sender = make_sender_with_threshold(capped_options(10_000), rows as usize);
        add_batch(&sender, shuffled(rows), named).await;
        let stats = settled(&sender, 75, |stats| stats.overlap.fresh_rows == 0).await;
        assert_eq!(stats.tantivy.num_docs, rows);
        let largest = stats.segments.iter().map(|s| s.docs).max().unwrap();
        assert!(largest >= 4_000, "{:?}", stats.segments);

        let ids = |range: std::ops::RangeInclusive<u64>, descending: bool| -> Vec<i64> {
            let ids = range.map(|id| id as i64);
            if descending {
                ids.rev().collect()
            } else {
                ids.collect()
            }
        };
        let before = sender.stats(make_index_key()).await.unwrap().walk;
        let (page, cursor) = search_ordered(&sender, "将军", 10, SearchWindow::default()).await;
        assert_eq!(page, ids(rows - 9..=rows, true));
        let after = sender.stats(make_index_key()).await.unwrap().walk;
        // A full read of the top segment would scan every one of its rows, thousands; a run is
        // read from its end, a stretch as wide as the page needs at the segment's density.
        let scanned = after.postings_scanned - before.postings_scanned;
        assert!(
            scanned <= 1_024,
            "{scanned} postings for a page of 10: {:?}",
            stats.segments
        );

        let (next, _) = search_ordered(
            &sender,
            "将军",
            10,
            resume(SortOrder::Desc, cursor.unwrap()),
        )
        .await;
        assert_eq!(next, ids(rows - 19..=rows - 10, true));

        let (first, cursor) = search_ordered(&sender, "将军", 10, ascending()).await;
        assert_eq!(first, ids(1..=10, false));
        let (next, _) =
            search_ordered(&sender, "将军", 10, resume(SortOrder::Asc, cursor.unwrap())).await;
        assert_eq!(next, ids(11..=20, false));

        let (ranged, _) = search_ordered(
            &sender,
            "将军",
            10,
            window(None, Some(sort_key(5_000)), Some(sort_key(6_000))),
        )
        .await;
        assert_eq!(ranged, ids(5_991..=6_000, true));

        // Past max_gram (3): candidates by grams, verified against the stored value. Ids whose
        // decimal form starts with 1, newest first.
        let (verified, _) = search_ordered(&sender, "ser1", 10, SearchWindow::default()).await;
        assert_eq!(verified, ids(19_990..=19_999, true));
    }

    fn third_named(primary: u64) -> String {
        let mut name = format!("user{primary}将军");
        if primary.is_multiple_of(3) {
            name += "逢三之数";
        }
        if (5_000..9_000).contains(&primary) {
            name += "中段之人";
        }
        if primary.is_multiple_of(700) {
            name += "千中稀客";
        }
        name
    }

    /// Every page of a search, followed from its cursor in either direction, holds the rows a
    /// full read would give, in order, whatever the segments' rows are laid out as: sorted runs
    /// a repair wrote, coarse runs rows arriving in order leave, and rows in no order at all.
    /// For a keyword every row holds, one a third of them hold, one past max_gram, verified,
    /// ones the rows of one stretch of the order hold, far from either end, and ones in a row of
    /// 700, a page of which lies in many runs.
    #[rstest]
    #[case::repaired("repaired")]
    #[case::in_order("stream")]
    #[case::in_order_in_large_commits("stream of 4000")]
    #[case::in_no_order("unrepaired")]
    #[timeout(Duration::from_secs(120))]
    #[tokio::test]
    async fn pages_hold_the_same_rows_whatever_the_runs(#[case] layout: &str) {
        let rows = 24_000u64;
        let options = IndexOptionsPattern {
            rewrite_wide_segments: RewriteWideSegments::from(layout == "repaired"),
            ..capped_options(8_000)
        };
        let sender = make_sender_with_threshold(options, rows as usize);
        if layout.starts_with("stream") {
            // The index is declared descending, so rows in ascending order lie with their
            // keys going down: in commits of 500 as a coarse run, each thread's share being
            // short, in commits of 4000 as sorted ones.
            let commit = if layout == "stream" { 500 } else { 4000 };
            for batch in (1..=rows).collect::<Vec<_>>().chunks(commit) {
                add_batch(&sender, batch.iter().copied(), third_named).await;
            }
        } else {
            add_batch(&sender, shuffled(rows), third_named).await;
        }
        if layout == "repaired" {
            settled(&sender, 90, |stats| stats.overlap.fresh_rows == 0).await;
        }
        let mut merging = 0;
        while merging < 5 {
            let stats = sender.stats(make_index_key()).await.unwrap();
            merging = if stats.merges.running == 0 {
                merging + 1
            } else {
                0
            };
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let stats = sender.stats(make_index_key()).await.unwrap();
        assert_eq!(stats.tantivy.num_docs, rows);

        if layout.starts_with("stream") {
            // The newest rows are in the last blocks of the newest segment: a page reads those,
            // not the segment.
            let before = stats.walk;
            let page = first_page(&sender, "将军", 20).await;
            assert_eq!(
                page,
                (rows as i64 - 19..=rows as i64).rev().collect::<Vec<_>>()
            );
            let after = sender.stats(make_index_key()).await.unwrap().walk;
            let scanned = after.postings_scanned - before.postings_scanned;
            assert!(scanned <= 2_048, "{scanned} postings: {:?}", stats.segments);
        }
        type Holds = fn(&i64) -> bool;
        let keywords: [(&str, Holds); 7] = [
            ("将军", |_| true),
            ("逢三之", |id| id % 3 == 0),
            ("逢三之数", |id| id % 3 == 0),
            ("中段", |id| (5_000..9_000).contains(id)),
            ("中段之人", |id| (5_000..9_000).contains(id)),
            ("千中", |id| id % 700 == 0),
            ("千中稀客", |id| id % 700 == 0),
        ];
        for (keyword, holds) in keywords {
            let expected: Vec<i64> = (1..=rows as i64).filter(holds).collect();
            for page in [20, 100] {
                let (ascending, _) = all_pages(&sender, keyword, page, SortOrder::Asc).await;
                assert!(ascending == expected, "{keyword} ascending by {page}");
                let (descending, _) = all_pages(&sender, keyword, page, SortOrder::Desc).await;
                assert!(
                    descending.iter().rev().eq(expected.iter()),
                    "{keyword} descending by {page}"
                );
            }
            let (ranged, _) = search_ordered(
                &sender,
                keyword,
                50,
                window(None, Some(sort_key(9_000)), Some(sort_key(15_000))),
            )
            .await;
            let within: Vec<i64> = expected
                .iter()
                .copied()
                .filter(|id| (9_000..=15_000).contains(id))
                .rev()
                .take(50)
                .collect();
            assert_eq!(ranged, within, "{keyword} in a range");
        }
    }

    static MATRIX_ROWS: AtomicU64 = AtomicU64::new(0);
    /// MATRIX_SPARSE: keywords in one row of 30, 100, 300 and 3000 as well, to find the number
    /// of matches at which reading a segment whole stops being the cheaper way.
    static MATRIX_SPARSE: AtomicBool = AtomicBool::new(false);

    /// A row of the matrix benchmark: keywords of every frequency, spread evenly over the ids
    /// and gathered at either end of them, and one whose grams most rows hold apart.
    fn matrix_name(id: u64) -> String {
        let rows = MATRIX_ROWS.load(Relaxed);
        let mut name = format!("用户{}号玩家{}", id % 9973, id);
        if id.is_multiple_of(9) {
            name += "逢九之数";
        }
        if id.is_multiple_of(1000) {
            name += "千禧";
        }
        if id.is_multiple_of(5) {
            name += "末等";
        }
        if id <= rows / 10 {
            name += if id.is_multiple_of(100) {
                "旧稀客"
            } else {
                "旧人"
            };
        }
        if id > rows - rows / 10 {
            name += "新人";
        }
        if MATRIX_SPARSE.load(Relaxed) {
            for (every, keyword) in [(30, "卅选"), (100, "百选"), (300, "叁选"), (3000, "仟选")]
            {
                if id.is_multiple_of(every) {
                    name += keyword;
                }
            }
        }
        if id.is_multiple_of(3000) {
            name += "甲乙丙丁";
        } else if id.is_multiple_of(3) {
            name += "甲乙丙之乙丙丁";
        }
        name
    }

    /// What a page costs over a matrix of cases, to see how flat the cost is rather than how low
    /// in one of them: keywords dense to absent, their matches spread evenly over the order or
    /// gathered at one end of it, one whose candidates are a thousand to a match, pages of 20 to
    /// 1000 rows from either end and from the middle. MATRIX_STATE picks the index:
    ///
    /// - `settled`: loaded shuffled and repaired;
    /// - `unrepaired`: the same with no repair, as under writes the repair gives way to
    ///   (MATRIX_RANDOM loads these two in no order at all);
    /// - `churned`: settled, then a tenth of the rows deleted and a tenth written again;
    /// - `tiers`: ordered by a column of five values, so that most rows tie;
    /// - `stream`: loaded in the index's order, MATRIX_BATCH rows (10000, a full commit) at a
    ///   time and committed as in production, every MATRIX_COMMIT_MS (1000) or at 10000 rows,
    ///   as rows stamped with the time arrive.
    ///
    /// MATRIX_PARTITION (token or scattered) makes it an index without ordering columns in that
    /// order of its partitions, loaded by token as a full scan loads it; the `stream` state then
    /// commits as in production and waits for the repair.
    ///
    /// Run with `--ignored --nocapture`; MATRIX_ROWS sets the size, MATRIX_CAP the segment cap.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn the_cost_of_a_page_over_a_matrix_of_cases() {
        let rows: u64 =
            std::env::var("MATRIX_ROWS").map_or(1_000_000, |rows| rows.parse().unwrap());
        let cap: u32 = std::env::var("MATRIX_CAP")
            .map_or(DEFAULT_SEGMENT_ROWS.get(), |cap| cap.parse().unwrap());
        let state = std::env::var("MATRIX_STATE").unwrap_or_else(|_| "settled".into());
        MATRIX_ROWS.store(rows, Relaxed);
        let sparse = std::env::var("MATRIX_SPARSE").is_ok();
        MATRIX_SPARSE.store(sparse, Relaxed);
        RECORD_RUN_SHAPES.store(true, Relaxed);
        // MATRIX_OFF switches one mechanism of the walk off, to see what it is worth.
        WALK_OFF.store(
            match std::env::var("MATRIX_OFF").as_deref() {
                Ok("take") => OFF_TAKE,
                Ok("halving") => OFF_HALVING,
                Ok("sweep") => OFF_SWEEP,
                Ok("whole_read") => OFF_WHOLE_READ,
                Ok("spread") => OFF_SPREAD,
                // The rows of a commit written as they came, not in the order of their keys.
                Ok("sorted_commits") => OFF_SORTED_COMMITS,
                Ok(other) => panic!("MATRIX_OFF: {other}"),
                Err(_) => 0,
            },
            Relaxed,
        );
        let tiers = state == "tiers";
        // MATRIX_PARTITION (token or scattered): an index without ordering columns, in that
        // order of its partitions, with the rows arriving by token, as a full scan sends them.
        let partition_order: Option<crate::PartitionOrder> = std::env::var("MATRIX_PARTITION")
            .ok()
            .map(|order| order.parse().unwrap());
        let options = IndexOptionsPattern {
            rewrite_wide_segments: RewriteWideSegments::from(state != "unrepaired"),
            ..capped_options(cap)
        };
        let options = match partition_order {
            Some(partition_order) => IndexOptionsPattern {
                order_by: OrderBy::Table,
                partition_order,
                ..options
            },
            None => options,
        };
        let table = if tiers {
            make_table(bigint_key, |id, _| Some(CqlValue::BigInt((id % 5) as i64)))
        } else {
            make_table(bigint_key, |id, _| Some(CqlValue::BigInt(id as i64)))
        };
        let sender = new(
            PatternIndexConfiguration {
                key: make_index_key(),
                options,
                key_layout: test_layout(),
            },
            table,
            worker::new(),
            make_memory_actor(Allocate::Can),
            std::env::var("MATRIX_COMMIT_MS").map_or(
                if state == "stream" {
                    COMMIT_INTERVAL
                } else {
                    TEST_COMMIT_INTERVAL
                },
                |millis| Duration::from_millis(millis.parse().unwrap()),
            ),
            MAX_UNCOMMITTED_THRESHOLD,
            None,
            // MATRIX_BUDGET_MS: the budget of a search request, none without it.
            SearchBudget {
                time: std::env::var("MATRIX_BUDGET_MS")
                    .ok()
                    .map(|millis| Duration::from_millis(millis.parse().unwrap())),
                work: None,
            },
        );
        let started = Instant::now();
        let stream = state == "stream";
        let batch: usize =
            std::env::var("MATRIX_BATCH").map_or(10_000, |batch| batch.parse().unwrap());
        let ids: Vec<u64> = if partition_order.is_some() {
            let mut ids: Vec<u64> = (1..=rows).collect();
            ids.sort_by_cached_key(|id| partition_place(*id, crate::PartitionOrder::Token));
            ids
        } else if stream {
            (1..=rows).collect()
        } else if std::env::var("MATRIX_RANDOM").is_ok() {
            // In no order at all: every commit's rows lie all over the order. The other
            // shuffle comes in two stretches of the order that move up as it goes, 3 ids at a
            // time, so a commit's rows lie close together.
            let mut ids: Vec<u64> = (1..=rows).collect();
            let mut state = 0x9E37_79B9_7F4A_7C15u64;
            for at in (1..ids.len()).rev() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ids.swap(at, (state % (at as u64 + 1)) as usize);
            }
            ids
        } else {
            (0..rows)
                .map(|i| (i * 1_000_003 + 977) % rows + 1)
                .collect()
        };
        for chunk in ids.chunks(if stream { batch } else { 100_000 }) {
            add_batch(&sender, chunk.iter().copied(), matrix_name).await;
        }
        let quiet = |sender: &mpsc::Sender<PatternIndex>| {
            let sender = sender.clone();
            async move {
                // No repair to wait for: until the merges stop.
                let mut still = 0;
                let deadline = Instant::now() + Duration::from_secs(1800);
                loop {
                    let stats = sender.stats(make_index_key()).await.unwrap();
                    still = if stats.merges.running == 0 && !stats.rewrite.pending {
                        still + 1
                    } else {
                        0
                    };
                    if still >= 20 || Instant::now() > deadline {
                        return stats;
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        };
        let mut stats = if state == "unrepaired" || stream {
            // Rows in order may leave the repair nothing to do.
            quiet(&sender).await
        } else {
            settled(&sender, 3600, |stats| stats.overlap.fresh_rows == 0).await
        };
        if state == "churned" {
            for chunk in ids.chunks(100_000) {
                for &id in chunk.iter().filter(|id| *id % 10 == 3) {
                    rm_doc_no_wait(&sender, id).await;
                }
                add_batch(
                    &sender,
                    chunk.iter().copied().filter(|id| id % 10 == 7),
                    matrix_name,
                )
                .await;
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
            stats = settled(&sender, 3600, |stats| stats.overlap.fresh_rows == 0).await;
        }
        println!(
            "MATRIX {state} {rows} rows, cap {cap}: ready in {:.0}s at {} segments, depth {}/{:.2}, {} fresh rows, {} deleted, {} docs",
            started.elapsed().as_secs_f64(),
            stats.tantivy.segment_count,
            stats.overlap.depth_max,
            stats.overlap.depth_rows as f64 / stats.overlap.rows.max(1) as f64,
            stats.overlap.fresh_rows,
            stats.overlap.rows_deleted,
            stats.tantivy.num_docs,
        );
        // The runs of the live segments, lowest key first.
        let live: HashSet<(u32, Option<u64>)> = stats
            .segments
            .iter()
            .map(|segment| (segment.docs + segment.deleted, segment.sort_min))
            .collect();
        for (&(max_doc, min), &(runs, short, short_rows)) in RUN_SHAPES.lock().unwrap().iter() {
            if live.contains(&(max_doc, Some((min >> 64) as u64))) {
                println!(
                    "MATRIX segment of {max_doc} rows from {:x}: {runs} sorted runs, {short} coarse with {short_rows} rows",
                    min >> 64
                );
            }
        }
        // MATRIX_STATS: what one read of the index's statistics costs, as the node's gauge
        // refresh makes one a second for an index that is searched or written.
        if std::env::var("MATRIX_STATS").is_ok() {
            let mut micros = Vec::new();
            for _ in 0..30 {
                let clock = Instant::now();
                let stats = sender.stats(make_index_key()).await.unwrap();
                micros.push((clock.elapsed().as_micros() as u64, stats.segments.len()));
            }
            let segments = micros[0].1;
            let mut times: Vec<u64> = micros.iter().map(|(time, _)| *time).collect();
            times.sort_unstable();
            println!(
                "STATS|{state}|{rows}|{segments}|{}|{}|{}",
                times[0],
                times[times.len() / 2],
                times[times.len() - 1]
            );
        }
        let mut queries = vec![
            ("玩家", "every row"),
            ("末等", "1/5 even"),
            ("逢九之", "1/9 even"),
            ("逢九之数", "1/9 even, verified"),
            ("号玩家1", "1/9 lead digit, verified"),
            ("千禧", "1/1000 even"),
            ("用户42号", "1/10000 even, verified"),
            ("新人", "top tenth"),
            ("旧人", "bottom tenth"),
            ("旧稀客", "1/1000 in bottom tenth"),
            ("甲乙丙丁", "1/3000 of 1/3 candidates"),
            ("无此名", "miss"),
            // Anchored patterns: a value's start or end is checked against the store, row by row.
            ("用户42号%", "1/10000 prefix, verified"),
            ("%甲乙丙丁", "1/3000 suffix of 1/3 candidates, verified"),
            ("%新人", "2/3 of top tenth suffix, verified"),
        ];
        if sparse {
            queries = vec![
                ("逢九之", "1/9 even"),
                ("卅选", "1/30 even"),
                ("百选", "1/100 even"),
                ("叁选", "1/300 even"),
                ("千禧", "1/1000 even"),
                ("仟选", "1/3000 even"),
            ];
        }
        // A keyword is looked for anywhere in the value; a pattern with `%` is taken as it is.
        let pattern_of = |keyword: &str| {
            if keyword.contains('%') {
                keyword.to_string()
            } else {
                containing(keyword)
            }
        };
        // In the tiers index the middle of the order is the middle tier.
        let middle = if tiers { 2 } else { (rows / 2) as i64 };
        let shapes = [
            ("newest", 20, SearchWindow::default()),
            ("ascending", 20, ascending()),
            ("middle", 20, window(None, None, Some(sort_key(middle)))),
            ("newest", 200, SearchWindow::default()),
            ("newest", 1000, SearchWindow::default()),
        ];
        // MATRIX_KEYWORD keeps one keyword, MATRIX_SECS the time a cell may take: to profile one.
        let ab: Vec<(&str, u32)> = if std::env::var("MATRIX_AB").is_ok() {
            vec![
                ("take", OFF_TAKE),
                ("halving", OFF_HALVING),
                ("sweep", OFF_SWEEP),
                ("whole_read", OFF_WHOLE_READ),
                ("spread", OFF_SPREAD),
            ]
        } else {
            Vec::new()
        };
        let only = std::env::var("MATRIX_KEYWORD").ok();
        let secs: u64 = std::env::var("MATRIX_SECS").map_or(3, |secs| secs.parse().unwrap());
        for (keyword, label) in queries {
            if only.as_ref().is_some_and(|only| only != keyword) {
                continue;
            }
            for (shape, limit_n, window) in &shapes {
                if std::env::var("MATRIX_SHAPE").is_ok_and(|only| only != *shape) {
                    continue;
                }
                let before = sender.stats(make_index_key()).await.unwrap().walk;
                let whole_before = SEGMENTS_READ_WHOLE.load(Relaxed);
                let clock = Instant::now();
                let mut searches = 0u32;
                let mut returned = 0;
                let enough = if only.is_some() { u32::MAX } else { 100 };
                while searches < enough
                    && (searches < 3 || clock.elapsed() < Duration::from_secs(secs))
                {
                    returned = like(&sender, &pattern_of(keyword), *limit_n, window.clone())
                        .await
                        .primary_keys
                        .len();
                    searches += 1;
                }
                let wall = clock.elapsed().as_secs_f64() * 1e6 / f64::from(searches);
                let after = sender.stats(make_index_key()).await.unwrap().walk;
                let whole = SEGMENTS_READ_WHOLE.load(Relaxed) - whole_before;
                let per = |a: u64, b: u64| (a - b) as f64 / f64::from(searches);
                println!(
                    "MATRIX|{state}|{cap}|{label}|{shape}|{limit_n}|{returned}|{:.0}|{:.0}|{:.1}|{:.1}|{:.1}|{:.1}|{:.1}",
                    per(after.postings_scanned, before.postings_scanned),
                    per(after.store_reads, before.store_reads),
                    per(after.segments_opened, before.segments_opened),
                    per(after.segments_wasted, before.segments_wasted),
                    whole as f64 / f64::from(searches),
                    per(after.walk_nanos, before.walk_nanos) / 1e3,
                    wall,
                );
                // With a budget a request may return a short page: what the whole page costs,
                // request after request from the cursors, and the longest of the requests.
                if std::env::var("MATRIX_BUDGET_MS").is_ok() {
                    let (mut rows, mut requests, mut longest) = (0usize, 0u32, Duration::ZERO);
                    let mut next = window.clone();
                    let clock = Instant::now();
                    loop {
                        let asked = Instant::now();
                        let page =
                            like(&sender, &pattern_of(keyword), *limit_n - rows, next.clone())
                                .await;
                        longest = longest.max(asked.elapsed());
                        requests += 1;
                        rows += page.primary_keys.len();
                        match page.next_cursor {
                            // Not for ever, if requests get nowhere.
                            Some(cursor) if rows < *limit_n && requests < 500 => {
                                next.cursor = Some(cursor)
                            }
                            _ => break,
                        }
                    }
                    println!(
                        "BUDGET|{state}|{label}|{shape}|{limit_n}|{rows}|{requests}|{:.0}|{:.0}",
                        clock.elapsed().as_secs_f64() * 1e6,
                        longest.as_secs_f64() * 1e6,
                    );
                }
                // MATRIX_WHOLE: every segment read whole, none, and as the walk chooses, by turns:
                // where the two readings cross, and whether the choice follows.
                if std::env::var("MATRIX_WHOLE").is_ok() {
                    // [runs, whole, as chosen] x (searches, postings, walk nanos)
                    let mut sides = [[0u64; 3]; 3];
                    let mut probes = [[0u64; 6]; 3];
                    for turn in 0..9 {
                        let side = turn % 3;
                        // MATRIX_WHOLE=spread: the walk without the spread of runs in place
                        // of the one that never reads whole.
                        let runs_side = if std::env::var("MATRIX_WHOLE").as_deref() == Ok("spread")
                        {
                            OFF_SPREAD
                        } else {
                            OFF_WHOLE_READ
                        };
                        WALK_OFF.store([runs_side, FORCE_WHOLE_READ, 0][side], Relaxed);
                        let before = sender.stats(make_index_key()).await.unwrap().walk;
                        let probed = WALK_PROBE.each_ref().map(|probe| probe.load(Relaxed));
                        let clock = Instant::now();
                        let mut searches = 0u64;
                        while searches < 30
                            && (searches < 2 || clock.elapsed() < Duration::from_millis(300))
                        {
                            like(&sender, &pattern_of(keyword), *limit_n, window.clone()).await;
                            searches += 1;
                        }
                        let after = sender.stats(make_index_key()).await.unwrap().walk;
                        sides[side][0] += searches;
                        sides[side][1] += after.postings_scanned - before.postings_scanned;
                        sides[side][2] += after.walk_nanos - before.walk_nanos;
                        for (what, probe) in WALK_PROBE.iter().enumerate() {
                            probes[side][what] += probe.load(Relaxed) - probed[what];
                        }
                    }
                    WALK_OFF.store(0, Relaxed);
                    let per = |side: usize, what: usize| {
                        sides[side][what] as f64 / sides[side][0].max(1) as f64
                    };
                    println!(
                        "WHOLE|{state}|{label}|{shape}|{limit_n}|{:.0}|{:.0}|{:.1}|{:.1}|{:.1}",
                        per(0, 1),
                        per(1, 1),
                        per(0, 2) / 1e3,
                        per(1, 2) / 1e3,
                        per(2, 2) / 1e3,
                    );
                    for (side, name) in ["runs", "whole", "chosen"].iter().enumerate() {
                        let per = |what: usize| probes[side][what] as f64 / sides[side][0] as f64;
                        println!(
                            "PROBE|{label}|{shape}|{limit_n}|{name}|passes {:.1}|regions {:.0}|scorers {:.1}|scan {:.0} us|verify {:.0} us|sweep {:.0} us",
                            per(0),
                            per(1),
                            per(2),
                            per(3) / 1e3,
                            per(4) / 1e3,
                            per(5) / 1e3,
                        );
                    }
                }
                // MATRIX_AB: each mechanism of the walk off and on by turns on this one index,
                // so that the layout, which differs from build to build, is the same for both.
                for &(name, mechanism) in &ab {
                    // [off, on] x (searches, postings, store reads, walk nanos)
                    let mut sides = [[0u64; 4]; 2];
                    for turn in 0..6 {
                        let side = turn % 2;
                        WALK_OFF.store(if side == 0 { mechanism } else { 0 }, Relaxed);
                        let before = sender.stats(make_index_key()).await.unwrap().walk;
                        let clock = Instant::now();
                        let mut searches = 0u64;
                        while searches < 30
                            && (searches < 2 || clock.elapsed() < Duration::from_millis(300))
                        {
                            like(&sender, &pattern_of(keyword), *limit_n, window.clone()).await;
                            searches += 1;
                        }
                        let after = sender.stats(make_index_key()).await.unwrap().walk;
                        sides[side][0] += searches;
                        sides[side][1] += after.postings_scanned - before.postings_scanned;
                        sides[side][2] += after.store_reads - before.store_reads;
                        sides[side][3] += after.walk_nanos - before.walk_nanos;
                    }
                    WALK_OFF.store(0, Relaxed);
                    let per = |side: usize, what: usize| {
                        sides[side][what] as f64 / sides[side][0].max(1) as f64
                    };
                    println!(
                        "AB|{state}|{label}|{shape}|{limit_n}|{name}|{:.0}|{:.0}|{:.0}|{:.0}|{:.1}|{:.1}",
                        per(1, 1),
                        per(0, 1),
                        per(1, 2),
                        per(0, 2),
                        per(1, 3) / 1e3,
                        per(0, 3) / 1e3,
                    );
                }
            }
        }
    }

    /// The repair as searches see it: rows loaded shuffled, then searches kept on one place of
    /// the order (HEAT_SHAPE: middle, newest or ascending) while the repair goes on. Each second:
    /// the fresh rows left, and what those searches cost. Made to measure the repair's order by
    /// where searches go, which is out for now; kept for the trace of the repair's steps.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn the_repair_under_searches_kept_on_one_place() {
        let rows: u64 =
            std::env::var("MATRIX_ROWS").map_or(1_000_000, |rows| rows.parse().unwrap());
        let cap: u32 = std::env::var("MATRIX_CAP")
            .map_or(DEFAULT_SEGMENT_ROWS.get(), |cap| cap.parse().unwrap());
        MATRIX_ROWS.store(rows, Relaxed);
        WALK_OFF.store(OFF_TRACE_REPAIR, Relaxed);
        let shape = std::env::var("HEAT_SHAPE").unwrap_or_else(|_| "middle".into());
        let place = match shape.as_str() {
            "middle" => window(None, None, Some(sort_key((rows / 2) as i64))),
            "newest" => SearchWindow::default(),
            "ascending" => ascending(),
            other => panic!("HEAT_SHAPE: {other}"),
        };
        let sender = new(
            PatternIndexConfiguration {
                key: make_index_key(),
                options: capped_options(cap),
                key_layout: test_layout(),
            },
            make_table(bigint_key, |id, _| Some(CqlValue::BigInt(id as i64))),
            worker::new(),
            make_memory_actor(Allocate::Can),
            TEST_COMMIT_INTERVAL,
            MAX_UNCOMMITTED_THRESHOLD,
            None,
            SearchBudget::default(),
        );
        let ids: Vec<u64> = (0..rows)
            .map(|i| (i * 1_000_003 + 977) % rows + 1)
            .collect();
        let started = Instant::now();
        if std::env::var("HEAT_STRAY").is_ok() {
            // One row above every other, committed alone: adopted into the ordered tier, which
            // is what the first commits of a shuffled load do now and then (see the stall of
            // 2026-11-19 in docs/dev/pattern/postponed.md).
            add_batch(&sender, [0], matrix_name).await;
        }
        let total = rows + u64::from(std::env::var("HEAT_STRAY").is_ok());
        if std::env::var("REPAIR_NO_PLACE").is_ok() {
            WALK_OFF.fetch_or(OFF_POOL_PLACE, Relaxed);
        }
        if std::env::var("HEAT_NO_HOLD_BOUND").is_ok() {
            WALK_OFF.fetch_or(OFF_HOLD_BOUND, Relaxed);
        }
        // HEAT_MERGE_ACROSS: ordered neighbours merged whatever fresh pieces reach between them.
        if std::env::var("HEAT_MERGE_ACROSS").is_ok() {
            WALK_OFF.fetch_or(OFF_FRESH_BETWEEN, Relaxed);
        }
        for chunk in ids.chunks(100_000) {
            add_batch(&sender, chunk.iter().copied(), matrix_name).await;
        }
        println!("HEAT loaded in {:.0}s", started.elapsed().as_secs_f64());
        let started = Instant::now();
        let mut moved = (0u64, Instant::now(), false);
        let mut settling: Option<(usize, u32)> = None;
        loop {
            let before = sender.stats(make_index_key()).await.unwrap();
            for _ in 0..20 {
                search_ordered(&sender, "玩家", 20, place.clone()).await;
            }
            let after = sender.stats(make_index_key()).await.unwrap();
            let per = |a: u64, b: u64| (a - b) as f64 / 20.0;
            println!(
                "HEAT|{shape}|{:.0}|{}|{}|{:.1}|{:.1}|{:.0}|phase {}|merges {}/{}/{}|segments {} fresh {} largest {}",
                started.elapsed().as_secs_f64(),
                after.overlap.fresh_rows,
                after.rewrite.steps,
                per(after.walk.segments_opened, before.walk.segments_opened),
                per(after.walk.segments_wasted, before.walk.segments_wasted),
                per(after.walk.walk_nanos, before.walk.walk_nanos) / 1e3,
                REPAIR_PHASE.load(Relaxed),
                after.merges.started,
                after.merges.ended,
                after.merges.running,
                after.tantivy.segment_count,
                after.overlap.fresh_segments,
                after
                    .segments
                    .iter()
                    .map(|segment| segment.docs)
                    .max()
                    .unwrap_or(0),
            );
            // A stall: the layout as it stands, once.
            if moved.0 != after.overlap.fresh_rows {
                moved = (after.overlap.fresh_rows, Instant::now(), false);
            } else if !moved.2 && moved.1.elapsed() > Duration::from_secs(45) {
                moved.2 = true;
                for segment in &after.segments {
                    println!(
                        "HEAT stalled: segment of {} rows, {} deleted, {:x?}..{:x?}",
                        segment.docs, segment.deleted, segment.sort_min, segment.sort_max
                    );
                }
            }
            // Repaired, and then until the merges that follow have ended: no merge running and
            // as many segments for five seconds.
            if after.overlap.fresh_rows == 0 && after.tantivy.num_docs == total {
                if settling.is_none() {
                    println!("HEAT repaired in {:.0}s", started.elapsed().as_secs_f64());
                }
                let segments = after.tantivy.segment_count;
                settling = match settling {
                    Some((count, quiet)) if count == segments && after.merges.running == 0 => {
                        Some((count, quiet + 1))
                    }
                    _ => Some((segments, 0)),
                };
            }
            if settling.is_some_and(|(_, quiet)| quiet >= 5)
                || started.elapsed() > Duration::from_secs(1800)
            {
                break;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    /// What the bounded lock holds of the repair are worth: rows loaded shuffled, then a writer
    /// adding a row every 10 ms, stamped as written then, while the repair goes on. Each second:
    /// the fresh rows left and how long the rows took to be searchable, by bucket.
    /// MATRIX_OFF=lock_chunks repairs each part of a step under one hold.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn the_writes_waiting_on_a_repair() {
        let rows: u64 =
            std::env::var("MATRIX_ROWS").map_or(1_000_000, |rows| rows.parse().unwrap());
        let cap: u32 = std::env::var("MATRIX_CAP")
            .map_or(DEFAULT_SEGMENT_ROWS.get(), |cap| cap.parse().unwrap());
        MATRIX_ROWS.store(rows, Relaxed);
        let off = std::env::var("MATRIX_OFF").is_ok_and(|off| off == "lock_chunks");
        WALK_OFF.store(
            if off { OFF_LOCK_CHUNKS } else { 0 }
                | OFF_TRACE_REPAIR
                | if std::env::var("MATRIX_NO_HANDOVER").is_ok() {
                    OFF_HANDOVER
                } else {
                    0
                }
                | if std::env::var("REPAIR_NO_PLACE").is_ok() {
                    OFF_POOL_PLACE
                } else {
                    0
                },
            Relaxed,
        );
        let sender = new(
            PatternIndexConfiguration {
                key: make_index_key(),
                options: capped_options(cap),
                key_layout: test_layout(),
            },
            make_table(bigint_key, |id, _| Some(CqlValue::BigInt(id as i64))),
            worker::new(),
            make_memory_actor(Allocate::Can),
            COMMIT_INTERVAL,
            MAX_UNCOMMITTED_THRESHOLD,
            None,
            SearchBudget::default(),
        );
        let ids: Vec<u64> = (0..rows)
            .map(|i| (i * 1_000_003 + 977) % rows + 1)
            .collect();
        let started = Instant::now();
        for chunk in ids.chunks(100_000) {
            add_batch(&sender, chunk.iter().copied(), matrix_name).await;
        }
        println!("LOCK loaded in {:.0}s", started.elapsed().as_secs_f64());
        let buckets = vec![0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0];
        let histogram = prometheus::Histogram::with_opts(
            prometheus::HistogramOpts::new("lag", "lag").buckets(buckets.clone()),
        )
        .unwrap();
        let writer = tokio::spawn({
            let sender = sender.clone();
            let histogram = histogram.clone();
            async move {
                let mut primary = rows;
                loop {
                    primary += 1;
                    let written = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap();
                    sender
                        .add_document(
                            test_partition_id(),
                            primary.into(),
                            matrix_name(primary),
                            AsyncInProgress::cdc(
                                histogram.clone(),
                                crate::Timestamp::from_millis(written.as_millis() as u64),
                            ),
                        )
                        .await
                        .unwrap();
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        });
        let limit: u64 = std::env::var("MATRIX_SECS").map_or(600, |secs| secs.parse().unwrap());
        let started = Instant::now();
        let mut worst = 0.0f64;
        let mut last = (0u64, 0.0f64);
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            let stats = sender.stats(make_index_key()).await.unwrap();
            let now = (histogram.get_sample_count(), histogram.get_sample_sum());
            let mean = (now.1 - last.1) / (now.0 - last.0).max(1) as f64;
            worst = worst.max(mean);
            println!(
                "LOCK|{}|{:.0}|{}|{}|{}|{:.2}",
                if off { "off" } else { "on" },
                started.elapsed().as_secs_f64(),
                stats.overlap.fresh_rows,
                stats.rewrite.steps,
                now.0 - last.0,
                mean,
            );
            last = now;
            if stats.overlap.fresh_rows < 400_000 || started.elapsed().as_secs() > limit {
                break;
            }
        }
        writer.abort();
        let proto = histogram.get_sample_count();
        println!(
            "LOCK done in {:.0}s: {proto} rows, mean {:.2}s, worst second's mean {worst:.2}s",
            started.elapsed().as_secs_f64(),
            histogram.get_sample_sum() / proto.max(1) as f64,
        );
        use prometheus::core::Collector;
        let family = &histogram.collect()[0];
        for bucket in family.get_metric()[0].get_histogram().get_bucket() {
            println!(
                "LOCK bucket <= {}s: {}",
                bucket.upper_bound(),
                bucket.cumulative_count()
            );
        }
    }

    /// Indexing and searches sharing the workers: rows written shuffled as fast as they are
    /// taken, stamped as written then, with MIXED_SEARCHERS (8) searches kept going. WORKER=fifo
    /// serves the workers' tasks first come first served, as without the two classes. Reports the
    /// rows' wait until searchable, the searches' time, and what held the repair.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn indexing_and_searches_sharing_the_workers() {
        let rows: u64 =
            std::env::var("MATRIX_ROWS").map_or(1_000_000, |rows| rows.parse().unwrap());
        let cap: u32 = std::env::var("MATRIX_CAP")
            .map_or(DEFAULT_SEGMENT_ROWS.get(), |cap| cap.parse().unwrap());
        let searchers: usize =
            std::env::var("MIXED_SEARCHERS").map_or(8, |searchers| searchers.parse().unwrap());
        MATRIX_ROWS.store(rows, Relaxed);
        let fifo = std::env::var("WORKER").is_ok_and(|worker| worker == "fifo");
        worker::FIFO.store(fifo, Relaxed);
        if std::env::var("REPAIR_NO_PLACE").is_ok() {
            WALK_OFF.fetch_or(OFF_POOL_PLACE, Relaxed);
        }
        let sender = new(
            PatternIndexConfiguration {
                key: make_index_key(),
                options: capped_options(cap),
                key_layout: test_layout(),
            },
            make_table(bigint_key, |id, _| Some(CqlValue::BigInt(id as i64))),
            worker::new(),
            make_memory_actor(Allocate::Can),
            COMMIT_INTERVAL,
            MAX_UNCOMMITTED_THRESHOLD,
            None,
            SearchBudget::default(),
        );
        let histogram = prometheus::Histogram::with_opts(
            prometheus::HistogramOpts::new("lag", "lag")
                .buckets(vec![0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 40.0]),
        )
        .unwrap();
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let times = Arc::new(Mutex::new(Vec::<u32>::new()));
        let mut tasks = Vec::new();
        for _ in 0..searchers {
            let (sender, done, times) = (sender.clone(), done.clone(), times.clone());
            tasks.push(tokio::spawn(async move {
                while !done.load(Relaxed) {
                    let clock = Instant::now();
                    search_ordered(&sender, "号玩家1", 200, SearchWindow::default()).await;
                    times
                        .lock()
                        .unwrap()
                        .push(clock.elapsed().as_micros() as u32);
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            }));
        }
        let started = Instant::now();
        let limit: u64 = std::env::var("MATRIX_SECS").map_or(900, |secs| secs.parse().unwrap());
        let mut rows = rows;
        let total = rows;
        for i in 0..total {
            if i % 100_000 == 0 {
                println!(
                    "MIXED sent {i} rows in {:.0}s, {} searches",
                    started.elapsed().as_secs_f64(),
                    times.lock().unwrap().len()
                );
                if started.elapsed().as_secs() > limit {
                    rows = i;
                    break;
                }
            }
            let primary = (i * 1_000_003 + 977) % total + 1;
            let written = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap();
            sender
                .add_document(
                    test_partition_id(),
                    primary.into(),
                    matrix_name(primary),
                    AsyncInProgress::cdc(
                        histogram.clone(),
                        crate::Timestamp::from_millis(written.as_millis() as u64),
                    ),
                )
                .await
                .unwrap();
        }
        let sent = started.elapsed().as_secs_f64();
        while histogram.get_sample_count() < rows && started.elapsed().as_secs() < 1800 {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        let indexed = started.elapsed().as_secs_f64();
        let stats = sender.stats(make_index_key()).await.unwrap();
        done.store(true, Relaxed);
        for task in tasks {
            task.await.unwrap();
        }
        let mut times = std::mem::take(&mut *times.lock().unwrap());
        times.sort_unstable();
        let at = |q: f64| times[((times.len() - 1) as f64 * q) as usize];
        let gate: Vec<u64> = GATE_TICKS.iter().map(|ticks| ticks.load(Relaxed)).collect();
        println!(
            "MIXED|{}|{rows}|{searchers}|sent {sent:.0}s|indexed {indexed:.0}s|wait mean {:.2}s|searches {}|p50 {}|p90 {}|p99 {}|max {} us|fresh {}|steps {}|ticks pending {} lagging {}",
            if fifo { "fifo" } else { "classes" },
            histogram.get_sample_sum() / histogram.get_sample_count().max(1) as f64,
            times.len(),
            at(0.5),
            at(0.9),
            at(0.99),
            at(1.0),
            stats.overlap.fresh_rows,
            stats.rewrite.steps,
            gate[0],
            gate[1],
        );
        use prometheus::core::Collector;
        for bucket in histogram.collect()[0].get_metric()[0]
            .get_histogram()
            .get_bucket()
        {
            println!(
                "MIXED bucket <= {}s: {}",
                bucket.upper_bound(),
                bucket.cumulative_count()
            );
        }
    }

    /// An index built by a full scan beside the searches of another, on one pool of workers:
    /// what the build costs the searches, and the searches the build. BUILD_SEARCHERS (8)
    /// searches are kept going on a settled index of MATRIX_ROWS rows, each BUILD_THINK_MS (0)
    /// after the last; after 10 s of them alone, a second index takes BUILD_ROWS rows of a full
    /// scan. BUILD_CLASS=indexing sends those as indexing, as before the background class.
    /// BUILD_INDEXES (1) indexes are built at once, each of BUILD_ROWS rows; with BUILD_REPAIR
    /// the searches go on until every one of them is repaired, and are reported for that time.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn a_build_beside_the_searches_of_another_index() {
        let env = |name: &str, default: u64| -> u64 {
            std::env::var(name).map_or(default, |value| value.parse().unwrap())
        };
        let rows = env("MATRIX_ROWS", 1_000_000);
        let build_rows = env("BUILD_ROWS", 1_000_000);
        let searchers = env("BUILD_SEARCHERS", 8);
        let think = Duration::from_millis(env("BUILD_THINK_MS", 0));
        let class = std::env::var("BUILD_CLASS").unwrap_or_else(|_| "background".into());
        worker::BACKGROUND_OFF.store(class == "indexing", Relaxed);
        // BUILD_CAP=off: as many background tasks at once as there are workers.
        worker::BACKGROUND_CAP_OFF.store(
            std::env::var("BUILD_CAP").is_ok_and(|cap| cap == "off"),
            Relaxed,
        );
        // BUILD_COMMIT=inline: commits on the runtime thread their worker is on.
        crate::tantivy::OFF_THE_RUNTIME_OFF.store(
            std::env::var("BUILD_COMMIT").is_ok_and(|commit| commit == "inline"),
            Relaxed,
        );
        // BUILD_META_LOCK=sleeping: the meta lock as a file of Tantivy's RAM directory.
        crate::tantivy::directory::SLEEPING.store(
            std::env::var("BUILD_META_LOCK").is_ok_and(|lock| lock == "sleeping"),
            Relaxed,
        );
        MATRIX_ROWS.store(rows.max(build_rows), Relaxed);
        let worker = worker::new();
        let index = |worker: &async_channel::Sender<Worker>| {
            new(
                PatternIndexConfiguration {
                    key: make_index_key(),
                    options: capped_options(DEFAULT_SEGMENT_ROWS.get()),
                    key_layout: test_layout(),
                },
                make_table(bigint_key, |id, _| Some(CqlValue::BigInt(id as i64))),
                worker.clone(),
                make_memory_actor(Allocate::Can),
                COMMIT_INTERVAL,
                MAX_UNCOMMITTED_THRESHOLD,
                None,
                SearchBudget {
                    time: Some(DEFAULT_SEARCH_BUDGET),
                    work: None,
                },
            )
        };
        let served = index(&worker);
        let shuffled = |rows: u64| (0..rows).map(move |i| (i * 1_000_003 + 977) % rows + 1);
        let ids: Vec<u64> = shuffled(rows).collect();
        for chunk in ids.chunks(100_000) {
            add_batch(&served, chunk.iter().copied(), matrix_name).await;
        }
        settled(&served, 3600, |stats| stats.overlap.fresh_rows == 0).await;

        // 0: warm-up, 1: the searches alone, 2: beside the build, 3: beside the repairs, 4: done.
        let phase = Arc::new(AtomicU32::new(0));
        let times = Arc::new(Mutex::new([Vec::<u32>::new(), Vec::new(), Vec::new()]));
        // BUILD_TRACE: the searches beside the build that took 5 ms or more, and the pool's
        // tasks that waited or ran 2 ms or more, by the time they started.
        let tracing = std::env::var("BUILD_TRACE").is_ok();
        let slow = Arc::new(Mutex::new(Vec::<(Instant, Duration)>::new()));
        let mut tasks = Vec::new();
        for _ in 0..searchers {
            let (served, phase, times) = (served.clone(), phase.clone(), times.clone());
            let slow = slow.clone();
            tasks.push(tokio::spawn(async move {
                loop {
                    let at = phase.load(Relaxed);
                    if at == 4 {
                        break;
                    }
                    let clock = Instant::now();
                    search_ordered(&served, "号玩家1", 200, SearchWindow::default()).await;
                    let took = clock.elapsed().as_micros() as u32;
                    if (1..=3).contains(&at) {
                        times.lock().unwrap()[at as usize - 1].push(took);
                    }
                    if at == 2 && took >= 5000 {
                        slow.lock().unwrap().push((clock, clock.elapsed()));
                    }
                    if !think.is_zero() {
                        tokio::time::sleep(think).await;
                    }
                }
            }));
        }
        // Where the time goes: the pool's totals per class, and the CPU time of the process's
        // threads by name, read before and after each phase.
        let cpu_by_thread = || -> std::collections::BTreeMap<String, f64> {
            let mut cpu = std::collections::BTreeMap::new();
            for task in std::fs::read_dir("/proc/self/task").unwrap().flatten() {
                let Ok(stat) = std::fs::read_to_string(task.path().join("stat")) else {
                    continue;
                };
                // pid (comm) state ...: the name may hold spaces, so split at the last ')'.
                let (head, tail) = stat.rsplit_once(')').unwrap();
                let name = head.split_once('(').unwrap().1;
                let name = name.trim_end_matches(|c: char| c.is_ascii_digit() || c == '-');
                let fields: Vec<&str> = tail.split_whitespace().collect();
                let ticks: f64 =
                    fields[11].parse::<f64>().unwrap() + fields[12].parse::<f64>().unwrap();
                *cpu.entry(name.to_string()).or_insert(0.0) += ticks / 100.0;
            }
            cpu
        };
        type Reading = (
            [(worker::TaskClass, u64, f64, f64); 3],
            std::collections::BTreeMap<String, f64>,
        );
        let reading = || -> Reading { (worker::STATS.snapshot(), cpu_by_thread()) };
        let spent = |label: &str, before: &Reading, after: &Reading, over: Duration| {
            for (b, a) in before.0.iter().zip(&after.0) {
                let tasks = (a.1 - b.1).max(1) as f64;
                println!(
                    "BUILD pool {label}: {} tasks {:.0}, in the queue {:.0} us each, running {:.0} us each, {:.2} workers busy",
                    a.0.name(),
                    tasks,
                    (a.2 - b.2) * 1e6 / tasks,
                    (a.3 - b.3) * 1e6 / tasks,
                    (a.3 - b.3) / over.as_secs_f64(),
                );
            }
            let cores: Vec<String> = after
                .1
                .iter()
                .map(|(name, cpu)| (name, cpu - before.1.get(name).copied().unwrap_or(0.0)))
                .filter(|(_, cpu)| *cpu / over.as_secs_f64() >= 0.02)
                .map(|(name, cpu)| format!("{name} {:.2}", cpu / over.as_secs_f64()))
                .collect();
            println!("BUILD cores {label}: {}", cores.join(", "));
        };
        tokio::time::sleep(Duration::from_secs(2)).await;
        phase.store(1, Relaxed);
        let alone = Duration::from_secs(10);
        let before = reading();
        tokio::time::sleep(alone).await;
        spent("alone", &before, &reading(), alone);

        phase.store(2, Relaxed);
        let before = reading();
        let started = Instant::now();
        if tracing {
            *worker::TRACE.lock().unwrap() = Some(Vec::new());
        }
        // Controls in place of a build, BUILD_SPIN threads for 10 s: BUILD_CLASS=spin keeps
        // them computing, which takes cores and nothing else; BUILD_CLASS=memory keeps them
        // copying 64 MB, which takes the memory and the caches too.
        if class == "spin" || class == "memory" {
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let threads: Vec<_> = (0..env("BUILD_SPIN", 2))
                .map(|_| {
                    let (stop, memory) = (stop.clone(), class == "memory");
                    std::thread::spawn(move || {
                        let from = vec![1u8; 64 << 20];
                        let mut to = vec![0u8; 64 << 20];
                        let mut sum = 0u64;
                        while !stop.load(Relaxed) {
                            if memory {
                                to.copy_from_slice(&from);
                                sum += u64::from(to[sum as usize % to.len()]);
                            } else {
                                for i in 0..1_000_000u64 {
                                    sum = sum.wrapping_mul(6364136223846793005).wrapping_add(i);
                                }
                            }
                        }
                        std::hint::black_box(sum);
                    })
                })
                .collect();
            tokio::time::sleep(alone).await;
            stop.store(true, Relaxed);
            for thread in threads {
                thread.join().unwrap();
            }
            let over = started.elapsed();
            spent("beside the control", &before, &reading(), over);
            phase.store(4, Relaxed);
            for task in tasks {
                task.await.unwrap();
            }
            let times = std::mem::take(&mut *times.lock().unwrap());
            for (label, mut times, over) in [
                ("alone", times[0].clone(), alone),
                ("beside", times[1].clone(), over),
            ] {
                times.sort_unstable();
                let at = |q: f64| times[((times.len() - 1) as f64 * q) as usize];
                println!(
                    "BUILD|{class}|{searchers} searchers|{label}: {:.0} q/s p50 {} p90 {} p99 {} max {} us",
                    times.len() as f64 / over.as_secs_f64(),
                    at(0.5),
                    at(0.9),
                    at(0.99),
                    at(1.0)
                );
            }
            return;
        }
        let built: Vec<_> = (0..env("BUILD_INDEXES", 1))
            .map(|_| index(&worker))
            .collect();
        let scans: Vec<_> = built
            .iter()
            .map(|built| {
                let built = built.clone();
                tokio::spawn(async move {
                    for primary in shuffled(build_rows) {
                        let (tx, _rx) = mpsc::channel(1);
                        built
                            .add_document(
                                test_partition_id(),
                                primary.into(),
                                matrix_name(primary),
                                AsyncInProgress::fullscan(tx),
                            )
                            .await
                            .unwrap();
                    }
                })
            })
            .collect();
        for scan in scans {
            scan.await.unwrap();
        }
        let sent = started.elapsed();
        for built in &built {
            while built
                .stats(make_index_key())
                .await
                .unwrap()
                .tantivy
                .num_docs
                < build_rows
            {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
        let searchable = started.elapsed();
        spent("beside the build", &before, &reading(), searchable);
        let mut repaired = Duration::ZERO;
        if std::env::var("BUILD_REPAIR").is_ok() {
            phase.store(3, Relaxed);
            let before = reading();
            let started = Instant::now();
            for built in &built {
                settled(built, 3600, |stats| stats.overlap.fresh_rows == 0).await;
            }
            repaired = started.elapsed();
            spent("beside the repairs", &before, &reading(), repaired);
        }
        if let Some(trace) = worker::TRACE.lock().unwrap().take() {
            let ms = |at: Instant| at.saturating_duration_since(started).as_secs_f64() * 1e3;
            let mut lines: Vec<(f64, String)> = trace
                .iter()
                .map(|(at, class, waited, ran)| {
                    (
                        ms(*at),
                        format!(
                            "task {} waited {:.1} ms ran {:.1} ms",
                            class.name(),
                            waited.as_secs_f64() * 1e3,
                            ran.as_secs_f64() * 1e3
                        ),
                    )
                })
                .chain(slow.lock().unwrap().iter().map(|(at, took)| {
                    (
                        ms(*at),
                        format!("search took {:.1} ms", took.as_secs_f64() * 1e3),
                    )
                }))
                .collect();
            lines.sort_by(|a, b| a.0.total_cmp(&b.0));
            for (at, line) in lines {
                println!("TRACE {at:9.1} ms {line}");
            }
        }
        phase.store(4, Relaxed);
        for task in tasks {
            task.await.unwrap();
        }
        let times = std::mem::take(&mut *times.lock().unwrap());
        let report = |mut times: Vec<u32>, over: Duration| {
            if times.is_empty() {
                return "no searches".to_string();
            }
            times.sort_unstable();
            let at = |q: f64| times[((times.len() - 1) as f64 * q) as usize];
            format!(
                "{:.0} q/s p50 {} p90 {} p99 {} max {} us",
                times.len() as f64 / over.as_secs_f64(),
                at(0.5),
                at(0.9),
                at(0.99),
                at(1.0)
            )
        };
        let [before, during, repairs] = times;
        if !repaired.is_zero() {
            println!(
                "BUILD repairs|{} indexes|repaired in {:.1}s|searches beside the repairs: {}",
                built.len(),
                repaired.as_secs_f64(),
                report(repairs, repaired),
            );
        }
        println!(
            "BUILD meta lock: {} takes waited 1 ms or more, {:.2}s in all",
            crate::tantivy::directory::WAITS[0].load(Relaxed),
            crate::tantivy::directory::WAITS[1].load(Relaxed) as f64 / 1e9,
        );
        println!(
            "BUILD|{class}|{searchers} searchers, think {} ms|{rows} rows served, {build_rows} built|alone: {}|beside the build: {}|rows sent in {:.1}s, searchable in {:.1}s",
            think.as_millis(),
            report(before, alone),
            report(during, searchable),
            sent.as_secs_f64(),
            searchable.as_secs_f64(),
        );
    }

    /// CDC changes that lag behind the table hold the repair back while they arrive, however
    /// many fresh rows pile up; once they stop, the idle ticks repair everything. AWS run 7a300b9d
    /// repaired while its indexes ingested a load minutes behind, and caught up 2.8x later.
    #[rstest]
    #[timeout(Duration::from_secs(60))]
    #[tokio::test]
    async fn cdc_changes_lagging_behind_hold_the_repair() {
        let cap = 100;
        let sender = make_sender_with_options(capped_options(cap));
        let written = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            - Duration::from_secs(60);
        let histogram =
            prometheus::Histogram::with_opts(prometheus::HistogramOpts::new("lag", "lag")).unwrap();
        let rows: Vec<u64> = shuffled(3000).collect();
        for (n, chunk) in rows.chunks(5).enumerate() {
            for &primary in chunk {
                sender
                    .add_document(
                        test_partition_id(),
                        primary.into(),
                        named(primary),
                        AsyncInProgress::cdc(
                            histogram.clone(),
                            crate::Timestamp::from_millis(written.as_millis() as u64),
                        ),
                    )
                    .await
                    .unwrap();
            }
            // A steady stream, so that no tick finds the index idle.
            tokio::time::sleep(Duration::from_millis(1)).await;
            if n % 100 == 99 {
                let stats = sender.stats(make_index_key()).await.unwrap();
                assert_eq!(
                    stats.rewrite.steps, 0,
                    "repaired while lagging: {:?} {:?}",
                    stats.rewrite, stats.overlap
                );
            }
        }
        let stats = settled(&sender, 45, |stats| {
            stats.tantivy.num_docs == 3000 && stats.overlap.fresh_rows == 0
        })
        .await;
        assert_eq!(stats.overlap.depth_max, 1, "{:?}", stats.overlap);
    }

    /// Nothing is repaired while a full scan goes on, idle tick or not. After it, an idle tick
    /// repairs whatever there is; while the index ingests, the repair waits for the fresh tier
    /// to hold `LOAD_FRESH_ROWS` rows, or a stretch past the cap.
    #[test]
    fn the_repair_waits_for_the_scan_and_for_rows_to_repair() {
        let under = LOAD_FRESH_ROWS - 1;
        // (idle, catching up, fresh rows, overfull)
        assert!(!repair_due(true, false, 0, false));
        assert!(repair_due(true, false, 1, false));
        assert!(!repair_due(true, true, LOAD_FRESH_ROWS, true));
        assert!(!repair_due(false, true, LOAD_FRESH_ROWS, true));
        assert!(!repair_due(false, false, under, false));
        assert!(repair_due(false, false, LOAD_FRESH_ROWS, false));
        assert!(repair_due(false, false, 0, true));
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
        // Between a segment holding the end of value 3 and one holding the start of value 4, it
        // overlaps neither: where it belongs.
        let segments = [
            ordered(20_000, layout(3, 0), layout(3, u64::MAX - 41), 1),
            ordered(20_000, layout(4, 91), layout(4, u64::MAX), 2),
            piece(500, min, max, 3),
        ];
        assert_eq!(ordered_tier(&segments), [true, true, true]);
        // The bounds as they were: both values whole, over both neighbours, to be repaired again
        // and again.
        let widened = piece(500, layout(3, 0), layout(4, u64::MAX), 3);
        let segments = [segments[0], segments[1], widened];
        assert_eq!(ordered_tier(&segments), [true, true, false]);
    }

    /// A request reads its budget from the configuration as it starts: a reload changes it for
    /// the requests that follow, without a restart.
    #[test]
    fn the_search_budget_follows_a_configuration_reload() {
        let config = |budget| {
            Arc::new(Config {
                pattern_search_budget: budget,
                ..Config::default()
            })
        };
        let (tx, rx) = watch::channel(config(None));
        let source = SearchBudgetSource::from(rx);
        assert_eq!((source.0)().time, Some(DEFAULT_SEARCH_BUDGET));
        tx.send(config(Some(Duration::from_millis(7)))).unwrap();
        assert_eq!((source.0)().time, Some(Duration::from_millis(7)));
        // Zero is no limit.
        tx.send(config(Some(Duration::ZERO))).unwrap();
        assert_eq!((source.0)().time, None);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn count_before_any_document_is_zero() {
        let sender = make_sender();

        assert_eq!(count(&sender).await, 0);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn add_document_increments_count() {
        let sender = make_sender();

        add_doc(&sender, 1, "AB-1024-XL").await;
        add_doc(&sender, 2, "CD-2048-S").await;

        assert_eq!(count(&sender).await, 2);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn remove_document_decrements_count() {
        let sender = make_sender();

        add_doc(&sender, 1, "AB-1024-XL").await;
        add_doc(&sender, 2, "CD-2048-S").await;
        rm_doc(&sender, 2).await;

        assert_eq!(count(&sender).await, 1);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn add_document_rejected_when_memory_exhausted() {
        let sender = make_sender_with_memory(Allocate::Cannot);

        add_doc(&sender, 1, "should not be indexed").await;

        assert_eq!(count(&sender).await, 0);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn like_before_any_document_returns_nothing() {
        let sender = make_sender();

        assert_eq!(like_ids(&sender, "%abc%").await, Vec::<i64>::new());
    }

    #[rstest]
    #[case::exact("AB-1024", &[1])]
    #[case::prefix("AB-1024%", &[1, 2])]
    #[case::suffix("%AB-1024", &[1, 3])]
    #[case::infix("%AB-1024%", &[1, 2, 3, 4])]
    #[case::short_infix("%XL%", &[2, 3, 4])]
    #[case::any_char("AB-_024", &[1, 6])]
    #[case::prefix_and_suffix("AB-%-XL", &[2])]
    #[case::two_infixes("%1024%S%", &[4])]
    #[case::case_sensitive("ab-%", &[7])]
    #[case::empty("", &[8])]
    #[case::percent("AB-50%", &[9, 10, 12])]
    #[case::escaped_percent(r"AB-50\%", &[9])]
    #[case::underscore("AB_50", &[11, 12, 13])]
    #[case::escaped_underscore(r"AB\_50", &[11])]
    #[case::escaped_backslash(r"AB\\50", &[13])]
    #[case::backslash_at_end(r"AB\", &[14])]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn like_matches_values(#[case] pattern: &str, #[case] expected: &[i64]) {
        let sender = make_sender();
        // Row 5 matches no pattern. Every value that holds all grams of a pattern also matches it,
        // so the result does not depend on the regex.
        add_docs(
            &sender,
            &[
                (1, "AB-1024"),
                (2, "AB-1024-XL"),
                (3, "XL-AB-1024"),
                (4, "XL-AB-1024-S"),
                (5, "CD-2048-M"),
                (6, "AB-2024"),
                (7, "ab-1024"),
                (8, ""),
                (9, "AB-50%"),
                (10, "AB-500"),
                (11, "AB_50"),
                (12, "AB-50"),
                (13, r"AB\50"),
                (14, r"AB\"),
            ],
        )
        .await;

        assert_eq!(like_ids(&sender, pattern).await, expected);
    }

    #[rstest]
    #[case::any_seq("%", &[1, 2, 3])]
    #[case::any_seqs("%%", &[1, 2, 3])]
    #[case::any_char("_", &[2])]
    #[case::any_chars("__", &[3])]
    #[case::any_char_between_any_seqs("%_%", &[2, 3])]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn like_matches_patterns_without_literal_characters(
        #[case] pattern: &str,
        #[case] expected: &[i64],
    ) {
        let sender = make_sender();
        // Values of 0, 1 and 2 characters.
        add_docs(&sender, &[(1, ""), (2, "a"), (3, "ab")]).await;

        assert_eq!(like_ids(&sender, pattern).await, expected);
    }

    #[rstest]
    #[case::exact("将军", &[1])]
    #[case::prefix("将军%", &[1, 2])]
    #[case::suffix("%将军", &[1, 3])]
    #[case::infix("%将军%", &[1, 2, 3, 4])]
    #[case::any_char("将_", &[1, 6])]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn like_supports_cjk(#[case] pattern: &str, #[case] expected: &[i64]) {
        let sender = make_sender();
        // Row 5 matches no pattern and row 6 only the one with '_'.
        add_docs(
            &sender,
            &[
                (1, "将军"),
                (2, "将军来了"),
                (3, "李将军"),
                (4, "李将军来了"),
                (5, "元帅"),
                (6, "将领"),
            ],
        )
        .await;

        assert_eq!(like_ids(&sender, pattern).await, expected);
    }

    #[rstest]
    // 'AB-10-1024' holds every 3-character piece of 'AB-1024' (AB-, B-1, -10, 102, 024), but the
    // pieces are not next to each other.
    #[case::grams_apart("%AB-1024%", "AB-10-1024")]
    // 'AB-1024-XL' holds 'XL' and '1024', but '1024' comes before 'XL'.
    #[case::wrong_order("%XL%1024%", "AB-1024-XL")]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn like_filters_out_values_that_grams_cannot_prove_by_applying_regex(
        #[case] pattern: &str,
        #[case] value_holding_all_grams: &str,
    ) {
        let sender = make_sender();
        // Both rows hold every gram of the pattern, so the gram query returns both of them as
        // candidates. Only row 2 really matches the pattern.
        add_docs(&sender, &[(1, value_holding_all_grams), (2, "XL-AB-1024")]).await;

        // The regex reads the stored value of each candidate and drops row 1.
        assert_eq!(like_ids(&sender, pattern).await, [2]);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn like_excludes_a_removed_document() {
        let sender = make_sender();
        add_docs(&sender, &[(1, "abc"), (2, "abcd")]).await;
        rm_doc(&sender, 2).await;

        assert_eq!(like_ids(&sender, "abc%").await, [1]);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn like_returns_a_row_indexed_twice_once() {
        let sender = make_sender();
        add_docs(&sender, &[(1, "abc"), (1, "abc")]).await;

        assert_eq!(like_ids(&sender, "%abc%").await, [1]);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn like_respects_limit() {
        let sender = make_sender();
        add_docs(&sender, &[(1, "abc"), (2, "abc"), (3, "abc")]).await;

        let page = like(&sender, "%abc%", 2, SearchWindow::default()).await;

        assert_eq!(page.primary_keys.len(), 2);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn like_matches_a_mark_character_in_the_pattern_only_in_the_value() {
        let sender = make_sender();
        // Both rows hold the gram '\u{2}ab': row 1 by its start mark, row 2 by its own text.
        add_docs(&sender, &[(1, "abc"), (2, "x\u{2}abc")]).await;

        assert_eq!(like_ids(&sender, "%\u{2}ab%").await, [2]);
    }

    #[rstest]
    #[case::checked_by_regex("Ab-%")]
    #[case::proven_by_grams("%aB%")]
    #[case::any_char("A_-1024")]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn like_ignores_letter_case_when_not_case_sensitive(#[case] pattern: &str) {
        let sender = make_case_insensitive_sender();
        // Rows 1 and 2 differ only in letter case; row 3 matches no pattern.
        add_docs(&sender, &[(1, "AB-1024"), (2, "ab-1024"), (3, "CD-2048")]).await;

        assert_eq!(like_ids(&sender, pattern).await, [1, 2]);
    }

    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn like_ignores_letter_case_of_greek_sigma_when_not_case_sensitive() {
        let sender = make_case_insensitive_sender();
        // Lowercase sigma depends on where it stands: 'ς' at the end of a word, 'σ' elsewhere.
        add_docs(&sender, &[(1, "ασα")]).await;

        // 'Σ' ends a word in the pattern but not in the value.
        assert_eq!(like_ids(&sender, "ΑΣ%").await, [1]);
    }
}
