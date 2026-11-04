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
//! * **Pruning depends on segment layout.** Cost is flat only where segments are disjoint in sort
//!   order. Two options keep them so: `segment_rows` installs a merge policy that keeps the
//!   segments placed in sort order (the ordered tier) apart from the pieces commits leave (the
//!   fresh tier) and only joins ordered neighbours under a cap, and `rewrite` repairs the fresh
//!   rows into the ordered tier, worst stretch first, continuously when idle and holding back
//!   while the index ingests. Without them, after an unordered backfill every segment spans
//!   everything and the search degrades to visiting every match: correct, but not fast. See
//!   `docs/dev/pattern/stage-2-ordering.md`.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BTreeSet;
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
use tantivy::DocSet;
use tantivy::SegmentReader;
use tantivy::TERMINATED;
use tantivy::TantivyDocument;
use tantivy::Term;
use tantivy::columnar::Column;
use tantivy::index::SegmentId;
use tantivy::index::SegmentMeta;
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

use crate::AsyncInProgress;
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
use crate::worker;
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
/// The fast field saying whether a repair step placed the row (1) or a commit of ingested rows
/// did (0): a segment whose every row says 1 is in the ordered tier however it was merged since.
/// Ordered indexes only.
const REPAIRED_FIELD: &str = "repaired";
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
    /// The worker pool's indexing load at the last tick, to tell whether indexing waited since.
    indexing_before: Mutex<worker::IndexingLoad>,
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
/// While an index ingests, the fresh rows, in caps, it holds before a repair step competes with
/// the writes for the node. Under that, the fresh tier waits for an idle tick.
const LOAD_FRESH_CAPS: u64 = 2;
/// Ticks without a row of a full scan before the scan counts as ended.
const SCAN_QUIET_TICKS: u32 = 5;
/// The mean time indexing tasks may have waited in the worker's queue since the last tick for
/// a repair step to run while the index ingests.
const INDEXING_WAIT_LIMIT: Duration = Duration::from_millis(1);

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
///   so the merged segment overlaps none. A cluster past the cap is left to the repair.
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
    for cluster in clusters(pieces, &ordered) {
        if cluster.docs > cap {
            flush(&mut run, &mut run_docs, &mut run_deleted, true);
            continue;
        }
        if !run.is_empty() && run_docs + cluster.docs > fill {
            flush(&mut run, &mut run_docs, &mut run_deleted, true);
        }
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
/// rows too, as it overlaps itself until split. The worst stretch is repaired, at most a cap of
/// its fresh rows at a time, lowest keys first. Pure, so it can be tested without an index.
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
        let columns = SegmentColumns {
            slots,
            layout_min,
            layout_max,
            primary_id,
            repaired,
        };
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
/// count. Zero for an unordered index, which keeps no bounds, but for the deleted rows.
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
            schema_builder.add_u64_field(REPAIRED_FIELD, FAST);
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
            doc.add_u64(
                schema.get_field(REPAIRED_FIELD).unwrap(),
                u64::from(row.repaired),
            );
        }
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
        if !self.orders_results() {
            // Nothing to order the segments by; the cap alone is not worth a policy.
            return None;
        }
        Some(Box::new(tantivy::indexer::NoMergePolicy))
    }

    /// Learns every live segment's bounds (which also warms the column cache), drops the retired
    /// ones, adopts the segments found where they belong, and asks the writer to merge the runs
    /// `merge_runs` cuts (see `merge_policy`). Not left to Tantivy, which evaluates merges after a
    /// commit: at that moment the bounds of the segments the commit produced are not known yet,
    /// so the runs they belong to would wait for a commit that may never come once the writes
    /// stop. Run on every tick, since the reader reloads on its own.
    fn after_reload(&self, state: &PatternIndexState) {
        if !self.orders_results() {
            return;
        }
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
    /// - Writes after that: on an idle tick always, so that an idle index converges to an empty
    ///   fresh tier and a disjoint ordered one; while the index ingests, once the fresh tier
    ///   holds `LOAD_FRESH_CAPS` caps, and only while indexing does not wait in the worker's
    ///   queue. A step holds the writer lock, so writes waiting for it back up the queue, and the
    ///   next tick holds off: a burst of writes grows the fresh tier, which searches pay for with
    ///   an open per fresh segment, rather than the lag.
    fn maybe_repair(&self, state: &PatternIndexState, key: &IndexKey, idle: bool) {
        // An unordered index has no sort order to repair into, nor the columns to read one from,
        // though a segment cap turns the rewrite option on for it as for any other.
        if !self.orders_results() || !*self.options.rewrite_wide_segments.as_ref() {
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
        let load = worker::STATS.indexing();
        let before = std::mem::replace(&mut *self.indexing_before.lock().unwrap(), load);
        let contended = load.contended_since(&before, INDEXING_WAIT_LIMIT);
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
        if !repair_due(idle, scanning, contended, fresh_rows, overfull, cap) {
            return;
        }
        if let Err(err) = repair_step(state) {
            error!("pattern: repair of {key} failed: {err}");
        }
    }

    fn rewrite_totals(&self) -> RewriteTotals {
        RewriteTotals {
            steps: self.repair.steps.load(Relaxed),
            docs_rewritten: self.repair.docs_rewritten.load(Relaxed),
            pending: self.repair.pending.load(Relaxed),
        }
    }
}

/// Whether a repair step may run now, given what there is to repair: never while a full scan
/// goes on; on an idle tick always; while the index ingests, once the fresh tier holds
/// `LOAD_FRESH_CAPS` caps or an ordered stretch is past the cap, and only if indexing has not
/// been `contended`. Pure, so it can be tested without an index.
fn repair_due(
    idle: bool,
    scanning: bool,
    contended: bool,
    fresh_rows: u64,
    overfull: bool,
    cap: u64,
) -> bool {
    if (fresh_rows == 0 && !overfull) || scanning {
        return false;
    }
    idle || (!contended && (fresh_rows >= LOAD_FRESH_CAPS * cap || overfull))
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
            .map(unframed)
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
/// by primary id and added again as repaired, under the writer lock, committed before and after.
/// A commit's rows land in pieces spanning no more than the keys they hold, so rows repaired into
/// a cluster's stretch overlap the cluster and merge into it, and rows of a gap make a cluster
/// of their own. Rows that would not fit beside the cluster under the cap are moved with the
/// cluster's own, in `split_rows` parts of disjoint key ranges, one commit each.
///
/// The lock is what keeps a concurrent update or delete of one of those rows from being undone
/// by the re-add: it waits, and lands with a higher opstamp. Searches take the reader, not this
/// lock. Returns whether rows were moved.
fn repair_step(state: &PatternIndexState) -> anyhow::Result<bool> {
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
            return Ok(false);
        }
    }

    let mut writer = state.lock_writer();
    state
        .commit_and_reload(&mut writer)
        .map_err(|e| anyhow!("pattern: failed to commit before a repair: {e}"))?;
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
    let Some(range) = plan_repair(&pieces, &sample, stride, cap) else {
        return Ok(false);
    };

    let mut moved = Vec::new();
    for segment in searcher.segment_readers() {
        let Some(piece) = fresh.get(&segment.segment_id()) else {
            continue;
        };
        if piece.max < range.low || piece.min > range.high {
            continue;
        }
        read_rows(
            state,
            segment,
            |key| range.low <= key && key <= range.high,
            &mut moved,
        )?;
    }
    let split = match &range.cluster {
        Some(cluster) => cluster.docs + moved.len() as u64 > u64::from(cap),
        None => moved.len() > cap as usize,
    };
    if split && let Some(cluster) = &range.cluster {
        for segment in searcher.segment_readers() {
            if cluster.ids.contains(&segment.segment_id()) {
                read_rows(state, segment, |_| true, &mut moved)?;
            }
        }
    }
    if moved.is_empty() {
        return Ok(false);
    }
    let count = moved.len();
    let parts = if split {
        moved.sort_by_key(|row| row.0);
        split_parts(moved, split_rows(cap))
    } else {
        vec![moved]
    };
    let part_count = parts.len();
    for part in parts {
        writer
            .rewrite_documents(part.into_iter().map(|(_, row)| row))
            .map_err(|e| anyhow!("pattern: failed to repair a range: {e}"))?;
        state
            .commit_and_reload(&mut writer)
            .map_err(|e| anyhow!("pattern: failed to commit a repair: {e}"))?;
    }
    drop(writer);

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
        repair: RepairCounters::default(),
        adopted: Mutex::new(HashSet::new()),
        scan_rows: AtomicBool::new(false),
        ticks_since_scan: AtomicU32::new(SCAN_QUIET_TICKS),
        indexing_before: Mutex::new(worker::STATS.indexing()),
        tick_running: AtomicBool::new(false),
        merging: Mutex::new(MergesInFlight::default()),
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
    let mut overlap = if state.backend().orders_results() {
        layout_overlap(&state.backend().geometry(&searcher))
    } else {
        LayoutOverlap::default()
    };
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
                            let scanned = matches!(in_progress, AsyncInProgress::Fullscan(_));
                            indexes
                                .add_document_with(primary_id, in_progress, move |index| {
                                    if scanned {
                                        index.backend().scan_rows.store(true, Relaxed);
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
                        tokio::task::spawn_blocking(move || {
                            let state = &work.0;
                            let backend = state.backend();
                            backend.after_reload(state);
                            backend.maybe_repair(state, &key, idle);
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
    use crate::DEFAULT_SEGMENT_ROWS;
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
        let overlap = before.overlap;
        assert_eq!(overlap.rows, before.tantivy.num_docs);
        assert!(overlap.depth_max >= 1, "{overlap:?}");
        assert!(overlap.depth_rows >= overlap.rows, "{overlap:?}");

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
            ordered(40, 60, 69, 9),      // alone, and no neighbour fits
            piece(40, 0, 100, 10),       // fresh, overlapping all of the above
            piece(50, 5, 100, 11),       // + 10 = 90
            piece(30, 10, 100, 12),      // 90 + 30 > 100: a run of its own
            piece(cap + 5, 20, 100, 13), // full: left out
            piece(5, 30, 100, 14),       // with 12: 35
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
    /// (CHURN_ROWS sets the size, CHURN_SECS how long to watch before the drop).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore]
    async fn a_fast_loaded_table_order_index_settles_and_stops_when_dropped() {
        let rows: u64 = std::env::var("CHURN_ROWS").map_or(2_000_000, |rows| rows.parse().unwrap());
        let watch = std::env::var("CHURN_SECS").map_or(900, |secs| secs.parse().unwrap());
        // RUST_LOG=info shows the merges ('Starting merge', 'pattern: asking for ... merges').
        let _ = tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .try_init();
        let sender = new(
            PatternIndexConfiguration {
                key: make_index_key(),
                // What db.rs gives a table-order index created with no options.
                options: IndexOptionsPattern {
                    segment_max_docs: SegmentMaxDocs::from(Some(DEFAULT_SEGMENT_ROWS)),
                    rewrite_wide_segments: RewriteWideSegments::from(true),
                    ..IndexOptionsPattern::default()
                },
                key_layout: test_layout(),
            },
            make_table(bigint_key, |_, _| None),
            worker::new(),
            make_memory_actor(),
            COMMIT_INTERVAL,
            MAX_UNCOMMITTED_THRESHOLD,
            None,
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
                    AsyncInProgress::Fullscan(tx.clone()),
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

    /// An unordered index with a segment cap, which turns the rewrite option on, never plans a
    /// rewrite: it has no order to rewrite into, nor the sort columns a plan reads.
    #[rstest]
    #[timeout(Duration::from_secs(30))]
    #[tokio::test]
    async fn an_unordered_index_plans_no_rewrite() {
        let cap = 20u32;
        let sender = make_sender_for(
            IndexOptionsPattern {
                order_by: "unordered".parse().unwrap(),
                segment_max_docs: cap.to_string().parse().unwrap(),
                rewrite_wide_segments: "true".parse().unwrap(),
                ..IndexOptionsPattern::default()
            },
            make_table(bigint_key, |_, _| None),
            test_layout(),
        );
        let mut acks = Vec::new();
        for id in 1..=200u64 {
            let (tx, rx) = mpsc::channel(1);
            sender
                .add_document(
                    test_partition_id(),
                    id.into(),
                    format!("user{id}"),
                    AsyncInProgress::Fullscan(tx),
                )
                .await
                .unwrap();
            acks.push(rx);
        }
        for mut ack in acks {
            ack.recv().await;
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
        let stats = sender.stats(make_index_key()).await.unwrap();
        assert_eq!(stats.tantivy.num_docs, 200);
        assert_eq!(stats.rewrite.steps, 0, "{:?}", stats.rewrite);
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
                    AsyncInProgress::Fullscan(tx),
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

    /// After most rows are deleted, the segments full of holes are merged until no deleted row
    /// is left and the survivors sit in a few segments.
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
            stats.overlap.rows_deleted == 0 && stats.tantivy.num_docs == 500
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

    /// Nothing is repaired while a full scan goes on, idle tick or not. After it, an idle tick
    /// repairs whatever there is; while the index ingests, the repair waits for the fresh tier
    /// to grow past two caps, or a stretch past the cap, and for indexing not to wait.
    #[test]
    fn the_repair_waits_for_the_scan_and_gives_way_to_indexing() {
        let cap = 100;
        // (idle, scanning, contended, fresh rows, overfull)
        assert!(!repair_due(true, false, false, 0, false, cap));
        assert!(repair_due(true, false, true, 1, false, cap));
        assert!(!repair_due(true, true, false, 1_000, true, cap));
        assert!(!repair_due(false, true, false, 1_000, true, cap));
        assert!(!repair_due(false, false, false, 199, false, cap));
        assert!(repair_due(false, false, false, 200, false, cap));
        assert!(!repair_due(false, false, true, 200, false, cap));
        assert!(repair_due(false, false, false, 0, true, cap));
        assert!(!repair_due(false, false, true, 0, true, cap));
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
}
