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
use std::sync::RwLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Duration;
use std::time::Instant;

use anyhow::anyhow;
use tantivy::DocSet;
use tantivy::SegmentReader;
use tantivy::TERMINATED;
use tantivy::TantivyDocument;
use tantivy::columnar::Column;
use tantivy::index::SegmentId;
use tantivy::query::Scorer;
use tantivy::schema::FAST;
use tantivy::schema::INDEXED;
use tantivy::schema::IndexRecordOption;
use tantivy::schema::STORED;
use tantivy::schema::Schema;
use tantivy::schema::TextFieldIndexing;
use tantivy::schema::TextOptions;
use tokio::sync::mpsc;
use tracing::debug;

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
use crate::worker::Worker;
use crate::worker::WorkerExt;

use super::actor::Cursor;
use super::actor::PatternIndex;
use super::actor::PatternLikeR;
use super::actor::PatternPage;
use super::actor::SearchWindow;
use super::factory::PatternIndexConfiguration;
use super::factory::PatternIndexFactory;
use super::gram;
use super::like::LikeQuery;
use super::order::KeyLayout;
use super::order::SortPrefix;

pub(crate) struct Factory {
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
    metrics: Arc<Metrics>,
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
        )
    }
}

pub(super) fn new_factory(
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
    metrics: Arc<Metrics>,
) -> Factory {
    Factory {
        worker,
        memory,
        metrics,
    }
}

/// The field the shared tantivy code deletes documents by; the pattern index also reads it back.
const PRIMARY_ID_FIELD: &str = "primary_id";
const TEXT_FIELD: &str = "text";
/// The fast field holding a row's token, the last slot of its sort prefix. The ordering columns'
/// slots come before it, as `sort_<i>`.
const TOKEN_FIELD: &str = "token";
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
    /// The fast fields of the sort prefix: the token.
    fn slot_fields(&self) -> Vec<String> {
        vec![TOKEN_FIELD.to_string()]
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

    /// Drops cache entries for segments a merge has retired, once they outnumber the live ones.
    fn prune_columns(&self, live: &[SegmentReader]) {
        let mut cache = self.columns.write().unwrap();
        if cache.len() > 2 * live.len().max(1) {
            let keep: HashSet<SegmentId> =
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
    segments_wasted: AtomicU64,
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
    pub(crate) segments: Vec<SegmentLayout>,
}

pub(crate) type PatternStatsR = anyhow::Result<PatternStats>;

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
        doc
    }
}

type PatternIndexState = Index<PatternBackend>;

impl PatternIndexState {
    fn backend(&self) -> &PatternBackend {
        &self.state
    }
}

/// A new tantivy index for `options`.
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
        options: options.clone(),
    };
    let schema = backend.build_schema();
    let analyzer = gram::build_ngram_analyzer()?;
    Index::with_state(schema, TOKENIZER_NAME, analyzer, key, worker, backend)
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

/// The row's place in table order: its token. None when the row's key is not known.
fn read_prefix(
    table: &RwLock<impl TableSearch>,
    key_layout: &KeyLayout,
    partition_id: PartitionId,
    primary_id: PrimaryId,
) -> Option<SortPrefix> {
    let primary_key = table
        .read()
        .unwrap()
        .primary_key(partition_id, primary_id)?;
    let mut prefix = SortPrefix::default();
    prefix.0[0] = key_layout.token_key(&primary_key)?;
    Some(prefix)
}

/// The part of the index's order a search may return, in *walk orientation*: the walk always
/// takes the highest row first, and a search in the declared order is the same walk over the
/// bitwise complement of every sort prefix, with ties between whole prefixes reversed too.
#[derive(Clone, Debug, Default)]
pub(crate) struct SortWindow {
    complement: bool,
    /// The previous page's last row, oriented: the page resumes strictly below it. Without a key
    /// the rows sharing its prefix are taken again rather than any skipped.
    cursor: Option<(SortPrefix, Option<PrimaryKey>)>,
}

impl SortWindow {
    /// `forward` walks the index's declared order from its start. `cursor` is in the index's own
    /// space.
    pub(crate) fn new(forward: bool, cursor: Option<Cursor>) -> Self {
        let orient = |prefix: SortPrefix| {
            if forward { prefix.complement() } else { prefix }
        };
        Self {
            complement: forward,
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

    /// Whether an oriented prefix is not above the cursor's. A prefix equal to the cursor's is
    /// decided by the keys, in [`SortWindow::below_cursor`].
    fn admits(&self, prefix: SortPrefix) -> bool {
        self.cursor
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

    /// Whether a segment whose lowest oriented layout key is `min` can hold anything in the
    /// window. Answered from the bound alone, so a segment ruled out here is never opened. The
    /// layout key is only the first two slots, so a bound equal to it may still hold rows on
    /// either side of it and stays in.
    fn overlaps(&self, min: u128) -> bool {
        self.cursor
            .as_ref()
            .is_none_or(|(cursor, _)| min <= cursor.layout_key())
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

/// One segment as an ordered walk reads it.
struct RegionWalk<'a> {
    segment: &'a SegmentReader,
    segment_ord: u32,
    columns: &'a SegmentColumns,
    window: &'a SortWindow,
    ties: &'a Ties<'a>,
    key_of: &'a dyn Fn(PrimaryId) -> Option<PrimaryKey>,
    like: &'a LikeQuery,
    /// Opened for the first candidate to verify.
    store: Option<tantivy::store::StoreReader>,
    /// The segment's postings for the query.
    scorer: Option<Box<dyn Scorer>>,
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

    /// Offers the page every matching row of the segment, in one pass of the postings.
    fn scan(
        &mut self,
        page: &mut Page,
        tally: &mut WalkTally,
        candidates: &mut Vec<(SortPrefix, u32)>,
    ) {
        let Some(mut scorer) = self.scorer.take() else {
            return;
        };
        let mut doc_id = scorer.doc();
        while doc_id != TERMINATED {
            tally.postings_scanned += 1;
            self.offer(page, tally, candidates, doc_id);
            doc_id = scorer.advance();
        }
    }

    /// Verifies `candidates` against the stored text, from the highest prefix down, and offers
    /// the page the ones that match, stopping at the first that can no longer enter it. The grams
    /// only nominate candidates, and the text that decides is in the document store. Postings arrive in doc order, so verifying as they come reads nearly every
    /// candidate when doc order is roughly ascending sort order: each later one beats the page
    /// the earlier ones built. From the highest down, the store is read for the page's rows plus
    /// the false positives above them.
    fn verify(
        &mut self,
        page: &mut Page,
        tally: &mut WalkTally,
        candidates: &mut Vec<(SortPrefix, u32)>,
    ) -> anyhow::Result<()> {
        // A heap, not a sort: of the many a wide pass gathers, few are looked at.
        let mut highest = BinaryHeap::from(std::mem::take(candidates));
        while let Some(&(prefix, doc_id)) = highest.peek() {
            if !page.may_take(prefix) {
                // Nor can any after it, now or once the page holds more.
                highest.clear();
                break;
            }
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
        Ok(())
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
/// Returns the page's primary ids, and the cursor of its last row when the page filled.
type OrderedPage = (Vec<PrimaryId>, Option<Cursor>);

fn collect_matches_ordered(
    state: &PatternIndexState,
    like: &LikeQuery,
    limit: usize,
    window: SortWindow,
    key_of: &dyn Fn(PrimaryId) -> Option<PrimaryKey>,
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
        if window.overlaps(lower_bound) {
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
    for (segment_ord, segment, columns, upper_bound) in segments {
        if let Some(weakest) = page.weakest()
            && upper_bound < weakest.prefix.layout_key()
        {
            // Nothing in this segment, nor in any later one, can enter the page. (A segment
            // ending exactly at the weakest layout key may still hold a row above it.)
            break;
        }
        let scorer = like.candidates(segment)?;
        tally.segments_opened += 1;
        candidates.clear();
        let mut walk = RegionWalk {
            segment,
            segment_ord,
            columns: &columns,
            window: &window,
            ties: &ties,
            key_of,
            like,
            store: None,
            scorer: Some(scorer),
        };
        walk.scan(&mut page, &mut tally, &mut candidates);
        walk.verify(&mut page, &mut tally, &mut candidates)?;
    }

    // A cursor says there may be more below this point. That is only true if the page filled: a
    // walk that ended with fewer than `limit` matches visited every segment overlapping the window,
    // so there is nothing left to resume from and a cursor would only cost the caller an empty
    // round trip. (Rows dropped later, in `handle_search`, shorten the page after this point and
    // must not suppress the cursor -- which is why this asks the page, not the returned rows.)
    let needed: HashSet<u32> = page.entries.iter().map(|entry| entry.segment).collect();
    tally.segments_wasted = tally.segments_opened - needed.len() as u64;
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
    Ok(PatternStats {
        tantivy,
        walk: state.backend().walk.snapshot(),
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
) -> PatternLikeR {
    let text_field = state.schema.get_field(TEXT_FIELD).unwrap();
    let pattern = state.backend().options.case_sensitive.fold(pattern);
    let like = LikeQuery::new(&state.schema, text_field, &pattern)?;
    let limit: usize = (*limit.as_ref()).into();
    let partition_id = find_partition_id(table.read().unwrap().deref(), index_key)?;
    let (primary_ids, next_cursor) = {
        let key_of =
            |primary_id: PrimaryId| table.read().unwrap().primary_key(partition_id, primary_id);
        // The index's declared order, from its start.
        let sort_window = SortWindow::new(true, window.cursor);
        collect_matches_ordered(state, &like, limit, sort_window, &key_of)?
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
                            let prefix = read_prefix(
                                table.as_ref(),
                                &index.key_layout,
                                partition_id,
                                primary_id,
                            );
                            if prefix.is_none() {
                                // A row the table does not know has no place in the order.
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
                            worker
                                .spawn_blocking(move || {
                                    let result = handle_like(
                                        &state,
                                        table.as_ref(),
                                        &index_key,
                                        pattern,
                                        limit,
                                        window,
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
                    indexes.commit_or_reload().await;
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
    use crate::PrimaryKey;
    use crate::memory::Allocate;
    use crate::table::IndexIdGenerator;
    use crate::table::MockTableSearch;
    use crate::table::PartitionId;
    use crate::tantivy::test_support::make_index_key;
    use crate::tantivy::test_support::make_memory_actor;
    use crate::tantivy::test_support::make_table_with_keys;
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

    const TEST_COMMIT_INTERVAL: Duration = Duration::from_millis(50);
    const TEST_COMMIT_THRESHOLD: usize = 100;

    fn options(case_sensitive: bool) -> IndexOptionsPattern {
        IndexOptionsPattern {
            case_sensitive: CaseSensitive::from(case_sensitive),
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
            make_memory_actor(Allocate::Can),
            TEST_COMMIT_INTERVAL,
            commit_threshold,
            None,
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
        )
    }

    /// The defaults: case-sensitive.
    fn make_sender() -> mpsc::Sender<PatternIndex> {
        make_sender_with_options(IndexOptionsPattern::default())
    }

    fn make_case_insensitive_sender() -> mpsc::Sender<PatternIndex> {
        make_sender_with_options(options(false))
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

    /// The window that resumes after a page.
    fn resume(cursor: Cursor) -> SearchWindow {
        SearchWindow {
            cursor: Some(cursor),
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

    /// `Stats` prices the searches served so far and describes the segments the pruning sees: an
    /// ordered index reports each segment's sort bounds, and the walk's tally grows per search.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn stats_count_the_walk_and_describe_the_segments() {
        let sender = make_sender_with_options(IndexOptionsPattern::default());
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

    /// The columns are opened once per segment: by the commit tick after the reload that made the
    /// segment searchable, or by a search that comes first, and never again -- and the time
    /// before the first posting is part of the walk time.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn columns_are_opened_once_per_segment() {
        let sender = make_sender_with_options(IndexOptionsPattern::default());
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

    /// No match at all is not a page to resume either.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn an_empty_page_offers_no_cursor() {
        let sender = make_sender_with_options(IndexOptionsPattern::default());
        add_docs(&sender, NICKNAMES).await;

        let (ids, cursor) = search_ordered(&sender, "没有人", 10, SearchWindow::default()).await;
        assert!(ids.is_empty(), "got {ids:?}");
        assert_eq!(cursor, None);
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
        let page = like(&sender, "%将军", 1, resume(cursor)).await;
        assert_eq!(ids_of(&page.primary_keys), expected[1..]);
        // A filled page always leaves a cursor; the page after it is empty and leaves none.
        let cursor = page.next_cursor.expect("a full page leaves a cursor");
        let page = like(&sender, "%将军", 1, resume(cursor)).await;
        assert!(page.primary_keys.is_empty());
        assert_eq!(page.next_cursor, None);
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
        )
    }

    fn bigint_key(id: u64) -> PrimaryKey {
        PrimaryKey::from(vec![CqlValue::BigInt(id as i64)])
    }

    /// Every page of a search, followed from its cursor; the pages concatenated.
    async fn every_page(sender: &mpsc::Sender<PatternIndex>, query: &str, page: usize) -> Vec<i64> {
        let mut seen = Vec::new();
        let mut window = SearchWindow::default();
        loop {
            let (ids, cursor) = search_ordered(sender, query, page, window).await;
            seen.extend(ids);
            match cursor {
                Some(cursor) => window = resume(cursor),
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
        assert_eq!(every_page(&sender, "将军", 5).await, expected);
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
                let found = like(&sender, "%将军%", page, window).await;
                seen.extend(found.primary_keys);
                match found.next_cursor {
                    Some(cursor) => {
                        window = SearchWindow {
                            cursor: Some(cursor),
                        }
                    }
                    None => break,
                }
            }
            assert_eq!(seen, expected, "page {page}");
        }
    }

    /// A segment holding the last tokens of one value and the first of the next spans what its
    /// rows do, not both values whole.
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
        assert_eq!(layout(4, high) - layout(3, low), 131);
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
