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
//! * **Ties are skipped.** The cursor is a sort key and the next page takes rows strictly below
//!   it, so when several rows share a sort key and the page boundary falls among them, the
//!   remainder are never returned. The fix is a composite `(sort_key, primary_id)` cursor, which
//!   needs the tie-break to be part of the comparison rather than of the heap entry only.
//! * **Only descending.** `ORDER BY ... ASC` would need the segment walk and the heap to invert;
//!   nothing here is inherently descending, but nothing takes a direction either.
//! * **A short page is ambiguous.** Rows dropped because the table no longer knows them shorten a
//!   page after the walk has filled it, so a caller cannot infer "no more results" from a page
//!   shorter than the limit, and must follow the cursor instead. The converse is exact: the walk
//!   reports a cursor only when it filled the page, so no cursor does mean no more results.
//! * **Pruning depends on segment layout.** Cost is flat only where a segment's span of the sort
//!   column is narrow. After an unordered backfill every segment spans everything and the search
//!   degrades to visiting every match -- correct, but not fast. Keeping segments narrow is a
//!   separate piece of work (see `docs/dev/substring/stage-2-ordering.md`).

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::BinaryHeap;
use std::ops::Bound;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use anyhow::anyhow;
use tantivy::DocAddress;
use tantivy::DocSet;
use tantivy::TERMINATED;
use tantivy::TantivyDocument;
use tantivy::Term;
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

use crate::CaseSensitive;
use crate::IndexKey;
use crate::IndexOptionsSubstring;
use crate::Limit;
use crate::memory::Allocate;
use crate::memory::Memory;
use crate::memory::MemoryExt;
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
use crate::tantivy_common::can_allocate_memory;
use crate::tantivy_common::commit;
use crate::tantivy_common::find_partition_id;
use crate::tantivy_common::get_or_create_state;
use crate::tantivy_common::get_state;
use crate::tantivy_common::handle_add_document;
use crate::tantivy_common::handle_remove_document;
use crate::tantivy_common::handle_stats;
use crate::worker::Worker;
use crate::worker::WorkerExt;

use super::actor::SubstringIndex;
use super::actor::SubstringPage;
use super::actor::SubstringSearchR;
use super::factory::SubstringIndexConfiguration;
use super::factory::SubstringIndexFactory;

pub(crate) struct TantivySubstringIndexFactory {
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
}

impl TantivySubstringIndexFactory {
    pub(crate) fn new(worker: async_channel::Sender<Worker>, memory: mpsc::Sender<Memory>) -> Self {
        Self { worker, memory }
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
        )
    }
}

const TEXT_FIELD: &str = "text";
const SORT_FIELD: &str = "sort_key";
const TOKENIZER_NAME: &str = "substring_ngram";
/// Values are short, so a single cached store block per segment covers consecutive lookups.
const STORE_CACHE_BLOCKS: usize = 1;

/// The substring flavour of a Tantivy index: the (normalized) value is stored verbatim for
/// verification and indexed as all of its `min_gram..=max_gram`-long substrings.
struct SubstringBackend {
    options: IndexOptionsSubstring,
}

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
        schema_builder.add_u64_field(PRIMARY_ID_FIELD, INDEXED | STORED);
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
        doc.add_text(text_field, normalized.as_ref());
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
    Candidates(Box<dyn Query>),
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
    let clauses = grams_of_length(normalized, max_gram)
        .iter()
        .map(|gram| term_query(gram))
        .collect();
    Ok(SubstringQuery::Candidates(Box::new(
        BooleanQuery::intersection(clauses),
    )))
}

/// Walks the matching documents in index order and returns the primary ids of the first
/// `limit` verified matches after skipping `offset` of them.
///
/// The query is executed once, unscored, and the walk stops as soon as enough matches are
/// found, so a hot single-character query does not pay for its whole posting list. The stored
/// document has to be read anyway for the primary id, so verifying the containment on the way
/// costs no extra I/O.
/// One page of an ordered search: the `limit` highest sort keys strictly below `cursor`, newest
/// first, with the cursor to resume from.
///
/// Two things keep this off the O(matches) path the obvious implementation lands on. Segments are
/// visited by descending upper bound and the walk stops once the next segment's bound cannot beat
/// the page's weakest entry, which skips whole segments unopened. And within a segment the sort key
/// -- a columnar read -- is checked before the document store is touched, so a candidate that
/// cannot make the page costs almost nothing. Measured, the second is worth 4-13x on its own and
/// does not depend on how the segments are laid out.
///
/// The cursor is a sort key rather than an offset, so a later page does not re-walk the earlier
/// ones. Rows sharing a sort key are a known gap; see the note in the module docs.
/// The window of sort keys a search may return: a range restriction, a paging cursor, or both.
///
/// The two arrive separately -- the range from the query's `WHERE`, the cursor from the previous
/// page -- but they constrain the same value, so they are resolved into one pair of bounds once
/// rather than checked separately on every candidate.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SortWindow {
    lower: Option<u64>,
    upper: Option<Bound<u64>>,
}

impl SortWindow {
    pub(crate) fn new(
        cursor: Option<u64>,
        min_sort_key: Option<u64>,
        max_sort_key: Option<u64>,
    ) -> Self {
        // The cursor excludes the key it names (the previous page ended there); a range bound
        // includes it. Where both apply, the tighter one wins.
        let upper = match (cursor, max_sort_key) {
            (Some(cursor), Some(max)) if max < cursor => Some(Bound::Included(max)),
            (Some(cursor), _) => Some(Bound::Excluded(cursor)),
            (None, Some(max)) => Some(Bound::Included(max)),
            (None, None) => None,
        };
        Self {
            lower: min_sort_key,
            upper,
        }
    }

    fn contains(&self, sort_key: u64) -> bool {
        let above_lower = self.lower.is_none_or(|lower| sort_key >= lower);
        let below_upper = match self.upper {
            None => true,
            Some(Bound::Included(upper)) => sort_key <= upper,
            Some(Bound::Excluded(upper)) => sort_key < upper,
            Some(Bound::Unbounded) => true,
        };
        above_lower && below_upper
    }

    /// Whether a segment spanning `[min, max]` can hold anything in the window. Answered from the
    /// segment's bounds alone, so a segment ruled out here is never opened.
    fn overlaps(&self, min: u64, max: u64) -> bool {
        let above = self.lower.is_none_or(|lower| max >= lower);
        let below = match self.upper {
            None => true,
            Some(Bound::Included(upper)) => min <= upper,
            Some(Bound::Excluded(upper)) => min < upper,
            Some(Bound::Unbounded) => true,
        };
        above && below
    }
}

fn collect_matches_ordered(
    state: &SubstringIndexState,
    normalized: &str,
    limit: usize,
    window: SortWindow,
) -> anyhow::Result<(Vec<PrimaryId>, Option<u64>)> {
    let text_field = state.schema.get_field(TEXT_FIELD).unwrap();
    let primary_id_field = state.schema.get_field(PRIMARY_ID_FIELD).unwrap();
    let sort_field_name = SORT_FIELD;

    let (query, needs_verification) = match build_query(state, normalized)? {
        SubstringQuery::Exact(query) => (query, false),
        SubstringQuery::Candidates(query) => (query, true),
    };

    let searcher = state.reader.searcher();
    let weight = query
        .weight(EnableScoring::disabled_from_searcher(&searcher))
        .map_err(|e| anyhow!("substring: failed to build the query: {e}"))?;

    let mut segments = Vec::with_capacity(searcher.segment_readers().len());
    for (segment_ord, segment) in searcher.segment_readers().iter().enumerate() {
        let column = segment
            .fast_fields()
            .u64(sort_field_name)
            .map_err(|e| anyhow!("substring: failed to open the sort column: {e}"))?;
        let (min, max) = (column.min_value(), column.max_value());
        // Ruled out from the bounds alone, so the segment is never opened.
        if window.overlaps(min, max) {
            segments.push((segment_ord as u32, segment, column, max));
        }
    }
    segments.sort_by_key(|(_, _, _, upper_bound)| Reverse(*upper_bound));

    // Min-heap of the best `limit` so far, so the root is the entry to beat. It holds document
    // addresses, not primary ids: an entrant is often pushed out again by a later, higher one, and
    // reading the document store to learn the id of every entrant is what dominated the walk --
    // 9-27x the cost of the walk itself (benches/substring_order.rs, `as_shipped`). The ids are
    // read once, for the page that survives.
    let mut best: BinaryHeap<Reverse<(u64, DocAddress)>> = BinaryHeap::with_capacity(limit + 1);
    for (segment_ord, segment, sort_column, upper_bound) in segments {
        if best.len() == limit
            && let Some(Reverse((weakest, _))) = best.peek()
            && upper_bound <= *weakest
        {
            // Nothing in this segment, nor in any later one, can enter the page.
            break;
        }

        let mut scorer = weight
            .scorer(segment, 1.0)
            .map_err(|e| anyhow!("substring: failed to run the query: {e}"))?;
        let alive = segment.alive_bitset();
        let store = needs_verification
            .then(|| segment.get_store_reader(STORE_CACHE_BLOCKS))
            .transpose()
            .map_err(|e| anyhow!("substring: failed to open the document store: {e}"))?;

        let mut doc_id = scorer.doc();
        while doc_id != TERMINATED {
            if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                let sort_key = sort_column.first(doc_id).unwrap_or(0);
                let beats_page = best.len() < limit
                    || best
                        .peek()
                        .is_none_or(|Reverse((weakest, _))| sort_key > *weakest);
                if window.contains(sort_key) && beats_page {
                    // Past max_gram the grams only nominate candidates, and the text is in the
                    // store; below it every match is exact and the store is not touched here.
                    let verified = match &store {
                        None => true,
                        Some(store) => store
                            .get::<TantivyDocument>(doc_id)
                            .map_err(|e| anyhow!("substring: failed to retrieve doc: {e}"))?
                            .get_first(text_field)
                            .and_then(|value| value.as_str())
                            .is_some_and(|text| text.contains(normalized)),
                    };
                    if verified {
                        best.push(Reverse((sort_key, DocAddress::new(segment_ord, doc_id))));
                        if best.len() > limit {
                            best.pop();
                        }
                    }
                }
            }
            doc_id = scorer.advance();
        }
    }

    // A cursor says there may be more below this point. That is only true if the page filled: a
    // walk that ended with fewer than `limit` matches visited every segment overlapping the window,
    // so there is nothing left to resume from and a cursor would only cost the caller an empty
    // round trip. (Rows dropped later, in `handle_search`, shorten the page after this point and
    // must not suppress the cursor -- which is why this asks the heap, not the returned page.)
    let next_cursor = (best.len() == limit)
        .then(|| best.peek().map(|Reverse((sort_key, _))| *sort_key))
        .flatten();
    let mut page: Vec<(u64, DocAddress)> = best.into_iter().map(|Reverse(entry)| entry).collect();
    page.sort_unstable_by_key(|(sort_key, _)| Reverse(*sort_key));
    let ids = page
        .into_iter()
        .map(|(_, address)| {
            let doc: TantivyDocument = searcher
                .doc(address)
                .map_err(|e| anyhow!("substring: failed to retrieve doc: {e}"))?;
            doc.get_first(primary_id_field)
                .and_then(|value| value.as_u64())
                .map(PrimaryId::from)
                .ok_or_else(|| anyhow!("substring: missing primary_id in doc"))
        })
        .collect::<anyhow::Result<_>>()?;
    Ok((ids, next_cursor))
}

fn collect_matches(
    state: &SubstringIndexState,
    normalized: &str,
    limit: usize,
    offset: usize,
) -> anyhow::Result<Vec<PrimaryId>> {
    let text_field = state.schema.get_field(TEXT_FIELD).unwrap();
    let primary_id_field = state.schema.get_field(PRIMARY_ID_FIELD).unwrap();

    let (query, needs_verification) = match build_query(state, normalized)? {
        SubstringQuery::Exact(query) => (query, false),
        SubstringQuery::Candidates(query) => (query, true),
    };

    let searcher = state.reader.searcher();
    let weight = query
        .weight(EnableScoring::disabled_from_searcher(&searcher))
        .map_err(|e| anyhow!("substring: failed to build the query: {e}"))?;

    let mut to_skip = offset;
    let mut matches = Vec::with_capacity(limit);
    'segments: for segment in searcher.segment_readers() {
        let mut scorer = weight
            .scorer(segment, 1.0)
            .map_err(|e| anyhow!("substring: failed to run the query: {e}"))?;
        let alive = segment.alive_bitset();
        let store = segment
            .get_store_reader(STORE_CACHE_BLOCKS)
            .map_err(|e| anyhow!("substring: failed to open the document store: {e}"))?;

        let mut doc_id = scorer.doc();
        while doc_id != TERMINATED {
            if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                let doc: TantivyDocument = store
                    .get(doc_id)
                    .map_err(|e| anyhow!("substring: failed to retrieve doc: {e}"))?;
                let verified = !needs_verification
                    || doc
                        .get_first(text_field)
                        .and_then(|value| value.as_str())
                        .is_some_and(|text| text.contains(normalized));
                if verified {
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
    Ok(matches)
}

fn handle_search(
    state: &SubstringIndexState,
    table: &RwLock<impl TableSearch>,
    index_key: &IndexKey,
    query: &str,
    limit: Limit,
    offset: usize,
    window: SortWindow,
) -> SubstringSearchR {
    let normalized = normalize(query, state.backend.options.case_sensitive);
    let limit: usize = (*limit.as_ref()).into();
    let (primary_ids, next_cursor) = if state.backend.orders_results() {
        collect_matches_ordered(state, &normalized, limit, window)?
    } else {
        (collect_matches(state, &normalized, limit, offset)?, None)
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
        next_cursor,
    })
}

pub(crate) fn new(
    index: SubstringIndexConfiguration,
    table: Arc<RwLock<impl TableSearch + Send + Sync + 'static>>,
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
    commit_interval: Duration,
    commit_threshold: usize,
) -> mpsc::Sender<SubstringIndex> {
    let (tx, mut rx) = mpsc::channel::<SubstringIndex>(perf::channel_size().into());
    tokio::spawn(async move {
        let key = index.key.clone();
        debug!("substring index actor starting for {key}");
        let mut states: BTreeMap<IndexId, Arc<SubstringIndexState>> = BTreeMap::new();
        let make_backend = || SubstringBackend {
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
                            limit,
                            offset,
                            window,
                            tx,
                        } => {
                            let Some(state) = get_state(&states, table.as_ref(), &index_key) else {
                                _ = tx.send(Ok(SubstringPage {
                                    primary_keys: vec![],
                                    next_cursor: None,
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
                                        limit,
                                        offset,
                                        window,
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
                                    let result = handle_stats(&state);
                                    _ = tx.send(result);
                                })
                                .await;
                        }
                    }
                }
                _ = interval.tick() => {
                    for state in states.values() {
                        if !state.writer.read().unwrap().has_uncommitted_docs() {
                            continue;
                        }
                        let state = Arc::clone(state);
                        let key = key.clone();
                        worker.spawn_blocking(move || commit(&state, &key)).await;
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
    use crate::PrimaryKey;
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
        new(
            SubstringIndexConfiguration {
                key: make_index_key(),
                options,
            },
            make_table_with_keys(),
            worker::new(),
            make_memory_actor(),
            TEST_COMMIT_INTERVAL,
            TEST_COMMIT_THRESHOLD,
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
        window: SortWindow,
    ) -> (Vec<i64>, Option<u64>) {
        let page = sender
            .search(make_index_key(), query.into(), limit(limit_n), 0, window)
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
                limit(limit_n),
                offset,
                SortWindow::default(),
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
                limit(10),
                0,
                SortWindow::default(),
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
                limit(10),
                0,
                SortWindow::default(),
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

        assert_eq!(stats.num_docs, 6);
        assert!(stats.segment_count > 0);
        assert!(stats.size_bytes > 0);
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

        let (ids, _) = search_ordered(&sender, "将军", 10, SortWindow::default()).await;
        let mut descending = ids.clone();
        descending.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(ids, descending, "not ordered by the sort column");
        assert_eq!(ids, vec![3, 2, 1]);
    }

    /// Without a sort column nothing changes: the index answers as it did before, and reports no
    /// cursor for the caller to page with.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn an_unordered_index_reports_no_cursor() {
        let sender = make_sender();
        add_docs(&sender, NICKNAMES).await;

        let (ids, cursor) = search_ordered(&sender, "将军", 10, SortWindow::default()).await;
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

        let (first, cursor) = search_ordered(&sender, "将军", 2, SortWindow::default()).await;
        assert_eq!(first, vec![3, 2]);
        let cursor = cursor.expect("a full page leaves a cursor");

        let (second, cursor) = search_ordered(
            &sender,
            "将军",
            2,
            SortWindow::new(Some(cursor), None, None),
        )
        .await;
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
        let (ids, cursor) = search_ordered(&sender, "将军", 10, SortWindow::default()).await;
        assert_eq!(ids, vec![3, 2, 1]);
        assert_eq!(
            cursor, None,
            "the walk ran out, so there is nothing to resume from"
        );

        // Exactly as many as there are: the walk stopped because the page was full, not because it
        // ran out, so it cannot tell that the next page would be empty and says so with a cursor.
        let (ids, cursor) = search_ordered(&sender, "将军", 3, SortWindow::default()).await;
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

        let (ids, cursor) = search_ordered(&sender, "没有人", 10, SortWindow::default()).await;
        assert!(ids.is_empty(), "got {ids:?}");
        assert_eq!(cursor, None);
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

        let (ids, _) = search_ordered(&sender, "将军来了", 10, SortWindow::default()).await;
        assert_eq!(ids, vec![3]);
    }

    /// Reading past the end yields an empty page rather than an error.
    #[rstest]
    #[timeout(Duration::from_secs(10))]
    #[tokio::test]
    async fn paging_past_the_last_match_is_empty() {
        let sender = make_sender_with_options(ordered_options());
        add_docs(&sender, NICKNAMES).await;

        let (ids, _) = search_ordered(
            &sender,
            "将军",
            10,
            SortWindow::new(Some(sort_key(1)), None, None),
        )
        .await;
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
        let window = SortWindow::new(None, Some(sort_key(2)), Some(sort_key(3)));
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
            SortWindow::new(None, Some(sort_key(3)), Some(sort_key(3))),
        )
        .await;
        assert_eq!(inclusive, vec![3]);

        let (exclusive, _) = search_ordered(
            &sender,
            "将军",
            10,
            SortWindow::new(Some(sort_key(3)), Some(sort_key(3)), None),
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
        let window = SortWindow::new(Some(sort_key(3)), Some(sort_key(1)), Some(sort_key(3)));
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
            SortWindow::new(None, Some(sort_key(900)), Some(sort_key(999))),
        )
        .await;
        assert!(ids.is_empty(), "got {ids:?}");
        assert_eq!(cursor, None);
    }
}
