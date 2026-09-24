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

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use anyhow::anyhow;
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
) -> SubstringSearchR {
    let normalized = normalize(query, state.backend.options.case_sensitive);
    let limit: usize = (*limit.as_ref()).into();
    let primary_ids = collect_matches(state, &normalized, limit, offset)?;

    let table = table.read().unwrap();
    let partition_id = find_partition_id::<SubstringBackend>(table.deref(), index_key)?;
    // A row the table cache no longer knows was deleted after the index snapshot was taken;
    // it is simply left out, as the full-text search does.
    Ok(primary_ids
        .into_iter()
        .filter_map(|primary_id| table.primary_key(partition_id, primary_id))
        .collect())
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
                            tx,
                        } => {
                            let Some(state) = get_state(&states, table.as_ref(), &index_key) else {
                                _ = tx.send(Ok(vec![]));
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

    async fn search_page(
        sender: &mpsc::Sender<SubstringIndex>,
        query: &str,
        limit_n: usize,
        offset: usize,
    ) -> Vec<i64> {
        let mut ids: Vec<i64> = sender
            .search(make_index_key(), query.into(), limit(limit_n), offset)
            .await
            .unwrap()
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
            .search(make_index_key(), "".into(), limit(10), 0)
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
            .search(make_index_key(), "宫".into(), limit(10), 0)
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
}
