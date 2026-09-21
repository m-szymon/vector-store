/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! Plumbing shared by every Tantivy-backed index kind.
//!
//! The full-text and the substring indexes differ in how they build a schema, tokenize text and
//! answer queries, but they manage the Tantivy writer, reader, commits, deletes, statistics and
//! memory gating in exactly the same way. That common part lives here, parametrised by a
//! [`TantivyBackend`] that supplies the kind-specific bits.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use anyhow::anyhow;
use tantivy::IndexWriter;
use tantivy::ReloadPolicy;
use tantivy::TantivyDocument;
use tantivy::indexer::IndexWriterOptions;
use tantivy::schema::Schema;
use tokio::sync::watch;
use tracing::error;

use crate::AsyncInProgress;
use crate::IndexKey;
use crate::memory::Allocate;
use crate::perf;
use crate::table::IndexId;
use crate::table::PartitionId;
use crate::table::PrimaryId;
use crate::table::TableSearch;

/// Name of the field storing the row identifier the index maps documents back to.
pub(crate) const PRIMARY_ID_FIELD: &str = "primary_id";

/// How often uncommitted documents are made searchable.
pub(crate) const COMMIT_INTERVAL: Duration = Duration::from_secs(3);
/// How many uncommitted documents force a commit before the interval elapses.
pub(crate) const MAX_UNCOMMITTED_THRESHOLD: usize = 10_000;

/// Kind-specific behaviour of a Tantivy-backed index.
pub(crate) trait TantivyBackend: Send + Sync + 'static {
    /// Prefix of log and error messages, e.g. `fts` or `substring`.
    const NAME: &'static str;

    /// Builds the schema of the index. It must contain the [`PRIMARY_ID_FIELD`] u64 field
    /// (indexed, so documents can be deleted by it, and stored, so it can be read back).
    fn build_schema(&self) -> Schema;

    /// Registers the tokenizers the schema refers to.
    fn register_tokenizers(&self, index: &tantivy::Index) -> anyhow::Result<()>;

    /// Turns a row's text into a document for the index.
    fn create_doc(&self, schema: &Schema, primary_id: PrimaryId, text: &str) -> TantivyDocument;
}

pub(crate) struct Writer {
    writer: IndexWriter,
    // In-progress guards for documents written to the writer but not yet committed. They are held
    // here so the index is not reported as caught up (SERVING) until the commit that makes those
    // documents searchable has succeeded.
    uncommitted_docs_in_progress_guards: Vec<AsyncInProgress>,
}

impl Writer {
    fn add_document(
        &mut self,
        doc: TantivyDocument,
        in_progress: AsyncInProgress,
    ) -> tantivy::Result<usize> {
        self.writer.add_document(doc)?;
        self.uncommitted_docs_in_progress_guards.push(in_progress);
        Ok(self.uncommitted_docs())
    }

    fn rm_document(&mut self, term: tantivy::Term, in_progress: AsyncInProgress) -> usize {
        self.writer.delete_term(term);
        self.uncommitted_docs_in_progress_guards.push(in_progress);
        self.uncommitted_docs()
    }

    fn commit(&mut self, reload: impl FnOnce() -> tantivy::Result<()>) -> tantivy::Result<()> {
        self.writer.commit()?;
        reload()?;
        self.uncommitted_docs_in_progress_guards.clear();
        Ok(())
    }

    fn uncommitted_docs(&self) -> usize {
        self.uncommitted_docs_in_progress_guards.len()
    }

    pub(crate) fn has_uncommitted_docs(&self) -> bool {
        !self.uncommitted_docs_in_progress_guards.is_empty()
    }
}

pub(crate) struct IndexState<B: TantivyBackend> {
    pub(crate) backend: B,
    pub(crate) index: tantivy::Index,
    pub(crate) writer: RwLock<Writer>,
    pub(crate) reader: tantivy::IndexReader,
    pub(crate) schema: Schema,
}

impl<B: TantivyBackend> IndexState<B> {
    pub(crate) fn new(backend: B) -> anyhow::Result<Self> {
        let schema = backend.build_schema();
        let index = tantivy::Index::create_in_ram(schema.clone());
        backend.register_tokenizers(&index)?;
        let options = IndexWriterOptions::builder()
            .num_worker_threads(perf::num_workers().into())
            .build();
        let writer = index
            .writer_with_options(options)
            .map_err(|e| anyhow!("{}: failed to create writer: {e}", B::NAME))?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| anyhow!("{}: failed to create reader: {e}", B::NAME))?;
        Ok(Self {
            backend,
            index,
            writer: RwLock::new(Writer {
                writer,
                uncommitted_docs_in_progress_guards: Vec::new(),
            }),
            reader,
            schema,
        })
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TantivyStats {
    pub(crate) num_docs: u64,
    pub(crate) size_bytes: u64,
    pub(crate) segment_count: usize,
}

pub(crate) type TantivyStatsR = anyhow::Result<TantivyStats>;

/// A query-related failure caused by the caller's input (an unparsable query, or a query
/// construct that this endpoint cannot process) rather than an internal/actor failure.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct QueryError(pub(crate) String);

pub(crate) fn commit<B: TantivyBackend>(state: &IndexState<B>, key: &IndexKey) {
    let result = state
        .writer
        .write()
        .unwrap()
        .commit(|| state.reader.reload());
    if let Err(err) = result {
        error!("{}: failed to commit for {key}: {err}", B::NAME);
    }
}

pub(crate) fn handle_add_document<B: TantivyBackend>(
    state: &IndexState<B>,
    primary_id: PrimaryId,
    document: String,
    in_progress: AsyncInProgress,
) -> usize {
    let doc = state
        .backend
        .create_doc(&state.schema, primary_id, &document);
    let mut writer = state.writer.write().unwrap();
    match writer.add_document(doc, in_progress) {
        Ok(pending) => pending,
        Err(err) => {
            error!("{}: failed to add document {primary_id:?}: {err}", B::NAME);
            writer.uncommitted_docs()
        }
    }
}

pub(crate) fn primary_id_term(schema: &Schema, primary_id: PrimaryId) -> tantivy::Term {
    let primary_id_field = schema.get_field(PRIMARY_ID_FIELD).unwrap();
    tantivy::Term::from_field_u64(primary_id_field, u64::from(primary_id))
}

pub(crate) fn handle_remove_document<B: TantivyBackend>(
    state: &IndexState<B>,
    primary_id: PrimaryId,
    in_progress: AsyncInProgress,
) -> usize {
    let term = primary_id_term(&state.schema, primary_id);
    state.writer.write().unwrap().rm_document(term, in_progress)
}

pub(crate) fn handle_stats<B: TantivyBackend>(state: &IndexState<B>) -> TantivyStatsR {
    let searcher = state.reader.searcher();
    let num_docs = searcher.num_docs();
    let segment_count = searcher.segment_readers().len();
    let size_bytes = searcher
        .space_usage()
        .map_err(|e| anyhow!("{}: failed to compute space usage: {e}", B::NAME))?
        .total()
        .get_bytes();
    Ok(TantivyStats {
        num_docs,
        size_bytes,
        segment_count,
    })
}

pub(crate) fn find_partition_id<B: TantivyBackend>(
    table: &impl TableSearch,
    index_key: &IndexKey,
) -> anyhow::Result<PartitionId> {
    let (partition_id, _) = table.partition_id(index_key, None).ok_or_else(|| {
        anyhow!(
            "{}: partition id not found for index key {index_key:?}",
            B::NAME
        )
    })?;
    Ok(partition_id)
}

pub(crate) fn get_or_create_state<B: TantivyBackend, T: TableSearch>(
    states: &mut BTreeMap<IndexId, Arc<IndexState<B>>>,
    table: &RwLock<T>,
    key: &IndexKey,
    make_backend: impl FnOnce() -> B,
) -> Option<Arc<IndexState<B>>> {
    let index_id = table.read().unwrap().index_id(key)?;
    if let Some(state) = states.get(&index_id) {
        return Some(Arc::clone(state));
    }
    match IndexState::new(make_backend()) {
        Ok(state) => {
            let state = Arc::new(state);
            states.insert(index_id, Arc::clone(&state));
            Some(state)
        }
        Err(err) => {
            error!("{}: failed to create index state for {key}: {err}", B::NAME);
            None
        }
    }
}

pub(crate) fn get_state<B: TantivyBackend, T: TableSearch>(
    states: &BTreeMap<IndexId, Arc<IndexState<B>>>,
    table: &RwLock<T>,
    key: &IndexKey,
) -> Option<Arc<IndexState<B>>> {
    let index_id = table.read().unwrap().index_id(key)?;
    states.get(&index_id).cloned()
}

pub(crate) fn can_allocate_memory(
    rx_allocate: &watch::Receiver<Allocate>,
    allocate_prev: &mut Allocate,
    key: &IndexKey,
) -> bool {
    let allocate = *rx_allocate.borrow();
    if allocate == Allocate::Cannot {
        if *allocate_prev == Allocate::Can {
            error!("Unable to add document for index {key}: not enough memory");
        }
        *allocate_prev = allocate;
        return false;
    }
    *allocate_prev = allocate;
    true
}
