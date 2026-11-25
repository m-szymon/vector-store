/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

pub(crate) mod directory;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::mem;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::RwLockWriteGuard;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::anyhow;
use tantivy::Directory;
use tantivy::IndexWriter;
use tantivy::Opstamp;
use tantivy::ReloadPolicy;
use tantivy::Searcher;
use tantivy::TantivyDocument;
use tantivy::directory::WatchCallback;
use tantivy::directory::WatchHandle;
use tantivy::index::SegmentId;
use tantivy::index::SegmentMeta;
use tantivy::indexer::IndexWriterOptions;
use tantivy::indexer::MergePolicy;
use tantivy::indexer::UserOperation;
use tantivy::schema::FAST;
use tantivy::schema::INDEXED;
use tantivy::schema::Schema;
use tantivy::schema::TextOptions;
use tantivy::tokenizer::TextAnalyzer;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::sync::watch;
use tracing::error;

use crate::AsyncInProgress;
use crate::IndexKey;
use crate::memory::Allocate;
use crate::memory::Memory;
use crate::memory::MemoryExt;
use crate::perf;
use crate::table::IndexId;
use crate::table::PrimaryId;
use crate::table::TableSearch;
use crate::worker::Worker;
use crate::worker::WorkerExt;

pub(crate) const COMMIT_INTERVAL: Duration = Duration::from_secs(1);
pub(crate) const MAX_UNCOMMITTED_THRESHOLD: usize = 10_000;

/// What the documents of one commit are ordered by, for an index that asks for it.
pub(crate) type CommitKey = Box<[u64]>;

/// The fewest documents of one commit written as a group: see `Writer::write_held`.
const HELD_GROUP_DOCS: usize = 1024;

pub(crate) struct Writer {
    writer: IndexWriter,
    /// How many threads `writer` indexes with.
    threads: usize,
    // Handed to the reader on commit, which holds them until the documents are searchable.
    uncommitted_docs_in_progress_guards: Vec<AsyncInProgress>,
    /// The documents added with a key since the last commit, which the commit writes in the
    /// order of their keys. Tantivy gives a segment's documents their ids in the order they were
    /// added, each of its threads taking some of them as they come: written in the order of
    /// the keys, a thread's share is in that order, a sorted stretch of the segment, and a
    /// search that wants the highest keys reads the end of each stretch. Written as they came,
    /// rows that arrive in no order make a segment in no order, and nothing in it is settled
    /// until all of it is read. A slot is emptied when its row is removed before the commit.
    held: Vec<Option<(CommitKey, TantivyDocument)>>,
    /// Where in `held` each row's documents are.
    held_at: HashMap<PrimaryId, Vec<usize>>,
}

impl Writer {
    fn new(index: &tantivy::Index) -> anyhow::Result<Self> {
        let threads: usize = perf::num_workers().into();
        let options = IndexWriterOptions::builder()
            .num_worker_threads(threads)
            .build();
        let writer = index
            .writer_with_options(options)
            .map_err(|e| anyhow!("tantivy: failed to create writer: {e}"))?;
        Ok(Self {
            writer,
            threads,
            uncommitted_docs_in_progress_guards: Vec::new(),
            held: Vec::new(),
            held_at: HashMap::new(),
        })
    }

    /// Adds `doc`. With a key, at the next commit, in the order of the keys among that
    /// commit's documents: see `held`.
    fn add_document(
        &mut self,
        primary_id: PrimaryId,
        key: Option<CommitKey>,
        doc: TantivyDocument,
        mut in_progress: AsyncInProgress,
    ) -> tantivy::Result<usize> {
        match key {
            Some(key) => {
                self.held_at
                    .entry(primary_id)
                    .or_default()
                    .push(self.held.len());
                self.held.push(Some((key, doc)));
            }
            None => {
                self.writer.add_document(doc)?;
            }
        }
        // A held document is as good as added: a removal that comes before the commit takes it
        // out of `held`.
        in_progress.applied();
        self.uncommitted_docs_in_progress_guards.push(in_progress);
        Ok(self.uncommitted_docs())
    }

    fn rm_document(
        &mut self,
        primary_id: PrimaryId,
        term: tantivy::Term,
        in_progress: AsyncInProgress,
    ) -> usize {
        // A delete reaches the documents written before it, and the held ones are written
        // after: the row's are dropped here instead.
        for at in self.held_at.remove(&primary_id).unwrap_or_default() {
            self.held[at] = None;
        }
        self.writer.delete_term(term);
        self.uncommitted_docs_in_progress_guards.push(in_progress);
        self.uncommitted_docs()
    }

    /// Writes the held documents, in the order of their keys.
    ///
    /// As groups (`IndexWriter::run`), each of which Tantivy writes into one segment in the
    /// order given: added one by one, the documents would go to whichever thread was free, and
    /// with many threads and few documents a thread's share would be too short a stretch to
    /// read as a run. The documents are dealt to the groups in turn, so every group is in the
    /// order of the keys and spans all of them; as many groups as threads, so that the threads
    /// all write, but none shorter than `HELD_GROUP_DOCS`.
    fn write_held(&mut self) {
        self.held_at.clear();
        let mut held: Vec<_> = self.held.drain(..).flatten().collect();
        if held.is_empty() {
            return;
        }
        // Stable: rows with one key stay in the order they came.
        held.sort_by(|(a, _), (b, _)| a.cmp(b));
        let groups = (held.len() / HELD_GROUP_DOCS).clamp(1, self.threads);
        let mut dealt: Vec<Vec<UserOperation>> = (0..groups)
            .map(|_| Vec::with_capacity(held.len() / groups + 1))
            .collect();
        for (at, (_, doc)) in held.into_iter().enumerate() {
            dealt[at % groups].push(UserOperation::Add(doc));
        }
        for group in dealt {
            if let Err(err) = self.writer.run(group) {
                error!("tantivy: failed to add documents: {err}");
            }
        }
    }

    /// Commits the documents to the index and returns the commit opstamp with their in-progress
    /// guards.
    fn commit(&mut self) -> tantivy::Result<(Opstamp, Vec<AsyncInProgress>)> {
        self.write_held();
        let opstamp = self.writer.commit()?;
        Ok((
            opstamp,
            mem::take(&mut self.uncommitted_docs_in_progress_guards),
        ))
    }

    fn uncommitted_docs(&self) -> usize {
        self.uncommitted_docs_in_progress_guards.len()
    }

    /// Re-adds `docs` under fresh opstamps after deleting whatever carried their terms: the delete
    /// applies to documents added before it, so the new copy survives. For moving rows between
    /// segments; nothing in progress is attached, since the rows were already indexed.
    pub(crate) fn rewrite_documents(
        &mut self,
        docs: impl IntoIterator<Item = (tantivy::Term, TantivyDocument)>,
    ) -> tantivy::Result<usize> {
        // A held document of one of these rows would be written after its copy here.
        self.write_held();
        let mut count = 0;
        for (term, doc) in docs {
            self.writer.delete_term(term);
            self.writer.add_document(doc)?;
            count += 1;
        }
        Ok(count)
    }

    /// Asks the writer to merge `segments`. The merge runs on Tantivy's own thread; the returned
    /// future only reports its outcome and can be dropped.
    pub(crate) fn merge(
        &mut self,
        segments: &[SegmentId],
    ) -> tantivy::FutureResult<Option<SegmentMeta>> {
        self.writer.merge(segments)
    }

    fn has_uncommitted_docs(&self) -> bool {
        !self.uncommitted_docs_in_progress_guards.is_empty()
    }
}

#[derive(Default)]
struct ReloadState {
    // Committed documents by commit opstamp, held until a reload makes them searchable.
    unsearchable_docs: BTreeMap<Opstamp, Vec<AsyncInProgress>>,
    reloaded_opstamp: Opstamp,
    reload_failed: bool,
}

/// A thin wrapper over the Tantivy reader that reloads it whenever the index metadata changes and
/// drops the in-progress guards of committed documents once they are searchable.
pub(crate) struct Reader {
    // Shared with the reload task.
    inner: Arc<ReaderInner>,
    _meta_watch: WatchHandle,
    // Lets the tick request a reload through the same channel as the watch callback.
    reload_tx: watch::Sender<()>,
}

struct ReaderInner {
    index: tantivy::Index,
    reader: tantivy::IndexReader,
    key: IndexKey,
    reload_state: Mutex<ReloadState>,
    // Held across a reload, so that the reload task and a caller reloading on its own
    // (`Index::commit_and_reload`) never store an older searcher or opstamp after a newer one.
    reloading: Mutex<()>,
    worker: async_channel::Sender<Worker>,
}

impl ReaderInner {
    /// Reloads the reader and drops the guards of the committed documents it made searchable.
    fn reload(&self) -> tantivy::Result<()> {
        let _reloading = self.reloading.lock().unwrap();
        // The index meta holds the opstamp of its last commit, so a reload after reading it
        // serves at least that commit.
        let reloaded = self.index.load_metas().and_then(|meta| {
            self.reader.reload()?;
            Ok(meta.opstamp)
        });
        let mut reload_state = self.reload_state.lock().unwrap();
        match reloaded {
            Ok(opstamp) => {
                reload_state.reload_failed = false;
                reload_state.reloaded_opstamp = opstamp;
                reload_state
                    .unsearchable_docs
                    .retain(|committed, _| *committed > opstamp);
                Ok(())
            }
            Err(err) => {
                reload_state.reload_failed = true;
                error!("tantivy: failed to reload reader for {}: {err}", self.key);
                Err(err)
            }
        }
    }

    /// Runs the reload on the worker and returns once it has finished.
    async fn reload_on_worker(self: &Arc<Self>) {
        let (done_tx, done_rx) = oneshot::channel();
        let inner = Arc::clone(self);
        self.worker
            .spawn_indexing_blocking(move || {
                _ = inner.reload();
                _ = done_tx.send(());
            })
            .await;
        _ = done_rx.await;
    }
}

impl Reader {
    fn new(
        index: &tantivy::Index,
        key: &IndexKey,
        worker: async_channel::Sender<Worker>,
    ) -> anyhow::Result<Self> {
        let reader: tantivy::IndexReader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| anyhow!("tantivy: failed to create reader: {e}"))?;
        let inner = Arc::new(ReaderInner {
            index: index.clone(),
            reader,
            key: key.clone(),
            reload_state: Mutex::new(ReloadState::default()),
            reloading: Mutex::new(()),
            worker,
        });
        let (reload_tx, mut reload_rx) = watch::channel(());
        let callback = {
            let reload_tx = reload_tx.clone();
            move || _ = reload_tx.send(())
        };
        let meta_watch = index
            .directory()
            .watch(WatchCallback::new(callback))
            .map_err(|e| anyhow!("tantivy: failed to watch index metas: {e}"))?;
        tokio::spawn({
            let inner = Arc::clone(&inner);
            async move {
                // The reloads run one at a time, so a reload never stores an older searcher or
                // opstamp after a newer one. Ends when the reader and its watch callback are
                // dropped.
                while reload_rx.changed().await.is_ok() {
                    inner.reload_on_worker().await;
                }
            }
        });
        Ok(Self {
            inner,
            _meta_watch: meta_watch,
            reload_tx,
        })
    }

    pub(crate) fn searcher(&self) -> Searcher {
        self.inner.reader.searcher()
    }

    fn request_reload(&self) {
        _ = self.reload_tx.send(());
    }

    /// Whether committed documents wait for a reload or the last reload failed.
    fn needs_reload(&self) -> bool {
        let reload_state = self.inner.reload_state.lock().unwrap();
        reload_state.reload_failed || !reload_state.unsearchable_docs.is_empty()
    }

    /// Holds the guards until a reload makes its documents searchable.
    fn drop_guards_when_searchable(&self, commit_opstamp: Opstamp, guards: Vec<AsyncInProgress>) {
        let mut reload_state = self.inner.reload_state.lock().unwrap();
        // The watch callback of this commit may have reloaded before the guards got here.
        if commit_opstamp > reload_state.reloaded_opstamp {
            reload_state
                .unsearchable_docs
                .insert(commit_opstamp, guards);
        }
    }
}

/// A tantivy index with its writer and reader, and `state`: whatever a kind of index keeps
/// alongside, such as caches of per-segment data. The full-text index keeps nothing.
pub(crate) struct Index<E = ()> {
    pub(crate) index: tantivy::Index,
    writer: RwLock<Writer>,
    /// The additions, removals and commits waiting for the writer now: see `let_writes_in`.
    writes_waiting: AtomicU64,
    pub(crate) reader: Reader,
    pub(crate) schema: Schema,
    pub(crate) state: E,
}

impl Index {
    pub(crate) fn new(
        schema: Schema,
        tokenizer_name: &str,
        tokenizer: TextAnalyzer,
        key: &IndexKey,
        worker: async_channel::Sender<Worker>,
    ) -> anyhow::Result<Self> {
        Self::with_state(schema, tokenizer_name, tokenizer, key, worker, ())
    }
}

impl<E> Index<E> {
    pub(crate) fn with_state(
        schema: Schema,
        tokenizer_name: &str,
        tokenizer: TextAnalyzer,
        key: &IndexKey,
        worker: async_channel::Sender<Worker>,
        state: E,
    ) -> anyhow::Result<Self> {
        // In a directory whose meta lock wakes its waiter: a reload and Tantivy's removal of
        // unused files meet on that lock, and the RAM directory's sleeps 100 ms between tries.
        let index = tantivy::Index::create(
            directory::MemoryDirectory::default(),
            schema.clone(),
            tantivy::IndexSettings::default(),
        )
        .map_err(|e| anyhow!("tantivy: failed to create index: {e}"))?;
        index.tokenizers().register(tokenizer_name, tokenizer);
        let reader = Reader::new(&index, key, worker)?;
        let writer = RwLock::new(Writer::new(&index)?);
        Ok(Self {
            index,
            writer,
            writes_waiting: AtomicU64::new(0),
            reader,
            schema,
            state,
        })
    }

    /// Replaces Tantivy's default merge policy.
    pub(crate) fn set_merge_policy(&self, policy: Box<dyn MergePolicy>) {
        self.writer.read().unwrap().writer.set_merge_policy(policy);
    }

    /// Whether the writer holds uncommitted documents, or None while someone holds the writer.
    /// Never waits for the writer: the actors ask on their ticks, and a writer held for seconds --
    /// a pattern index moving rows between segments -- would stop an actor answering anything.
    pub(crate) fn has_uncommitted_docs(&self) -> Option<bool> {
        let writer = self.writer.try_read().ok()?;
        Some(writer.has_uncommitted_docs())
    }

    /// The writer, locked, for a caller that has to read the index and write what it read back
    /// with nothing written in between, as moving rows between segments does.
    pub(crate) fn lock_writer(&self) -> RwLockWriteGuard<'_, Writer> {
        self.writer.write().unwrap()
    }

    /// The writer, locked for an addition, a removal or a commit, counted while it waits.
    fn write(&self) -> RwLockWriteGuard<'_, Writer> {
        self.writes_waiting.fetch_add(1, Ordering::Relaxed);
        let writer = self.writer.write().unwrap();
        self.writes_waiting.fetch_sub(1, Ordering::Relaxed);
        writer
    }

    /// Waits, for `limit` at most, until no addition, removal or commit waits for the writer.
    /// For a caller of `lock_writer` that takes the writer again and again: the lock is not
    /// handed over in the order it was asked for, and a thread that releases it and asks at once
    /// gets it back before the ones waiting have woken, so they would wait for the last of its
    /// holds. A repair moving 10M rows in holds of 0.7 s waited for none of its 206 holds, while
    /// a sixth of the rows written meanwhile waited over 10 s; with the writes let in between,
    /// none waited 5 s. The limit keeps a steady stream of writes from stopping the caller.
    pub(crate) fn let_writes_in(&self, limit: Duration) {
        let deadline = std::time::Instant::now() + limit;
        while self.writes_waiting.load(Ordering::Relaxed) > 0
            && std::time::Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Commits `writer` and reloads the reader before returning, for a caller that has to read
    /// what it has just committed rather than wait for the reload task.
    pub(crate) fn commit_and_reload(&self, writer: &mut Writer) -> tantivy::Result<()> {
        let (commit_opstamp, guards) = writer.commit()?;
        self.reader
            .drop_guards_when_searchable(commit_opstamp, guards);
        self.reader.inner.reload()
    }
}

pub(crate) fn build_schema(text_options: TextOptions) -> Schema {
    let mut schema_builder = Schema::builder();
    schema_builder.add_u64_field("primary_id", INDEXED | FAST);
    schema_builder.add_text_field("text", text_options);
    schema_builder.build()
}

fn create_doc(schema: &Schema, primary_id: PrimaryId, document: &str) -> TantivyDocument {
    let primary_id_field = schema.get_field("primary_id").unwrap();
    let text_field = schema.get_field("text").unwrap();

    let mut doc = TantivyDocument::new();
    doc.add_u64(primary_id_field, u64::from(primary_id));
    doc.add_text(text_field, document);
    doc
}

fn commit<E>(index: &Index<E>, key: &IndexKey) {
    let result = index.write().commit();
    match result {
        Ok((commit_opstamp, guards)) => index
            .reader
            .drop_guards_when_searchable(commit_opstamp, guards),
        Err(err) => error!("tantivy: failed to commit for {key}: {err}"),
    }
}

fn add_document<E>(
    index: &Index<E>,
    primary_id: PrimaryId,
    commit_key: Option<CommitKey>,
    doc: TantivyDocument,
    in_progress: AsyncInProgress,
) -> usize {
    let mut writer = index.write();
    match writer.add_document(primary_id, commit_key, doc, in_progress) {
        Ok(pending) => pending,
        Err(err) => {
            error!("tantivy: failed to add document {primary_id:?}: {err}");
            writer.uncommitted_docs()
        }
    }
}

fn create_term(schema: &Schema, primary_id: PrimaryId) -> tantivy::Term {
    let primary_id_field = schema.get_field("primary_id").unwrap();
    tantivy::Term::from_field_u64(primary_id_field, u64::from(primary_id))
}

fn remove_document<E>(
    index: &Index<E>,
    primary_id: PrimaryId,
    in_progress: AsyncInProgress,
) -> usize {
    let term = create_term(&index.schema, primary_id);
    index.write().rm_document(primary_id, term, in_progress)
}

/// The tantivy indexes of one index key, one per index id.
pub(crate) struct Indexes<T, F, E = ()> {
    key: IndexKey,
    table: Arc<RwLock<T>>,
    worker: async_channel::Sender<Worker>,
    commit_interval: tokio::time::Interval,
    commit_threshold: usize,
    indexes: BTreeMap<IndexId, Arc<Index<E>>>,
    allocate_rx: watch::Receiver<Allocate>,
    allocate_prev: Allocate,
    new_index: F,
}

impl<T, F, E> Indexes<T, F, E>
where
    T: TableSearch,
    F: Fn(&IndexKey, async_channel::Sender<Worker>) -> anyhow::Result<Index<E>>,
    E: Send + Sync + 'static,
{
    pub(crate) async fn new(
        key: IndexKey,
        table: Arc<RwLock<T>>,
        worker: async_channel::Sender<Worker>,
        memory: mpsc::Sender<Memory>,
        commit_interval: Duration,
        commit_threshold: usize,
        new_index: F,
    ) -> Self {
        let allocate_rx = memory.subscribe_allocate().await;
        let mut commit_interval = tokio::time::interval(commit_interval);
        commit_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        Self {
            key,
            table,
            worker,
            commit_interval,
            commit_threshold,
            indexes: BTreeMap::new(),
            allocate_rx,
            allocate_prev: Allocate::Can,
            new_index,
        }
    }

    pub(crate) async fn add_document(
        &mut self,
        primary_id: PrimaryId,
        document: String,
        in_progress: AsyncInProgress,
    ) {
        self.add_document_with(primary_id, in_progress, None, move |index| {
            create_doc(&index.schema, primary_id, &document)
        })
        .await;
    }

    /// Adds the document `make_doc` builds, for an index whose documents carry more than the
    /// text. It runs on the worker, with the index the document goes to. With a `commit_key`
    /// the documents of a commit are written in the order of their keys.
    pub(crate) async fn add_document_with(
        &mut self,
        primary_id: PrimaryId,
        in_progress: AsyncInProgress,
        commit_key: Option<CommitKey>,
        make_doc: impl FnOnce(&Index<E>) -> TantivyDocument + Send + 'static,
    ) {
        let Some(index) = self.get_or_create() else {
            return;
        };
        if !self.can_allocate_memory() {
            return;
        }
        let key = self.key.clone();
        let commit_threshold = self.commit_threshold;
        // A row of a full scan builds an index nobody searches yet: it waits for the searches
        // and the live writes of the other indexes.
        let background = in_progress.is_fullscan();
        self.worker
            .spawn_write_blocking(background, move || {
                let doc = make_doc(&index);
                let pending = add_document(&index, primary_id, commit_key, doc, in_progress);
                if pending >= commit_threshold {
                    commit(&index, &key);
                }
            })
            .await;
    }

    pub(crate) async fn remove_document(
        &mut self,
        primary_id: PrimaryId,
        in_progress: AsyncInProgress,
    ) {
        let Some(index) = self.get_or_create() else {
            return;
        };
        let key = self.key.clone();
        let commit_threshold = self.commit_threshold;
        // In the class of the add that follows it, so that the two run in the order sent.
        let background = in_progress.is_fullscan();
        self.worker
            .spawn_write_blocking(background, move || {
                let pending = remove_document(&index, primary_id, in_progress);
                if pending >= commit_threshold {
                    commit(&index, &key);
                }
            })
            .await;
    }

    pub(crate) fn get(&self, index_key: &IndexKey) -> Option<Arc<Index<E>>> {
        let index_id = self.table.read().unwrap().index_id(index_key)?;
        self.indexes.get(&index_id).cloned()
    }

    pub(crate) fn count(&self, index_key: &IndexKey) -> usize {
        self.get(index_key)
            .map(|index| index.reader.searcher().num_docs() as usize)
            .unwrap_or(0)
    }

    /// Every index of the key, one per index id.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Arc<Index<E>>> {
        self.indexes.values()
    }

    pub(crate) async fn commit_interval_tick(&mut self) {
        self.commit_interval.tick().await;
    }

    /// Commits every index with uncommitted documents and requests a reload of every other index
    /// whose committed documents are not searchable yet. An index whose writer is held right now
    /// is left for the next tick rather than waited for.
    pub(crate) async fn commit_or_reload(&self) {
        for index in self.indexes.values() {
            let Some(uncommitted) = index.has_uncommitted_docs() else {
                continue;
            };
            if uncommitted {
                let index = Arc::clone(index);
                let key = self.key.clone();
                self.worker
                    .spawn_indexing_blocking(move || commit(&index, &key))
                    .await;
            } else if index.reader.needs_reload() {
                index.reader.request_reload();
            }
        }
    }

    fn get_or_create(&mut self) -> Option<Arc<Index<E>>> {
        let index_id = self.table.read().unwrap().index_id(&self.key)?;
        if let Some(index) = self.indexes.get(&index_id) {
            return Some(Arc::clone(index));
        }
        match (self.new_index)(&self.key, self.worker.clone()) {
            Ok(index) => {
                let index = Arc::new(index);
                self.indexes.insert(index_id, Arc::clone(&index));
                Some(index)
            }
            Err(err) => {
                error!("tantivy: failed to create index for {}: {err}", self.key);
                None
            }
        }
    }

    fn can_allocate_memory(&mut self) -> bool {
        let allocate = *self.allocate_rx.borrow();
        if allocate == Allocate::Cannot {
            if self.allocate_prev == Allocate::Can {
                error!(
                    "Unable to add document for index {}: not enough memory",
                    self.key
                );
            }
            self.allocate_prev = allocate;
            return false;
        }
        self.allocate_prev = allocate;
        true
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;
    use std::sync::RwLock;

    use scylla::value::CqlValue;
    use tokio::sync::mpsc;
    use tokio::sync::watch;

    use crate::IndexKey;
    use crate::PrimaryKey;
    use crate::memory::Allocate;
    use crate::memory::Memory;
    use crate::table::IndexIdGenerator;
    use crate::table::MockTableSearch;
    use crate::table::PartitionId;

    pub(crate) fn make_table_with_keys() -> Arc<RwLock<MockTableSearch>> {
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

    pub(crate) fn make_index_key() -> IndexKey {
        IndexKey::new(&"ks".into(), &"idx".into())
    }

    pub(crate) fn make_memory_actor(allocate: Allocate) -> mpsc::Sender<Memory> {
        let (tx, mut rx) = mpsc::channel::<Memory>(1);
        tokio::spawn(async move {
            let (watch_tx, _) = watch::channel(allocate);
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
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use tantivy::schema::STORED;
    use tantivy::schema::TEXT;
    use tantivy::schema::Value;
    use tantivy::tokenizer::SimpleTokenizer;
    use tantivy::tokenizer::TextAnalyzer;

    use super::test_support::make_index_key;
    use super::*;

    /// A reload that finds the meta lock held, as Tantivy holds it after a commit while it looks
    /// for files to delete, goes on as soon as the lock is released, not up to 100 ms later.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_reload_goes_on_as_soon_as_the_meta_lock_is_released() {
        use tantivy::directory::META_LOCK;
        let mut schema = Schema::builder();
        schema.add_u64_field("primary_id", INDEXED | STORED | FAST);
        schema.add_text_field("text", TEXT | STORED);
        let index = Index::new(
            schema.build(),
            "default",
            TextAnalyzer::from(SimpleTokenizer::default()),
            &make_index_key(),
            crate::worker::new(),
        )
        .unwrap();
        let held = index.index.directory().acquire_lock(&META_LOCK).unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            // Taken before the release, so the reload cannot have the lock earlier.
            let released = std::time::Instant::now();
            drop(held);
            released
        });
        index.reader.inner.reload().unwrap();
        let reloaded = std::time::Instant::now();
        let released = release.join().unwrap();
        assert!(reloaded >= released);
        // With Tantivy's RAM directory the reload would retry 90 ms after the release.
        let after_release = reloaded - released;
        assert!(
            after_release < Duration::from_millis(50),
            "{after_release:?}"
        );
    }

    /// The documents added with a key are written at the commit in the order of the keys; a row
    /// removed before the commit is not written, and one added again after that is there once,
    /// as it was added last.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn held_documents_are_written_in_the_order_of_their_keys_less_the_removed() {
        let mut schema = Schema::builder();
        schema.add_u64_field("primary_id", INDEXED | STORED | FAST);
        schema.add_text_field("text", TEXT | STORED);
        let index = Index::new(
            schema.build(),
            "default",
            TextAnalyzer::from(SimpleTokenizer::default()),
            &make_index_key(),
            crate::worker::new(),
        )
        .unwrap();
        let guard = || AsyncInProgress::fullscan(tokio::sync::mpsc::channel(1).0);
        let doc = |id: u64, text: &str| create_doc(&index.schema, id.into(), text);
        let key = |key: u64| Some(CommitKey::from([key]));
        let term = |id: u64| create_term(&index.schema, id.into());
        {
            let mut writer = index.lock_writer();
            writer
                .add_document(1.into(), key(30), doc(1, "one"), guard())
                .unwrap();
            writer
                .add_document(2.into(), key(10), doc(2, "gone"), guard())
                .unwrap();
            writer
                .add_document(3.into(), key(20), doc(3, "old"), guard())
                .unwrap();
            writer
                .add_document(4.into(), key(5), doc(4, "four"), guard())
                .unwrap();
            writer.rm_document(2.into(), term(2), guard());
            writer.rm_document(3.into(), term(3), guard());
            writer
                .add_document(3.into(), key(20), doc(3, "new"), guard())
                .unwrap();
            index.commit_and_reload(&mut writer).unwrap();
        }
        let searcher = index.reader.searcher();
        assert_eq!(searcher.num_docs(), 3);
        // One group, so one segment, its documents in the order of the keys: 5, 20, 30.
        let segments = searcher.segment_readers();
        assert_eq!(segments.len(), 1);
        let ids = segments[0].fast_fields().u64("primary_id").unwrap();
        let written: Vec<u64> = (0..3).filter_map(|doc_id| ids.first(doc_id)).collect();
        assert_eq!(written, vec![4, 3, 1]);
        let text = index.schema.get_field("text").unwrap();
        let stored: TantivyDocument = searcher.doc(tantivy::DocAddress::new(0, 1)).unwrap();
        assert_eq!(
            stored.get_first(text).and_then(|value| value.as_str()),
            Some("new")
        );
    }

    /// A write that waits for the writer gets it before a holder that lets the writes in takes
    /// it again; without that the holder, asking at once, would get it back first.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_waiting_write_goes_before_the_next_hold() {
        let mut schema = Schema::builder();
        schema.add_u64_field("primary_id", INDEXED | STORED | FAST);
        schema.add_text_field("text", TEXT | STORED);
        let index = Arc::new(
            Index::new(
                schema.build(),
                "default",
                TextAnalyzer::from(SimpleTokenizer::default()),
                &make_index_key(),
                crate::worker::new(),
            )
            .unwrap(),
        );
        for round in 1..=20 {
            let held = index.lock_writer();
            let waiter = std::thread::spawn({
                let index = Arc::clone(&index);
                move || remove_document(&index, PrimaryId::from(1), AsyncInProgress::none())
            });
            while index.writes_waiting.load(Ordering::Relaxed) == 0 {
                std::thread::sleep(Duration::from_millis(1));
            }
            drop(held);
            index.let_writes_in(Duration::from_secs(10));
            // The waiting write has the writer or has had it: the next hold comes after it.
            let held = index.lock_writer();
            assert_eq!(held.uncommitted_docs(), round);
            drop(held);
            waiter.join().unwrap();
        }
    }
}
