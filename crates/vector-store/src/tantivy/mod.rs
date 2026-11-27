/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

pub(crate) mod directory;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::mem;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::RwLock;
use std::sync::RwLockWriteGuard;
use std::sync::TryLockError;
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
use tantivy::schema::Field;
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

/// How many rows a stretch of a build holds, on average, when the fullest one is written: see
/// `Stretches`. The fullest holds about twice as many, which is what a commit of one writes.
const STRETCH_HELD_ROWS: usize = 4096;

/// The fewest rows of a build held before its size is estimated and the stretches cut, and the
/// fewest for each stretch cut.
const STRETCH_SAMPLE_ROWS: usize = 16_384;
const STRETCH_SAMPLE_ROWS_EACH: usize = 16;

/// The most stretches a build is cut into.
const STRETCH_MAX: usize = 65_536;

/// What a row of a build brings for `Stretches`, besides its key.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BuildRow {
    /// The row's token, as an unsigned key: how densely the first rows' tokens lie on the ring
    /// says how many rows the table has.
    pub(crate) token: u64,
    /// The rows a stretch is to end with.
    pub(crate) stretch_rows: u32,
    /// The fast field that says a document was written where it belongs in the order: 1 on
    /// the documents written by stretch, 0 on the ones written as they came.
    pub(crate) placed: &'static str,
}

type HeldRow = (CommitKey, PrimaryId, u64, TantivyDocument, AsyncInProgress);

/// The rows of a build, held by stretch of the order.
///
/// A build's rows come in no order, a commit of them makes segments that span the whole order,
/// and a repair then writes every row again: of a build of 10M rows, 240 s of 300. Held here
/// instead, by which stretch of the order their key is in, and written a stretch at a time --
/// the fullest, once `STRETCH_HELD_ROWS` a stretch are held -- they make segments of one
/// stretch each, which merge into the segment the stretch ends as, and nothing is written
/// twice: 90 s. A row keeps its guard until the commit that writes it, so the scan that sent
/// it counts it as indexed only then.
///
/// How many stretches depends on how many rows the build brings, which nothing says. It is
/// estimated from the tokens of the first rows (`estimate_rows`), and the stretches' bounds are
/// then cut from those rows' keys, as many of them to each. That takes keys that come in no
/// order; where the estimate is not to be had -- rows that mostly share partitions, which come
/// together -- the rows are written as they came, as before (`plain`).
struct Stretches {
    stretch_rows: usize,
    placed: Field,
    /// The lowest key of each stretch but the first; empty until cut, and while it is, the
    /// rows wait in one bucket.
    bounds: Vec<CommitKey>,
    /// Whether the stretches are cut, and the rows held at which to try.
    cut: bool,
    cut_at: usize,
    /// Whether the build's rows are written as they came: no estimate.
    plain: bool,
    buckets: Vec<Vec<Option<HeldRow>>>,
    /// Where each held row's documents are: bucket and place in it.
    at: HashMap<PrimaryId, Vec<(u32, u32)>>,
    rows: usize,
    /// The rows that came since the last commit, and whether any came since the last commit
    /// of a tick.
    since_commit: usize,
    arrived: bool,
}

/// How many rows a table has, from the tokens of `tokens.len()` of them read by a scan in
/// flight, or None when they do not say.
///
/// A scan reads some ranges of the ring at once, each from its lowest token up, so the tokens
/// it has read lie in as many clumps, as dense as the table's partitions: the median gap
/// between neighbours is ln 2 of the mean gap between partitions, whatever lies between the
/// clumps. The rows are as many times the partitions as the sample's. With fewer than a
/// quarter as many partitions as rows there is nothing to cut: the rows of a partition come
/// together and lie together.
fn estimate_rows(tokens: &mut Vec<u64>) -> Option<u64> {
    let rows = tokens.len();
    tokens.sort_unstable();
    tokens.dedup();
    if tokens.len() < 2 || tokens.len() * 4 < rows {
        return None;
    }
    let mut gaps: Vec<u64> = tokens.windows(2).map(|pair| pair[1] - pair[0]).collect();
    let middle = gaps.len() / 2;
    let median = *gaps.select_nth_unstable(middle).1;
    let partitions = std::f64::consts::LN_2 * 2f64.powi(64) / median as f64;
    Some((partitions * rows as f64 / tokens.len() as f64) as u64)
}

impl Stretches {
    fn new(stretch_rows: usize, placed: Field) -> Self {
        Self {
            stretch_rows: stretch_rows.max(1),
            placed,
            bounds: Vec::new(),
            cut: false,
            cut_at: STRETCH_SAMPLE_ROWS,
            plain: false,
            buckets: vec![Vec::new()],
            at: HashMap::new(),
            rows: 0,
            since_commit: 0,
            arrived: false,
        }
    }

    fn add(&mut self, row: HeldRow) {
        self.since_commit += 1;
        self.arrived = true;
        self.hold(row);
        if !self.cut && self.rows >= self.cut_at {
            self.cut();
        }
    }

    fn hold(&mut self, row: HeldRow) {
        let bucket = self.bounds.partition_point(|bound| *bound <= row.0);
        self.at
            .entry(row.1)
            .or_default()
            .push((bucket as u32, self.buckets[bucket].len() as u32));
        self.buckets[bucket].push(Some(row));
        self.rows += 1;
    }

    /// Cuts the order into stretches of as many of the held rows each, if there are enough
    /// rows for the stretches the build's estimated size asks for; `plain` if it has none, or
    /// one stretch would do.
    fn cut(&mut self) {
        let mut tokens: Vec<u64> = self.buckets[0].iter().flatten().map(|row| row.2).collect();
        let Some(rows) = estimate_rows(&mut tokens) else {
            self.plain = true;
            return;
        };
        let count = (rows as usize).div_ceil(self.stretch_rows).min(STRETCH_MAX);
        if count < 2 {
            // One stretch is the whole order: nothing to hold the rows for.
            self.plain = true;
            return;
        }
        if self.rows < count * STRETCH_SAMPLE_ROWS_EACH {
            self.cut_at = count * STRETCH_SAMPLE_ROWS_EACH;
            return;
        }
        let mut held: Vec<HeldRow> = self.buckets[0].drain(..).flatten().collect();
        held.sort_by(|a, b| a.0.cmp(&b.0));
        self.bounds = (1..count)
            .map(|stretch| held[stretch * held.len() / count].0.clone())
            .collect();
        self.bounds.dedup();
        self.cut = true;
        self.buckets = (0..=self.bounds.len()).map(|_| Vec::new()).collect();
        self.at.clear();
        self.rows = 0;
        for row in held {
            self.hold(row);
        }
    }

    /// Takes out the held documents of `primary_id`, and returns their guards.
    fn remove(&mut self, primary_id: PrimaryId) -> Vec<AsyncInProgress> {
        let mut guards = Vec::new();
        for (bucket, place) in self.at.remove(&primary_id).unwrap_or_default() {
            if let Some(row) = self.buckets[bucket as usize][place as usize].take() {
                self.rows -= 1;
                guards.push(row.4);
            }
        }
        guards
    }

    /// Whether more rows are held than a build keeps: the fullest stretch is to be written.
    fn is_full(&self) -> bool {
        self.cut && self.rows > self.buckets.len() * STRETCH_HELD_ROWS
    }

    /// The rows of one stretch, in the order of their keys: the fullest, or with `first` the
    /// first that holds any.
    fn take(&mut self, first: bool) -> Vec<HeldRow> {
        let bucket = if first {
            self.buckets.iter().position(|bucket| !bucket.is_empty())
        } else {
            (0..self.buckets.len()).max_by_key(|&bucket| self.buckets[bucket].len())
        };
        let Some(bucket) = bucket else {
            return Vec::new();
        };
        let mut held: Vec<HeldRow> = std::mem::take(&mut self.buckets[bucket])
            .into_iter()
            .flatten()
            .collect();
        for row in &held {
            if let Some(places) = self.at.get_mut(&row.1) {
                places.retain(|place| place.0 as usize != bucket);
                if places.is_empty() {
                    self.at.remove(&row.1);
                }
            }
        }
        self.rows -= held.len();
        held.sort_by(|a, b| a.0.cmp(&b.0));
        held
    }
}

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
    /// The rows of a build held by stretch of the order, once one has come.
    stretches: Option<Stretches>,
}

impl Writer {
    fn new(index: &tantivy::Index) -> anyhow::Result<Self> {
        // Half as many threads as the pool has workers. They are the writer's own, outside the
        // pool and its policy, and every index has a writer: with as many as workers, four
        // indexes built at once on four cores had sixteen, and the searches of a fifth a p99 of
        // 10.6 ms; with half, 8.2 ms, the builds as fast. What it costs is the repair of a
        // pattern index, which writes through them: 12% longer on an idle node.
        let workers: usize = perf::num_workers().into();
        let threads = (workers / 2).max(1);
        // A benchmark test's switch: another number of threads.
        #[cfg(test)]
        let threads = match WRITER_THREADS.load(Ordering::Relaxed) {
            0 => threads,
            threads => threads as usize,
        };
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
            stretches: None,
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
        build: Option<BuildRow>,
    ) -> tantivy::Result<usize> {
        // A row of a build is held by stretch of the order, or, where the build's size is not
        // to be had, written as the other rows are: see `Stretches`.
        let mut doc = doc;
        if let (Some(_), Some(build)) = (&key, build) {
            if self.stretches.is_none() {
                let placed = self.writer.index().schema().get_field(build.placed)?;
                self.stretches = Some(Stretches::new(build.stretch_rows as usize, placed));
            }
            let stretches = self.stretches.as_mut().unwrap();
            if !stretches.plain {
                in_progress.applied();
                let key = key.unwrap();
                stretches.add((key, primary_id, build.token, doc, in_progress));
                if stretches.plain {
                    self.write_plain();
                }
                return Ok(self.uncommitted_docs());
            }
            doc.add_u64(stretches.placed, 0);
        }
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
        if let Some(stretches) = &mut self.stretches {
            let guards = stretches.remove(primary_id);
            self.uncommitted_docs_in_progress_guards.extend(guards);
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

    /// Writes one stretch of a build's held rows, as contiguous groups in the order of the
    /// keys, one for each thread, marked as placed: the fullest stretch, or with `first` the
    /// first that holds any. Their guards go with the next commit's.
    fn write_stretch(&mut self, first: bool) {
        let Some(stretches) = &mut self.stretches else {
            return;
        };
        let placed = stretches.placed;
        let held = stretches.take(first);
        let groups = (held.len() / HELD_GROUP_DOCS).clamp(1, self.threads);
        let size = held.len().div_ceil(groups).max(1);
        let mut group = Vec::with_capacity(size);
        for (_, _, _, mut doc, guard) in held {
            doc.add_u64(placed, 1);
            group.push(UserOperation::Add(doc));
            self.uncommitted_docs_in_progress_guards.push(guard);
            if group.len() == size {
                self.run(std::mem::replace(&mut group, Vec::with_capacity(size)));
            }
        }
        self.run(group);
    }

    fn run(&mut self, group: Vec<UserOperation>) {
        if group.is_empty() {
            return;
        }
        if let Err(err) = self.writer.run(group) {
            error!("tantivy: failed to add documents: {err}");
        }
    }

    /// Hands the rows of a build held so far to `held`, as rows written as they came: the
    /// build's size is not to be had, or the build ended before it was.
    fn write_plain(&mut self) {
        let Some(stretches) = &mut self.stretches else {
            return;
        };
        let placed = stretches.placed;
        while stretches.rows > 0 {
            for (key, primary_id, _, mut doc, guard) in stretches.take(true) {
                doc.add_u64(placed, 0);
                self.held_at
                    .entry(primary_id)
                    .or_default()
                    .push(self.held.len());
                self.held.push(Some((key, doc)));
                self.uncommitted_docs_in_progress_guards.push(guard);
            }
        }
    }

    /// The commit of a tick. When no row of a build came since the last one, the build has
    /// ended or stalled, and the rows held by stretch are all written: a stretch a commit, so
    /// that no segment holds two.
    fn commit_on_tick(&mut self) -> tantivy::Result<(Opstamp, Vec<AsyncInProgress>)> {
        let quiet = self
            .stretches
            .as_mut()
            .is_some_and(|stretches| !std::mem::take(&mut stretches.arrived));
        if quiet {
            if self
                .stretches
                .as_ref()
                .is_some_and(|stretches| !stretches.cut)
            {
                self.write_plain();
            }
            self.write_held();
            while self.stretches.as_ref().is_some_and(|held| held.rows > 0) {
                self.write_stretch(true);
                self.writer.commit()?;
            }
        }
        self.commit()
    }

    /// Commits the documents to the index and returns the commit opstamp with their in-progress
    /// guards.
    fn commit(&mut self) -> tantivy::Result<(Opstamp, Vec<AsyncInProgress>)> {
        self.write_held();
        // A stretch a commit: two written in one would share the threads' segments.
        let mut opstamp = self.writer.commit()?;
        while self.stretches.as_ref().is_some_and(Stretches::is_full) {
            self.write_stretch(false);
            opstamp = self.writer.commit()?;
        }
        if let Some(stretches) = &mut self.stretches {
            stretches.since_commit = 0;
        }
        Ok((
            opstamp,
            mem::take(&mut self.uncommitted_docs_in_progress_guards),
        ))
    }

    fn uncommitted_docs(&self) -> usize {
        self.uncommitted_docs_in_progress_guards.len()
            + self
                .stretches
                .as_ref()
                .map_or(0, |stretches| stretches.since_commit)
    }

    /// Whether a build's rows are held by stretch of the order.
    fn holds_by_stretch(&self) -> bool {
        self.stretches
            .as_ref()
            .is_some_and(|stretches| stretches.cut)
    }

    /// Re-adds `docs` under fresh opstamps after deleting whatever carried their terms: the delete
    /// applies to documents added before it, so the new copy survives. For moving rows between
    /// segments; nothing in progress is attached, since the rows were already indexed.
    pub(crate) fn rewrite_documents(
        &mut self,
        docs: impl IntoIterator<Item = (tantivy::Term, TantivyDocument)>,
    ) -> tantivy::Result<usize> {
        // A held document of one of these rows would be written after its copy here.
        self.write_plain();
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
            || self
                .stretches
                .as_ref()
                .is_some_and(|stretches| stretches.rows > 0)
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
    /// Whether the last write of the index was a row of a full scan: its reloads are then the
    /// build's too, and wait as its rows and commits do.
    building: std::sync::atomic::AtomicBool,
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
    ///
    /// The reload of an index being built is background work, as the build's rows and commits
    /// are: nobody searches that index, and as indexing it went before the searches of every
    /// other one, ten times a second for each index being built.
    async fn reload_on_worker(self: &Arc<Self>) {
        let (done_tx, done_rx) = oneshot::channel();
        let inner = Arc::clone(self);
        let building = self.building.load(Ordering::Relaxed);
        // BUILD_RELOAD=indexing, a benchmark test's switch: as before.
        #[cfg(test)]
        let building = building && !RELOAD_INDEXING.load(Ordering::Relaxed);
        self.worker
            .spawn_write_blocking(building, move || {
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
            building: std::sync::atomic::AtomicBool::new(false),
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

    /// Says whether the index's last write was a row of a full scan: see `ReaderInner::building`.
    fn set_building(&self, building: bool) {
        self.inner.building.store(building, Ordering::Relaxed);
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
/// A change to an index, as it waits for the writer: see `Index::change`.
enum Change {
    Add {
        primary_id: PrimaryId,
        commit_key: Option<CommitKey>,
        doc: TantivyDocument,
        in_progress: AsyncInProgress,
        /// The uncommitted documents at which the change commits.
        commit_threshold: usize,
        /// What a row of a build brings, to be held by stretch of the order.
        build: Option<BuildRow>,
    },
    Remove {
        primary_id: PrimaryId,
        in_progress: AsyncInProgress,
        commit_threshold: usize,
    },
    Commit,
}

/// The most changes that wait for a held writer in an index's queue. One more waits for the
/// writer on its worker instead, which slows the reader that sends them: a holder of seconds
/// would otherwise let the queue grow with everything written meanwhile.
const CHANGES_WAITING_MAX: usize = 65_536;

/// The writer of an index, held. As it is let go, it applies the changes that came while it
/// was held: see `Index::change`.
pub(crate) struct WriterGuard<'a, E> {
    index: &'a Index<E>,
    held: Option<RwLockWriteGuard<'a, Writer>>,
}

impl<E> Deref for WriterGuard<'_, E> {
    type Target = Writer;

    fn deref(&self) -> &Writer {
        self.held.as_ref().unwrap()
    }
}

impl<E> DerefMut for WriterGuard<'_, E> {
    fn deref_mut(&mut self) -> &mut Writer {
        self.held.as_mut().unwrap()
    }
}

impl<E> Drop for WriterGuard<'_, E> {
    fn drop(&mut self) {
        if let Some(mut writer) = self.held.take() {
            self.index.apply_changes(&mut writer);
        }
        // One queued between the last look at the queue and the writer let go.
        self.index.apply_changes_waiting();
    }
}

pub(crate) struct Index<E = ()> {
    pub(crate) index: tantivy::Index,
    writer: RwLock<Writer>,
    /// The changes that found the writer held, in the order they came: see `change`.
    changes: Mutex<VecDeque<Change>>,
    /// The changes waiting for the writer on their workers now, the queue being full: see
    /// `let_writes_in`.
    writes_waiting: AtomicU64,
    /// Whether a build's rows are held by stretch of the order: see `Stretches`.
    pub(crate) held_by_stretch: std::sync::atomic::AtomicBool,
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
            changes: Mutex::new(VecDeque::new()),
            writes_waiting: AtomicU64::new(0),
            held_by_stretch: std::sync::atomic::AtomicBool::new(false),
            reader,
            schema,
            state,
        })
    }

    /// Replaces Tantivy's default merge policy.
    pub(crate) fn set_merge_policy(&self, policy: Box<dyn MergePolicy>) {
        self.lock_writer().writer.set_merge_policy(policy);
    }

    /// Whether the writer holds uncommitted documents, or None while someone holds the writer.
    /// Never waits for the writer: the actors ask on their ticks, and a writer held for seconds --
    /// a pattern index moving rows between segments -- would stop an actor answering anything.
    pub(crate) fn has_uncommitted_docs(&self) -> Option<bool> {
        // A change left in the queue is uncommitted, and the commit asked for it applies it.
        if !self.changes.lock().unwrap().is_empty() {
            return Some(true);
        }
        let writer = self.writer.try_read().ok()?;
        Some(writer.has_uncommitted_docs())
    }

    /// The writer, locked, for a caller that has to read the index and write what it read back
    /// with nothing written in between, as moving rows between segments does. The changes that
    /// come meanwhile are applied as the caller lets it go.
    pub(crate) fn lock_writer(&self) -> WriterGuard<'_, E> {
        WriterGuard {
            index: self,
            held: Some(self.writer.write().unwrap()),
        }
    }

    /// Applies `change` after those that came before it, without waiting for the writer.
    ///
    /// A change runs on a worker, and a writer may be held for most of a second: by a repair
    /// of a pattern index moving rows, by a commit for tens of milliseconds. Waiting for it
    /// there, the changes of one index took every worker in turn, and the searches of every
    /// index waited for the hold to end (p99 275 ms beside a repair). So a change goes into the
    /// index's queue, and whoever has the writer applies the queue in order: this change's
    /// worker if the writer is free, else the holder as it lets go (`WriterGuard`). Only with
    /// `CHANGES_WAITING_MAX` changes waiting does a change wait for the writer on its worker.
    fn change(&self, change: Change) {
        let full = {
            let mut changes = self.changes.lock().unwrap();
            changes.push_back(change);
            changes.len() > CHANGES_WAITING_MAX
        };
        if full {
            self.writes_waiting.fetch_add(1, Ordering::Relaxed);
            let writer = self.lock_writer();
            self.writes_waiting.fetch_sub(1, Ordering::Relaxed);
            drop(writer);
        } else {
            self.apply_changes_waiting();
        }
    }

    /// Applies the changes in the queue if the writer is free; if it is held, its holder does.
    fn apply_changes_waiting(&self) {
        while !self.changes.lock().unwrap().is_empty() {
            let mut writer = match self.writer.try_write() {
                Ok(writer) => writer,
                Err(TryLockError::WouldBlock) => return,
                Err(TryLockError::Poisoned(err)) => panic!("tantivy: writer lock poisoned: {err}"),
            };
            self.apply_changes(&mut writer);
        }
    }

    /// Applies the changes in the queue to `writer`, in the order they came.
    fn apply_changes(&self, writer: &mut Writer) {
        loop {
            // The queue is not held while a change is applied: a commit takes long.
            let Some(change) = self.changes.lock().unwrap().pop_front() else {
                return;
            };
            let tick = matches!(change, Change::Commit);
            let commit = match change {
                Change::Add {
                    primary_id,
                    commit_key,
                    doc,
                    in_progress,
                    commit_threshold,
                    build,
                } => {
                    let pending = writer
                        .add_document(primary_id, commit_key, doc, in_progress, build)
                        .unwrap_or_else(|err| {
                            error!("tantivy: failed to add document {primary_id:?}: {err}");
                            writer.uncommitted_docs()
                        });
                    if build.is_some() {
                        self.held_by_stretch
                            .store(writer.holds_by_stretch(), Ordering::Relaxed);
                    }
                    pending >= commit_threshold
                }
                Change::Remove {
                    primary_id,
                    in_progress,
                    commit_threshold,
                } => {
                    let term = create_term(&self.schema, primary_id);
                    writer.rm_document(primary_id, term, in_progress) >= commit_threshold
                }
                Change::Commit => writer.has_uncommitted_docs(),
            };
            if commit {
                match off_the_runtime(|| {
                    if tick {
                        writer.commit_on_tick()
                    } else {
                        writer.commit()
                    }
                }) {
                    Ok((commit_opstamp, guards)) => self
                        .reader
                        .drop_guards_when_searchable(commit_opstamp, guards),
                    Err(err) => error!(
                        "tantivy: failed to commit for {}: {err}",
                        self.reader.inner.key
                    ),
                }
            }
        }
    }

    /// Waits, for `limit` at most, until no change waits for the writer on its worker.
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

/// Runs `f`, which takes long, so that the runtime thread it may be on keeps nothing waiting.
///
/// A worker of the pool is a task of the runtime and runs its work on a runtime thread. A
/// commit takes tens of milliseconds, and for that long whatever the thread had just woken --
/// the request waiting for the search the worker finished before -- sat in the thread's own
/// slot, which no other thread takes from: a search that took 69 ms beside a commit of 69 ms.
/// `block_in_place` hands the thread's tasks to another before `f` runs. Outside a
/// multi-threaded runtime (the pool's own thread, a repair's) there is nothing to hand over.
fn off_the_runtime<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(test)]
    if OFF_THE_RUNTIME_OFF.load(Ordering::Relaxed) {
        return f();
    }
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(f)
        }
        _ => f(),
    }
}

/// A benchmark test's switch: the tick's commit of an index being built as indexing.
#[cfg(test)]
pub(crate) static TICK_COMMIT_INDEXING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// A benchmark test's switch: the reloads of an index being built as indexing.
#[cfg(test)]
pub(crate) static RELOAD_INDEXING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// A benchmark test's switch: the threads a writer indexes with, when not zero.
#[cfg(test)]
pub(crate) static WRITER_THREADS: AtomicU64 = AtomicU64::new(0);

/// A benchmark test's switch: commits on the runtime thread their worker is on.
#[cfg(test)]
pub(crate) static OFF_THE_RUNTIME_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

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

fn create_term(schema: &Schema, primary_id: PrimaryId) -> tantivy::Term {
    let primary_id_field = schema.get_field("primary_id").unwrap();
    tantivy::Term::from_field_u64(primary_id_field, u64::from(primary_id))
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
    /// Whether the last write was a row of a full scan: the commits of the tick are then the
    /// build's too, and wait as its rows do. See `commit_or_reload`.
    building: bool,
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
            building: false,
            new_index,
        }
    }

    pub(crate) async fn add_document(
        &mut self,
        primary_id: PrimaryId,
        document: String,
        in_progress: AsyncInProgress,
    ) {
        self.add_document_with(primary_id, in_progress, None, None, move |index| {
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
        build: Option<BuildRow>,
        make_doc: impl FnOnce(&Index<E>) -> TantivyDocument + Send + 'static,
    ) {
        let Some(index) = self.get_or_create() else {
            return;
        };
        if !self.can_allocate_memory() {
            return;
        }
        let commit_threshold = self.commit_threshold;
        // A row of a full scan builds an index nobody searches yet: it waits for the searches
        // and the live writes of the other indexes.
        let background = in_progress.is_fullscan();
        self.building = background;
        index.reader.set_building(background);
        self.worker
            .spawn_write_blocking(background, move || {
                let doc = make_doc(&index);
                index.change(Change::Add {
                    primary_id,
                    commit_key,
                    doc,
                    in_progress,
                    commit_threshold,
                    build,
                });
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
        let commit_threshold = self.commit_threshold;
        // In the class of the add that follows it, so that the two run in the order sent.
        let background = in_progress.is_fullscan();
        self.building = background;
        index.reader.set_building(background);
        self.worker
            .spawn_write_blocking(background, move || {
                index.change(Change::Remove {
                    primary_id,
                    in_progress,
                    commit_threshold,
                });
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
                // BUILD_TICK=indexing, a benchmark test's switch: as before.
                #[cfg(test)]
                let building = self.building && !TICK_COMMIT_INDEXING.load(Ordering::Relaxed);
                #[cfg(not(test))]
                let building = self.building;
                self.worker
                    .spawn_write_blocking(building, move || index.change(Change::Commit))
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
                .add_document(1.into(), key(30), doc(1, "one"), guard(), None)
                .unwrap();
            writer
                .add_document(2.into(), key(10), doc(2, "gone"), guard(), None)
                .unwrap();
            writer
                .add_document(3.into(), key(20), doc(3, "old"), guard(), None)
                .unwrap();
            writer
                .add_document(4.into(), key(5), doc(4, "four"), guard(), None)
                .unwrap();
            writer.rm_document(2.into(), term(2), guard());
            writer.rm_document(3.into(), term(3), guard());
            writer
                .add_document(3.into(), key(20), doc(3, "new"), guard(), None)
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

    /// A change that finds the writer held does not wait for it: the holder applies it as it
    /// lets the writer go, after the changes that came before.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_change_that_finds_the_writer_held_is_applied_by_its_holder() {
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
        let add = |id: u64, text: &str| Change::Add {
            primary_id: id.into(),
            commit_key: None,
            doc: create_doc(&index.schema, id.into(), text),
            in_progress: AsyncInProgress::none(),
            commit_threshold: 4,
            build: None,
        };
        let held = index.lock_writer();
        index.change(add(1, "old"));
        index.change(Change::Remove {
            primary_id: 1.into(),
            in_progress: AsyncInProgress::none(),
            commit_threshold: 4,
        });
        index.change(add(1, "new"));
        assert_eq!(held.uncommitted_docs(), 0);
        assert_eq!(index.has_uncommitted_docs(), Some(true));
        drop(held);
        assert_eq!(index.lock_writer().uncommitted_docs(), 3);
        // With the writer free a change is applied at once, and the fourth commits.
        index.change(add(2, "two"));
        assert_eq!(index.lock_writer().uncommitted_docs(), 0);
        index.reader.inner.reload().unwrap();
        let searcher = index.reader.searcher();
        assert_eq!(searcher.num_docs(), 2);
    }

    fn random(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    /// The tokens a scan has read so far -- the lowest of some ranges of the ring -- say how
    /// many rows the table has; the rows of a few wide partitions do not.
    #[test]
    fn the_rows_of_a_table_are_estimated_from_the_tokens_a_scan_has_read() {
        let mut seed = 0x9e37_79b9_7f4a_7c15;
        let mut table: Vec<u64> = (0..1_000_000).map(|_| random(&mut seed)).collect();
        table.sort_unstable();
        // 128 ranges of as many rows, 36 of them being read, 500 rows into each.
        let mut read: Vec<u64> = table
            .chunks(table.len() / 128)
            .take(36)
            .flat_map(|range| range[..500].to_vec())
            .collect();
        let rows = estimate_rows(&mut read).unwrap();
        assert!((850_000..1_150_000).contains(&rows), "{rows}");

        // Two rows a partition, in a table of 200,000 partitions.
        let mut table: Vec<u64> = (0..200_000).map(|_| random(&mut seed)).collect();
        table.sort_unstable();
        let mut read: Vec<u64> = table
            .chunks(table.len() / 128)
            .take(36)
            .flat_map(|range| range[..250].iter().flat_map(|&token| [token; 2]))
            .collect();
        let rows = estimate_rows(&mut read).unwrap();
        assert!((340_000..460_000).contains(&rows), "{rows}");

        // A hundred rows a partition come together, and lie together: nothing to cut.
        let mut read: Vec<u64> = table[..200]
            .iter()
            .flat_map(|&token| [token; 100])
            .collect();
        assert_eq!(estimate_rows(&mut read), None);
    }

    /// The rows of a build, which come in no order, are held and written a stretch of the
    /// order at a time: every segment holds the rows of one stretch, marked as placed. A row
    /// keeps its guard until it is written, and one removed while held is not written.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_rows_of_a_build_are_written_a_stretch_of_the_order_at_a_time() {
        const ROWS: u64 = 60_000;
        let mut schema = Schema::builder();
        schema.add_u64_field("primary_id", INDEXED | STORED | FAST);
        schema.add_text_field("text", TEXT | STORED);
        schema.add_u64_field("key", FAST);
        schema.add_u64_field("placed", FAST);
        let index = Index::new(
            schema.build(),
            "default",
            TextAnalyzer::from(SimpleTokenizer::default()),
            &make_index_key(),
            crate::worker::new(),
        )
        .unwrap();
        // No merges: the segments are looked at as they were written.
        index.set_merge_policy(Box::new(tantivy::indexer::NoMergePolicy));
        let key_field = index.schema.get_field("key").unwrap();
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let mut seed = 0x2545_f491_4f6c_dd1d;
        // A scan of one range over the whole ring: the tokens going up.
        let mut tokens: Vec<u64> = (0..ROWS).map(|_| random(&mut seed)).collect();
        tokens.sort_unstable();
        let mut writer = index.lock_writer();
        for id in 1..=ROWS {
            let key = random(&mut seed);
            let mut doc = create_doc(&index.schema, id.into(), "row");
            doc.add_u64(key_field, key);
            let build = BuildRow {
                token: tokens[id as usize - 1],
                stretch_rows: 6000,
                placed: "placed",
            };
            let guard = AsyncInProgress::fullscan(tx.clone());
            let uncommitted = writer
                .add_document(
                    id.into(),
                    Some(CommitKey::from([key])),
                    doc,
                    guard,
                    Some(build),
                )
                .unwrap();
            if id == 5 || id == 30_000 {
                let guard = AsyncInProgress::fullscan(tx.clone());
                writer.rm_document(id.into(), create_term(&index.schema, id.into()), guard);
            }
            if uncommitted >= 10_000 {
                index.commit_and_reload(&mut writer).unwrap();
                // The guards left are of the rows not written yet.
                let searchable = index.reader.searcher().num_docs();
                let removed = u64::from(id >= 5) + u64::from(id >= 30_000);
                assert_eq!(tx.strong_count() as u64 - 1, id - removed - searchable);
            }
        }
        assert!(writer.holds_by_stretch());
        let searchable = index.reader.searcher().num_docs();
        assert!((1..ROWS - 2).contains(&searchable), "{searchable}");
        // A tick after the last row writes them all.
        let (opstamp, guards) = writer.commit_on_tick().unwrap();
        index.reader.drop_guards_when_searchable(opstamp, guards);
        let (opstamp, guards) = writer.commit_on_tick().unwrap();
        index.reader.drop_guards_when_searchable(opstamp, guards);
        index.reader.inner.reload().unwrap();
        drop(writer);
        assert_eq!(tx.strong_count(), 1);

        let searcher = index.reader.searcher();
        assert_eq!(searcher.num_docs(), ROWS - 2);
        // The segments' spans, lowest first, fall into ten stretches that do not overlap.
        let mut spans: Vec<(u64, u64, u64)> = searcher
            .segment_readers()
            .iter()
            .map(|segment| {
                let placed = segment.fast_fields().u64("placed").unwrap();
                assert_eq!(placed.min_value(), 1);
                let keys = segment.fast_fields().u64("key").unwrap();
                (
                    keys.min_value(),
                    keys.max_value(),
                    u64::from(segment.num_docs()),
                )
            })
            .collect();
        spans.sort_unstable();
        let mut stretches: Vec<(u64, u64)> = Vec::new();
        for (min, max, rows) in spans {
            match stretches.last_mut() {
                Some((reach, held)) if min <= *reach => {
                    *reach = (*reach).max(max);
                    *held += rows;
                }
                _ => stretches.push((max, rows)),
            }
        }
        assert_eq!(stretches.len(), 10, "{stretches:?}");
        for (_, rows) in stretches {
            assert!((4500..7500).contains(&rows), "{rows}");
        }
    }
}
