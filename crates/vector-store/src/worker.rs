/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! The pool of runtime workers that runs index work: searches, and the writes that keep the
//! indexes up to date.
//!
//! The two compete for the same cores, and a single queue let searches win: under an unthrottled
//! query load every row a CDC reader handed over waited behind the searches queued before it, and
//! on a 4-vCPU AWS node (run 5df052ac, 2026-11-03) the write-to-searchable lag of a pattern index
//! grew from about a second to 19-31 s with nothing wrong with the index. So work comes in two
//! classes, [`TaskClass::Indexing`] and [`TaskClass::Search`], each with its own queue, and a free
//! worker picks by [`Scheduling`]: indexing first once its oldest task has waited past a budget
//! (zero by default -- the lag is held, and searches get what indexing leaves), but never so much
//! that searches waiting get less than a minimum share of the workers' time. A short burst of
//! writes then slows the searches and lets the lag grow for a while rather than stopping either;
//! a load that keeps both queues waiting is one the node is too small for, which the queue waits
//! in [`STATS`] show.
//!
//! A third class, [`TaskClass::Background`], is for the writes nobody waits for: the rows of a
//! full scan, building an index that does not serve yet. As indexing, they would take the
//! workers from the searches of every other index for as long as the build lasts. They run when
//! neither of the other classes waits, and otherwise get a small minimum share of the workers'
//! time, so that a build under a steady load still ends. At most half of the workers run them at
//! once ([`background_cap`]): they all take their index's writer, and would otherwise all sit
//! behind one commit. The pool takes them into their queue
//! at once and tells the sender when the queue has room
//! ([`WorkerExt::spawn_background_blocking`]): a full scan then waits for that, at the pace of
//! its share, without holding up what the other classes send through the same channel.

use crate::perf;
use async_channel::Sender;
use futures::future::BoxFuture;
use std::collections::VecDeque;
use std::future::Future;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;
use std::time::Instant;
use tokio::runtime::Builder;
use tokio::sync::Notify;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tracing::Instrument;
use tracing::debug;
use tracing::error_span;

/// A lambda creating a future, instead of the future itself. The future has to be
/// created by the runtime which polls it.
type NewTaskFuture = Box<dyn FnOnce() -> BoxFuture<'static, ()> + Send>;

fn new_task_future<Fut>(f: impl FnOnce() -> Fut + Send + 'static) -> NewTaskFuture
where
    Fut: Future<Output = ()> + Send + 'static,
{
    Box::new(move || -> BoxFuture<'static, ()> { Box::pin(f()) })
}

/// Which queue a task waits in; see the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TaskClass {
    /// Everything not marked otherwise: searches, and the vector indexes' own work.
    Search,
    /// The writes that make rows searchable: a row added or removed, a commit.
    Indexing,
    /// Writes nobody waits for: the rows of a full scan, building an index.
    Background,
}

/// The number of task classes.
const CLASSES: usize = 3;

impl TaskClass {
    const ALL: [TaskClass; CLASSES] = [
        TaskClass::Search,
        TaskClass::Indexing,
        TaskClass::Background,
    ];

    fn index(self) -> usize {
        self as usize
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            TaskClass::Search => "search",
            TaskClass::Indexing => "indexing",
            TaskClass::Background => "background",
        }
    }
}

#[allow(clippy::enum_variant_names)]
pub(crate) enum Worker {
    SpawnNonBlocking {
        f: Box<dyn FnOnce() + Send>,
    },
    SpawnBlocking {
        f: Box<dyn FnOnce() + Send>,
        class: TaskClass,
        /// For a background task: told once the pool has room for the task, which its sender
        /// waits for.
        admitted: Option<oneshot::Sender<()>>,
    },
    SpawnAsyncNonBlocking {
        f: NewTaskFuture,
    },
    SpawnAsyncBlocking {
        f: NewTaskFuture,
    },
}

impl Worker {
    fn class(&self) -> TaskClass {
        match self {
            Worker::SpawnBlocking { class, .. } => *class,
            _ => TaskClass::Search,
        }
    }
}

pub(crate) trait WorkerExt {
    async fn spawn_non_blocking(&self, f: impl FnOnce() + Send + 'static);
    async fn spawn_blocking(&self, f: impl FnOnce() + Send + 'static);
    /// As `spawn_blocking`, for work that makes rows searchable: see [`TaskClass::Indexing`].
    async fn spawn_indexing_blocking(&self, f: impl FnOnce() + Send + 'static);
    /// As `spawn_blocking`, for a write nobody waits for: see [`TaskClass::Background`]. Returns
    /// once the pool has room for it, which under load is at the pace of the class's share.
    async fn spawn_background_blocking(&self, f: impl FnOnce() + Send + 'static);
    /// A write of an index: `spawn_background_blocking` for one of a full scan,
    /// `spawn_indexing_blocking` otherwise.
    async fn spawn_write_blocking(&self, background: bool, f: impl FnOnce() + Send + 'static) {
        if background {
            self.spawn_background_blocking(f).await
        } else {
            self.spawn_indexing_blocking(f).await
        }
    }
    async fn spawn_async_non_blocking<Fut>(&self, f: impl FnOnce() -> Fut + Send + 'static)
    where
        Fut: Future<Output = ()> + Send + 'static;
    async fn spawn_async_blocking<Fut>(&self, f: impl FnOnce() -> Fut + Send + 'static)
    where
        Fut: Future<Output = ()> + Send + 'static;
}

impl WorkerExt for Sender<Worker> {
    #[hotpath::measure]
    async fn spawn_non_blocking(&self, f: impl FnOnce() + Send + 'static) {
        self.send(Worker::SpawnNonBlocking { f: Box::new(f) })
            .await
            .expect("WorkerExt::spawn_non_blocking: internal actor should receive request");
    }

    #[hotpath::measure]
    async fn spawn_blocking(&self, f: impl FnOnce() + Send + 'static) {
        self.send(Worker::SpawnBlocking {
            f: Box::new(f),
            class: TaskClass::Search,
            admitted: None,
        })
        .await
        .expect("WorkerExt::spawn_blocking: internal actor should receive request");
    }

    #[hotpath::measure]
    async fn spawn_indexing_blocking(&self, f: impl FnOnce() + Send + 'static) {
        self.send(Worker::SpawnBlocking {
            f: Box::new(f),
            class: TaskClass::Indexing,
            admitted: None,
        })
        .await
        .expect("WorkerExt::spawn_indexing_blocking: internal actor should receive request");
    }

    #[hotpath::measure]
    async fn spawn_background_blocking(&self, f: impl FnOnce() + Send + 'static) {
        // The pool takes the task at once, so that the channel every class shares keeps moving,
        // and answers when its background queue has room: a full scan waits here, not there.
        let (admitted_tx, admitted_rx) = oneshot::channel();
        self.send(Worker::SpawnBlocking {
            f: Box::new(f),
            class: TaskClass::Background,
            admitted: Some(admitted_tx),
        })
        .await
        .expect("WorkerExt::spawn_background_blocking: internal actor should receive request");
        // An error is the pool gone, with the task: nothing to wait for.
        _ = admitted_rx.await;
    }

    #[hotpath::measure]
    async fn spawn_async_non_blocking<Fut>(&self, f: impl FnOnce() -> Fut + Send + 'static)
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.send(Worker::SpawnAsyncNonBlocking {
            f: new_task_future(f),
        })
        .await
        .expect("WorkerExt::spawn_async_non_blocking: internal actor should receive request");
    }

    #[hotpath::measure]
    async fn spawn_async_blocking<Fut>(&self, f: impl FnOnce() -> Fut + Send + 'static)
    where
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.send(Worker::SpawnAsyncBlocking {
            f: new_task_future(f),
        })
        .await
        .expect("WorkerExt::spawn_async_blocking: internal actor should receive request");
    }
}

/// The most background tasks that run at once: half of the workers. The writes of a build all
/// take their index's writer, one at a time, and a commit holds it for tens of milliseconds;
/// uncapped, every worker picked one in a moment with no search waiting and sat behind the
/// commit, and the searches of the other indexes that came next waited for it to end.
fn background_cap(workers: usize) -> usize {
    (workers / 2).max(1)
}

/// How a free worker chooses between the queues when more than one has work.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Scheduling {
    /// How long the oldest indexing task may wait while searches are served first. Zero: indexing
    /// always first (`VECTOR_STORE_INDEXING_QUEUE_BUDGET`).
    pub(crate) indexing_queue_budget: Duration,
    /// The share of the workers' recent time, 0.0 to 1.0, that searches waiting get however long
    /// indexing has waited (`VECTOR_STORE_MIN_SEARCH_SHARE`, in percent).
    pub(crate) min_search_share: f64,
    /// The share of the workers' recent time, 0.0 to 1.0, that background tasks waiting get while
    /// the other classes wait too (`VECTOR_STORE_MIN_BACKGROUND_SHARE`, in percent). Zero: only
    /// when nothing else waits.
    pub(crate) min_background_share: f64,
}

impl Default for Scheduling {
    fn default() -> Self {
        Self {
            indexing_queue_budget: Duration::ZERO,
            min_search_share: DEFAULT_MIN_SEARCH_SHARE,
            min_background_share: DEFAULT_MIN_BACKGROUND_SHARE,
        }
    }
}

pub(crate) const DEFAULT_MIN_SEARCH_SHARE: f64 = 0.25;
pub(crate) const DEFAULT_MIN_BACKGROUND_SHARE: f64 = 0.10;

/// What waits in the queues: whether a search does, how long the oldest indexing task has, and
/// whether a background task does.
#[derive(Clone, Copy, Debug, Default)]
struct Waiting {
    search: bool,
    oldest_indexing: Option<Duration>,
    background: bool,
}

/// The classes' shares of the workers' recent time.
#[derive(Clone, Copy, Debug)]
struct Shares {
    search: f64,
    background: f64,
}

/// The class the next task comes from, given what waits: `None` when nothing does. Pure, so
/// that the policy can be tested without a runtime.
fn pick_class(waiting: Waiting, shares: Shares, scheduling: &Scheduling) -> Option<TaskClass> {
    let searches_starve = waiting.search && shares.search < scheduling.min_search_share;
    let foreground = match (waiting.search, waiting.oldest_indexing) {
        (false, None) => None,
        (true, None) => Some(TaskClass::Search),
        (false, Some(_)) => Some(TaskClass::Indexing),
        (true, Some(waited)) => {
            if waited < scheduling.indexing_queue_budget || searches_starve {
                Some(TaskClass::Search)
            } else {
                Some(TaskClass::Indexing)
            }
        }
    };
    match foreground {
        None => waiting.background.then_some(TaskClass::Background),
        // Below its share a background task goes first, unless searches are below theirs: the
        // two floors together leave indexing the most of the pool, and searches come first.
        Some(_)
            if waiting.background
                && shares.background < scheduling.min_background_share
                && !searches_starve =>
        {
            Some(TaskClass::Background)
        }
        foreground => foreground,
    }
}

/// How fast past work stops counting towards a class's share: its weight halves every
/// `SHARE_HALF_LIFE`, so the share follows the last few seconds.
const SHARE_HALF_LIFE: Duration = Duration::from_secs(1);

/// The workers' recent time per class, decaying with `SHARE_HALF_LIFE`.
struct RecentBusy {
    seconds: [f64; CLASSES],
    at: Instant,
}

impl RecentBusy {
    fn new(now: Instant) -> Self {
        Self {
            seconds: [0.0; CLASSES],
            at: now,
        }
    }

    fn decay(&mut self, now: Instant) {
        let elapsed = now.saturating_duration_since(self.at).as_secs_f64();
        let factor = 0.5_f64.powf(elapsed / SHARE_HALF_LIFE.as_secs_f64());
        self.seconds
            .iter_mut()
            .for_each(|seconds| *seconds *= factor);
        self.at = now;
    }

    fn add(&mut self, class: TaskClass, busy: Duration, now: Instant) {
        self.decay(now);
        self.seconds[class.index()] += busy.as_secs_f64();
    }

    /// A class's share of the recent time; 1.0 when nothing ran, so that an idle pool does not
    /// read as one starving the class.
    fn share(&mut self, class: TaskClass, now: Instant) -> f64 {
        self.decay(now);
        let total: f64 = self.seconds.iter().sum();
        if total <= 0.0 {
            1.0
        } else {
            self.seconds[class.index()] / total
        }
    }

    fn shares(&mut self, now: Instant) -> Shares {
        Shares {
            search: self.share(TaskClass::Search, now),
            background: self.share(TaskClass::Background, now),
        }
    }
}

struct Queued {
    msg: Worker,
    enqueued: Instant,
}

#[cfg(test)]
pub(crate) static FIFO: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

struct Queues {
    by_class: [VecDeque<Queued>; CLASSES],
    /// The senders of the background tasks queued past the queue's room, oldest first: each is
    /// told as a task is taken. There are at most as many as there are senders.
    not_admitted: VecDeque<oneshot::Sender<()>>,
    /// Background tasks running now, and the most that may: see [`background_cap`].
    background_running: usize,
    background_cap: usize,
    recent: RecentBusy,
    closed: bool,
}

impl Queues {
    fn take(&mut self, scheduling: &Scheduling, now: Instant) -> Option<(Queued, TaskClass)> {
        let waiting = Waiting {
            search: !self.by_class[TaskClass::Search.index()].is_empty(),
            oldest_indexing: self.by_class[TaskClass::Indexing.index()]
                .front()
                .map(|queued| now.saturating_duration_since(queued.enqueued)),
            background: !self.by_class[TaskClass::Background.index()].is_empty()
                && self.background_running < self.background_cap,
        };
        let class = pick_class(waiting, self.recent.shares(now), scheduling)?;
        // A benchmark test's switch: one queue, first come first served, as without classes.
        #[cfg(test)]
        let class = if FIFO.load(Ordering::Relaxed) {
            TaskClass::ALL
                .into_iter()
                .filter_map(|class| Some((self.by_class[class.index()].front()?.enqueued, class)))
                .min_by_key(|&(enqueued, _)| enqueued)?
                .1
        } else {
            class
        };
        let queued = self.by_class[class.index()].pop_front()?;
        STATS.queued[class.index()].fetch_sub(1, Ordering::Relaxed);
        if class == TaskClass::Background {
            self.background_running += 1;
            if let Some(admitted) = self.not_admitted.pop_front() {
                _ = admitted.send(());
            }
        }
        Some((queued, class))
    }

    fn is_empty(&self) -> bool {
        self.by_class.iter().all(VecDeque::is_empty)
    }
}

/// The pool's queues, shared by the task that fills them and the workers that empty them.
struct Shared {
    queues: Mutex<Queues>,
    /// Signalled when a task is queued or the input channel closes.
    work: Notify,
    /// Signalled when an indexing task is taken, so that a full indexing queue takes more.
    room: Notify,
    scheduling: Scheduling,
}

/// Totals since the process started, over every pool, for `/metrics`: per class, the tasks run,
/// their time in the queue, and their time running.
pub(crate) struct WorkerStats {
    tasks: [AtomicU64; CLASSES],
    queue_wait_nanos: [AtomicU64; CLASSES],
    busy_nanos: [AtomicU64; CLASSES],
    /// Tasks in the queues now.
    queued: [AtomicU64; CLASSES],
}

pub(crate) static STATS: WorkerStats = WorkerStats {
    tasks: [const { AtomicU64::new(0) }; CLASSES],
    queue_wait_nanos: [const { AtomicU64::new(0) }; CLASSES],
    busy_nanos: [const { AtomicU64::new(0) }; CLASSES],
    queued: [const { AtomicU64::new(0) }; CLASSES],
};

/// How the indexing class fares at one moment, over every pool: what background work that
/// competes with indexing for the node -- the repair of a pattern index's layout -- reads to hold
/// back while indexing waits. Totals since the process started, so a caller diffs two readings.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexingLoad {
    /// Indexing tasks in the queues now.
    pub(crate) queued: u64,
    pub(crate) tasks: u64,
    pub(crate) queue_wait_nanos: u64,
}

impl IndexingLoad {
    /// Whether indexing waited since `before`: tasks are queued now, or the tasks taken since
    /// waited more than `limit` on average.
    pub(crate) fn contended_since(&self, before: &IndexingLoad, limit: Duration) -> bool {
        let tasks = self.tasks.saturating_sub(before.tasks);
        let waited = self
            .queue_wait_nanos
            .saturating_sub(before.queue_wait_nanos);
        self.queued > 0 || (tasks > 0 && waited / tasks > limit.as_nanos() as u64)
    }
}

impl WorkerStats {
    fn record(&self, class: TaskClass, queue_wait: Duration, busy: Duration) {
        let class = class.index();
        self.tasks[class].fetch_add(1, Ordering::Relaxed);
        self.queue_wait_nanos[class].fetch_add(queue_wait.as_nanos() as u64, Ordering::Relaxed);
        self.busy_nanos[class].fetch_add(busy.as_nanos() as u64, Ordering::Relaxed);
    }

    pub(crate) fn indexing(&self) -> IndexingLoad {
        let i = TaskClass::Indexing.index();
        IndexingLoad {
            queued: self.queued[i].load(Ordering::Relaxed),
            tasks: self.tasks[i].load(Ordering::Relaxed),
            queue_wait_nanos: self.queue_wait_nanos[i].load(Ordering::Relaxed),
        }
    }

    /// Per class: (class, tasks run, seconds queued, seconds running).
    pub(crate) fn snapshot(&self) -> [(TaskClass, u64, f64, f64); CLASSES] {
        TaskClass::ALL.map(|class| {
            let i = class.index();
            (
                class,
                self.tasks[i].load(Ordering::Relaxed),
                self.queue_wait_nanos[i].load(Ordering::Relaxed) as f64 / 1e9,
                self.busy_nanos[i].load(Ordering::Relaxed) as f64 / 1e9,
            )
        })
    }
}

fn try_acquire_thread(
    id: usize,
    in_flow: usize,
    workers: usize,
    thread_operations_in_flow: &AtomicUsize,
) -> bool {
    // If all workers are busy, we need to execute the blocking task in a
    // separate thread to avoid starving runtime.
    if in_flow != workers {
        return false;
    }
    let thread_in_flow = thread_operations_in_flow.fetch_add(1, Ordering::Relaxed) + 1;
    hotpath::val!("worker::thread_in_flow").set(&(id, in_flow, thread_in_flow));
    if thread_in_flow == 1 {
        true
    } else {
        thread_operations_in_flow.fetch_sub(1, Ordering::Relaxed);
        false
    }
}

async fn run_in_thread(
    tx_thread: &mpsc::Sender<(NewTaskFuture, oneshot::Sender<()>)>,
    thread_operations_in_flow: &AtomicUsize,
    f: NewTaskFuture,
) {
    let (tx_call, rx_call) = oneshot::channel();
    tx_thread
        .send((f, tx_call))
        .await
        .expect("Worker: internal thread should receive task");
    _ = rx_call.await;
    thread_operations_in_flow.fetch_sub(1, Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn new() -> Sender<Worker> {
    with_scheduling(Scheduling::default())
}

pub(crate) fn with_scheduling(scheduling: Scheduling) -> Sender<Worker> {
    let capacity = perf::channel_size().into();
    let (tx_worker, rx_worker) = async_channel::bounded::<Worker>(capacity);
    let (tx_thread, mut rx_thread) = mpsc::channel::<(NewTaskFuture, oneshot::Sender<()>)>(1);

    // Dedicated thread for long tasks to avoid starving runtime.
    thread::spawn(move || {
        let runtime = Builder::new_current_thread().enable_all().build().unwrap();

        runtime.block_on(async move {
            while let Some((f, tx_call)) = rx_thread.recv().await {
                f().await;
                _ = tx_call.send(());
            }
        });
    });

    let shared = Arc::new(Shared {
        queues: Mutex::new(Queues {
            by_class: std::array::from_fn(|_| VecDeque::new()),
            not_admitted: VecDeque::new(),
            background_running: 0,
            background_cap: background_cap(perf::num_workers().into()),
            recent: RecentBusy::new(Instant::now()),
            closed: false,
        }),
        work: Notify::new(),
        room: Notify::new(),
        scheduling,
    });

    // Moves what the channel brings into the queue of its class. The indexing queue holds at
    // most the channel's capacity, so that a CDC reader faster than indexing still waits on its
    // sends rather than filling memory; searches already wait as their requests' own tasks, so
    // their queue holds whatever is sent; a background task is queued at once, and its sender
    // told when the queue is within the same capacity.
    tokio::spawn({
        let shared = Arc::clone(&shared);
        async move {
            while let Ok(mut msg) = rx_worker.recv().await {
                let class = msg.class();
                let admitted = match &mut msg {
                    Worker::SpawnBlocking { admitted, .. } => admitted.take(),
                    _ => None,
                };
                loop {
                    let room = shared.room.notified();
                    {
                        let mut queues = shared.queues.lock().unwrap();
                        let queue = &mut queues.by_class[class.index()];
                        if class != TaskClass::Indexing || queue.len() < capacity {
                            queue.push_back(Queued {
                                msg,
                                enqueued: Instant::now(),
                            });
                            STATS.queued[class.index()].fetch_add(1, Ordering::Relaxed);
                            if let Some(admitted) = admitted {
                                if queues.by_class[class.index()].len() <= capacity {
                                    _ = admitted.send(());
                                } else {
                                    queues.not_admitted.push_back(admitted);
                                }
                            }
                            break;
                        }
                    }
                    room.await;
                }
                shared.work.notify_one();
            }
            shared.queues.lock().unwrap().closed = true;
            shared.work.notify_waiters();
        }
    });

    let operations_in_flow = Arc::new(AtomicUsize::new(0));
    let thread_operations_in_flow = Arc::new(AtomicUsize::new(0));
    let workers = perf::num_workers().into();
    (0..workers).for_each(|id| {
        let shared = Arc::clone(&shared);
        let tx_thread = tx_thread.clone();
        let operations_in_flow = Arc::clone(&operations_in_flow);
        let thread_operations_in_flow = Arc::clone(&thread_operations_in_flow);
        tokio::spawn(perf::hotpath_async(
            async move {
                debug!("starting");

                loop {
                    let work = shared.work.notified();
                    let next = {
                        let mut queues = shared.queues.lock().unwrap();
                        let next = queues.take(&shared.scheduling, Instant::now());
                        if next.is_none() && queues.closed && queues.is_empty() {
                            break;
                        }
                        // Pass the wake-up on: a notification that found every worker busy
                        // wakes only one when it is consumed.
                        if next.is_some() && !queues.is_empty() {
                            shared.work.notify_one();
                        }
                        next
                    };
                    let Some((queued, class)) = next else {
                        work.await;
                        continue;
                    };
                    if class == TaskClass::Indexing {
                        shared.room.notify_one();
                    }
                    let started = Instant::now();
                    let queue_wait = started.saturating_duration_since(queued.enqueued);

                    let in_flow = operations_in_flow.fetch_add(1, Ordering::Relaxed) + 1;
                    hotpath::val!("worker::in_flow").set(&(id, in_flow));
                    match queued.msg {
                        Worker::SpawnNonBlocking { f } => {
                            f();
                        }
                        Worker::SpawnAsyncNonBlocking { f } => {
                            f().await;
                        }
                        Worker::SpawnBlocking { f, .. } => {
                            if try_acquire_thread(id, in_flow, workers, &thread_operations_in_flow)
                            {
                                run_in_thread(
                                    &tx_thread,
                                    &thread_operations_in_flow,
                                    new_task_future(move || async move { f() }),
                                )
                                .await;
                            } else {
                                f();
                            }
                        }
                        Worker::SpawnAsyncBlocking { f } => {
                            if try_acquire_thread(id, in_flow, workers, &thread_operations_in_flow)
                            {
                                run_in_thread(&tx_thread, &thread_operations_in_flow, f).await;
                            } else {
                                f().await;
                            }
                        }
                    }
                    operations_in_flow.fetch_sub(1, Ordering::Relaxed);

                    let finished = Instant::now();
                    let busy = finished.saturating_duration_since(started);
                    STATS.record(class, queue_wait, busy);
                    let mut queues = shared.queues.lock().unwrap();
                    queues.recent.add(class, busy, finished);
                    if class == TaskClass::Background {
                        queues.background_running -= 1;
                        // A worker that found the class at its cap may be waiting for this.
                        if !queues.by_class[class.index()].is_empty() {
                            shared.work.notify_one();
                        }
                    }
                    drop(queues);
                }

                debug!("finished");
            }
            .instrument(error_span!("worker", id)),
        ));
    });
    tx_worker
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUDGET_100MS: Scheduling = Scheduling {
        indexing_queue_budget: Duration::from_millis(100),
        min_search_share: 0.25,
        min_background_share: 0.10,
    };

    /// Indexing is contended while tasks wait in its queue, or when the tasks taken since the
    /// last reading waited past the limit on average; a reading with nothing taken and nothing
    /// queued is not.
    #[test]
    fn indexing_is_contended_while_its_tasks_wait() {
        let limit = Duration::from_millis(1);
        let before = IndexingLoad {
            queued: 0,
            tasks: 100,
            queue_wait_nanos: 1_000_000,
        };
        assert!(!before.contended_since(&before, limit));
        let quick = IndexingLoad {
            tasks: 200,
            queue_wait_nanos: 1_000_000 + 100 * 50_000,
            ..before
        };
        assert!(!quick.contended_since(&before, limit));
        let slow = IndexingLoad {
            queue_wait_nanos: 1_000_000 + 100 * 2_000_000,
            ..quick
        };
        assert!(slow.contended_since(&before, limit));
        let backlog = IndexingLoad {
            queued: 3,
            ..before
        };
        assert!(backlog.contended_since(&before, limit));
    }

    /// What waits, and the shares to go with it: searches' as given, background's above its
    /// minimum unless a test says otherwise.
    fn pick(
        search: bool,
        oldest_indexing: Option<Duration>,
        search_share: f64,
        scheduling: &Scheduling,
    ) -> Option<TaskClass> {
        pick_class(
            Waiting {
                search,
                oldest_indexing,
                background: false,
            },
            Shares {
                search: search_share,
                background: 1.0,
            },
            scheduling,
        )
    }

    #[test]
    fn an_empty_queue_gives_nothing_and_a_lone_class_goes_first() {
        let scheduling = Scheduling::default();
        assert_eq!(pick(false, None, 1.0, &scheduling), None);
        assert_eq!(pick(true, None, 0.0, &scheduling), Some(TaskClass::Search));
        assert_eq!(
            pick(false, Some(Duration::ZERO), 1.0, &scheduling),
            Some(TaskClass::Indexing)
        );
    }

    /// The default: indexing first whenever both wait, so the lag holds.
    #[test]
    fn by_default_indexing_goes_before_searches() {
        let scheduling = Scheduling::default();
        assert_eq!(
            pick(true, Some(Duration::ZERO), 0.9, &scheduling),
            Some(TaskClass::Indexing)
        );
    }

    #[test]
    fn indexing_within_its_budget_lets_searches_go_first() {
        assert_eq!(
            pick(true, Some(Duration::from_millis(50)), 0.9, &BUDGET_100MS),
            Some(TaskClass::Search)
        );
        assert_eq!(
            pick(true, Some(Duration::from_millis(150)), 0.9, &BUDGET_100MS),
            Some(TaskClass::Indexing)
        );
    }

    /// However long indexing has waited, searches below their minimum share go first: a burst
    /// of writes slows the searches and lets the lag grow rather than stopping the searches.
    #[test]
    fn searches_below_their_minimum_share_go_first_however_long_indexing_waited() {
        let scheduling = Scheduling::default();
        assert_eq!(
            pick(true, Some(Duration::from_secs(60)), 0.1, &scheduling),
            Some(TaskClass::Search)
        );
        assert_eq!(
            pick(true, Some(Duration::from_secs(60)), 0.3, &scheduling),
            Some(TaskClass::Indexing)
        );
    }

    /// A background task runs when nothing else waits; while something does, only below its
    /// minimum share, and never ahead of searches below theirs.
    #[test]
    fn background_runs_when_the_others_do_not_wait_or_below_its_share() {
        let scheduling = Scheduling::default();
        let pick = |search, oldest_indexing, search_share, background_share| {
            pick_class(
                Waiting {
                    search,
                    oldest_indexing,
                    background: true,
                },
                Shares {
                    search: search_share,
                    background: background_share,
                },
                &scheduling,
            )
        };
        let waited = Some(Duration::from_millis(5));
        assert_eq!(pick(false, None, 1.0, 1.0), Some(TaskClass::Background));
        assert_eq!(pick(false, None, 0.0, 0.0), Some(TaskClass::Background));
        // Above its share it waits for both of the others.
        assert_eq!(pick(true, None, 0.8, 0.2), Some(TaskClass::Search));
        assert_eq!(pick(false, waited, 0.8, 0.2), Some(TaskClass::Indexing));
        // Below it, it goes ahead of either.
        assert_eq!(pick(true, None, 0.95, 0.05), Some(TaskClass::Background));
        assert_eq!(pick(false, waited, 0.5, 0.05), Some(TaskClass::Background));
        assert_eq!(pick(true, waited, 0.5, 0.05), Some(TaskClass::Background));
        // Searches below their share come first even then.
        assert_eq!(pick(true, waited, 0.1, 0.05), Some(TaskClass::Search));
        // A share of zero: only when nothing else waits.
        let never = Scheduling {
            min_background_share: 0.0,
            ..scheduling
        };
        let waiting = Waiting {
            search: true,
            oldest_indexing: None,
            background: true,
        };
        let shares = Shares {
            search: 1.0,
            background: 0.0,
        };
        assert_eq!(pick_class(waiting, shares, &never), Some(TaskClass::Search));
    }

    /// Background tasks run at most `background_cap` at once: with that many running, a worker
    /// takes none, and takes the next when one has ended.
    #[test]
    fn background_tasks_running_at_once_are_capped() {
        let now = Instant::now();
        let background = || Queued {
            msg: Worker::SpawnBlocking {
                f: Box::new(|| ()),
                class: TaskClass::Background,
                admitted: None,
            },
            enqueued: now,
        };
        let mut queues = Queues {
            by_class: std::array::from_fn(|_| VecDeque::new()),
            not_admitted: VecDeque::new(),
            background_running: 0,
            background_cap: 1,
            recent: RecentBusy::new(now),
            closed: false,
        };
        for _ in 0..2 {
            queues.by_class[TaskClass::Background.index()].push_back(background());
            STATS.queued[TaskClass::Background.index()].fetch_add(1, Ordering::Relaxed);
        }
        let scheduling = Scheduling::default();
        assert!(queues.take(&scheduling, now).is_some());
        assert!(queues.take(&scheduling, now).is_none());
        queues.background_running -= 1;
        assert!(queues.take(&scheduling, now).is_some());
        assert_eq!(background_cap(4), 2);
        assert_eq!(background_cap(1), 1);
    }

    #[test]
    fn the_recent_share_follows_the_last_seconds() {
        let start = Instant::now();
        let mut recent = RecentBusy::new(start);
        assert_eq!(recent.share(TaskClass::Search, start), 1.0);
        assert_eq!(recent.share(TaskClass::Background, start), 1.0);
        recent.add(TaskClass::Indexing, Duration::from_millis(800), start);
        recent.add(TaskClass::Background, Duration::from_millis(100), start);
        recent.add(TaskClass::Search, Duration::from_millis(100), start);
        assert!((recent.share(TaskClass::Search, start) - 0.1).abs() < 1e-9);
        assert!((recent.shares(start).background - 0.1).abs() < 1e-9);
        // Ten half-lives later, the old work weighs next to nothing against new searches.
        let later = start + 10 * SHARE_HALF_LIFE;
        recent.add(TaskClass::Search, Duration::from_millis(100), later);
        assert!(recent.share(TaskClass::Search, later) > 0.99);
    }

    async fn run_order(scheduling: Scheduling, sent: &[TaskClass]) -> Vec<TaskClass> {
        let worker = with_scheduling(scheduling);
        // Hold every worker, so that what is sent next queues up behind them.
        let workers: usize = perf::num_workers().into();
        let (release_tx, release_rx) = async_channel::bounded::<()>(1);
        let (held_tx, held_rx) = async_channel::unbounded::<()>();
        for _ in 0..workers {
            let release_rx = release_rx.clone();
            let held_tx = held_tx.clone();
            worker
                .spawn_async_non_blocking(move || async move {
                    held_tx.send(()).await.unwrap();
                    _ = release_rx.recv().await;
                })
                .await;
        }
        for _ in 0..workers {
            held_rx.recv().await.unwrap();
        }
        let (ran_tx, ran_rx) = async_channel::unbounded();
        for &class in sent {
            let ran_tx = ran_tx.clone();
            let task = move || ran_tx.try_send(class).unwrap();
            match class {
                TaskClass::Search => worker.spawn_blocking(task).await,
                TaskClass::Indexing => worker.spawn_indexing_blocking(task).await,
                TaskClass::Background => worker.spawn_background_blocking(task).await,
            }
        }
        // Let the dispatcher queue all of it before the workers are let go.
        tokio::time::sleep(Duration::from_millis(50)).await;
        release_tx.close();
        let mut ran = Vec::with_capacity(sent.len());
        for _ in sent {
            ran.push(ran_rx.recv().await.unwrap());
        }
        ran
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn queued_indexing_runs_before_searches_queued_earlier() {
        use TaskClass::Indexing;
        use TaskClass::Search;
        let ran = run_order(Scheduling::default(), &[Search, Search, Indexing, Indexing]).await;
        assert_eq!(&ran[..2], &[Indexing, Indexing], "{ran:?}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn with_a_long_budget_the_queues_are_served_searches_first() {
        use TaskClass::Indexing;
        use TaskClass::Search;
        let scheduling = Scheduling {
            indexing_queue_budget: Duration::from_secs(3600),
            ..Scheduling::default()
        };
        let ran = run_order(scheduling, &[Indexing, Search, Search]).await;
        assert_eq!(&ran[..2], &[Search, Search], "{ran:?}");
    }

    /// Background tasks queued first run after the searches and the indexing queued behind
    /// them: with no share of their own, after all of them.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn queued_background_runs_after_the_other_classes() {
        use TaskClass::Background;
        use TaskClass::Indexing;
        use TaskClass::Search;
        let scheduling = Scheduling {
            min_background_share: 0.0,
            ..Scheduling::default()
        };
        let sent = [Background, Background, Search, Indexing, Search];
        let ran = run_order(scheduling, &sent).await;
        assert_eq!(&ran[3..], &[Background, Background], "{ran:?}");
    }

    /// The pool takes as many background tasks as its queue has room for; the sender of one
    /// more waits until a task is taken, while a search sent meanwhile is queued at once.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn a_sender_of_background_tasks_waits_for_room_and_holds_up_no_one() {
        let worker = new();
        let capacity: usize = perf::channel_size().into();
        let (release_tx, release_rx) = async_channel::bounded::<()>(1);
        let (held_tx, held_rx) = async_channel::unbounded::<()>();
        let workers: usize = perf::num_workers().into();
        for _ in 0..workers {
            let (release_rx, held_tx) = (release_rx.clone(), held_tx.clone());
            worker
                .spawn_async_non_blocking(move || async move {
                    held_tx.send(()).await.unwrap();
                    _ = release_rx.recv().await;
                })
                .await;
        }
        for _ in 0..workers {
            held_rx.recv().await.unwrap();
        }
        let (ran_tx, ran_rx) = async_channel::unbounded();
        for _ in 0..capacity {
            let ran_tx = ran_tx.clone();
            tokio::time::timeout(
                Duration::from_secs(5),
                worker.spawn_background_blocking(move || ran_tx.try_send("background").unwrap()),
            )
            .await
            .expect("the queue has room");
        }
        let one_more = tokio::spawn({
            let (worker, ran_tx) = (worker.clone(), ran_tx.clone());
            async move {
                worker
                    .spawn_background_blocking(move || ran_tx.try_send("background").unwrap())
                    .await
            }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!one_more.is_finished(), "sent past the queue's room");
        tokio::time::timeout(
            Duration::from_secs(5),
            worker.spawn_blocking(move || ran_tx.try_send("search").unwrap()),
        )
        .await
        .expect("a search is queued while a background sender waits");
        release_tx.close();
        tokio::time::timeout(Duration::from_secs(5), one_more)
            .await
            .expect("room once tasks are taken")
            .unwrap();
        let mut ran = Vec::new();
        for _ in 0..capacity + 2 {
            ran.push(ran_rx.recv().await.unwrap());
        }
        assert_eq!(ran.iter().filter(|class| **class == "search").count(), 1);
    }
}
