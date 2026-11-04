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
}

impl TaskClass {
    const ALL: [TaskClass; 2] = [TaskClass::Search, TaskClass::Indexing];

    fn index(self) -> usize {
        self as usize
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            TaskClass::Search => "search",
            TaskClass::Indexing => "indexing",
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
        })
        .await
        .expect("WorkerExt::spawn_blocking: internal actor should receive request");
    }

    #[hotpath::measure]
    async fn spawn_indexing_blocking(&self, f: impl FnOnce() + Send + 'static) {
        self.send(Worker::SpawnBlocking {
            f: Box::new(f),
            class: TaskClass::Indexing,
        })
        .await
        .expect("WorkerExt::spawn_indexing_blocking: internal actor should receive request");
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

/// How a free worker chooses between the two queues when both have work.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Scheduling {
    /// How long the oldest indexing task may wait while searches are served first. Zero: indexing
    /// always first (`VECTOR_STORE_INDEXING_QUEUE_BUDGET`).
    pub(crate) indexing_queue_budget: Duration,
    /// The share of the workers' recent time, 0.0 to 1.0, that searches waiting get however long
    /// indexing has waited (`VECTOR_STORE_MIN_SEARCH_SHARE`, in percent).
    pub(crate) min_search_share: f64,
}

impl Default for Scheduling {
    fn default() -> Self {
        Self {
            indexing_queue_budget: Duration::ZERO,
            min_search_share: DEFAULT_MIN_SEARCH_SHARE,
        }
    }
}

pub(crate) const DEFAULT_MIN_SEARCH_SHARE: f64 = 0.25;

/// The class the next task comes from, given what waits: `None` when nothing does. Pure, so
/// that the policy can be tested without a runtime.
fn pick_class(
    search_waiting: bool,
    oldest_indexing_wait: Option<Duration>,
    search_share: f64,
    scheduling: &Scheduling,
) -> Option<TaskClass> {
    match (search_waiting, oldest_indexing_wait) {
        (false, None) => None,
        (true, None) => Some(TaskClass::Search),
        (false, Some(_)) => Some(TaskClass::Indexing),
        (true, Some(waited)) => {
            if waited < scheduling.indexing_queue_budget
                || search_share < scheduling.min_search_share
            {
                Some(TaskClass::Search)
            } else {
                Some(TaskClass::Indexing)
            }
        }
    }
}

/// How fast past work stops counting towards a class's share: its weight halves every
/// `SHARE_HALF_LIFE`, so the share follows the last few seconds.
const SHARE_HALF_LIFE: Duration = Duration::from_secs(1);

/// The workers' recent time per class, decaying with `SHARE_HALF_LIFE`.
struct RecentBusy {
    seconds: [f64; 2],
    at: Instant,
}

impl RecentBusy {
    fn new(now: Instant) -> Self {
        Self {
            seconds: [0.0; 2],
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

    /// Searches' share of the recent time; 1.0 when nothing ran, so that an idle pool does not
    /// read as one starving its searches.
    fn search_share(&mut self, now: Instant) -> f64 {
        self.decay(now);
        let total: f64 = self.seconds.iter().sum();
        if total <= 0.0 {
            1.0
        } else {
            self.seconds[TaskClass::Search.index()] / total
        }
    }
}

struct Queued {
    msg: Worker,
    enqueued: Instant,
}

struct Queues {
    by_class: [VecDeque<Queued>; 2],
    recent: RecentBusy,
    closed: bool,
}

impl Queues {
    fn take(&mut self, scheduling: &Scheduling, now: Instant) -> Option<(Queued, TaskClass)> {
        let oldest_indexing_wait = self.by_class[TaskClass::Indexing.index()]
            .front()
            .map(|queued| now.saturating_duration_since(queued.enqueued));
        let search_waiting = !self.by_class[TaskClass::Search.index()].is_empty();
        let share = self.recent.search_share(now);
        let class = pick_class(search_waiting, oldest_indexing_wait, share, scheduling)?;
        let queued = self.by_class[class.index()].pop_front()?;
        STATS.queued[class.index()].fetch_sub(1, Ordering::Relaxed);
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
    tasks: [AtomicU64; 2],
    queue_wait_nanos: [AtomicU64; 2],
    busy_nanos: [AtomicU64; 2],
    /// Tasks in the queues now.
    queued: [AtomicU64; 2],
}

pub(crate) static STATS: WorkerStats = WorkerStats {
    tasks: [AtomicU64::new(0), AtomicU64::new(0)],
    queue_wait_nanos: [AtomicU64::new(0), AtomicU64::new(0)],
    busy_nanos: [AtomicU64::new(0), AtomicU64::new(0)],
    queued: [AtomicU64::new(0), AtomicU64::new(0)],
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
    pub(crate) fn snapshot(&self) -> [(TaskClass, u64, f64, f64); 2] {
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
            by_class: [VecDeque::new(), VecDeque::new()],
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
    // their queue holds whatever is sent.
    tokio::spawn({
        let shared = Arc::clone(&shared);
        async move {
            while let Ok(msg) = rx_worker.recv().await {
                let class = msg.class();
                loop {
                    let room = shared.room.notified();
                    {
                        let mut queues = shared.queues.lock().unwrap();
                        let queue = &mut queues.by_class[class.index()];
                        if class == TaskClass::Search || queue.len() < capacity {
                            queue.push_back(Queued {
                                msg,
                                enqueued: Instant::now(),
                            });
                            STATS.queued[class.index()].fetch_add(1, Ordering::Relaxed);
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
                    shared
                        .queues
                        .lock()
                        .unwrap()
                        .recent
                        .add(class, busy, finished);
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

    #[test]
    fn an_empty_queue_gives_nothing_and_a_lone_class_goes_first() {
        let scheduling = Scheduling::default();
        assert_eq!(pick_class(false, None, 1.0, &scheduling), None);
        assert_eq!(
            pick_class(true, None, 0.0, &scheduling),
            Some(TaskClass::Search)
        );
        assert_eq!(
            pick_class(false, Some(Duration::ZERO), 1.0, &scheduling),
            Some(TaskClass::Indexing)
        );
    }

    /// The default: indexing first whenever both wait, so the lag holds.
    #[test]
    fn by_default_indexing_goes_before_searches() {
        let scheduling = Scheduling::default();
        assert_eq!(
            pick_class(true, Some(Duration::ZERO), 0.9, &scheduling),
            Some(TaskClass::Indexing)
        );
    }

    #[test]
    fn indexing_within_its_budget_lets_searches_go_first() {
        assert_eq!(
            pick_class(true, Some(Duration::from_millis(50)), 0.9, &BUDGET_100MS),
            Some(TaskClass::Search)
        );
        assert_eq!(
            pick_class(true, Some(Duration::from_millis(150)), 0.9, &BUDGET_100MS),
            Some(TaskClass::Indexing)
        );
    }

    /// However long indexing has waited, searches below their minimum share go first: a burst
    /// of writes slows the searches and lets the lag grow rather than stopping the searches.
    #[test]
    fn searches_below_their_minimum_share_go_first_however_long_indexing_waited() {
        let scheduling = Scheduling::default();
        assert_eq!(
            pick_class(true, Some(Duration::from_secs(60)), 0.1, &scheduling),
            Some(TaskClass::Search)
        );
        assert_eq!(
            pick_class(true, Some(Duration::from_secs(60)), 0.3, &scheduling),
            Some(TaskClass::Indexing)
        );
    }

    #[test]
    fn the_recent_share_follows_the_last_seconds() {
        let start = Instant::now();
        let mut recent = RecentBusy::new(start);
        assert_eq!(recent.search_share(start), 1.0);
        recent.add(TaskClass::Indexing, Duration::from_millis(900), start);
        recent.add(TaskClass::Search, Duration::from_millis(100), start);
        assert!((recent.search_share(start) - 0.1).abs() < 1e-9);
        // Ten half-lives later, the old work weighs next to nothing against new searches.
        let later = start + 10 * SHARE_HALF_LIFE;
        recent.add(TaskClass::Search, Duration::from_millis(100), later);
        assert!(recent.search_share(later) > 0.99);
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
            min_search_share: 0.25,
        };
        let ran = run_order(scheduling, &[Indexing, Search, Search]).await;
        assert_eq!(&ran[..2], &[Search, Search], "{ran:?}");
    }
}
