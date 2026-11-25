/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! A RAM directory for the Tantivy indexes whose meta lock wakes a waiter as soon as it is
//! released.
//!
//! Tantivy takes the meta lock to open the segments in a reload, and to find the files it can
//! delete after every commit and merge. Both hold it for about a millisecond. The RAM directory
//! of Tantivy has no lock of its own: a lock is a file, and a blocking take of a lock held by
//! someone else sleeps 100 ms between tries. A reload starts as a commit writes the meta, just
//! before the removal of unused files, so the two often met, and whoever came second slept
//! 100 ms: a reload on a worker, or the removal in the commit, with every write to the index
//! waiting for the commit. Tantivy's file system directory waits on an operating system lock
//! instead, which wakes the waiter at once; this directory does the same in memory.

use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::PoisonError;
use std::time::Duration;

use tantivy::directory::Directory;
use tantivy::directory::DirectoryLock;
use tantivy::directory::FileHandle;
use tantivy::directory::FileSlice;
use tantivy::directory::Lock;
use tantivy::directory::META_LOCK;
use tantivy::directory::RamDirectory;
use tantivy::directory::WatchCallback;
use tantivy::directory::WatchHandle;
use tantivy::directory::WritePtr;
use tantivy::directory::error::DeleteError;
use tantivy::directory::error::LockError;
use tantivy::directory::error::OpenReadError;
use tantivy::directory::error::OpenWriteError;

/// How long a blocking take of the meta lock waits before it fails, as long as Tantivy's own
/// tries last (100 tries, 100 ms apart).
const META_LOCK_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Debug, Default)]
pub(crate) struct MemoryDirectory {
    files: RamDirectory,
    meta_lock: Arc<MetaLock>,
}

#[derive(Debug, Default)]
struct MetaLock {
    held: Mutex<bool>,
    released: Condvar,
}

/// A benchmark test's switch and counters: with `SLEEPING` the meta lock is the RAM
/// directory's file, as before this directory; `WAITS` counts the blocking takes of the meta
/// lock that waited a millisecond or more, and sums their waits in nanoseconds.
#[cfg(test)]
pub(crate) static SLEEPING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
#[cfg(test)]
pub(crate) static WAITS: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl MetaLock {
    // A panic elsewhere cannot leave the flag half-written, so a poisoned mutex is used as is.
    fn acquire(self: &Arc<Self>, blocking: bool) -> Result<DirectoryLock, LockError> {
        let held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        let mut held = if blocking {
            let (held, timeout) = self
                .released
                .wait_timeout_while(held, META_LOCK_TIMEOUT, |held| *held)
                .unwrap_or_else(PoisonError::into_inner);
            if timeout.timed_out() {
                return Err(LockError::LockBusy);
            }
            held
        } else if *held {
            return Err(LockError::LockBusy);
        } else {
            held
        };
        *held = true;
        Ok(DirectoryLock::from(Box::new(MetaLockGuard(Arc::clone(
            self,
        )))))
    }
}

struct MetaLockGuard(Arc<MetaLock>);

impl Drop for MetaLockGuard {
    fn drop(&mut self) {
        *self.0.held.lock().unwrap_or_else(PoisonError::into_inner) = false;
        self.0.released.notify_one();
    }
}

impl Directory for MemoryDirectory {
    fn get_file_handle(&self, path: &Path) -> Result<Arc<dyn FileHandle>, OpenReadError> {
        self.files.get_file_handle(path)
    }

    fn open_read(&self, path: &Path) -> Result<FileSlice, OpenReadError> {
        self.files.open_read(path)
    }

    fn delete(&self, path: &Path) -> Result<(), DeleteError> {
        self.files.delete(path)
    }

    fn exists(&self, path: &Path) -> Result<bool, OpenReadError> {
        self.files.exists(path)
    }

    fn open_write(&self, path: &Path) -> Result<WritePtr, OpenWriteError> {
        self.files.open_write(path)
    }

    fn atomic_read(&self, path: &Path) -> Result<Vec<u8>, OpenReadError> {
        self.files.atomic_read(path)
    }

    fn atomic_write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.files.atomic_write(path, data)
    }

    fn sync_directory(&self) -> io::Result<()> {
        self.files.sync_directory()
    }

    fn watch(&self, watch_callback: WatchCallback) -> tantivy::Result<WatchHandle> {
        self.files.watch(watch_callback)
    }

    fn acquire_lock(&self, lock: &Lock) -> Result<DirectoryLock, LockError> {
        if lock.filepath != META_LOCK.filepath {
            return self.files.acquire_lock(lock);
        }
        #[cfg(test)]
        let clock = std::time::Instant::now();
        #[cfg(test)]
        let acquired = if SLEEPING.load(std::sync::atomic::Ordering::Relaxed) {
            self.files.acquire_lock(lock)
        } else {
            self.meta_lock.acquire(lock.is_blocking)
        };
        #[cfg(not(test))]
        let acquired = self.meta_lock.acquire(lock.is_blocking);
        #[cfg(test)]
        if lock.is_blocking && clock.elapsed() >= Duration::from_millis(1) {
            use std::sync::atomic::Ordering::Relaxed;
            WAITS[0].fetch_add(1, Relaxed);
            WAITS[1].fetch_add(clock.elapsed().as_nanos() as u64, Relaxed);
        }
        acquired
    }
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Instant;

    use super::*;

    fn non_blocking_meta_lock() -> Lock {
        Lock {
            filepath: META_LOCK.filepath.clone(),
            is_blocking: false,
        }
    }

    /// A blocking take of a held meta lock goes on as soon as the lock is released, not after
    /// the 100 ms Tantivy's RAM directory sleeps between its tries: released 10 ms after the
    /// first try, the RAM directory's lock would be taken 90 ms after the release.
    #[test]
    fn a_held_meta_lock_wakes_its_waiter_when_released() {
        let directory = MemoryDirectory::default();
        let held = directory.acquire_lock(&META_LOCK).unwrap();
        let release = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            // Taken before the release, so the waiter cannot have the lock earlier.
            let released = Instant::now();
            drop(held);
            released
        });
        let _lock = directory.acquire_lock(&META_LOCK).unwrap();
        let acquired = Instant::now();
        let released = release.join().unwrap();
        assert!(acquired >= released);
        let after_release = acquired - released;
        assert!(
            after_release < Duration::from_millis(50),
            "{after_release:?}"
        );
    }

    /// A non-blocking take fails while the meta lock is held by a clone of the directory, as
    /// Tantivy's reader and writer each hold one, and succeeds once it is released.
    #[test]
    fn a_non_blocking_take_fails_while_the_meta_lock_is_held() {
        let directory = MemoryDirectory::default();
        let clone = directory.clone();
        let held = directory.acquire_lock(&META_LOCK).unwrap();
        assert!(matches!(
            clone.acquire_lock(&non_blocking_meta_lock()),
            Err(LockError::LockBusy)
        ));
        drop(held);
        assert!(clone.acquire_lock(&non_blocking_meta_lock()).is_ok());
    }

    /// The other locks, the writer's among them, stay files of the RAM directory.
    #[test]
    fn other_locks_are_unchanged() {
        let directory = MemoryDirectory::default();
        let writer_lock = || Lock {
            filepath: "writer.lock".into(),
            is_blocking: false,
        };
        let held = directory.acquire_lock(&writer_lock()).unwrap();
        assert!(matches!(
            directory.acquire_lock(&writer_lock()),
            Err(LockError::LockBusy)
        ));
        let _meta = directory.acquire_lock(&META_LOCK).unwrap();
        drop(held);
        assert!(directory.acquire_lock(&writer_lock()).is_ok());
    }
}
