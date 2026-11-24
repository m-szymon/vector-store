/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! Pattern index: `LIKE` patterns served by an n-gram Tantivy index.

mod actor;
mod factory;
mod gram;
mod like;
mod order;
mod tantivy;

use crate::Config;
use crate::memory::Memory;
use crate::metrics::Metrics;
use crate::worker::Worker;
pub(crate) use actor::Cursor;
pub(crate) use actor::PatternIndex;
pub(crate) use actor::PatternIndexExt;
pub(crate) use actor::SearchWindow;
pub(crate) use actor::SortOrder;
pub(crate) use factory::PatternIndexConfiguration;
pub(crate) use factory::PatternIndexFactory;
pub use order::KeyLayout;
pub(crate) use order::PREFIX_SLOTS;
pub(crate) use order::SortPrefix;
pub(crate) use order::is_key_type_supported;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::watch;

/// A search request's budget of time is read from `config` as each request starts, so a
/// configuration reload changes it for the requests that follow.
pub(crate) fn new_factory_tantivy(
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
    metrics: Arc<Metrics>,
    config: watch::Receiver<Arc<Config>>,
) -> Box<dyn PatternIndexFactory + Send + Sync> {
    Box::new(tantivy::new_factory(worker, memory, metrics, config))
}
