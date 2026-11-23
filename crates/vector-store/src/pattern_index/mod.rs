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
use std::time::Duration;
use tokio::sync::mpsc;

pub(crate) use tantivy::DEFAULT_SEARCH_BUDGET;

/// `search_budget` is how long one search request may take before it returns what it has
/// settled; None for no limit.
pub(crate) fn new_factory_tantivy(
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
    metrics: Arc<Metrics>,
    search_budget: Option<Duration>,
) -> Box<dyn PatternIndexFactory + Send + Sync> {
    Box::new(tantivy::new_factory(worker, memory, metrics, search_budget))
}
