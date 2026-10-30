/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! Pattern index: `LIKE` patterns served by an n-gram Tantivy index.

mod actor;
mod factory;
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
pub(crate) use like::parse as parse_like_pattern;
pub use order::KeyLayout;
pub(crate) use order::PREFIX_SLOTS;
pub(crate) use order::SortPrefix;
pub(crate) use order::is_key_type_supported;
use std::sync::Arc;
use tantivy::TantivyPatternIndexFactory;
use tokio::sync::mpsc;

pub(crate) fn new_pattern_index_factory_tantivy(
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
    metrics: Arc<Metrics>,
) -> Box<dyn PatternIndexFactory + Send + Sync> {
    Box::new(TantivyPatternIndexFactory::new(worker, memory, metrics))
}
