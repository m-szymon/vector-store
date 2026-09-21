/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! Substring (infix containment) index: `LIKE '%kw%'` served by an n-gram Tantivy index.

mod actor;
mod factory;
mod tantivy;

use crate::memory::Memory;
use crate::worker::Worker;
pub(crate) use actor::SubstringIndex;
pub(crate) use actor::SubstringIndexExt;
pub(crate) use factory::SubstringIndexConfiguration;
pub(crate) use factory::SubstringIndexFactory;
use tantivy::TantivySubstringIndexFactory;
use tokio::sync::mpsc;

pub(crate) fn new_substring_index_factory_tantivy(
    worker: async_channel::Sender<Worker>,
    memory: mpsc::Sender<Memory>,
) -> Box<dyn SubstringIndexFactory + Send + Sync> {
    Box::new(TantivySubstringIndexFactory::new(worker, memory))
}
