/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

use crate::IndexKey;
use crate::IndexOptionsPattern;
use crate::pattern_index::actor::PatternIndex;
use crate::table::Table;
use std::sync::Arc;
use std::sync::RwLock;
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub(crate) struct PatternIndexConfiguration {
    pub key: IndexKey,
    pub options: IndexOptionsPattern,
}

pub(crate) trait PatternIndexFactory {
    fn create_index(
        &self,
        index: PatternIndexConfiguration,
        table: Arc<RwLock<Table>>,
    ) -> mpsc::Sender<PatternIndex>;
}
