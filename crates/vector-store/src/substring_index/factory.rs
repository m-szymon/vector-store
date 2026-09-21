/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

use crate::IndexKey;
use crate::IndexOptionsSubstring;
use crate::substring_index::actor::SubstringIndex;
use crate::table::Table;
use std::sync::Arc;
use std::sync::RwLock;
use tokio::sync::mpsc;

#[derive(Clone, Debug)]
pub(crate) struct SubstringIndexConfiguration {
    pub key: IndexKey,
    pub options: IndexOptionsSubstring,
}

pub(crate) trait SubstringIndexFactory {
    fn create_index(
        &self,
        index: SubstringIndexConfiguration,
        table: Arc<RwLock<Table>>,
    ) -> mpsc::Sender<SubstringIndex>;
}
