/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

use super::tantivy::SortWindow;
use super::tantivy::SubstringStatsR;
use crate::AsyncInProgress;
use crate::IndexKey;
use crate::Limit;
use crate::PrimaryKey;
use crate::table::PartitionId;
use crate::table::PrimaryId;
use crate::vs_index::CountR;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

/// One page of a substring search.
#[derive(Debug)]
pub(crate) struct SubstringPage {
    /// Primary keys of the rows whose indexed value contains the query. Ordered by the index's
    /// sort column when it has one, otherwise in whatever order the walk found them.
    pub(crate) primary_keys: Vec<PrimaryKey>,
    /// Where the next page starts, for an ordered search that filled this one. `None` means there
    /// is nothing more to read, or the index is unordered and does not page this way.
    pub(crate) next_cursor: Option<u64>,
}

pub(crate) type SubstringSearchR = anyhow::Result<SubstringPage>;

pub(crate) enum SubstringIndex {
    AddDocument {
        /// Needed to read the sort column back out of the table: a value is keyed by the partition
        /// it lives in as well as by its primary id.
        partition_id: PartitionId,
        primary_id: PrimaryId,
        document: String,
        in_progress: AsyncInProgress,
    },
    RemoveDocument {
        primary_id: PrimaryId,
        in_progress: AsyncInProgress,
    },
    Count {
        index_key: IndexKey,
        tx: oneshot::Sender<CountR>,
    },
    Search {
        index_key: IndexKey,
        query: String,
        limit: Limit,
        /// Number of matching rows to skip before collecting `limit` of them. Ignored by an
        /// ordered search, which pages by cursor instead.
        offset: usize,
        /// Which sort keys the answer may come from: a range restriction, a paging cursor, or
        /// both. Ignored by an unordered index.
        window: SortWindow,
        tx: oneshot::Sender<SubstringSearchR>,
    },
    Stats {
        index_key: IndexKey,
        tx: oneshot::Sender<SubstringStatsR>,
    },
}

pub(crate) trait SubstringIndexExt {
    async fn add_document(
        &self,
        partition_id: PartitionId,
        primary_id: PrimaryId,
        document: String,
        in_progress: AsyncInProgress,
    ) -> anyhow::Result<()>;
    async fn remove_document(
        &self,
        primary_id: PrimaryId,
        in_progress: AsyncInProgress,
    ) -> anyhow::Result<()>;
    async fn count(&self, index_key: IndexKey) -> CountR;
    async fn search(
        &self,
        index_key: IndexKey,
        query: String,
        limit: Limit,
        offset: usize,
        window: SortWindow,
    ) -> SubstringSearchR;
    async fn stats(&self, index_key: IndexKey) -> SubstringStatsR;
}

impl SubstringIndexExt for mpsc::Sender<SubstringIndex> {
    async fn add_document(
        &self,
        partition_id: PartitionId,
        primary_id: PrimaryId,
        document: String,
        in_progress: AsyncInProgress,
    ) -> anyhow::Result<()> {
        Ok(self
            .send(SubstringIndex::AddDocument {
                partition_id,
                primary_id,
                document,
                in_progress,
            })
            .await?)
    }

    async fn remove_document(
        &self,
        primary_id: PrimaryId,
        in_progress: AsyncInProgress,
    ) -> anyhow::Result<()> {
        Ok(self
            .send(SubstringIndex::RemoveDocument {
                primary_id,
                in_progress,
            })
            .await?)
    }

    async fn count(&self, index_key: IndexKey) -> CountR {
        let (tx, rx) = oneshot::channel();
        self.send(SubstringIndex::Count { index_key, tx }).await?;
        rx.await?
    }

    async fn search(
        &self,
        index_key: IndexKey,
        query: String,
        limit: Limit,
        offset: usize,
        window: SortWindow,
    ) -> SubstringSearchR {
        let (tx, rx) = oneshot::channel();
        self.send(SubstringIndex::Search {
            index_key,
            query,
            limit,
            offset,
            window,
            tx,
        })
        .await?;
        rx.await?
    }

    async fn stats(&self, index_key: IndexKey) -> SubstringStatsR {
        let (tx, rx) = oneshot::channel();
        self.send(SubstringIndex::Stats { index_key, tx }).await?;
        rx.await?
    }
}
