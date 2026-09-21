/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

use crate::AsyncInProgress;
use crate::IndexKey;
use crate::Limit;
use crate::PrimaryKey;
use crate::table::PrimaryId;
use crate::tantivy_common::TantivyStatsR;
use crate::vs_index::CountR;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

/// Primary keys of the rows whose indexed value contains the query, in index order.
pub(crate) type SubstringSearchR = anyhow::Result<Vec<PrimaryKey>>;

pub(crate) enum SubstringIndex {
    AddDocument {
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
        /// Number of matching rows to skip before collecting `limit` of them.
        offset: usize,
        tx: oneshot::Sender<SubstringSearchR>,
    },
    Stats {
        index_key: IndexKey,
        tx: oneshot::Sender<TantivyStatsR>,
    },
}

pub(crate) trait SubstringIndexExt {
    async fn add_document(
        &self,
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
    ) -> SubstringSearchR;
    async fn stats(&self, index_key: IndexKey) -> TantivyStatsR;
}

impl SubstringIndexExt for mpsc::Sender<SubstringIndex> {
    async fn add_document(
        &self,
        primary_id: PrimaryId,
        document: String,
        in_progress: AsyncInProgress,
    ) -> anyhow::Result<()> {
        Ok(self
            .send(SubstringIndex::AddDocument {
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
    ) -> SubstringSearchR {
        let (tx, rx) = oneshot::channel();
        self.send(SubstringIndex::Search {
            index_key,
            query,
            limit,
            offset,
            tx,
        })
        .await?;
        rx.await?
    }

    async fn stats(&self, index_key: IndexKey) -> TantivyStatsR {
        let (tx, rx) = oneshot::channel();
        self.send(SubstringIndex::Stats { index_key, tx }).await?;
        rx.await?
    }
}
