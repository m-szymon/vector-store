/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

use super::order::SortPrefix;
use super::tantivy::PatternStatsR;
use crate::AsyncInProgress;
use crate::IndexKey;
use crate::Limit;
use crate::PrimaryKey;
use crate::table::PartitionId;
use crate::table::PrimaryId;
use crate::vs_index::CountR;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

/// The direction an ordered search asks for, on the index's first ordering column. The node turns
/// it into "the declared order" or "its reverse"; a search that names none gets the declared order,
/// which for an index in table order is ascending.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SortOrder {
    Desc,
    Asc,
}

/// Where a page ended: the sort prefix of its last row and that row's primary key, so that the
/// next page resumes among the rows sharing that prefix instead of skipping them. The primary key
/// is `None` when the row was gone by the time the page was assembled; a resume from such a cursor
/// takes every row sharing the prefix again rather than risk skipping one.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Cursor {
    pub(crate) prefix: SortPrefix,
    pub(crate) primary_key: Option<PrimaryKey>,
}

/// What an ordered search may return: a direction, the cursor of the previous page, and a range
/// on the first ordering column, as sort keys of that column's values.
#[derive(Clone, Debug, Default)]
pub(crate) struct SearchWindow {
    pub(crate) order: Option<SortOrder>,
    pub(crate) cursor: Option<Cursor>,
    pub(crate) min_sort_key: Option<u64>,
    pub(crate) max_sort_key: Option<u64>,
}

/// One page of a pattern search.
#[derive(Debug)]
pub(crate) struct PatternPage {
    /// Primary keys of the rows whose indexed value matches the pattern, in the index's order.
    pub(crate) primary_keys: Vec<PrimaryKey>,
    /// Where the next page starts, for a search that filled this one. `None` means there is
    /// nothing more to read.
    pub(crate) next_cursor: Option<Cursor>,
}

pub(crate) type PatternLikeR = anyhow::Result<PatternPage>;

pub(crate) enum PatternIndex {
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
    Like {
        index_key: IndexKey,
        pattern: String,
        limit: Limit,
        /// What of the index's order the page may hold: see `SearchWindow`.
        window: SearchWindow,
        tx: oneshot::Sender<PatternLikeR>,
    },
    Stats {
        index_key: IndexKey,
        tx: oneshot::Sender<PatternStatsR>,
    },
}

pub(crate) trait PatternIndexExt {
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
    async fn like(
        &self,
        index_key: IndexKey,
        pattern: String,
        limit: Limit,
        window: SearchWindow,
    ) -> PatternLikeR;
    async fn stats(&self, index_key: IndexKey) -> PatternStatsR;
}

impl PatternIndexExt for mpsc::Sender<PatternIndex> {
    async fn add_document(
        &self,
        partition_id: PartitionId,
        primary_id: PrimaryId,
        document: String,
        in_progress: AsyncInProgress,
    ) -> anyhow::Result<()> {
        Ok(self
            .send(PatternIndex::AddDocument {
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
            .send(PatternIndex::RemoveDocument {
                primary_id,
                in_progress,
            })
            .await?)
    }

    async fn count(&self, index_key: IndexKey) -> CountR {
        let (tx, rx) = oneshot::channel();
        self.send(PatternIndex::Count { index_key, tx }).await?;
        rx.await?
    }

    async fn like(
        &self,
        index_key: IndexKey,
        pattern: String,
        limit: Limit,
        window: SearchWindow,
    ) -> PatternLikeR {
        let (tx, rx) = oneshot::channel();
        self.send(PatternIndex::Like {
            index_key,
            pattern,
            limit,
            window,
            tx,
        })
        .await?;
        rx.await?
    }

    async fn stats(&self, index_key: IndexKey) -> PatternStatsR {
        let (tx, rx) = oneshot::channel();
        self.send(PatternIndex::Stats { index_key, tx }).await?;
        rx.await?
    }
}
