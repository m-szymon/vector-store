/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

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

/// Where in the value the keyword has to occur.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum MatchKind {
    /// Anywhere: `LIKE '%keyword%'`.
    #[default]
    Contains,
    /// At the start: `LIKE 'keyword%'`.
    Prefix,
    /// At the end: `LIKE '%keyword'`.
    Suffix,
}

/// Which way an ordered search walks the sort column.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SortOrder {
    /// Highest sort key first: newest first for a timestamp.
    #[default]
    Desc,
    Asc,
}

/// Where a page ended: the sort key of its last row and that row's primary key, so that the next
/// page resumes among the rows sharing that sort key instead of skipping them. The primary key is
/// what makes the cursor survive a rebuild of the index, whose internal ids are not stable; it is
/// `None` when the row was gone by the time the page was assembled, and a resume from such a
/// cursor takes every row sharing the sort key again rather than risk skipping one.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Cursor {
    pub(crate) sort_key: u64,
    pub(crate) primary_key: Option<PrimaryKey>,
}

/// What an ordered search may return: a direction, the cursor of the previous page, and a range
/// on the sort column. Ignored by an unordered index.
#[derive(Clone, Debug, Default)]
pub(crate) struct SearchWindow {
    pub(crate) order: SortOrder,
    pub(crate) cursor: Option<Cursor>,
    pub(crate) min_sort_key: Option<u64>,
    pub(crate) max_sort_key: Option<u64>,
}

/// One page of a substring search.
#[derive(Debug)]
pub(crate) struct SubstringPage {
    /// Primary keys of the rows whose indexed value contains the query. Ordered by the index's
    /// sort column when it has one, otherwise in whatever order the walk found them.
    pub(crate) primary_keys: Vec<PrimaryKey>,
    /// Where the next page starts, for an ordered search that filled this one. `None` means there
    /// is nothing more to read, or the index is unordered and does not page this way.
    pub(crate) next_cursor: Option<Cursor>,
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
        kind: MatchKind,
        limit: Limit,
        /// Number of matching rows to skip before collecting `limit` of them. Ignored by an
        /// ordered search, which pages by cursor instead.
        offset: usize,
        /// The direction, the previous page's cursor and the range restriction. Ignored by an
        /// unordered index.
        window: SearchWindow,
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
        kind: MatchKind,
        limit: Limit,
        offset: usize,
        window: SearchWindow,
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
        kind: MatchKind,
        limit: Limit,
        offset: usize,
        window: SearchWindow,
    ) -> SubstringSearchR {
        let (tx, rx) = oneshot::channel();
        self.send(SubstringIndex::Search {
            index_key,
            query,
            kind,
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
