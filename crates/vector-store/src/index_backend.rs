/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

use crate::ColumnName;
use crate::Dimensions;
use crate::IndexName;
use crate::KeyspaceName;
use crate::TableName;
use async_trait::async_trait;
use futures::TryStreamExt;
use regex::Regex;
use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;
use std::collections::BTreeMap;
use std::num::NonZeroUsize;

pub(crate) struct IndexLocation {
    pub keyspace: KeyspaceName,
    pub table: TableName,
    pub index: IndexName,
}

#[async_trait]
pub(crate) trait IndexBackend: Send + Sync {
    async fn get_dimensions(
        &self,
        session: &Session,
        st_get_index_target_type: &PreparedStatement,
        re_get_index_target_type: &Regex,
        st_get_index_options: &PreparedStatement,
        location: IndexLocation,
    ) -> anyhow::Result<Option<Dimensions>>;
}

pub(crate) struct CqlBackend {
    target_column: ColumnName,
}

pub(crate) struct AlternatorBackend {}

pub(crate) fn new(keyspace: &KeyspaceName, target_column: ColumnName) -> Box<dyn IndexBackend> {
    if keyspace.is_alternator() {
        Box::new(AlternatorBackend {})
    } else {
        Box::new(CqlBackend { target_column })
    }
}

#[async_trait]
impl IndexBackend for CqlBackend {
    async fn get_dimensions(
        &self,
        session: &Session,
        st_get_index_target_type: &PreparedStatement,
        re_get_index_target_type: &Regex,
        _st_get_index_options: &PreparedStatement,
        location: IndexLocation,
    ) -> anyhow::Result<Option<Dimensions>> {
        let column_type = session
            .execute_iter(
                st_get_index_target_type.clone(),
                (
                    location.keyspace,
                    location.table,
                    self.target_column.clone(),
                ),
            )
            .await?
            .rows_stream::<(String,)>()?
            .try_next()
            .await?;
        let dimensions = column_type
            .and_then(|(typ,)| {
                re_get_index_target_type
                    .captures(&typ)
                    .and_then(|captures| captures["dimensions"].parse::<usize>().ok())
            })
            .and_then(|dimensions| {
                NonZeroUsize::new(dimensions).map(|dimensions| dimensions.into())
            });
        Ok(dimensions)
    }
}

#[async_trait]
impl IndexBackend for AlternatorBackend {
    async fn get_dimensions(
        &self,
        session: &Session,
        _st_get_index_target_type: &PreparedStatement,
        _re_get_index_target_type: &Regex,
        st_get_index_options: &PreparedStatement,
        location: IndexLocation,
    ) -> anyhow::Result<Option<Dimensions>> {
        let index_options = session
            .execute_iter(
                st_get_index_options.clone(),
                (location.keyspace, location.table, location.index),
            )
            .await?
            .rows_stream::<(BTreeMap<String, String>,)>()?
            .try_next()
            .await?;
        let dimensions = index_options
            .and_then(|(mut options,)| {
                options
                    .remove("dimensions")
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .and_then(|dimensions| {
                NonZeroUsize::new(dimensions).map(|dimensions| dimensions.into())
            });
        Ok(dimensions)
    }
}
