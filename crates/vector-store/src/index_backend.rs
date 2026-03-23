/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

use crate::ColumnName;
use crate::Dimensions;
use crate::IndexName;
use crate::KeyspaceName;
use crate::TableName;
use crate::Vector;
use crate::vector;
use async_trait::async_trait;
use futures::TryStreamExt;
use regex::Regex;
use scylla::client::session::Session;
use scylla::statement::prepared::PreparedStatement;
use scylla::value::CqlValue;
use std::collections::BTreeMap;
use std::num::NonZeroUsize;

pub(crate) struct IndexLocation {
    pub keyspace: KeyspaceName,
    pub table: TableName,
    pub index: IndexName,
}

#[async_trait]
pub(crate) trait IndexBackend: Send + Sync {
    fn vector_column_name(&self) -> &str;

    fn extract_vector(&self, value: CqlValue) -> anyhow::Result<Option<Vector>>;

    fn range_scan_query(
        &self,
        keyspace: &KeyspaceName,
        table: &TableName,
        primary_key_list: &str,
        partition_key_list: &str,
    ) -> String;

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

pub(crate) struct AlternatorBackend {
    target_column: ColumnName,
}

pub(crate) fn new(keyspace: &KeyspaceName, target_column: ColumnName) -> Box<dyn IndexBackend> {
    if keyspace.is_alternator() {
        Box::new(AlternatorBackend { target_column })
    } else {
        Box::new(CqlBackend { target_column })
    }
}

#[async_trait]
impl IndexBackend for CqlBackend {
    fn vector_column_name(&self) -> &str {
        self.target_column.as_ref()
    }

    fn extract_vector(&self, value: CqlValue) -> anyhow::Result<Option<Vector>> {
        Vector::try_from(value).map(Some)
    }

    fn range_scan_query(
        &self,
        keyspace: &KeyspaceName,
        table: &TableName,
        primary_key_list: &str,
        partition_key_list: &str,
    ) -> String {
        let vector = self.target_column.as_ref();
        format!(
            "
            SELECT {primary_key_list}, {vector}, writetime({vector})
            FROM {keyspace}.{table}
            WHERE
                token({partition_key_list}) >= ?
                AND token({partition_key_list}) <= ?
            BYPASS CACHE
            "
        )
    }

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
    fn vector_column_name(&self) -> &str {
        ":attrs"
    }

    fn extract_vector(&self, value: CqlValue) -> anyhow::Result<Option<Vector>> {
        vector::AlternatorAttrs {
            attrs: value,
            target_column: self.target_column.as_ref(),
        }
        .try_into()
    }

    /// Alternator stores non-key attributes in a `map<bytes, bytes>` column named `:attrs`.
    /// The range scan selects the target attribute from the map using `":attrs"['<name>']`.
    fn range_scan_query(
        &self,
        keyspace: &KeyspaceName,
        table: &TableName,
        primary_key_list: &str,
        partition_key_list: &str,
    ) -> String {
        let vector = self.target_column.as_ref();
        format!(
            "
            SELECT {primary_key_list}, \":attrs\"['{vector}'], writetime(\":attrs\"['{vector}'])
            FROM \"{keyspace}\".\"{table}\"
            WHERE
                token({partition_key_list}) >= ?
                AND token({partition_key_list}) <= ?
            BYPASS CACHE
            "
        )
    }

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
