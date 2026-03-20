/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.0
 */

use crate::ColumnName;
use crate::CqlLiteral;
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
use scylla_cdc::CqlIdentifier;
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
        keyspace: &CqlIdentifier,
        table: &CqlIdentifier,
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
        keyspace: &CqlIdentifier,
        table: &CqlIdentifier,
        primary_key_list: &str,
        partition_key_list: &str,
    ) -> String {
        let vector = CqlIdentifier::new(self.target_column.as_ref());
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
        keyspace: &CqlIdentifier,
        table: &CqlIdentifier,
        primary_key_list: &str,
        partition_key_list: &str,
    ) -> String {
        let attributes = CqlIdentifier::new(":attrs");
        let vector = CqlLiteral::new(self.target_column.as_ref());
        format!(
            "
            SELECT {primary_key_list}, {attributes}[{vector}], writetime({attributes}[{vector}])
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

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;

    #[test]
    fn range_scan_query_quotes_lowercase_identifiers() {
        let backend = CqlBackend {
            target_column: ColumnName::from("embedding"),
        };
        let query = backend.range_scan_query(
            &CqlIdentifier::new("ks"),
            &CqlIdentifier::new("tbl"),
            &CqlIdentifier::new("id").to_string(),
            &CqlIdentifier::new("id").to_string(),
        );
        assert!(query.contains(r#""embedding""#));
        assert!(query.contains(r#"FROM "ks"."tbl""#));
        assert!(query.contains(r#"token("id")"#));
    }

    #[test]
    fn range_scan_query_quotes_mixed_case_identifiers() {
        let pk_list = [
            CqlIdentifier::new("UserId"),
            CqlIdentifier::new("CreatedAt"),
        ]
        .iter()
        .join(", ");
        let backend = CqlBackend {
            target_column: ColumnName::from("EmbeddingCol"),
        };
        let query = backend.range_scan_query(
            &CqlIdentifier::new("MyKeyspace"),
            &CqlIdentifier::new("MyTable"),
            &pk_list,
            &CqlIdentifier::new("UserId").to_string(),
        );
        assert!(
            query.contains(r#""EmbeddingCol""#),
            "mixed-case embedding column must be quoted"
        );
        assert!(
            query.contains(r#"FROM "MyKeyspace"."MyTable""#),
            "mixed-case keyspace/table must be quoted"
        );
        assert!(
            query.contains(r#""UserId", "CreatedAt""#),
            "mixed-case primary key columns must be quoted"
        );
    }

    #[test]
    fn range_scan_query_quotes_uppercase_identifiers() {
        let backend = CqlBackend {
            target_column: ColumnName::from("VEC"),
        };
        let query = backend.range_scan_query(
            &CqlIdentifier::new("UPPER_KS"),
            &CqlIdentifier::new("UPPER_TBL"),
            &CqlIdentifier::new("ID").to_string(),
            &CqlIdentifier::new("ID").to_string(),
        );
        assert!(
            query.contains(r#""VEC""#),
            "uppercase embedding column must be quoted"
        );
        assert!(
            query.contains(r#"FROM "UPPER_KS"."UPPER_TBL""#),
            "uppercase keyspace/table must be quoted"
        );
    }

    #[test]
    fn range_scan_query_quotes_special_character_identifiers() {
        let pk_list = [CqlIdentifier::new(":pk"), CqlIdentifier::new(":sk")]
            .iter()
            .join(", ");
        let backend = CqlBackend {
            target_column: ColumnName::from("my-vector"),
        };
        let query = backend.range_scan_query(
            &CqlIdentifier::new("alternator_my-app"),
            &CqlIdentifier::new("my-table:v1"),
            &pk_list,
            &CqlIdentifier::new(":pk").to_string(),
        );
        assert!(
            query.contains(r#""my-vector""#),
            "hyphenated embedding column must be quoted"
        );
        assert!(
            query.contains(r#"FROM "alternator_my-app"."my-table:v1""#),
            "special-character keyspace/table must be quoted"
        );
        assert!(
            query.contains(r#"token(":pk")"#),
            "special-character partition key must be quoted"
        );
    }

    #[test]
    fn alternator_range_scan_query_basic() {
        let pk_list = [CqlIdentifier::new(":pk"), CqlIdentifier::new(":sk")]
            .iter()
            .join(", ");
        let backend = AlternatorBackend {
            target_column: ColumnName::from("v"),
        };
        let query = backend.range_scan_query(
            &CqlIdentifier::new("alternator_my-app"),
            &CqlIdentifier::new("my-table"),
            &pk_list,
            &CqlIdentifier::new(":pk").to_string(),
        );
        assert!(
            query.contains(r#"":attrs"['v']"#),
            "attribute name must be single-quoted inside :attrs map access: {query}"
        );
        assert!(
            query.contains(r#"writetime(":attrs"['v'])"#),
            "writetime must wrap the same :attrs map access: {query}"
        );
        assert!(
            query.contains(r#"FROM "alternator_my-app"."my-table""#),
            "keyspace and table must be double-quoted: {query}"
        );
        assert!(
            query.contains(r#"token(":pk")"#),
            "partition key must be double-quoted: {query}"
        );
    }

    #[test]
    fn alternator_range_scan_query_special_attribute_name() {
        let pk_list = CqlIdentifier::new(":pk").to_string();
        let backend = AlternatorBackend {
            target_column: ColumnName::from("my-vector:v1"),
        };
        let query = backend.range_scan_query(
            &CqlIdentifier::new("ks"),
            &CqlIdentifier::new("tbl"),
            &pk_list,
            &pk_list,
        );
        assert!(
            query.contains(r#"":attrs"['my-vector:v1']"#),
            "special characters in attribute name must appear verbatim inside single quotes: {query}"
        );
        assert!(
            query.contains(r#"writetime(":attrs"['my-vector:v1'])"#),
            "writetime must use the same single-quoted attribute access: {query}"
        );
    }

    #[test]
    fn alternator_range_scan_query_mixed_case_attribute() {
        let pk_list = CqlIdentifier::new("pk").to_string();
        let backend = AlternatorBackend {
            target_column: ColumnName::from("EmbeddingCol"),
        };
        let query = backend.range_scan_query(
            &CqlIdentifier::new("Ks"),
            &CqlIdentifier::new("Tbl"),
            &pk_list,
            &pk_list,
        );
        assert!(
            query.contains(r#"":attrs"['EmbeddingCol']"#),
            "mixed-case attribute name must be preserved as-is inside single quotes: {query}"
        );
        assert!(
            query.contains(r#"FROM "Ks"."Tbl""#),
            "mixed-case keyspace/table must be double-quoted: {query}"
        );
    }

    #[test]
    fn alternator_range_scan_query_attribute_with_quotes() {
        let pk_list = CqlIdentifier::new(":pk").to_string();
        let backend = AlternatorBackend {
            target_column: ColumnName::from("it's a \"test\""),
        };
        let query = backend.range_scan_query(
            &CqlIdentifier::new("ks"),
            &CqlIdentifier::new("tbl"),
            &pk_list,
            &pk_list,
        );
        assert!(
            query.contains(r#"":attrs"['it''s a "test"']"#),
            "single quotes in attribute name must be escaped by doubling: {query}"
        );
        assert!(
            query.contains(r#"writetime(":attrs"['it''s a "test"'])"#),
            "writetime must use the same escaped attribute access: {query}"
        );
    }
}
