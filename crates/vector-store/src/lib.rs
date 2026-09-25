/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

mod alternator;
mod async_in_progress;
mod config_manager;
mod cql_types;
pub mod db;
mod db_cdc;
pub mod db_index;
mod db_index_backend;
mod db_value;
mod distance;
mod engine;
mod file_monitor;
mod fts_index;
mod httproutes;
mod httpserver;
mod index_key;
mod indexes;
mod info;
mod internals;
mod invariant_key;
mod memory;
mod metrics;
mod monitor_indexes;
mod monitor_items;
pub mod node_state;
mod nonempty;
mod partition_key;
mod perf;
mod primary_key;
mod similarity;
mod substring_index;
mod table;
mod tantivy_common;
mod timestamp;
pub mod tls;
mod vector;
mod vs_index;
mod worker;

pub use crate::async_in_progress::AsyncInProgress;
pub use crate::config_manager::ConfigManager;
pub use crate::config_manager::ConfigReceivers;
pub use crate::config_manager::HttpServerConfig;
pub use crate::config_manager::load_config;
pub use crate::distance::Distance;
pub use crate::httpserver::HttpServer;
pub use crate::httpserver::HttpServerExt;
pub use crate::index_key::IndexKey;
use crate::indexes::Indexes;
pub use crate::info::Info;
use crate::metrics::Metrics;
use crate::node_state::NodeState;
pub use crate::nonempty::NonemptyArc;
pub use crate::nonempty::NonemptyBox;
pub use crate::nonempty::NonemptyIteratorExt;
pub use crate::partition_key::PartitionKey;
pub use crate::primary_key::PrimaryKey;
pub use crate::similarity::SimilarityScore;
pub use crate::table::PartitionId;
pub use crate::table::PrimaryId;
pub use crate::timestamp::Timestamp;
pub use crate::timestamp::Timestamped;
pub use crate::vector::Vector;
use crate::vs_index::VsIndexFactory;
use db::Db;
use scylla::cluster::metadata::ColumnType;
use scylla::cluster::metadata::NativeType;
use scylla::serialize::SerializationError;
use scylla::serialize::value::SerializeValue;
use scylla::serialize::writers::CellWriter;
use scylla::serialize::writers::WrittenCellProof;
use scylla::value::CqlValue;
use scylla_cdc::CqlIdentifier;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::hash::Hash;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;
use tokio::runtime::Builder;
use tokio::signal;
use tokio::sync::mpsc::Sender;
use utoipa::openapi::OpenApi;
use uuid::Uuid;

/// A CQL string literal that is always properly single-quoted when formatted
/// for use in CQL statements.
///
/// The inner value stores the already-quoted form of the string.
/// The [`Display`](std::fmt::Display) implementation outputs it in single quotes
/// with embedded single-quote characters escaped by doubling them
/// (`'` -> `''`), following the CQL grammar for string constants.
pub(crate) struct CqlLiteral {
    quoted: String,
}

impl CqlLiteral {
    /// Creates a new `CqlLiteral`, preserving the value exactly as given.
    ///
    /// The value will be single-quoted when formatted, with any embedded
    /// single quotes escaped by doubling.
    pub(crate) fn new(value: impl AsRef<str>) -> Self {
        let quoted = format!("'{}'", value.as_ref().replace('\'', "''"));
        Self { quoted }
    }
}

impl std::fmt::Display for CqlLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.quoted)
    }
}

pub(crate) struct KeyspaceIdentifier {
    cql_identifier: CqlIdentifier,
    is_alternator: bool,
}

impl<T: AsRef<str>> From<T> for KeyspaceIdentifier {
    fn from(value: T) -> Self {
        let value = value.as_ref();
        Self {
            cql_identifier: CqlIdentifier::new(value),
            is_alternator: value.starts_with("alternator_"),
        }
    }
}

impl KeyspaceIdentifier {
    pub(crate) fn is_alternator(&self) -> bool {
        self.is_alternator
    }
}

impl std::fmt::Display for KeyspaceIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cql_identifier.fmt(f)
    }
}

pub(crate) struct TableIdentifier {
    cql_identifier: CqlIdentifier,
}

impl<T: AsRef<str>> From<T> for TableIdentifier {
    fn from(value: T) -> Self {
        Self {
            cql_identifier: CqlIdentifier::new(value.as_ref()),
        }
    }
}

impl std::fmt::Display for TableIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cql_identifier.fmt(f)
    }
}

/// Which data provider a DiskANN index is built on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DiskannBackendKind {
    #[default]
    Inmem,
    Scylla,
}

impl FromStr for DiskannBackendKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "inmem" => Ok(Self::Inmem),
            "scylla" => Ok(Self::Scylla),
            _ => Err(anyhow::anyhow!("Unknown DiskANN backend: {s}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct DiskannAlpha(f32);

impl DiskannAlpha {
    pub fn new(value: f32) -> Result<Self, String> {
        if !value.is_finite() {
            Err(format!("DiskannAlpha must be finite, got {value}"))
        } else if value <= 0.0 {
            Err(format!("DiskannAlpha must be > 0, got {value}"))
        } else {
            Ok(Self(value))
        }
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub vector_store_addr: std::net::SocketAddr,
    pub scylladb_uri: String,
    pub threads: Option<usize>,
    pub memory_limit: Option<u64>,
    pub memory_usage_check_interval: Option<Duration>,
    pub opensearch_addr: Option<String>,
    pub credentials: Option<Credentials>,
    pub usearch_simulator: Option<Vec<Duration>>,
    pub diskann_alpha: Option<DiskannAlpha>,
    pub diskann_max_points: Option<NonZeroUsize>,
    pub diskann_backend: Option<DiskannBackendKind>,
    pub alter_index_simulator: bool,
    pub fulltext_indexes: bool,
    pub substring_indexes: bool,
    pub cql_connection_timeout: Option<Duration>,
    pub cql_keepalive_interval: Option<Duration>,
    pub cql_keepalive_timeout: Option<Duration>,
    pub cql_tcp_keepalive_interval: Option<Duration>,
    pub cql_uri_translation_map: Option<HashMap<SocketAddr, SocketAddr>>,
    pub cql_preferred_datacenter: Option<String>,
    pub cql_preferred_rack: Option<String>,
    pub cdc_safety_interval: Option<Duration>,
    pub cdc_sleep_interval: Option<Duration>,
    pub cdc_fine_safety_interval: Option<Duration>,
    pub cdc_fine_sleep_interval: Option<Duration>,
    pub monitor_indexes_interval: Option<Duration>,
    pub engine_status_update_interval: Option<Duration>,
    pub disable_colors: bool,
    pub tls_cert_path: Option<std::path::PathBuf>,
    pub tls_key_path: Option<std::path::PathBuf>,
    pub mtls_addr: SocketAddr,
    pub mtls_ca_cert_path: Option<std::path::PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vector_store_addr: "127.0.0.1:6080".parse().unwrap(),
            scylladb_uri: "127.0.0.1:9042".to_string(),
            threads: None,
            memory_limit: None,
            memory_usage_check_interval: None,
            opensearch_addr: None,
            credentials: None,
            usearch_simulator: None,
            diskann_alpha: None,
            diskann_max_points: None,
            diskann_backend: None,
            alter_index_simulator: false,
            fulltext_indexes: true,
            substring_indexes: true,
            disable_colors: false,
            tls_cert_path: None,
            tls_key_path: None,
            mtls_addr: "127.0.0.1:6081".parse().unwrap(),
            mtls_ca_cert_path: None,
            cql_connection_timeout: None,
            cql_keepalive_interval: None,
            cql_keepalive_timeout: None,
            cql_tcp_keepalive_interval: None,
            cql_uri_translation_map: None,
            cql_preferred_datacenter: None,
            cql_preferred_rack: None,
            cdc_safety_interval: None,
            cdc_sleep_interval: None,
            cdc_fine_safety_interval: None,
            cdc_fine_sleep_interval: None,
            monitor_indexes_interval: None,
            engine_status_update_interval: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Credentials {
    pub username: Option<String>,
    pub password: Option<secrecy::SecretString>,
    pub certificate_path: Option<std::path::PathBuf>,
}

#[derive(
    Clone,
    Debug,
    Eq,
    Hash,
    PartialEq,
    derive_more::AsRef,
    derive_more::Display,
    derive_more::From,
    derive_more::Into,
)]
#[from(String, &String, &str)]
#[as_ref(str)]
/// A keyspace name in a db.
pub struct KeyspaceName(String);

impl KeyspaceName {
    /// Returns true if this keyspace is backed by Alternator (DynamoDB-compatible API).
    /// Alternator keyspaces are prefixed with `alternator_`.
    fn is_alternator(&self) -> bool {
        self.0.starts_with("alternator_")
    }
}

impl SerializeValue for KeyspaceName {
    fn serialize<'b>(
        &self,
        typ: &ColumnType,
        writer: CellWriter<'b>,
    ) -> Result<WrittenCellProof<'b>, SerializationError> {
        <String as SerializeValue>::serialize(&self.0, typ, writer)
    }
}

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    derive_more::From,
    derive_more::AsRef,
    derive_more::Into,
    derive_more::Display,
)]
#[from(String, &String, &str)]
#[as_ref(str)]
/// A name of the vector index in a db.
pub struct IndexName(String);

impl SerializeValue for IndexName {
    fn serialize<'b>(
        &self,
        typ: &ColumnType,
        writer: CellWriter<'b>,
    ) -> Result<WrittenCellProof<'b>, SerializationError> {
        <String as SerializeValue>::serialize(&self.0, typ, writer)
    }
}

#[derive(
    Clone, Debug, PartialEq, Eq, Hash, derive_more::From, derive_more::AsRef, derive_more::Display,
)]
#[from(String, &String, &str)]
#[as_ref(str)]
/// A table name of the table with vectors in a db
pub struct TableName(String);

impl SerializeValue for TableName {
    fn serialize<'b>(
        &self,
        typ: &ColumnType,
        writer: CellWriter<'b>,
    ) -> Result<WrittenCellProof<'b>, SerializationError> {
        <String as SerializeValue>::serialize(&self.0, typ, writer)
    }
}

#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Ord,
    PartialOrd,
    derive_more::From,
    derive_more::Into,
    derive_more::AsRef,
    derive_more::Display,
)]
#[from(String, &String, &str)]
#[as_ref(str)]
/// Name of the column in a db table.
pub struct ColumnName(String);

impl SerializeValue for ColumnName {
    fn serialize<'b>(
        &self,
        typ: &ColumnType,
        writer: CellWriter<'b>,
    ) -> Result<WrittenCellProof<'b>, SerializationError> {
        <String as SerializeValue>::serialize(&self.0, typ, writer)
    }
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    derive_more::AsRef,
    derive_more::From,
    derive_more::Display,
    PartialOrd,
)]
/// Dimensions of embeddings
pub struct Dimensions(NonZeroUsize);

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    derive_more::AsRef,
    derive_more::From,
    derive_more::Display,
    derive_more::FromStr,
)]
/// Limit number of neighbors per graph node
pub struct Connectivity(usize);

impl Default for Connectivity {
    fn default() -> Self {
        Self(16)
    }
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    derive_more::AsRef,
    derive_more::From,
    derive_more::Display,
    derive_more::FromStr,
)]
/// Control the recall of indexing
pub struct ExpansionAdd(usize);

impl Default for ExpansionAdd {
    fn default() -> Self {
        Self(128)
    }
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    derive_more::AsRef,
    derive_more::From,
    derive_more::Display,
    derive_more::FromStr,
)]
/// Control the quality of the search
pub struct ExpansionSearch(usize);

impl Default for ExpansionSearch {
    fn default() -> Self {
        Self(64)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default, derive_more::From)]
pub enum SpaceType {
    Euclidean,
    #[default]
    Cosine,
    DotProduct,
    Hamming,
}

impl FromStr for SpaceType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "EUCLIDEAN" => Ok(Self::Euclidean),
            "COSINE" => Ok(Self::Cosine),
            "DOT_PRODUCT" => Ok(Self::DotProduct),
            "HAMMING" => Ok(Self::Hamming),
            _ => Err(anyhow::anyhow!("Unknown space type: {s}")),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
/// Represents the quantization type for vectors.
///
/// Quantization is a process that reduces the precision of floating-point numbers,
/// which can lead to significant memory savings.
pub enum Quantization {
    /// 32-bit single-precision IEEE 754 floating-point.
    #[default]
    F32,
    /// 16-bit standard half-precision floating-point (IEEE 754).
    F16,
    /// 16-bit "Brain" floating-point.
    BF16,
    /// 8-bit signed integer.
    I8,
    /// 1-bit binary value (packed 8 per byte).
    B1,
}

impl FromStr for Quantization {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "I8" => Ok(Self::I8),
            "B1" => Ok(Self::B1),
            _ => Err(anyhow::anyhow!("Unknown quantization type: {s}")),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default, derive_more::Display)]
/// A text analyzer a full-text index uses to split text into tokens.
pub enum Analyzer {
    #[default]
    #[display("standard")]
    Standard,
    #[display("english")]
    English,
    #[display("german")]
    German,
    #[display("french")]
    French,
    #[display("spanish")]
    Spanish,
    #[display("italian")]
    Italian,
    #[display("portuguese")]
    Portuguese,
    #[display("russian")]
    Russian,
    #[display("simple")]
    Simple,
    #[display("whitespace")]
    Whitespace,
}

impl FromStr for Analyzer {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "standard" => Ok(Self::Standard),
            "english" => Ok(Self::English),
            "german" => Ok(Self::German),
            "french" => Ok(Self::French),
            "spanish" => Ok(Self::Spanish),
            "italian" => Ok(Self::Italian),
            "portuguese" => Ok(Self::Portuguese),
            "russian" => Ok(Self::Russian),
            "simple" => Ok(Self::Simple),
            "whitespace" => Ok(Self::Whitespace),
            _ => Err(anyhow::anyhow!("Unknown analyzer: {s}")),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, derive_more::AsRef, derive_more::From)]
/// Whether a full-text index stores token positions, which are required by phrase queries.
pub struct Positions(bool);

impl Default for Positions {
    fn default() -> Self {
        Self(true)
    }
}

impl FromStr for Positions {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "true" => Ok(Self(true)),
            "false" => Ok(Self(false)),
            _ => Err(anyhow::anyhow!("Unknown positions value: {s}")),
        }
    }
}

/// Upper bound of `min_gram` and `max_gram` of a substring index. Every substring of a value up
/// to `max_gram` characters becomes an indexed term, so the bound keeps the index size sane.
pub const MAX_GRAM_LIMIT: usize = 8;

fn parse_gram(s: &str, name: &str) -> anyhow::Result<NonZeroUsize> {
    let value: usize = s
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("Unknown {name} value: {s}"))?;
    if value == 0 || value > MAX_GRAM_LIMIT {
        anyhow::bail!("{name} must be in 1..={MAX_GRAM_LIMIT}, got {value}");
    }
    Ok(NonZeroUsize::new(value).unwrap())
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, derive_more::AsRef, derive_more::From)]
/// Length in characters of the shortest term a substring index stores for a value.
/// A query shorter than this cannot be answered by the index.
pub struct MinGram(NonZeroUsize);

impl Default for MinGram {
    fn default() -> Self {
        Self(NonZeroUsize::new(1).unwrap())
    }
}

impl FromStr for MinGram {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_gram(s, "min_gram").map(Self)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, derive_more::AsRef, derive_more::From)]
/// Length in characters of the longest term a substring index stores for a value.
/// A query up to this length is a single exact term lookup; a longer one is answered by
/// intersecting its `max_gram`-long substrings and verifying the candidates.
pub struct MaxGram(NonZeroUsize);

impl Default for MaxGram {
    fn default() -> Self {
        Self(NonZeroUsize::new(3).unwrap())
    }
}

impl FromStr for MaxGram {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parse_gram(s, "max_gram").map(Self)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, derive_more::AsRef, derive_more::From)]
/// Whether a substring index matches letter case exactly (the CQL `LIKE` semantics) or
/// lowercases both the indexed values and the queries.
pub struct CaseSensitive(bool);

impl Default for CaseSensitive {
    fn default() -> Self {
        Self(true)
    }
}

impl FromStr for CaseSensitive {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "true" => Ok(Self(true)),
            "false" => Ok(Self(false)),
            _ => Err(anyhow::anyhow!("Unknown case_sensitive value: {s}")),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, derive_more::AsRef, derive_more::From)]
/// The column a substring index orders its results by, if it was given one.
///
/// Unset means the index answers in unspecified order, which is what lets a search stop at the
/// limit instead of examining every match. An index with a sort column pays for the ordering, so
/// this is opt-in rather than always on.
pub struct OrderBy(Option<ColumnName>);

impl FromStr for OrderBy {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let name = s.trim();
        if name.is_empty() {
            return Err(anyhow::anyhow!("order_by must name a column"));
        }
        Ok(Self(Some(ColumnName::from(name))))
    }
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, derive_more::AsRef, derive_more::From,
)]
/// Whether a substring index keeps the primary id as a `FAST` column as well as in the document
/// store. Off by default: it costs index size, and exists to measure what the store reads cost.
///
/// ScyllaDB knows this only as the placeholder `poc_option_1`, which it validates as non-empty and
/// hands through; the meaning lives here.
pub struct PrimaryIdFast(bool);

impl FromStr for PrimaryIdFast {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "on" | "yes" => Ok(Self(true)),
            "false" | "0" | "off" | "no" => Ok(Self(false)),
            _ => Err(anyhow::anyhow!("Unknown primary_id_fast value: {s}")),
        }
    }
}

#[derive(Clone, Copy, derive_more::AsRef, derive_more::Display, derive_more::From)]
/// Limit the number of search result
pub struct Limit(NonZeroUsize);

impl Default for Limit {
    fn default() -> Self {
        Self(NonZeroUsize::new(1).unwrap())
    }
}

/// A restriction provided in a CQL query for filtering ANN search results.
#[derive(Debug, PartialEq)]
pub enum Restriction {
    Eq {
        lhs: ColumnName,
        rhs: CqlValue,
    },
    In {
        lhs: ColumnName,
        rhs: Vec<CqlValue>,
    },
    Lt {
        lhs: ColumnName,
        rhs: CqlValue,
    },
    Lte {
        lhs: ColumnName,
        rhs: CqlValue,
    },
    Gt {
        lhs: ColumnName,
        rhs: CqlValue,
    },
    Gte {
        lhs: ColumnName,
        rhs: CqlValue,
    },
    EqTuple {
        lhs: Vec<ColumnName>,
        rhs: Vec<CqlValue>,
    },
    InTuple {
        lhs: Vec<ColumnName>,
        rhs: Vec<Vec<CqlValue>>,
    },
    LtTuple {
        lhs: Vec<ColumnName>,
        rhs: Vec<CqlValue>,
    },
    LteTuple {
        lhs: Vec<ColumnName>,
        rhs: Vec<CqlValue>,
    },
    GtTuple {
        lhs: Vec<ColumnName>,
        rhs: Vec<CqlValue>,
    },
    GteTuple {
        lhs: Vec<ColumnName>,
        rhs: Vec<CqlValue>,
    },
}

/// A filter to apply to an ANN search. It contains restrictions from a CQL query and a flag to
/// indicate whether ALLOW FILTERING was specified in the CQL query.
#[derive(Debug)]
pub struct Filter {
    pub restrictions: Vec<Restriction>,
    pub allow_filtering: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, derive_more::From)]
pub struct IndexVersion(Uuid);

impl IndexVersion {
    fn gregorian_ticks(&self) -> u64 {
        self.0
            .get_timestamp()
            .map(|ts| ts.to_gregorian().0)
            .unwrap_or(0)
    }
}

impl PartialOrd for IndexVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.gregorian_ticks().cmp(&other.gregorian_ticks())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Vector-search-specific index configuration.
pub struct IndexOptionsVs {
    pub dimensions: Dimensions,
    pub connectivity: Connectivity,
    pub expansion_add: ExpansionAdd,
    pub expansion_search: ExpansionSearch,
    pub space_type: SpaceType,
    pub quantization: Quantization,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
/// Full-text-search-specific index configuration.
pub struct IndexOptionsFts {
    pub analyzer: Analyzer,
    pub positions: Positions,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
/// Substring-search-specific index configuration.
///
/// Not `Copy`: `order_by` holds a column name. Clone it rather than reaching for a `*`.
pub struct IndexOptionsSubstring {
    pub min_gram: MinGram,
    pub max_gram: MaxGram,
    pub case_sensitive: CaseSensitive,
    /// The column results are ordered by, or none for unspecified order.
    pub order_by: OrderBy,
    /// Whether the primary id is also kept as a columnar (`FAST`) field, so that resolving a page
    /// never touches the document store. Read from the `poc_option_1` placeholder.
    pub primary_id_fast: PrimaryIdFast,
}

impl IndexOptionsSubstring {
    /// Checks the cross-option invariant `min_gram <= max_gram`. ScyllaDB rejects such an index
    /// at creation, so a violation here means the options were tampered with; fall back to the
    /// defaults with a warning, the same way an unparsable single option is handled.
    pub fn validated(self) -> Self {
        if self.min_gram.as_ref().get() > self.max_gram.as_ref().get() {
            tracing::warn!(
                "substring index options min_gram={} > max_gram={}, using the defaults",
                self.min_gram.as_ref(),
                self.max_gram.as_ref()
            );
            // Only the grams are in doubt, so only the grams are reset. Dropping the sort column
            // here would silently turn an ordered index into an unordered one, and the queries
            // would come back in the wrong order rather than failing.
            return Self {
                order_by: self.order_by,
                primary_id_fast: self.primary_id_fast,
                ..Self::default()
            };
        }
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Discriminates between vector-search, full-text-search and substring-search index.
pub enum IndexKind {
    Vs(IndexOptionsVs),
    Fts(IndexOptionsFts),
    Substring(IndexOptionsSubstring),
}

impl IndexKind {
    pub fn as_vs(&self) -> Option<&IndexOptionsVs> {
        match self {
            IndexKind::Vs(vs) => Some(vs),
            IndexKind::Fts(_) | IndexKind::Substring(_) => None,
        }
    }

    pub fn as_fts(&self) -> Option<&IndexOptionsFts> {
        match self {
            IndexKind::Fts(fts) => Some(fts),
            IndexKind::Vs(_) | IndexKind::Substring(_) => None,
        }
    }

    pub fn as_substring(&self) -> Option<&IndexOptionsSubstring> {
        match self {
            IndexKind::Substring(substring) => Some(substring),
            IndexKind::Vs(_) | IndexKind::Fts(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Information about an index
pub struct IndexMetadata {
    pub keyspace_name: KeyspaceName,
    pub index_name: IndexName,
    pub table_name: TableName,
    pub primary_key_columns: NonemptyArc<ColumnName>,
    pub partition_key_count: NonZeroUsize,
    pub target_columns: NonemptyArc<ColumnName>,
    pub partitioning: DbIndexPartitioning,
    pub filtering_columns: Arc<[ColumnName]>,
    /// For Alternator tables: the NativeType to decode each partition-key/
    /// filtering column with no real column in the table's CQL schema as
    /// (its value lives only in the ":attrs" map), derived from its
    /// declared DynamoDB type ("S", "N" or "B"). Empty for CQL-native
    /// tables and for columns that are real columns.
    pub alternator_attribute_types: Arc<BTreeMap<ColumnName, NativeType>>,
    pub version: IndexVersion,
    pub kind: IndexKind,
}

// Manual impl because `NativeType` (in `alternator_attribute_types`) doesn't
// implement `Hash`. Its variants are all fieldless, so hashing by
// `mem::discriminant` is exactly as precise as hashing the value itself.
impl std::hash::Hash for IndexMetadata {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.keyspace_name.hash(state);
        self.index_name.hash(state);
        self.table_name.hash(state);
        self.primary_key_columns.hash(state);
        self.partition_key_count.hash(state);
        self.target_columns.hash(state);
        self.partitioning.hash(state);
        self.filtering_columns.hash(state);
        self.alternator_attribute_types.len().hash(state);
        for (name, native_type) in self.alternator_attribute_types.iter() {
            name.hash(state);
            std::mem::discriminant(native_type).hash(state);
        }
        self.version.hash(state);
        self.kind.hash(state);
    }
}

impl IndexMetadata {
    pub fn key(&self) -> IndexKey {
        IndexKey::new(&self.keyspace_name, &self.index_name)
    }

    pub fn vs(&self) -> Option<&IndexOptionsVs> {
        self.kind.as_vs()
    }

    pub fn fts(&self) -> Option<&IndexOptionsFts> {
        self.kind.as_fts()
    }

    pub fn substring(&self) -> Option<&IndexOptionsSubstring> {
        self.kind.as_substring()
    }

    /// The NativeType to treat `column` as, if it's a virtual Alternator
    /// attribute (see alternator_attribute_types) - None otherwise.
    pub fn alternator_native_type(&self, column: &ColumnName) -> Option<NativeType> {
        self.alternator_attribute_types.get(column).cloned()
    }

    fn discard_version(&self) -> Self {
        let mut copy = self.clone();
        copy.version = IndexVersion(Uuid::nil());
        copy
    }

    fn nonpk_partition_key_columns(&self) -> Option<impl Iterator<Item = &ColumnName>> {
        match &self.partitioning {
            DbIndexPartitioning::Global => None,
            DbIndexPartitioning::Local(pk_columns) => Some(
                pk_columns
                    .iter()
                    .filter(|col| !self.primary_key_columns.contains(col)),
            ),
        }
    }

    /// The subset of `filtering_columns` not already in `primary_key_columns`.
    /// Table::new() stores a primary-key column as Column::PrimaryKey, not a
    /// value slot, so it can't also be fetched/stored as a filtering column.
    fn nonpk_filtering_columns(&self) -> impl Iterator<Item = &ColumnName> {
        self.filtering_columns
            .iter()
            .filter(|col| !self.primary_key_columns.contains(col))
    }

    /// Every column whose value has to be fetched and kept alongside the indexed one: the filtering
    /// columns, plus a substring index's sort column.
    ///
    /// Three places have to agree on this set -- the full-scan SELECT, the CDC re-SELECT and the
    /// table's column slots -- and a disagreement shows up as a row count mismatch far from the
    /// cause, so they all go through here rather than each assembling their own list.
    pub(crate) fn ingested_value_columns(&self) -> impl Iterator<Item = &ColumnName> {
        let sort_column = self
            .kind
            .as_substring()
            .and_then(|options| options.order_by.as_ref().as_ref())
            .filter(|col| {
                !self.primary_key_columns.contains(col) && !self.filtering_columns.contains(col)
            });
        self.nonpk_filtering_columns().chain(sort_column)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DbIndexPartitioning {
    Global,
    Local(NonemptyArc<ColumnName>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// The kind of custom index as declared in ScyllaDB index options.
pub enum DbIndexKind {
    VectorSearch,
    FullTextSearch,
    Substring,
}

#[derive(Debug)]
pub struct DbCustomIndex {
    pub keyspace: KeyspaceName,
    pub index: IndexName,
    pub table: TableName,
    pub primary_key_columns: NonemptyArc<ColumnName>,
    pub partition_key_count: NonZeroUsize,
    pub target_columns: NonemptyArc<ColumnName>,
    pub partitioning: DbIndexPartitioning,
    pub filtering_columns: Arc<[ColumnName]>,
    /// See IndexMetadata::alternator_attribute_types.
    pub alternator_attribute_types: Arc<BTreeMap<ColumnName, NativeType>>,
    pub kind: DbIndexKind,
}

impl DbCustomIndex {
    pub fn key(&self) -> IndexKey {
        IndexKey::new(&self.keyspace, &self.index)
    }
}

#[derive(Clone, Debug, PartialEq)]
/// The indexed value read from a CDC row or full scan.
pub enum DbIndexedValue {
    Vector(Vector),
    Document(String),
    Filtering(CqlValue),
}

/// The operation read from a CDC row or full scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DbIndexedOperation {
    Upsert(NonemptyBox<Timestamped<DbIndexedValue>>),
    Delete(Timestamp),
}

/// A row read from a CDC row or full scan, containing the primary key and the operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DbIndexedRow {
    pub primary_key: PrimaryKey,
    pub operation: DbIndexedOperation,
}

pub fn block_on<Output>(threads: Option<usize>, f: impl AsyncFnOnce() -> Output) -> Output {
    let mut builder = match threads {
        Some(0) | None => Builder::new_multi_thread(),
        Some(1) => Builder::new_current_thread(),
        Some(threads) => {
            let mut builder = Builder::new_multi_thread();
            builder.worker_threads(threads);
            builder
        }
    };
    builder
        .enable_all()
        .build()
        .unwrap()
        .block_on(async move { f().await })
}

pub async fn run(
    node_state: Option<Sender<NodeState>>,
    db_actor: Option<Sender<Db>>,
    config_receivers: ConfigReceivers,
) -> anyhow::Result<(Sender<HttpServer>, Sender<HttpServer>)> {
    let node_state = if let Some(node_state) = node_state {
        node_state
    } else {
        new_node_state().await
    };

    let config_rx = config_receivers.config.clone();
    let opensearch_addr = config_rx.borrow().opensearch_addr.clone();
    let diskann_backend = config_rx.borrow().diskann_backend;

    let internals = internals::new();
    let memory = memory::new(internals.clone(), config_rx.clone());
    let worker = worker::new();

    let vs_index_factory = if let Some(addr) = opensearch_addr {
        tracing::info!("Using OpenSearch index factory at {addr}");
        vs_index::new_index_factory_opensearch(addr, config_rx.clone())?
    } else if let Some(backend) = diskann_backend {
        tracing::info!("Using DiskANN index factory with the {backend:?} backend");
        vs_index::new_index_factory_diskann(config_rx.clone(), worker.clone(), memory.clone())?
    } else {
        tracing::info!("Using Usearch index factory");
        vs_index::new_index_factory_usearch(config_rx.clone(), worker.clone(), memory.clone())?
    };

    let metrics = Arc::new(Metrics::new());
    let db_actor = if let Some(db_actor) = db_actor {
        db_actor
    } else {
        db::new(
            node_state.clone(),
            internals.clone(),
            config_rx,
            Arc::clone(&metrics),
        )
        .await?
    };

    let index_engine_version = vs_index_factory.index_engine_version();
    let indexes = Arc::new(RwLock::new(Indexes::new()));
    let fts_index_factory =
        fts_index::new_fts_index_factory_tantivy(worker.clone(), memory.clone());
    let substring_index_factory =
        substring_index::new_substring_index_factory_tantivy(worker, memory);
    let engine = engine::new(
        db_actor,
        engine::IndexFactories {
            vs: vs_index_factory,
            fts: fts_index_factory,
            substring: substring_index_factory,
        },
        node_state.clone(),
        metrics.clone(),
        Arc::clone(&indexes),
        config_receivers.config,
    )
    .await?;

    let main = httpserver::new(
        node_state.clone(),
        Arc::clone(&indexes),
        engine.clone(),
        metrics.clone(),
        internals.clone(),
        index_engine_version.clone(),
        config_receivers.http,
    )
    .await?;

    let mtls = httpserver::new(
        node_state,
        indexes,
        engine,
        metrics,
        internals,
        index_engine_version,
        config_receivers.mtls_http,
    )
    .await?;

    Ok((main, mtls))
}

pub async fn new_node_state() -> Sender<NodeState> {
    node_state::new().await
}

pub fn openapi() -> OpenApi {
    httproutes::api()
}

pub async fn wait_for_shutdown() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await
    };
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Percentage {
    value: f64,
}

impl Percentage {
    pub fn get(&self) -> f64 {
        self.value
    }
}

impl TryFrom<f64> for Percentage {
    type Error = String;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if !(0.0..=100.0).contains(&value) {
            Err(format!(
                "Percentage must be between 0 and 100, got: {value}"
            ))
        } else {
            Ok(Self { value })
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Progress {
    Done,
    InProgress(Percentage),
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_percentage_from_f64() {
        assert_eq!(Percentage::try_from(50.0).unwrap().get(), 50.0);
        assert!(Percentage::try_from(-1.0).is_err());
        assert!(Percentage::try_from(101.0).is_err());
        assert!(Percentage::try_from(0.0).is_ok());
        assert!(Percentage::try_from(100.0).is_ok());
    }

    #[test]
    fn analyzer_parses_every_value_scylladb_accepts() {
        for (text, expected) in [
            ("standard", Analyzer::Standard),
            ("english", Analyzer::English),
            ("german", Analyzer::German),
            ("french", Analyzer::French),
            ("spanish", Analyzer::Spanish),
            ("italian", Analyzer::Italian),
            ("portuguese", Analyzer::Portuguese),
            ("russian", Analyzer::Russian),
            ("simple", Analyzer::Simple),
            ("whitespace", Analyzer::Whitespace),
        ] {
            assert_eq!(text.parse::<Analyzer>().unwrap(), expected);
            assert_eq!(text.to_uppercase().parse::<Analyzer>().unwrap(), expected);
            assert_eq!(expected.to_string(), text);
        }
    }

    #[test]
    fn analyzer_rejects_unknown_value() {
        assert!("klingon".parse::<Analyzer>().is_err());
    }

    fn index_metadata_with_alternator_types(
        types: impl IntoIterator<Item = (&'static str, NativeType)>,
    ) -> IndexMetadata {
        IndexMetadata {
            keyspace_name: "alternator_ks".into(),
            index_name: "idx".into(),
            table_name: "tbl".into(),
            primary_key_columns: NonemptyArc::new(["pk"]).unwrap(),
            partition_key_count: NonZeroUsize::new(1).unwrap(),
            target_columns: NonemptyArc::new(["embedding"]).unwrap(),
            partitioning: DbIndexPartitioning::Global,
            filtering_columns: Arc::new([]),
            alternator_attribute_types: Arc::new(
                types
                    .into_iter()
                    .map(|(name, typ)| (ColumnName::from(name), typ))
                    .collect(),
            ),
            version: IndexVersion(Uuid::new_v4()),
            kind: IndexKind::Vs(IndexOptionsVs {
                dimensions: NonZeroUsize::new(3).unwrap().into(),
                connectivity: Default::default(),
                expansion_add: Default::default(),
                expansion_search: Default::default(),
                space_type: SpaceType::Euclidean,
                quantization: Default::default(),
            }),
        }
    }

    #[test]
    fn alternator_native_type_none_for_unknown_column() {
        let metadata = index_metadata_with_alternator_types([("color", NativeType::Text)]);
        assert_eq!(metadata.alternator_native_type(&"size".into()), None);
    }

    #[test]
    fn alternator_native_type_returns_declared_type() {
        let metadata = index_metadata_with_alternator_types([("color", NativeType::Text)]);
        assert_eq!(
            metadata.alternator_native_type(&"color".into()),
            Some(NativeType::Text)
        );
    }

    #[test]
    fn analyzer_defaults_to_standard() {
        assert_eq!(Analyzer::default(), Analyzer::Standard);
    }

    #[test]
    fn positions_parses_booleans_case_insensitively() {
        assert!(*"true".parse::<Positions>().unwrap().as_ref());
        assert!(*"TRUE".parse::<Positions>().unwrap().as_ref());
        assert!(!*"false".parse::<Positions>().unwrap().as_ref());
        assert!(!*"False".parse::<Positions>().unwrap().as_ref());
        assert!("yes".parse::<Positions>().is_err());
    }

    #[test]
    fn positions_defaults_to_enabled() {
        assert!(*Positions::default().as_ref());
    }

    #[test]
    fn grams_parse_within_bounds_only() {
        assert_eq!("3".parse::<MinGram>().unwrap().as_ref().get(), 3);
        assert_eq!(" 8 ".parse::<MaxGram>().unwrap().as_ref().get(), 8);
        assert!("0".parse::<MinGram>().is_err());
        assert!("9".parse::<MaxGram>().is_err());
        assert!("three".parse::<MaxGram>().is_err());
    }

    #[test]
    fn grams_default_to_one_and_three() {
        assert_eq!(MinGram::default().as_ref().get(), 1);
        assert_eq!(MaxGram::default().as_ref().get(), 3);
    }

    #[test]
    fn case_sensitive_parses_booleans_and_defaults_to_true() {
        assert!(*"true".parse::<CaseSensitive>().unwrap().as_ref());
        assert!(!*"FALSE".parse::<CaseSensitive>().unwrap().as_ref());
        assert!("yes".parse::<CaseSensitive>().is_err());
        assert!(*CaseSensitive::default().as_ref());
    }

    #[test]
    fn substring_options_fall_back_to_defaults_when_min_exceeds_max() {
        let inverted = IndexOptionsSubstring {
            min_gram: "4".parse().unwrap(),
            max_gram: "3".parse().unwrap(),
            case_sensitive: CaseSensitive::from(false),
            order_by: OrderBy::default(),
            primary_id_fast: PrimaryIdFast::default(),
        };
        assert_eq!(inverted.validated(), IndexOptionsSubstring::default());

        // The sort column survives a gram fallback: dropping it would answer in the wrong order
        // rather than failing, which is far harder to notice.
        let inverted_ordered = IndexOptionsSubstring {
            min_gram: "4".parse().unwrap(),
            max_gram: "3".parse().unwrap(),
            case_sensitive: CaseSensitive::from(false),
            order_by: "registered_at".parse().unwrap(),
            primary_id_fast: PrimaryIdFast::from(true),
        };
        assert_eq!(
            inverted_ordered.validated(),
            IndexOptionsSubstring {
                order_by: "registered_at".parse().unwrap(),
                primary_id_fast: PrimaryIdFast::from(true),
                ..IndexOptionsSubstring::default()
            }
        );

        let valid = IndexOptionsSubstring {
            min_gram: "2".parse().unwrap(),
            max_gram: "2".parse().unwrap(),
            case_sensitive: CaseSensitive::from(false),
            order_by: OrderBy::default(),
            primary_id_fast: PrimaryIdFast::default(),
        };
        assert_eq!(valid.clone().validated(), valid);

        assert!(*"TRUE".parse::<PrimaryIdFast>().unwrap().as_ref());
        assert!(!*"off".parse::<PrimaryIdFast>().unwrap().as_ref());
        assert!("maybe".parse::<PrimaryIdFast>().is_err());
    }

    #[test]
    fn nonpk_filtering_columns_excludes_primary_key_columns() {
        let metadata = IndexMetadata {
            keyspace_name: "ks".into(),
            index_name: "idx".into(),
            table_name: "tbl".into(),
            // "pk" is the partition key, "ck" a clustering key.
            primary_key_columns: NonemptyArc::new(["pk", "ck"]).unwrap(),
            partition_key_count: NonZeroUsize::new(1).unwrap(),
            target_columns: NonemptyArc::new(["embedding"]).unwrap(),
            partitioning: DbIndexPartitioning::Local(NonemptyArc::new(["pk"]).unwrap()),
            // "ck" is also a primary-key column; "f" is a genuine value column.
            filtering_columns: Arc::new(["ck".into(), "f".into()]),
            alternator_attribute_types: Arc::new(BTreeMap::new()),
            version: Uuid::new_v4().into(),
            kind: IndexKind::Vs(IndexOptionsVs {
                dimensions: Dimensions(NonZeroUsize::new(3).unwrap()),
                connectivity: Default::default(),
                expansion_add: Default::default(),
                expansion_search: Default::default(),
                space_type: Default::default(),
                quantization: Default::default(),
            }),
        };

        let filtering_columns: Vec<_> = metadata.nonpk_filtering_columns().cloned().collect();
        assert_eq!(filtering_columns, vec!["f".into()]);
    }

    fn substring_metadata(
        filtering_columns: Arc<[ColumnName]>,
        order_by: OrderBy,
    ) -> IndexMetadata {
        IndexMetadata {
            keyspace_name: "ks".into(),
            index_name: "idx".into(),
            table_name: "tbl".into(),
            primary_key_columns: NonemptyArc::new(["pk"]).unwrap(),
            partition_key_count: NonZeroUsize::new(1).unwrap(),
            target_columns: NonemptyArc::new(["nickname"]).unwrap(),
            partitioning: DbIndexPartitioning::Global,
            filtering_columns,
            alternator_attribute_types: Arc::new(BTreeMap::new()),
            version: Uuid::new_v4().into(),
            kind: IndexKind::Substring(IndexOptionsSubstring {
                order_by,
                ..IndexOptionsSubstring::default()
            }),
        }
    }

    #[test]
    fn a_sort_column_joins_the_columns_whose_value_is_ingested() {
        let metadata = substring_metadata(Arc::new([]), "registered_at".parse().unwrap());
        let columns: Vec<_> = metadata.ingested_value_columns().cloned().collect();
        assert_eq!(columns, vec!["registered_at".into()]);
    }

    #[test]
    fn without_a_sort_column_nothing_is_ingested_beyond_the_filtering_ones() {
        let metadata = substring_metadata(Arc::new(["f".into()]), OrderBy::default());
        let columns: Vec<_> = metadata.ingested_value_columns().cloned().collect();
        assert_eq!(columns, vec!["f".into()]);
    }

    /// The three sites that build this list feed `Table::new()`'s column slots and the SELECT that
    /// fills them. A column listed twice would be fetched twice and throw the row-length check off
    /// far from the cause.
    #[test]
    fn a_sort_column_is_not_ingested_twice() {
        let already_filtering = substring_metadata(Arc::new(["f".into()]), "f".parse().unwrap());
        let columns: Vec<_> = already_filtering
            .ingested_value_columns()
            .cloned()
            .collect();
        assert_eq!(columns, vec!["f".into()]);

        // A primary-key column is stored as a key offset rather than a value slot, so it must not
        // be fetched as one either.
        let primary_key = substring_metadata(Arc::new([]), "pk".parse().unwrap());
        assert_eq!(primary_key.ingested_value_columns().count(), 0);
    }

    #[test]
    fn order_by_names_a_column_or_nothing() {
        assert_eq!(
            *"registered_at".parse::<OrderBy>().unwrap().as_ref(),
            Some("registered_at".into())
        );
        // Trimmed, because CQL options come through as free text.
        assert_eq!(
            *"  registered_at  ".parse::<OrderBy>().unwrap().as_ref(),
            Some("registered_at".into())
        );
        assert!("".parse::<OrderBy>().is_err());
        assert!("   ".parse::<OrderBy>().is_err());
        assert_eq!(*OrderBy::default().as_ref(), None);
    }
}
