/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! The order a pattern index answers in, and the parts of it that are not plain integers.
//!
//! An ordered index keeps its rows by a tuple: the declared ordering columns, each as an
//! order-preserving `u64` (complemented for a `DESC` column), then the row's token, then its
//! primary key in table order. Everything up to the token is read from fast columns and compared
//! as integers; the primary key is compared only when all of that is equal, which happens for rows
//! of one partition with equal ordering values, or on a token collision. Table order -- the order
//! of a scan of the table -- is token, then partition key, then clustering key in the schema's
//! clustering order; this module reproduces ScyllaDB's comparators for the key types it supports.

use crate::MAX_ORDER_COLUMNS;
use crate::PrimaryKey;
use scylla::cluster::metadata::ColumnType;
use scylla::cluster::metadata::NativeType;
use scylla::routing::partitioner::Murmur3Partitioner;
use scylla::routing::partitioner::Partitioner;
use scylla::routing::partitioner::PartitionerHasher;
use scylla::value::CqlValue;
use std::cmp::Ordering;
use std::net::IpAddr;
use std::sync::Arc;

/// Slots in a [`SortPrefix`]: every ordering column, and the token.
pub(crate) const PREFIX_SLOTS: usize = MAX_ORDER_COLUMNS + 1;

/// The integer part of a row's place in its index's order: the ordering columns, then the token,
/// in the index's own (ascending) space. Slots past the index's columns are zero for every row,
/// so whole prefixes compare correctly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SortPrefix(pub(crate) [u64; PREFIX_SLOTS]);

impl SortPrefix {
    /// Every slot complemented: the same prefix in the reverse order.
    pub(crate) fn complement(self) -> Self {
        Self(self.0.map(|slot| !slot))
    }

    /// The first two slots as one number, high slot first: what segment bounds, the merge policy
    /// and the rewrite work with. For one ordering column that is the column and the token, so a
    /// run of rows sharing a value can still be split; for table order, the token.
    pub(crate) fn layout_key(&self) -> u128 {
        (u128::from(self.0[0]) << 64) | u128::from(self.0[1])
    }
}

/// A token as a sort key: biased so that the signed order of tokens is the unsigned order of keys.
pub(crate) fn token_key(token: i64) -> u64 {
    (token as u64) ^ (1 << 63)
}

/// How an index's primary keys are laid out, for table order and tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyLayout {
    /// The first this many primary-key columns are the partition key.
    pub(crate) partition_key_count: usize,
    /// For each clustering column, in key order, whether the schema says `DESC`.
    pub(crate) clustering_descending: Arc<[bool]>,
}

impl KeyLayout {
    /// The layout of a key whose first `partition_key_count` columns are the partition key,
    /// followed by clustering columns declared `DESC` where `clustering_descending` says so.
    pub fn new(partition_key_count: usize, clustering_descending: impl Into<Arc<[bool]>>) -> Self {
        Self {
            partition_key_count,
            clustering_descending: clustering_descending.into(),
        }
    }

    /// The partition key in ScyllaDB's legacy form: a single column's value, or each column as a
    /// 2-byte length, the value and a zero byte. It is what the token is hashed from and what
    /// orders two partitions that share a token. None for a value of a type the module does not
    /// serialize, which index creation refuses.
    pub(crate) fn partition_bytes(&self, key: &PrimaryKey) -> Option<Vec<u8>> {
        if self.partition_key_count == 1 {
            return serialize_value(&key.get(0)?);
        }
        let mut bytes = Vec::new();
        for idx in 0..self.partition_key_count {
            let value = serialize_value(&key.get(idx)?)?;
            bytes.extend_from_slice(&u16::try_from(value.len()).ok()?.to_be_bytes());
            bytes.extend_from_slice(&value);
            bytes.push(0);
        }
        Some(bytes)
    }

    /// The row's Murmur3 token, as a sort key.
    pub(crate) fn token_key(&self, key: &PrimaryKey) -> Option<u64> {
        let bytes = self.partition_bytes(key)?;
        let mut hasher = Murmur3Partitioner.build_hasher();
        hasher.write(&bytes);
        Some(token_key(hasher.finish().value()))
    }

    /// Table order of two rows already known to share a token: partition key bytes, then the
    /// clustering columns, each by its type's order and reversed where declared `DESC`.
    pub(crate) fn compare_same_token(&self, a: &PrimaryKey, b: &PrimaryKey) -> Ordering {
        let partitions = self.partition_bytes(a).cmp(&self.partition_bytes(b));
        if partitions != Ordering::Equal {
            return partitions;
        }
        for (offset, descending) in self.clustering_descending.iter().enumerate() {
            let idx = self.partition_key_count + offset;
            let order = match (a.get(idx), b.get(idx)) {
                (Some(a), Some(b)) => compare_values(&a, &b),
                (a, b) => a.is_some().cmp(&b.is_some()),
            };
            let order = if *descending { order.reverse() } else { order };
            if order != Ordering::Equal {
                return order;
            }
        }
        Ordering::Equal
    }
}

/// Whether a primary-key column of this type can take part in table order: the types whose order
/// [`compare_values`] reproduces and whose serialization [`serialize_value`] knows.
pub(crate) fn is_key_type_supported(typ: &ColumnType) -> bool {
    match typ {
        ColumnType::Native(native) => matches!(
            native,
            NativeType::TinyInt
                | NativeType::SmallInt
                | NativeType::Int
                | NativeType::BigInt
                | NativeType::Varint
                | NativeType::Text
                | NativeType::Ascii
                | NativeType::Blob
                | NativeType::Boolean
                | NativeType::Timestamp
                | NativeType::Date
                | NativeType::Time
                | NativeType::Uuid
                | NativeType::Timeuuid
                | NativeType::Inet
                | NativeType::Float
                | NativeType::Double
        ),
        ColumnType::Tuple(elements) => elements.iter().all(is_key_type_supported),
        _ => false,
    }
}

/// A value's CQL serialization, for the types table order supports.
pub(crate) fn serialize_value(value: &CqlValue) -> Option<Vec<u8>> {
    Some(match value {
        CqlValue::Empty => Vec::new(),
        CqlValue::TinyInt(v) => v.to_be_bytes().to_vec(),
        CqlValue::SmallInt(v) => v.to_be_bytes().to_vec(),
        CqlValue::Int(v) => v.to_be_bytes().to_vec(),
        CqlValue::BigInt(v) => v.to_be_bytes().to_vec(),
        CqlValue::Varint(v) => v.as_signed_bytes_be_slice().to_vec(),
        CqlValue::Text(v) | CqlValue::Ascii(v) => v.as_bytes().to_vec(),
        CqlValue::Blob(v) => v.clone(),
        CqlValue::Boolean(v) => vec![u8::from(*v)],
        CqlValue::Timestamp(v) => v.0.to_be_bytes().to_vec(),
        CqlValue::Date(v) => v.0.to_be_bytes().to_vec(),
        CqlValue::Time(v) => v.0.to_be_bytes().to_vec(),
        CqlValue::Uuid(v) => v.as_bytes().to_vec(),
        CqlValue::Timeuuid(v) => v.as_bytes().to_vec(),
        CqlValue::Inet(IpAddr::V4(v)) => v.octets().to_vec(),
        CqlValue::Inet(IpAddr::V6(v)) => v.octets().to_vec(),
        CqlValue::Float(v) => v.to_be_bytes().to_vec(),
        CqlValue::Double(v) => v.to_be_bytes().to_vec(),
        CqlValue::Tuple(elements) => {
            let mut bytes = Vec::new();
            for element in elements {
                match element {
                    None => bytes.extend_from_slice(&(-1i32).to_be_bytes()),
                    Some(element) => {
                        let element = serialize_value(element)?;
                        bytes.extend_from_slice(&i32::try_from(element.len()).ok()?.to_be_bytes());
                        bytes.extend_from_slice(&element);
                    }
                }
            }
            bytes
        }
        _ => return None,
    })
}

/// ScyllaDB's order of two values of one type (`compare_visitor` in its types.cc), for the types
/// table order supports. An empty value sorts before any other.
pub(crate) fn compare_values(a: &CqlValue, b: &CqlValue) -> Ordering {
    match (a, b) {
        (CqlValue::Empty, CqlValue::Empty) => Ordering::Equal,
        (CqlValue::Empty, _) => Ordering::Less,
        (_, CqlValue::Empty) => Ordering::Greater,
        (CqlValue::TinyInt(a), CqlValue::TinyInt(b)) => a.cmp(b),
        (CqlValue::SmallInt(a), CqlValue::SmallInt(b)) => a.cmp(b),
        (CqlValue::Int(a), CqlValue::Int(b)) => a.cmp(b),
        (CqlValue::BigInt(a), CqlValue::BigInt(b)) => a.cmp(b),
        (CqlValue::Varint(a), CqlValue::Varint(b)) => {
            compare_varints(a.as_signed_bytes_be_slice(), b.as_signed_bytes_be_slice())
        }
        (CqlValue::Text(a) | CqlValue::Ascii(a), CqlValue::Text(b) | CqlValue::Ascii(b)) => {
            a.as_bytes().cmp(b.as_bytes())
        }
        (CqlValue::Blob(a), CqlValue::Blob(b)) => a.cmp(b),
        (CqlValue::Boolean(a), CqlValue::Boolean(b)) => a.cmp(b),
        (CqlValue::Timestamp(a), CqlValue::Timestamp(b)) => a.0.cmp(&b.0),
        (CqlValue::Date(a), CqlValue::Date(b)) => a.0.cmp(&b.0),
        (CqlValue::Time(a), CqlValue::Time(b)) => a.0.cmp(&b.0),
        (CqlValue::Uuid(a), CqlValue::Uuid(b)) => compare_uuids(a.as_bytes(), b.as_bytes()),
        (CqlValue::Timeuuid(a), CqlValue::Timeuuid(b)) => {
            compare_timeuuids(a.as_bytes(), b.as_bytes())
        }
        (CqlValue::Inet(_), CqlValue::Inet(_)) => serialize_value(a).cmp(&serialize_value(b)),
        (CqlValue::Float(a), CqlValue::Float(b)) => compare_floats(*a, *b),
        (CqlValue::Double(a), CqlValue::Double(b)) => compare_floats(*a, *b),
        (CqlValue::Tuple(a), CqlValue::Tuple(b)) => compare_tuples(a, b),
        // Values of one column are of one type; anything else is a bug elsewhere, and a fixed
        // answer keeps the order total.
        _ => serialize_value(a).cmp(&serialize_value(b)),
    }
}

/// Two's-complement big-endian integers of any length, by value.
fn compare_varints(a: &[u8], b: &[u8]) -> Ordering {
    let negative = |v: &[u8]| v.first().is_some_and(|byte| byte & 0x80 != 0);
    // Without redundant sign bytes, a longer number is further from zero.
    let trim = |v: &[u8]| -> Vec<u8> {
        let fill = if negative(v) { 0xff } else { 0x00 };
        let mut start = 0;
        while start + 1 < v.len() && v[start] == fill && (v[start + 1] & 0x80 == fill & 0x80) {
            start += 1;
        }
        v[start..].to_vec()
    };
    let (a, b) = (trim(a), trim(b));
    match (negative(&a), negative(&b)) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => a.len().cmp(&b.len()).then_with(|| a.cmp(&b)),
        (true, true) => b.len().cmp(&a.len()).then_with(|| a.cmp(&b)),
    }
}

/// The timestamp half of a time-based UUID, reordered so that it compares as an integer
/// (`timeuuid_read_msb`).
fn timeuuid_msb(b: &[u8; 16]) -> u64 {
    let byte = |i: usize| u64::from(b[i]);
    (byte(6) & 0xf) << 56
        | byte(7) << 48
        | byte(4) << 40
        | byte(5) << 32
        | byte(0) << 24
        | byte(1) << 16
        | byte(2) << 8
        | byte(3)
}

fn uuid_lsb(b: &[u8; 16]) -> u64 {
    u64::from_be_bytes(b[8..16].try_into().unwrap())
}

/// `timeuuid`: by time, then the rest compared as signed bytes, as Cassandra did.
fn compare_timeuuids(a: &[u8; 16], b: &[u8; 16]) -> Ordering {
    let signed_lsb = |b: &[u8; 16]| uuid_lsb(b) ^ 0x8080_8080_8080_8080;
    timeuuid_msb(a)
        .cmp(&timeuuid_msb(b))
        .then_with(|| signed_lsb(a).cmp(&signed_lsb(b)))
}

/// `uuid`: by version first; two time-based ones by time, then the rest unsigned; any other two
/// as bytes.
fn compare_uuids(a: &[u8; 16], b: &[u8; 16]) -> Ordering {
    let version = |b: &[u8; 16]| (b[6] >> 4) & 0x0f;
    version(a).cmp(&version(b)).then_with(|| {
        if version(a) == 1 {
            timeuuid_msb(a)
                .cmp(&timeuuid_msb(b))
                .then_with(|| uuid_lsb(a).cmp(&uuid_lsb(b)))
        } else {
            a.cmp(b)
        }
    })
}

/// By value, with every NaN equal and above everything else, as Java's order has it.
fn compare_floats<F: PartialOrd>(a: F, b: F) -> Ordering {
    #[allow(clippy::eq_op)]
    let nan = |v: &F| v != v;
    match (nan(&a), nan(&b)) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// Element by element, a null before any value; a tuple that only adds trailing nulls is equal.
fn compare_tuples(a: &[Option<CqlValue>], b: &[Option<CqlValue>]) -> Ordering {
    for (a, b) in a.iter().zip(b) {
        let order = match (a, b) {
            (Some(a), Some(b)) => compare_values(a, b),
            (a, b) => a.is_some().cmp(&b.is_some()),
        };
        if order != Ordering::Equal {
            return order;
        }
    }
    let rest_set = |rest: &[Option<CqlValue>]| rest.iter().any(Option::is_some);
    match a.len().cmp(&b.len()) {
        Ordering::Greater if rest_set(&a[b.len()..]) => Ordering::Greater,
        Ordering::Less if rest_set(&b[a.len()..]) => Ordering::Less,
        _ => Ordering::Equal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scylla::serialize::value::SerializeValue;
    use scylla::serialize::writers::CellWriter;
    use scylla::value::CqlDate;
    use scylla::value::CqlTime;
    use scylla::value::CqlTimestamp;
    use scylla::value::CqlTimeuuid;
    use scylla::value::CqlVarint;
    use uuid::Uuid;

    fn samples() -> Vec<(ColumnType<'static>, Vec<CqlValue>)> {
        let native = ColumnType::Native;
        vec![
            (
                native(NativeType::TinyInt),
                vec![
                    CqlValue::TinyInt(-3),
                    CqlValue::TinyInt(0),
                    CqlValue::TinyInt(7),
                ],
            ),
            (
                native(NativeType::Int),
                vec![CqlValue::Int(i32::MIN), CqlValue::Int(-1), CqlValue::Int(5)],
            ),
            (
                native(NativeType::BigInt),
                vec![
                    CqlValue::BigInt(-10),
                    CqlValue::BigInt(0),
                    CqlValue::BigInt(10),
                ],
            ),
            (
                native(NativeType::Varint),
                vec![
                    CqlValue::Varint(CqlVarint::from_signed_bytes_be(vec![0x80, 0x00])),
                    CqlValue::Varint(CqlVarint::from_signed_bytes_be(vec![0xff])),
                    CqlValue::Varint(CqlVarint::from_signed_bytes_be(vec![0x00, 0x05])),
                    CqlValue::Varint(CqlVarint::from_signed_bytes_be(vec![0x01, 0x00])),
                ],
            ),
            (
                native(NativeType::Text),
                vec![
                    CqlValue::Text("".into()),
                    CqlValue::Text("a".into()),
                    CqlValue::Text("ab".into()),
                    CqlValue::Text("ż".into()),
                ],
            ),
            (
                native(NativeType::Blob),
                vec![
                    CqlValue::Blob(vec![0x01]),
                    CqlValue::Blob(vec![0x01, 0x00]),
                    CqlValue::Blob(vec![0xff]),
                ],
            ),
            (
                native(NativeType::Boolean),
                vec![CqlValue::Boolean(false), CqlValue::Boolean(true)],
            ),
            (
                native(NativeType::Timestamp),
                vec![
                    CqlValue::Timestamp(CqlTimestamp(-5)),
                    CqlValue::Timestamp(CqlTimestamp(9)),
                ],
            ),
            (
                native(NativeType::Date),
                vec![CqlValue::Date(CqlDate(1)), CqlValue::Date(CqlDate(1 << 31))],
            ),
            (
                native(NativeType::Time),
                vec![CqlValue::Time(CqlTime(0)), CqlValue::Time(CqlTime(86_399))],
            ),
            (
                native(NativeType::Inet),
                // As bytes: an IPv6 address starting with zeros sorts before 10.x.
                vec![
                    CqlValue::Inet("::1".parse().unwrap()),
                    CqlValue::Inet("10.0.0.1".parse().unwrap()),
                    CqlValue::Inet("10.0.0.2".parse().unwrap()),
                ],
            ),
            (
                native(NativeType::Double),
                vec![
                    CqlValue::Double(f64::NEG_INFINITY),
                    CqlValue::Double(-1.5),
                    CqlValue::Double(2.0),
                    CqlValue::Double(f64::NAN),
                ],
            ),
            (
                native(NativeType::Float),
                vec![CqlValue::Float(-1.0), CqlValue::Float(3.5)],
            ),
        ]
    }

    /// What the driver writes for a value of `typ`.
    fn driver_bytes(value: &CqlValue, typ: &ColumnType) -> Vec<u8> {
        let mut out = Vec::new();
        value.serialize(typ, CellWriter::new(&mut out)).unwrap();
        // A cell is its 4-byte length, then the value.
        out[4..].to_vec()
    }

    #[test]
    fn values_serialize_as_the_driver_writes_them() {
        for (typ, values) in samples() {
            for value in &values {
                assert_eq!(
                    serialize_value(value).unwrap(),
                    driver_bytes(value, &typ),
                    "{value:?}"
                );
            }
        }
        let tuple_type = ColumnType::Tuple(vec![
            ColumnType::Native(NativeType::Int),
            ColumnType::Native(NativeType::Text),
        ]);
        let tuple = CqlValue::Tuple(vec![Some(CqlValue::Int(3)), None]);
        assert_eq!(
            serialize_value(&tuple).unwrap(),
            driver_bytes(&tuple, &tuple_type)
        );
        let uuid = CqlValue::Uuid(Uuid::new_v4());
        assert_eq!(
            serialize_value(&uuid).unwrap(),
            driver_bytes(&uuid, &ColumnType::Native(NativeType::Uuid))
        );
    }

    #[test]
    fn samples_are_listed_in_scylla_order() {
        for (_, values) in samples() {
            for (i, a) in values.iter().enumerate() {
                for (j, b) in values.iter().enumerate() {
                    assert_eq!(compare_values(a, b), i.cmp(&j), "{a:?} vs {b:?}");
                }
            }
        }
        assert_eq!(
            compare_values(&CqlValue::Empty, &CqlValue::Int(i32::MIN)),
            Ordering::Less
        );
    }

    #[test]
    fn varints_compare_by_value_whatever_their_padding() {
        let v = |bytes: &[u8]| CqlValue::Varint(CqlVarint::from_signed_bytes_be(bytes.to_vec()));
        assert_eq!(
            compare_values(&v(&[0x00, 0x05]), &v(&[0x05])),
            Ordering::Equal
        );
        assert_eq!(
            compare_values(&v(&[0xff, 0xff]), &v(&[0xff])),
            Ordering::Equal
        );
        assert_eq!(compare_values(&v(&[0xfe]), &v(&[0xff])), Ordering::Less);
        assert_eq!(
            compare_values(&v(&[0xff, 0x00]), &v(&[0x80])),
            Ordering::Less
        );
        assert_eq!(
            compare_values(&v(&[0x7f]), &v(&[0x00, 0x80])),
            Ordering::Less
        );
    }

    #[test]
    fn timeuuids_compare_by_time_first() {
        // Version 1, time_low 0x00000002 against 0x00000001 with a higher time_hi.
        let earlier: [u8; 16] = [0, 0, 0, 2, 0, 0, 0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let later: [u8; 16] = [0, 0, 0, 1, 0, 0, 0x10, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        let t = |b: [u8; 16]| CqlValue::Timeuuid(CqlTimeuuid::from_bytes(b));
        assert_eq!(compare_values(&t(earlier), &t(later)), Ordering::Less);
        // The clock sequence and node compare as signed bytes.
        let mut low = earlier;
        low[8] = 0x80;
        assert_eq!(compare_values(&t(low), &t(earlier)), Ordering::Less);
        // As a plain uuid the same tail compares unsigned.
        let u = |b: [u8; 16]| CqlValue::Uuid(Uuid::from_bytes(b));
        assert_eq!(compare_values(&u(low), &u(earlier)), Ordering::Greater);
        // A version 4 uuid sorts after any version 1 one.
        let mut random = [0xffu8; 16];
        random[6] = 0x4f;
        assert_eq!(compare_values(&u(later), &u(random)), Ordering::Less);
    }

    #[test]
    fn tuples_compare_element_by_element() {
        let t = |items: Vec<Option<CqlValue>>| CqlValue::Tuple(items);
        let one = Some(CqlValue::Int(1));
        let two = Some(CqlValue::Int(2));
        assert_eq!(
            compare_values(
                &t(vec![one.clone(), two.clone()]),
                &t(vec![two.clone(), one.clone()])
            ),
            Ordering::Less
        );
        assert_eq!(
            compare_values(&t(vec![None]), &t(vec![one.clone()])),
            Ordering::Less
        );
        assert_eq!(
            compare_values(&t(vec![one.clone(), None]), &t(vec![one.clone()])),
            Ordering::Equal
        );
        assert_eq!(
            compare_values(&t(vec![one.clone(), two]), &t(vec![one])),
            Ordering::Greater
        );
    }

    fn layout(partition_key_count: usize, clustering_descending: &[bool]) -> KeyLayout {
        KeyLayout {
            partition_key_count,
            clustering_descending: clustering_descending.into(),
        }
    }

    #[test]
    fn tokens_are_murmur3_of_the_legacy_partition_key() {
        // Known Murmur3 tokens of ScyllaDB/Cassandra for an int partition key.
        let key = |pk: i32| PrimaryKey::from([CqlValue::Int(pk)]);
        let single = layout(1, &[]);
        assert_eq!(
            single.token_key(&key(1)),
            Some(token_key(-4069959284402364209))
        );
        assert_eq!(
            single.token_key(&key(2)),
            Some(token_key(-3248873570005575792))
        );
        // A compound key hashes each part with its length and a zero byte.
        let compound = layout(2, &[]);
        let bytes = compound
            .partition_bytes(&PrimaryKey::from([
                CqlValue::Int(1),
                CqlValue::Text("ab".into()),
            ]))
            .unwrap();
        assert_eq!(bytes, vec![0, 4, 0, 0, 0, 1, 0, 0, 2, b'a', b'b', 0]);
    }

    #[test]
    fn rows_of_one_partition_follow_the_clustering_order() {
        let row = |pk: i32, a: i32, b: &str| {
            PrimaryKey::from([
                CqlValue::Int(pk),
                CqlValue::Int(a),
                CqlValue::Text(b.into()),
            ])
        };
        let ascending = layout(1, &[false, false]);
        assert_eq!(
            ascending.compare_same_token(&row(1, 1, "z"), &row(1, 2, "a")),
            Ordering::Less
        );
        assert_eq!(
            ascending.compare_same_token(&row(1, 1, "a"), &row(1, 1, "b")),
            Ordering::Less
        );
        assert_eq!(
            ascending.compare_same_token(&row(1, 1, "a"), &row(1, 1, "a")),
            Ordering::Equal
        );
        // CLUSTERING ORDER BY (a DESC, b ASC).
        let mixed = layout(1, &[true, false]);
        assert_eq!(
            mixed.compare_same_token(&row(1, 1, "z"), &row(1, 2, "a")),
            Ordering::Greater
        );
        assert_eq!(
            mixed.compare_same_token(&row(1, 1, "a"), &row(1, 1, "b")),
            Ordering::Less
        );
        // Two partitions sharing a token compare by their key bytes, before any clustering.
        assert_eq!(
            ascending.compare_same_token(&row(1, 9, "z"), &row(2, 1, "a")),
            Ordering::Less
        );
    }

    #[test]
    fn supported_key_types() {
        assert!(is_key_type_supported(&ColumnType::Native(
            NativeType::Timeuuid
        )));
        assert!(is_key_type_supported(&ColumnType::Tuple(vec![
            ColumnType::Native(NativeType::Int),
            ColumnType::Native(NativeType::Text),
        ])));
        assert!(!is_key_type_supported(&ColumnType::Native(
            NativeType::Decimal
        )));
        assert!(!is_key_type_supported(&ColumnType::Tuple(vec![
            ColumnType::Native(NativeType::Duration)
        ])));
    }

    #[test]
    fn a_prefix_reverses_by_complement_and_lays_out_by_its_first_two_slots() {
        let a = SortPrefix([1, 5, 0, 0, 0]);
        let b = SortPrefix([1, 6, 0, 0, 0]);
        assert!(a < b && a.complement() > b.complement());
        assert_eq!(a.layout_key(), (1u128 << 64) | 5);
    }
}
