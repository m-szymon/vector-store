/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

use crate::TestActors;
use crate::common;
use async_backtrace::framed;
use aws_sdk_dynamodb::primitives::Blob;
use aws_sdk_dynamodb::types::AttributeValue;
use aws_sdk_dynamodb::types::ScalarAttributeType;
use e2etest::TestCase;
use std::collections::HashSet;
use tracing::info;

use crate::alternator;
use crate::alternator::Item;
use crate::alternator::TableContext;
use crate::alternator::TableShape;
use crate::alternator::query::QueryBuilderExt;

/// Shared logic for all key-type tests: creates a table with 2 initial items,
/// adds a 3rd via PutItem, then queries with a keys-only projection and asserts
/// that only the pk attribute is returned.
async fn query_with_key_type(
    actors: &TestActors,
    shape: &TableShape,
    initial: &[Item],
    extra: Item,
) {
    let vec_attr = shape.vec().unwrap();
    let ctx = TableContext::create_with_data(actors, shape, initial).await;

    ctx.put(&extra).await;
    ctx.wait_for_count(3).await;

    let items = ctx
        .client
        .query()
        .table_name(&ctx.table_name)
        .index_name(ctx.index.index.as_ref())
        .limit(3)
        .projection_expression("#pk")
        .expression_attribute_names("#pk", shape.pk())
        .vector_search([1.0, 1.0, 1.0])
        .send()
        .await
        .expect("Query with VectorSearch should succeed")
        .items()
        .to_vec();

    assert!(
        !items.is_empty(),
        "keys-only query should return at least one item"
    );
    let expected_keys: HashSet<&str> = [shape.pk()].into();
    for item in &items {
        let got_keys: HashSet<&str> = item.keys().map(String::as_str).collect();
        assert_eq!(
            got_keys,
            expected_keys,
            "projected item should contain only pk '{}'",
            shape.pk()
        );
        assert!(
            !item.contains_key(vec_attr),
            "projected item should NOT contain vector '{vec_attr}'"
        );
    }

    ctx.done().await;
}

#[framed]
async fn query_with_string_key(actors: TestActors) {
    info!("started");
    let shape = TableShape {
        table_prefix: None,
        index_prefix: None,
        pk_name: "Pk-StrKey".into(),
        sk_name: None,
        vec_name: Some("Vec-StrKey".into()),
        pk_type: ScalarAttributeType::S,
    };
    let v = shape.vec().unwrap();
    let initial = [
        Item::new(shape.pk(), AttributeValue::S("str-a".into())).vec(v, [1.0, 1.0, 1.0]),
        Item::new(shape.pk(), AttributeValue::S("str-b".into())).vec(v, [1.0, 2.0, 4.0]),
    ];
    let extra = Item::new(shape.pk(), AttributeValue::S("str-c".into())).vec(v, [1.0, 4.0, 8.0]);
    query_with_key_type(&actors, &shape, &initial, extra).await;
    info!("finished");
}

#[framed]
async fn query_with_number_key(actors: TestActors) {
    info!("started");
    let shape = TableShape {
        table_prefix: None,
        index_prefix: None,
        pk_name: "Pk-NumKey".into(),
        sk_name: None,
        vec_name: Some("Vec-NumKey".into()),
        pk_type: ScalarAttributeType::N,
    };
    let v = shape.vec().unwrap();
    let initial = [
        Item::new(shape.pk(), AttributeValue::N("1".into())).vec(v, [1.0, 1.0, 1.0]),
        Item::new(shape.pk(), AttributeValue::N("2".into())).vec(v, [1.0, 2.0, 4.0]),
    ];
    let extra = Item::new(shape.pk(), AttributeValue::N("3".into())).vec(v, [1.0, 4.0, 8.0]);
    query_with_key_type(&actors, &shape, &initial, extra).await;
    info!("finished");
}

#[framed]
async fn query_with_binary_key(actors: TestActors) {
    info!("started");
    let shape = TableShape {
        table_prefix: None,
        index_prefix: None,
        pk_name: "Pk-BinKey".into(),
        sk_name: None,
        vec_name: Some("Vec-BinKey".into()),
        pk_type: ScalarAttributeType::B,
    };
    let v = shape.vec().unwrap();
    let initial = [
        Item::new(shape.pk(), AttributeValue::B(Blob::new(vec![0x01u8]))).vec(v, [1.0, 1.0, 1.0]),
        Item::new(shape.pk(), AttributeValue::B(Blob::new(vec![0x02u8]))).vec(v, [1.0, 2.0, 4.0]),
    ];
    let extra =
        Item::new(shape.pk(), AttributeValue::B(Blob::new(vec![0x03u8]))).vec(v, [1.0, 4.0, 8.0]);
    query_with_key_type(&actors, &shape, &initial, extra).await;
    info!("finished");
}

/// Verifies queries work with both FLOAT32VECTOR-encoded items and query vectors.
#[framed]
async fn query_with_optimized_vector_type(actors: TestActors) {
    info!("started");

    let shape = TableShape {
        table_prefix: None,
        index_prefix: None,
        pk_name: "Pk-VecType".into(),
        sk_name: None,
        vec_name: Some("Vec-VecType".into()),
        pk_type: ScalarAttributeType::S,
    };

    let pk_name = shape.pk();
    let vec_name = shape.vec().unwrap();
    let pk_l_scan = "pk-l-scan";
    let pk_v_scan = "pk-v-scan";
    let pk_l_live = "pk-l-live";
    let pk_v_live = "pk-v-live";

    // Both L-type and FLOAT32VECTOR-type items inserted before index creation go through
    // the initial scan path.
    let initial =
        [Item::new(pk_name, AttributeValue::S(pk_l_scan.into())).vec(vec_name, [1.0, 0.0, 0.0])];
    let ctx = TableContext::create_with_data(&actors, &shape, &initial).await;
    ctx.put_vector(&AttributeValue::S(pk_v_scan.into()), [1.0, 0.0, 0.0])
        .await;

    ctx.wait_for_count(2).await;

    // Both L-type and FLOAT32VECTOR-type items inserted after index creation go through
    // the live path.
    ctx.put(
        &Item::new(pk_name, AttributeValue::S(pk_l_live.into())).vec(vec_name, [1.0, 0.0, 0.0]),
    )
    .await;
    ctx.put_vector(&AttributeValue::S(pk_v_live.into()), [1.0, 0.0, 0.0])
        .await;

    ctx.wait_for_count(4).await;

    // Query with standard L-type vector encoding.
    let items = ctx
        .client
        .query()
        .table_name(&ctx.table_name)
        .index_name(ctx.index.index.as_ref())
        .limit(4)
        .vector_search([1.0_f32, 0.0, 0.0])
        .send()
        .await
        .expect("Query with L-type vector should succeed")
        .items()
        .to_vec();

    for pk in [pk_l_scan, pk_v_scan, pk_l_live, pk_v_live] {
        assert!(
            items
                .iter()
                .any(|item| item.get(pk_name) == Some(&AttributeValue::S(pk.into()))),
            "L-type query should return '{pk}'"
        );
    }

    // Query with optimized FLOAT32VECTOR vector encoding.
    let items = ctx
        .client
        .query()
        .table_name(&ctx.table_name)
        .index_name(ctx.index.index.as_ref())
        .limit(4)
        .vector_search_optimized([1.0_f32, 0.0, 0.0])
        .send()
        .await
        .expect("Query with FLOAT32VECTOR vector should succeed")
        .items()
        .to_vec();

    for pk in [pk_l_scan, pk_v_scan, pk_l_live, pk_v_live] {
        assert!(
            items
                .iter()
                .any(|item| item.get(pk_name) == Some(&AttributeValue::S(pk.into()))),
            "FLOAT32VECTOR query should return '{pk}'"
        );
    }

    ctx.done().await;

    info!("finished");
}

pub(super) async fn new() -> TestCase<TestActors> {
    TestCase::empty()
        .with_init(common::DEFAULT_TEST_TIMEOUT, alternator::init)
        .with_cleanup(common::DEFAULT_TEST_TIMEOUT, common::cleanup)
        .with_test(
            "query_with_string_key",
            common::DEFAULT_TEST_TIMEOUT,
            query_with_string_key,
        )
        .with_test(
            "query_with_number_key",
            common::DEFAULT_TEST_TIMEOUT,
            query_with_number_key,
        )
        .with_test(
            "query_with_binary_key",
            common::DEFAULT_TEST_TIMEOUT,
            query_with_binary_key,
        )
        .with_test(
            "query_with_optimized_vector_type",
            common::DEFAULT_TEST_TIMEOUT,
            query_with_optimized_vector_type,
        )
}
