/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

use crate::create_config_channels;
use crate::db_basic;
use crate::db_basic::DbBasic;
use crate::db_basic::ScanFn;
use crate::db_basic::Table;
use crate::wait_for;
use httpapi::IndexNotReadyReason;
use httpapi::IndexStatus;
use httpclient::HttpClient;
use reqwest::StatusCode;
use scylla::cluster::metadata::NativeType;
use scylla::value::CqlValue;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use uuid::Uuid;
use vector_store::ColumnName;
use vector_store::Config;
use vector_store::DbIndexPartitioning;
use vector_store::HttpServerExt;
use vector_store::IndexKind;
use vector_store::IndexMetadata;
use vector_store::IndexOptionsSubstring;
use vector_store::NonemptyArc;
use vector_store::Percentage;
use vector_store::Timestamp;
use vector_store::node_state::NodeState;

fn substring_index_metadata(options: IndexOptionsSubstring) -> IndexMetadata {
    IndexMetadata {
        keyspace_name: "search_demo".into(),
        table_name: "users".into(),
        index_name: "users_nickname_sub".into(),
        primary_key_columns: NonemptyArc::new(["pk"]).unwrap(),
        partition_key_count: NonZeroUsize::new(1).unwrap(),
        target_columns: NonemptyArc::new(["nickname"]).unwrap(),
        partitioning: DbIndexPartitioning::Global,
        filtering_columns: Arc::new([]),
        alternator_attribute_types: Default::default(),
        version: Uuid::new_v4().into(),
        kind: IndexKind::Substring(options),
    }
}

async fn setup_substring_store(
    options: IndexOptionsSubstring,
    substring_indexes: bool,
    fullscan_fn: Option<ScanFn>,
) -> (
    impl std::future::Future<Output = (HttpClient, impl Sized, impl Sized)>,
    IndexMetadata,
    DbBasic,
    Sender<NodeState>,
) {
    let config = Config {
        vector_store_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
        substring_indexes,
        ..Default::default()
    };

    let node_state = vector_store::new_node_state().await;
    let (db_actor, db) = db_basic::new(node_state.clone());

    let index = substring_index_metadata(options);

    db.add_table(
        index.keyspace_name.clone(),
        index.table_name.clone(),
        Table {
            primary_keys: index.primary_key_columns.clone(),
            partition_key_count: 1,
            columns: Arc::new(HashMap::from([(ColumnName::from("pk"), NativeType::Int)])),
            dimensions: HashMap::new(),
        },
    )
    .unwrap();

    db.add_index(index.clone(), fullscan_fn, None).unwrap();

    let (receivers, senders) = create_config_channels(config).await;

    let run = {
        let node_state = node_state.clone();
        async move {
            let (server, _mtls) = vector_store::run(Some(node_state), Some(db_actor), receivers)
                .await
                .unwrap();
            let addr = (*server.address().await.borrow()).unwrap();

            (HttpClient::new(addr), server, senders)
        }
    };

    (run, index, db, node_state)
}

fn documents(rows: &[(i32, &str)]) -> ScanFn {
    let rows: Vec<_> = rows
        .iter()
        .enumerate()
        .map(|(i, (pk, text))| {
            (
                [CqlValue::Int(*pk)].into(),
                Some(text.to_string()),
                Timestamp::from_millis(10 * (i as u64 + 1)),
            )
        })
        .collect();
    db_basic::scan_fn_documents(rows)
}

const NICKNAMES: &[(i32, &str)] = &[
    (1, "宇将军"),
    (2, "李将军"),
    (3, "将军来了"),
    (4, "将领"),
    (5, "元帅"),
    (6, "南宫月"),
    (7, "小南宫粉丝团"),
    (8, "宫南"),
];

const USERNAMES: &[(i32, &str)] = &[
    (1, "NGgamer"),
    (2, "kingNG"),
    (3, "gn"),
    (4, "925555"),
    (5, "9255551"),
    (6, "925"),
];

async fn setup_and_wait(
    options: IndexOptionsSubstring,
    rows: &[(i32, &str)],
) -> (
    HttpClient,
    httpapi::KeyspaceName,
    httpapi::IndexName,
    DbBasic,
    impl Sized,
) {
    let (run, index, db, node_state) =
        setup_substring_store(options, true, Some(documents(rows))).await;

    let (client, server, config_tx) = run.await;
    let keyspace_name = index.keyspace_name.clone().into();
    let index_name = index.index_name.clone().into();

    wait_for(
        || async {
            client
                .index_status(&keyspace_name, &index_name)
                .await
                .is_ok_and(|status| {
                    status.status == IndexStatus::Serving && status.count == rows.len()
                })
        },
        "Waiting for the substring index to be serving",
    )
    .await;

    (
        client,
        keyspace_name,
        index_name,
        db,
        (server, config_tx, node_state),
    )
}

async fn contains(
    client: &HttpClient,
    keyspace_name: &httpapi::KeyspaceName,
    index_name: &httpapi::IndexName,
    query: &str,
    limit: usize,
    offset: usize,
) -> Vec<i64> {
    let primary_keys = client
        .contains(
            keyspace_name,
            index_name,
            query.into(),
            NonZeroUsize::new(limit).unwrap().into(),
            offset,
        )
        .await;
    let mut ids: Vec<i64> = primary_keys
        .get(&"pk".into())
        .expect("pk column in the response")
        .iter()
        .map(|value| value.as_i64().unwrap())
        .collect();
    ids.sort_unstable();
    ids
}

#[tokio::test]
async fn substring_index_returns_proper_count() {
    crate::enable_tracing();

    let (client, keyspace_name, index_name, _db, _hold) =
        setup_and_wait(IndexOptionsSubstring::default(), NICKNAMES).await;

    let status = client
        .index_status(&keyspace_name, &index_name)
        .await
        .unwrap();

    assert_eq!(status.status, IndexStatus::Serving);
    assert_eq!(status.count, NICKNAMES.len());
}

#[tokio::test]
async fn substring_contains_returns_cjk_infix_matches() {
    crate::enable_tracing();

    let (client, ks, idx, _db, _hold) =
        setup_and_wait(IndexOptionsSubstring::default(), NICKNAMES).await;

    assert_eq!(contains(&client, &ks, &idx, "将军", 20, 0).await, [1, 2, 3]);
    assert_eq!(contains(&client, &ks, &idx, "南宫", 20, 0).await, [6, 7]);
    assert_eq!(contains(&client, &ks, &idx, "宫", 20, 0).await, [6, 7, 8]);
    assert_eq!(contains(&client, &ks, &idx, "将军来了", 20, 0).await, [3]);
}

#[tokio::test]
async fn substring_contains_case_insensitive_option() {
    crate::enable_tracing();

    let options = IndexOptionsSubstring {
        case_sensitive: false.into(),
        ..Default::default()
    };
    let (client, ks, idx, _db, _hold) = setup_and_wait(options, USERNAMES).await;

    assert_eq!(contains(&client, &ks, &idx, "ng", 20, 0).await, [1, 2]);
    assert_eq!(contains(&client, &ks, &idx, "925555", 20, 0).await, [4, 5]);
    assert_eq!(contains(&client, &ks, &idx, "925", 20, 0).await, [4, 5, 6]);
}

#[tokio::test]
async fn substring_contains_is_case_sensitive_by_default() {
    crate::enable_tracing();

    let (client, ks, idx, _db, _hold) =
        setup_and_wait(IndexOptionsSubstring::default(), USERNAMES).await;

    // Only "kingNG" contains a lowercase "ng" (inside "king").
    assert_eq!(contains(&client, &ks, &idx, "ng", 20, 0).await, [2]);
    assert_eq!(contains(&client, &ks, &idx, "NG", 20, 0).await, [1, 2]);
}

#[tokio::test]
async fn substring_contains_returns_empty_for_no_match() {
    crate::enable_tracing();

    let (client, ks, idx, _db, _hold) =
        setup_and_wait(IndexOptionsSubstring::default(), NICKNAMES).await;

    assert!(contains(&client, &ks, &idx, "元将", 20, 0).await.is_empty());
}

#[tokio::test]
async fn substring_contains_respects_limit_and_offset() {
    crate::enable_tracing();

    let (client, ks, idx, _db, _hold) =
        setup_and_wait(IndexOptionsSubstring::default(), NICKNAMES).await;

    let first = contains(&client, &ks, &idx, "宫", 2, 0).await;
    let rest = contains(&client, &ks, &idx, "宫", 2, 2).await;
    assert_eq!(first.len(), 2);
    assert_eq!(rest.len(), 1);
    let mut all: Vec<_> = first.into_iter().chain(rest).collect();
    all.sort_unstable();
    assert_eq!(all, [6, 7, 8]);
}

#[tokio::test]
async fn substring_contains_not_found_returns_404() {
    crate::enable_tracing();

    let (client, _ks, _idx, _db, _hold) =
        setup_and_wait(IndexOptionsSubstring::default(), NICKNAMES).await;

    let response = client
        .post_contains(
            &"nonexistent_ks".into(),
            &"nonexistent_idx".into(),
            "将军".into(),
            NonZeroUsize::new(10).unwrap().into(),
            0,
        )
        .await;

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn substring_contains_empty_query_returns_400() {
    crate::enable_tracing();

    let (client, ks, idx, _db, _hold) =
        setup_and_wait(IndexOptionsSubstring::default(), NICKNAMES).await;

    let response = client
        .post_contains(
            &ks,
            &idx,
            "".into(),
            NonZeroUsize::new(10).unwrap().into(),
            0,
        )
        .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn substring_contains_returns_503_and_node_bootstrapping_reason_while_the_first_index_builds()
{
    crate::enable_tracing();

    let (run, index, _db, _node_state) = setup_substring_store(
        IndexOptionsSubstring::default(),
        true,
        Some(db_basic::pending_scan_fn()),
    )
    .await;
    let (client, _server, _config_tx) = run.await;

    let keyspace_name = index.keyspace_name.clone().into();
    let index_name = index.index_name.clone().into();

    wait_for(
        || async {
            client
                .index_status(&keyspace_name, &index_name)
                .await
                .is_ok_and(|status| status.status == IndexStatus::Bootstrapping)
        },
        "Waiting for the substring index to be bootstrapping",
    )
    .await;

    let response = client
        .post_contains(
            &keyspace_name,
            &index_name,
            "将军".into(),
            NonZeroUsize::new(10).unwrap().into(),
            0,
        )
        .await;

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let reason: IndexNotReadyReason = response.json().await.unwrap();
    assert_eq!(reason, IndexNotReadyReason::NodeBootstrapping);
}

#[tokio::test]
async fn substring_contains_returns_503_and_index_building_reason_while_a_later_index_builds() {
    crate::enable_tracing();

    let (client, _ks, _idx, db, _hold) =
        setup_and_wait(IndexOptionsSubstring::default(), NICKNAMES).await;

    let index = IndexMetadata {
        index_name: "users_username_sub".into(),
        ..substring_index_metadata(IndexOptionsSubstring::default())
    };
    db.add_index(index.clone(), Some(db_basic::pending_scan_fn()), None)
        .unwrap();
    db.set_next_full_scan_progress(vector_store::Progress::InProgress(
        Percentage::try_from(75.0).unwrap(),
    ));

    let keyspace_name = index.keyspace_name.clone().into();
    let index_name = index.index_name.clone().into();

    wait_for(
        || async {
            client
                .index_status(&keyspace_name, &index_name)
                .await
                .is_ok_and(|status| status.status == IndexStatus::Bootstrapping)
        },
        "Waiting for the substring index to be bootstrapping",
    )
    .await;

    let response = client
        .post_contains(
            &keyspace_name,
            &index_name,
            "将军".into(),
            NonZeroUsize::new(10).unwrap().into(),
            0,
        )
        .await;

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let reason: IndexNotReadyReason = response.json().await.unwrap();
    let IndexNotReadyReason::IndexBuilding { message } = reason else {
        panic!("expected IndexBuilding, got {reason:?}");
    };
    assert!(
        message.contains("progress: 75.000%"),
        "unexpected message: {message}"
    );
}

#[tokio::test]
async fn substring_index_appears_in_indexes_list_with_its_options() {
    crate::enable_tracing();

    let options = IndexOptionsSubstring {
        min_gram: "2".parse().unwrap(),
        max_gram: "4".parse().unwrap(),
        case_sensitive: false.into(),
        ..Default::default()
    };
    let (client, ks, idx, _db, _hold) = setup_and_wait(options, NICKNAMES).await;

    let indexes = client.indexes().await;
    let listed = indexes
        .iter()
        .find(|i| i.keyspace == ks && i.index == idx)
        .expect("the substring index is listed");

    assert_eq!(
        listed.options,
        httpapi::IndexOptions::Substring(httpapi::SubstringIndexOptions {
            min_gram: 2,
            max_gram: 4,
            case_sensitive: false,
        })
    );
    assert_eq!(
        client.index_info(&ks, &idx).await.unwrap().options,
        listed.options
    );
}

#[tokio::test]
async fn substring_indexes_are_ignored_when_disabled_by_config() {
    crate::enable_tracing();

    let (run, index, _db, _node_state) = setup_substring_store(
        IndexOptionsSubstring::default(),
        false,
        Some(documents(NICKNAMES)),
    )
    .await;
    let (client, _server, _config_tx) = run.await;

    wait_for(
        || async {
            client
                .status()
                .await
                .is_ok_and(|status| status == httpapi::NodeStatus::Serving)
        },
        "Waiting for the node to be serving",
    )
    .await;

    let keyspace_name: httpapi::KeyspaceName = index.keyspace_name.clone().into();
    let index_name: httpapi::IndexName = index.index_name.clone().into();
    assert!(
        client
            .indexes()
            .await
            .iter()
            .all(|i| !(i.keyspace == keyspace_name && i.index == index_name))
    );
}
