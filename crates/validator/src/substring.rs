/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! End-to-end tests of the substring (`LIKE '%kw%'`) index.
//!
//! Until ScyllaDB routes `LIKE` to the index node, the queries go straight to the
//! vector-store `/contains` endpoint of every node.

use crate::TestActors;
use crate::common::*;
use async_backtrace::framed;
use httpapi::IndexInfo;
use httpapi::IndexOptions;
use httpapi::KeyspaceName;
use httpapi::SubstringIndexOptions;
use httpclient::HttpClient;
use scylla::client::session::Session;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::RwLock;
use tracing::info;

e2etest::group!(
    name = substring,
    fixtures = (Cluster),
    parent = crate::validator
);

struct Cluster {
    actors: Arc<TestActors>,
}

impl e2etest::Fixture for Cluster {
    async fn setup(setup: &mut impl e2etest::Setup) -> Option<Self> {
        let actors = setup.setup::<TestActors>().await?;
        init(&actors).await;
        Some(Self { actors })
    }

    async fn teardown(self) {
        cleanup(&self.actors).await;
    }
}

struct Fixture {
    session: Arc<Session>,
    keyspace: KeyspaceName,
    table: TableName,
    clients: Vec<HttpClient>,
    index: RwLock<Option<Arc<IndexInfo>>>,
}

impl e2etest::Fixture for Fixture {
    async fn setup(setup: &mut impl e2etest::Setup) -> Option<Self> {
        let cluster = setup.setup::<Cluster>().await?;
        let (session, clients) = prepare_connection(&cluster.actors).await;
        let keyspace = create_keyspace(&session).await;
        let table = create_table(&session, "pk INT PRIMARY KEY, nickname TEXT", None).await;
        Some(Self {
            session,
            keyspace,
            table,
            clients,
            index: RwLock::new(None),
        })
    }

    async fn teardown(self) {
        self.session
            .query_unpaged(format!("DROP KEYSPACE {}", self.keyspace), ())
            .await
            .expect("failed to drop a keyspace");
    }
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

impl Fixture {
    #[framed]
    async fn insert_rows(&self, rows: &[(i32, &str)]) {
        let stmt = self
            .session
            .prepare(format!(
                "INSERT INTO {} (pk, nickname) VALUES (?, ?)",
                self.table
            ))
            .await
            .expect("failed to prepare insert statement");
        for (pk, nickname) in rows {
            self.session
                .execute_unpaged(&stmt, (pk, nickname.to_string()))
                .await
                .expect("failed to insert data");
        }
    }

    #[framed]
    async fn delete_row(&self, pk: i32) {
        self.session
            .query_unpaged(format!("DELETE FROM {} WHERE pk = ?", self.table), (pk,))
            .await
            .expect("failed to delete data");
    }

    #[framed]
    async fn create_index_with_options(&self, options: &[(&str, &str)]) {
        let index = create_index(
            CreateIndexQuery::new(&self.session, &self.clients, &self.table, "nickname")
                .index_type("substring_index")
                .options(options.iter().copied()),
        )
        .await;
        for client in &self.clients {
            wait_for_index(client, &index).await;
        }
        *self.index.write().unwrap() = Some(Arc::new(index));
    }

    fn index(&self) -> Arc<IndexInfo> {
        Arc::clone(
            self.index
                .read()
                .unwrap()
                .as_ref()
                .expect("the substring index has not been created yet"),
        )
    }

    /// The primary keys containing `query`, sorted, as seen by one node.
    #[framed]
    async fn contains_on(&self, client: &HttpClient, query: &str) -> Vec<i32> {
        let index = self.index();
        let primary_keys = client
            .contains(
                &index.keyspace,
                &index.index,
                query.into(),
                NonZeroUsize::new(100).unwrap().into(),
                0,
            )
            .await;
        let mut pks: Vec<i32> = primary_keys
            .get(&"pk".into())
            .expect("pk column in the response")
            .iter()
            .map(|value| value.as_i64().expect("an integer primary key") as i32)
            .collect();
        pks.sort_unstable();
        pks
    }

    /// Asserts every node answers `query` with exactly `expected`.
    async fn assert_contains(&self, query: &str, expected: &[i32]) {
        for client in &self.clients {
            assert_eq!(
                self.contains_on(client, query).await,
                expected,
                "query {query:?} at {}",
                client.url()
            );
        }
    }

    /// Waits until every node answers `query` with exactly `expected`.
    async fn wait_for_contains(&self, query: &str, expected: &[i32]) {
        for client in &self.clients {
            wait_for(
                || async { self.contains_on(client, query).await == expected },
                format!("query {query:?} to return {expected:?} at {}", client.url()),
                DEFAULT_OPERATION_TIMEOUT,
            )
            .await;
        }
    }
}

#[e2etest::test(group = substring)]
async fn substring_index_lifecycle(actors: Arc<TestActors>) {
    info!("started");

    let (session, clients) = prepare_connection(&actors).await;
    let keyspace = create_keyspace(&session).await;
    let table = create_table(&session, "pk INT PRIMARY KEY, nickname TEXT", None).await;

    info!("Creating substring index with explicit options");
    let index = create_index(
        CreateIndexQuery::new(&session, &clients, &table, "nickname")
            .index_type("substring_index")
            .options([
                ("min_gram", "2"),
                ("max_gram", "4"),
                ("case_sensitive", "false"),
            ]),
    )
    .await;

    info!("Verifying index is SERVING on all nodes with its options");
    for client in &clients {
        let serving = wait_for_index(client, &index).await;
        assert_eq!(
            serving.options,
            IndexOptions::Substring(SubstringIndexOptions {
                min_gram: 2,
                max_gram: 4,
                case_sensitive: false,
            })
        );
    }

    info!("Dropping substring index");
    session
        .query_unpaged(format!("DROP INDEX {}", index.index), ())
        .await
        .expect("failed to drop index");

    info!("Verifying index is removed on all nodes");
    for client in &clients {
        wait_for_no_index(client, &index).await;
    }

    session
        .query_unpaged(format!("DROP KEYSPACE {keyspace}"), ())
        .await
        .expect("failed to drop a keyspace");

    info!("finished");
}

#[e2etest::test(group = substring)]
async fn contains_cjk_nicknames(fixture: Arc<Fixture>) {
    info!("started");

    fixture.insert_rows(NICKNAMES).await;
    fixture.create_index_with_options(&[]).await;

    fixture.assert_contains("将军", &[1, 2, 3]).await;
    fixture.assert_contains("南宫", &[6, 7]).await;
    fixture.assert_contains("宫", &[6, 7, 8]).await;
    fixture.assert_contains("将军来了", &[3]).await;
    fixture.assert_contains("元将", &[]).await;

    info!("finished");
}

#[e2etest::test(group = substring)]
async fn contains_case_insensitive_usernames(fixture: Arc<Fixture>) {
    info!("started");

    fixture.insert_rows(USERNAMES).await;
    fixture
        .create_index_with_options(&[("case_sensitive", "false")])
        .await;

    fixture.assert_contains("ng", &[1, 2]).await;
    fixture.assert_contains("NG", &[1, 2]).await;
    fixture.assert_contains("925555", &[4, 5]).await;
    fixture.assert_contains("925", &[4, 5, 6]).await;

    info!("finished");
}

#[e2etest::test(group = substring)]
async fn contains_is_case_sensitive_by_default(fixture: Arc<Fixture>) {
    info!("started");

    fixture.insert_rows(USERNAMES).await;
    fixture.create_index_with_options(&[]).await;

    // Only "kingNG" contains a lowercase "ng" (inside "king").
    fixture.assert_contains("ng", &[2]).await;
    fixture.assert_contains("NG", &[1, 2]).await;

    info!("finished");
}

#[e2etest::test(group = substring)]
async fn substring_crud_insert(fixture: Arc<Fixture>) {
    info!("started");

    fixture.create_index_with_options(&[]).await;
    fixture.insert_rows(&[(1, "宇将军")]).await;

    fixture.wait_for_contains("将军", &[1]).await;

    info!("finished");
}

#[e2etest::test(group = substring)]
async fn substring_crud_update(fixture: Arc<Fixture>) {
    info!("started");

    fixture.create_index_with_options(&[]).await;
    fixture.insert_rows(&[(1, "宇将军")]).await;
    fixture.wait_for_contains("将军", &[1]).await;

    fixture.insert_rows(&[(1, "南宫月")]).await;

    fixture.wait_for_contains("将军", &[]).await;
    fixture.wait_for_contains("南宫", &[1]).await;

    info!("finished");
}

#[e2etest::test(group = substring)]
async fn substring_crud_delete(fixture: Arc<Fixture>) {
    info!("started");

    fixture.insert_rows(NICKNAMES).await;
    fixture.create_index_with_options(&[]).await;
    fixture.assert_contains("将军", &[1, 2, 3]).await;

    fixture.delete_row(1).await;

    fixture.wait_for_contains("将军", &[2, 3]).await;

    info!("finished");
}
