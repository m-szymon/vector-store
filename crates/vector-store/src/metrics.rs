/*
 * Copyright 2025-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */
use dashmap::DashMap;
use dashmap::DashSet;
use prometheus::CounterVec;
use prometheus::GaugeVec;
use prometheus::HistogramVec;
use prometheus::Registry;
use std::sync::Arc;

pub(crate) const OP_INSERT: &str = "insert";
pub(crate) const OP_UPDATE: &str = "update";
pub(crate) const OP_REMOVE: &str = "remove";
pub(crate) const OPERATIONS: &[&str] = &[OP_INSERT, OP_UPDATE, OP_REMOVE];
pub(crate) const OP_ANN: &str = "ann";
pub(crate) const OP_BM25: &str = "bm25";
pub(crate) const OP_HIGHLIGHT: &str = "highlight";
pub(crate) const OP_LIKE: &str = "like";
pub(crate) const REQUEST_OPERATIONS: &[&str] = &[OP_ANN, OP_BM25, OP_HIGHLIGHT, OP_LIKE];

#[derive(Clone)]
pub(crate) struct Metrics {
    pub registry: Registry,
    pub latency: HistogramVec,
    pub size: GaugeVec,
    pub modified: CounterVec,
    pub indexing_lag: HistogramVec,
    pub cdc_reader_up: GaugeVec,
    pub cdc_handler_errors_total: CounterVec,
    pub cdc_reader_restarts_total: CounterVec,
    pub cdc_last_processed_timestamp_seconds: GaugeVec,
    pub fts_index_size_bytes: GaugeVec,
    pub fts_segment_count: GaugeVec,
    pub pattern_index_size_bytes: GaugeVec,
    pub pattern_segment_count: GaugeVec,
    /// What a pattern index's searches have cost, summed since it was created. Read off the
    /// index on scrape, so they are gauges that only ever grow; a client diffs two scrapes to
    /// price the queries in between. One entry per [`PATTERN_SEARCH_TOTALS`].
    pub pattern_search_totals: Vec<GaugeVec>,
    /// Progress of a pattern index's rewrite of wide segments. One entry per
    /// [`PATTERN_REWRITE_GAUGES`].
    pub pattern_rewrite: Vec<GaugeVec>,
    /// A pattern index's segments, one label set per segment ordinal: how many rows each holds
    /// and the span of sort keys its pruning is judged by. See [`PATTERN_SEGMENT_GAUGES`].
    pub pattern_segment_layout: Vec<GaugeVec>,
    dirty_indexes: Arc<DashSet<(String, String)>>,
    /// How many segment label sets each pattern index exported last time, so a scrape after a
    /// merge can drop the ordinals that no longer exist.
    pattern_segments_exported: Arc<DashMap<(String, String), usize>>,
}

/// The per-index search totals, in the order `pattern_search_totals` holds them.
pub(crate) const PATTERN_SEARCH_TOTALS: &[(&str, &str)] = &[
    (
        "pattern_search_total",
        "Pattern searches served by an index since it was created",
    ),
    (
        "pattern_search_segments_considered_total",
        "Segments a pattern search could not rule out from their bounds, summed over searches",
    ),
    (
        "pattern_search_segments_opened_total",
        "Segments a pattern search walked, summed over searches",
    ),
    (
        "pattern_search_postings_scanned_total",
        "Postings a pattern search visited, summed over searches",
    ),
    (
        "pattern_search_heap_entrants_total",
        "Matches that entered a pattern search's page or top-k heap, summed over searches",
    ),
    (
        "pattern_search_store_reads_total",
        "Documents a pattern search read from the document store, summed over searches",
    ),
    (
        "pattern_search_walk_seconds_total",
        "Time pattern searches spent walking the index, summed over searches",
    ),
    (
        "pattern_search_column_opens_total",
        "Columns a pattern search had to open (misses of the per-segment column cache), summed over searches",
    ),
    (
        "pattern_search_prepare_seconds_total",
        "Time pattern searches spent reading segment bounds before the first posting (part of the walk time), summed over searches",
    ),
    (
        "pattern_search_page_resolve_seconds_total",
        "Time pattern searches spent turning the page into primary ids (part of the walk time), summed over searches",
    ),
];

/// The rewrite (P3b) gauges, in the order `pattern_rewrite` holds them.
pub(crate) const PATTERN_REWRITE_GAUGES: &[(&str, &str)] = &[
    (
        "pattern_rewrite_ranges_total",
        "Ranges the current or last rewrite of wide segments was split into",
    ),
    (
        "pattern_rewrite_ranges_done",
        "Ranges of that rewrite moved so far",
    ),
    (
        "pattern_rewrite_docs_total",
        "Rows moved by rewrites since the index was created",
    ),
    (
        "pattern_l0_docs",
        "Rows in wide segments when the current or last rewrite was planned",
    ),
];

/// The per-segment gauges, in the order `pattern_segment_layout` holds them. The sort bounds
/// are exported with the sign bias of `cql_types::to_sort_key` undone, so a timestamp or integer
/// column reads as its own value. A `date` column carries no bias and so comes out offset by
/// 2^63; the spans and overlaps these gauges exist to show are unaffected.
pub(crate) const PATTERN_SEGMENT_GAUGES: &[(&str, &str)] = &[
    (
        "pattern_segment_docs",
        "Live rows in one segment of a pattern index",
    ),
    (
        "pattern_segment_sort_min",
        "Lowest sort-column value in one segment of an ordered pattern index",
    ),
    (
        "pattern_segment_sort_max",
        "Highest sort-column value in one segment of an ordered pattern index",
    ),
];

impl Metrics {
    pub(crate) fn new() -> Self {
        let registry = Registry::new();

        // Custom buckets: 0.1ms to 10s
        let buckets = vec![
            0.0001, // 0.1 ms
            0.0002, // 0.2 ms
            0.0005, // 0.5 ms
            0.001,  // 1 ms
            0.002,  // 2 ms
            0.005,  // 5 ms
            0.01,   // 10 ms
            0.02,   // 20 ms
            0.05,   // 50 ms
            0.1,    // 0.1 second
            0.2,    // 0.2 seconds
            0.5,    // 0.5 seconds
            1.0,    // 1 seconds
            2.0,    // 2 seconds
            5.0,    // 5 seconds
            10.0,   // 10 seconds
        ];

        let latency = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "request_latency_seconds",
                "Latency per index and operation (seconds)",
            )
            .buckets(buckets),
            &["keyspace", "index_name", "operation"],
        )
        .unwrap();

        let size = GaugeVec::new(
            prometheus::Opts::new("index_size", "Number of Vector per index"),
            &["keyspace", "index_name"],
        )
        .unwrap();

        let modified: CounterVec = CounterVec::new(
            prometheus::Opts::new("index_modified", "Number of modified items per index"),
            &["keyspace", "index_name", "operation"],
        )
        .unwrap();

        // Custom buckets spanning the realtime (sub-second) and consistent
        // (tens of seconds) CDC reader regimes, up to a few minutes to surface
        // stalled indexing.
        let lag_buckets = vec![
            0.05,  // 50 ms
            0.1,   // 0.1 second
            0.25,  // 0.25 seconds
            0.5,   // 0.5 seconds
            1.0,   // 1 second
            2.5,   // 2.5 seconds
            5.0,   // 5 seconds
            10.0,  // 10 seconds
            30.0,  // 30 seconds
            60.0,  // 1 minute
            120.0, // 2 minutes
            300.0, // 5 minutes
        ];

        let indexing_lag = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "indexing_lag_seconds",
                "Time in seconds between a CDC-recorded change in ScyllaDB and its indexing in the vector store, \
                 per CDC reader: the fine reader's is how long a change takes to become searchable, the wide \
                 reader's that of its delayed re-read",
            )
            .buckets(lag_buckets),
            &["keyspace", "index_name", "reader"],
        )
        .unwrap();

        let cdc_reader_up = GaugeVec::new(
            prometheus::Opts::new(
                "cdc_reader_up",
                "Whether the CDC reader for an index is currently running (1) or stopped (0)",
            ),
            &["keyspace", "index_name", "reader"],
        )
        .unwrap();

        let cdc_handler_errors_total = CounterVec::new(
            prometheus::Opts::new(
                "cdc_handler_errors_total",
                "Total number of CDC handler errors per index and reader",
            ),
            &["keyspace", "index_name", "reader"],
        )
        .unwrap();

        let cdc_reader_restarts_total = CounterVec::new(
            prometheus::Opts::new(
                "cdc_reader_restarts_total",
                "Total number of CDC reader restart attempts after an error, per index and reader",
            ),
            &["keyspace", "index_name", "reader"],
        )
        .unwrap();

        let cdc_last_processed_timestamp_seconds = GaugeVec::new(
            prometheus::Opts::new(
                "cdc_last_processed_timestamp_seconds",
                "Unix timestamp (seconds) up to which the CDC log has been fully consumed. \
                 This is the reader's checkpoint position, not the wall-clock time of the last mutation.",
            ),
            &["keyspace", "index_name", "reader"],
        )
        .unwrap();

        let fts_index_size_bytes = GaugeVec::new(
            prometheus::Opts::new(
                "fts_index_size_bytes",
                "Total size of a full-text search index (bytes)",
            ),
            &["keyspace", "index_name"],
        )
        .unwrap();

        let fts_segment_count = GaugeVec::new(
            prometheus::Opts::new(
                "fts_segment_count",
                "Number of segments in a full-text search index",
            ),
            &["keyspace", "index_name"],
        )
        .unwrap();

        let pattern_index_size_bytes = GaugeVec::new(
            prometheus::Opts::new(
                "pattern_index_size_bytes",
                "Total size of a pattern search index (bytes)",
            ),
            &["keyspace", "index_name"],
        )
        .unwrap();

        let pattern_segment_count = GaugeVec::new(
            prometheus::Opts::new(
                "pattern_segment_count",
                "Number of segments in a pattern search index",
            ),
            &["keyspace", "index_name"],
        )
        .unwrap();

        let pattern_search_totals = PATTERN_SEARCH_TOTALS
            .iter()
            .map(|(name, help)| {
                GaugeVec::new(
                    prometheus::Opts::new(*name, *help),
                    &["keyspace", "index_name"],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let pattern_rewrite = PATTERN_REWRITE_GAUGES
            .iter()
            .map(|(name, help)| {
                GaugeVec::new(
                    prometheus::Opts::new(*name, *help),
                    &["keyspace", "index_name"],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let pattern_segment_layout = PATTERN_SEGMENT_GAUGES
            .iter()
            .map(|(name, help)| {
                GaugeVec::new(
                    prometheus::Opts::new(*name, *help),
                    &["keyspace", "index_name", "segment"],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        registry.register(Box::new(latency.clone())).unwrap();
        registry.register(Box::new(size.clone())).unwrap();
        registry.register(Box::new(modified.clone())).unwrap();
        registry.register(Box::new(indexing_lag.clone())).unwrap();
        registry.register(Box::new(cdc_reader_up.clone())).unwrap();
        registry
            .register(Box::new(cdc_handler_errors_total.clone()))
            .unwrap();
        registry
            .register(Box::new(cdc_reader_restarts_total.clone()))
            .unwrap();
        registry
            .register(Box::new(cdc_last_processed_timestamp_seconds.clone()))
            .unwrap();
        registry
            .register(Box::new(fts_index_size_bytes.clone()))
            .unwrap();
        registry
            .register(Box::new(fts_segment_count.clone()))
            .unwrap();
        registry
            .register(Box::new(pattern_index_size_bytes.clone()))
            .unwrap();
        registry
            .register(Box::new(pattern_segment_count.clone()))
            .unwrap();
        for gauge in pattern_search_totals
            .iter()
            .chain(pattern_rewrite.iter())
            .chain(pattern_segment_layout.iter())
        {
            registry.register(Box::new(gauge.clone())).unwrap();
        }

        Self {
            registry,
            latency,
            size,
            modified,
            indexing_lag,
            cdc_reader_up,
            cdc_handler_errors_total,
            cdc_reader_restarts_total,
            cdc_last_processed_timestamp_seconds,
            fts_index_size_bytes,
            fts_segment_count,
            pattern_index_size_bytes,
            pattern_segment_count,
            pattern_search_totals,
            pattern_rewrite,
            pattern_segment_layout,
            dirty_indexes: Arc::new(DashSet::new()),
            pattern_segments_exported: Arc::new(DashMap::new()),
        }
    }

    /// Replace the segment label sets of one pattern index with `segments`, dropping the
    /// ordinals a merge has retired since the last scrape.
    pub(crate) fn set_pattern_segments(
        &self,
        keyspace: &str,
        index_name: &str,
        segments: &[[Option<f64>; 3]],
    ) {
        let key = (keyspace.to_owned(), index_name.to_owned());
        let previous = self
            .pattern_segments_exported
            .insert(key, segments.len())
            .unwrap_or(0);
        for (ordinal, values) in segments.iter().enumerate() {
            let ordinal = ordinal.to_string();
            let labels = [keyspace, index_name, ordinal.as_str()];
            for (gauge, value) in self.pattern_segment_layout.iter().zip(values) {
                match value {
                    Some(value) => gauge.with_label_values(&labels).set(*value),
                    None => _ = gauge.remove_label_values(&labels),
                }
            }
        }
        for ordinal in segments.len()..previous {
            let ordinal = ordinal.to_string();
            for gauge in &self.pattern_segment_layout {
                let _ = gauge.remove_label_values(&[keyspace, index_name, ordinal.as_str()]);
            }
        }
    }

    pub(crate) fn mark_dirty(&self, keyspace: &str, index_name: &str) {
        self.dirty_indexes
            .insert((keyspace.to_owned(), index_name.to_owned()));
    }
    pub(crate) fn take_dirty_indexes(&self) -> Vec<(String, String)> {
        // Collect, then remove.
        let keys: Vec<_> = self
            .dirty_indexes
            .iter()
            .map(|entry| entry.clone())
            .collect();
        for k in &keys {
            self.dirty_indexes.remove(k);
        }
        keys
    }

    pub(crate) fn remove_index_labels(&self, keyspace: &str, index_name: &str) {
        for op in REQUEST_OPERATIONS {
            let _ = self
                .latency
                .remove_label_values(&[keyspace, index_name, op]);
        }
        let _ = self.size.remove_label_values(&[keyspace, index_name]);
        for reader in [crate::db_cdc::READER_FINE, crate::db_cdc::READER_WIDE] {
            let _ = self
                .indexing_lag
                .remove_label_values(&[keyspace, index_name, reader]);
        }
        let _ = self
            .fts_index_size_bytes
            .remove_label_values(&[keyspace, index_name]);
        let _ = self
            .fts_segment_count
            .remove_label_values(&[keyspace, index_name]);
        let _ = self
            .pattern_index_size_bytes
            .remove_label_values(&[keyspace, index_name]);
        let _ = self
            .pattern_segment_count
            .remove_label_values(&[keyspace, index_name]);
        for gauge in self
            .pattern_search_totals
            .iter()
            .chain(self.pattern_rewrite.iter())
        {
            let _ = gauge.remove_label_values(&[keyspace, index_name]);
        }
        self.set_pattern_segments(keyspace, index_name, &[]);
        for op in OPERATIONS {
            let _ = self
                .modified
                .remove_label_values(&[keyspace, index_name, op]);
        }
        self.dirty_indexes
            .remove(&(keyspace.to_owned(), index_name.to_owned()));
    }

    pub(crate) fn remove_reader_labels(&self, keyspace: &str, index_name: &str, reader: &str) {
        let _ = self
            .cdc_reader_up
            .remove_label_values(&[keyspace, index_name, reader]);
        let _ = self
            .cdc_handler_errors_total
            .remove_label_values(&[keyspace, index_name, reader]);
        let _ = self
            .cdc_reader_restarts_total
            .remove_label_values(&[keyspace, index_name, reader]);
        let _ = self
            .cdc_last_processed_timestamp_seconds
            .remove_label_values(&[keyspace, index_name, reader]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db_cdc::READER_FINE;
    use crate::db_cdc::READER_WIDE;
    use prometheus::Encoder;
    use prometheus::TextEncoder;

    fn metric_families_text(metrics: &Metrics) -> String {
        let mut buf = Vec::new();
        TextEncoder::new()
            .encode(&metrics.registry.gather(), &mut buf)
            .unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn remove_index_labels_clears_all_metric_series_for_index() {
        let metrics = Metrics::new();

        metrics.size.with_label_values(&["ks", "idx"]).set(10.0);
        metrics
            .modified
            .with_label_values(&["ks", "idx", OP_INSERT])
            .inc();
        metrics
            .latency
            .with_label_values(&["ks", "idx", OP_BM25])
            .observe(0.001);
        metrics
            .indexing_lag
            .with_label_values(&["ks", "idx", READER_FINE])
            .observe(0.001);
        metrics
            .latency
            .with_label_values(&["ks", "idx", OP_HIGHLIGHT])
            .observe(0.001);

        metrics.remove_index_labels("ks", "idx");

        let output = metric_families_text(&metrics);
        assert!(
            !output.contains(r#"keyspace="ks""#),
            "metric output should not contain labels for deleted index, got:\n{output}"
        );
    }

    #[test]
    fn pattern_search_totals_and_segment_layout_are_exported() {
        let metrics = Metrics::new();

        metrics.pattern_search_totals[0]
            .with_label_values(&["ks", "idx"])
            .set(7.0);
        metrics.set_pattern_segments(
            "ks",
            "idx",
            &[[Some(10.0), Some(1.0), Some(5.0)], [Some(3.0), None, None]],
        );

        let output = metric_families_text(&metrics);
        assert!(
            output.contains(r#"pattern_search_total{index_name="idx",keyspace="ks"} 7"#),
            "search total missing from export:\n{output}"
        );
        assert!(
            output
                .contains(r#"pattern_segment_docs{index_name="idx",keyspace="ks",segment="0"} 10"#),
            "segment 0 docs missing from export:\n{output}"
        );
        assert!(
            output.contains(
                r#"pattern_segment_sort_max{index_name="idx",keyspace="ks",segment="0"} 5"#
            ),
            "segment 0 sort max missing from export:\n{output}"
        );
        assert!(
            output
                .contains(r#"pattern_segment_docs{index_name="idx",keyspace="ks",segment="1"} 3"#),
            "segment 1 docs missing from export:\n{output}"
        );
        assert!(
            !output.contains(
                r#"pattern_segment_sort_min{index_name="idx",keyspace="ks",segment="1"}"#
            ),
            "an unordered segment must export no bounds:\n{output}"
        );
    }

    #[test]
    fn set_pattern_segments_drops_retired_ordinals() {
        let metrics = Metrics::new();
        let segment = |docs| [Some(docs), Some(0.0), Some(1.0)];

        metrics.set_pattern_segments("ks", "idx", &[segment(1.0), segment(2.0), segment(3.0)]);
        metrics.set_pattern_segments("ks", "idx", &[segment(6.0)]);

        let output = metric_families_text(&metrics);
        assert!(
            output.contains(r#"segment="0"} 6"#),
            "the merged segment should be exported:\n{output}"
        );
        assert!(
            !output.contains(r#"segment="1""#) && !output.contains(r#"segment="2""#),
            "retired ordinals should be gone:\n{output}"
        );

        metrics.remove_index_labels("ks", "idx");
        assert!(
            !metric_families_text(&metrics).contains("pattern_segment"),
            "removing the index should clear its segments"
        );
    }

    #[test]
    fn remove_index_labels_is_noop_when_index_has_no_metrics() {
        let metrics = Metrics::new();
        metrics.remove_index_labels("ks", "nonexistent");
        assert!(metric_families_text(&metrics).is_empty());
    }

    #[test]
    fn remove_index_labels_only_removes_target_index() {
        let metrics = Metrics::new();

        metrics.size.with_label_values(&["ks", "idx1"]).set(1.0);
        metrics.size.with_label_values(&["ks", "idx2"]).set(2.0);

        metrics.remove_index_labels("ks", "idx1");

        let output = metric_families_text(&metrics);
        assert!(
            !output.contains(r#"index_name="idx1""#),
            "idx1 labels should be removed"
        );
        assert!(
            output.contains(r#"index_name="idx2""#),
            "idx2 labels should remain"
        );
    }

    #[test]
    fn remove_index_labels_clears_dirty_index_entry() {
        let metrics = Metrics::new();

        metrics.mark_dirty("ks", "idx");
        assert_eq!(metrics.dirty_indexes.len(), 1);

        metrics.remove_index_labels("ks", "idx");
        assert!(metrics.dirty_indexes.is_empty());
    }

    #[test]
    fn indexing_lag_is_observed_and_exported() {
        use crate::AsyncInProgress;
        use crate::Timestamp;

        let metrics = Metrics::new();

        // Create and immediately drop a Cdc marker to trigger observation.
        let histogram = metrics
            .indexing_lag
            .with_label_values(&["ks", "idx", READER_FINE]);
        drop(AsyncInProgress::cdc(histogram.clone(), Timestamp::MIN));

        assert_eq!(histogram.get_sample_count(), 1);
        assert!(histogram.get_sample_sum() > 0.0);

        // The metric must be exported through the same registry the `/metrics`
        // endpoint scrapes, with the expected name, type, and labels.
        let output = metric_families_text(&metrics);
        assert!(
            output.contains("# TYPE indexing_lag_seconds histogram"),
            "indexing_lag_seconds histogram missing from export:\n{output}"
        );
        assert!(
            output.contains(r#"keyspace="ks""#) && output.contains(r#"index_name="idx""#),
            "expected keyspace/index_name labels in export:\n{output}"
        );
        assert!(
            output.contains(
                r#"indexing_lag_seconds_count{index_name="idx",keyspace="ks",reader="fine"} 1"#
            ),
            "expected a single observed sample in export:\n{output}"
        );
    }

    #[test]
    fn indexing_lag_not_exported_until_observed() {
        let metrics = Metrics::new();

        // A histogram with no observations is still registered, but reports a
        // zero sample count for any label set.
        assert_eq!(
            metrics
                .indexing_lag
                .with_label_values(&["ks", "idx", READER_FINE])
                .get_sample_count(),
            0
        );
    }

    #[test]
    fn cdc_reader_up_is_exported() {
        let metrics = Metrics::new();

        metrics
            .cdc_reader_up
            .with_label_values(&["ks", "idx", READER_WIDE])
            .set(1.0);

        let output = metric_families_text(&metrics);
        assert!(
            output.contains("# TYPE cdc_reader_up gauge"),
            "cdc_reader_up gauge missing from export:\n{output}"
        );
        assert!(
            output.contains(r#"cdc_reader_up{index_name="idx",keyspace="ks",reader="wide"} 1"#),
            "expected cdc_reader_up=1 for wide reader in export:\n{output}"
        );
    }

    #[test]
    fn cdc_handler_errors_total_is_exported() {
        let metrics = Metrics::new();

        metrics
            .cdc_handler_errors_total
            .with_label_values(&["ks", "idx", READER_FINE])
            .inc();

        let output = metric_families_text(&metrics);
        assert!(
            output.contains("# TYPE cdc_handler_errors_total counter"),
            "cdc_handler_errors_total counter missing from export:\n{output}"
        );
        assert!(
            output.contains(
                r#"cdc_handler_errors_total{index_name="idx",keyspace="ks",reader="fine"} 1"#
            ),
            "expected a single observed handler error in export:\n{output}"
        );
    }

    #[test]
    fn cdc_reader_restarts_total_is_exported() {
        let metrics = Metrics::new();

        metrics
            .cdc_reader_restarts_total
            .with_label_values(&["ks", "idx", READER_FINE])
            .inc_by(2.0);

        let output = metric_families_text(&metrics);
        assert!(
            output.contains("# TYPE cdc_reader_restarts_total counter"),
            "cdc_reader_restarts_total counter missing from export:\n{output}"
        );
        assert!(
            output.contains(
                r#"cdc_reader_restarts_total{index_name="idx",keyspace="ks",reader="fine"} 2"#
            ),
            "expected two observed restarts in export:\n{output}"
        );
    }

    #[test]
    fn remove_index_labels_does_not_clear_cdc_reader_metrics() {
        let metrics = Metrics::new();

        // Non-CDC index-scoped metric; must be cleared.
        metrics.size.with_label_values(&["ks", "idx"]).set(1.0);

        // CDC reader metrics are owned by the CDC actor; remove_index_labels must not touch them.
        metrics
            .cdc_reader_up
            .with_label_values(&["ks", "idx", READER_WIDE])
            .set(1.0);

        metrics.remove_index_labels("ks", "idx");

        let output = metric_families_text(&metrics);
        assert!(
            !output.contains(r#"index_size{index_name="idx",keyspace="ks"}"#),
            "index_size should be removed, got:\n{output}"
        );
        assert!(
            output.contains(r#"cdc_reader_up{index_name="idx",keyspace="ks",reader="wide"} 1"#),
            "cdc_reader_up must survive remove_index_labels, got:\n{output}"
        );
    }

    #[test]
    fn cdc_last_processed_timestamp_seconds_is_exported() {
        let metrics = Metrics::new();

        metrics
            .cdc_last_processed_timestamp_seconds
            .with_label_values(&["ks", "idx", READER_WIDE])
            .set(1_700_000_000.0);

        let output = metric_families_text(&metrics);
        assert!(
            output.contains("# TYPE cdc_last_processed_timestamp_seconds gauge"),
            "cdc_last_processed_timestamp_seconds gauge missing from export:\n{output}"
        );
        assert!(
            output.contains(
                r#"cdc_last_processed_timestamp_seconds{index_name="idx",keyspace="ks",reader="wide"} 1700000000"#
            ),
            "expected observed value in export:\n{output}"
        );
    }

    #[test]
    fn remove_reader_labels_clears_only_the_given_readers_cdc_metrics() {
        let metrics = Metrics::new();

        for reader in [READER_WIDE, READER_FINE] {
            metrics
                .cdc_reader_up
                .with_label_values(&["ks", "idx", reader])
                .set(1.0);
            metrics
                .cdc_handler_errors_total
                .with_label_values(&["ks", "idx", reader])
                .inc();
            metrics
                .cdc_reader_restarts_total
                .with_label_values(&["ks", "idx", reader])
                .inc();
            metrics
                .cdc_last_processed_timestamp_seconds
                .with_label_values(&["ks", "idx", reader])
                .set(1_700_000_000.0);
        }

        metrics.remove_reader_labels("ks", "idx", READER_WIDE);

        let output = metric_families_text(&metrics);
        assert!(
            !output.contains(r#"reader="wide""#),
            "the wide reader's CDC metric series should be removed, got:\n{output}"
        );
        assert!(
            output.contains(r#"reader="fine""#),
            "the fine reader's CDC metric series must be unaffected, got:\n{output}"
        );
    }

    #[test]
    fn fts_index_size_bytes_is_exported() {
        let metrics = Metrics::new();

        metrics
            .fts_index_size_bytes
            .with_label_values(&["ks", "idx"])
            .set(1024.0);

        let output = metric_families_text(&metrics);
        assert!(
            output.contains("# TYPE fts_index_size_bytes gauge"),
            "fts_index_size_bytes gauge missing from export:\n{output}"
        );
        assert!(
            output.contains(r#"fts_index_size_bytes{index_name="idx",keyspace="ks"} 1024"#),
            "expected observed value in export:\n{output}"
        );
    }

    #[test]
    fn pattern_gauges_are_exported_and_removed_with_the_index() {
        let metrics = Metrics::new();

        metrics
            .pattern_index_size_bytes
            .with_label_values(&["ks", "idx"])
            .set(2048.0);
        metrics
            .pattern_segment_count
            .with_label_values(&["ks", "idx"])
            .set(2.0);

        let output = metric_families_text(&metrics);
        assert!(
            output.contains("# TYPE pattern_index_size_bytes gauge"),
            "pattern_index_size_bytes gauge missing from export:\n{output}"
        );
        assert!(
            output.contains(r#"pattern_index_size_bytes{index_name="idx",keyspace="ks"} 2048"#),
            "expected observed value in export:\n{output}"
        );
        assert!(
            output.contains(r#"pattern_segment_count{index_name="idx",keyspace="ks"} 2"#),
            "expected observed value in export:\n{output}"
        );

        metrics.remove_index_labels("ks", "idx");

        let output = metric_families_text(&metrics);
        assert!(
            !output.contains(r#"index_name="idx""#),
            "the index's series should be removed, got:\n{output}"
        );
    }

    #[test]
    fn fts_segment_count_is_exported() {
        let metrics = Metrics::new();

        metrics
            .fts_segment_count
            .with_label_values(&["ks", "idx"])
            .set(3.0);

        let output = metric_families_text(&metrics);
        assert!(
            output.contains("# TYPE fts_segment_count gauge"),
            "fts_segment_count gauge missing from export:\n{output}"
        );
        assert!(
            output.contains(r#"fts_segment_count{index_name="idx",keyspace="ks"} 3"#),
            "expected observed value in export:\n{output}"
        );
    }

    #[test]
    fn request_latency_is_exported() {
        let metrics = Metrics::new();

        for op in REQUEST_OPERATIONS {
            metrics
                .latency
                .with_label_values(&["ks", "idx", op])
                .observe(0.001);
        }

        let output = metric_families_text(&metrics);

        assert!(
            output.contains("# TYPE request_latency_seconds histogram"),
            "request_latency_seconds histogram missing from export:\n{output}"
        );
        // Listed explicitly so that dropping an operation from REQUEST_OPERATIONS fails the test.
        for op in ["ann", "bm25", "highlight", "like"] {
            assert!(
                output.contains(&format!(
                    r#"request_latency_seconds_count{{index_name="idx",keyspace="ks",operation="{op}"}} 1"#
                )),
                "expected a single {op} latency sample in export:\n{output}"
            );
        }
    }

    #[test]
    fn remove_index_labels_clears_fts_metrics() {
        let metrics = Metrics::new();

        metrics
            .fts_index_size_bytes
            .with_label_values(&["ks", "idx"])
            .set(1024.0);
        metrics
            .fts_segment_count
            .with_label_values(&["ks", "idx"])
            .set(3.0);

        metrics.remove_index_labels("ks", "idx");

        let output = metric_families_text(&metrics);
        assert!(
            !output.contains(r#"keyspace="ks""#),
            "metric output should not contain labels for deleted index, got:\n{output}"
        );
    }
}
