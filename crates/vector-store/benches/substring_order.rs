/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! What ordering a substring search by a sort column costs, as a function of how many documents
//! the keyword matches.
//!
//! Stage 1 of the substring index answers `LIKE '%kw%' LIMIT n` in unspecified order, which lets
//! it stop walking as soon as it has `n` matches: a hot keyword costs the same as a rare one.
//! Stage 2 wants `ORDER BY <column> DESC`, and the obvious way to get it -- a `FAST` field and
//! `TopDocs::order_by_fast_field` -- cannot stop early, because the newest `n` matches are only
//! known once every match has been looked at. Tantivy reserves its pruning collector
//! (`for_each_pruning`, Block-WAND) for sorting by score; every other sort key falls through to
//! `default_collect_segment_impl`, which is a plain `for_each` over the whole docset.
//!
//! So the ordered variants here are expected to be linear in the match count while the unordered
//! one is flat. The question this benchmark answers is not whether that is true but what the
//! constant is, and what can be done about it. The answers, at 2M names: ordering costs about
//! 13.5 ns per match unverified and 580--7,700 ns per match verified; restricting the order to the
//! newest hundredth of the corpus is about 100 times cheaper and tracks the window rather than the
//! match count; and moving verification off the document store makes it worse, not better.
//!
//! Measured at each keyword frequency:
//!
//! * `walk_unordered` -- stage 1 as it ships: walk, verify, stop at the limit.
//! * `topdocs_ordered` -- stage 2 for keywords within `max_gram`: one term, ordered top-k.
//! * `walk_ordered_verify` -- stage 2 for keywords past `max_gram`: every candidate must be read
//!   from the store and verified before the top-k is known, so this is the case that turns bounded
//!   store reads into unbounded ones. Compared against `walk_unordered_verify`, the same query
//!   shape without ordering.
//! * `deep_page` -- page 50 rather than page 1. A cursor on the sort value does not rescue
//!   ordering: for `DESC` by time it excludes only the rows already returned, so every page costs
//!   what the first one did.
//! * `short_keyword_window` / `long_keyword_window` -- ordering only the newest slice of the
//!   corpus, which is what partitioning the index by the sort column would amount to. A `doc_id`
//!   bound stands in for a bucket boundary, and `DocSet::seek` skips the rest over the posting
//!   list's skip lists.
//! * `walk_ordered_verify_fast` -- verifying from a `FAST` text column rather than the document
//!   store. Measured and rejected; behind `SUBSTRING_BENCH_FAST_TEXT` so it stays reproducible.
//!   See `fast_text_enabled`.
//!
//! The corpus is synthetic and built so that match counts are exact rather than estimated: filler
//! characters and keyword characters are drawn from disjoint alphabets, so a keyword occurs in
//! precisely the documents it was planted in and nowhere by accident.
//!
//! Run with the defaults (fast, and enough to see the shape):
//!
//! ```sh
//! cargo bench --bench substring_order
//! ```
//!
//! Run at a size closer to the design target (slower to build, ~30 s for the corpus):
//!
//! ```sh
//! SUBSTRING_BENCH_NAMES=2000000 cargo bench --bench substring_order
//! ```

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use criterion::criterion_group;
use criterion::criterion_main;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::collections::BinaryHeap;
use std::hint::black_box;
use std::sync::LazyLock;
use tantivy::DocAddress;
use tantivy::Index;
use tantivy::IndexReader;
use tantivy::Order;
use tantivy::TantivyDocument;
use tantivy::Term;
use tantivy::collector::TopDocs;
use tantivy::query::BooleanQuery;
use tantivy::query::EnableScoring;
use tantivy::query::Query;
use tantivy::query::TermQuery;
use tantivy::schema::FAST;
use tantivy::schema::Field;
use tantivy::schema::INDEXED;
use tantivy::schema::IndexRecordOption;
use tantivy::schema::STORED;
use tantivy::schema::Schema;
use tantivy::schema::TextFieldIndexing;
use tantivy::schema::TextOptions;
use tantivy::schema::Value;
use tantivy::tokenizer::NgramTokenizer;
use tantivy::tokenizer::TextAnalyzer;

// The index under test, mirroring substring_index::tantivy: the same field names, the same
// tokenizer settings and the same `Basic` record option, plus the `FAST` sort column stage 2
// would add. Anything that differs here would make the numbers describe a different index.
const PRIMARY_ID_FIELD: &str = "primary_id";
const TEXT_FIELD: &str = "text";
const SORT_FIELD: &str = "sort_key";
const TOKENIZER_NAME: &str = "substring_ngram";
const MIN_GRAM: usize = 1;
const MAX_GRAM: usize = 3;
const STORE_CACHE_BLOCKS: usize = 1;

/// The page a search box asks for.
const LIMIT: usize = 20;
/// Deep-paging probe: far enough in that offset paging is visibly doing extra work.
const DEEP_PAGE: usize = 50;

const DEFAULT_NAMES: usize = 500_000;
const NAME_LEN: usize = 12;
const WRITER_HEAP_BYTES: usize = 256 << 20;

/// Share of the corpus each planted keyword matches. The top of the range is what a one- or
/// two-character CJK keyword really does to a display-name corpus; the bottom is a rare keyword,
/// where ordering is affordable whatever the algorithm.
const FREQUENCIES: [f64; 5] = [0.0001, 0.001, 0.01, 0.05, 0.20];

/// Filler characters, drawn from CJK Unified Ideographs. Names are built only from these, so they
/// can never accidentally contain a keyword.
const FILLER: &str =
    "的一是不了人我在有他这为之大来以个中上们到国说和地也子时道出而要于就下得可你年生";

/// Keyword characters, drawn from Hangul syllables -- disjoint from FILLER, so a planted keyword
/// occurs exactly as often as it was planted.
const KEYWORD_ALPHABET: &str = "가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허";

/// Whether to add the `FAST` text column that `walk_ordered_verified_fast` needs.
///
/// Measured and rejected: the column is dictionary-encoded and every name is distinct, so each
/// lookup is an sstable seek into a dictionary with one entry per document, at a flat ~7,700 ns per
/// candidate. Reading the document store instead costs 580--5,600 ns and gets cheaper as matches
/// get denser, because the walk picks up block locality the column has no equivalent of. At 400,000
/// matches that is 233 ms against 3,088 ms. Kept behind a flag rather than deleted so the negative
/// result can be re-checked against a later Tantivy.
fn fast_text_enabled() -> bool {
    std::env::var("SUBSTRING_BENCH_FAST_TEXT").is_ok_and(|v| v != "0")
}

/// Documents per commit, which is what decides how many segments the index has and therefore how
/// finely a query can prune. 0 keeps Tantivy's own behaviour, where a segment is cut whenever the
/// writer's per-thread budget fills. Anything else also switches the merge policy off, so the
/// segment count is exactly what was asked for rather than whatever merging leaves behind.
fn segment_docs() -> usize {
    std::env::var("SUBSTRING_BENCH_SEGMENT_DOCS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Whether the sort column disagrees with insertion order.
///
/// Live traffic arrives in registration order, so each segment covers a narrow span and a query can
/// skip most of them. The initial CDC backfill does not: it reads per stream per token range, which
/// is uncorrelated with registration time, so every segment spans nearly the whole range and
/// nothing can be skipped. This models the second case.
fn shuffled_sort_keys() -> bool {
    std::env::var("SUBSTRING_BENCH_SHUFFLE").is_ok_and(|v| v != "0")
}

fn corpus_size() -> usize {
    std::env::var("SUBSTRING_BENCH_NAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_NAMES)
}

/// A keyword and the number of documents it was planted in.
struct Planted {
    keyword: String,
    matches: usize,
}

struct Corpus {
    _dir: tempfile::TempDir,
    reader: IndexReader,
    schema: Schema,
    /// Keywords of `MAX_GRAM` characters or fewer: one term lookup, no verification.
    short: Vec<Planted>,
    /// Keywords longer than `MAX_GRAM`: a gram intersection whose candidates must be verified.
    long: Vec<Planted>,
}

fn build_schema() -> Schema {
    let indexing = TextFieldIndexing::default()
        .set_tokenizer(TOKENIZER_NAME)
        .set_index_option(IndexRecordOption::Basic);
    // Off by default: the shipped index does not have this column, and adding it inflates the index
    // and slows the other variants by about a fifth. It exists so the rejected alternative stays
    // reproducible -- see `walk_ordered_verified_fast`.
    let mut text_options = TextOptions::default()
        .set_indexing_options(indexing)
        .set_stored();
    if fast_text_enabled() {
        text_options = text_options.set_fast(None);
    }
    let mut builder = Schema::builder();
    builder.add_u64_field(PRIMARY_ID_FIELD, INDEXED | STORED);
    builder.add_text_field(TEXT_FIELD, text_options);
    // The sort column stage 2 would order by: a registration timestamp, which is immutable and
    // therefore safe to sort or partition on, unlike the name itself.
    builder.add_u64_field(SORT_FIELD, FAST | STORED);
    builder.build()
}

/// Builds the corpus once, for every benchmark in this file.
///
/// Each frequency gets its own keyword, planted in exactly `round(frequency * names)` documents
/// chosen by a stride rather than at random, so that matches are spread evenly across segments
/// instead of clustering in whichever ones happened to be written first.
fn build_corpus() -> Corpus {
    let names = corpus_size();
    let chars: Vec<char> = FILLER.chars().collect();
    let kw_chars: Vec<char> = KEYWORD_ALPHABET.chars().collect();

    let mut short = Vec::new();
    let mut long = Vec::new();
    for (i, frequency) in FREQUENCIES.iter().enumerate() {
        let matches = ((names as f64) * frequency).round() as usize;
        // Two characters: within max_gram, so a single term lookup.
        short.push(Planted {
            keyword: format!("{}{}", kw_chars[i * 2], kw_chars[i * 2 + 1]),
            matches,
        });
        // Four characters: past max_gram, so the query intersects 3-grams and every candidate is
        // verified against the stored text.
        long.push(Planted {
            keyword: format!(
                "{}{}{}{}",
                kw_chars[10 + i * 2],
                kw_chars[10 + i * 2 + 1],
                kw_chars[11 + i * 2],
                kw_chars[10 + i * 2]
            ),
            matches,
        });
    }

    let dir = tempfile::tempdir().expect("bench: failed to create the index directory");
    let schema = build_schema();
    let index = Index::create_in_dir(dir.path(), schema.clone())
        .expect("bench: failed to create the index");
    let tokenizer =
        NgramTokenizer::new(MIN_GRAM, MAX_GRAM, false).expect("bench: bad n-gram range");
    index
        .tokenizers()
        .register(TOKENIZER_NAME, TextAnalyzer::builder(tokenizer).build());

    let primary_id_field = schema.get_field(PRIMARY_ID_FIELD).unwrap();
    let text_field = schema.get_field(TEXT_FIELD).unwrap();
    let sort_field = schema.get_field(SORT_FIELD).unwrap();

    let mut writer = index
        .writer::<TantivyDocument>(WRITER_HEAP_BYTES)
        .expect("bench: failed to create the writer");
    let mut rng = StdRng::seed_from_u64(0x5eed);

    let per_segment = segment_docs();
    if per_segment > 0 {
        // Otherwise the log policy merges by size tier and the segment count stops being the thing
        // the benchmark is varying.
        writer.set_merge_policy(Box::new(tantivy::indexer::NoMergePolicy));
    }

    // The sort column. In order it is the document's own position, so segments come out disjoint
    // and narrow; shuffled, every segment spans nearly the whole range and pruning has nothing to
    // work with.
    let mut sort_keys: Vec<u64> = (0..names as u64).collect();
    if shuffled_sort_keys() {
        for i in (1..names).rev() {
            let j = rng.random_range(0..=i);
            sort_keys.swap(i, j);
        }
    }

    for (doc_id, &sort_key) in sort_keys.iter().enumerate().take(names) {
        // Whichever keywords claim this document. A document can carry more than one, which is what
        // a real corpus does too -- a name holding one common character often holds another.
        let to_plant: Vec<&str> = short
            .iter()
            .chain(long.iter())
            .filter(|planted| {
                if planted.matches == 0 {
                    return false;
                }
                let stride = names / planted.matches;
                stride > 0 && doc_id % stride == 0 && doc_id / stride < planted.matches
            })
            .map(|planted| planted.keyword.as_str())
            .collect();

        // Build the name in one pass, as filler chunks with the keywords between them, rather than
        // by inserting keywords into a finished string: an insertion can land inside a keyword
        // planted earlier and destroy it, which makes a keyword match fewer documents than it was
        // planted in. Every chunk holds at least one filler character, so two keywords are never
        // adjacent and no substring spanning a boundary can look like a planted keyword.
        let chunks = to_plant.len() + 1;
        let filler_len = NAME_LEN.max(chunks);
        let mut chunk_sizes = vec![1usize; chunks];
        for _ in 0..filler_len - chunks {
            chunk_sizes[rng.random_range(0..chunks)] += 1;
        }
        let mut name = String::new();
        for (i, size) in chunk_sizes.iter().enumerate() {
            for _ in 0..*size {
                name.push(chars[rng.random_range(0..chars.len())]);
            }
            if let Some(keyword) = to_plant.get(i) {
                name.push_str(keyword);
            }
        }

        let mut doc = TantivyDocument::new();
        doc.add_u64(primary_id_field, doc_id as u64);
        doc.add_text(text_field, &name);
        // Stand-in for register_time: monotonically increasing, so "newest first" is a meaningful
        // ordering and the top-k is spread across the whole corpus rather than sitting in one spot.
        doc.add_u64(sort_field, sort_key);
        writer
            .add_document(doc)
            .expect("bench: failed to add a doc");
        if per_segment > 0 && (doc_id + 1) % per_segment == 0 {
            writer.commit().expect("bench: failed to commit");
        }
    }
    writer.commit().expect("bench: failed to commit");

    let reader = index.reader().expect("bench: failed to open a reader");
    let bytes: u64 = std::fs::read_dir(dir.path())
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| entry.metadata().ok())
        .map(|meta| meta.len())
        .sum();
    eprintln!(
        "corpus: {names} names, index {:.1} MiB ({:.2} bytes/name), {} segments",
        bytes as f64 / (1 << 20) as f64,
        bytes as f64 / names as f64,
        reader.searcher().segment_readers().len(),
    );
    Corpus {
        _dir: dir,
        reader,
        schema,
        short,
        long,
    }
}

static CORPUS: LazyLock<Corpus> = LazyLock::new(build_corpus);

fn term_query(text_field: Field, gram: &str) -> Box<dyn Query> {
    Box::new(TermQuery::new(
        Term::from_field_text(text_field, gram),
        IndexRecordOption::Basic,
    ))
}

/// The grams that take part in a query longer than `max_gram`, as substring_index builds them.
fn grams_of_length(text: &str, gram_len: usize) -> BTreeSet<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < gram_len {
        return BTreeSet::new();
    }
    (0..=chars.len() - gram_len)
        .map(|start| chars[start..start + gram_len].iter().collect())
        .collect()
}

/// Builds the query substring_index would build, and says whether its candidates need verifying.
fn build_query(text_field: Field, keyword: &str) -> (Box<dyn Query>, bool) {
    if keyword.chars().count() <= MAX_GRAM {
        return (term_query(text_field, keyword), false);
    }
    let clauses = grams_of_length(keyword, MAX_GRAM)
        .iter()
        .map(|gram| term_query(text_field, gram))
        .collect();
    (Box::new(BooleanQuery::intersection(clauses)), true)
}

/// Stage 1: walk the docset in index order and stop as soon as `limit` verified matches have been
/// found past `offset`. This is `substring_index::tantivy::collect_matches`, reproduced so the
/// benchmark measures the shipped algorithm rather than an approximation of it.
fn walk_unordered(corpus: &Corpus, keyword: &str, limit: usize, offset: usize) -> Vec<u64> {
    let text_field = corpus.schema.get_field(TEXT_FIELD).unwrap();
    let primary_id_field = corpus.schema.get_field(PRIMARY_ID_FIELD).unwrap();
    let (query, needs_verification) = build_query(text_field, keyword);

    let searcher = corpus.reader.searcher();
    let weight = query
        .weight(EnableScoring::disabled_from_searcher(&searcher))
        .expect("bench: failed to build the weight");

    let mut to_skip = offset;
    let mut matches = Vec::with_capacity(limit);
    'segments: for segment in searcher.segment_readers() {
        let mut scorer = weight.scorer(segment, 1.0).expect("bench: failed to score");
        let alive = segment.alive_bitset();
        let store = segment
            .get_store_reader(STORE_CACHE_BLOCKS)
            .expect("bench: failed to open the store");

        let mut doc_id = scorer.doc();
        while doc_id != tantivy::TERMINATED {
            if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                let doc: TantivyDocument = store.get(doc_id).expect("bench: failed to read a doc");
                let verified = !needs_verification
                    || doc
                        .get_first(text_field)
                        .and_then(|value| value.as_str())
                        .is_some_and(|text| text.contains(keyword));
                if verified {
                    if to_skip > 0 {
                        to_skip -= 1;
                    } else {
                        let id = doc
                            .get_first(primary_id_field)
                            .and_then(|value| value.as_u64())
                            .expect("bench: missing primary id");
                        matches.push(id);
                        if matches.len() == limit {
                            break 'segments;
                        }
                    }
                }
            }
            doc_id = scorer.advance();
        }
    }
    matches
}

/// Stage 2, for a keyword no longer than `max_gram`: no verification is needed, so the ordering
/// can be left entirely to Tantivy's top-k collector over the `FAST` sort field.
fn topdocs_ordered(corpus: &Corpus, keyword: &str, limit: usize, offset: usize) -> usize {
    let text_field = corpus.schema.get_field(TEXT_FIELD).unwrap();
    let (query, needs_verification) = build_query(text_field, keyword);
    assert!(
        !needs_verification,
        "bench: topdocs_ordered is only valid for keywords within max_gram"
    );

    let searcher = corpus.reader.searcher();
    let collector = TopDocs::with_limit(limit)
        .and_offset(offset)
        .order_by_fast_field::<u64>(SORT_FIELD, Order::Desc);
    let hits: Vec<(Option<u64>, DocAddress)> = searcher
        .search(&query, &collector)
        .expect("bench: failed to search");
    hits.len()
}

/// Stage 2, for a keyword longer than `max_gram`: verification decides membership, so it has to
/// happen before the top-k is known. Every candidate is read from the store and checked, and only
/// then does it compete for a place in the page -- the store reads that stage 1 bounds by the
/// limit are unbounded here.
fn walk_ordered_verified(corpus: &Corpus, keyword: &str, limit: usize, offset: usize) -> usize {
    let text_field = corpus.schema.get_field(TEXT_FIELD).unwrap();
    let primary_id_field = corpus.schema.get_field(PRIMARY_ID_FIELD).unwrap();
    let sort_field_name = SORT_FIELD;
    let (query, needs_verification) = build_query(text_field, keyword);

    let searcher = corpus.reader.searcher();
    let weight = query
        .weight(EnableScoring::disabled_from_searcher(&searcher))
        .expect("bench: failed to build the weight");

    // A bounded max-heap would do; a Vec plus a partial sort keeps the benchmark honest about the
    // part that costs, which is visiting and verifying every candidate rather than the top-k.
    let mut scored: Vec<(u64, u64)> = Vec::new();
    for segment in searcher.segment_readers() {
        let mut scorer = weight.scorer(segment, 1.0).expect("bench: failed to score");
        let alive = segment.alive_bitset();
        let store = segment
            .get_store_reader(STORE_CACHE_BLOCKS)
            .expect("bench: failed to open the store");
        let sort_column = segment
            .fast_fields()
            .u64(sort_field_name)
            .expect("bench: missing the sort fast field");

        let mut doc_id = scorer.doc();
        while doc_id != tantivy::TERMINATED {
            if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                let doc: TantivyDocument = store.get(doc_id).expect("bench: failed to read a doc");
                let verified = !needs_verification
                    || doc
                        .get_first(text_field)
                        .and_then(|value| value.as_str())
                        .is_some_and(|text| text.contains(keyword));
                if verified {
                    let id = doc
                        .get_first(primary_id_field)
                        .and_then(|value| value.as_u64())
                        .expect("bench: missing primary id");
                    let sort_key = sort_column.first(doc_id).unwrap_or(0);
                    scored.push((sort_key, id));
                }
            }
            doc_id = scorer.advance();
        }
    }
    scored.sort_unstable_by_key(|(sort_key, _)| std::cmp::Reverse(*sort_key));
    scored.into_iter().skip(offset).take(limit).count()
}

/// Stage 2 past `max_gram`, verifying from the `FAST` text column instead of the document store.
///
/// Identical to `walk_ordered_verified` except for where the text comes from. The store read is
/// what makes that function cost roughly 545 ns per candidate rather than the 13.5 ns the
/// unverified path pays, so this asks whether a columnar read is cheaper. It is not obvious that it
/// is: the str column is dictionary-encoded and every name is distinct, so the dictionary has as
/// many entries as the corpus has documents and each lookup is an sstable seek.
fn walk_ordered_verified_fast(
    corpus: &Corpus,
    keyword: &str,
    limit: usize,
    offset: usize,
) -> usize {
    let text_field = corpus.schema.get_field(TEXT_FIELD).unwrap();
    let (query, needs_verification) = build_query(text_field, keyword);

    let searcher = corpus.reader.searcher();
    let weight = query
        .weight(EnableScoring::disabled_from_searcher(&searcher))
        .expect("bench: failed to build the weight");

    let mut scored: Vec<(u64, u32)> = Vec::new();
    for segment in searcher.segment_readers() {
        let mut scorer = weight.scorer(segment, 1.0).expect("bench: failed to score");
        let alive = segment.alive_bitset();
        let sort_column = segment
            .fast_fields()
            .u64(SORT_FIELD)
            .expect("bench: missing the sort fast field");
        let text_column = segment
            .fast_fields()
            .str(TEXT_FIELD)
            .expect("bench: failed to open the text column")
            .expect("bench: missing the text fast field");
        let mut buffer = String::new();

        let mut doc_id = scorer.doc();
        while doc_id != tantivy::TERMINATED {
            if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                let verified = !needs_verification || {
                    let mut found = false;
                    for ord in text_column.term_ords(doc_id) {
                        buffer.clear();
                        text_column
                            .ord_to_str(ord, &mut buffer)
                            .expect("bench: failed to read the text column");
                        if buffer.contains(keyword) {
                            found = true;
                            break;
                        }
                    }
                    found
                };
                if verified {
                    let sort_key = sort_column.first(doc_id).unwrap_or(0);
                    scored.push((sort_key, doc_id));
                }
            }
            doc_id = scorer.advance();
        }
    }
    scored.sort_unstable_by_key(|(sort_key, _)| std::cmp::Reverse(*sort_key));
    scored.into_iter().skip(offset).take(limit).count()
}

/// What one bucket of a time-partitioned index would cost.
///
/// Bucketing works because a document's bucket is decided by the sort column, so the newest bucket
/// holds the newest documents and ordering it answers page one without the rest of the corpus being
/// looked at. This simulates that without building separate indexes: documents are written in sort
/// order, so a `doc_id` lower bound is the same restriction a bucket boundary would be, and
/// `DocSet::seek` jumps there over the posting list's skip lists rather than advancing through it.
///
/// The measurement to take from this is not the absolute number but that it tracks the matches
/// inside the window rather than the matches in the corpus.
fn windowed_ordered(corpus: &Corpus, keyword: &str, limit: usize, window: f64) -> usize {
    let text_field = corpus.schema.get_field(TEXT_FIELD).unwrap();
    let (query, needs_verification) = build_query(text_field, keyword);

    let searcher = corpus.reader.searcher();
    let weight = query
        .weight(EnableScoring::disabled_from_searcher(&searcher))
        .expect("bench: failed to build the weight");

    let mut scored: Vec<(u64, u32)> = Vec::new();
    for segment in searcher.segment_readers() {
        let max_doc = segment.max_doc();
        let first_in_window = ((max_doc as f64) * (1.0 - window)) as u32;
        let mut scorer = weight.scorer(segment, 1.0).expect("bench: failed to score");
        let alive = segment.alive_bitset();
        let sort_column = segment
            .fast_fields()
            .u64(SORT_FIELD)
            .expect("bench: missing the sort fast field");

        let store = segment
            .get_store_reader(STORE_CACHE_BLOCKS)
            .expect("bench: failed to open the store");

        // The whole point: skip the older part of the posting list rather than walk it.
        let mut doc_id = scorer.seek(first_in_window);
        while doc_id != tantivy::TERMINATED {
            if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                let verified = !needs_verification || {
                    let doc: TantivyDocument =
                        store.get(doc_id).expect("bench: failed to read a doc");
                    doc.get_first(text_field)
                        .and_then(|value| value.as_str())
                        .is_some_and(|text| text.contains(keyword))
                };
                if verified {
                    let sort_key = sort_column.first(doc_id).unwrap_or(0);
                    scored.push((sort_key, doc_id));
                }
            }
            doc_id = scorer.advance();
        }
    }
    scored.sort_unstable_by_key(|(sort_key, _)| std::cmp::Reverse(*sort_key));
    scored.into_iter().take(limit).count()
}

/// Ordering with segment-level pruning: the design that uses Tantivy's own structure instead of
/// partitioning the index.
///
/// A segment is already an immutable bucket with its own postings and columns, and `Column` carries
/// conservative `min_value`/`max_value` bounds for a `FAST` field that can be read without touching
/// the data. So a query can visit segments newest-first, keep a running top-k, and stop as soon as
/// the next segment's upper bound cannot beat the k-th best it already holds.
///
/// This is correct whatever order documents arrived in -- the bounds are always valid, so a stale
/// or shuffled ingestion order costs time and never answers. What insertion order decides is how
/// tight the bounds are, which is what `SUBSTRING_BENCH_SHUFFLE` exists to measure.
fn segment_pruned_ordered(corpus: &Corpus, keyword: &str, limit: usize) -> usize {
    let text_field = corpus.schema.get_field(TEXT_FIELD).unwrap();
    let (query, needs_verification) = build_query(text_field, keyword);

    let searcher = corpus.reader.searcher();
    let weight = query
        .weight(EnableScoring::disabled_from_searcher(&searcher))
        .expect("bench: failed to build the weight");

    let mut segments: Vec<_> = searcher
        .segment_readers()
        .iter()
        .map(|segment| {
            let column = segment
                .fast_fields()
                .u64(SORT_FIELD)
                .expect("bench: missing the sort fast field");
            let upper_bound = column.max_value();
            (segment, column, upper_bound)
        })
        .collect();
    segments.sort_by_key(|(_, _, upper_bound)| std::cmp::Reverse(*upper_bound));

    // Min-heap of the best `limit` seen so far, so the root is the one to beat.
    let mut best: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
    for (segment, sort_column, upper_bound) in segments {
        if best.len() == limit
            && let Some(Reverse((kth, _))) = best.peek()
            && upper_bound <= *kth
        {
            // Nothing in this segment, or any later one, can enter the page.
            break;
        }

        let mut scorer = weight.scorer(segment, 1.0).expect("bench: failed to score");
        let alive = segment.alive_bitset();
        let store = segment
            .get_store_reader(STORE_CACHE_BLOCKS)
            .expect("bench: failed to open the store");

        let mut doc_id = scorer.doc();
        while doc_id != tantivy::TERMINATED {
            if alive.is_none_or(|alive| alive.is_alive(doc_id)) {
                let sort_key = sort_column.first(doc_id).unwrap_or(0);
                // Check the cheap thing first: a candidate that cannot make the page need not be
                // read from the store, which is what makes pruning pay on the verified path too.
                let worth_it = best.len() < limit
                    || best.peek().is_none_or(|Reverse((kth, _))| sort_key > *kth);
                if worth_it {
                    let verified = !needs_verification || {
                        let doc: TantivyDocument =
                            store.get(doc_id).expect("bench: failed to read a doc");
                        doc.get_first(text_field)
                            .and_then(|value| value.as_str())
                            .is_some_and(|text| text.contains(keyword))
                    };
                    if verified {
                        best.push(Reverse((sort_key, doc_id)));
                        if best.len() > limit {
                            best.pop();
                        }
                    }
                }
            }
            doc_id = scorer.advance();
        }
    }
    best.len()
}

/// How tight the segments are: the mean segment's span of the sort column, as a fraction of the
/// whole corpus's span. Near `1 / segments` when ingestion follows the sort column, near 1 when it
/// does not -- and it is the direct predictor of how much `segment_pruned_ordered` can skip.
fn segment_tightness(corpus: &Corpus) -> (usize, f64) {
    let searcher = corpus.reader.searcher();
    let spans: Vec<(u64, u64)> = searcher
        .segment_readers()
        .iter()
        .map(|segment| {
            let column = segment
                .fast_fields()
                .u64(SORT_FIELD)
                .expect("bench: missing the sort fast field");
            (column.min_value(), column.max_value())
        })
        .collect();
    let lowest = spans.iter().map(|(min, _)| *min).min().unwrap_or(0);
    let highest = spans.iter().map(|(_, max)| *max).max().unwrap_or(0);
    let whole = (highest - lowest).max(1) as f64;
    let mean = spans
        .iter()
        .map(|(min, max)| (max - min) as f64 / whole)
        .sum::<f64>()
        / spans.len().max(1) as f64;
    (spans.len(), mean)
}

/// Two-character keywords: the typical search-box query, and the comparison that matters. The
/// unordered walk should be flat across frequencies and the ordered top-k linear in them.
fn bench_short_keywords(c: &mut Criterion) {
    let corpus = &*CORPUS;
    verify_corpus(corpus);
    let names = corpus_size();
    let mut group = c.benchmark_group("short_keyword_page1");
    for planted in &corpus.short {
        let label = format!("{}of{}", planted.matches, names);
        group.throughput(Throughput::Elements(planted.matches as u64));
        group.bench_with_input(
            BenchmarkId::new("walk_unordered", &label),
            &planted.keyword,
            |b, keyword| b.iter(|| black_box(walk_unordered(corpus, keyword, LIMIT, 0))),
        );
        group.bench_with_input(
            BenchmarkId::new("topdocs_ordered", &label),
            &planted.keyword,
            |b, keyword| b.iter(|| black_box(topdocs_ordered(corpus, keyword, LIMIT, 0))),
        );
    }
    group.finish();
}

/// Four-character keywords: past `max_gram`, so every candidate costs a store read. Stage 1 pays
/// for about `limit` of them; ordering makes it pay for all of them.
fn bench_long_keywords(c: &mut Criterion) {
    let corpus = &*CORPUS;
    let names = corpus_size();
    let mut group = c.benchmark_group("long_keyword_page1");
    for planted in &corpus.long {
        let label = format!("{}of{}", planted.matches, names);
        group.throughput(Throughput::Elements(planted.matches as u64));
        group.bench_with_input(
            BenchmarkId::new("walk_unordered_verify", &label),
            &planted.keyword,
            |b, keyword| b.iter(|| black_box(walk_unordered(corpus, keyword, LIMIT, 0))),
        );
        group.bench_with_input(
            BenchmarkId::new("walk_ordered_verify", &label),
            &planted.keyword,
            |b, keyword| b.iter(|| black_box(walk_ordered_verified(corpus, keyword, LIMIT, 0))),
        );
        if fast_text_enabled() {
            group.bench_with_input(
                BenchmarkId::new("walk_ordered_verify_fast", &label),
                &planted.keyword,
                |b, keyword| {
                    b.iter(|| black_box(walk_ordered_verified_fast(corpus, keyword, LIMIT, 0)))
                },
            );
        }
    }
    group.finish();
}

/// Page 50 rather than page 1. Offset paging makes the unordered walk pay for the rows it skips,
/// which is the linear-and-unstable cost stage 1 accepted; the ordered variant was already paying
/// for every match on page 1 and pays it again on every later page, which is why a keyset cursor
/// does not rescue it.
fn bench_deep_page(c: &mut Criterion) {
    let corpus = &*CORPUS;
    let names = corpus_size();
    let offset = DEEP_PAGE * LIMIT;
    let mut group = c.benchmark_group("short_keyword_deep_page");
    for planted in &corpus.short {
        // Only meaningful where there are enough matches to page that far into.
        if planted.matches < offset + LIMIT {
            continue;
        }
        let label = format!("{}of{}", planted.matches, names);
        group.throughput(Throughput::Elements(planted.matches as u64));
        group.bench_with_input(
            BenchmarkId::new("walk_unordered", &label),
            &planted.keyword,
            |b, keyword| b.iter(|| black_box(walk_unordered(corpus, keyword, LIMIT, offset))),
        );
        group.bench_with_input(
            BenchmarkId::new("topdocs_ordered", &label),
            &planted.keyword,
            |b, keyword| b.iter(|| black_box(topdocs_ordered(corpus, keyword, LIMIT, offset))),
        );
    }
    group.finish();
}

/// Checks the corpus is what the labels claim before any timing is reported.
///
/// Every number this benchmark prints is labelled with a match count, and the comparison between
/// ordered and unordered only means anything if both answer the same question. A keyword occurring
/// by accident, a planting skipped, or an ordering that does not actually order would all leave the
/// benchmark running happily and describing a workload nobody intended. With `harness = false`
/// there is no test runner to put this in, so it runs as the first thing the benchmark does.
fn verify_corpus(corpus: &Corpus) {
    let text_field = corpus.schema.get_field(TEXT_FIELD).unwrap();
    let searcher = corpus.reader.searcher();

    for planted in corpus.short.iter().chain(corpus.long.iter()) {
        let (query, needs_verification) = build_query(text_field, &planted.keyword);
        let found = if needs_verification {
            // Verified matches, not gram-intersection candidates.
            walk_ordered_verified(corpus, &planted.keyword, usize::MAX, 0)
        } else {
            searcher
                .search(&query, &tantivy::collector::Count)
                .expect("bench: count failed")
        };
        assert_eq!(
            found, planted.matches,
            "bench: keyword {:?} matches {found} documents but was planted in {}",
            planted.keyword, planted.matches
        );
    }

    if fast_text_enabled() {
        for planted in &corpus.long {
            assert_eq!(
                walk_ordered_verified_fast(corpus, &planted.keyword, usize::MAX, 0),
                planted.matches,
                "bench: verifying from the fast text column disagrees with the document store"
            );
        }
    }

    for planted in &corpus.short {
        let expected = planted.matches.min(LIMIT);
        assert_eq!(
            windowed_ordered(corpus, &planted.keyword, LIMIT, 1.0),
            expected,
            "bench: a full-corpus window must return the same page as no window at all"
        );
        assert_eq!(
            segment_pruned_ordered(corpus, &planted.keyword, LIMIT),
            expected,
            "bench: segment pruning must not drop rows from the page"
        );
        assert_eq!(
            walk_unordered(corpus, &planted.keyword, LIMIT, 0).len(),
            expected,
            "bench: the unordered walk returned the wrong page size"
        );
        assert_eq!(
            topdocs_ordered(corpus, &planted.keyword, LIMIT, 0),
            expected,
            "bench: the ordered top-k returned the wrong page size"
        );
    }

    let planted = corpus
        .short
        .iter()
        .find(|p| p.matches > LIMIT)
        .expect("bench: expected a keyword with more matches than fit on a page");
    let (query, _) = build_query(text_field, &planted.keyword);
    let hits: Vec<(Option<u64>, DocAddress)> = searcher
        .search(
            &query,
            &TopDocs::with_limit(LIMIT).order_by_fast_field::<u64>(SORT_FIELD, Order::Desc),
        )
        .expect("bench: search failed");
    let keys: Vec<u64> = hits.iter().filter_map(|(key, _)| *key).collect();
    assert_eq!(
        keys.len(),
        LIMIT,
        "bench: short page from an ordered search"
    );
    assert!(
        keys.windows(2).all(|pair| pair[0] >= pair[1]),
        "bench: the ordered search did not return the newest first: {keys:?}"
    );
}

/// Ordering the newest slice of the corpus rather than all of it: what bucketing would buy.
///
/// The 1% window is the shape worth designing against -- a bucket holding a hundredth of the corpus
/// turns the hot keyword from "every match" into "every match registered recently", which is the
/// difference between tens of milliseconds and a fraction of one.
fn bench_windowed(c: &mut Criterion) {
    let corpus = &*CORPUS;
    let names = corpus_size();
    let mut group = c.benchmark_group("short_keyword_window");
    for planted in &corpus.short {
        let label = format!("{}of{}", planted.matches, names);
        group.throughput(Throughput::Elements(planted.matches as u64));
        for (name, window) in [
            ("whole_corpus", 1.0),
            ("newest_10pct", 0.1),
            ("newest_1pct", 0.01),
        ] {
            group.bench_with_input(
                BenchmarkId::new(name, &label),
                &planted.keyword,
                |b, keyword| b.iter(|| black_box(windowed_ordered(corpus, keyword, LIMIT, window))),
            );
        }
    }
    group.finish();
}

/// The one case bucketing might not rescue: a keyword past `max_gram` inside a window.
///
/// Windowing cuts how many candidates there are, but each surviving candidate still costs a
/// document store read, and a narrow window makes those reads sparser and so individually dearer --
/// the store-read cost per candidate runs from about 580 ns when matches are dense to 5,600 ns when
/// they are not. Whether the smaller count or the worse locality wins is not something to reason
/// about from the other numbers.
fn bench_windowed_verified(c: &mut Criterion) {
    let corpus = &*CORPUS;
    let names = corpus_size();
    let mut group = c.benchmark_group("long_keyword_window");
    for planted in &corpus.long {
        let label = format!("{}of{}", planted.matches, names);
        group.throughput(Throughput::Elements(planted.matches as u64));
        for (name, window) in [
            ("whole_corpus", 1.0),
            ("newest_10pct", 0.1),
            ("newest_1pct", 0.01),
        ] {
            group.bench_with_input(
                BenchmarkId::new(name, &label),
                &planted.keyword,
                |b, keyword| b.iter(|| black_box(windowed_ordered(corpus, keyword, LIMIT, window))),
            );
        }
    }
    group.finish();
}

/// Segment pruning against no pruning, which is the measurement that decides the design.
fn bench_segment_pruning(c: &mut Criterion) {
    let corpus = &*CORPUS;
    let names = corpus_size();
    let (segments, tightness) = segment_tightness(corpus);
    eprintln!(
        "segments: {segments}, mean span {:.1}% of the corpus ({})",
        tightness * 100.0,
        if shuffled_sort_keys() {
            "shuffled ingestion"
        } else {
            "ingestion follows the sort column"
        },
    );
    let mut group = c.benchmark_group("short_keyword_segment_pruning");
    for planted in &corpus.short {
        let label = format!("{}of{}", planted.matches, names);
        group.throughput(Throughput::Elements(planted.matches as u64));
        group.bench_with_input(
            BenchmarkId::new("no_pruning", &label),
            &planted.keyword,
            |b, keyword| b.iter(|| black_box(topdocs_ordered(corpus, keyword, LIMIT, 0))),
        );
        group.bench_with_input(
            BenchmarkId::new("segment_pruned", &label),
            &planted.keyword,
            |b, keyword| b.iter(|| black_box(segment_pruned_ordered(corpus, keyword, LIMIT))),
        );
    }
    group.finish();
}

/// The same, for keywords past `max_gram`, where a skipped candidate also skips a store read.
fn bench_segment_pruning_verified(c: &mut Criterion) {
    let corpus = &*CORPUS;
    let names = corpus_size();
    let mut group = c.benchmark_group("long_keyword_segment_pruning");
    for planted in &corpus.long {
        let label = format!("{}of{}", planted.matches, names);
        group.throughput(Throughput::Elements(planted.matches as u64));
        group.bench_with_input(
            BenchmarkId::new("no_pruning", &label),
            &planted.keyword,
            |b, keyword| b.iter(|| black_box(walk_ordered_verified(corpus, keyword, LIMIT, 0))),
        );
        group.bench_with_input(
            BenchmarkId::new("segment_pruned", &label),
            &planted.keyword,
            |b, keyword| b.iter(|| black_box(segment_pruned_ordered(corpus, keyword, LIMIT))),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_short_keywords,
    bench_long_keywords,
    bench_deep_page,
    bench_windowed,
    bench_windowed_verified,
    bench_segment_pruning,
    bench_segment_pruning_verified
);
criterion_main!(benches);
