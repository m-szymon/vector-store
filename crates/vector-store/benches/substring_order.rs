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
//! constant is, because that decides whether plain ordering survives for realistic search-box
//! keywords or whether stage 2 needs to partition the index by the sort column.
//!
//! Four things are measured, at each keyword frequency:
//!
//! * `walk_unordered` -- stage 1 as it ships: walk, verify, stop at the limit.
//! * `topdocs_ordered` -- stage 2 for keywords within `max_gram`: one term, ordered top-k.
//! * `walk_ordered_verify` -- stage 2 for keywords past `max_gram`: every candidate must be read
//!   from the store and verified before the top-k is known, so this is the case that turns bounded
//!   store reads into unbounded ones. Compared against `walk_unordered_verify`, the same query
//!   shape without ordering.
//! * `deep_page` -- page 50 rather than page 1, unordered (offset) and ordered, to show that a
//!   cursor does not rescue either: for `DESC` by time the cursor excludes only the rows already
//!   returned.
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
use std::collections::BTreeSet;
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
    let text_options = TextOptions::default()
        .set_indexing_options(indexing)
        .set_stored();
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

    for doc_id in 0..names {
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
        doc.add_u64(sort_field, doc_id as u64);
        writer
            .add_document(doc)
            .expect("bench: failed to add a doc");
    }
    writer.commit().expect("bench: failed to commit");

    let reader = index.reader().expect("bench: failed to open a reader");
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

    for planted in &corpus.short {
        let expected = planted.matches.min(LIMIT);
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

criterion_group!(
    benches,
    bench_short_keywords,
    bench_long_keywords,
    bench_deep_page
);
criterion_main!(benches);
