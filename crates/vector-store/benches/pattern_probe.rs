/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! What a keyword past `max_gram` costs per segment once the segment skip has let the segment
//! through.
//!
//! The walk asks each segment's term dictionary for every gram of the keyword before opening it
//! (`GramProbe` in `pattern_index::tantivy`), builds the postings from the entries it found,
//! intersects them and verifies each candidate against the stored text (`check`, then
//! `all grams`). Two ways to cut that cost are measured against it:
//!
//! - `rarest k`: intersect only the `k` grams with the fewest documents in the segment, and let
//!   the verification drop what the others would have. Fewer posting lists opened, more
//!   candidates read from the store.
//! - a larger `max_gram` (`PATTERN_PROBE_MAX_GRAM`): longer grams are rarer, so fewer segments
//!   hold all of a keyword's grams without holding the keyword, and fewer candidates survive the
//!   intersection. The index grows; its size is printed.
//!
//! Runs on a real corpus rather than a synthetic one, since what matters is how many segments hold
//! every gram of a keyword without holding it:
//!
//! ```sh
//! PATTERN_PROBE_NAMES=.../names_10M_long/shards/names_000.tsv,.../names_001.tsv \
//! PATTERN_PROBE_QUERIES=.../queries_char8.tsv,.../queries_char16.tsv \
//! PATTERN_PROBE_MAX_GRAM=3 PATTERN_PROBE_RAREST=2,3,4 \
//!     cargo bench -p vector-store --bench pattern_probe
//! ```
//!
//! Names files are `id<TAB>name<TAB>sort key`, query files `id<TAB>keyword`.

use std::collections::BTreeSet;
use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;
use tantivy::DocSet;
use tantivy::Index;
use tantivy::Searcher;
use tantivy::SegmentReader;
use tantivy::TERMINATED;
use tantivy::TantivyDocument;
use tantivy::Term;
use tantivy::indexer::NoMergePolicy;
use tantivy::postings::TermInfo;
use tantivy::query::ConstScorer;
use tantivy::query::Scorer;
use tantivy::query::intersect_scorers;
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

// Mirrors pattern_index::tantivy: field names, tokenizer, record option, frame marks.
const TOKENIZER_NAME: &str = "pattern_ngram";
const MIN_GRAM: usize = 1;
const STORE_CACHE_BLOCKS: usize = 1;
const WRITER_HEAP_BYTES: usize = 512 << 20;
const REPEATS: usize = 5;

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn grams_of_length(text: &str, gram_len: usize) -> BTreeSet<String> {
    let chars: Vec<char> = text.chars().collect();
    chars
        .windows(gram_len)
        .map(|window| window.iter().collect())
        .collect()
}

struct Built {
    searcher: Searcher,
    text: Field,
    max_gram: usize,
}

fn build(names_paths: &str, docs: usize, segment_docs: usize, max_gram: usize) -> Built {
    let mut schema = Schema::builder();
    let primary_id = schema.add_u64_field("primary_id", INDEXED | STORED | FAST);
    let text = schema.add_text_field(
        "text",
        TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer(TOKENIZER_NAME)
                    .set_index_option(IndexRecordOption::Basic),
            )
            .set_stored(),
    );
    let sort_key = schema.add_u64_field("sort_key", FAST);
    let index = Index::create_in_ram(schema.build());
    index.tokenizers().register(
        TOKENIZER_NAME,
        TextAnalyzer::builder(NgramTokenizer::new(MIN_GRAM, max_gram, false).unwrap()).build(),
    );
    // One thread, so that one commit is one segment of `segment_docs` names.
    let mut writer = index.writer_with_num_threads(1, WRITER_HEAP_BYTES).unwrap();
    writer.set_merge_policy(Box::new(NoMergePolicy));
    let content: String = names_paths
        .split(',')
        .map(|path| std::fs::read_to_string(path).expect("PATTERN_PROBE_NAMES is readable"))
        .collect();
    for (n, line) in content.lines().take(docs).enumerate() {
        let mut columns = line.split('\t');
        let (_, name, key) = (columns.next(), columns.next().unwrap(), columns.next());
        let mut doc = TantivyDocument::new();
        doc.add_u64(primary_id, n as u64);
        doc.add_text(text, format!("\u{2}{name}\u{3}"));
        doc.add_u64(
            sort_key,
            key.and_then(|key| key.parse().ok()).unwrap_or(n as u64),
        );
        writer.add_document(doc).unwrap();
        if (n + 1) % segment_docs == 0 {
            writer.commit().unwrap();
        }
    }
    writer.commit().unwrap();
    writer.wait_merging_threads().unwrap();
    let searcher = index.reader().unwrap().searcher();
    Built {
        searcher,
        text,
        max_gram,
    }
}

/// Moves the gram that ruled a segment out to the front, as `GramProbe` does.
fn missing_gram(terms: &mut [Term], segment: &SegmentReader, text: Field) -> bool {
    let inverted = segment.inverted_index(text).unwrap();
    for i in 0..terms.len() {
        if inverted.get_term_info(&terms[i]).unwrap().is_none() {
            terms[..=i].rotate_right(1);
            return true;
        }
    }
    false
}

/// Like `missing_gram`, but keeps what it found.
fn found_grams(
    terms: &mut [Term],
    segment: &SegmentReader,
    text: Field,
    found: &mut Vec<TermInfo>,
) -> bool {
    found.clear();
    let inverted = segment.inverted_index(text).unwrap();
    for i in 0..terms.len() {
        match inverted.get_term_info(&terms[i]).unwrap() {
            Some(info) => found.push(info),
            None => {
                terms[..=i].rotate_right(1);
                return false;
            }
        }
    }
    true
}

#[derive(Default, Clone, Copy)]
struct Walk {
    opened: u64,
    candidates: u64,
    matches: u64,
}

/// Reads every candidate's text from the store and counts those that hold the keyword.
fn verify(mut docs: impl DocSet, segment: &SegmentReader, built: &Built, keyword: &str) -> Walk {
    let store = segment.get_store_reader(STORE_CACHE_BLOCKS).unwrap();
    let mut walk = Walk::default();
    let mut doc = docs.doc();
    while doc != TERMINATED {
        walk.candidates += 1;
        let text: TantivyDocument = store.get(doc).unwrap();
        if text
            .get_first(built.text)
            .and_then(|value| value.as_str())
            .is_some_and(|text| text.contains(keyword))
        {
            walk.matches += 1;
        }
        doc = docs.advance();
    }
    walk
}

/// The check only, to split the walk's cost.
fn walk_check_only(built: &Built, segments: &[SegmentReader], keyword: &str) -> Walk {
    let mut terms: Vec<Term> = grams_of_length(keyword, built.max_gram)
        .iter()
        .map(|gram| Term::from_field_text(built.text, gram))
        .collect();
    let mut walk = Walk::default();
    for segment in segments {
        if !missing_gram(&mut terms, segment, built.text) {
            walk.opened += 1;
        }
    }
    walk
}

/// The check keeps its entries and the postings are built from them: what the node does today.
/// With `rarest`, only that many of the grams with the fewest documents are intersected.
fn walk_reuse(
    built: &Built,
    segments: &[SegmentReader],
    keyword: &str,
    rarest: Option<usize>,
) -> Walk {
    let mut terms: Vec<Term> = grams_of_length(keyword, built.max_gram)
        .iter()
        .map(|gram| Term::from_field_text(built.text, gram))
        .collect();
    let mut found = Vec::with_capacity(terms.len());
    let mut walk = Walk::default();
    for segment in segments {
        if !found_grams(&mut terms, segment, built.text, &mut found) {
            continue;
        }
        walk.opened += 1;
        if let Some(rarest) = rarest {
            found.sort_unstable_by_key(|info| info.doc_freq);
            found.truncate(rarest);
        }
        let inverted = segment.inverted_index(built.text).unwrap();
        let scorers: Vec<Box<dyn Scorer>> = found
            .iter()
            .map(|info| {
                Box::new(ConstScorer::new(
                    inverted
                        .read_postings_from_terminfo(info, IndexRecordOption::Basic)
                        .unwrap(),
                    1.0,
                )) as Box<dyn Scorer>
            })
            .collect();
        let one = verify(
            intersect_scorers(scorers, segment.max_doc()),
            segment,
            built,
            keyword,
        );
        walk.candidates += one.candidates;
        walk.matches += one.matches;
    }
    walk
}

fn time(
    built: &Built,
    segments: &[SegmentReader],
    keywords: &[String],
    walk: &dyn Fn(&Built, &[SegmentReader], &str) -> Walk,
) -> (Duration, Walk) {
    let mut best = Duration::MAX;
    let mut total = Walk::default();
    for _ in 0..REPEATS {
        total = Walk::default();
        let started = Instant::now();
        for keyword in keywords {
            let one = black_box(walk(built, segments, keyword));
            total.opened += one.opened;
            total.candidates += one.candidates;
            total.matches += one.matches;
        }
        best = best.min(started.elapsed());
    }
    (best, total)
}

fn main() {
    let names = std::env::var("PATTERN_PROBE_NAMES").expect("set PATTERN_PROBE_NAMES");
    let queries = std::env::var("PATTERN_PROBE_QUERIES").expect("set PATTERN_PROBE_QUERIES");
    let docs = env_or("PATTERN_PROBE_DOCS", 1_000_000);
    let segment_docs = env_or("PATTERN_PROBE_SEGMENT_DOCS", 66_000);
    let max_queries = env_or("PATTERN_PROBE_MAX_QUERIES", 1_000);
    let max_gram = env_or("PATTERN_PROBE_MAX_GRAM", 3);
    let rarest: Vec<usize> = std::env::var("PATTERN_PROBE_RAREST")
        .unwrap_or_default()
        .split(',')
        .filter_map(|k| k.trim().parse().ok())
        .collect();

    let started = Instant::now();
    let built = build(&names, docs, segment_docs, max_gram);
    let segments = built.searcher.segment_readers().to_vec();
    let bytes = built.searcher.space_usage().unwrap().total().get_bytes();
    eprintln!(
        "built {docs} names, max_gram {max_gram}, in {} segments in {:.1} s, {:.1} bytes a name",
        segments.len(),
        started.elapsed().as_secs_f64(),
        bytes as f64 / docs as f64,
    );

    for path in queries.split(',') {
        let keywords: Vec<String> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter_map(|line| line.split('\t').nth(1).map(str::to_string))
            .filter(|keyword| keyword.chars().count() > max_gram)
            .take(max_queries)
            .collect();
        if keywords.is_empty() {
            println!("{path}\n  no keyword longer than max_gram {max_gram}");
            continue;
        }
        let n = keywords.len() as f64;
        let per_query = |d: Duration| d.as_secs_f64() * 1e6 / n;
        let (check, checked) = time(&built, &segments, &keywords, &walk_check_only);
        let (reuse, reused) = time(&built, &segments, &keywords, &|b, s, k| {
            walk_reuse(b, s, k, None)
        });
        assert_eq!(checked.opened, reused.opened);
        println!(
            "{path}\n  {n} keywords, {:.1} of {} segments opened, {:.2} candidates, {:.2} matches a query\n  \
             check     {:7.1} us/query\n  \
             all grams {:7.1} us/query",
            reused.opened as f64 / n,
            segments.len(),
            reused.candidates as f64 / n,
            reused.matches as f64 / n,
            per_query(check),
            per_query(reuse),
        );
        for &k in &rarest {
            let (took, walk) = time(&built, &segments, &keywords, &|b, s, kw| {
                walk_reuse(b, s, kw, Some(k))
            });
            assert_eq!(walk.matches, reused.matches);
            println!(
                "  rarest {k}  {:7.1} us/query  {:.2} candidates a query  ({:.0}% of all grams)",
                per_query(took),
                walk.candidates as f64 / n,
                100.0 * took.as_secs_f64() / reuse.as_secs_f64(),
            );
        }
    }
}
