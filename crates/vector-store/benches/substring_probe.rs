/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! What a keyword past `max_gram` costs per segment once the segment skip has let the segment
//! through, and how much of that is looking its grams up a second time.
//!
//! The walk asks each segment's term dictionary for every gram of the keyword before opening it
//! (`GramProbe` in `substring_index::tantivy`), and then builds the intersection through a
//! `BooleanQuery` of `TermQuery`s, whose weight looks every gram up again and also opens a
//! field-norm reader per gram that an unscored walk never reads. The `reuse` variant builds the
//! postings from the entries the check already found and intersects them directly.
//!
//! Runs on a real corpus rather than a synthetic one, since what matters is how many segments hold
//! every gram of a keyword without holding it:
//!
//! ```sh
//! SUBSTRING_PROBE_NAMES=.../names_10M_long/shards/names_000.tsv \
//! SUBSTRING_PROBE_QUERIES=.../queries_char8.tsv,.../queries_char16.tsv \
//!     cargo bench -p vector-store --bench substring_probe
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
use tantivy::query::BooleanQuery;
use tantivy::query::ConstScorer;
use tantivy::query::EnableScoring;
use tantivy::query::Query;
use tantivy::query::Scorer;
use tantivy::query::TermQuery;
use tantivy::query::intersect_scorers;
use tantivy::schema::FAST;
use tantivy::schema::Field;
use tantivy::schema::INDEXED;
use tantivy::schema::IndexRecordOption;
use tantivy::schema::STORED;
use tantivy::schema::Schema;
use tantivy::schema::TextFieldIndexing;
use tantivy::schema::TextOptions;
use tantivy::tokenizer::NgramTokenizer;
use tantivy::tokenizer::TextAnalyzer;

// Mirrors substring_index::tantivy: field names, tokenizer, record option, frame marks.
const TOKENIZER_NAME: &str = "substring_ngram";
const MIN_GRAM: usize = 1;
const MAX_GRAM: usize = 3;
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
}

fn build(names_path: &str, docs: usize, segment_docs: usize) -> Built {
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
        TextAnalyzer::builder(NgramTokenizer::new(MIN_GRAM, MAX_GRAM, false).unwrap()).build(),
    );
    // One thread, so that one commit is one segment of `segment_docs` names.
    let mut writer = index.writer_with_num_threads(1, WRITER_HEAP_BYTES).unwrap();
    writer.set_merge_policy(Box::new(NoMergePolicy));
    let content = std::fs::read_to_string(names_path).expect("SUBSTRING_PROBE_NAMES is readable");
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
    Built { searcher, text }
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

fn drain(mut docs: impl DocSet) -> u64 {
    let mut count = 0;
    let mut doc = docs.doc();
    while doc != TERMINATED {
        count += 1;
        doc = docs.advance();
    }
    count
}

#[derive(Default, Clone, Copy)]
struct Walk {
    opened: u64,
    candidates: u64,
}

/// Today's walk: the check, then the query's weight per segment.
fn walk_current(built: &Built, segments: &[SegmentReader], keyword: &str) -> Walk {
    let grams = grams_of_length(keyword, MAX_GRAM);
    let mut terms: Vec<Term> = grams
        .iter()
        .map(|gram| Term::from_field_text(built.text, gram))
        .collect();
    let clauses: Vec<Box<dyn Query>> = terms
        .iter()
        .map(|term| {
            Box::new(TermQuery::new(term.clone(), IndexRecordOption::Basic)) as Box<dyn Query>
        })
        .collect();
    let weight = BooleanQuery::intersection(clauses)
        .weight(EnableScoring::disabled_from_searcher(&built.searcher))
        .unwrap();
    let mut walk = Walk::default();
    for segment in segments {
        if missing_gram(&mut terms, segment, built.text) {
            continue;
        }
        walk.opened += 1;
        walk.candidates += drain(weight.scorer(segment, 1.0).unwrap());
    }
    walk
}

/// The check only, to split the walk's cost.
fn walk_check_only(built: &Built, segments: &[SegmentReader], keyword: &str) -> Walk {
    let mut terms: Vec<Term> = grams_of_length(keyword, MAX_GRAM)
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

/// The check keeps its entries and the postings are built from them.
fn walk_reuse(built: &Built, segments: &[SegmentReader], keyword: &str) -> Walk {
    let mut terms: Vec<Term> = grams_of_length(keyword, MAX_GRAM)
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
        walk.candidates += drain(intersect_scorers(scorers, segment.max_doc()));
    }
    walk
}

type WalkFn = fn(&Built, &[SegmentReader], &str) -> Walk;

fn time(
    built: &Built,
    segments: &[SegmentReader],
    keywords: &[String],
    walk: WalkFn,
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
        }
        best = best.min(started.elapsed());
    }
    (best, total)
}

fn main() {
    let names = std::env::var("SUBSTRING_PROBE_NAMES").expect("set SUBSTRING_PROBE_NAMES");
    let queries = std::env::var("SUBSTRING_PROBE_QUERIES").expect("set SUBSTRING_PROBE_QUERIES");
    let docs = env_or("SUBSTRING_PROBE_DOCS", 1_000_000);
    let segment_docs = env_or("SUBSTRING_PROBE_SEGMENT_DOCS", 66_000);
    let max_queries = env_or("SUBSTRING_PROBE_MAX_QUERIES", 1_000);

    let started = Instant::now();
    let built = build(&names, docs, segment_docs);
    let segments = built.searcher.segment_readers().to_vec();
    eprintln!(
        "built {docs} names in {} segments in {:.1} s",
        segments.len(),
        started.elapsed().as_secs_f64()
    );

    for path in queries.split(',') {
        let keywords: Vec<String> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .filter_map(|line| line.split('\t').nth(1).map(str::to_string))
            .filter(|keyword| keyword.chars().count() > MAX_GRAM)
            .take(max_queries)
            .collect();
        let n = keywords.len() as f64;
        let (current, walk) = time(&built, &segments, &keywords, walk_current);
        let (check, checked) = time(&built, &segments, &keywords, walk_check_only);
        let (reuse, reused) = time(&built, &segments, &keywords, walk_reuse);
        assert_eq!(walk.opened, checked.opened);
        assert_eq!(walk.opened, reused.opened);
        assert_eq!(walk.candidates, reused.candidates);
        let per_query = |d: Duration| d.as_secs_f64() * 1e6 / n;
        let per_open = |d: Duration| d.as_secs_f64() * 1e9 / walk.opened.max(1) as f64;
        println!(
            "{path}\n  {n} keywords, {:.1} of {} segments opened, {:.2} candidates a query\n  \
             current {:7.1} us/query  {:6.0} ns per opened segment\n  \
             check   {:7.1} us/query\n  \
             reuse   {:7.1} us/query  {:6.0} ns per opened segment  ({:.0}% of current)",
            walk.opened as f64 / n,
            segments.len(),
            walk.candidates as f64 / n,
            per_query(current),
            per_open(current),
            per_query(check),
            per_query(reuse),
            per_open(reuse),
            100.0 * reuse.as_secs_f64() / current.as_secs_f64(),
        );
    }
}
