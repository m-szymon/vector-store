/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! A LIKE pattern with the rules of ScyllaDB's LIKE operator: the pattern matches the whole
//! value, `%` matches any sequence of characters, `_` matches one character, `\` makes the
//! character after it literal, and a `\` ending the pattern matches `\`.

use super::gram;
use super::gram::Token;
use crate::QueryError;
use anyhow::anyhow;
use regex::Regex;
use tantivy::DocId;
use tantivy::SegmentReader;
use tantivy::TantivyDocument;
use tantivy::Term;
use tantivy::index::SegmentId;
use tantivy::postings::TermInfo;
use tantivy::query::AllQuery;
use tantivy::query::ConstScorer;
use tantivy::query::EnableScoring;
use tantivy::query::Query;
use tantivy::query::Scorer;
use tantivy::query::Weight;
use tantivy::query::intersect_scorers;
use tantivy::schema::Field;
use tantivy::schema::IndexRecordOption;
use tantivy::schema::Schema;
use tantivy::schema::Value;
use tantivy::store::StoreReader;

/// The candidates of a pattern: the documents that hold every gram a matching value must hold.
enum Candidates {
    /// The pattern has no gram, so every document is a candidate.
    All(Box<dyn Weight>),
    Grams(GramProbe),
}

/// Opens the documents that hold every one of several grams, one segment at a time.
///
/// A segment whose term dictionary lacks any one of the grams holds no candidate, so it is
/// skipped before a single posting list is opened -- which, for a pattern that occurs in a
/// handful of values, is nearly every segment, and what made a rare long pattern cost a full walk
/// of every segment. The gram last found missing is asked first (move to front): a rare pattern
/// usually has one rare gram, which is absent from nearly every segment, so after the first miss
/// a segment costs one dictionary lookup.
///
/// A segment that holds every gram has its postings built from the entries the check just
/// found, rather than through a `TermQuery` weight per gram, which would look each gram up again
/// and open a field-norm reader an unscored walk never reads. Measured on the 10M-name corpus
/// (`benches/pattern_probe.rs`), that second pass was a third of the walk for 8 to 32
/// characters.
///
/// With `rarest`, only that many of the grams found, those with the fewest documents in the
/// segment, are intersected: the verification drops what the others would have. A long pattern's
/// grams are mostly common ones, and each costs a posting list opened and walked; a few rare ones
/// narrow the candidates nearly as far (`benches/pattern_probe.rs`: at 16 characters, 4 of the
/// pattern's 14 grams took the walk to a quarter, with the same matches).
struct GramProbe {
    terms: Vec<Term>,
    field: Field,
    found: Vec<TermInfo>,
    /// The segment `found` is of.
    opened: Option<SegmentId>,
    rarest: Option<usize>,
}

impl GramProbe {
    fn new(terms: Vec<Term>, field: Field, rarest: Option<usize>) -> Self {
        let found = Vec::with_capacity(terms.len());
        Self {
            terms,
            field,
            found,
            opened: None,
            rarest,
        }
    }

    /// The segment's candidates, or `None` when it certainly holds none.
    fn open(&mut self, segment: &SegmentReader) -> anyhow::Result<Option<Box<dyn Scorer>>> {
        let inverted = segment
            .inverted_index(self.field)
            .map_err(|e| anyhow!("pattern: failed to open the term dictionary: {e}"))?;
        if self.opened != Some(segment.segment_id()) {
            self.opened = None;
            self.found.clear();
            for i in 0..self.terms.len() {
                let info = inverted
                    .get_term_info(&self.terms[i])
                    .map_err(|e| anyhow!("pattern: failed to look a gram up: {e}"))?;
                let Some(info) = info else {
                    self.terms[..=i].rotate_right(1);
                    return Ok(None);
                };
                self.found.push(info);
            }
            // Fewer grams than the pattern has cannot prove a match, and need not: a pattern
            // with more than one gram is always verified.
            if let Some(rarest) = self.rarest
                && rarest < self.found.len()
            {
                self.found.sort_unstable_by_key(|info| info.doc_freq);
                self.found.truncate(rarest);
            }
            self.opened = Some(segment.segment_id());
        }
        let postings = self
            .found
            .iter()
            .map(|info| {
                let postings = inverted
                    .read_postings_from_terminfo(info, IndexRecordOption::Basic)
                    .map_err(|e| anyhow!("pattern: failed to open a gram's postings: {e}"))?;
                Ok(Box::new(ConstScorer::new(postings, 1.0)) as Box<dyn Scorer>)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Some(intersect_scorers(postings, segment.max_doc())))
    }
}

/// The grams every stored value matching a LIKE pattern holds, and the matcher a stored value
/// must still pass when the grams do not prove the match.
pub(super) struct LikeQuery {
    field: Field,
    candidates: Candidates,
    regex_matcher: Option<Regex>,
}

impl LikeQuery {
    /// `min_gram` and `max_gram` are the lengths of the grams the index keeps. A pattern whose
    /// literal characters all stand in runs shorter than `min_gram` has no gram to look up, and
    /// is a [`QueryError`]. With `rarest_grams`, only that many of the pattern's grams are
    /// intersected in a segment, the rarest there.
    pub(super) fn new(
        schema: &Schema,
        field: Field,
        pattern: &str,
        min_gram: usize,
        max_gram: usize,
        rarest_grams: Option<usize>,
    ) -> anyhow::Result<Self> {
        let framed = gram::framed_pattern(&parse(pattern));
        let grams = gram::grams(&framed, min_gram, max_gram);
        let regex_matcher = matcher_unless_grams_prove_match(&framed, min_gram, max_gram)?;
        let candidates = if !grams.is_empty() {
            let terms = grams
                .iter()
                .map(|gram| Term::from_field_text(field, gram))
                .collect();
            Candidates::Grams(GramProbe::new(terms, field, rarest_grams))
        } else if framed
            .iter()
            .any(|token| matches!(token, Token::Literal(_)))
        {
            return Err(QueryError(format!(
                "pattern: {pattern:?} has no run of at least min_gram={min_gram} literal \
                characters, which is the shortest the index can look up"
            ))
            .into());
        } else {
            // Scoring is off, as LIKE results are not ordered by a score, so the weight needs
            // only the schema.
            let weight = AllQuery
                .weight(EnableScoring::disabled_from_schema(schema))
                .map_err(|e| anyhow!("pattern: failed to build the query: {e}"))?;
            Candidates::All(weight)
        };
        Ok(Self {
            field,
            candidates,
            regex_matcher,
        })
    }

    /// Whether a document that holds every gram may still not match, so that its stored value
    /// has to be checked (`value_matches`).
    pub(super) fn needs_verification(&self) -> bool {
        self.regex_matcher.is_some()
    }

    /// The documents of `segment` that hold every gram of the pattern, in doc id order, the
    /// deleted ones among them, or `None` when the segment certainly holds none.
    pub(super) fn candidates(
        &mut self,
        segment: &SegmentReader,
    ) -> anyhow::Result<Option<Box<dyn Scorer>>> {
        match &mut self.candidates {
            Candidates::All(weight) => weight
                .scorer(segment, 1.0)
                .map(Some)
                .map_err(|e| anyhow!("pattern: failed to run the query: {e}")),
            Candidates::Grams(probe) => probe.open(segment),
        }
    }

    /// Whether the stored value of a candidate matches the pattern. No value is read when the
    /// grams already prove the match.
    pub(super) fn value_matches(&self, store: &StoreReader, doc_id: DocId) -> anyhow::Result<bool> {
        let Some(regex) = &self.regex_matcher else {
            return Ok(true);
        };
        let doc: TantivyDocument = store
            .get(doc_id)
            .map_err(|e| anyhow!("pattern: failed to read the document: {e}"))?;
        Ok(doc
            .get_first(self.field)
            .and_then(|value| value.as_str())
            .and_then(gram::unframed)
            .is_some_and(|value| regex.is_match(value)))
    }
}

/// Consecutive `%` are collapsed into one [`Token::AnySeq`], as they match the same values.
fn parse(pattern: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = pattern.chars();
    while let Some(pattern_char) = chars.next() {
        tokens.push(match pattern_char {
            '\\' => Token::Literal(chars.next().unwrap_or('\\')),
            '%' => Token::AnySeq,
            '_' => Token::AnyChar,
            literal => Token::Literal(literal),
        });
    }
    tokens.dedup_by(|token, previous| *token == Token::AnySeq && *previous == Token::AnySeq);
    tokens
}

/// The regex that checks the stored value, or `None` when holding every gram already proves a
/// match, so no value has to be read.
fn matcher_unless_grams_prove_match(
    framed: &[Token],
    min_gram: usize,
    max_gram: usize,
) -> anyhow::Result<Option<Regex>> {
    if gram::grams_prove_match(framed, min_gram, max_gram) {
        return Ok(None);
    }
    matcher(framed).map(Some)
}

/// A regex matching a value that matches the pattern.
fn matcher(framed: &[Token]) -> anyhow::Result<Regex> {
    let body: String = framed
        .iter()
        .map(|token| match token {
            Token::AnyChar => ".".to_string(),
            Token::AnySeq => ".*".to_string(),
            Token::Literal(literal) => regex::escape(&literal.to_string()),
            Token::ValueStart => "^".to_string(),
            Token::ValueEnd => "$".to_string(),
        })
        .collect();
    Regex::new(&format!("(?s){body}"))
        .map_err(|e| anyhow!("pattern: failed to build the matcher: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn like(pattern: &str, value: &str) -> bool {
        matcher(&gram::framed_pattern(&parse(pattern)))
            .unwrap()
            .is_match(value)
    }

    /// The rules of ScyllaDB's LIKE.
    #[rstest]
    #[case::literal("abc", "abc", true)]
    #[case::literal("abc", "abcd", false)]
    #[case::case_sensitive("abc", "ABC", false)]
    #[case::empty("", "", true)]
    #[case::empty("", "a", false)]
    #[case::underscore_start("_a", "Шa", true)]
    #[case::underscore_start("_a", "a", false)]
    #[case::underscore_middle("a_c", "abc", true)]
    #[case::underscore_middle("a_c", "ac", false)]
    #[case::underscore_end("a_", "ab", true)]
    #[case::underscore_end("a_", "abc", false)]
    #[case::underscore_consecutive("a__", "abc", true)]
    #[case::underscore_consecutive("a__", "ab", false)]
    #[case::percent_start("%a", "ba", true)]
    #[case::percent_start("%a", "ab", false)]
    #[case::percent_middle("a%z", "az", true)]
    #[case::percent_middle("a%z", "abz", true)]
    #[case::percent_middle("a%z", "za", false)]
    #[case::percent_end("a%", "a", true)]
    #[case::percent_end("a%", "ba", false)]
    #[case::percent_multiple("%a%z%", "xaxzx", true)]
    #[case::percent_multiple("%a%z%", "xzxax", false)]
    #[case::escape_percent(r"a\%", "a%", true)]
    #[case::escape_percent(r"a\%", "ab", false)]
    #[case::escape_underscore(r"a\_", "a_", true)]
    #[case::escape_underscore(r"a\_", "ab", false)]
    #[case::escape_any_char(r"a\b", "ab", true)]
    #[case::single_backslash_at_end(r"a\", r"a\", true)]
    #[case::single_backslash_at_end(r"a\", "a", false)]
    #[case::double_backslash_at_end(r"a\\", r"a\", true)]
    #[case::regex_special_characters(".*+?^$()[]{}|", ".*+?^$()[]{}|", true)]
    #[case::regex_special_characters("a.c", "abc", false)]
    #[case::any_seq_spans_newlines("a%z", "a\nb\nz", true)]
    #[case::any_char_matches_a_newline("a_z", "a\nz", true)]
    #[case::any_char_is_a_code_point_not_a_byte("a_c", "a€c", true)]
    #[case::cjk_any_char_is_one_code_point("__将军", "宇将军", false)]
    fn regex_matcher_matches_like_scylladb(
        #[case] pattern: &str,
        #[case] value: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(like(pattern, value), expected);
    }
}
