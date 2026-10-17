/*
 * Copyright 2026-present ScyllaDB
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

//! The `LIKE` patterns a pattern index serves.
//!
//! ScyllaDB sends the pattern as the query wrote it, so it is read here with the rules of
//! ScyllaDB's own `LIKE` (`utils/like_matcher.cc`): `%` matches any run of characters, `_` exactly
//! one, `\` makes the next character literal whatever it is, a `\` that ends the pattern is a
//! literal backslash, and the whole value has to match. An index that answered a pattern with
//! other rules would return other rows than a filtered `LIKE`.

use super::actor::MatchKind;
use crate::QueryError;

#[derive(Debug, PartialEq, Eq)]
enum Token {
    Literal(char),
    AnyRun,
    AnyOne,
}

fn tokens(pattern: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        tokens.push(match c {
            '\\' => Token::Literal(chars.next().unwrap_or('\\')),
            '%' => Token::AnyRun,
            '_' => Token::AnyOne,
            c => Token::Literal(c),
        });
    }
    tokens
}

/// The keyword a pattern looks for and where in the value: `'%kw%'` anywhere, `'kw%'` at the
/// start, `'%kw'` at the end. Repeated `%` are one. Any other pattern is a [`QueryError`], which
/// the endpoint answers with 400: a `_`, a `%` between literal characters, a pattern without `%`
/// (the whole value), and one without a literal character (every value).
pub(crate) fn parse(pattern: &str) -> Result<(MatchKind, String), QueryError> {
    let refuse = |why: &str| {
        QueryError(format!(
            "pattern: the index serves only '%keyword%', 'keyword%' and '%keyword', \
            but {pattern:?} {why}"
        ))
    };
    let mut tokens = tokens(pattern);
    if tokens.contains(&Token::AnyOne) {
        return Err(refuse("has a '_' wildcard"));
    }
    tokens.dedup_by(|a, b| *a == Token::AnyRun && *b == Token::AnyRun);
    let leading = tokens.first() == Some(&Token::AnyRun);
    let trailing = tokens.len() > usize::from(leading) && tokens.last() == Some(&Token::AnyRun);
    let inner = &tokens[usize::from(leading)..tokens.len() - usize::from(trailing)];
    let mut keyword = String::with_capacity(pattern.len());
    for token in inner {
        match token {
            Token::Literal(c) => keyword.push(*c),
            _ => return Err(refuse("has a '%' between literal characters")),
        }
    }
    if keyword.is_empty() {
        return Err(refuse("has no literal character"));
    }
    match (leading, trailing) {
        (true, true) => Ok((MatchKind::Contains, keyword)),
        (false, true) => Ok((MatchKind::Prefix, keyword)),
        (true, false) => Ok((MatchKind::Suffix, keyword)),
        (false, false) => Err(refuse("has no '%' and would match the whole value")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn served(pattern: &str) -> (MatchKind, String) {
        parse(pattern).unwrap_or_else(|err| panic!("{pattern:?} refused: {err}"))
    }

    fn refused(pattern: &str) -> String {
        match parse(pattern) {
            Ok(served) => panic!("{pattern:?} served as {served:?}"),
            Err(err) => err.to_string(),
        }
    }

    #[test]
    fn the_three_shapes_are_served() {
        assert_eq!(served("%将军%"), (MatchKind::Contains, "将军".into()));
        assert_eq!(served("将军%"), (MatchKind::Prefix, "将军".into()));
        assert_eq!(served("%将军"), (MatchKind::Suffix, "将军".into()));
    }

    #[test]
    fn repeated_percent_signs_are_one() {
        assert_eq!(served("%%abc%%%"), (MatchKind::Contains, "abc".into()));
        assert_eq!(served("abc%%"), (MatchKind::Prefix, "abc".into()));
    }

    #[test]
    fn a_backslash_makes_the_next_character_literal() {
        assert_eq!(served("%50\\%%"), (MatchKind::Contains, "50%".into()));
        assert_eq!(served("%a\\_b%"), (MatchKind::Contains, "a_b".into()));
        assert_eq!(served("%a\\\\b%"), (MatchKind::Contains, "a\\b".into()));
        assert_eq!(served("%\\x%"), (MatchKind::Contains, "x".into()));
        // An escaped `%` at an end is part of the keyword, not an anchor.
        assert_eq!(served("\\%abc%"), (MatchKind::Prefix, "%abc".into()));
        assert_eq!(served("%abc\\%"), (MatchKind::Suffix, "abc%".into()));
    }

    #[test]
    fn a_trailing_backslash_is_literal() {
        // like_matcher adds a second backslash to a pattern ending in one.
        assert_eq!(served("%abc\\"), (MatchKind::Suffix, "abc\\".into()));
    }

    #[test]
    fn other_patterns_are_refused_with_the_reason() {
        assert!(refused("%a_b%").contains("'_'"));
        assert!(refused("_abc%").contains("'_'"));
        assert!(refused("%ab%cd%").contains("between literal"));
        assert!(refused("abc").contains("whole value"));
        assert!(refused("%").contains("no literal"));
        assert!(refused("%%").contains("no literal"));
        assert!(refused("").contains("no literal"));
    }
}
