---
marp: true
theme: default
paginate: true
title: FTS Query Syntax for ScyllaDB CQL
---

# FTS Query Syntax for ScyllaDB CQL

How should CQL express full-text search conditions?

*Discussion starter*

---

# Current: Vector Search Query Syntax

```sql
SELECT id, title
FROM articles
ORDER BY embedding ANN OF [0.12, 0.34, ..., 0.05]
LIMIT 10;
```

Meaning: find, using ANN\* algorithm, `10` rows whose `embedding` column is most similar\*\* to the vector `[0.12, 0.34, ..., 0.05]`.

  \* ANN = Approximate Nearest Neighbor — not exact top-N, trades precision for speed

  \*\* "Similar" is defined by index options: similarity function, quantization settings, ...

Cassandra-compatible syntax choice.


<!-- presenter notes:
This syntax comes from Apache Cassandra. We adopted it for compatibility.

Key points to emphasize:
- "Approximate" — ANN doesn't guarantee finding the true nearest neighbors,
  it's a speed/accuracy tradeoff. This is inherent to the algorithm.
- "Nearest neighbor" definition is not in the query — it's in the index.
  The similarity function (cosine, dot product, euclidean) is set at
  CREATE INDEX time. Quantization further changes what "nearest" means:
  a quantized index may use a different internal metric (e.g., Hamming
  distance on binary codes) than the original similarity function.
  Rescoring can then recompute exact similarity on the top candidates.
- ORDER BY here is not classical SQL sorting. It doesn't sort an existing
  result set — it defines which rows are returned. The query is routed to
  the vector index node which performs the ANN search.
- The query vector in ANN OF is the only search parameter in the query
  itself. Everything else (how to compare, how approximate) comes from
  the index definition.
-->

---

# Incoming: Full Text Search

Ref: [FTS Requirement Document](https://scylladb.atlassian.net/wiki/spaces/RND/pages/263585939),
[M1 Plan](https://scylladb.atlassian.net/wiki/spaces/RND/pages/304415149)

Find, `N` rows best matching\* given text contrains.  
  \* Match is ranked by **BM25** score - based on term frequency, document length and corpus statistics.

```sql
... BM25(body, 'database') ...                         -- single term
... BM25(body, 'error AND timeout') ...                -- boolean AND
... BM25(body, 'scylla OR cassandra') ...              -- boolean OR
... BM25(body, 'error NOT warning') ...                -- boolean NOT
... BM25(body, '"out of memory"') ...                  -- phrase
... BM25(body, '(error OR fault) AND critical') ...    -- grouping
... BM25(body, 'databse~1') ...                        -- fuzzy (M3)
... BM25(body, 'scyll*') ...                           -- wildcard (M3)
... BM25((title, body), 'vector database') ...         -- multi-column (M3)
```

This is **not** boolean matching — every result has a score.
The query defines both *what* matches and *how well* it matches.

Unlike ANN, BM25 is **exact** and **deterministic** — true top-N is always
returned; increasing N only appends rows without changing existing order.

<!-- presenter notes:
BM25 always produces a ranking score. Even though the query language has
boolean operators (AND, OR, NOT), the result is not just "matches or
doesn't match." Every matching row gets a BM25 score that depends on:
- Term frequency in that specific row
- How rare the term is across the entire index (IDF)
- Document length normalization

So 'error AND timeout' doesn't just find rows containing both words —
it ranks them by how relevant they are. A short row with both terms
prominent will score higher than a long row where they appear once each.

Key difference from ANN: BM25 over an inverted index is exact — every
matching document is found and scored deterministically. The true top-N
by BM25 score is always returned. Increasing LIMIT from 10 to 20 adds
rows 11-20 without changing the first 10. This IS classical ORDER BY
behavior — a total ordering over a well-defined set.

ANN, by contrast, is approximate and non-deterministic — it may miss
true nearest neighbors, results can vary between runs, and changing
LIMIT can reshuffle earlier results. ANN lives in ORDER BY but doesn't
truly order in the classical sense. BM25 actually fits ORDER BY
semantics more naturally than ANN does.

We use BM25() here as illustrative syntax to make the scoring nature
explicit. The actual function name is part of what we're discussing.

M1 scope: single column, standard analyzer (Unicode tokenizer, lowercase,
English stop words), boolean operators, phrase queries.

M3 extends the query language with fuzzy matching (edit distance tolerance),
wildcard prefix search, and multi-column indexes with per-field boosting.
These are extensions of the search expression — they don't change where
in CQL the expression appears, which is the main topic of this presentation.
-->

---

# Combining Multiple Search Indexes (M2, M4)

A single query may involve **two (multiple?) indexes** — each a different search:

| Operation | Indexes involved | Result |
|---|---|---|
| FTS only | 1 FTS index | Rows ranked by BM25 |
| ANN only | 1 vector index | Rows ranked by similarity |
| FTS + ANN fusion | FTS + vector | Fused ranking (RRF, weighted) |
| FTS pre-filter + ANN | FTS + vector | FTS narrows candidates, ANN ranks |
| FTS post-filter + ANN | FTS + vector | ANN ranks, FTS filters results |
| Multi-signal FTS ? | 2 FTS indexes | e.g., search title AND body separately |
| Multi-column VS ? | 2 VS indexes |  |

The syntax must express:
- **Which** search operations to perform (and on which columns/indexes)
- **How** they combine — fusion? filtering? which one ranks?

<!-- presenter notes:
This is the key syntax challenge beyond basic FTS. The query needs to
coordinate multiple index-backed search operations in one statement.

In SQL systems, this is implicit — the query planner picks indexes from
predicates. In Elasticsearch/OpenSearch, this is explicit — the user
declares separate retrieval operations (knn + query) and a fusion strategy
(RRF). DSE allows solr_query + ANN OF in the same WHERE clause, with one
typically pre-filtering for the other.

Pre-filtering (M4): FTS narrows the candidate set before ANN search —
the BM25 score is discarded, only the match/no-match matters. This is
semantically different from fusion where both scores contribute to ranking.

Post-filtering (M2): ANN returns candidates, FTS filters out non-matching
rows after the fact. Simpler to implement but may discard good ANN results
that don't match the text query.

Fusion (M2): Both FTS and ANN produce ranked lists, which are merged via
Reciprocal Rank Fusion (RRF) or weighted linear combination. Both scores
contribute to the final ranking.

Multi-signal FTS: searching title and body with different queries, each
hitting a separate FTS index, with results fused. Similar to multi-column
search but with independent queries per column.

The syntax choice (WHERE vs ORDER BY) directly determines how naturally
each of these operations can be expressed.

Existing CQL precedent for multi-index queries: SAI in Cassandra 5.0+,
Astra DB, and DSE supports querying multiple column indexes in one query
with AND/OR (via Token Flow intersection/union). However, this is all
boolean filtering — no scored search, no ranking fusion. Astra DB can
combine SAI text filter (':') + vector (ORDER BY ANN OF) in one query,
but text is a boolean pre-filter only.

You cannot mix index types in one query (no SAI + DSE Search, no SAI +
SASI) — the query planner dispatches to one index implementation.

No CQL implementation has ever combined scored search results from
multiple indexes. Hybrid FTS + ANN fusion — where both produce ranked
results that are merged — would be genuinely new territory for CQL.
-->

---

# JSON Systems: Elasticsearch, OpenSearch, MongoDB

**Elasticsearch / OpenSearch**:

```json
{ "query": { "bool": {
    "must":     [{ "match": { "body": "vector database" } }],
    "should":   [{ "match": { "title": "search" } }],
    "must_not": [{ "match": { "status": "draft" } }]
} } }
```

- **OpenSearch**: essentially the same DSL (Elasticsearch fork)
- **MongoDB Atlas Search**: very similar (`compound: must/should/mustNot`)
- **DynamoDB**: no FTS — delegates entirely to OpenSearch

**Key takeaway:** JSON makes it easy to extend — add fields, nest objects.
Match conditions and ranking are unified in one query structure.
Not directly applicable to CQL grammar, but shows the "ideal" design freedom.

<!-- presenter notes:
All these systems are heavily Lucene-influenced and converged on nearly
identical query structures: boolean compound queries with must (AND),
should (OR), must_not (NOT), and filter (non-scoring AND).

The JSON format gives them freedom we don't have in CQL. They can add new
query types, nesting, and parameters without changing grammar. In CQL we
need to fit into a more rigid SQL-like syntax.

DynamoDB is interesting as a precedent: it's a key-value/document store
(like Scylla) that chose not to build FTS natively but instead integrates
with OpenSearch. The pattern is: write to DynamoDB, stream changes to
OpenSearch via CDC, query OpenSearch for search. This is architecturally
similar to our vector-store approach, but they expose it as a separate
query endpoint, not through the primary query language.
-->

---

# SQL Systems

| Database | Filter (WHERE) | Ranking (ORDER BY) |
|---|---|---|
| PostgreSQL | `WHERE tsvec @@ tsquery(...)` | `ORDER BY ts_rank(...)` |
| MySQL | `WHERE MATCH(col) AGAINST('text')` | `ORDER BY MATCH(col) AGAINST('text')` |
| SQL Server | `WHERE CONTAINS(col, 'text')` | `CONTAINSTABLE(...)` as join |
| Oracle | `WHERE CONTAINS(col, 'text', 1) > 0` | `SCORE(1)` companion function |

**Common pattern:** match predicate and relevance ranking are always separated.
WHERE = "does it match?", ORDER BY / SELECT = "how well?"

**MySQL** is the closest analogy to our design space:
```sql
SELECT *, MATCH(title, body) AGAINST('vector database') AS score
FROM articles
WHERE MATCH(title, body) AGAINST('vector database')
ORDER BY score DESC;
```

Same expression serves as both filter and score source.

<!-- presenter notes:
Every SQL system separates the boolean match (WHERE) from the ranking
(ORDER BY/SELECT), but the mechanisms vary:

PostgreSQL is the most explicit: @@ is purely boolean, ts_rank() is a
separate function. You must use both if you want ranked results.

  SELECT id, title, ts_rank(to_tsvector('english', body),
                            to_tsquery('vector & database')) AS score
  FROM articles
  WHERE to_tsvector('english', body) @@ to_tsquery('vector & database')
  ORDER BY ts_rank(to_tsvector('english', body),
                   to_tsquery('vector & database')) DESC;

MySQL's MATCH...AGAINST is dual-purpose: in WHERE it's a boolean predicate,
in ORDER BY / SELECT it returns a float relevance score. Same expression,
different meaning depending on context.

  SELECT id, title, MATCH(title, body) AGAINST('vector database') AS score
  FROM articles
  WHERE MATCH(title, body) AGAINST('vector database')
  ORDER BY MATCH(title, body) AGAINST('vector database') DESC;

SQL Server takes a different approach: CONTAINS is filter-only, and if you
want ranking you must use CONTAINSTABLE which returns a virtual table with
KEY and RANK columns that you JOIN against.

  SELECT a.id, a.title, ft.RANK
  FROM articles a
  INNER JOIN CONTAINSTABLE(articles, body, 'vector AND database') AS ft
    ON a.id = ft.[KEY]
  ORDER BY ft.RANK DESC;

Oracle uses a label-based system: CONTAINS(col, 'query', 1) tags the search
with label 1, and SCORE(1) retrieves the score for that label.

  SELECT id, title, SCORE(1) AS score
  FROM articles
  WHERE CONTAINS(body, 'vector AND database', 1) > 0
  ORDER BY SCORE(1) DESC;

None of these had to deal with a pre-existing non-classical ORDER BY
like our ANN OF. That's what makes our situation unique.
-->

---

# CQL World

**Apache Cassandra:** No FTS.
- SASI indexes: `WHERE col LIKE '%term%'` — pattern matching, no ranking
- SAI indexes (5.0): improved secondary indexes, still no FTS

**Astra DB** (DataStax) — SAI with analyzers:
```sql
-- Boolean filter only — no ranking, no scoring
SELECT * FROM products WHERE description : 'hiking';

-- Text filter + vector search (text is pre-filter for ANN)
SELECT * FROM products
WHERE description : 'waterproof'
ORDER BY embedding ANN OF [0.1, ...] LIMIT 10;
```

**DSE Search** (DataStax Enterprise) — Solr integration:
```sql
-- Scored search (results ordered by relevance — Solr default)
SELECT * FROM articles
WHERE solr_query = 'title:vector AND body:database';
```

<!-- presenter notes:
Two CQL-based systems have text search, both from DataStax:

ASTRA DB uses SAI (Storage Attached Indexes) with Lucene analyzers.
The ':' operator does tokenized matching — it's boolean only, no
relevance scoring. Text search in CQL is purely a filter. For hybrid
search, text acts as a pre-filter for vector ANN — same syntax as
our current proposal (WHERE filter + ORDER BY ANN OF).

Astra also has a separate Data API (JSON, not CQL) with richer
capabilities: BM25-like lexical scoring, hybrid search via
findAndRerank with neural reranking, and projectable scores. But
this is a different interface, not CQL.

DSE SEARCH embeds Solr's query language as a string literal in WHERE.
Everything lives in WHERE — but WHERE also determines ordering.
Scored by default (relevance descending); sort controlled inside
the string. Score is used for ordering but NOT accessible in CQL
SELECT. CQL grammar doesn't know if query is scored or boolean —
Solr decides (q = scored, fq = boolean filter).

DSE Search is NOT available in Astra DB Serverless — they are
completely different mechanisms despite both being DataStax products.

No other CQL system (YugabyteDB YCQL, Cosmos DB Cassandra API) has
FTS of any kind.

Key observations for our design:
- Astra DB's CQL approach (SAI ':' operator) is boolean-only —
  insufficient for ranked FTS. They needed a separate JSON API for
  scored search.
- DSE's approach (solr_query string) is powerful but opaque — CQL
  doesn't understand the search semantics.
- Neither system exposes relevance score as a CQL-projectable value.
- We want: scored FTS natively in CQL, with projectable scores,
  and grammar-level distinction between filter and ranking modes.

Regarding multi-index queries in CQL: SAI (Cassandra 5.0+) supports
querying multiple column indexes in one query with AND/OR via Token
Flow intersection/union — this is the closest CQL precedent. But it's
all boolean filtering, no scoring. No CQL system has combined scored
results from different indexes in a single query.
-->

---

# Decision for M1 - FTS in ORDER BY — Name and Form

**To confirm:** BM25 is a ranking function. It fits ORDER BY naturally — more so than ANN.
```sql
SELECT id, title FROM articles
ORDER BY <??? body, '(error OR fault) AND critical' ???>
LIMIT 10;
```

**To decide:** Form (function vs Cassandra-style vs something else) and name.
| Name | Function style | Cassandra style |
|---|---|---|
| BM25 | `ORDER BY BM25(body, 'query')` | `ORDER BY body BM25 OF 'query'` |
| MATCH | `ORDER BY MATCH(body, 'query')` | `ORDER BY body MATCH OF 'query'` |
| SEARCH | `ORDER BY SEARCH(body, 'query')` | `ORDER BY body SEARCH OF 'query'` |
| FTS | `ORDER BY FTS(body, 'query')` | `ORDER BY body FTS OF 'query'` |

<!-- presenter notes:
Two independent decisions here:

FORM: Function style vs Cassandra style.
- Function style: F(column, query) — standard SQL-like function call.
  Can naturally appear in other clauses (WHERE, SELECT) in the future.
- Cassandra style: column KEYWORD OF query — matches existing ANN syntax.
  More consistent with current VS, but harder to reuse in WHERE for
  boolean filtering. Also more verbose with multiple search expressions.

NAME: What to call it.
- BM25: explicit about the algorithm. But the algorithm is really an index
  concern — the query shouldn't need to know. What if we add TF-IDF later?
  Also: ANN OF doesn't name the similarity function (cosine, dot product),
  it names the search strategy. BM25 names the scoring formula — different
  level of abstraction.
- MATCH: generic, widely understood. But in SQL (MySQL), MATCH traditionally
  implies boolean matching, not ranking. Could be confusing.
- SEARCH: neutral, clearly about search. No strong SQL baggage. But also
  quite generic — search for what?
- FTS: explicit about the feature. But abbreviations can be obscure.

Other candidates worth considering: QUERY, RANK, TEXT, RELEVANT...

The name choice matters less for M1 alone, but has long-term implications
for how the syntax reads when extended to hybrid search and potentially
unified with vector search.
-->

---

# Next Features: Unified ANN Syntax

If M1 chooses a generic name (MATCH/SEARCH), it extends to ANN:
```sql
ORDER BY MATCH(embedding, [0.1, ...])  -- alternative to ANN OF
```
Existing `ANN OF` remains for Cassandra compatibility.

If M1 chooses FTS-specific name (BM25/FTS), hybrid queries
mix two patterns — less uniform:
```sql
ORDER BY BM25(body, 'query'), embedding ANN OF [0.1, ...]
```

<!-- presenter notes:
A generic name like MATCH or SEARCH naturally extends to vector search,
because the index type determines the algorithm, not the function name.
MATCH(vector_col, query_vector) = ANN search, MATCH(text_col, query) = FTS.

An FTS-specific name forces two different syntactic patterns in the same
query for hybrid search. Not a blocker, but less elegant.
-->

---

# Next Features: Hybrid Search Syntax

How to express fusion of multiple search signals?

**A) Comma-separated** — but breaks ORDER BY semantics:
```sql
ORDER BY MATCH(body, 'q'), MATCH(emb, [0.1, ...])
```
Normal `ORDER BY a, b` = "sort by a, break ties with b" — not fusion.

**B) Wrapper function** — explicit, but new syntax:
```sql
ORDER BY FUSION(MATCH(body, 'q'), MATCH(emb, [0.1, ...]))
```

**C) Keyword composition:**
```sql
ORDER BY MATCH(body, 'q') FUSED WITH MATCH(emb, [0.1, ...])
```

**D) Tuple arguments** — single call, avoids ORDER BY ambiguity:
```sql
ORDER BY MATCH((body, emb), ('query', [0.1, ...]))
```
But: CQL has no heterogeneous tuple args; variable arity; per-search options hard.

<!-- presenter notes:
Each approach has tradeoffs:

A) Comma-separated is the simplest extension of M1 syntax, but silently
redefines ORDER BY semantics. Without explicit fusion, it's ambiguous:
does it mean default fusion (RRF)? Sequential filter+rerank? The first
search selects candidates and the second rescores them? These are
fundamentally different operations.

B) FUSION() wrapper makes fusion explicit at the syntax level. The function
name itself signals "this is combined ranking, not sequential ordering."
Downside: it's a new syntactic construct.

C) FUSED WITH keyword reads naturally but is also a new syntactic invention.
Harder to parse and extend.

D) Tuple arguments avoid the ORDER BY ambiguity entirely — it's a single
expression. But CQL doesn't support heterogeneous tuples as function args.
Variable arity (2, 3, ... indexes) complicates the function signature.
Attaching per-search options (e.g. different oversampling for each search)
becomes harder inside a single call.

All of these push beyond what CQL currently supports. The M1 decision
constrains which extensions are natural later.
-->

---

# Next Features: Index Filtering

**Boolean pre-filter** — MATCH in WHERE, no ranking:
```sql
SELECT id FROM articles
WHERE MATCH(body, 'required terms')       -- boolean filter
ORDER BY embedding ANN OF [0.1, ...]
LIMIT 10;
```

**Score as filter** — threshold on ranking score:
```sql
SELECT id FROM articles
WHERE search_score() > 0.5
ORDER BY MATCH(body, 'query')
LIMIT 10;
```
Syntactically WHERE precedes ORDER BY, but logically the score
doesn't exist until after ranking. May need HAVING or post-filter clause.

Requires `search_score()` — see [proposal](https://scylladb.atlassian.net/wiki/spaces/RND/pages/284131411).
Alternative: repeat the search expression in WHERE with a score threshold,
e.g. `WHERE MATCH(body, 'query') > 0.5` — but complex queries make this
verbose and error-prone.

<!-- presenter notes:
Two different uses of search in WHERE:

1. Boolean pre-filter: MATCH in WHERE means "does this row match?" —
BM25 score is computed internally but discarded. Only match/no-match
matters. This narrows candidates before another search (ANN) ranks them.
Same function name, different semantics depending on clause:
WHERE = filter, ORDER BY = rank.

2. Score threshold filter: search_score() > N filters out low-quality
results after ranking. This is a post-filter — the search runs normally,
then rows below the threshold are excluded.

The search_score() function is proposed as a general ranking value
(see Confluence page 284131411). As a filter it connects to the
score-projection discussion but serves a different purpose: controlling
result quality rather than returning score to the caller.

Note: WHERE precedes ORDER BY syntactically, but search_score() depends
on ranking that hasn't happened yet. This is a logical contradiction —
the score doesn't exist until after the search runs. Possible solutions:
- HAVING clause (post-aggregation filter, SQL precedent)
- A new post-filter clause
- Allowing search_score() in WHERE as a special case that the engine
  evaluates after ranking internally
-->

---

# Next Features: Per-Query Options

How to pass search options (oversampling, rescoring, fusion strategy)?

**A) `USING` clause** — as a single options map or individual keywords:
```sql
-- Map style 
... USING SEARCH_OPTIONS { 'fusion': {'strategy': 'rrf'},
                           'oversampling': '5.0' };

-- Keyword style
... USING FUSION {'strategy': 'rrf'}
       AND OVERSAMPLING '5.0'
       AND RESCORING 'true';
```
Map is extensible without grammar changes; keywords are more readable
and CQL-native but each new option requires parser updates.

**B) Third argument to MATCH** — per-search options:
```sql
ORDER BY MATCH(body, 'query', {'boost': '2.0'}),
         MATCH(emb, [0.1, ...], {'oversampling': '5.0'})
```
Per-search granularity, but fusion options don't belong to either MATCH.

**C) Extended ORDER BY block** — options attached to expressions:
```sql
ORDER BY MATCH(body, 'q') WITH {'boost': '2.0'},
         MATCH(emb, [...]) WITH {'oversampling': '5.0'}
USING FUSION {'strategy': 'rrf'}
```

Likely **combination needed**: per-search + query-level options.

<!-- presenter notes:
The Search Option Validation document (Confluence page 284065844) proposes
USING SEARCH_OPTIONS as the main mechanism. It covers oversampling,
rescoring, and fusion. Three design options are discussed there:

Option 1: Literal structured map (no bind variables, full prepare-time
validation). Best for Scylla-side planning.

Option 2: Flat bindable string map (bind variables work, but weaker
prepare-time validation, nested options awkward).

Option 3: Split prepare-time/execute-time options (complex, harder to
explain).

The document notes Option 1 is simplest. No recommendation yet.

The map-style vs keyword-style tradeoff under approach A:
- Map style (USING SEARCH_OPTIONS {...}): a single opaque map. New options
  don't require grammar changes — just add keys. But Scylla parser can't
  validate individual options at parse time; validation moves to backend.
- Keyword style (USING FUSION {...} AND OVERSAMPLING '5.0'): each option
  is a grammar-level clause. More readable, CQL-native, and Scylla can
  validate syntax at parse time. But every new option requires a parser
  update and a Scylla release.

Per-search options (approach B/C) are not covered in that document but
are a natural extension. Some options are clearly per-search (boost,
oversampling for a specific index), while others are query-level (fusion
strategy, global limit behavior).

A third MATCH argument works if CQL can accept map literals as function
parameters. This is plausible but not standard CQL today.

The extended ORDER BY block (approach C) separates per-search options
from query-level options syntactically: WITH for per-search, USING for
query-level. Clean separation but more new syntax to parse.

In practice, a combination is likely needed: USING SEARCH_OPTIONS for
query-level options, and some per-search mechanism for index-specific
tuning.
-->

---

# Out of Scope: Extended Search Info in SELECT

Not the focus of this presentation, but worth noting — future work
also needs syntax for **returning** search-related data:

- **Score:** `search_score()` — general ranking value
  (ref: [Search Score Function](https://scylladb.atlassian.net/wiki/spaces/RND/pages/284131411))
- **Highlighting:** text snippets with matched terms marked
- **Facets:** count-by-category buckets alongside results
- **Aggregations:** stats over search results

These belong in the **SELECT** clause and are separate design questions.

<!-- presenter notes:
The search_score() proposal is documented in the General Search Score
Function page. It proposes a single parameterless function that returns
the ranking value used for the current row, regardless of search mode
(ANN, BM25, hybrid fusion). This replaces the earlier FTS_SCORE() idea.

Highlighting, facets, and aggregations are M4 features. They affect what
data is returned alongside search results but don't change how the search
condition itself is expressed. They'll need their own syntax discussion
(likely new functions or clauses in SELECT) but that's a separate topic.
-->

---

# Summary

**Decision needed for M1:**
- FTS search condition in **ORDER BY** (not WHERE as initially proposed)
- Form: function-style `F(col, query)`, Cassandra-style `col F OF query`, other ideas?
- Name: BM25 / MATCH / SEARCH / FTS / other?

**To keep in mind — M1 choice affects future syntax for:**
- Unified with ANN syntax
- Hybrid search composition (comma-separated, FUSION wrapper, keywords)
- ORDER BY semantics with multiple search expressions (ambiguity without explicit fusion)
- Boolean filtering vs scored filtering in WHERE
- Per-query options (`USING` clause, per-search arguments, or both)
- Score projection and filtering (`search_score()` function)
