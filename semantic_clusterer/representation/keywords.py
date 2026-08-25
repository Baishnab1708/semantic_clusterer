"""Keyword extraction for cluster output using class-based TF-IDF (c-TF-IDF).

Pure post-processing layer — called AFTER clustering is complete.
Does not influence cluster assignments, parameter selection, or any pipeline.

State-of-the-art c-TF-IDF with:
  1. L1-normalised TF per class (prevents long documents from dominating).
  2. BM25-style sublinear saturation (dampens very frequent terms).
  3. Corpus-aware automatic stop-word detection (domain-specific).
  4. MMR-style keyword diversification via character trigram Jaccard.
  5. Two-tier topic labels (short: 2 words, long: 3-4 words).
"""

from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Predicate verbs to demote during label generation.
# These words remain in the keywords list (they are informative) but are
# scored near-zero when picking words for the human-readable topic label.
# Root forms cover the most common sentence predicates in factual text.
# ---------------------------------------------------------------------------
_LABEL_VERB_DEMOTION: frozenset = frozenset({
    "add", "adds",
    "affect", "affects",
    "allow", "allows",
    "analyze", "analyzes",
    "apply", "applies",
    "assign", "assigns",
    "automate", "automates",
    "build", "builds",
    "calculate", "calculates",
    "capture", "captures",
    "classify", "classifies",
    "cluster", "clusters",
    "collect", "collects",
    "combine", "combines",
    "compute", "computes",
    "configure", "configures",
    "connect", "connects",
    "contain", "contains",
    "control", "controls",
    "convey", "conveys",
    "convert", "converts",
    "create", "creates",
    "decrease", "decreases",
    "define", "defines",
    "deploy", "deploys",
    "depend", "depends",
    "detect", "detects",
    "determine", "determines",
    "develop", "develops",
    "distribute", "distributes",
    "drive", "drives",
    "enable", "enables",
    "enhance", "enhances",
    "ensure", "ensures",
    "estimate", "estimates",
    "evaluate", "evaluates",
    "execute", "executes",
    "expand", "expands",
    "exploit", "exploits",
    "expose", "exposes",
    "extend", "extends",
    "extract", "extracts",
    "filter", "filters",
    "generate", "generates",
    "govern", "governs",
    "guide", "guides",
    "handle", "handles",
    "harm", "harms",
    "help", "helps",
    "identify", "identifies",
    "implement", "implements",
    "improve", "improves",
    "include", "includes",
    "increase", "increases",
    "integrate", "integrates",
    "interpret", "interprets",
    "involve", "involves",
    "learn", "learns",
    "limit", "limits",
    "link", "links",
    "maintain", "maintains",
    "make", "makes",
    "manage", "manages",
    "measure", "measures",
    "monitor", "monitors",
    "normalize", "normalizes",
    "occur", "occurs",
    "optimize", "optimizes",
    "organize", "organizes",
    "predict", "predicts",
    "prevent", "prevents",
    "process", "processes",
    "promote", "promotes",
    "protect", "protects",
    "provide", "provides",
    "reduce", "reduces",
    "regulate", "regulates",
    "relate", "relates",
    "remain", "remains",
    "remove", "removes",
    "replace", "replaces",
    "represent", "represents",
    "require", "requires",
    "resolve", "resolves",
    "restore", "restores",
    "retrieve", "retrieves",
    "reveal", "reveals",
    "run", "runs",
    "scale", "scales",
    "schedule", "schedules",
    "serve", "serves",
    "settle", "settles",
    "shape", "shapes",
    "show", "shows",
    "solve", "solves",
    "store", "stores",
    "support", "supports",
    "track", "tracks",
    "train", "trains",
    "transfer", "transfers",
    "transform", "transforms",
    "treat", "treats",
    "trigger", "triggers",
    "update", "updates",
    "use", "uses",
    "validate", "validates",
    "vary", "varies",
    "verify", "verifies",
    "visualize", "visualizes",
    "work", "works",
})

# Adjective/past-participle words that are poor anchors for topic labels.
# They are kept in the keyword list but demoted in label scoring.
_LABEL_ADJECTIVE_DEMOTION: frozenset = frozenset({
    "decentralized", "distributed", "automated", "optimized", "integrated",
    "adaptive", "collaborative", "scalable", "renewable", "sustainable",
    "secure", "encrypted", "digital", "virtual", "global", "local",
    "advanced", "multiple", "various", "common", "general", "specific",
    "effective", "efficient", "flexible", "robust", "reliable", "critical",
    "essential", "important", "primary", "secondary", "based", "driven",
})

# Suffixes that strongly indicate the word is a noun. Used to boost candidates
# for the topic label. Based on English morphology — generalises across domains.
_NOUN_SUFFIXES: Tuple[str, ...] = (
    "tion", "sion",   # encryption, classification, automation, compression
    "ity",            # security, reliability, scalability, electricity
    "ness",           # awareness, robustness, effectiveness, correctness
    "ment",           # management, development, deployment, assessment
    "ology", "logy",  # psychology, technology, biology, neurology
    "ics",            # analytics, robotics, mechanics, genomics, ethics
    "ance", "ence",   # performance, intelligence, resilience, compliance
    "ure",            # architecture, infrastructure, procedure, failure
    "ware",           # software, hardware, malware, firmware, middleware
    "work",           # network, framework, teamwork, groundwork
    "acy",            # privacy, accuracy, democracy, literacy
    "ery",            # discovery, recovery, surgery, delivery
    "gy",             # strategy, technology, energy
    "ty",             # safety, quality, property
    "al",             # signal, protocol, interval (weaker — also adjective)
)

# BM25 saturation constant — controls how quickly term frequency saturates.
_BM25_K1 = 1.5


def _compute_label_score(keyword: str, tfidf_score: float) -> float:
    """Score a keyword for topic-label suitability (not for keyword ranking).

    Applies multipliers to the raw TF-IDF score:
      bigram_mult:  clean 2-word bigrams strongly preferred (noun phrases)
      noun_mult:    words with noun-indicating suffixes are boosted
      verb_mult:    generic predicate verbs are heavily demoted or zeroed
      adj_mult:     standalone adjectives are demoted (poor label anchors)
    
    Bigrams with 3+ words in their parts, or bigrams where either word is a
    verb/adjective are penalised so they don't create messy labels.
    """
    words = keyword.lower().strip().split()

    # Reject keywords longer than 2 words outright — they produce messy labels
    if len(words) > 2:
        return 0.0

    is_bigram = len(words) == 2

    # Predicate verb check — if ANY word is a generic verb, reject entirely.
    # Topic labels should be pure noun phrases, not verb phrases.
    verb_count = sum(1 for w in words if w in _LABEL_VERB_DEMOTION)
    if verb_count > 0:
        return 0.0

    # Adjective/past-participle demotion
    adj_count = sum(1 for w in words if w in _LABEL_ADJECTIVE_DEMOTION)
    if adj_count == len(words):
        return 0.0   # All words are weak adjectives — skip
    adj_mult = 0.30 if adj_count > 0 else 1.0

    # Bigram bonus — only clean bigrams
    bigram_mult = 1.55 if is_bigram else 1.0

    # Noun suffix boost — check the last word
    last_word = words[-1]
    noun_mult = 1.0
    for suffix in _NOUN_SUFFIXES:
        if last_word.endswith(suffix) and len(last_word) > len(suffix) + 2:
            noun_mult = 1.35
            break

    return tfidf_score * bigram_mult * noun_mult * adj_mult


def _format_label_word(keyword: str) -> str:
    """Title-case each word in a keyword."""
    return " ".join(w.capitalize() for w in keyword.split())


def _char_trigrams(word: str) -> set:
    """Extract character trigrams from a word for fuzzy overlap detection."""
    w = word.lower()
    if len(w) < 3:
        return {w}
    return {w[i:i + 3] for i in range(len(w) - 2)}


def _char_trigram_jaccard(a: str, b: str) -> float:
    """Character trigram Jaccard similarity — more robust than prefix matching."""
    ta = _char_trigrams(a)
    tb = _char_trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _detect_corpus_stopwords(
    ctfidf: np.ndarray,
    vocab: np.ndarray,
    n_clusters: int,
    threshold: float = 0.80,
) -> set:
    """Identify words that appear uniformly across ≥threshold of clusters.

    These are corpus-specific stop words (e.g., "data" in a tech corpus,
    "patient" in a medical corpus) that are uninformative for distinguishing
    clusters. They are NOT removed from keyword lists but their c-TF-IDF
    scores are dampened by 0.1x.
    """
    if n_clusters < 2:
        return set()

    # A word is a corpus stopword if it appears (nonzero score) in ≥threshold
    # fraction of clusters AND its scores are relatively uniform (CV < 0.5).
    presence = (ctfidf > 0).sum(axis=0)  # per-word count of clusters it appears in
    prevalence = presence / n_clusters

    corpus_stops = set()
    for word_idx in range(ctfidf.shape[1]):
        if prevalence[word_idx] >= threshold:
            # Check coefficient of variation — if scores are similar across
            # clusters, the word is truly generic.
            scores = ctfidf[:, word_idx]
            nonzero_scores = scores[scores > 0]
            if len(nonzero_scores) >= 2:
                cv = float(np.std(nonzero_scores)) / (float(np.mean(nonzero_scores)) + 1e-10)
                if cv < 0.5:
                    corpus_stops.add(int(word_idx))

    return corpus_stops


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_cluster_keywords(
    texts: List[str],
    labels: np.ndarray,
    top_n: int = 10,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 1,
) -> Dict[int, List[Tuple[str, float]]]:
    """Extract top keywords per cluster using enhanced class-based TF-IDF.

    State-of-the-art c-TF-IDF with:
      - L1 normalisation per class (length invariance)
      - BM25 sublinear saturation (dampens frequency dominance)
      - Corpus-aware automatic stop-word detection
      - Higher keyword fidelity through these combined improvements

    Pure output-enrichment. Has no effect on cluster assignments.

    Args:
        texts: The texts that were clustered (aligned with labels).
        labels: Cluster label per text (-1 = noise, excluded).
        top_n: Number of top keywords to return per cluster.
        ngram_range: Min/max n-gram size for keyword candidates.
        min_df: Minimum document frequency for a word to be considered.

    Returns:
        Dict mapping cluster_id -> [(keyword, score), ...] sorted descending.
        Noise cluster (-1) is excluded.
    """
    try:
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError:
        return {}

    labels = np.asarray(labels)
    unique_labels = sorted(int(lb) for lb in np.unique(labels) if lb >= 0)

    if not unique_labels or not texts:
        return {}

    # One concatenated document per cluster
    cluster_docs: List[str] = []
    doc_label_order: List[int] = []

    for label in unique_labels:
        mask = labels == label
        cluster_texts = [
            texts[i] for i in range(len(texts))
            if i < len(labels) and mask[i]
        ]
        if cluster_texts:
            cluster_docs.append(" ".join(cluster_texts))
            doc_label_order.append(label)

    if not cluster_docs:
        return {}

    try:
        vectorizer = CountVectorizer(
            min_df=min_df,
            max_features=15_000,
            ngram_range=ngram_range,
            stop_words="english",
            strip_accents="unicode",
            lowercase=True,
        )
        tf_matrix = vectorizer.fit_transform(cluster_docs)
    except Exception:
        return {}

    vocab = vectorizer.get_feature_names_out()
    tf_dense = tf_matrix.toarray().astype(np.float64)

    # ── L1-normalise TF per class (length invariance) ────────────────
    row_sums = tf_dense.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    tf_norm = tf_dense / row_sums

    # ── BM25-style sublinear saturation ──────────────────────────────
    # Dampens the impact of very frequent terms while boosting rare-but-
    # distinctive ones. k1=1.5 is the standard BM25 saturation constant.
    tf_saturated = (tf_norm * (_BM25_K1 + 1)) / (tf_norm + _BM25_K1)

    # ── c-TF-IDF:  score(word, cluster) = tf_saturated * log(1 + A / tf_global)
    words_per_doc = tf_dense.sum(axis=1)
    A = max(float(np.mean(words_per_doc)), 1.0)
    tf_global = tf_dense.sum(axis=0)
    idf = np.log1p(A / (tf_global + 1e-10))
    ctfidf = tf_saturated * idf[np.newaxis, :]

    # ── Corpus-aware stop-word demotion ──────────────────────────────
    n_clusters = len(doc_label_order)
    corpus_stops = _detect_corpus_stopwords(ctfidf, vocab, n_clusters)
    if corpus_stops:
        for word_idx in corpus_stops:
            ctfidf[:, word_idx] *= 0.1  # demote, don't remove

    result: Dict[int, List[Tuple[str, float]]] = {}
    for idx, label in enumerate(doc_label_order):
        scores = ctfidf[idx]
        nonzero = np.where(tf_dense[idx] > 0)[0]
        if len(nonzero) == 0:
            result[label] = []
            continue
        sort_order = np.argsort(scores[nonzero])[::-1]
        top_indices = nonzero[sort_order[:top_n]]
        result[label] = [
            (str(vocab[i]), float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    return result


def generate_topic_label(
    keywords: List[Tuple[str, float]],
    max_words: int = 4,
) -> str:
    """Generate a concise human-readable topic label from ranked keywords.

    Scores all candidates (unigrams and bigrams) by label suitability:
    - Bigrams are preferred — they are almost always noun phrases
    - Noun-suffix words are boosted (e.g. 'encryption', 'analytics')
    - Generic predicate verbs are demoted (e.g. 'protects', 'ensures')
    - MMR-style diversification via character trigram Jaccard

    Args:
        keywords: List of (keyword, tfidf_score) tuples, score descending.
        max_words: Kept for API compatibility. Labels use top-2 candidates.

    Returns:
        A human-readable topic label, e.g. "Machine Learning" or
        "Cybersecurity & Encryption". Falls back gracefully on edge cases.
    """
    if not keywords:
        return ""

    # Score top-20 candidates
    scored: List[Tuple[str, float]] = []
    for kw, score in keywords[:20]:
        ls = _compute_label_score(kw, score)
        if ls > 0.0:
            scored.append((kw, ls))

    if not scored:
        # Fallback: title-case the top keyword as-is
        return _format_label_word(keywords[0][0])

    scored.sort(key=lambda x: x[1], reverse=True)

    # Pick top-2 non-overlapping candidates using MMR-style diversification.
    # Use character trigram Jaccard instead of simple 5-char prefix matching
    # to catch near-synonyms like "security"/"secure", "transaction"/"transactions".
    selected: List[str] = []
    selected_trigrams: List[set] = []

    for kw, _ in scored:
        # Check character trigram Jaccard overlap with all already-selected words
        kw_words = kw.lower().split()
        too_similar = False
        for sel_kw in selected:
            sel_words = sel_kw.lower().split()
            # Check each word pair for overlap
            for w1 in kw_words:
                for w2 in sel_words:
                    if _char_trigram_jaccard(w1, w2) > 0.5:
                        too_similar = True
                        break
                if too_similar:
                    break
            if too_similar:
                break

        if too_similar:
            continue

        selected.append(kw)
        if len(selected) == 2:
            break

    if not selected:
        return _format_label_word(keywords[0][0])

    parts = [_format_label_word(w) for w in selected]
    return parts[0] if len(parts) == 1 else f"{parts[0]} & {parts[1]}"
