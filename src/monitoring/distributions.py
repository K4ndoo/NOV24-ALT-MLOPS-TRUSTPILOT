from collections import Counter


def word_counts(df):
    """Nombre d'occurrences de chaque mot (plus de deux lettres, lettres uniquement)."""
    words = Counter()
    for text in df["text"]:
        words.update(w for w in text.split() if len(w) > 2 and w.isalpha())
    return words


def exclusive_words(df, other, n=15, min_count=3):
    """Mots fréquents dans `df` qui n'apparaissent jamais dans `other`."""
    seen_elsewhere = word_counts(other)
    own = word_counts(df)
    return [(w, c) for w, c in own.most_common() if c >= min_count and w not in seen_elsewhere][:n]


def as_text(pairs):
    """« mot (12), autre (8) » : format lisible pour les métadonnées d'un rapport."""
    return ", ".join(f"{w} ({c})" for w, c in pairs) or "aucun"
