import re

def tokenize(text):
    return set(re.findall(r'\w+', text.lower()))

def compute_reward(source: str, candidate: str) -> float:
    source_words = tokenize(source)
    candidate_words = tokenize(candidate)

    if not candidate_words:
        return 0.0

    # 1. Core overlap (strong weight)
    overlap_ratio = len(source_words & candidate_words) / len(source_words)

    # 2. Negation mismatch detection
    negation_words = {"not", "no", "never"}
    source_neg = any(w in source_words for w in negation_words)
    candidate_neg = any(w in candidate_words for w in negation_words)

    negation_penalty = 0.4 if source_neg != candidate_neg else 0.0

    # 3. Directional contradiction
    contradiction_pairs = [
        ("east", "west"),
        ("minima", "maxima"),
        ("landlocked", "coastal")
    ]

    contradiction_penalty = 0.0
    for a, b in contradiction_pairs:
        if a in source_words and b in candidate_words:
            contradiction_penalty += 0.4
        if b in source_words and a in candidate_words:
            contradiction_penalty += 0.4

    # 4. Soft hallucination penalty (lighter now)
    hallucinated = candidate_words - source_words
    hallucination_ratio = len(hallucinated) / max(len(candidate_words), 1)
    hallucination_penalty = hallucination_ratio * 0.2  # reduced weight

    # 5. Verbosity penalty
    length_ratio = len(candidate_words) / len(source_words)
    verbosity_penalty = 0.2 if length_ratio > 2 else 0.0

    # Weighted scoring (overlap dominates)
    score = (
        0.8 * overlap_ratio
        - negation_penalty
        - contradiction_penalty
        - hallucination_penalty
        - verbosity_penalty
    )

    return max(0.0, min(1.0, score))