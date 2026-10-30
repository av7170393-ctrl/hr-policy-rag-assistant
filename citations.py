"""Pick the exact sentence that supports an answer and highlight it."""
from app.rag.textutils import split_sentences, tokenize


def best_sentence(chunk_text: str, query: str, answer: str = "") -> str | None:
    target = set(tokenize(query + " " + answer))
    best, best_score = None, 0.0
    for sentence in split_sentences(chunk_text):
        tokens = set(tokenize(sentence))
        overlap = len(target & tokens)
        # normalise by length so a focused sentence beats a long one with the same overlap
        score = overlap / (len(tokens) ** 0.5) if overlap else 0.0
        if score > best_score:
            best, best_score = sentence, score
    return best


def highlight(text: str, sentence: str | None) -> str:
    if sentence and sentence in text:
        return text.replace(sentence, f"**{sentence}**", 1)
    return text
