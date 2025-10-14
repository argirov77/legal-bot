import random
import string

from app.chunker import chunk_text


def generate_text(words: int = 200) -> str:
    random.seed(42)
    alphabet = string.ascii_letters + "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    tokens = []
    for _ in range(words):
        length = random.randint(3, 12)
        token_chars = [random.choice(alphabet) for _ in range(length)]
        tokens.append("".join(token_chars))
    return " ".join(tokens)


def test_chunk_text_covers_entire_input():
    text = generate_text(120)
    chunk_size = 80
    overlap = 15

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    assert chunks, "Expected at least one chunk"
    assert chunks[0]["meta"]["start"] == 0
    assert chunks[-1]["meta"]["end"] == len(text)

    coverage = [False] * len(text)
    previous_end = 0

    for chunk in chunks:
        start = chunk["meta"]["start"]
        end = chunk["meta"]["end"]

        assert 0 <= start < end <= len(text)
        assert start <= previous_end
        assert end > previous_end or previous_end == 0

        for index in range(start, end):
            coverage[index] = True

        previous_end = end

    assert all(coverage)


def test_chunk_text_respects_overlap_boundaries():
    text = generate_text(80)
    chunk_size = 60
    overlap = 10

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    for current, nxt in zip(chunks, chunks[1:]):
        current_end = current["meta"]["end"]
        next_start = nxt["meta"]["start"]

        assert current_end >= next_start
        assert current_end - next_start <= overlap
