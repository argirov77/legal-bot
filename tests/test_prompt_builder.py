from pathlib import Path

from app.prompt_builder import build_prompt


def test_build_prompt_includes_system_and_sorted_contexts() -> None:
    question = "Какъв е срокът за обжалване?"
    contexts = [
        {
            "content": "Член 2 регламентира срок от седем дни.",
            "score": 0.25,
            "metadata": {"source": "law-2"},
        },
        {
            "content": "Съгласно член 1 срокът е четиринадесет дни.",
            "score": 0.9,
            "metadata": {"source": "law-1"},
        },
    ]

    prompt = build_prompt(question, contexts)

    system_path = Path(__file__).resolve().parents[1] / "prompts" / "bg" / "system.txt"
    system_text = system_path.read_text(encoding="utf-8").strip()

    assert system_text in prompt

    first_context_content = contexts[1]["content"].strip()
    second_context_content = contexts[0]["content"].strip()

    assert prompt.index(first_context_content) < prompt.index(second_context_content)

    assert "[citation:law-1]" in prompt
    assert "[citation:law-2]" in prompt

    assert question in prompt
