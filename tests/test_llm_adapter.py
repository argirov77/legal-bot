from app.llm import MockLLMAdapter


def test_mock_llm_adapter_generate_returns_expected_string() -> None:
    adapter = MockLLMAdapter()
    adapter.load(model_path="mock-model")

    result = adapter.generate("Привет", max_tokens=32)

    assert result == "Mock response to 'Привет' with max_tokens=32."
