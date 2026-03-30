import pytest
from unittest.mock import patch, MagicMock
from game.core.llm import LLM, build_tools


def test_llm_init():
    llm = LLM()
    assert llm.model == 'openai/gpt-4o-mini'
    assert llm.temperature == 0
    assert llm.max_tokens == 2048


def test_llm_init_custom_params():
    llm = LLM(model='custom_model', temperature=0.5, max_tokens=100)
    assert llm.model == 'custom_model'
    assert llm.temperature == 0.5
    assert llm.max_tokens == 100


@patch('game.core.llm.completion')
def test_llm_generate_without_tools(mock_completion):
    llm = LLM()
    prompt = [{'role': 'user', 'content': 'Hello!'}]
    llm.generate(prompt=prompt, max_tokens=50)
    mock_completion.assert_called_once_with(
        model=llm.model,
        messages=prompt,
        temperature=llm.temperature,
        max_tokens=50
    )


@patch('game.core.llm.completion')
def test_llm_generate_with_tools(mock_completion):
    llm = LLM()
    prompt = [{'role': 'user', 'content': 'Hello!'}]
    tools = [{'type': 'function', 'function': {'name': 'test_function', 'description': 'A test function', 'parameters': {}}}]
    llm.generate(prompt=prompt, tools=tools)
    mock_completion.assert_called_once_with(
        model=llm.model,
        messages=prompt,
        temperature=llm.temperature,
        tools=tools,
        tool_choice='auto',
        max_tokens=llm.max_tokens
    )


@patch('game.core.llm.LLM.generate')
def test_llm_generate_text(mock_generate):
    llm = LLM()
    prompt = [{'role': 'user', 'content': 'Hello!'}]
    mock_generate.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='Generated text'))])
    result = llm.generate_text(prompt=prompt, max_tokens=50)
    mock_generate.assert_called_once_with(prompt=prompt, max_tokens=50)
    assert result == 'Generated text'


@patch('game.core.llm.ActionRegistry')
def test_build_tools(mock_action_registry):
    mock_action_registry_instance = mock_action_registry.return_value
    action1 = MagicMock(name='action1', description='First action', parameters={})
    action2 = MagicMock(name='action2', description='Second action', parameters={})
    mock_action_registry_instance.list_actions.return_value = [action1, action2]
    tools = build_tools(mock_action_registry_instance)
    assert len(tools) == 2
    assert tools[0]['function']['name'] == action1.name
    assert tools[1]['function']['name'] == action2.name