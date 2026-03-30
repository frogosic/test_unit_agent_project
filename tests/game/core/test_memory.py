import pytest
from unittest.mock import Mock
import json
from game.core.memory import Memory


def test_add_memory():
    memory = Memory()
    memory.add_memory({'key': 'value'})
    assert len(memory.get_memories()) == 1
    assert memory.get_memories()[0] == {'key': 'value'}


def test_add_user_message():
    memory = Memory()
    memory.add_user_message('Hello, world!')
    assert len(memory.get_memories()) == 1
    assert memory.get_memories()[0] == {'role': 'user', 'content': 'Hello, world!'}


def test_add_assistant_message():
    memory = Memory()
    mock_message = Mock()
    mock_message.content = 'Hello from assistant'
    mock_message.tool_calls = []
    memory.add_assistant_message(mock_message)
    assert len(memory.get_memories()) == 1
    assert memory.get_memories()[0] == {'role': 'assistant', 'content': 'Hello from assistant'}


def test_add_assistant_message_with_tool_calls():
    memory = Memory()
    mock_function = Mock(name='example_function', arguments={})
    mock_tool_call = Mock(id='1', type='example', function=mock_function)
    mock_message = Mock()
    mock_message.content = 'Hello from assistant'
    mock_message.tool_calls = [mock_tool_call]
    memory.add_assistant_message(mock_message)
    assert len(memory.get_memories()) == 1
    assert memory.get_memories()[0]['tool_calls'][0]['id'] == '1'


def test_add_tool_result():
    memory = Memory()
    memory.add_tool_result('tool_call_1', {'result': 'success'})
    assert len(memory.get_memories()) == 1
    assert memory.get_memories()[0] == {'role': 'tool', 'tool_call_id': 'tool_call_1', 'content': json.dumps({'result': 'success'})}


def test_get_memories():
    memory = Memory()
    assert len(memory.get_memories()) == 0