import pytest
from game.core.memory import Memory


@pytest.fixture
def memory():
    return Memory()


def test_add_memory(memory):
    test_memory = {'key': 'value'}
    memory.add_memory(test_memory)
    assert memory.get_memories() == [test_memory]


def test_add_user_message(memory):
    content = 'Hello, world!'
    memory.add_user_message(content)
    assert memory.get_memories() == [{'role': 'user', 'content': content}]


def test_add_assistant_message(memory):
    class MockMessage:
        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    message = MockMessage('Assistant response')
    memory.add_assistant_message(message)
    assert memory.get_memories()[-1] == {'role': 'assistant', 'content': 'Assistant response'}


def test_add_tool_result(memory):
    tool_call_id = '123'
    result = {'result_key': 'result_value'}
    memory.add_tool_result(tool_call_id, result)
    assert memory.get_memories()[-1] == {'role': 'tool', 'tool_call_id': tool_call_id, 'content': '{"result_key": "result_value"}'}


def test_get_memories(memory):
    assert memory.get_memories() == []
    memory.add_user_message('Test message')
    assert len(memory.get_memories()) == 1
    assert memory.get_memories()[0]['role'] == 'user'