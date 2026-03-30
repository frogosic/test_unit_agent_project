import pytest
from game.core.agent_registry import AgentRegistry, RegisteredAgent


def mock_callable():
    pass


@pytest.fixture
def agent_registry():
    return AgentRegistry()


def test_register_agent_success(agent_registry):
    agent_registry.register_agent('agent1', mock_callable, 'An agent for testing')
    assert len(agent_registry._agents) == 1
    assert agent_registry.get_agent('agent1') == RegisteredAgent(name='agent1', run_callable=mock_callable, description='An agent for testing')


def test_register_agent_empty_name(agent_registry):
    with pytest.raises(ValueError):
        agent_registry.register_agent('', mock_callable)


def test_register_agent_duplicate_name(agent_registry):
    agent_registry.register_agent('agent1', mock_callable)
    with pytest.raises(ValueError):
        agent_registry.register_agent('agent1', mock_callable)


def test_get_agent_success(agent_registry):
    agent_registry.register_agent('agent1', mock_callable)
    assert agent_registry.get_agent('agent1') == RegisteredAgent(name='agent1', run_callable=mock_callable)


def test_get_agent_non_existing(agent_registry):
    assert agent_registry.get_agent('non_existing_agent') is None


def test_require_agent_success(agent_registry):
    agent_registry.register_agent('agent1', mock_callable)
    assert agent_registry.require_agent('agent1') == RegisteredAgent(name='agent1', run_callable=mock_callable)


def test_require_agent_non_existing(agent_registry):
    with pytest.raises(ValueError):
        agent_registry.require_agent('non_existing_agent')


def test_allow_calls_success(agent_registry):
    agent_registry.register_agent('agent1', mock_callable)
    agent_registry.register_agent('agent2', mock_callable)
    agent_registry.allow_calls('agent1', ['agent2'])
    assert agent_registry.can_call('agent1', 'agent2') is True


def test_allow_calls_unregistered(agent_registry):
    agent_registry.register_agent('agent1', mock_callable)
    with pytest.raises(ValueError):
        agent_registry.allow_calls('agent1', ['non_existing_agent'])


def test_can_call_success(agent_registry):
    agent_registry.register_agent('agent1', mock_callable)
    agent_registry.register_agent('agent2', mock_callable)
    agent_registry.allow_calls('agent1', ['agent2'])
    assert agent_registry.can_call('agent1', 'agent2') is True


def test_can_call_failure(agent_registry):
    agent_registry.register_agent('agent1', mock_callable)
    agent_registry.register_agent('agent3', mock_callable)
    assert agent_registry.can_call('agent1', 'agent3') is False


def test_normalize_name_success():
    assert AgentRegistry._normalize_name(' agent1 ') == 'agent1'


def test_normalize_name_empty():
    with pytest.raises(ValueError):
        AgentRegistry._normalize_name(' ')