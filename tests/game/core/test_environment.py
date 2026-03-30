import pytest
from unittest.mock import MagicMock, patch
from game.core.environment import Environment

class TestEnvironment:
    @pytest.fixture
    def environment(self):
        return Environment()

    @pytest.fixture
    def mock_action_instance(self):
        return MagicMock()

    @pytest.fixture
    def mock_action_context(self):
        return MagicMock()

    def test_successful_action_execution(self, environment, mock_action_instance, mock_action_context):
        mock_action_instance.execute.return_value = 'mock_result'
        action_args = {'arg1': 'value1'}
        result = environment.execute_action(mock_action_instance, action_args, mock_action_context)
        assert result['tool_executed'] is True
        assert result['result'] is not None
        assert 'timestamp' in result

    def test_action_execution_raises_exception(self, environment, mock_action_instance, mock_action_context):
        mock_action_instance.execute.side_effect = Exception('mock error')
        action_args = {'arg1': 'value1'}
        result = environment.execute_action(mock_action_instance, action_args, mock_action_context)
        assert result['tool_executed'] is False
        assert 'error' in result
        assert 'traceback' in result
        assert 'timestamp' in result

    def test_successful_result_formatting(self, environment):
        mock_result = 'mock_result'
        result = environment.format_result(mock_result)
        assert result['tool_executed'] is True
        assert result['result'] == mock_result
        assert 'timestamp' in result

    def test_empty_result_formatting(self, environment):
        result = environment.format_result(None)
        assert result['tool_executed'] is True
        assert result['result'] is None
        assert 'timestamp' in result