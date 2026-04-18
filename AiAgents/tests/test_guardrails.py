"""Tests for guardrail validators."""
import pytest
from unittest.mock import MagicMock


class TestInputValidator:
    """Test input validation and safety checks."""

    def test_input_validator_initialization(self):
        from src.guardrails.input_validator import InputValidator
        validator = InputValidator()
        assert validator is not None

    def test_input_validator_has_validate(self):
        from src.guardrails.input_validator import InputValidator
        validator = InputValidator()
        assert hasattr(validator, "validate")


class TestOutputValidator:
    """Test output validation."""

    def test_output_validator_initialization(self):
        from src.guardrails.output_validator import OutputValidator
        validator = OutputValidator()
        assert validator is not None


class TestActionValidator:
    """Test action validation for dangerous tool calls."""

    def test_action_validator_initialization(self):
        from src.guardrails.action_validator import ActionValidator
        validator = ActionValidator()
        assert validator is not None


class TestSafetyConfig:
    """Test safety configuration loading."""

    def test_safety_config_import(self):
        from src.guardrails.safety_config import SafetyConfig
        config = SafetyConfig()
        assert config is not None
