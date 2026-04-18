"""Tests for FastAPI backend API."""
import pytest
from fastapi.testclient import TestClient


class TestBackendAPI:
    """Test the FastAPI backend endpoints."""

    def _get_client(self):
        from src.backend_api.app import app
        return TestClient(app)

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        client = self._get_client()
        response = client.get("/health")
        assert response.status_code == 200

    def test_root_endpoint(self):
        """Test the root endpoint."""
        client = self._get_client()
        response = client.get("/")
        assert response.status_code == 200


class TestCalculatorServer:
    """Test calculator tool functions."""

    def test_calculator_import(self):
        from src.servers.calculator.server import mcp
        assert mcp.name == "calculator_server"

    def test_calculate_basic_math(self):
        from src.servers.calculator.server import calculate
        assert calculate("2 + 2") == "4"
        assert calculate("10 * 5") == "50"

    def test_calculate_division_by_zero(self):
        from src.servers.calculator.server import calculate
        result = calculate("1 / 0")
        assert "Error" in result

    def test_calculate_rejects_code_injection(self):
        from src.servers.calculator.server import calculate
        result = calculate("__import__('os').system('rm -rf /')")
        assert "Error" in result

    def test_fibonacci(self):
        from src.servers.calculator.server import fibonacci
        assert fibonacci(0) == "0"
        assert fibonacci(1) == "1"
        assert fibonacci(10) == "55"

    def test_fibonacci_out_of_range(self):
        from src.servers.calculator.server import fibonacci
        result = fibonacci(-1)
        assert "Error" in result
        result = fibonacci(101)
        assert "Error" in result
