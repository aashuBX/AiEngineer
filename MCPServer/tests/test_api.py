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



