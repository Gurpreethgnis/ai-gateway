#!/usr/bin/env python3
"""Stage 7 smoke tests for dashboard API additions."""

import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.routers.dashboard_api import router as dashboard_api_router


class Stage7DashboardEndpointTests(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(dashboard_api_router)
        self.client = TestClient(self.app)

    def test_provider_summary_endpoint_shape(self):
        with patch("gateway.config.ENABLE_DASHBOARD_AUTH", False):
            resp = self.client.get("/api/providers/summary")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("providers", data)
        self.assertIsInstance(data["providers"], list)
        # expected static set exists even if none are configured
        names = {item.get("provider") for item in data["providers"]}
        for expected in {"anthropic", "openai", "gemini", "groq", "ollama"}:
            self.assertIn(expected, names)

    def test_routing_trace_endpoint_shape(self):
        with patch("gateway.config.ENABLE_DASHBOARD_AUTH", False):
            resp = self.client.get("/api/routing/trace?limit=5")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("events", data)
        self.assertIsInstance(data["events"], list)

    def test_provider_status_endpoint_no_crash(self):
        with patch("gateway.config.ENABLE_DASHBOARD_AUTH", False):
            resp = self.client.get("/api/providers/status")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("providers", data)
        self.assertIsInstance(data["providers"], dict)


if __name__ == "__main__":
    unittest.main()
