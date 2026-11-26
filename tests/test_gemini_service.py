import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import httpx
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel

from gemini_service import GENERATION_UNAVAILABLE, GeminiGenerator, discover_text_models


class Report(BaseModel):
    score: int


def advertised(name, *actions):
    return SimpleNamespace(name=f"models/{name}", supported_actions=actions or ["generateContent"])


def client(*models):
    stub = Mock()
    stub.models.list.return_value = models
    stub.models.generate_content.return_value = SimpleNamespace(text='{"score": 85}')
    return stub


def api_error(code):
    return APIError(code, {"error": {"message": "provider detail", "status": "ERROR"}})


class GeminiGeneratorTests(TestCase):
    def test_discovers_only_versioned_text_models(self):
        stub = client(
            advertised("gemini-flash-latest"),
            advertised("gemini-3.1-flash-image"),
            advertised("gemini-3.1-flash", "embedContent"),
            advertised("gemini-3.1-flash-lite-preview"),
            advertised("gemini-2.5-flash"),
        )
        self.assertEqual(discover_text_models(stub), ["gemini-2.5-flash", "gemini-3.1-flash-lite-preview"])

    def test_falls_back_to_beta_and_reuses_working_model_after_v1_404(self):
        stable = client(advertised("gemini-2.5-flash"))
        stable.models.generate_content.side_effect = api_error(404)
        beta = client(advertised("gemini-3.1-flash-lite"))
        generator = GeminiGenerator({"v1": stable, "v1beta": beta})

        for _ in range(2):
            report, error = generator.generate("prompt", Report)
            self.assertIsNone(error)
            self.assertEqual(report.score, 85)
        stable.models.generate_content.assert_called_once()
        beta.models.list.assert_called_once()
        call = beta.models.generate_content.call_args.kwargs
        self.assertEqual(call["model"], "gemini-3.1-flash-lite")
        self.assertEqual(call["config"].response_mime_type, "application/json")

    def test_tries_another_model_for_malformed_output(self):
        stable = client(advertised("gemini-3.1-flash-lite"), advertised("gemini-2.5-flash"))
        stable.models.generate_content.side_effect = [
            SimpleNamespace(text="not JSON"),
            SimpleNamespace(text='{"score": 91}'),
        ]
        generator = GeminiGenerator({"v1": stable}, "models/gemini-2.5-flash")

        self.assertEqual(generator.generate("prompt", Report), (Report(score=91), None))
        self.assertEqual(
            [call.kwargs["model"] for call in stable.models.generate_content.call_args_list],
            ["gemini-2.5-flash", "gemini-3.1-flash-lite"],
        )

    def test_catalog_failure_does_not_prevent_generation(self):
        stable = client()
        stable.models.list.side_effect = httpx.ReadTimeout("listing timed out")
        generator = GeminiGenerator({"v1": stable})

        self.assertEqual(generator.generate("prompt", Report), (Report(score=85), None))

    def test_exhausted_models_do_not_expose_provider_errors(self):
        stable = client(advertised("gemini-2.5-flash"))
        stable.models.generate_content.side_effect = api_error(429)
        beta = client(advertised("gemini-3.1-flash-lite"))
        beta.models.generate_content.side_effect = api_error(503)

        self.assertEqual(
            GeminiGenerator({"v1": stable, "v1beta": beta}).generate("prompt", Report),
            (None, GENERATION_UNAVAILABLE),
        )

    def test_stops_fallback_when_time_budget_runs_out(self):
        stable = client(advertised("gemini-2.5-flash"))
        beta = client(advertised("gemini-3.1-flash-lite"))
        generator = GeminiGenerator({"v1": stable, "v1beta": beta})
        elapsed = [0]

        def timeout(**kwargs):
            self.assertEqual(kwargs["config"].http_options.timeout, 8_000)
            elapsed[0] = 10
            raise httpx.ReadTimeout("generation timed out")

        stable.models.generate_content.side_effect = timeout
        with patch("gemini_service.monotonic", side_effect=lambda: elapsed[0]):
            result = generator.generate("prompt", Report, budget_ms=10_000, attempt_timeout_ms=8_000)
        self.assertEqual(result, (None, GENERATION_UNAVAILABLE))
        beta.models.list.assert_not_called()

    def test_sdk_lists_and_generates_on_the_same_api_version(self):
        paths = []

        def handle(request):
            path = request.url.path
            paths.append(path)
            if request.method == "GET":
                return httpx.Response(200, json={"models": [{
                    "name": "models/gemini-2.5-flash",
                    "supportedGenerationMethods": ["generateContent"],
                }]})
            if path.startswith("/v1/"):
                return httpx.Response(404, json={"error": {"message": "not found", "status": "NOT_FOUND"}})
            config = json.loads(request.content)["generationConfig"]
            self.assertEqual(config["responseMimeType"], "application/json")
            return httpx.Response(200, json={"candidates": [{
                "content": {"parts": [{"text": '{"score": 85}'}]},
            }]})

        with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
            clients = {
                version: genai.Client(
                    api_key="test-key",
                    http_options=types.HttpOptions(api_version=version, httpx_client=transport),
                )
                for version in ("v1", "v1beta")
            }
            self.assertEqual(GeminiGenerator(clients).generate("prompt", Report), (Report(score=85), None))
        self.assertEqual(paths, [
            "/v1/models",
            "/v1/models/gemini-2.5-flash:generateContent",
            "/v1beta/models",
            "/v1beta/models/gemini-2.5-flash:generateContent",
        ])
