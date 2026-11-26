"""Select a Gemini text model available to this API key and validate its output."""

import logging
import re
from time import monotonic

import httpx
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import ValidationError

try:
    import httpx2
except ImportError:  # google-genai 1.x uses httpx instead
    httpx2 = None

logger = logging.getLogger(__name__)
API_VERSIONS = ("v1", "v1beta")
FLASH_MODEL = re.compile(r"gemini-(\d+(?:\.\d+)*)-flash(-lite)?(-preview(?:-\d+(?:-\d+)*)?)?(?:-\d{3})?")
CATALOG_TTL = 900
GENERATION_UNAVAILABLE = "Gemini is temporarily unavailable. Please try again in a few minutes."
TRANSPORT_ERRORS = (httpx.TransportError,) + ((httpx2.TransportError,) if httpx2 else ())
REQUEST_ERRORS = (APIError,) + TRANSPORT_ERRORS
INVALID_RESPONSE_ERRORS = (ValidationError,) + TRANSPORT_ERRORS


def make_client(api_key: str, version: str, timeout_ms: int = 35_000):
    return genai.Client(
        api_key=api_key.strip(),
        http_options=types.HttpOptions(api_version=version, timeout=timeout_ms),
    )


def discover_text_models(client) -> list[str]:
    """Only choose versioned Flash models advertised for generateContent."""
    models = set()
    for model in client.models.list():
        name = (model.name or "").removeprefix("models/")
        if "generateContent" in (model.supported_actions or []) and FLASH_MODEL.fullmatch(name):
            models.add(name)

    def priority(name):
        version, lite, preview = FLASH_MODEL.fullmatch(name).groups()
        return (not preview, tuple(map(int, version.split("."))), not lite, name)

    return sorted(models, key=priority, reverse=True)


class GeminiGenerator:
    def __init__(self, clients, preferred_model: str = ""):
        self.clients = clients
        self.preferred_model = preferred_model.strip().removeprefix("models/")
        self.catalogs = {}
        self.last_working = None

    @classmethod
    def from_key(cls, api_key: str, preferred_model: str = ""):
        return cls({version: make_client(api_key, version) for version in API_VERSIONS}, preferred_model)

    def _models_for(self, version):
        now = monotonic()
        expires, models = self.catalogs.get(version, (0, []))
        if now >= expires:
            try:
                models = discover_text_models(self.clients[version])
                self.catalogs[version] = (monotonic() + CATALOG_TTL, models)
            except REQUEST_ERRORS as error:
                logger.warning("Gemini model listing failed on %s (%s)", version, getattr(error, "code", type(error).__name__))
                # Some deployments can generate even if model listing is unavailable.
                if not models:
                    models = ["gemini-3.1-flash-lite", "gemini-2.5-flash"]
                    if FLASH_MODEL.fullmatch(self.preferred_model):
                        models.insert(0, self.preferred_model)
                self.catalogs[version] = (monotonic() + 60, models)
        preferred = [self.preferred_model] if self.preferred_model in models else []
        last_working = self.last_working
        if last_working and last_working[0] == version and last_working[1] in models:
            preferred.insert(0, last_working[1])
        return list(dict.fromkeys(preferred + models))

    def generate(self, prompt, schema):
        versions = list(self.clients)
        last_working = self.last_working
        if last_working:
            versions.remove(last_working[0])
            versions.insert(0, last_working[0])
        config = types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema, temperature=0.1
        )
        for version in versions:
            for model in self._models_for(version)[:2]:
                try:
                    response = self.clients[version].models.generate_content(
                        model=model, contents=prompt, config=config
                    )
                    report = schema.model_validate_json(response.text or "")
                    self.last_working = (version, model)
                    return report, None
                except APIError as error:
                    logger.warning("Gemini generation failed on %s/%s (HTTP %s)", version, model, error.code)
                    if error.code == 404:
                        expires, current = self.catalogs[version]
                        remaining = [name for name in current if name != model]
                        if not remaining:
                            expires = min(expires, monotonic() + 60)
                        self.catalogs[version] = (expires, remaining)
                    if error.code == 401:
                        return None, GENERATION_UNAVAILABLE
                except INVALID_RESPONSE_ERRORS as error:
                    logger.warning("Gemini generation failed on %s/%s (%s)", version, model, type(error).__name__)
        return None, GENERATION_UNAVAILABLE
