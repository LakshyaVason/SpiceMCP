"""List the models available on the TAMU AI Chat API.

Run this once to discover which model strings the proxy actually exposes, then set
SPICE_MCP_MODEL in .env (or edit the default in spice_mcp_app/config.py).

    .venv\\Scripts\\activate
    python scripts/list_tamu_models.py

The key is read from TAMU_API_KEY in the environment or a git-ignored .env at the
repo root. It is never hardcoded and never logged.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "https://gateway.api.tamu.ai"


def resolve_base_url() -> str:
    """Use the same global override as the app runtime when available."""
    value = (os.environ.get("SPICE_MCP_BASE_URL") or "").strip().rstrip("/")
    if not value:
        return DEFAULT_BASE_URL
    for suffix in ("/v1", "/openai", "/api"):
        if value.lower().endswith(suffix):
            value = value[: -len(suffix)]
    return value or DEFAULT_BASE_URL


def load_api_key() -> str:
    """Read TAMU_API_KEY from the environment, falling back to the repo .env."""
    key = os.environ.get("TAMU_API_KEY")
    if key:
        return key.strip()

    # Minimal .env reader so this script stays dependency-light (requests only).
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == "TAMU_API_KEY":
                return value.strip().strip("'\"")

    sys.exit(
        "TAMU_API_KEY is not set.\n\n"
        f"Create {env_path} containing:\n"
        "    TAMU_API_KEY=your_key_here\n\n"
        ".env is already covered by .gitignore, so it will not be committed."
    )


def call_models_api(api_key: str) -> dict:
    base_url = resolve_base_url()
    url = f"{base_url}/v1/models"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> int:
    api_key = load_api_key()
    try:
        result = call_models_api(api_key)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        print(f"Error calling API: HTTP {status}", file=sys.stderr)
        if status == 401:
            print("The key was rejected. Check TAMU_API_KEY.", file=sys.stderr)
        if exc.response is not None:
            print(exc.response.text[:500], file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Error calling API: {exc}", file=sys.stderr)
        return 1

    # The response is OpenAI-shaped: {"object": "list", "data": [{"id": ...}, ...]}
    models = result.get("data")
    if isinstance(models, list):
        base_url = resolve_base_url()
        ids = sorted(str(m.get("id", "?")) for m in models)
        print(f"{len(ids)} model(s) available on {base_url}:\n")
        for model_id in ids:
            print(f"  {model_id}")
        print(
            "\nPick one and set it in .env:\n"
            "    SPICE_MCP_MODEL=<model-id>\n"
            "\nPrefer a model that supports tool/function calling - the agent loop "
            "depends on it. Verify with:\n"
            "    python scripts/probe_tool_calling.py <model-id>"
        )
    else:
        # Unexpected shape; dump it so we can adapt.
        print("Unexpected response shape, dumping raw JSON:\n")
        print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
