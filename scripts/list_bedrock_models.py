"""List the Claude models and inference profiles this AWS account can actually invoke.

Run this before anything that costs tokens. `SPICE_MCP_MODEL` is a Bedrock *inference
profile id*, availability is region-scoped, and access has to be granted per model in the
account - so a wrong region and an ungranted model fail identically from the app's point
of view. This separates the two.

    .venv\\Scripts\\activate
    python scripts/list_bedrock_models.py
    python scripts/list_bedrock_models.py us-west-2      # probe another region

Bedrock has no HTTP `GET /v1/models`; the equivalent is the **control plane**, which is
the `bedrock` boto3 client and not `bedrock-runtime`. Credentials come from the standard
AWS chain - this script holds no key, exactly as the app holds none.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGION = "us-east-1"

# What the app actually asks for. Flagged in the output so a missing profile is obvious
# rather than something to spot by eye in a long list.
APP_DEFAULT_MODEL = "us.anthropic.claude-opus-5"


def resolve_region() -> str:
    """Resolve the region the same way the app runtime does.

    Deliberately a standalone copy rather than an import: this script is meant to run from
    `scripts/requirements.txt` alone, without the app's dependency tree. The precedence
    order must match `spice_mcp_app.config._resolve_region`, and
    `tests/test_config.py` loads this module by path to assert that it still does.
    """
    for name in ("SPICE_MCP_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value

    # Minimal .env reader so the script stays dependency-light (no python-dotenv).
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "SPICE_MCP_AWS_REGION":
                cleaned = value.strip().strip("'\"")
                if cleaned:
                    return cleaned

    return DEFAULT_REGION


def show_inference_profiles(client: object) -> list[str]:
    """Print the cross-region inference profiles, which is what SPICE_MCP_MODEL names."""
    profiles: list[dict] = []
    token: str | None = None
    while True:
        kwargs = {"maxResults": 100}
        if token:
            kwargs["nextToken"] = token
        page = client.list_inference_profiles(**kwargs)
        profiles.extend(page.get("inferenceProfileSummaries", []))
        token = page.get("nextToken")
        if not token:
            break

    claude = sorted(
        (p for p in profiles if "claude" in str(p.get("inferenceProfileId", "")).lower()),
        key=lambda p: str(p.get("inferenceProfileId")),
    )
    print(f"Claude inference profiles ({len(claude)} of {len(profiles)} total):\n")
    ids: list[str] = []
    for profile in claude:
        profile_id = str(profile.get("inferenceProfileId", "?"))
        ids.append(profile_id)
        status = str(profile.get("status", "?"))
        mark = "  <-- SPICE_MCP_MODEL default" if profile_id == APP_DEFAULT_MODEL else ""
        print(f"  {profile_id:<48} {status}{mark}")
    if not claude:
        print("  (none - see the note about model access below)")
    return ids


def show_foundation_models(client: object) -> None:
    """Print the base model ids, which say what the region carries at all.

    A model listed here but absent from the profiles above usually means on-demand access
    exists but the cross-region profile does not, or vice versa - worth seeing both.
    """
    response = client.list_foundation_models(byProvider="anthropic")
    summaries = response.get("modelSummaries", [])
    active = [
        m
        for m in summaries
        if str(m.get("modelLifecycle", {}).get("status", "ACTIVE")).upper() == "ACTIVE"
    ]
    print(f"\nAnthropic foundation models in this region ({len(active)} active):\n")
    for model in sorted(active, key=lambda m: str(m.get("modelId"))):
        kinds = ",".join(model.get("inferenceTypesSupported", []) or ["?"])
        print(f"  {str(model.get('modelId', '?')):<48} {kinds}")


def main() -> int:
    region = sys.argv[1] if len(sys.argv) > 1 else resolve_region()
    print(f"Region: {region}\n")

    # The control plane. `bedrock-runtime` is the invoke path and has no listing calls.
    client = boto3.client("bedrock", region_name=region)

    try:
        profile_ids = show_inference_profiles(client)
        show_foundation_models(client)
    except NoCredentialsError:
        print(
            "\nNo AWS credentials resolved.\n\n"
            "  The default chain looks at AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY,\n"
            "  AWS_PROFILE, ~/.aws/credentials, and instance/container roles. Set up any\n"
            "  one of them - `aws configure` is the usual answer - or put the keys in the\n"
            "  git-ignored .env (see .env.example).",
            file=sys.stderr,
        )
        return 1
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "?")
        print(f"\nBedrock refused the call: {code}", file=sys.stderr)
        if code in ("AccessDeniedException", "UnrecognizedClientException"):
            print(
                "  The credentials resolved but lack bedrock:ListInferenceProfiles /\n"
                "  bedrock:ListFoundationModels. Equivalent CLI call, for comparison:\n"
                f"    aws bedrock list-inference-profiles --region {region}",
                file=sys.stderr,
            )
        else:
            print(f"  {exc}", file=sys.stderr)
        return 1
    except BotoCoreError as exc:
        print(f"\nCould not reach Bedrock: {exc}", file=sys.stderr)
        return 1

    print(
        "\nSet the profile id in .env:\n"
        "    SPICE_MCP_MODEL=<inference-profile-id>\n"
        f"    SPICE_MCP_AWS_REGION={region}\n\n"
        "`us.` profiles are billed at a 10% premium over `global.` - prefer `global.`\n"
        "unless you need US-only routing. Then confirm the model does native tool use:\n"
        "    python scripts/probe_tool_calling.py"
    )
    if APP_DEFAULT_MODEL not in profile_ids:
        print(
            f"\nNOTE: {APP_DEFAULT_MODEL} is not in the list above. Either pick one that\n"
            f"      is, or grant access to it: Bedrock console -> Model access, in {region}.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
