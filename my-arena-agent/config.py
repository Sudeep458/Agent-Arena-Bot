"""
config.py — centralized environment loader and validator
"""
from pathlib import Path
import os
from dotenv import load_dotenv

# Try to load a project-local .env first (or a secrets submodule)
_env_paths = [Path(__file__).parent / '.env', Path(__file__).parent.parent / 'secrets' / '.env']
for p in _env_paths:
    if p.exists():
        load_dotenv(p)
        break
else:
    # fallback to system environment
    load_dotenv()

REQUIRED_SECRETS = ["ID_TOKEN", "GEMINI_API_KEY"]
OPTIONAL_SECRETS = ["TRACELOOP_API_KEY", "OPENCODE_GO_API_KEY", "OPENCODE_GO_MODEL"]


def load_config() -> dict:
    """Load and validate environment configuration.

    Returns a dict with keys for all known config variables.
    Raises ValueError if required secrets are missing.
    """
    cfg = {}
    missing = []
    for key in REQUIRED_SECRETS:
        cfg[key] = os.environ.get(key, "").strip()
        if not cfg[key]:
            missing.append(key)

    for key in OPTIONAL_SECRETS:
        if key == "OPENCODE_GO_MODEL":
            cfg[key] = os.environ.get(key, "kimi-k2.6")
        else:
            cfg[key] = os.environ.get(key, "")

    if missing:
        raise ValueError(f"Missing required secrets: {', '.join(missing)}. Copy .env.example to .env and fill values.")

    return cfg


if __name__ == '__main__':
    try:
        cfg = load_config()
        print("All required config values present.")
    except Exception as e:
        print(f"Configuration error: {e}")
        raise SystemExit(1)
