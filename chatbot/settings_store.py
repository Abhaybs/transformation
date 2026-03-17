import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class ChatbotSettings:
    llm_provider: str = "OpenAI"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_temperature: float = 0.4
    llm_max_tokens: int = 500


SETTINGS_DIR = Path(__file__).resolve().parent / ".runtime"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"


def load_settings() -> ChatbotSettings:
    if not SETTINGS_PATH.exists():
        return ChatbotSettings()

    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ChatbotSettings()

    return ChatbotSettings(
        llm_provider=str(data.get("llm_provider", "OpenAI")),
        llm_api_key=str(data.get("llm_api_key", "")),
        llm_model=str(data.get("llm_model", "gpt-4o-mini")),
        llm_base_url=str(data.get("llm_base_url", "https://api.openai.com/v1")),
        llm_temperature=float(data.get("llm_temperature", 0.4)),
        llm_max_tokens=int(data.get("llm_max_tokens", 500)),
    )


def save_settings(settings: ChatbotSettings) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")

    try:
        os.chmod(SETTINGS_PATH, 0o600)
    except OSError:
        # Some Windows environments may not fully support Unix-style chmod semantics.
        pass
