from dataclasses import dataclass
from typing import Iterable, Mapping, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

SYSTEM_PROMPT = (
    "You are a strict fitness assistant. "
    "Only answer topics related to fitness, exercise, workouts, nutrition, recovery, sleep for performance, "
    "body composition, and healthy training habits. "
    "If a user asks about any non-fitness topic, refuse briefly and redirect to fitness topics. "
    "Do not provide illegal, dangerous, self-harm, hateful, sexual, or violent guidance. "
    "Do not claim medical diagnosis. Keep answers practical and concise."
)


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.4
    max_tokens: int = 500


def _to_openai_messages(messages: Iterable[Mapping[str, str]]) -> list[ChatCompletionMessageParam]:
    out: list[ChatCompletionMessageParam] = [
        cast(ChatCompletionMessageParam, {"role": "system", "content": SYSTEM_PROMPT})
    ]
    for item in messages:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role not in {"user", "assistant"}:
            continue
        out.append(cast(ChatCompletionMessageParam, {"role": role, "content": content}))
    return out


def generate_reply(messages: Iterable[Mapping[str, str]], config: LLMConfig) -> str:
    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    completion = client.chat.completions.create(
        model=config.model,
        messages=_to_openai_messages(messages),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    text = completion.choices[0].message.content or ""
    cleaned = text.strip()
    if not cleaned:
        return "I could not generate a response. Please try again."

    return cleaned
