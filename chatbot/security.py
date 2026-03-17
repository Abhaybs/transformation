import io
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

MAX_MESSAGE_CHARS = 700
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_REQUESTS_PER_WINDOW = 20
RATE_LIMIT_WINDOW_SECONDS = 60
MIN_SECONDS_BETWEEN_REQUESTS = 2
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

FITNESS_KEYWORDS = {
    "fitness",
    "workout",
    "exercise",
    "training",
    "muscle",
    "fat",
    "diet",
    "nutrition",
    "protein",
    "cardio",
    "strength",
    "mobility",
    "weight",
    "physique",
    "gym",
    "reps",
    "sets",
    "calories",
    "body",
    "transformation",
    "meal",
    "sleep",
    "hydration",
}

DISALLOWED_PATTERNS = [
    re.compile(r"ignore\\s+previous\\s+instructions", re.IGNORECASE),
    re.compile(r"system\\s+prompt", re.IGNORECASE),
    re.compile(r"<script", re.IGNORECASE),
    re.compile(r"\\.\\./"),
    re.compile(r"cmd\\.exe", re.IGNORECASE),
    re.compile(r"powershell", re.IGNORECASE),
    re.compile(r"rm\\s+-rf", re.IGNORECASE),
    re.compile(r"drop\\s+table", re.IGNORECASE),
]


@dataclass(frozen=True)
class MessageValidationResult:
    allowed: bool
    reason: Optional[str] = None


class RateLimiter:
    def __init__(self) -> None:
        self.request_timestamps = []

    def check(self, now: Optional[float] = None) -> MessageValidationResult:
        current = now if now is not None else time.time()
        self.request_timestamps = [
            ts for ts in self.request_timestamps if current - ts <= RATE_LIMIT_WINDOW_SECONDS
        ]

        if self.request_timestamps and current - self.request_timestamps[-1] < MIN_SECONDS_BETWEEN_REQUESTS:
            return MessageValidationResult(
                allowed=False,
                reason="You are sending messages too quickly. Please wait a moment.",
            )

        if len(self.request_timestamps) >= MAX_REQUESTS_PER_WINDOW:
            return MessageValidationResult(
                allowed=False,
                reason="Rate limit reached. Please wait and try again shortly.",
            )

        self.request_timestamps.append(current)
        return MessageValidationResult(allowed=True)


def sanitize_message(text: str) -> str:
    no_control_chars = re.sub(r"[\\x00-\\x1f\\x7f]", " ", text)
    return re.sub(r"\\s+", " ", no_control_chars).strip()


def validate_message(text: str) -> MessageValidationResult:
    sanitized = sanitize_message(text)

    if not sanitized:
        return MessageValidationResult(False, "Please enter a message.")

    if len(sanitized) > MAX_MESSAGE_CHARS:
        return MessageValidationResult(
            False,
            f"Message too long. Keep it under {MAX_MESSAGE_CHARS} characters.",
        )

    for pattern in DISALLOWED_PATTERNS:
        if pattern.search(sanitized):
            return MessageValidationResult(
                False,
                "Message blocked by security guardrails. Please rephrase.",
            )

    return MessageValidationResult(True)


def is_fitness_related(text: str) -> bool:
    normalized = sanitize_message(text).lower()

    if normalized.startswith("/"):
        return True

    return any(keyword in normalized for keyword in FITNESS_KEYWORDS)


def validate_image_upload(file_name: str, file_bytes: bytes) -> MessageValidationResult:
    extension = Path(file_name).suffix.lower()

    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        return MessageValidationResult(
            False,
            "Unsupported image format. Use JPG, JPEG, PNG, or WEBP.",
        )

    if len(file_bytes) > MAX_IMAGE_BYTES:
        return MessageValidationResult(
            False,
            f"Image too large. Keep it under {MAX_IMAGE_BYTES // (1024 * 1024)}MB.",
        )

    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError):
        return MessageValidationResult(False, "Invalid image file.")

    return MessageValidationResult(True)


def load_rgb_image(file_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(file_bytes)) as image:
        return image.convert("RGB")


def secure_output_path(output_dir: Path, prefix: str, extension: str = ".png") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{prefix}_{uuid.uuid4().hex}{extension}"
    return output_dir / safe_name
