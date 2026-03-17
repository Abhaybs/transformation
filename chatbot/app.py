import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fitness_ai.config import ModelConfig
from fitness_ai.goals import list_goals
from fitness_ai.pipeline import create_pipeline
from fitness_ai.schemas import GenerationRequest
from fitness_ai.transform_service import TransformationService

from chatbot.security import (
    RateLimiter,
    is_fitness_related,
    load_rgb_image,
    sanitize_message,
    secure_output_path,
    validate_image_upload,
    validate_message,
)
from chatbot.llm_service import LLMConfig, generate_reply
from chatbot.settings_store import ChatbotSettings, load_settings, save_settings

APP_ROOT = Path(__file__).resolve().parent
GENERATED_DIR = APP_ROOT / "generated"
LOG_PATH = APP_ROOT / "logs" / "audit.log"
DEFAULT_OUTPUT_EXT = ".png"
PROVIDER_DEFAULT_BASE_URLS = {
    "OpenAI": "https://api.openai.com/v1",
    "OpenRouter": "https://openrouter.ai/api/v1",
    "Groq": "https://api.groq.com/openai/v1",
    "Together": "https://api.together.xyz/v1",
    "DeepSeek": "https://api.deepseek.com/v1",
    "Custom": "",
}


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def enforce_optional_access_token() -> None:
    expected_token = os.getenv("FITNESS_CHATBOT_ACCESS_TOKEN", "").strip()
    if not expected_token:
        return

    provided_token = st.sidebar.text_input("Access token", type="password")
    if provided_token != expected_token:
        st.error("Unauthorized. Provide a valid access token.")
        st.stop()


@st.cache_resource(show_spinner="Loading fitness transformation model...")
def get_transform_service() -> TransformationService:
    config = ModelConfig(
        device="cpu",
        dtype="float32",
        disable_safety_checker=False,
        use_dpm_solver=True,
    )
    pipeline = create_pipeline(config)
    return TransformationService(pipeline)


def generate_fitness_response(message: str) -> str:
    text = message.lower()

    if any(token in text for token in ["diet", "nutrition", "meal", "protein", "calories"]):
        return (
            "For nutrition, prioritize protein intake, whole foods, hydration, and consistent meal timing. "
            "A simple baseline is lean protein each meal, vegetables, and calorie control matched to your goal."
        )

    if any(token in text for token in ["workout", "exercise", "training", "gym", "reps", "sets"]):
        return (
            "For training, keep a structured split and progressive overload. "
            "Start with 3 to 4 sessions per week, focus on compound lifts, and track reps and load weekly."
        )

    if any(token in text for token in ["fat loss", "cut", "lean", "weight loss"]):
        return (
            "For fat loss, use a moderate calorie deficit, keep protein high, "
            "and include resistance training plus light cardio for recovery and adherence."
        )

    if any(token in text for token in ["muscle", "bulk", "gain", "hypertrophy"]):
        return (
            "For muscle gain, aim for a small calorie surplus, progressive overload, "
            "and enough recovery sleep. Keep your weekly volume consistent before adding more exercises."
        )

    if any(token in text for token in ["sleep", "recovery", "stress"]):
        return (
            "Recovery drives results. Target 7 to 9 hours of sleep, manage stress, and keep at least 1 full rest day weekly."
        )

    return (
        "I can help with fitness training, nutrition, recovery, and body transformation planning. "
        "If you want image transformation, upload a photo and use /transform muscle_gain or /transform fat_loss."
    )


def parse_goal_from_message(message: str, selected_goal: str) -> str:
    tokens = sanitize_message(message).lower().split()
    if len(tokens) >= 2 and tokens[0] == "/transform":
        candidate = tokens[1].strip()
        if candidate in set(list_goals()):
            return candidate
    return selected_goal


def run_transformation(
    uploaded_file,
    goal: str,
    prompt_suffix: str,
    strength: Optional[float],
    guidance_scale: Optional[float],
    steps: Optional[int],
    seed: Optional[int],
) -> Tuple[str, Optional[Path]]:
    if uploaded_file is None:
        return "Please upload an image in the sidebar before requesting a transformation.", None

    image_bytes = uploaded_file.getvalue()
    image_check = validate_image_upload(uploaded_file.name, image_bytes)
    if not image_check.allowed:
        return image_check.reason or "Image validation failed.", None

    input_path = secure_output_path(GENERATED_DIR, "input", extension=DEFAULT_OUTPUT_EXT)
    output_path = secure_output_path(GENERATED_DIR, "output", extension=DEFAULT_OUTPUT_EXT)

    image = load_rgb_image(image_bytes)
    image.save(input_path)

    service = get_transform_service()
    request = GenerationRequest(
        input_image_path=str(input_path),
        output_image_path=str(output_path),
        goal=goal,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        prompt_suffix=prompt_suffix,
        seed=seed,
    )

    result = service.generate(request)
    summary = (
        "Transformation complete. "
        f"Goal={result.goal}, Strength={result.strength:.2f}, Guidance={result.guidance_scale:.2f}, "
        f"Steps={result.num_inference_steps}, Latency={result.latency_seconds:.2f}s"
    )
    return summary, Path(result.output_image_path)


def initialize_state() -> None:
    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "I am your fitness assistant. I only answer fitness topics. "
                    "Use /transform muscle_gain or /transform fat_loss after uploading an image."
                ),
                "image": None,
            }
        ]

    if "rate_limiter" not in st.session_state:
        st.session_state.rate_limiter = RateLimiter()


def render_llm_controls() -> ChatbotSettings:
    current: ChatbotSettings = st.session_state.settings

    st.header("LLM Settings")
    providers = list(PROVIDER_DEFAULT_BASE_URLS.keys())
    default_index = providers.index(current.llm_provider) if current.llm_provider in providers else 0
    llm_provider = st.selectbox("API Provider", options=providers, index=default_index)

    provider_default_url = PROVIDER_DEFAULT_BASE_URLS[llm_provider]
    base_url_help = "Endpoint for selected provider. For custom provider, set full OpenAI-compatible base URL."
    if llm_provider != "Custom":
        llm_base_url = st.text_input("Base URL", value=provider_default_url, help=base_url_help)
    else:
        llm_base_url = st.text_input("Base URL", value=current.llm_base_url, help=base_url_help)

    llm_model = st.text_input("Model", value=current.llm_model)
    llm_temperature = st.slider("Temperature", 0.0, 1.0, float(current.llm_temperature), 0.05)
    llm_max_tokens = st.slider("Max tokens", 128, 1200, int(current.llm_max_tokens), 32)
    llm_api_key = st.text_input("LLM API Key", value=current.llm_api_key, type="password")

    col_save, col_clear = st.columns(2)
    if col_save.button("Save LLM settings"):
        resolved_base_url = llm_base_url.strip() or provider_default_url
        updated = ChatbotSettings(
            llm_provider=llm_provider,
            llm_api_key=llm_api_key.strip(),
            llm_model=llm_model.strip() or "gpt-4o-mini",
            llm_base_url=resolved_base_url,
            llm_temperature=float(llm_temperature),
            llm_max_tokens=int(llm_max_tokens),
        )
        save_settings(updated)
        st.session_state.settings = updated
        st.success("LLM settings saved on this machine.")

    if col_clear.button("Clear saved key"):
        resolved_base_url = llm_base_url.strip() or provider_default_url
        cleared = ChatbotSettings(
            llm_provider=llm_provider,
            llm_api_key="",
            llm_model=llm_model.strip() or "gpt-4o-mini",
            llm_base_url=resolved_base_url,
            llm_temperature=float(llm_temperature),
            llm_max_tokens=int(llm_max_tokens),
        )
        save_settings(cleared)
        st.session_state.settings = cleared
        st.info("Saved API key removed.")

    resolved_base_url = llm_base_url.strip() or provider_default_url
    return ChatbotSettings(
        llm_provider=llm_provider,
        llm_api_key=llm_api_key.strip(),
        llm_model=llm_model.strip() or current.llm_model,
        llm_base_url=resolved_base_url or current.llm_base_url,
        llm_temperature=float(llm_temperature),
        llm_max_tokens=int(llm_max_tokens),
    )


def generate_llm_fitness_response(settings: ChatbotSettings) -> str:
    if not settings.llm_api_key:
        return "No API key configured. Add your LLM API key in sidebar and click Save LLM settings."

    chat_messages = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages
        if item["role"] in {"user", "assistant"}
    ]

    cfg = LLMConfig(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    try:
        return generate_reply(chat_messages, cfg)
    except Exception:
        logging.exception("llm_generation_failed")
        return "LLM request failed. Check provider, API key, model, base URL, and network connectivity."


def render_chat_history() -> None:
    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.write(item["content"])
            if item.get("image"):
                st.image(str(item["image"]), caption="Generated transformation")


def main() -> None:
    setup_logging()
    initialize_state()

    st.set_page_config(page_title="Fitness AI Chatbot", page_icon="💪", layout="wide")
    st.title("Fitness AI Chatbot")
    st.caption("Fitness-only chat assistant with secure image transformation controls.")

    with st.sidebar:
        active_settings = render_llm_controls()
        st.divider()
        st.header("Transformation Controls")
        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear face/body image for transformation.",
        )
        selected_goal = st.selectbox("Goal", options=list_goals(), index=0)
        prompt_suffix = st.text_input("Prompt suffix (optional)", value="")
        custom_strength = st.checkbox("Override strength", value=False)
        strength = st.slider("Strength", 0.2, 0.7, 0.35, 0.01) if custom_strength else None
        custom_guidance = st.checkbox("Override guidance scale", value=False)
        guidance_scale = (
            st.slider("Guidance scale", 6.0, 12.0, 8.5, 0.1) if custom_guidance else None
        )
        custom_steps = st.checkbox("Override steps", value=False)
        steps = st.slider("Inference steps", 15, 60, 30, 1) if custom_steps else None
        custom_seed = st.checkbox("Use fixed seed", value=False)
        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=7) if custom_seed else None
        enforce_optional_access_token()

        st.markdown("### Commands")
        st.markdown("- /transform muscle_gain")
        st.markdown("- /transform fat_loss")
        st.markdown("- /help")

    render_chat_history()

    user_message = st.chat_input("Ask a fitness question or run /transform")
    if not user_message:
        return

    raw_message = user_message
    sanitized_message = sanitize_message(raw_message)

    st.session_state.messages.append({"role": "user", "content": sanitized_message, "image": None})
    with st.chat_message("user"):
        st.write(sanitized_message)

    rate_result = st.session_state.rate_limiter.check()
    if not rate_result.allowed:
        response = rate_result.reason or "Rate limit exceeded."
        st.session_state.messages.append({"role": "assistant", "content": response, "image": None})
        with st.chat_message("assistant"):
            st.write(response)
        return

    msg_result = validate_message(sanitized_message)
    if not msg_result.allowed:
        response = msg_result.reason or "Message blocked."
        st.session_state.messages.append({"role": "assistant", "content": response, "image": None})
        with st.chat_message("assistant"):
            st.write(response)
        logging.warning("blocked_message | reason=%s", response)
        return

    if sanitized_message.lower().startswith("/help"):
        help_text = (
            "I can only discuss fitness.\n"
            "Use /transform muscle_gain or /transform fat_loss to run image transformation.\n"
            "Upload an image in the sidebar first."
        )
        st.session_state.messages.append({"role": "assistant", "content": help_text, "image": None})
        with st.chat_message("assistant"):
            st.write(help_text)
        return

    if sanitized_message.lower().startswith("/transform"):
        resolved_goal = parse_goal_from_message(sanitized_message, selected_goal)
        with st.chat_message("assistant"):
            with st.spinner("Generating transformation..."):
                try:
                    reply, image_path = run_transformation(
                        uploaded_file=uploaded_file,
                        goal=resolved_goal,
                        prompt_suffix=prompt_suffix,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        steps=steps,
                        seed=int(seed) if seed is not None else None,
                    )
                except Exception:
                    logging.exception("transform_failed")
                    reply, image_path = (
                        "Transformation failed due to an internal error. Check logs and retry.",
                        None,
                    )

            st.write(reply)
            if image_path:
                st.image(str(image_path), caption="Generated transformation")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
                "image": str(image_path) if image_path else None,
            }
        )
        logging.info("transform_request | goal=%s | status=%s", resolved_goal, "ok" if image_path else "failed")
        return

    if not is_fitness_related(sanitized_message):
        refusal = (
            "I can only assist with fitness topics such as workouts, nutrition, recovery, "
            "and body transformation."
        )
        st.session_state.messages.append({"role": "assistant", "content": refusal, "image": None})
        with st.chat_message("assistant"):
            st.write(refusal)
        logging.info("off_topic_blocked")
        return

    response = generate_llm_fitness_response(active_settings)
    st.session_state.messages.append({"role": "assistant", "content": response, "image": None})
    with st.chat_message("assistant"):
        st.write(response)

    logging.info("fitness_chat | prompt=%s", sanitized_message)


if __name__ == "__main__":
    main()
