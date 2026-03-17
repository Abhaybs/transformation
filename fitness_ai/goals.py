from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class GoalPreset:
    key: str
    display_name: str
    positive_prompt: str
    negative_prompt: str
    recommended_strength: float
    recommended_guidance_scale: float
    recommended_steps: int


_GOAL_PRESETS: Dict[str, GoalPreset] = {
    "muscle_gain": GoalPreset(
        key="muscle_gain",
        display_name="Muscle Gain",
        positive_prompt=(
            "same person, preserve exact facial identity, preserve face shape, "
            "athletic muscular physique, toned arms and shoulders, realistic skin texture, "
            "gym setting, high detail, photorealistic"
        ),
        negative_prompt=(
            "different person, altered face, blurry, deformed body, extra limbs, "
            "cartoon, low quality, nsfw, shirtless"
        ),
        recommended_strength=0.35,
        recommended_guidance_scale=8.5,
        recommended_steps=30,
    ),
    "fat_loss": GoalPreset(
        key="fat_loss",
        display_name="Fat Loss",
        positive_prompt=(
            "same person, preserve exact facial identity, preserve face shape, "
            "lean athletic physique, lower body fat, realistic skin texture, "
            "gym setting, high detail, photorealistic"
        ),
        negative_prompt=(
            "different person, altered face, blurry, deformed body, extra limbs, "
            "cartoon, low quality, nsfw, shirtless"
        ),
        recommended_strength=0.33,
        recommended_guidance_scale=8.2,
        recommended_steps=30,
    ),
}

_GOAL_ALIASES = {
    "1": "muscle_gain",
    "2": "fat_loss",
    "musclegain": "muscle_gain",
    "fatloss": "fat_loss",
    "muscle": "muscle_gain",
    "lean": "fat_loss",
    "cut": "fat_loss",
}


def list_goals() -> List[str]:
    return sorted(_GOAL_PRESETS.keys())


def get_goal_preset(goal: str) -> GoalPreset:
    normalized_goal = goal.strip().lower().replace(" ", "_")
    normalized_goal = _GOAL_ALIASES.get(normalized_goal, normalized_goal)

    if normalized_goal not in _GOAL_PRESETS:
        available = ", ".join(list_goals())
        raise ValueError(f"Unknown goal '{goal}'. Supported goals: {available}")

    return _GOAL_PRESETS[normalized_goal]
