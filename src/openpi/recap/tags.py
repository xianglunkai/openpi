"""Shared ACP prompt tags used by both training and inference.

Matches Evo-RL ``lerobot.rl.acp_tags`` and RLinf OpenPI CFG prompts.
"""

ACP_TAG_KEY = "Advantage"
ACP_POSITIVE_VALUE = "positive"
ACP_NEGATIVE_VALUE = "negative"

ACP_POSITIVE_TAG = f"{ACP_TAG_KEY}: {ACP_POSITIVE_VALUE}"
ACP_NEGATIVE_TAG = f"{ACP_TAG_KEY}: {ACP_NEGATIVE_VALUE}"


def build_acp_tagged_task(task: str | None, is_positive: bool) -> str:
    tag = ACP_POSITIVE_TAG if is_positive else ACP_NEGATIVE_TAG
    base_task = task or ""
    if not base_task:
        return tag
    return f"{base_task}\n{tag}"
