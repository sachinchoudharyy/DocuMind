from datetime import datetime, timezone


class ChatHistory:
    """Utilities for formatting persisted conversation messages."""

    @staticmethod
    def format_messages(messages: list[dict], max_messages: int = 10) -> str:
        if not messages:
            return "No previous conversation."

        recent_messages = messages[-max_messages:]
        formatted = []

        for message in recent_messages:
            role = message.get("role", "unknown").capitalize()
            content = message.get("content", "").strip()

            if content:
                formatted.append(f"{role}: {content}")

        return "\n".join(formatted) or "No previous conversation."

    @staticmethod
    def timestamp_now() -> str:
        return datetime.now(timezone.utc).isoformat()
