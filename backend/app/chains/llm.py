
from langchain_groq import ChatGroq
from app.config import settings


class LLMProvider:
    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Add it to your .env file."
            )

        self.llm = ChatGroq(
            model=settings.GROQ_MODEL,
            temperature=0,
            api_key=settings.GROQ_API_KEY,
        )

    def get_model(self):
        return self.llm
