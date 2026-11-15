
from app.chains.llm import LLMProvider


def main():
    llm = LLMProvider().get_model()

    response = llm.invoke(
        "Reply with exactly: DocuMind Groq connection successful."
    )

    print("\nGroq response:")
    print(response.content)


if __name__ == "__main__":
    main()
