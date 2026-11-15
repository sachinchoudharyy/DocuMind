import unittest
from types import SimpleNamespace

from app.chains.rag_chain import (
    ABSTENTION_ANSWER,
    DocumentRAGChain,
)


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.last_query = None

    def retrieve(self, query, k, neighbor_window):
        self.last_query = query
        return self.documents


class FakeAnswerChain:
    def __init__(self, answer):
        self.answer = answer
        self.last_input = None

    def invoke(self, payload):
        self.last_input = payload
        return self.answer


def make_document(text, source="annual_report.pdf", page=1):
    return SimpleNamespace(
        page_content=text,
        metadata={"source": source, "page": page},
    )


class RAGQualityTests(unittest.TestCase):
    def make_chain(self, docs, answer):
        chain = object.__new__(DocumentRAGChain)
        chain.retriever = FakeRetriever(docs)
        chain.top_k = 5
        chain.neighbor_window = 1
        chain.max_context_chars = 10000
        chain.answer_chain = FakeAnswerChain(answer)
        chain._resolve_question = lambda question, chat_history: (
            "What was the company's revenue in FY24-25?"
        )
        return chain

    def test_followup_uses_resolved_question_not_original_followup(self):
        chain = self.make_chain(
            [make_document("Revenue in FY24-25 was INR 10 crore.")],
            "The company's revenue in FY24-25 was INR 10 crore. [S1]",
        )
        result = chain.ask(
            "And what about the previous financial year?",
            chat_history=(
                "User: What was the company's revenue in FY25-26?\\n"
                "Assistant: Revenue was INR 12.5 crore. [S1]"
            ),
        )
        self.assertEqual(
            chain.answer_chain.last_input["question"],
            "What was the company's revenue in FY24-25?",
        )
        self.assertNotIn("chat_history", chain.answer_chain.last_input)
        self.assertIn("INR 10 crore", result["answer"])

    def test_unknown_citation_is_removed_and_uncited_answer_abstains(self):
        chain = self.make_chain(
            [make_document("Revenue in FY24-25 was INR 10 crore.")],
            "Revenue was INR 10 crore. [S99]",
        )
        result = chain.ask("What was revenue?")
        self.assertEqual(result["answer"], ABSTENTION_ANSWER)
        self.assertEqual(result["sources"], [])

    def test_valid_citation_is_returned_with_its_source(self):
        chain = self.make_chain(
            [make_document("Revenue in FY24-25 was INR 10 crore.")],
            "Revenue was INR 10 crore.[S1]",
        )
        result = chain.ask("What was revenue?")
        self.assertEqual(result["answer"], "Revenue was INR 10 crore. [S1]")
        self.assertEqual(result["sources"][0]["source"], "annual_report.pdf")

    def test_empty_retrieval_abstains(self):
        chain = self.make_chain([], "This answer should not be used. [S1]")
        result = chain.ask("What is the travel allowance?")
        self.assertEqual(result["answer"], ABSTENTION_ANSWER)
        self.assertEqual(result["sources"], [])
        self.assertEqual(chain.answer_chain.last_input, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
