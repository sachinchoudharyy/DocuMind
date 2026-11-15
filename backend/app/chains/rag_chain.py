from __future__ import annotations

import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.chains.llm import LLMProvider
from app.retrieval.retriever import ContextualRetriever


QUESTION_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Convert the latest user message into ONE precise, standalone
document-search question.

Use the conversation history only to resolve references such as "it",
"that", "the previous year", or "what about the other one".

CRITICAL SCOPE RULES:
- Preserve the exact subject, entity, metric, policy, and comparison requested
  in the latest user message.
- If the latest message refers to a previous period, change ONLY the period;
  keep the metric from the immediately preceding user question.
- Do not broaden a question about revenue into a request for all financial
  metrics. Do not add related facts or extra metrics.
- Do not combine multiple earlier questions unless the latest message clearly
  asks for a comparison or summary.
- If the latest message is already standalone, return it unchanged.
- Do not answer the question and do not use document knowledge.
- Return only the standalone question, with no explanation or quotation marks.""",
        ),
        (
            "human",
            "Conversation history:\n{chat_history}\n\n"
            "Latest user message:\n{question}",
        ),
    ]
)


SYSTEM_PROMPT = """You are DocuMind, an evidence-grounded document assistant.

Your task is to answer ONLY the precise question supplied by the user, using
the retrieved document excerpts as evidence.

EVIDENCE AND SCOPE:
1. Treat the document excerpts as the only factual evidence.
2. Conversation history is not evidence. It may have been used to resolve the
   user's follow-up, but it must not cause you to answer additional questions.
3. Answer only the metric, entity, period, policy, or topic requested. Do not
   add adjacent metrics or a general summary unless explicitly requested.
4. Never invent facts, numbers, dates, policies, calculations, or explanations.
5. If the retrieved excerpts do not support an answer, say:
   "I could not find this information in the uploaded documents."
6. When a question asks whether a policy or entitlement is specified and the
   retrieved text does not state it, use careful wording such as:
   "The retrieved policy content does not specify [the requested item]."
   Do NOT say that the document explicitly states something is absent unless
   the document actually says so. Absence from retrieved excerpts is not an
   explicit denial or proof that no such policy exists elsewhere.
7. If only part of the requested answer is supported, give only that supported
   part and clearly identify what could not be found.

CITATIONS:
8. Cite every factual answer or material factual claim with one or more source
   labels that directly support that claim.
9. Use only labels supplied in the document context, in exact ASCII format:
   [S1], [S2], etc.
10. Do not invent source labels and do not cite a source that does not support
    the specific claim.
11. For an answer that reports that a retrieved policy excerpt does not specify
    an item, cite the relevant policy excerpt if one is present.

RESPONSE STYLE:
12. Be concise, professional, and direct.
13. Do not repeat the question.
14. Do not add a heading such as "Answer" unless it materially improves clarity.
15. Do not place a citation by itself on a separate line.

DOCUMENT EXCERPTS:
{context}
"""


ABSTENTION_ANSWER = (
    "I could not find this information in the uploaded documents."
)


class DocumentRAGChain:
    """Session-scoped RAG chain with focused follow-up resolution and citations."""

    def __init__(
        self,
        vector_store,
        all_chunks,
        top_k: int = 5,
        neighbor_window: int = 1,
        max_context_chars: int = 10000,
        llm: Any | None = None,
    ):
        self.retriever = ContextualRetriever(
            vector_store=vector_store,
            all_chunks=all_chunks,
        )
        self.top_k = top_k
        self.neighbor_window = neighbor_window
        self.max_context_chars = max_context_chars

        self.llm = llm or LLMProvider().get_model()

        self.question_rewriter = (
            QUESTION_REWRITE_PROMPT
            | self.llm
            | StrOutputParser()
        )

        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Precise user question (answer this question only):\n"
                    "{question}",
                ),
            ]
        )

        self.answer_chain = (
            self.answer_prompt
            | self.llm
            | StrOutputParser()
        )

    def _resolve_question(self, question: str, chat_history: str) -> str:
        history = (chat_history or "").strip()
        if not history or history == "No previous conversation.":
            return question.strip()

        rewritten = self.question_rewriter.invoke(
            {
                "chat_history": history,
                "question": question.strip(),
            }
        )
        standalone = str(rewritten).strip().strip('"').strip("'")
        return standalone or question.strip()

    def _prepare_context(self, documents):
        context_parts: list[str] = []
        source_references: list[dict] = []
        source_ids: dict[tuple[str, Any], str] = {}
        used_chars = 0

        for doc in documents:
            source = str(doc.metadata.get("source", "Unknown source"))
            page = doc.metadata.get("page", "Unknown")
            source_key = (source, page)

            if source_key not in source_ids:
                source_id = f"S{len(source_references) + 1}"
                source_ids[source_key] = source_id
                source_references.append(
                    {
                        "id": source_id,
                        "source": source,
                        "page": page,
                    }
                )
            else:
                source_id = source_ids[source_key]

            section = (
                f"[{source_id}] Source: {source}, Page: {page}\n"
                f"{doc.page_content.strip()}"
            )

            remaining = self.max_context_chars - used_chars
            if remaining <= 0:
                break

            if len(section) > remaining:
                section = section[:remaining]

            if section.strip():
                context_parts.append(section)
                used_chars += len(section)

        return "\n\n".join(context_parts), source_references

    @staticmethod
    def _normalize_citations(answer: str, valid_ids: set[str] | None = None) -> str:
        """Normalize common model citation brackets and remove unknown IDs."""
        text = str(answer or "").strip()
        text = re.sub(
            r"[【［〔]\s*(S\d+)\s*[】］〕]",
            r"[\1]",
            text,
        )
        text = re.sub(r"(?<=[A-Za-z0-9.,])(\[S\d+\])", r" \1", text)

        if valid_ids is not None:
            def keep_or_remove(match: re.Match) -> str:
                source_id = match.group(1)
                return match.group(0) if source_id in valid_ids else ""

            text = re.sub(r"\[(S\d+)\]", keep_or_remove, text)
            text = re.sub(r"[ \t]{2,}", " ", text)
            text = re.sub(r" +([.,;:])", r"\1", text)
            text = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", text)

        return text.strip()

    @staticmethod
    def _keep_cited_sources(
        answer: str,
        source_references: list[dict],
    ) -> list[dict]:
        cited_ids = set(re.findall(r"\[(S\d+)\]", answer))
        return [
            source
            for source in source_references
            if source["id"] in cited_ids
        ]

    @staticmethod
    def _has_valid_citation(answer: str, source_references: list[dict]) -> bool:
        valid_ids = {item["id"] for item in source_references}
        cited_ids = set(re.findall(r"\[(S\d+)\]", answer))
        return bool(valid_ids.intersection(cited_ids))

    def ask(self, question: str, chat_history: str = "") -> dict:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        standalone_question = self._resolve_question(
            question=question,
            chat_history=chat_history,
        )

        retrieved_documents = self.retriever.retrieve(
            query=standalone_question,
            k=self.top_k,
            neighbor_window=self.neighbor_window,
        )

        if not retrieved_documents:
            return {
                "answer": ABSTENTION_ANSWER,
                "sources": [],
                "resolved_question": standalone_question,
            }

        context, source_references = self._prepare_context(retrieved_documents)
        if not context.strip() or not source_references:
            return {
                "answer": ABSTENTION_ANSWER,
                "sources": [],
                "resolved_question": standalone_question,
            }

        # Important: send the resolved, focused question to answer generation.
        # Do not send the original ambiguous follow-up or the full chat history.
        # This prevents the answer model from expanding "previous year?" into
        # every metric mentioned in earlier turns.
        raw_answer = self.answer_chain.invoke(
            {
                "context": context,
                "question": standalone_question,
            }
        )

        valid_ids = {item["id"] for item in source_references}
        answer = self._normalize_citations(raw_answer, valid_ids=valid_ids)

        # A factual response without a usable source citation is not presented
        # as grounded. Return a safe abstention instead of an uncited assertion.
        if not answer or not self._has_valid_citation(answer, source_references):
            answer = ABSTENTION_ANSWER
            cited_sources = []
        else:
            cited_sources = self._keep_cited_sources(answer, source_references)

        return {
            "answer": answer,
            "sources": cited_sources,
            "resolved_question": standalone_question,
        }
