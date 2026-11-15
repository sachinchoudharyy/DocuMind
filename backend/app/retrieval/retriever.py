from collections import defaultdict

from langchain_core.documents import Document


class ContextualRetriever:
    """
    Retrieves relevant FAISS chunks and joins nearby chunks from the same
    source document and PDF page so facts split at chunk boundaries remain
    available to the answer-generation step.
    """

    def __init__(self, vector_store, all_chunks: list[Document]):
        self.vector_store = vector_store
        self.all_chunks = all_chunks

        self.chunk_lookup = {}
        for chunk in all_chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            chunk_id = chunk.metadata.get("chunk_id")

            if source is None or page is None or chunk_id is None:
                continue

            self.chunk_lookup[(source, page, chunk_id)] = chunk

    @staticmethod
    def _join_with_overlap(left: str, right: str) -> str:
        """Join adjacent chunk text without repeating their shared overlap."""
        max_overlap = min(len(left), len(right))

        for overlap_size in range(max_overlap, 0, -1):
            if left[-overlap_size:] == right[:overlap_size]:
                return left + right[overlap_size:]

        return left.rstrip() + "\n" + right.lstrip()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        neighbor_window: int = 1,
    ) -> list[Document]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if neighbor_window < 0:
            raise ValueError("neighbor_window cannot be negative.")

        search_results = self.vector_store.search(query, k=k)

        if not search_results:
            return []

        # Keep the best FAISS score for each chunk selected by similarity search.
        selected_scores = {}
        for doc, score in search_results:
            source = doc.metadata.get("source")
            page = doc.metadata.get("page")
            chunk_id = doc.metadata.get("chunk_id")

            if source is None or page is None or chunk_id is None:
                continue

            key = (source, page, chunk_id)
            selected_scores[key] = min(
                float(score),
                selected_scores.get(key, float("inf")),
            )

        # Expand each selected chunk to nearby chunks from the same PDF page.
        expanded_keys = set()
        for source, page, chunk_id in selected_scores:
            for offset in range(-neighbor_window, neighbor_window + 1):
                neighbor_id = chunk_id + offset
                key = (source, page, neighbor_id)
                if key in self.chunk_lookup:
                    expanded_keys.add(key)

        # Group expanded chunks by source and page, then combine consecutive IDs.
        grouped = defaultdict(list)
        for key in expanded_keys:
            source, page, chunk_id = key
            grouped[(source, page)].append(chunk_id)

        contextual_documents = []

        for (source, page), chunk_ids in grouped.items():
            sorted_ids = sorted(chunk_ids)
            consecutive_groups = []
            current_group = []

            for chunk_id in sorted_ids:
                if not current_group or chunk_id == current_group[-1] + 1:
                    current_group.append(chunk_id)
                else:
                    consecutive_groups.append(current_group)
                    current_group = [chunk_id]

            if current_group:
                consecutive_groups.append(current_group)

            for group in consecutive_groups:
                first_chunk = self.chunk_lookup[(source, page, group[0])]
                combined_text = first_chunk.page_content

                for chunk_id in group[1:]:
                    next_chunk = self.chunk_lookup[(source, page, chunk_id)]
                    combined_text = self._join_with_overlap(
                        combined_text,
                        next_chunk.page_content,
                    )

                group_scores = [
                    selected_scores[(source, page, chunk_id)]
                    for chunk_id in group
                    if (source, page, chunk_id) in selected_scores
                ]

                metadata = dict(first_chunk.metadata)
                metadata["chunk_ids"] = group
                metadata["source"] = source
                metadata["page"] = page
                metadata["retrieved_score"] = (
                    min(group_scores) if group_scores else None
                )

                contextual_documents.append(
                    Document(
                        page_content=combined_text,
                        metadata=metadata,
                    )
                )

        # Prefer contextual groups containing the strongest original match.
        contextual_documents.sort(
            key=lambda doc: (
                doc.metadata["retrieved_score"]
                if doc.metadata["retrieved_score"] is not None
                else float("inf")
            )
        )

        return contextual_documents
