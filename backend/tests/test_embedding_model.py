import unittest
from unittest.mock import patch

from app.embeddings import embedding_model
from app.embeddings.embedding_model import EmbeddingModel
from app.vectorstore.faiss_store import FAISSVectorStore


class EmbeddingModelReuseTests(unittest.TestCase):
    def setUp(self):
        self.previous_model = embedding_model._embedding_model
        embedding_model._embedding_model = None

    def tearDown(self):
        embedding_model._embedding_model = self.previous_model

    @patch("app.embeddings.embedding_model.HuggingFaceEmbeddings")
    def test_wrapper_and_faiss_share_one_model_instance(self, model_class):
        fake_model = object()
        model_class.return_value = fake_model

        first_wrapper = EmbeddingModel()
        second_wrapper = EmbeddingModel()
        first_store = FAISSVectorStore()
        second_store = FAISSVectorStore(index_path="temporary-index")

        self.assertIs(first_wrapper.get_model(), fake_model)
        self.assertIs(second_wrapper.get_model(), fake_model)
        self.assertIs(first_store.embedding_model, fake_model)
        self.assertIs(second_store.embedding_model, fake_model)
        model_class.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
