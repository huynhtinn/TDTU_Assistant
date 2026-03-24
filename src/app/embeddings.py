"""
Shared E5 Embedding wrapper.

E5 (intfloat/multilingual-e5-base) requires mandatory prefixes:
  - 'passage: ' when indexing documents (embed_documents)
  - 'query: '   when searching (embed_query)
Skipping the prefix degrades retrieval accuracy by ~5-10%.
"""

from langchain_huggingface import HuggingFaceEmbeddings


class E5Embeddings(HuggingFaceEmbeddings):

    def embed_documents(self, texts: list) -> list:
        prefixed = [f"passage: {t}" for t in texts]
        return super().embed_documents(prefixed)

    def embed_query(self, text: str) -> list:
        return super().embed_query(f"query: {text}")


def get_shared_embedding_model() -> E5Embeddings:
    return E5Embeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
