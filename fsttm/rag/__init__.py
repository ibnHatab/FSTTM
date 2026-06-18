"""Manual RAG: ingest a PDF → embed → retrieve grounded context for how-to /
where-is / explain questions. Pairs with the `manual` intent domain."""
from fsttm.rag.store import Embedder, VectorStore  # noqa: F401
from fsttm.rag.retrieve import Retriever, build_answer_prompt  # noqa: F401
