from abc import ABC, abstractmethod

from app.logger import setup_logger
from app.product.catalog import load_products, products_by_id

logger = setup_logger("retrieval")


class ProductRepository(ABC):
    """Interface for product data access.

    The concrete implementation talks to a vector database for semantic search.
    """

    @abstractmethod
    def search(self, query: str) -> list[dict[str, object]]: ...

    @abstractmethod
    def get_all(self) -> list[dict[str, object]]: ...


class ChromaProductRepository(ProductRepository):
    """Semantic product search backed by ChromaDB (RAG retrieval step).

    The catalog is embedded once by the offline indexer (embedding/index.py) into
    a remote ChromaDB collection. At query time the user's question is embedded
    with the same model and the collection returns the top-k most similar products
    by cosine distance — this is real retrieval, not keyword matching.

    The ChromaDB client is connected lazily on first use so that importing this
    module (and the DI container) never performs network I/O. Catalog metadata is
    kept local: the collection returns ids, which we map back to full products.
    """

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        embedding_model: str = "text-embedding-3-small",
        api_key_env_var: str = "OPENAI_API_KEY",
        top_k: int = 3,
        max_distance: float | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._embedding_model = embedding_model
        self._api_key_env_var = api_key_env_var
        self._top_k = top_k
        # Optional relevance cutoff: drop hits whose cosine distance exceeds this.
        # None keeps the raw top-k (Q&A always gets some context to work with).
        self._max_distance = max_distance
        self._by_id = products_by_id()
        self._collection = None

    def _get_collection(self):
        """Connect to the remote ChromaDB collection (lazy, cached).

        chromadb is imported here, not at module top level, so unit tests and the
        DI container can run without the chromadb package or a live server.
        """
        if self._collection is not None:
            return self._collection

        import chromadb
        from chromadb.utils import embedding_functions

        # The query is embedded client-side with the same model used for indexing,
        # so the server only ever stores and compares vectors (no API key needed there).
        # The EF reads the key from the env var instead of holding the secret directly.
        embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key_env_var=self._api_key_env_var,
            model_name=self._embedding_model,
        )
        client = chromadb.HttpClient(host=self._host, port=self._port)
        # Pass the EF explicitly: get_collection otherwise defaults to a local ONNX
        # model, which embeds queries with the wrong dimensions vs the index.
        self._collection = client.get_collection(
            name=self._collection_name,
            embedding_function=embedding_fn,
        )
        return self._collection

    def search(self, query: str) -> list[dict[str, object]]:
        collection = self._get_collection()
        result = collection.query(
            query_texts=[query],
            n_results=self._top_k,
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]

        products: list[dict[str, object]] = []
        for product_id, distance in zip(ids, distances, strict=True):
            if self._max_distance is not None and distance > self._max_distance:
                continue
            product = self._by_id.get(product_id)
            if product is not None:
                products.append(product)

        hits = ", ".join(f"{pid} (d={dist:.3f})" for pid, dist in zip(ids, distances, strict=True))
        logger.info("RETRIEVAL | query=%r | top_k=%d | %s", query, self._top_k, hits or "(none)")
        return products

    def get_all(self) -> list[dict[str, object]]:
        return list(load_products())
