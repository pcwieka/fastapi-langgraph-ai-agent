"""Offline embedding indexer — builds the ChromaDB product collection.

Run once (and again whenever the catalog changes) with `make index`. It reads the
canonical catalog, embeds each product with OpenAI, and upserts the vectors into a
remote ChromaDB collection. The running app never builds the index itself — it only
queries the collection at request time.

This separation mirrors a real RAG pipeline: indexing is a batch job, retrieval is
online. Keeping it out of the request path means startup stays fast and embeddings
are computed exactly once.

Usage:
    python -m embedding.index
"""

import os
import sys

import chromadb
from chromadb.utils import embedding_functions

from app.product.catalog import load_products


def product_to_text(product: dict) -> str:
    """Flatten a product into the text that gets embedded.

    Only the semantically meaningful fields are included — name, category, brand
    and the free-text fields. Price and stock are noise for similarity search.
    """
    fields = ("name", "category", "brand", "specs", "description")
    return " | ".join(str(product.get(field, "")) for field in fields)


def build_index() -> None:
    host = os.environ.get("CHROMA_HOST", "localhost")
    port = int(os.environ.get("CHROMA_PORT", "8000"))
    collection_name = os.environ.get("CHROMA_COLLECTION", "products")
    embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY is not set — cannot create embeddings.")

    products = load_products()
    print(f"Loaded {len(products)} products from the catalog.")

    # Point the EF at the env var holding the key instead of passing the secret
    # directly. Embedding runs client-side (here).
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key_env_var="OPENAI_API_KEY",
        model_name=embedding_model,
    )

    client = chromadb.HttpClient(host=host, port=port)
    print(f"Connected to ChromaDB at {host}:{port}.")

    # Recreate the collection from scratch so re-indexing is idempotent.
    try:
        client.delete_collection(collection_name)
        print(f"Dropped existing collection '{collection_name}'.")
    except Exception:
        # Collection did not exist yet — nothing to drop on a first run.
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        # Cosine distance is the standard metric for normalized text embeddings.
        configuration={"hnsw": {"space": "cosine"}},
    )

    # Embedding happens here: passing documents lets the embedding function turn
    # each product text into a vector before storing it in the collection.
    collection.add(
        ids=[p["id"] for p in products],
        documents=[product_to_text(p) for p in products],
        metadatas=[{"name": p["name"], "category": p["category"]} for p in products],
    )

    print(f"Indexed {collection.count()} products into '{collection_name}' (model: {embedding_model}).")


if __name__ == "__main__":
    build_index()
