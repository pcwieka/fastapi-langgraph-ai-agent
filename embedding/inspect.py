"""Quick inspector for the ChromaDB product collection.

Prints what is actually stored in the vector store: collections, row count, and
each product's id, metadata and the document text that was embedded. Read-only —
uses get(), which does not embed anything, so no OpenAI key is needed.

Two modes:
    documents   id + metadata + the embedded text (default)
    embeddings  the above plus a preview of each stored vector

Run (inside the app container via `make inspect-documents` / `make inspect-embeddings`),
or directly from the host (Chroma is mapped to localhost:8001 by docker-compose):
    python -m embedding.inspect --mode embeddings
"""

import argparse
import os

import chromadb

# How many vector components to show in embeddings mode (full vectors are 1536 floats).
_EMBEDDING_PREVIEW = 8


def inspect_collection(mode: str) -> None:
    host = os.environ.get("CHROMA_HOST", "localhost")
    # Host-side port from docker-compose (8001:8000). Inside the container env sets 8000.
    port = int(os.environ.get("CHROMA_PORT", "8001"))
    collection_name = os.environ.get("CHROMA_COLLECTION", "products")

    client = chromadb.HttpClient(host=host, port=port)
    print(f"Connected to ChromaDB at {host}:{port} | heartbeat={client.heartbeat()}")
    print(f"Collections: {[c.name for c in client.list_collections()]}")

    collection = client.get_collection(collection_name)
    print(f"\nCollection '{collection_name}' — {collection.count()} items (mode: {mode})\n")

    include = ["documents", "metadatas"]
    if mode == "embeddings":
        include.append("embeddings")

    data = collection.get(include=include)
    embeddings = data["embeddings"] if mode == "embeddings" else [None] * len(data["ids"])

    for product_id, document, metadata, embedding in zip(
        data["ids"], data["documents"], data["metadatas"], embeddings, strict=True
    ):
        print(f"- {product_id} | {metadata} | {document[:70]}")
        if embedding is not None:
            preview = ", ".join(f"{x:.4f}" for x in list(embedding)[:_EMBEDDING_PREVIEW])
            print(f"    vector[{len(embedding)}]: [{preview}, ...]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the ChromaDB product collection.")
    parser.add_argument(
        "--mode",
        choices=["documents", "embeddings"],
        default="documents",
        help="documents: text + metadata; embeddings: also show vector previews",
    )
    inspect_collection(parser.parse_args().mode)
