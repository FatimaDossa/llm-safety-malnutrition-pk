import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-miniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_kb")
collection = client.get_collection("malnutrition_kb")


def retrieve(query, top_k=3):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    retrieved = []

    for doc, meta in zip(docs, metadatas):
        retrieved.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "topic": meta.get("topic", "unknown"),
            "is_emergency": meta.get("is_emergency", False)
        })

    return retrieved