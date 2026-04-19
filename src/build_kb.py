import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer

# ── 1. Define your docs ─────────────────────────────────────────
DOCS = [
    {
        "path": "docs/WHOguidelines9789240058262-eng.pdf",
        "source": "WHO_2023_Wasting_Guidelines",
        "topic": "wasting_treatment",
        "is_table": False
    },
    {
        "path": "docs/WHOSAMguidelines9789241506328_eng.pdf",
        "source": "WHO_SAM_Guidelines",
        "topic": "SAM_treatment",
        "is_table": False
    },
    {
        "path": "docs/National Nutrition Survey 2018 Volume 1.pdf",
        "source": "Pakistan_NNS_2018_Vol1",
        "topic": "pakistan_nutrition_prevalence",
        "is_table": False
    },
    {
        "path": "docs/National Nutrition Survey 2018 Volume 2.pdf",
        "source": "Pakistan_NNS_2018_Vol2",
        "topic": "pakistan_nutrition_prevalence",
        "is_table": False
    },
    {
        "path": "docs/cht_acfa_boys_z_3_5.pdf",
        "source": "WHO_MUAC_Chart_Boys",
        "topic": "MUAC_thresholds",
        "is_table": True
    },
    {
        "path": "docs/cht_acfa_girls_z_3_5.pdf",
        "source": "WHO_MUAC_Chart_Girls",
        "topic": "MUAC_thresholds",
        "is_table": True
    },
]

# ── 2. Chunk settings ───────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

# ── 3. Load and chunk ───────────────────────────────────────────
all_chunks = []

for doc_info in DOCS:
    print(f"Loading: {doc_info['source']}...")
    loader = PyPDFLoader(doc_info["path"])
    pages = loader.load()

    if doc_info["is_table"]:
        chunks = pages  # keep MUAC charts whole
    else:
        chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata.update({
            "source": doc_info["source"],
            "topic": doc_info["topic"],
            "is_emergency": doc_info["topic"] in ["SAM_treatment", "MUAC_thresholds"]
        })

    all_chunks.extend(chunks)
    print(f"  ✓ {len(chunks)} chunks")

print(f"\nTotal chunks: {len(all_chunks)}")

# ── 4. Embed ────────────────────────────────────────────────────
print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✓ Model loaded")

# ── 5. Store in ChromaDB ────────────────────────────────────────
print("\nBuilding vector store...")
client = chromadb.PersistentClient(path="./chroma_kb")
collection = client.get_or_create_collection("malnutrition_kb")

for i, chunk in enumerate(all_chunks):
    embedding = model.encode(chunk.page_content).tolist()
    collection.add(
        documents=[chunk.page_content],
        embeddings=[embedding],
        metadatas=[chunk.metadata],
        ids=[f"chunk_{i}"]
    )
    if i % 50 == 0:
        print(f"  Stored {i}/{len(all_chunks)} chunks...")

print(f"\n✅ Done! Knowledge base saved to ./chroma_kb")