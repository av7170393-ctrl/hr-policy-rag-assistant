"""Load policy files, chunk them with access-level metadata and build the vector index.

Run:  python -m app.rag.indexer
"""
import json
import re
from pathlib import Path

from app import config
from app.rag.textutils import split_text


def load_access() -> dict:
    if config.ACCESS_FILE.exists():
        return json.loads(config.ACCESS_FILE.read_text(encoding="utf-8"))
    return {}


def save_access(data: dict) -> None:
    config.ACCESS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _md_sections(text: str):
    for part in re.split(r"\n(?=## )", text):
        lines = part.strip().split("\n")
        first = lines[0]
        if first.startswith("## "):
            heading, body = first[3:].strip(), "\n".join(lines[1:])
        elif first.startswith("# "):
            heading, body = first[2:].strip(), "\n".join(lines[1:])
        else:
            heading, body = "Overview", part
        if body.strip():
            yield heading, body.strip()


def make_chunks(data_dir=None, access=None) -> list[dict]:
    data_dir = Path(data_dir or config.DATA_DIR)
    access = load_access() if access is None else access
    chunks = []
    for f in sorted(data_dir.iterdir()):
        ext = f.suffix.lower()
        min_role = access.get(f.name, "employee")
        level = config.ROLE_LEVEL.get(min_role, 1)
        pieces = []
        if ext in {".md", ".txt"}:
            pieces = [(1, h, b) for h, b in _md_sections(f.read_text(encoding="utf-8"))]
        elif ext == ".pdf":
            from pypdf import PdfReader
            for i, page in enumerate(PdfReader(str(f)).pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pieces.append((i, f"Page {i}", text))
        n = 0
        for page, heading, body in pieces:
            for part in split_text(body):
                chunks.append({"id": f"{f.name}::{n}", "text": part, "source": f.name, "page": page,
                               "heading": heading, "min_role": min_role, "level": level})
                n += 1
    return chunks


def build_index() -> dict:
    import chromadb
    from app.rag.models import get_embedder

    chunks = make_chunks()
    if not chunks:
        raise RuntimeError(f"No policy documents found in {config.DATA_DIR}")
    config.STORE_DIR.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_FILE.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection("hr_policies")
    except Exception:
        pass
    col = client.create_collection("hr_policies", metadata={"hnsw:space": "cosine"})
    texts = [f"{c['heading']}. {c['text']}" for c in chunks]
    vectors = get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
    for i in range(0, len(chunks), 100):
        batch = chunks[i:i + 100]
        col.add(ids=[c["id"] for c in batch], embeddings=vectors[i:i + 100],
                documents=[c["text"] for c in batch],
                metadatas=[{"source": c["source"], "page": c["page"], "heading": c["heading"],
                            "level": c["level"]} for c in batch])
    return {"documents": len({c["source"] for c in chunks}), "chunks": len(chunks)}


if __name__ == "__main__":
    print("Indexed:", build_index())
