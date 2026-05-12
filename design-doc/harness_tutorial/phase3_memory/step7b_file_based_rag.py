"""
PHASE 3 — STEP 7b: File-Based RAG — Zero Database Required
===========================================================
Harness Pillar: Context Management (Memory without Infrastructure)

Goal:
  Implement RAG using ONLY flat files on disk.
  Four approaches, from zero-dependency to production-grade.
  No Postgres, no Redis, no SQLite, no server.

Why decouple from databases?
  ✓ Zero infra — no server to provision, no migrations to run
  ✓ Git-friendly — markdown files are human-readable, diffable, versionable
  ✓ Portable — zip a folder, move agent memory anywhere
  ✓ Auditable — open any memory entry with a text editor

Memory Format (shared across all options):
  memory/
  ├── episodic/          ← event-based memories (what happened)
  │   └── 2026-05-10_signal_analysis.md
  ├── semantic/          ← domain facts and knowledge
  │   └── sm2040_spec_notes.md
  └── procedural/        ← how-to workflows
      └── analysis_workflow.md

  Each .md file has YAML frontmatter:
  ---
  title: SM-2040 ADC Spec Notes
  tags: [hardware, adc, spec, voltage]
  date: 2026-05-10
  importance: high
  ---
  The SM-2040 ADC board operates at 3.3V reference...

Four Retrieval Approaches:
  A) BM25 + Markdown         → no embeddings, rank_bm25 only
  B) Numpy Cosine + .npy     → sentence-transformers, pure numpy cosine sim
  C) FAISS + Markdown        → faiss-cpu, single .index binary file
  D) LanceDB                 → Apache Lance columnar files, native vector search

Install per approach:
  pip install rank-bm25                        # A only
  pip install sentence-transformers            # B + C + D
  pip install faiss-cpu                        # C only
  pip install lancedb                          # D only
  pip install -r requirements.txt              # everything
"""

import json
import os
import re
import math
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


# ═══════════════════════════════════════════════════════════════════════════
# SHARED: Markdown Memory Format
#   All four approaches read from the same directory of .md files.
#   The markdown files ARE the database — no separate storage layer.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryChunk:
    """A single retrievable unit of memory."""
    chunk_id: str
    file_path: str
    title: str
    tags: list[str]
    date: str
    importance: str          # high | medium | low
    content: str             # the actual text chunk
    char_start: int = 0      # position in original document


def parse_markdown_file(path: Path) -> list[MemoryChunk]:
    """
    Parse a .md file with YAML frontmatter into MemoryChunks.

    Frontmatter format:
    ---
    title: My Note
    tags: [physics, signal]
    date: 2026-05-10
    importance: high
    ---
    Content here...
    """
    raw = path.read_text(encoding="utf-8")

    # ── Parse YAML frontmatter ──
    metadata: dict = {}
    content_start = 0

    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            fm_text = raw[3:end].strip()
            content_start = end + 4
            # Minimal YAML parser (avoids pyyaml dep for portability)
            for line in fm_text.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    v = v.strip()
                    # Parse list: [a, b, c]
                    if v.startswith("[") and v.endswith("]"):
                        v = [x.strip().strip('"\'') for x in v[1:-1].split(",")]
                    metadata[k.strip()] = v

    content = raw[content_start:].strip()

    title = str(metadata.get("title", path.stem))
    tags = metadata.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    date = str(metadata.get("date", ""))
    importance = str(metadata.get("importance", "medium"))

    # ── Chunk content (512-char chunks, 64-char overlap) ──
    chunks = []
    chunk_size = 512
    overlap = 64
    pos = 0
    idx = 0

    while pos < len(content):
        end = min(pos + chunk_size, len(content))
        # Try to break at paragraph boundary
        if end < len(content):
            newline = content.rfind("\n\n", pos, end)
            if newline != -1 and newline > pos + overlap:
                end = newline

        chunk_text = content[pos:end].strip()
        if chunk_text:
            chunks.append(MemoryChunk(
                chunk_id=f"{path.stem}_chunk{idx}",
                file_path=str(path),
                title=title,
                tags=tags,
                date=date,
                importance=importance,
                content=chunk_text,
                char_start=pos,
            ))
            idx += 1

        pos = end - overlap if end < len(content) else end

    # If file is short (< chunk_size), return as one chunk
    if not chunks and content:
        chunks.append(MemoryChunk(
            chunk_id=f"{path.stem}_chunk0",
            file_path=str(path),
            title=title, tags=tags, date=date, importance=importance,
            content=content, char_start=0,
        ))

    return chunks


def load_memory_dir(memory_dir: str | Path) -> list[MemoryChunk]:
    """Load all .md files from a directory tree into MemoryChunks."""
    memory_dir = Path(memory_dir)
    all_chunks: list[MemoryChunk] = []
    for md_file in sorted(memory_dir.rglob("*.md")):
        try:
            all_chunks.extend(parse_markdown_file(md_file))
        except Exception as e:
            print(f"  ⚠ Skipping {md_file.name}: {e}")
    return all_chunks


def save_memory(memory_dir: Path, subfolder: str, filename: str, title: str,
                tags: list[str], content: str, importance: str = "medium") -> Path:
    """Write a new memory as a markdown file."""
    from datetime import date
    folder = memory_dir / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    today = date.today().isoformat()
    tags_str = "[" + ", ".join(tags) + "]"
    frontmatter = f"---\ntitle: {title}\ntags: {tags_str}\ndate: {today}\nimportance: {importance}\n---\n\n"
    path.write_text(frontmatter + content, encoding="utf-8")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# OPTION A: BM25 + Markdown
# ═══════════════════════════════════════════════════════════════════════════
# ─ What it is: BM25 (Best Match 25) is a ranking function used by search
#   engines like Elasticsearch. Better than TF-IDF because it accounts for
#   document length. No embeddings, no model, no GPU.
#
# ─ When to use: domain-specific terminology (spec doc keywords, part names,
#   error codes) where exact term matching beats semantic similarity.
#
# ─ Deps: pip install rank-bm25
# ─ Files: only the .md files themselves. Nothing extra saved to disk.
# ═══════════════════════════════════════════════════════════════════════════

class BM25Retriever:
    """
    BM25 retrieval over a directory of Markdown files.
    Index is built in-memory at startup — no separate index file.
    """

    def __init__(self, memory_dir: str | Path):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("pip install rank-bm25")

        self.chunks = load_memory_dir(memory_dir)
        if not self.chunks:
            raise ValueError(f"No .md files found in {memory_dir}")

        # Tokenise: lowercase, split on non-alphanumeric
        def tokenise(text: str) -> list[str]:
            return re.findall(r"[a-z0-9]+", text.lower())

        corpus = [tokenise(c.content + " " + c.title + " " + " ".join(c.tags))
                  for c in self.chunks]

        self.bm25 = BM25Okapi(corpus)
        self._tokenise = tokenise
        print(f"✓ BM25 index built — {len(self.chunks)} chunks from {memory_dir}")

    def retrieve(self, query: str, k: int = 3) -> list[MemoryChunk]:
        tokens = self._tokenise(query)
        scores = self.bm25.get_scores(tokens)
        # Boost high-importance chunks
        for i, chunk in enumerate(self.chunks):
            if chunk.importance == "high":
                scores[i] *= 1.3
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in top_k if scores[i] > 0]


# ═══════════════════════════════════════════════════════════════════════════
# OPTION B: Numpy Cosine + .emb.npy Files
# ═══════════════════════════════════════════════════════════════════════════
# ─ What it is: semantic similarity via dense vector embeddings.
#   Each chunk is embedded once and saved as a .npy file next to its .md.
#   Retrieval = cosine(query_embedding, chunk_embeddings).
#
# ─ When to use: semantic queries ("what were the conclusions?") where
#   exact keywords may not appear in the stored memory.
#
# ─ Persistence: embeddings cached as  memory/semantic/doc.emb.npy
#   If the .md file changes, delete the .npy to trigger re-embedding.
#
# ─ Deps: pip install sentence-transformers numpy
# ═══════════════════════════════════════════════════════════════════════════

class NumpyCosineRetriever:
    """
    Semantic retrieval using numpy cosine similarity.
    Embeddings persisted as .npy files — no vector database.
    """

    def __init__(self, memory_dir: str | Path, model_name: str = "all-MiniLM-L6-v2"):
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("pip install sentence-transformers numpy")

        self.np = np
        self.memory_dir = Path(memory_dir)
        self.chunks = load_memory_dir(memory_dir)

        print(f"→ Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        self.embeddings = self._load_or_build_embeddings()
        print(f"✓ Numpy cosine index ready — {len(self.chunks)} chunks")

    def _embed_texts(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def _load_or_build_embeddings(self):
        np = self.np
        cache_file = self.memory_dir / ".embeddings_cache.npy"
        meta_file = self.memory_dir / ".embeddings_meta.json"

        chunk_ids = [c.chunk_id for c in self.chunks]

        # ── Check if cache is valid ──
        if cache_file.exists() and meta_file.exists():
            cached_ids = json.loads(meta_file.read_text())
            if cached_ids == chunk_ids:
                print("  ↳ Loading embeddings from cache")
                return np.load(cache_file)

        # ── Build embeddings ──
        print(f"  ↳ Embedding {len(self.chunks)} chunks (first run)...")
        texts = [c.content + " [SEP] " + c.title for c in self.chunks]
        embeddings = self._embed_texts(texts)

        # ── Save to disk ──
        np.save(cache_file, embeddings)
        meta_file.write_text(json.dumps(chunk_ids))
        print(f"  ↳ Cached to {cache_file}")
        return embeddings

    def retrieve(self, query: str, k: int = 3) -> list[MemoryChunk]:
        np = self.np
        q_emb = self._embed_texts([query])[0]                  # shape: (d,)
        scores = self.embeddings @ q_emb                       # cosine (normalised)
        top_k = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top_k]

    def invalidate_cache(self):
        """Call after adding/editing .md files."""
        for f in [self.memory_dir / ".embeddings_cache.npy",
                  self.memory_dir / ".embeddings_meta.json"]:
            if f.exists():
                f.unlink()


# ═══════════════════════════════════════════════════════════════════════════
# OPTION C: FAISS + Markdown
# ═══════════════════════════════════════════════════════════════════════════
# ─ What it is: Facebook's FAISS (Approximate Nearest Neighbor) as a
#   single binary .index file. Scales to millions of chunks while still
#   fitting on a laptop, no server required.
#
# ─ When to use: larger memory stores (>10k chunks) where numpy cosine
#   becomes slow. FAISS is 100-1000× faster for large indexes.
#
# ─ Persistence:
#   memory/.faiss_index        ← vector index binary
#   memory/.faiss_meta.json    ← chunk_id → chunk metadata mapping
#
# ─ Deps: pip install faiss-cpu sentence-transformers numpy
# ═══════════════════════════════════════════════════════════════════════════

class FAISSRetriever:
    """
    FAISS approximate-nearest-neighbour retrieval.
    Index saved to a single .index binary file.
    """

    def __init__(self, memory_dir: str | Path, model_name: str = "all-MiniLM-L6-v2"):
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("pip install faiss-cpu sentence-transformers numpy")

        self.faiss = faiss
        self.np = np
        self.memory_dir = Path(memory_dir)
        self.chunks = load_memory_dir(memory_dir)

        self.model = SentenceTransformer(model_name)
        self.index, self.chunk_ids = self._load_or_build_index()
        print(f"✓ FAISS index ready — {self.index.ntotal} vectors")

    def _embed(self, texts: list[str]):
        embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return self.np.array(embs, dtype="float32")

    def _load_or_build_index(self):
        faiss = self.faiss
        index_file = self.memory_dir / ".faiss_index"
        meta_file = self.memory_dir / ".faiss_meta.json"

        chunk_ids = [c.chunk_id for c in self.chunks]

        if index_file.exists() and meta_file.exists():
            cached_ids = json.loads(meta_file.read_text())
            if cached_ids == chunk_ids:
                print("  ↳ Loading FAISS index from disk")
                return faiss.read_index(str(index_file)), chunk_ids

        print(f"  ↳ Building FAISS index for {len(self.chunks)} chunks...")
        texts = [c.content + " [SEP] " + c.title for c in self.chunks]
        embeddings = self._embed(texts)

        dim = embeddings.shape[1]
        # IndexFlatIP = inner product (= cosine sim when vectors are normalised)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        faiss.write_index(index, str(index_file))
        meta_file.write_text(json.dumps(chunk_ids))
        print(f"  ↳ FAISS index saved to {index_file}")
        return index, chunk_ids

    def retrieve(self, query: str, k: int = 3) -> list[MemoryChunk]:
        q_emb = self._embed([query])
        scores, indices = self.index.search(q_emb, k)

        id_to_chunk = {c.chunk_id: c for c in self.chunks}
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunk_ids):
                cid = self.chunk_ids[idx]
                if cid in id_to_chunk:
                    results.append(id_to_chunk[cid])
        return results

    def invalidate_index(self):
        """Delete cached index files to force rebuild."""
        for f in [self.memory_dir / ".faiss_index",
                  self.memory_dir / ".faiss_meta.json"]:
            if f.exists():
                f.unlink()


# ═══════════════════════════════════════════════════════════════════════════
# OPTION D: LanceDB (Apache Lance Columnar Files)
# ═══════════════════════════════════════════════════════════════════════════
# ─ What it is: LanceDB is an embedded vector database that stores data
#   as Apache Lance columnar files. No server, no SQL, just a folder.
#
# ─ Hybrid search: vector similarity AND metadata filters in one query.
#   e.g. "find similar chunks WHERE importance = 'high' AND 'signal' in tags"
#
# ─ Persistence:
#   memory/.lancedb/          ← Apache Lance files (columnar format)
#   └── chunks.lance/
#
# ─ Why it beats FAISS for agents: metadata filtering without post-filtering.
#   FAISS retrieves top-k by vector, then you filter. LanceDB filters first.
#
# ─ Deps: pip install lancedb sentence-transformers numpy
# ═══════════════════════════════════════════════════════════════════════════

class LanceDBRetriever:
    """
    LanceDB retrieval with hybrid vector + metadata filtering.
    Stored as Apache Lance files — no database server.
    """

    def __init__(self, memory_dir: str | Path, model_name: str = "all-MiniLM-L6-v2"):
        try:
            import lancedb
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("pip install lancedb sentence-transformers numpy")

        self.lancedb = lancedb
        self.np = np
        self.memory_dir = Path(memory_dir)
        self.chunks = load_memory_dir(memory_dir)
        self.model = SentenceTransformer(model_name)

        db_path = self.memory_dir / ".lancedb"
        self.db = lancedb.connect(str(db_path))
        self.table = self._load_or_build_table()
        print(f"✓ LanceDB ready — {self.table.count_rows()} vectors at {db_path}")

    def _embed(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True,
                                 show_progress_bar=False).tolist()

    def _load_or_build_table(self):
        chunk_ids = [c.chunk_id for c in self.chunks]
        table_name = "chunks"

        # ── Check if table is up-to-date ──
        if table_name in self.db.table_names():
            table = self.db.open_table(table_name)
            stored_ids = [r["chunk_id"] for r in table.to_arrow().to_pydict()["chunk_id"]]
            if stored_ids == chunk_ids:
                print("  ↳ Loading LanceDB table from disk")
                return table
            # Stale — drop and rebuild
            self.db.drop_table(table_name)

        print(f"  ↳ Building LanceDB table for {len(self.chunks)} chunks...")
        texts = [c.content + " [SEP] " + c.title for c in self.chunks]
        embeddings = self._embed(texts)

        rows = [
            {
                "chunk_id": c.chunk_id,
                "title": c.title,
                "tags": ", ".join(c.tags),
                "importance": c.importance,
                "date": c.date,
                "content": c.content,
                "file_path": c.file_path,
                "vector": emb,
            }
            for c, emb in zip(self.chunks, embeddings)
        ]

        table = self.db.create_table(table_name, data=rows)
        print(f"  ↳ LanceDB table saved to {self.memory_dir / '.lancedb'}")
        return table

    def retrieve(self, query: str, k: int = 3,
                 importance_filter: Optional[str] = None,
                 tag_filter: Optional[str] = None) -> list[MemoryChunk]:
        """
        Hybrid search: vector similarity + optional metadata filters.

        Args:
            query: natural-language query
            k: number of results
            importance_filter: "high" | "medium" | "low"
            tag_filter: tag string that must appear in the tags field
        """
        q_emb = self._embed([query])[0]
        search = self.table.search(q_emb).limit(k)

        # ── Apply metadata filters (pre-filter, not post-filter) ──
        where_clauses = []
        if importance_filter:
            where_clauses.append(f"importance = '{importance_filter}'")
        if tag_filter:
            where_clauses.append(f"tags LIKE '%{tag_filter}%'")
        if where_clauses:
            search = search.where(" AND ".join(where_clauses))

        results = search.to_list()
        id_to_chunk = {c.chunk_id: c for c in self.chunks}
        return [id_to_chunk[r["chunk_id"]] for r in results if r["chunk_id"] in id_to_chunk]

    def add_memory(self, chunk: MemoryChunk):
        """Incrementally add one chunk without full rebuild."""
        emb = self._embed([chunk.content + " [SEP] " + chunk.title])[0]
        self.table.add([{
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "tags": ", ".join(chunk.tags),
            "importance": chunk.importance,
            "date": chunk.date,
            "content": chunk.content,
            "file_path": chunk.file_path,
            "vector": emb,
        }])


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED INTERFACE
# Swap retriever backend with one parameter — no agent code changes needed.
# ═══════════════════════════════════════════════════════════════════════════

class FileBasedMemory:
    """
    Unified memory interface wrapping any retriever.
    Format context for injection into Claude's system prompt.
    """

    BACKENDS = {
        "bm25":    BM25Retriever,
        "numpy":   NumpyCosineRetriever,
        "faiss":   FAISSRetriever,
        "lancedb": LanceDBRetriever,
    }

    def __init__(self, memory_dir: str | Path, backend: str = "bm25", **kwargs):
        """
        Args:
            memory_dir: path to directory of .md files
            backend: "bm25" | "numpy" | "faiss" | "lancedb"
            **kwargs: passed to the retriever (e.g., importance_filter for lancedb)
        """
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of: {list(self.BACKENDS)}")
        cls = self.BACKENDS[backend]
        self.retriever = cls(memory_dir, **kwargs)
        self.backend = backend
        self.memory_dir = Path(memory_dir)

    def retrieve(self, query: str, k: int = 3, **kwargs) -> list[MemoryChunk]:
        return self.retriever.retrieve(query, k=k, **kwargs)

    def format_for_context(self, query: str, k: int = 3, **kwargs) -> str:
        """Return a formatted string ready to inject into a system prompt."""
        chunks = self.retrieve(query, k=k, **kwargs)
        if not chunks:
            return "(no relevant memory found)"

        lines = ["<memory>"]
        for i, chunk in enumerate(chunks, 1):
            tags = ", ".join(chunk.tags) if chunk.tags else "none"
            lines.append(f"  <item index=\"{i}\" title=\"{chunk.title}\" "
                         f"tags=\"{tags}\" importance=\"{chunk.importance}\" "
                         f"date=\"{chunk.date}\">")
            lines.append(f"    {chunk.content.strip()}")
            lines.append(f"  </item>")
        lines.append("</memory>")
        return "\n".join(lines)

    def save_new_memory(self, subfolder: str, filename: str,
                        title: str, tags: list[str], content: str,
                        importance: str = "medium") -> Path:
        """Write a new .md memory file and invalidate index if needed."""
        path = save_memory(self.memory_dir, subfolder, filename,
                           title, tags, content, importance)
        # Invalidate caches for numpy/faiss backends
        if hasattr(self.retriever, "invalidate_cache"):
            self.retriever.invalidate_cache()
        if hasattr(self.retriever, "invalidate_index"):
            self.retriever.invalidate_index()
        return path


# ═══════════════════════════════════════════════════════════════════════════
# COMPARISON DEMO
# Run all four backends against the same query, compare results.
# ═══════════════════════════════════════════════════════════════════════════

def create_sample_memory_dir(base: Path) -> Path:
    """Create a small sample memory directory for the demo."""
    memory_dir = base / "memory"

    # ── Semantic knowledge ──
    save_memory(memory_dir, "semantic", "sm2040_spec.md",
                title="SM-2040 ADC Specification",
                tags=["hardware", "adc", "spec", "voltage", "frequency"],
                importance="high",
                content="""
The SM-2040 is a 24-bit differential ADC board designed for high-precision signal acquisition.

Operating parameters:
- Input voltage range: ±10V differential
- Sampling rate: 1 kHz to 100 kHz (configurable)
- Reference voltage: 3.3V internal, ±0.1% accuracy
- SNR: 110 dB typical at 1 kHz
- THD: -120 dBc at full scale

Valid analysis window for frequency domain:
- Nyquist limit: fs / 2 (do not interpret components above this)
- Fundamental frequency must be ≥ 1 Hz and ≤ 40 kHz
- Harmonics beyond 5th order are likely noise floor artefacts

Physical constraints for data quality:
- Reject samples where |V| > 9.9V (clipping)
- Reject samples where |V| < 0.001V (noise floor)
- Valid SNR range for trustworthy FFT: 60 dB – 130 dB
""")

    save_memory(memory_dir, "semantic", "signal_processing.md",
                title="Signal Processing Fundamentals",
                tags=["dsp", "fft", "windowing", "sampling"],
                importance="medium",
                content="""
Key DSP concepts for data analyst agents:

FFT Analysis:
- Always apply a window function (Hann, Blackman) before FFT to reduce spectral leakage
- Use zero-padding to interpolate the spectrum (does not add frequency resolution)
- Frequency resolution = fs / N where fs = sample rate, N = number of samples

Noise handling:
- Gaussian noise: mean=0, use RMS as noise floor estimate
- Impulsive noise: median filter > mean filter
- 1/f (pink) noise: use PSD (power spectral density) in log-log plot

THD (Total Harmonic Distortion) calculation:
  THD = sqrt(V2² + V3² + V4² + ...) / V1
where V1 = fundamental amplitude, V2..Vn = harmonic amplitudes.
""")

    # ── Episodic memories ──
    save_memory(memory_dir, "episodic", "2026-05-08_analysis.md",
                title="May 8 Signal Analysis Session",
                tags=["analysis", "sm2040", "results", "clipping"],
                importance="high",
                content="""
Analysis session on SM-2040 dataset (2000 samples at 10 kHz):

Findings:
- 47 samples (2.3%) showed clipping at ±9.9V — excluded from FFT analysis
- Fundamental frequency identified at 1000 Hz with amplitude 7.2V peak
- 2nd harmonic at 2000 Hz: -42 dBc (within spec, < -40 dBc limit)
- 3rd harmonic at 3000 Hz: -65 dBc (well within spec)
- SNR measured: 88.5 dB (good, above 60 dB threshold)
- THD computed: 0.79% (acceptable)

Recommendation:
- Increase sampling duration (currently 200ms) for better frequency resolution
- Apply Hann window before FFT to reduce spectral leakage
- Investigation of clipping source needed — may be upstream amplifier saturation
""")

    # ── Procedural workflows ──
    save_memory(memory_dir, "procedural", "analysis_workflow.md",
                title="Standard Data Analysis Workflow",
                tags=["workflow", "procedure", "analysis", "steps"],
                importance="medium",
                content="""
Standard workflow for SM-2040 data analysis:

Step 1: Data validation
  - Load CSV, check column names (timestamp_s, voltage_V, channel)
  - Reject samples outside ±9.9V (clipping per spec)
  - Reject samples with |V| < 0.001V (noise floor)
  - Log rejection rate — if > 5%, flag for manual review

Step 2: Time-domain analysis
  - Plot voltage vs time (first 100ms for visual inspection)
  - Compute RMS, peak, crest factor
  - Check for stationarity (rolling mean should be stable)

Step 3: Frequency-domain analysis
  - Apply Hann window to each segment
  - Compute FFT (numpy.fft.rfft)
  - Identify fundamental and harmonics
  - Compute THD if fundamental > -60 dBFS

Step 4: Report generation
  - Summary statistics table
  - Time-domain plot (raw + filtered)
  - FFT spectrum plot (linear scale → dB scale)
  - Flag any spec violations in red
""")

    print(f"✓ Sample memory created at {memory_dir}")
    return memory_dir


# ═══════════════════════════════════════════════════════════════════════════
# CLAUDE INTEGRATION
# The agent uses memory.format_for_context() to inject relevant knowledge
# into each turn — RAG without a database.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """\
You are a domain-aware data analyst specialising in hardware signal acquisition.

You have access to a structured memory bank (Markdown files on disk).
Relevant knowledge retrieved for the current query is injected below.
Use it to constrain your analysis to physically meaningful ranges.

{memory_context}

When answering:
1. Reference spec limits from memory before interpreting data values.
2. Flag any values that violate physical constraints.
3. Recommend the next analysis step based on the procedural workflow in memory.
"""


def run_claude_with_memory(query: str, memory: FileBasedMemory) -> str:
    """Single-turn Claude call augmented with file-based RAG memory."""
    memory_context = memory.format_for_context(query, k=3)

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT_TEMPLATE.format(memory_context=memory_context),
        messages=[{"role": "user", "content": query}],
    )
    return response.content[0].text


def compare_backends(memory_dir: Path, query: str):
    """
    Show the difference in retrieved chunks across all four backends
    for the same query.
    """
    print(f"\n{'═'*65}")
    print(f"QUERY: {query}")
    print(f"{'═'*65}")

    backends_to_test: list[tuple[str, dict]] = [
        ("bm25", {}),
    ]

    # Try to add semantic backends if deps are available
    for backend, extra in [("numpy", {}), ("faiss", {}), ("lancedb", {})]:
        try:
            FileBasedMemory(memory_dir, backend=backend)
            backends_to_test.append((backend, extra))
        except ImportError as e:
            print(f"⚠ Skipping {backend}: {e}")

    for backend, kwargs in backends_to_test:
        print(f"\n── Option {['A','B','C','D'][['bm25','numpy','faiss','lancedb'].index(backend)]}"
              f": {backend.upper()} ──")
        mem = FileBasedMemory(memory_dir, backend=backend)
        chunks = mem.retrieve(query, k=2, **kwargs)
        for i, c in enumerate(chunks, 1):
            preview = c.content[:120].replace("\n", " ").strip()
            print(f"  [{i}] {c.title!r}  (importance={c.importance})")
            print(f"      {preview}...")

    # Run Claude with whichever backend is available
    primary_backend = backends_to_test[0][0]
    mem = FileBasedMemory(memory_dir, backend=primary_backend)
    print(f"\n── Claude response (using {primary_backend}) ──")
    answer = run_claude_with_memory(query, mem)
    print(answer)


# ═══════════════════════════════════════════════════════════════════════════
# BACKEND COMPARISON CHEAT SHEET
# ═══════════════════════════════════════════════════════════════════════════

COMPARISON_TABLE = """
╔══════════════════╦══════════╦═════════════════╦══════════════╦══════════════╗
║ Feature          ║ BM25 (A) ║ Numpy Cosine (B)║  FAISS (C)   ║ LanceDB (D)  ║
╠══════════════════╬══════════╬═════════════════╬══════════════╬══════════════╣
║ Semantic search  ║    ✗     ║       ✓         ║      ✓       ║      ✓       ║
║ Exact keywords   ║    ✓✓    ║       ✓         ║      ✓       ║      ✓       ║
║ Metadata filter  ║  manual  ║    manual        ║   manual     ║  ✓ native    ║
║ Scale (chunks)   ║  <50k    ║    <10k          ║   millions   ║   millions   ║
║ GPU needed       ║    ✗     ║       ✗          ║   optional   ║   optional   ║
║ Extra files      ║  none    ║  .npy + .json    ║ .faiss+.json ║  .lancedb/   ║
║ Incremental add  ║  rebuild ║    rebuild       ║   rebuild    ║  ✓ native    ║
║ Git-trackable    ║    ✓✓    ║    partial       ║      ✗       ║      ✗       ║
║ Dependencies     ║ rank_bm25║  ST + numpy      ║ ST+numpy+    ║ lancedb+ST   ║
║                  ║          ║                 ║  faiss-cpu   ║              ║
╠══════════════════╬══════════╬═════════════════╬══════════════╬══════════════╣
║ Best for         ║ domain   ║ semantic recall  ║ large scale  ║ hybrid query ║
║                  ║ keywords,║ general queries  ║ >10k chunks  ║ + metadata   ║
║                  ║ spec docs║                 ║              ║ filtering    ║
╚══════════════════╩══════════╩═════════════════╩══════════════╩══════════════╝

Recommendation for data analyst + spec docs:
  • Small team / fast iteration → BM25 (A): spec terms are exact keywords
  • Semantic user queries       → Numpy (B): <10k chunks, no infra
  • Large corpus (reports, logs)→ FAISS (C): fast at scale
  • Need date/importance filter → LanceDB (D): hybrid search is a killer feature
"""


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import tempfile

    print(__doc__)
    print(COMPARISON_TABLE)

    # ── Create a temporary memory directory with sample .md files ──
    with tempfile.TemporaryDirectory(prefix="harness_memory_") as tmp:
        memory_dir = create_sample_memory_dir(Path(tmp))

        # ── Test queries ──
        queries = [
            "What is the valid voltage range and how should I handle clipping?",
            "What happened in the last analysis session and what were the THD findings?",
            "How do I apply a window function before FFT?",
        ]

        for query in queries:
            compare_backends(memory_dir, query)
            print()

    # ── Persistence demo ──
    print("\n── Persistence Demo: BM25 ──")
    print("BM25 rebuilds index from .md files each startup.")
    print("No extra files are written. Memory = the .md files themselves.")
    print()
    print("── Persistence Demo: Numpy / FAISS ──")
    print("Embeddings cached to:  memory/.embeddings_cache.npy")
    print("Chunk metadata:        memory/.embeddings_meta.json")
    print("Delete these to force re-embedding after editing .md files.")
    print()
    print("── Persistence Demo: LanceDB ──")
    print("Lance columnar files:  memory/.lancedb/chunks.lance/")
    print("Supports incremental add via table.add([...]) — no full rebuild.")
    print()
    print("Key takeaway: ALL four backends use the same .md files as the")
    print("source of truth. The vector index is a derived, disposable cache.")
    print("Delete the index, rebuild from .md — memory is never lost.")
