"""
Hybrid vector store: FAISS (dense) + BM25 (sparse) s RRF fúziou.
"""

from __future__ import annotations

import logging
import pickle
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Slovenské + anglické bežné stop words (light; BM25 bez stemmingu zvláda dobre)
_STOPWORDS = {
    "a", "aj", "ako", "ale", "alebo", "ani", "ano", "áno", "bez", "bol",
    "bola", "boli", "bolo", "by", "byť", "co", "čo", "či", "do", "ho", "i",
    "ich", "im", "je", "jej", "ju", "k", "ke", "keď", "ktorá", "ktoré",
    "ktorí", "ktorý", "kto", "mu", "my", "na", "nad", "nám", "nás", "nech",
    "nie", "no", "o", "od", "on", "ona", "ono", "pre", "po", "pod", "pri",
    "s", "sa", "si", "so", "som", "sme", "sú", "svoj", "ta", "tam",
    "tak", "taký", "tej", "ten", "to", "toto", "tu", "tvoj", "ty",
    "u", "v", "vás", "vo", "všetko", "vy", "za", "zo", "že",
    "the", "an", "and", "or", "of", "in", "on", "at", "to", "for",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "could", "may", "might",
    "this", "that", "these", "those", "it", "its", "as", "by", "with", "from",
}

# Parametre vyhľadávania
DEFAULT_TOPK_RETRIEVAL = 40     # celkový recall (Stage 1)
RRF_K = 60                      # RRF konštanta


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     E5 Embeddings wrapper                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class E5Embeddings(Embeddings):
    """
    Wrapper nad HuggingFaceEmbeddings, ktorý pridáva prefixy potrebné pre
    multilingual-e5-* modely: "passage: " k dokumentom, "query: " k otázkam.

    Auto-detect CUDA: ak je GPU dostupná, embedding beží na nej (kritické
    pre rýchlosť pri dlhých dokumentoch s tisíckami chunkov).
    """

    def __init__(self, model_name: str, device: Optional[str] = None,
                 batch_size: int = 32):
        if device is None:
            device = self._detect_device()
        logger.info(f"E5Embeddings: model={model_name}, device={device}, batch={batch_size}")
        self._device = device
        self._base = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": batch_size,
            },
        )

    @staticmethod
    def _detect_device() -> str:
        """CUDA ak dostupná, inak CPU. Kontroluje cez torch."""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                logger.info(f"Embedding GPU detected: {name}")
                return "cuda"
        except Exception as e:
            logger.info(f"torch.cuda check: {e}")
        return "cpu"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._base.embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(f"query: {text}")


# ─── Singleton embeddingov (drahé načítanie) ───────────────────────────────
_EMBEDDINGS: Optional[E5Embeddings] = None


def get_embeddings() -> E5Embeddings:
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        logger.info(f"Načítavam embedding model: {EMBEDDING_MODEL}")
        _EMBEDDINGS = E5Embeddings(EMBEDDING_MODEL)
    return _EMBEDDINGS


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                      NotebookVectorStore                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class NotebookVectorStore:
    """FAISS + BM25 per notebook. Drží kompletný zoznam child chunkov."""

    def __init__(self, nb_path: Path):
        self.nb_path    = nb_path
        self.vs_dir     = nb_path / "vector_store"
        self.vs_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.vs_dir / "faiss_index"
        self.docs_path  = self.vs_dir / "docs.pkl"

        self.embeddings                     = get_embeddings()
        self._faiss: Optional[FAISS]        = None
        self._all_docs: list[Document]      = []
        self._bm25: Optional[BM25Okapi]     = None
        self._load()

    # ─── Persist ─────────────────────────────────────────────────────────────
    def _load(self):
        if self.docs_path.exists():
            try:
                with open(self.docs_path, "rb") as f:
                    self._all_docs = pickle.load(f)
            except Exception as e:
                logger.error(f"Načítanie docs.pkl zlyhalo: {e}")
                self._all_docs = []

        if self.index_path.exists():
            try:
                self._faiss = FAISS.load_local(
                    str(self.index_path),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception as e:
                logger.error(f"Načítanie FAISS zlyhalo: {e}")
                self._faiss = None

        if self._all_docs:
            self._build_bm25()

    def _save(self):
        if self._faiss is not None:
            self._faiss.save_local(str(self.index_path))
        with open(self.docs_path, "wb") as f:
            pickle.dump(self._all_docs, f)

    # ─── Index builds ────────────────────────────────────────────────────────
    def _build_bm25(self):
        tokenized = [self._tokenize(d.page_content) for d in self._all_docs]
        if not any(tokenized):
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower()
        toks = re.findall(r"[a-zá-žä-ü0-9]+", text, flags=re.UNICODE)
        return [t for t in toks if t not in _STOPWORDS and len(t) > 1]

    # ─── Public API používaná app.py ─────────────────────────────────────────
    def has_documents(self) -> bool:
        return bool(self._all_docs)

    def add_documents(self, docs: list[Document]) -> None:
        if not docs:
            return
        texts = [d.page_content for d in docs]
        metas = [d.metadata for d in docs]

        if self._faiss is None:
            self._faiss = FAISS.from_texts(
                texts, embedding=self.embeddings, metadatas=metas,
            )
            self._all_docs = list(docs)
        else:
            self._faiss.add_texts(texts, metadatas=metas)
            self._all_docs.extend(docs)

        self._build_bm25()
        self._save()
        logger.info(
            f"Indexované +{len(docs)} chunkov (total: {len(self._all_docs)})."
        )

    def remove_source(self, source: str) -> None:
        remaining = [d for d in self._all_docs if d.metadata.get("source") != source]
        if len(remaining) == len(self._all_docs):
            return
        self.rebuild_index(remaining)

    def rebuild_index(self, docs: list[Document]) -> None:
        self._faiss = None
        self._all_docs = []
        self._bm25 = None

        if self.index_path.exists():
            shutil.rmtree(self.index_path, ignore_errors=True)
        if self.docs_path.exists():
            try:
                self.docs_path.unlink()
            except Exception:
                pass

        if docs:
            self.add_documents(docs)
        else:
            self._save()

    # ─── Searches ────────────────────────────────────────────────────────────
    def similarity_search(self, query: str, k: int = 30) -> list[tuple[Document, float]]:
        if self._faiss is None:
            return []
        results = self._faiss.similarity_search_with_score(query, k=k)
        return [(doc, float(-score)) for doc, score in results]

    def bm25_search(self, query: str, k: int = 30) -> list[tuple[Document, float]]:
        if self._bm25 is None or not self._all_docs:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            (self._all_docs[i], float(scores[i]))
            for i in top_idx
            if scores[i] > 0
        ]

    def hybrid_search(
        self,
        query: str,
        k: int = DEFAULT_TOPK_RETRIEVAL,
        rrf_k: int = RRF_K,
        vec_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> list[tuple[Document, float]]:
        wide = max(k * 2, 50)
        vec_results = self.similarity_search(query, k=wide)
        bm_results  = self.bm25_search(query, k=wide)

        fused: dict[str, tuple[Document, float]] = {}

        def _key(doc: Document) -> str:
            import hashlib
            h = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:12]
            src = doc.metadata.get("source", "")
            pg  = doc.metadata.get("page", 0)
            return f"{src}#{pg}#{h}"

        def _add(results: list[tuple[Document, float]], weight: float):
            for rank, (doc, _s) in enumerate(results):
                k_ = _key(doc)
                cur = fused.get(k_, (doc, 0.0))
                fused[k_] = (doc, cur[1] + weight / (rrf_k + rank + 1))

        _add(vec_results, vec_weight)
        _add(bm_results, bm25_weight)

        ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    @property
    def all_docs(self) -> list[Document]:
        return list(self._all_docs)

    def stats(self) -> dict:
        per_source: dict[str, int] = defaultdict(int)
        for d in self._all_docs:
            per_source[d.metadata.get("source", "?")] += 1
        return {
            "total_chunks": len(self._all_docs),
            "per_source":   dict(per_source),
            "has_faiss":    self._faiss is not None,
            "has_bm25":     self._bm25 is not None,
        }