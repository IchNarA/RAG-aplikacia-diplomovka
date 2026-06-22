"""
ParentStore –  úložisko parent chunkov pre hierarchický RAG.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                           ParentStore                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class ParentStore:
    """
    In-memory dict + JSON persistence pre parent chunky.

    Štruktúra JSON súboru:
        {
          "parents": {
            "<parent_id>": {
              "page_content": "...",
              "metadata": {...}
            },
            ...
          },
          "version": 1
        }
    """

    FILENAME = "parents.json"
    VERSION  = 1

    def __init__(self, nb_path: Path):
        self.nb_path = nb_path
        self.dir     = nb_path / "parent_store"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file    = self.dir / self.FILENAME

        self._parents: dict[str, Document] = {}
        self._load()

    # ─── Persistence ─────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not self.file.exists():
            return
        try:
            with open(self.file, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("parents", {})
            for pid, doc_dict in raw.items():
                self._parents[pid] = Document(
                    page_content=doc_dict.get("page_content", ""),
                    metadata=doc_dict.get("metadata", {}),
                )
            logger.info(f"ParentStore načítaný: {len(self._parents)} parent chunkov.")
        except Exception as e:
            logger.error(f"ParentStore load zlyhal: {e}")
            self._parents = {}

    def _save(self) -> None:
        payload = {
            "version": self.VERSION,
            "parents": {
                pid: {
                    "page_content": d.page_content,
                    "metadata":     d.metadata,
                }
                for pid, d in self._parents.items()
            },
        }
        # Atomický zápis cez temp file (chráni pred poškodením pri páde)
        tmp = self.file.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(self.file)

    # ─── Public API (volá sa z app.py) ───────────────────────────────────────
    def add_many(self, parents: list[Document]) -> int:
        """Pridá parent chunky. Kľúč je metadata['parent_id']. Vracia počet pridaných."""
        if not parents:
            return 0

        added = 0
        for doc in parents:
            pid = doc.metadata.get("parent_id")
            if not pid:
                logger.warning(
                    f"ParentStore.add_many: parent bez parent_id preskakujem "
                    f"(source={doc.metadata.get('source')})."
                )
                continue
            self._parents[pid] = doc
            added += 1

        if added:
            self._save()
            logger.info(
                f"ParentStore: +{added} parentov (celkom: {len(self._parents)})."
            )
        return added

    def remove_source(self, source: str) -> int:
        """Odstráni všetky parent chunky zo zdroja `source`. Vracia počet odstránených."""
        to_remove = [
            pid for pid, d in self._parents.items()
            if d.metadata.get("source") == source
        ]
        if not to_remove:
            return 0
        for pid in to_remove:
            del self._parents[pid]
        self._save()
        logger.info(f"ParentStore: -{len(to_remove)} parentov (source={source}).")
        return len(to_remove)

    def rebuild(self, parents: list[Document]) -> None:
        """Úplný reset + znovuvybudovanie. Voľ prázdny list na úplné vyčistenie."""
        self._parents = {}
        if parents:
            for doc in parents:
                pid = doc.metadata.get("parent_id")
                if pid:
                    self._parents[pid] = doc
        self._save()
        logger.info(f"ParentStore rebuild: {len(self._parents)} parentov.")

    # ─── Lookup API (volá sa z rag_graph.py) ─────────────────────────────────
    def get(self, parent_id: str) -> Optional[Document]:
        """Vráti parent Document podľa ID, alebo None ak neexistuje."""
        if not parent_id:
            return None
        return self._parents.get(parent_id)

    def get_many(self, parent_ids: list[str]) -> list[Document]:
        """Vráti zoznam parent Documentov v poradí ID-čiek (chýbajúce sú preskočené)."""
        out: list[Document] = []
        for pid in parent_ids:
            doc = self._parents.get(pid)
            if doc is not None:
                out.append(doc)
        return out

    # ─── Introspection ───────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._parents)

    def __contains__(self, parent_id: str) -> bool:
        return parent_id in self._parents

    @property
    def all_parents(self) -> list[Document]:
        return list(self._parents.values())

    def sources(self) -> set[str]:
        """Zoznam unikátnych zdrojov v ParentStore."""
        return {
            d.metadata.get("source", "?") for d in self._parents.values()
        }

    def stats(self) -> dict:
        from collections import defaultdict
        per_source: dict[str, int] = defaultdict(int)
        total_chars = 0
        for d in self._parents.values():
            per_source[d.metadata.get("source", "?")] += 1
            total_chars += len(d.page_content)

        return {
            "total_parents": len(self._parents),
            "per_source":    dict(per_source),
            "total_chars":   total_chars,
            "avg_chars":     (total_chars // len(self._parents))
                             if self._parents else 0,
        }