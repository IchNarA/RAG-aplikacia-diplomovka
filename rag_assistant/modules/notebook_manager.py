"""
Správa zošitov: vytváranie, mazanie, ukladanie metadát.
Každý zošit = priečinok s vlastnou FAISS DB, súbormi a JSON metadátami.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import NOTEBOOKS_DIR


# ─── Štruktúra zošita ────────────────────────────────────────────────────────
#  data/notebooks/{notebook_id}/
#      meta.json          – názov, dátum vytvorenia
#      files/             – nahrané originálne súbory
#      vector_store/      – FAISS index
#      chat_history.json  – história konverzácie
#      page_images/       – renderované obrázky stránok PDF

class NotebookManager:

    def get_all_notebooks(self) -> list[dict]:
        """Vráti zoznam všetkých zošitov (zoradených od najnovšieho)."""
        notebooks = []
        for path in NOTEBOOKS_DIR.iterdir():
            if path.is_dir():
                meta_file = path / "meta.json"
                if meta_file.exists():
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["id"] = path.name
                    notebooks.append(meta)
        return sorted(notebooks, key=lambda x: x.get("created_at", ""), reverse=True)

    def create_notebook(self, name: str) -> dict:
        """Vytvorí nový zošit a vráti jeho metadáta."""
        nb_id   = f"nb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        nb_path = NOTEBOOKS_DIR / nb_id

        # Vytvor podpriečinky
        (nb_path / "files").mkdir(parents=True)
        (nb_path / "vector_store").mkdir()
        (nb_path / "page_images").mkdir()

        meta = {
            "id":         nb_id,
            "name":       name.strip() or "Nový zošit",
            "created_at": datetime.now().isoformat(),
            "files":      [],
        }
        with open(nb_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Prázdna história chatu
        with open(nb_path / "chat_history.json", "w", encoding="utf-8") as f:
            json.dump([], f)

        return meta

    def delete_notebook(self, nb_id: str) -> bool:
        """Vymaže zošit aj so všetkými súbormi."""
        nb_path = NOTEBOOKS_DIR / nb_id
        if nb_path.exists():
            shutil.rmtree(nb_path)
            return True
        return False

    def get_notebook(self, nb_id: str) -> Optional[dict]:
        """Načíta metadáta zošita."""
        meta_file = NOTEBOOKS_DIR / nb_id / "meta.json"
        if not meta_file.exists():
            return None
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["id"] = nb_id
        return meta

    def get_notebook_path(self, nb_id: str) -> Path:
        return NOTEBOOKS_DIR / nb_id

    # ─── Súbory ──────────────────────────────────────────────────────────────

    def add_file_to_notebook(self, nb_id: str, filename: str) -> None:
        """Zapíše názov súboru do metadát zošita."""
        meta = self.get_notebook(nb_id)
        if meta and filename not in meta.get("files", []):
            meta["files"].append(filename)
            nb_path = NOTEBOOKS_DIR / nb_id
            with open(nb_path / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    def remove_file_from_notebook(self, nb_id: str, filename: str) -> None:
        """Vymaže súbor zo zošita (súbor + záznam v meta)."""
        meta    = self.get_notebook(nb_id)
        nb_path = NOTEBOOKS_DIR / nb_id

        if meta:
            meta["files"] = [f for f in meta.get("files", []) if f != filename]
            with open(nb_path / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        # Vymaž fyzický súbor
        file_path = nb_path / "files" / filename
        if file_path.exists():
            file_path.unlink()

        # Vymaž obrázky tejto stránky
        img_dir = nb_path / "page_images" / filename
        if img_dir.exists():
            shutil.rmtree(img_dir)

    def get_files(self, nb_id: str) -> list[str]:
        meta = self.get_notebook(nb_id)
        return meta.get("files", []) if meta else []

    # ─── História chatu ──────────────────────────────────────────────────────

    def load_chat_history(self, nb_id: str) -> list[dict]:
        history_file = NOTEBOOKS_DIR / nb_id / "chat_history.json"
        if not history_file.exists():
            return []
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_chat_history(self, nb_id: str, history: list[dict]) -> None:
        history_file = NOTEBOOKS_DIR / nb_id / "chat_history.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def clear_chat_history(self, nb_id: str) -> None:
        self.save_chat_history(nb_id, [])
