"""
Pomocný modul pre zobrazenie zdrojových obrázkov stránok.
"""

from pathlib import Path
from typing import Optional
from PIL import Image

from config import NOTEBOOKS_DIR


def get_source_image(nb_id: str, filename: str, page_no: int) -> Optional[Path]:
    """
    Vráti cestu k obrázku stránky pre daný zošit, súbor a číslo stránky.
    """
    img_path = NOTEBOOKS_DIR / nb_id / "page_images" / filename / f"page_{page_no}.png"
    return img_path if img_path.exists() else None


def get_all_source_images(nb_id: str, source_docs: list) -> list[dict]:
    """
    Pripraví zoznam zdrojových obrázkov pre zobrazenie v pravom paneli.
    Vráti deduplikovaný zoznam (source, page, image_path).
    """
    seen    = set()
    sources = []

    for doc in source_docs:
        filename = doc.metadata.get("source", "")
        page_no  = doc.metadata.get("page", 1)
        key      = (filename, page_no)

        if key not in seen:
            seen.add(key)
            img_path = get_source_image(nb_id, filename, page_no)
            sources.append({
                "filename": filename,
                "page":     page_no,
                "img_path": img_path,
                "text":     doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })

    return sources
