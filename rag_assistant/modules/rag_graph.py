"""
RAG pipeline cez LangGraph.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, TypedDict, Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from config import LLM_MODEL, OLLAMA_BASE_URL
from modules.vector_store import NotebookVectorStore
from modules.parent_store import ParentStore

logger = logging.getLogger(__name__)

NOT_FOUND_MSG = "Túto informáciu som v nahraných dokumentoch nenašiel."

# ── Parametre pipeline ───────────────────────────────────────────────────────
RERANKER_MODEL      = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
INITIAL_K           = 40       
RERANK_KEEP_K       = 10        # koľko kandidátov prejde rerankerom
MAX_CONTEXT_CHARS   = 6000     # rozpočet pre kontext (gemma3:4b má 8k ctx)
MAX_PARENTS         = 6        # horný limit parentov v kontexte
MIN_RERANK_SCORE    = -4

# ── Reranker singleton ───────────────────────────────────────────────────────
_RERANKER = None


def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        logger.info(f"Načítavam reranker: {RERANKER_MODEL} na {device}")
        _RERANKER = CrossEncoder(RERANKER_MODEL, device=device, max_length=512)
    return _RERANKER


# ── Deiktiká pre query rewriting ─────────────────────────────────────────────
_DEICTIC_PATTERNS = [
    r"\ba (čo|aký|aká|ako|kedy|prečo|potom|ďalej|ten|tá|to|teda)\b",
    r"\b(ten|tá|to|tie|toto|túto|tomto|týmto) ",
    r"\b(vysvetli|rozveď|podrobnejšie|viac|ešte)\b",
    r"\b(predchádzajúc|predošl|prvý|druhý|tretí|ďalší|ďalšia)\b",
]
_DEICTIC_RE = re.compile("|".join(_DEICTIC_PATTERNS), re.IGNORECASE)


def _needs_rewrite(question: str) -> bool:
    q = question.strip()
    if len(q.split()) < 4:
        return True
    return bool(_DEICTIC_RE.search(q))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                           RAGState                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class RAGState(TypedDict, total=False):
    question: str
    chat_history: list[dict]
    standalone_question: str
    retrieved: list[tuple[Document, float]]
    reranked: list[tuple[Document, float]]
    context_docs: list[Document]
    context_text: str
    answer: str
    source_docs: list[Document]
    retrieval_debug: dict


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                           RAGGraph                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class RAGGraph:
    """Hlavná RAG trieda — LangGraph pipeline s parent/child retrievalom."""

    def __init__(self, vector_store: NotebookVectorStore, parent_store: ParentStore):
        self.vs = vector_store
        self.ps = parent_store

        # Hlavný generátor: nízka teplota pre faktualitu
        self.llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            num_predict=1024,
            repeat_penalty=1.15,
            num_ctx=6000,
        )
        # Rýchly LLM pre rewrite (kratšie výstupy)
        self.rewriter_llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
            num_predict=150,
            num_ctx=2048,
        )

        self.graph = self._build_graph()

    # ─── Build graph ─────────────────────────────────────────────────────────
    def _build_graph(self):
        g = StateGraph(RAGState)
        g.add_node("rewrite",  self._rewrite_node)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("rerank",   self._rerank_node)
        g.add_node("expand",   self._expand_node)
        g.add_node("generate", self._generate_node)

        g.set_entry_point("rewrite")
        g.add_edge("rewrite", "retrieve")

        g.add_conditional_edges(
            "retrieve",
            lambda s: "empty" if not s.get("retrieved") else "ok",
            {"empty": END, "ok": "rerank"},
        )
        g.add_conditional_edges(
            "rerank",
            lambda s: "empty" if not s.get("reranked") else "ok",
            {"empty": END, "ok": "expand"},
        )
        g.add_edge("expand", "generate")
        g.add_edge("generate", END)

        return g.compile()

    # ─── Node: rewrite ───────────────────────────────────────────────────────
    def _rewrite_node(self, state: RAGState) -> dict:
        question = state["question"]
        history  = state.get("chat_history") or []

        # Bez histórie alebo otázka je zjavne samostatná → skip
        if not history or not _needs_rewrite(question):
            return {"standalone_question": question}

        # Posledné 4 správy ako kontext
        recent = history[-4:]
        convo = "\n".join(
            f"{'Študent' if m.get('role') == 'user' else 'Asistent'}: {m.get('content','')}"
            for m in recent
        )

        prompt = (
            "Daná je konverzácia a posledná otázka študenta. Ak otázka odkazuje na "
            "predchádzajúci kontext (napr. 'a čo to druhé?', 'vysvetli to'), prepíš ju "
            "ako samostatnú, úplnú otázku v slovenčine. Ak je už samostatná, vráť ju nezmenenú.\n"
            "VRÁŤ IBA prepísanú otázku. Žiadne úvody, žiadne vysvetlenia, žiadne úvodzovky.\n\n"
            f"KONVERZÁCIA:\n{convo}\n\n"
            f"POSLEDNÁ OTÁZKA: {question}\n\n"
            "SAMOSTATNÁ OTÁZKA:"
        )

        try:
            resp = self.rewriter_llm.invoke([HumanMessage(content=prompt)])
            rewritten = resp.content.strip().strip('"').strip("'").strip()
            # Odstráň prípadný prefix typu "Samostatná otázka: ..."
            rewritten = re.sub(r"^(samostatn[aá]?\s*ot[áa]zka[:\-]?\s*)", "", rewritten, flags=re.I)
            if 5 < len(rewritten) < 400:
                logger.info(f"Rewrite: {question!r} → {rewritten!r}")
                return {"standalone_question": rewritten}
        except Exception as e:
            logger.warning(f"Rewrite zlyhal: {e}")

        return {"standalone_question": question}

    # ─── Node: hybrid retrieve ───────────────────────────────────────────────
    def _retrieve_node(self, state: RAGState) -> dict:
        query = state.get("standalone_question") or state["question"]

        if not self.vs.has_documents():
            logger.info("Retrieve: vector store je prázdny.")
            return {
                "retrieved": [],
                "answer": NOT_FOUND_MSG,
                "source_docs": [],
                "retrieval_debug": {"query": query, "note": "prázdny index"},
            }

        results = self.vs.hybrid_search(query, k=INITIAL_K)
        logger.info(f"Retrieve: {len(results)} kandidátov pre {query!r}")

        if not results:
            return {
                "retrieved": [],
                "answer": NOT_FOUND_MSG,
                "source_docs": [],
                "retrieval_debug": {"query": query, "note": "hybrid search 0 výsledkov"},
            }

        return {"retrieved": results}

    # ─── Node: rerank ────────────────────────────────────────────────────────
    def _rerank_node(self, state: RAGState) -> dict:
        query   = state.get("standalone_question") or state["question"]
        results = state.get("retrieved", [])

        if not results:
            return {"reranked": [], "answer": NOT_FOUND_MSG, "source_docs": []}

        reranker = get_reranker()
        docs  = [doc for doc, _ in results]
        pairs = [(query, d.page_content) for d in docs]

        try:
            scores = reranker.predict(pairs, show_progress_bar=False, batch_size=16)
            scores = [float(s) for s in scores]
        except Exception as e:
            logger.error(f"Reranker zlyhal: {e}")
            # Fallback — hybrid skóre
            scores = [float(s) for _, s in results]

        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Filter slabých kandidátov
        kept = [(d, s) for d, s in scored[:RERANK_KEEP_K] if s > MIN_RERANK_SCORE]
        top_raw = [round(s, 3) for _, s in scored[:5]]

        logger.info(f"Rerank: kept={len(kept)} / {len(scored)}; top_raw={top_raw}")

        if not kept:
            return {
                "reranked": [],
                "answer": NOT_FOUND_MSG,
                "source_docs": [],
                "retrieval_debug": {
                    "query": query,
                    "note": f"žiadny kandidát nad prahom {MIN_RERANK_SCORE}",
                    "top_raw_scores": top_raw,
                },
            }

        return {
            "reranked": kept,
            "retrieval_debug": {
                "query": query,
                "initial_retrieved": len(results),
                "after_rerank": len(kept),
                "top_scores": [round(s, 3) for _, s in kept],
            },
        }

    # ─── Node: parent expansion ──────────────────────────────────────────────
    def _expand_node(self, state: RAGState) -> dict:
        reranked = state.get("reranked", [])
        if not reranked:
            return {"context_docs": [], "context_text": "", "source_docs": []}

        # 1) Pokús sa rozšíriť na parentov (ak ParentStore ponúka `get`)
        parent_order: list[str] = []
        seen: set[str] = set()
        for doc, _ in reranked:
            pid = doc.metadata.get("parent_id")
            if pid and pid not in seen:
                seen.add(pid)
                parent_order.append(pid)

        parents: list[Document] = []
        for pid in parent_order[:MAX_PARENTS]:
            p = self._fetch_parent(pid)
            if p is not None:
                parents.append(p)

        # 2) Ak parents nie sú dostupné, použi rerankované child chunky
        context_docs = parents if parents else [d for d, _ in reranked[:RERANK_KEEP_K]]

        # 3) Rozpočet znakov
        limited: list[Document] = []
        total = 0
        for d in context_docs:
            L = len(d.page_content)
            if limited and total + L > MAX_CONTEXT_CHARS:
                break
            limited.append(d)
            total += L

        # 4) source_docs pre UI = child chunky (majú presné čísla strán + images)
        source_docs = [d for d, _ in reranked[:RERANK_KEEP_K]]
        context_text = self._format_context(limited)

        logger.info(f"Kontext: {len(limited)} docs, ~{total} znakov, "
                    f"{'parenti' if parents else 'childovia'}")

        return {
            "context_docs": limited,
            "context_text": context_text,
            "source_docs":  source_docs,
        }

    def _fetch_parent(self, parent_id: str) -> Optional[Document]:
        """Robustne skúsi rôzne rozhrania ParentStore."""
        if not parent_id or self.ps is None:
            return None
        # Skúsi `get`, `fetch`, `mget`, `__getitem__`
        for method_name in ("get", "fetch"):
            fn = getattr(self.ps, method_name, None)
            if callable(fn):
                try:
                    r = fn(parent_id)
                    if isinstance(r, Document):
                        return r
                    if isinstance(r, list) and r and isinstance(r[0], Document):
                        return r[0]
                except Exception:
                    continue
        # mget (langchain storage interface)
        mget = getattr(self.ps, "mget", None)
        if callable(mget):
            try:
                rs = mget([parent_id])
                if rs and rs[0] is not None:
                    r = rs[0]
                    return r if isinstance(r, Document) else None
            except Exception:
                pass
        return None

    # ─── Node: generate ──────────────────────────────────────────────────────
    def _generate_node(self, state: RAGState) -> dict:
        context_docs = state.get("context_docs", [])
        context      = state.get("context_text", "")
        q_orig       = state["question"]
        q_std        = state.get("standalone_question") or q_orig

        if not context.strip():
            return {"answer": NOT_FOUND_MSG, "source_docs": []}

        # Zoznam reálnych súborov, ktoré sú teraz v kontexte
        # → dáme ich modelu explicitne, aby vedel, že INÉ súbory neexistujú
        available_sources = sorted({
            d.metadata.get("source", "") for d in context_docs
            if d.metadata.get("source")
        })

        system = self._system_prompt(available_sources)
        user   = self._user_prompt(q_std, context)

        try:
            resp = self.llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            answer = resp.content.strip()
        except Exception as e:
            logger.error(f"LLM zlyhal: {e}")
            return {"answer": f"⚠️ Chyba pri generovaní: {e}", "source_docs": []}

        if self._looks_like_refusal(answer):
            logger.info("Model sám priznal neznalosť → NOT_FOUND_MSG")
            return {"answer": NOT_FOUND_MSG, "source_docs": []}

        cited_sources = self._filter_cited_sources(answer, state.get("source_docs", []))
        return {"answer": answer, "source_docs": cited_sources}

    # ─── Prompty ─────────────────────────────────────────────────────────────
    @staticmethod
    def _system_prompt(available_sources: list[str]) -> str:
        # Vytvor explicitný zoznam dostupných zdrojov
        if available_sources:
            src_list = "\n".join(f"  • {s}" for s in available_sources)
            src_block = (
                f"DOSTUPNÉ ZDROJE (existujú IBA tieto súbory — žiadne iné):\n{src_list}\n\n"
            )
        else:
            src_block = ""

        return (
            "Si študijný asistent pre vysokoškolských študentov. Odpovedáš VÝHRADNE "
            "na základe zdrojov poskytnutých v sekcii KONTEXT. Si vecný, presný a pedagogický.\n\n"
            f"{src_block}"
            "━━━━━━━━━━━━━━ PRAVIDLÁ (DODRŽIAVAJ PRÍSNE) ━━━━━━━━━━━━━━\n"
            "1. Používaj IBA informácie z KONTEXTU. NIKDY nedopĺňaj vlastné znalosti.\n"
            f"2. Ak odpoveď v KONTEXTE NIE JE, vráť PRESNE: \"{NOT_FOUND_MSG}\"\n"
            "3. CITÁCIE — KRITICKY DÔLEŽITÉ:\n"
            "   • Cituj v hranatých zátvorkách s NÁZVOM SÚBORU a číslom strany.\n"
            "   • Názov súboru musí byť PRESNE ten zo zoznamu DOSTUPNÝCH ZDROJOV.\n"
            "   • NIKDY nepoužívaj čísla zdrojov ako [1, s. X], [2, s. X].\n"
            "   • NIKDY nevymýšľaj súbory, ktoré nie sú v zozname vyššie.\n"
            "   • Každé faktografické tvrdenie má mať citáciu priamo za vetou.\n"
            "4. MATEMATIKU PÍŠ V LATEXu ak sa vzorce nachádzajú v KONTEXTE:\n"
            "   • inline vzorec: obal ho do $ $\n"
            "   • samostatný vzorec na novom riadku: obal ho do $$ $$\n"
            "   • Ak KONTEXT neobsahuje vzorce, NEPÍŠ žiadny LaTeX.\n"
            "5. Odpovedaj v SLOVENČINE. Odborné EN termíny v zátvorke.\n"
            "6. Ak sú zdroje protichodné, uveď oba pohľady s citáciami.\n"
            "7. Žiadne frázy 'všeobecne', 'typicky', 'zvyčajne', pokiaľ to nie je v KONTEXTE.\n"
            "8. Ak kontext obsahuje IBA príklad použitia pojmu bez jeho definície, povedz to "
            "   explicitne. Nikdy nereprodukuj príklad ako keby to bola definícia.\n"
            "9. NIKDY neopakuj tú istú vetu alebo frázu viackrát v jednej odpovedi.\n"
        )

    def _user_prompt(self, question: str, context: str) -> str:
        return (
            "KONTEXT — JEDINÝ zdroj, z ktorého smieš čerpať (každý úryvok má svoj názov súboru a stranu):\n"
            "═══════════════════════════════════════════════\n"
            f"{context}\n"
            "═══════════════════════════════════════════════\n\n"
            f"OTÁZKA ŠTUDENTA: {question}\n\n"
            "Odpoveď v slovenčine s citáciami presne podľa vzoru [súbor.pdf, s. X] "
            "a LaTeX vzorcami. Cituj iba reálne názvy súborov z KONTEXTU:"
        )

    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        """
        Formát: namiesto ZDROJ [N] sa priamo uvedie [názov_súboru, s. X].
        LLM si to len presne skopíruje do odpovede — nevymyslí čísla zdrojov.
        """
        blocks = []
        for d in docs:
            src  = d.metadata.get("source", "neznámy_zdroj")
            page = d.metadata.get("page", "?")
            blocks.append(
                f"━━━ [{src}, s. {page}] ━━━\n"
                f"{d.page_content.strip()}"
            )
        return "\n\n".join(blocks)

    # ─── Post-processing helpers ─────────────────────────────────────────────
    @staticmethod
    def _looks_like_refusal(answer: str) -> bool:
        """Detekcia, keď model namiesto NOT_FOUND_MSG píše voľné odmietnutie."""
        if NOT_FOUND_MSG in answer:
            return False  # už je to správna forma
        low = answer.lower()
        triggers = [
            "nie je uvedené v dokumentoch",
            "v dokumentoch som nenašiel",
            "v zdrojoch nie je",
            "v kontexte sa nenachádza",
            "nemám k dispozícii informácie",
            "v poskytnutých zdrojoch nie",
            "nenašiel som informáciu",
        ]
        # Iba ak je to krátka odpoveď a obsahuje trigger
        return len(answer) < 300 and any(t in low for t in triggers)

    @staticmethod
    def _filter_cited_sources(answer: str, source_docs: list[Document]) -> list[Document]:
        """
        Z kandidátov na zdroje nechaj IBA tie, ktoré model skutočne citoval v odpovedi.
        Tak bude pravý panel zobrazovať presne tie strany, ktoré figurovali v texte.
        """
        if not source_docs:
            return []

        # [súbor.pdf, s. 3]  |  [súbor, strana 3]  |  [súbor.pdf, p. 3]
        pat = re.compile(
            r"\[([^\[\]\n]+?)[,;]\s*(?:s\.?|str\.?|strana|strane|page|p\.?)\s*(\d+)\s*\]",
            re.IGNORECASE,
        )
        cited: set[tuple[str, int]] = set()
        for m in pat.finditer(answer):
            src  = m.group(1).strip().lower()
            page = int(m.group(2))
            cited.add((src, page))

        if not cited:
            # Model necitoval v štandardnom formáte — vráť všetko, nech má študent čo overovať
            return source_docs

        kept: list[Document] = []
        seen: set[tuple[str, int]] = set()
        for d in source_docs:
            d_src  = (d.metadata.get("source") or "").lower()
            d_page = int(d.metadata.get("page") or 0)
            key = (d_src, d_page)
            if key in seen:
                continue
            # Fuzzy match: dovoľ aj bez extension-u a substring
            hit = False
            for c_src, c_page in cited:
                if c_page != d_page:
                    continue
                if c_src == d_src or c_src in d_src or d_src in c_src:
                    hit = True
                    break
            if hit:
                seen.add(key)
                kept.append(d)

        return kept if kept else source_docs

    # ─── Public API ──────────────────────────────────────────────────────────
    def query(
        self,
        question: str,
        chat_history: Optional[list[dict]] = None,
    ) -> tuple[str, list[Document], dict]:
        """
        Spusti RAG pipeline.

        Returns:
            (answer, source_docs, retrieval_debug)
            - answer: slovenská odpoveď s [citáciami] a LaTeXom
            - source_docs: iba dokumenty reálne citované v odpovedi (pre UI panel)
            - retrieval_debug: dict s info o retrievale (top_scores, counts)
        """
        init_state: RAGState = {
            "question":     question,
            "chat_history": chat_history or [],
        }
        try:
            final = self.graph.invoke(init_state)
        except Exception as e:
            logger.error(f"RAG graph pipeline zlyhal: {e}", exc_info=True)
            return f"⚠️ Chyba RAG pipeline: {e}", [], {}

        answer = (final.get("answer") or NOT_FOUND_MSG).strip()
        sources = final.get("source_docs", []) or []
        debug   = final.get("retrieval_debug", {}) or {}

        # Ak je odpoveď = NOT_FOUND, neukazuj žiadne zdroje (boli by zavádzajúce)
        if answer == NOT_FOUND_MSG:
            sources = []

        return answer, sources, debug
        