"""
RAG Study Assistant — Hlavná Streamlit aplikácia
Vzdelávacia pomôcka pre študentov s RAG pipeline.
"""

import logging
import re
import hashlib
import urllib.parse
from pathlib import Path

import streamlit as st

from config import NOTEBOOKS_DIR
from modules.notebook_manager import NotebookManager
from modules.document_processor import DocumentProcessor
from modules.vector_store import NotebookVectorStore
from modules.parent_store import ParentStore
from modules.rag_graph import RAGGraph
from modules.page_renderer import get_all_source_images
from config import IS_REDUCED_QUALITY, LLM_MODEL, get_hardware_info

NOT_FOUND_MSG = "Túto informáciu som v nahraných dokumentoch nenašiel."

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def show_hardware_warning():
    """Zobrazí upozornenie pri slabom hardvéri."""
    if IS_REDUCED_QUALITY:
        info = get_hardware_info()
        st.warning(
            f"⚠️ **Detegovaný slabší hardvér** ({info['total_ram_gb']} GB RAM, "
            f"bez GPU). Aplikácia používa úsporný model **{LLM_MODEL}** s nižšou "
            f"kvalitou odpovedí. Pre plnú kvalitu odporúčame zariadenie s **min. 12 GB "
            f"RAM** alebo dedikovanou grafickou kartou.",
            icon="⚠️",
        )

# ─── Streamlit konfigurácia ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Študijný asistent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Styling ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --primary:      #4F46E5;
        --primary-dark: #3730A3;
        --accent:       #10B981;
        --bg-sidebar:   #0F172A;
        --bg-chat:      #1E293B;
        --bg-card:      #334155;
        --text-main:    #F1F5F9;
        --text-muted:   #94A3B8;
        --border:       #475569;
        --user-bubble:  #4F46E5;
        --ai-bubble:    #1E3A5F;
        --source-bg:    #0F2744;
    }

    /*#MainMenu, footer, header { visibility: hidden; }*/
    .block-container { padding-top: 3rem !important; }

    [data-testid="stSidebar"] {
        background: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text-main) !important; }

    .chat-message-user {
        background: var(--user-bubble);
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 4px 0 4px 20%;
        color: white;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .chat-message-ai {
        background: var(--ai-bubble);
        border: 1px solid #2D4A70;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 4px 20% 4px 0;
        color: var(--text-main);
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    .source-card {
        background: var(--source-bg);
        border: 1px solid #1E4A80;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .source-card h4 { color: #60A5FA; font-size: 0.8rem; margin-bottom: 6px; }

    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79,70,229,0.3);
    }

    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border) !important;
        border-radius: 10px !important;
    }
    hr { border-color: var(--border) !important; }
    [data-testid="stChatInput"] textarea {
        background: #1E293B !important;
        border-radius: 12px !important;
        border-color: var(--border) !important;
    }

    .info-box {
        background: #1E3A5F;
        border-left: 4px solid #3B82F6;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.85rem;
        color: #BAE6FD;
    }
    .warn-box {
        background: #422006;
        border-left: 4px solid #F59E0B;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.85rem;
        color: #FDE68A;
    }

    .app-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-bottom: 1px solid var(--border);
        padding: 12px 20px;
        border-radius: 10px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .app-header h1 {
        margin: 0;
        font-size: 1.4rem;
        background: linear-gradient(90deg, #818CF8, #34D399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ─── Myšlienková mapa ─────────────────────────────────────── */
    .mm-central {
        background: linear-gradient(135deg, #6366F1 0%, #10B981 100%);
        color: white;
        text-align: center;
        padding: 18px 20px;
        border-radius: 14px;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 20px;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
        letter-spacing: 0.02em;
    }
    .mm-branch-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
        transition: all 0.2s;
    }
    .mm-branch-card:hover {
        border-color: #6366F1;
        box-shadow: 0 4px 14px rgba(99,102,241,0.2);
        transform: translateY(-1px);
    }
    .mm-branch-title {
        display: block;
        color: #A5B4FC !important;
        font-weight: 700;
        font-size: 1rem;
        text-decoration: none !important;
        margin-bottom: 8px;
        padding: 8px 12px;
        border-radius: 8px;
        background: rgba(129, 140, 248, 0.14);
        border-left: 3px solid #6366F1;
    }
    .mm-branch-title:hover {
        background: rgba(129, 140, 248, 0.28);
        color: #C7D2FE !important;
    }
    .mm-subs {
        padding-left: 12px;
        display: flex;
        flex-direction: column;
        gap: 3px;
        margin-top: 4px;
    }
    .mm-sub {
        color: #94A3B8 !important;
        font-size: 0.86rem;
        text-decoration: none !important;
        padding: 5px 10px;
        border-radius: 5px;
        transition: all 0.15s;
    }
    .mm-sub:hover {
        color: #E2E8F0 !important;
        background: rgba(148, 163, 184, 0.12);
        padding-left: 14px;
    }

    /* ─── Súhrn ─────────────────────────────────────── */
    .summary-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        margin-top: 12px;
        line-height: 1.65;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session state inicializácia ──────────────────────────────────────────────
def init_session():
    defaults = {
        "active_notebook_id":  None,
        "chat_history":        [],
        "source_docs":         [],
        "vector_store":        None,
        "processing":          False,
        "last_sources":        [],
        "debug_chunks":        {},
        "debug_last_retrieval": None,
        "tools_panel_open":    False,
        "summary_result":      None,
        "flashcards_result":   None,
        "mindmap_result":      None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─── Cached stores ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_vector_store(nb_id: str) -> NotebookVectorStore:
    return NotebookVectorStore(NOTEBOOKS_DIR / nb_id)


@st.cache_resource(show_spinner=False)
def get_parent_store(nb_id: str) -> ParentStore:
    return ParentStore(NOTEBOOKS_DIR / nb_id)


@st.cache_resource(show_spinner=False)
def get_rag_graph(nb_id: str) -> RAGGraph:
    """RAGGraph je cache-ovaný — reranker a LLM sa načítajú iba raz."""
    return RAGGraph(get_vector_store(nb_id), get_parent_store(nb_id))


# ─── Helper: spracovanie otázky (zdieľaný kód pre chat_input + mindmap) ──────
def process_user_message(manager: NotebookManager, nb_id: str, question: str):
    """Pridá otázku do histórie, spustí RAG a uloží odpoveď + zdroje."""
    st.session_state.chat_history.append({"role": "user", "content": question})

    try:
        rag = get_rag_graph(nb_id)
        answer, source_docs, retrieval_debug = rag.query(
            question=question,
            chat_history=st.session_state.chat_history[:-1],
        )
        if retrieval_debug:
            st.session_state.debug_last_retrieval = retrieval_debug
    except Exception as e:
        logger.error(f"RAG chyba: {e}", exc_info=True)
        answer      = f"⚠️ Chyba pri spracovaní: {e}"
        source_docs = []

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    manager.save_chat_history(nb_id, st.session_state.chat_history)
    st.session_state.last_sources = get_all_source_images(nb_id, source_docs)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar(manager: NotebookManager):
    st.sidebar.markdown("## 📚 Moje zošity")
    st.sidebar.markdown("---")

    with st.sidebar.expander("➕ Nový zošit", expanded=False):
        nb_name = st.text_input(
            "Názov zošita",
            placeholder="napr. Fyzika – Mechanika",
            key="new_nb_name",
            label_visibility="collapsed",
        )
        if st.button("Vytvoriť", key="create_nb", use_container_width=True, type="primary"):
            if nb_name.strip():
                new_nb = manager.create_notebook(nb_name)
                st.session_state.active_notebook_id = new_nb["id"]
                st.session_state.chat_history = []
                st.session_state.last_sources = []
                st.rerun()
            else:
                st.warning("Zadaj názov zošita.")

    st.sidebar.markdown("---")
    notebooks = manager.get_all_notebooks()

    if not notebooks:
        st.sidebar.markdown(
            '<div class="warn-box">Žiadne zošity. Vytvor si prvý zošit! 👆</div>',
            unsafe_allow_html=True,
        )
        return

    active_id = st.session_state.active_notebook_id

    for nb in notebooks:
        nb_id = nb["id"]
        is_active = nb_id == active_id
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            btn_label = f"{'▶ ' if is_active else ''}{nb['name']}"
            if st.button(
                btn_label,
                key=f"select_nb_{nb_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    st.session_state.active_notebook_id = nb_id
                    st.session_state.chat_history = manager.load_chat_history(nb_id)
                    st.session_state.last_sources = []
                    get_vector_store.clear()
                    get_parent_store.clear()
                    get_rag_graph.clear()
                    st.rerun()
        with col2:
            if st.button("🗑", key=f"del_nb_{nb_id}", help="Vymazať zošit"):
                manager.delete_notebook(nb_id)
                if is_active:
                    st.session_state.active_notebook_id = None
                    st.session_state.chat_history = []
                    st.session_state.last_sources = []
                    get_vector_store.clear()
                    get_parent_store.clear()
                    get_rag_graph.clear()
                st.rerun()

    st.sidebar.markdown("---")
    if active_id:
        render_file_panel(manager, active_id)


def render_file_panel(manager: NotebookManager, nb_id: str):
    nb = manager.get_notebook(nb_id)
    name = nb.get("name", "Zošit") if nb else "Zošit"
    st.sidebar.markdown(f"### 📁 {name}")

    files = manager.get_files(nb_id)
    if files:
        for fname in files:
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                ext_icons = {".pdf": "📄", ".xlsx": "📊", ".xls": "📊", ".docx": "📝", ".doc": "📝"}
                ext = Path(fname).suffix.lower()
                icon = ext_icons.get(ext, "📎")
                st.markdown(
                    f"<small>{icon} {fname[:28]}{'…' if len(fname) > 28 else ''}</small>",
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("✕", key=f"del_file_{nb_id}_{fname}", help="Odstrániť súbor"):
                    with st.spinner("Odstraňujem..."):
                        manager.remove_file_from_notebook(nb_id, fname)
                        _rebuild_index_without_file(manager, nb_id, fname)
                        get_vector_store.clear()
                        get_parent_store.clear()
                        get_rag_graph.clear()
                    st.rerun()
    else:
        st.sidebar.markdown(
            '<div class="info-box">Žiadne súbory. Nahraj dokumenty nižšie.</div>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📤 Nahrať súbory**")

    mode_option = st.sidebar.radio(
        "Spôsob spracovania PDF:",
        options=["⚡ Rýchle", "🔬 Kvalitné"],
        index=0,
        key=f"processing_mode_{nb_id}",
        help=(
            "⚡ Rýchle – čistý text, rýchly.\n\n"
            "🔬 Kvalitné – zachováva kvalitnú štruktúru vzorcov, ale pomalší"
        ),
    )
    processing_mode = "fast" if "Rýchle" in mode_option else "quality"

    if processing_mode == "quality":
        st.sidebar.markdown(
            '<div class="info-box">🔬 <strong>MinerU</strong> – zachováva vzorce ako LaTeX.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="info-box">⚡ <strong>PyMuPDF</strong> – rýchla extrakcia čistého textu.</div>',
            unsafe_allow_html=True,
        )

    uploaded = st.sidebar.file_uploader(
        "Vyber súbory",
        type=["pdf", "xlsx", "xls", "docx", "doc"],
        accept_multiple_files=True,
        key=f"uploader_{nb_id}",
        label_visibility="collapsed",
    )

    if uploaded:
        nb_path = manager.get_notebook_path(nb_id)
        files_dir = nb_path / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        st.session_state.setdefault("uploaded_file_hashes", {})
        st.session_state["uploaded_file_hashes"].setdefault(nb_id, {})
        prev_hashes: dict[str, str] = st.session_state["uploaded_file_hashes"][nb_id]

        new_files = []
        for uf in uploaded:
            dest = files_dir / uf.name
            buf = bytes(uf.getbuffer())
            digest = hashlib.sha256(buf).hexdigest()[:16]
            ext = Path(uf.name).suffix.lower()
            mode_tag = processing_mode if ext == ".pdf" else "fast"
            digest_key = f"{digest}|{mode_tag}"
            if prev_hashes.get(uf.name) == digest_key and dest.exists():
                continue
            dest.write_bytes(buf)
            prev_hashes[uf.name] = digest_key
            new_files.append(uf.name)

        if new_files:
            mode_label = "MinerU 🔬" if processing_mode == "quality" else "PyMuPDF ⚡"
            with st.sidebar.status(f"⚙️ Spracovávam dokumenty ({mode_label})...", expanded=True) as status:
                processor = DocumentProcessor(nb_path)
                vs = NotebookVectorStore(nb_path)
                ps = ParentStore(nb_path)
                for fname in new_files:
                    ext = Path(fname).suffix.lower()
                    effective_mode = processing_mode if ext == ".pdf" else "fast"
                    status.write(f"📄 Parsujem: {fname}")
                    try: vs.remove_source(fname)
                    except Exception: pass
                    try: ps.remove_source(fname)
                    except Exception: pass
                    parents, children = processor.process_file(fname, mode=effective_mode)
                    if children:
                        vs.add_documents(children)
                        ps.add_many(parents)
                        manager.add_file_to_notebook(nb_id, fname)
                get_vector_store.clear()
                get_parent_store.clear()
                get_rag_graph.clear()
                status.update(label="✅ Spracovanie dokončené!", state="complete")
            st.rerun()

    st.sidebar.markdown("---")
    

    st.sidebar.markdown("---")
    if st.sidebar.button("🗑 Vymazať históriu chatu", use_container_width=True):
        manager.clear_chat_history(nb_id)
        st.session_state.chat_history = []
        st.session_state.last_sources = []
        st.rerun()


def _rebuild_index_without_file(manager, nb_id, removed_file):
    nb_path = manager.get_notebook_path(nb_id)
    processor = DocumentProcessor(nb_path)
    vs = NotebookVectorStore(nb_path)
    ps = ParentStore(nb_path)
    all_parents, all_children = [], []
    for fname in manager.get_files(nb_id):
        if fname != removed_file:
            parents, children = processor.process_file(fname, mode="fast")
            all_parents.extend(parents)
            all_children.extend(children)
    vs.rebuild_index(all_children)
    ps.rebuild(all_parents)


# ─── LLM helper pre tools (bez RAG retrievalu) ────────────────────────────────
def _call_llm_for_tools(prompt: str, max_tokens: int = 2000, temperature: float = 0.2) -> str:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage
    from config import LLM_MODEL, OLLAMA_BASE_URL

    llm = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=max_tokens,
        num_ctx=8192,
    )
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content.strip()
    except Exception as e:
        return f"⚠️ Chyba: {e}"


def _get_diverse_sample(nb_id: str, max_chars: int = 90000, seed: int = 42) -> str:
    """
    Vyberie reprezentatívnu vzorku chunkov rovnomerne z celého dokumentu.
    Deterministická (seed) → konzistentné výsledky pri opakovanom generovaní.
    """
    import random
    vs = get_vector_store(nb_id)
    if not vs.has_documents():
        return ""

    docs = vs.all_docs
    if not docs:
        return ""

    total = len(docs)
    # Rovnomerne rozložené chunky + málo náhodných (mix)
    if total <= 30:
        selected = list(docs)
    else:
        step = max(1, total // 30)
        evenly = [docs[i] for i in range(0, total, step)][:30]
        rng = random.Random(seed)
        randomly = rng.sample(docs, min(15, total))
        seen, selected = set(), []
        for d in evenly + randomly:
            k = id(d)
            if k not in seen:
                seen.add(k)
                selected.append(d)

    # Text + zdroj na každý chunk (pre lepšie atribúcie v LLM výstupe)
    parts, used = [], 0
    for d in selected:
        src = d.metadata.get("source", "zdroj")
        pg  = d.metadata.get("page", "?")
        body = d.page_content.strip()
        chunk = f"[{src}, s. {pg}]\n{body}"
        if used + len(chunk) > max_chars and parts:
            break
        parts.append(chunk)
        used += len(chunk)

    return "\n\n---\n\n".join(parts)


# ─── Sumarizácia ──────────────────────────────────────────────────────────────
def _extract_key_concepts(nb_id: str, sample: str, n: int = 8) -> list[str]:
    """Prvá fáza: LLM vyberie reálne kľúčové pojmy z textu (nie si vymýšľa)."""
    import json as _json

    prompt = (
        f"Z TEXTU nižšie vyber {n} najdôležitejších odborných pojmov/tém. "
        "PRAVIDLÁ:\n"
        "• IBA pojmy, ktoré sa v texte DOSLOVNE vyskytujú (nie synonymá ani preklady).\n"
        "• Žiadne všeobecnosti ('analýza dát'), iba konkrétne odborné termíny.\n"
        "• Jeden pojem = 1-4 slová.\n"
        "• Odpovedaj v jazyku originálu (väčšinou EN pri odbornej literatúre).\n\n"
        "Formát — VRÁŤ IBA JSON pole reťazcov, nič iné:\n"
        '["ANOVA", "factorial design", ...]\n\n'
        f"TEXT:\n{sample}"
    )
    raw = _call_llm_for_tools(prompt, max_tokens=400, temperature=0.0)
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        items = _json.loads(m.group())
        return [str(x).strip() for x in items if isinstance(x, str) and x.strip()][:n]
    except Exception:
        return []


def render_summary_tab(nb_id: str):
    st.markdown("### 📋 Sumarizácia dokumentov")
    

    if st.button("🔄 Generovať súhrn", use_container_width=True, type="primary"):
        vs = get_vector_store(nb_id)
        if not vs.has_documents():
            st.warning("Žiadne dokumenty nie sú nahrané.")
            return

        with st.status("Generujem súhrn...", expanded=True) as status:
            # ─── Fáza 1: extrakcia kľúčových pojmov ──────────────────────
            status.write("🔍 Analyzujem dokument a hľadám kľúčové pojmy...")
            sample = _get_diverse_sample(nb_id, max_chars=20000, seed=7)
            if not sample:
                status.update(label="❌ Prázdny dokument", state="error")
                return

            concepts = _extract_key_concepts(nb_id, sample, n=8)
            if not concepts:
                st.error("Nepodarilo sa extrahovať pojmy.")
                return

            status.write(f"✅ Našiel som {len(concepts)} pojmov: {', '.join(concepts)}")

            # ─── Fáza 2: RAG citácie pre každý pojem ──────────────────────
            status.write("📚 Vyhľadávam zdrojové pasáže pre každý pojem...")
            concept_blocks = []
            for concept in concepts:
                hits = vs.hybrid_search(concept, k=3)
                if not hits:
                    continue
                top_doc = hits[0][0]
                src  = top_doc.metadata.get("source", "zdroj")
                page = top_doc.metadata.get("page", "?")
                # Krátka pasáž (~400 znakov), ktorá skutočne obsahuje pojem
                body = top_doc.page_content.strip()[:500]
                concept_blocks.append(
                    f"POJEM: {concept}\n"
                    f"ZDROJ: [{src}, s. {page}]\n"
                    f"PASÁŽ: {body}"
                )

            if not concept_blocks:
                st.error("Nenašli sa žiadne relevantné pasáže.")
                return

            grounded_context = "\n\n---\n\n".join(concept_blocks)

            # ─── Fáza 3: finálny štruktúrovaný súhrn ──────────────────────
            status.write("✍️ Skladám finálny súhrn...")
            final_prompt = (
                "Si pedagogický asistent. Vytvor ŠTRUKTÚROVANÝ SÚHRN v slovenčine "
                "VÝHRADNE na základe POJMOV A PASÁŽÍ nižšie.\n\n"
                "PRAVIDLÁ:\n"
                "• Píš výhradne v SPISOVNEJ SLOVENČINE — NIE česky. "
                "Slovenské varianty: 'vrátane', 'rôznymi', 'pomocou', 'niektoré', 'medzi'.\n"
                "• Sekcia 'Kľúčové pojmy' = STRUČNÁ DEFINÍCIA každého pojmu (čo to je).\n"
                "• Sekcia 'Hlavné myšlienky' = DÔLEŽITÉ TVRDENIA o vzťahoch medzi pojmami "
                "alebo o spôsobe ich použitia. NESMÚ to byť zopakované definície!\n"
                "• FORMÁT CITÁCIÍ — KRITICKY DÔLEŽITÉ:\n"
                "  – POUŽI presne tú citáciu, ktorá je v sekcii 'ZDROJ:' alebo 'ZDROJE:' v dátach.\n"
                "  – Ak je v dátach 'ZDROJ: [Montgomery.pdf, s. 237]', "
                "v odpovedi napíš [Montgomery.pdf, s. 237].\n"
                "  – NIKDY nepíš [súbor, s. X] — to je iba zástupný zápis, nie reálna citácia.\n"
                "• Odborné anglické termíny zachovaj s prekladom: 'replication (replikácia)'.\n"
                "• Vzorce v LaTeX, ak sú v pasáži. Ak nie, nevymýšľaj.\n\n"
                "FORMÁT (presne takto):\n\n"
                "## 🎯 O čom dokument je\n"
                "2-3 vety — hlavná téma a cieľ knihy/dokumentu.\n\n"
                "## 🔑 Kľúčové pojmy\n"
                "Pre KAŽDÝ pojem urob jednu odrážku — DEFINÍCIU (čo to je):\n"
                "- **Pojem (EN termín)** — definícia v 1-2 vetách. [presný názov súboru, s. X]\n\n"
                "## 📖 Hlavné myšlienky a vzťahy\n"
                "3-5 číslovaných bodov o tom, AKO sa pojmy POUŽÍVAJÚ alebo AKO SÚVISIA. "
                "Žiadne opakovanie definícií! Zameraj sa na praktické tvrdenia, súvislosti, postupy.\n"
                "PRÍKLAD DOBRÉHO BODU:\n"
                "  1. ANOVA umožňuje rozhodnúť, či sú rozdiely medzi skupinami štatisticky významné, "
                "porovnávaním rozptylu medzi skupinami a rozptylu vnútri skupín. [Montgomery.pdf, s. 237]\n"
                "PRÍKLAD ZLÉHO BODU:\n"
                "  1. ANOVA je analýza rozptylu. [Montgomery.pdf, s. 237]  ← toto je definícia, nie myšlienka\n\n"
                "## ⚡ Čo si zapamätať\n"
                "2 vety — najdôležitejší praktický záver pre študenta.\n\n"
                "DÔLEŽITÉ: Tvoj výstup MUSÍ KONČIŤ sekciou '⚡ Čo si zapamätať'. "
                "Nepridávaj žiadne ďalšie poznámky, oddeľovače, vety typu 'Tento dokument bol vytvorený...' "
                "ani záverečné komentáre.\n\n"
                "DÁTA (iba z nich čerpaj — citácie kopíruj presne tak, ako sú uvedené v ZDROJ:/ZDROJE:):\n\n"
                f"{grounded_context}"
            )
            result = _call_llm_for_tools(final_prompt, max_tokens=2000, temperature=0.1)
            st.session_state.summary_result = result
            status.update(label="✅ Súhrn pripravený!", state="complete")

    if st.session_state.get("summary_result"):
        st.markdown(
            f'<div class="summary-card">{_render_markdown_with_latex(st.session_state.summary_result)}</div>',
            unsafe_allow_html=True,
        )


def _render_markdown_with_latex(text: str) -> str:
    """Escaping pre HTML výstup so zachovaním LaTeX-u a základného markdownu."""
    # Markdown nadpisy → HTML
    text = re.sub(r"^## (.+)$", r"<h3 style='color:#A5B4FC;margin-top:16px;'>\1</h3>", text, flags=re.M)
    text = re.sub(r"^### (.+)$", r"<h4 style='color:#C7D2FE;'>\1</h4>", text, flags=re.M)
    # Bold/italic
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^\*]+?)\*(?!\*)", r"<em>\1</em>", text)
    # Odrážky
    text = re.sub(r"^- (.+)$", r"<li>\1</li>", text, flags=re.M)
    text = re.sub(r"(<li>.*?</li>(\s*<li>.*?</li>)*)", r"<ul>\1</ul>", text, flags=re.S)
    # Newlines
    text = text.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return text


# ─── Myšlienková mapa ─────────────────────────────────────────────────────────
def render_mindmap_tab(nb_id: str):
    import streamlit.components.v1 as components
    import json as _json

    st.markdown("### 🧠 Myšlienková mapa")
    

    if st.button("🔄 Generovať mapu", use_container_width=True, type="primary"):
        with st.spinner("Analyzujem dokumenty..."):
            sample = _get_diverse_sample(nb_id, max_chars=10000, seed=11)
            if not sample:
                st.warning("Žiadne dokumenty nie sú nahrané.")
                return

            prompt = (
                "Z TEXTU nižšie vytvor MYŠLIENKOVÚ MAPU v SLOVENČINE.\n\n"
                "PRAVIDLÁ:\n"
                "• VŠETKY texty v mape (central, topic, subtopics) musia byť v SLOVENČINE.\n"
                "• Aj keď je zdrojový text v angličtine, pojmy PREKLADAJ do slovenčiny.\n"
                "• Pri zaužívaných odborných termínoch použij slovenský preklad s anglickým "
                "ekvivalentom v zátvorke: regresia (regression), replikácia (replication).\n"
                "• Ak slovenský preklad neexistuje alebo nie je zaužívaný (napr. ANOVA, "
                "p-hodnota), nechaj originálny pojem.\n"
                "• NEVYMÝŠĽAJ pojmy — iba tie, ktoré reálne v texte figurujú.\n"
                "• Krátko a vecne: topic 2–4 slová, subtopic 1–3 slová.\n"
                "• KRITICKÉ: V hodnotách JSON-u NEPOUŽÍVAJ úvodzovky (\") ani apostrofy ('). "
                "Ak by si potreboval ohraničiť časť textu, použi ZÁTVORKY napr. (text).\n\n"
                "VRÁŤ IBA JSON (nič iné, žiaden markdown, žiadne ```json):\n"
                "{\n"
                '  "central": "Hlavná téma po slovensky (2-5 slov)",\n'
                '  "branches": [\n'
                '    {"topic": "Téma po slovensky", "subtopics": ["pojem", "pojem", "pojem"]}\n'
                '  ]\n'
                "}\n\n"
                "PRÍKLAD pre anglickú učebnicu o štatistike:\n"
                '{"central": "Plánovanie experimentov",\n'
                ' "branches": [\n'
                '   {"topic": "Faktoriálne dizajny", "subtopics": ["replikácia (replication)", "blokovanie (blocking)", "interakcie"]},\n'
                '   {"topic": "Analýza variancie (ANOVA)", "subtopics": ["F-test", "rezíduá", "p-hodnota"]}\n'
                ' ]}\n\n'
                "Vytvor presne 6 vetiev. Každá má 3–4 subtopicy.\n\n"
                f"TEXT:\n{sample}"
            )
            raw = _call_llm_for_tools(prompt, max_tokens=1500, temperature=0.2)

            try:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if not m:
                    st.error("Model nevrátil validný JSON.")
                    return
                parsed = _json.loads(m.group())
                if "central" not in parsed or "branches" not in parsed:
                    st.error("Chýba 'central' alebo 'branches'.")
                    return
                # Validácia & trim
                parsed["branches"] = [
                    {
                        "topic": b["topic"].strip(),
                        "subtopics": [s.strip() for s in b.get("subtopics", [])
                                      if isinstance(s, str) and s.strip()][:4],
                    }
                    for b in parsed["branches"]
                    if isinstance(b, dict) and "topic" in b and b["topic"].strip()
                ][:6]
                if not parsed["branches"]:
                    st.error("Mapa neobsahuje validné vetvy.")
                    return
                st.session_state.mindmap_result = parsed
            except _json.JSONDecodeError as e:
                st.error(f"JSON chyba: {e}")
                return

    mindmap = st.session_state.get("mindmap_result")
    if not mindmap:
        return

    # ─── SVG radiálny render ──────────────────────────────────────────────
    data_json = _json.dumps(mindmap, ensure_ascii=False)

    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: radial-gradient(ellipse at center, #1a2540 0%, #0B1220 70%);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  overflow: hidden;
}
#wrap {
  width: 100vw;
  height: 100vh;
  position: relative;
}
svg {
  width: 100%;
  height: 100%;
  cursor: grab;
}
svg:active { cursor: grabbing; }

.link {
  fill: none;
  stroke-opacity: 0.65;
  stroke-linecap: round;
}

.node-central rect {
  fill: url(#gradCentral);
  stroke: #818CF8;
  stroke-width: 2;
  filter: drop-shadow(0 4px 14px rgba(99,102,241,0.5));
}
.node-central text {
  fill: white;
  font-weight: 700;
  font-size: 15px;
  text-anchor: middle;
  pointer-events: none;
  letter-spacing: 0.02em;
}

.node-topic { transition: transform 0.15s; }
.node-topic:hover { transform: scale(1.05); transform-origin: center; }
.node-topic rect {
  stroke-width: 2;
  transition: filter 0.15s;
}
.node-topic:hover rect {
  filter: brightness(1.25) drop-shadow(0 4px 10px rgba(0,0,0,0.5));
}
.node-topic text {
  fill: white;
  font-weight: 600;
  font-size: 12.5px;
  text-anchor: middle;
  pointer-events: none;
}

.node-sub {  }
.node-sub rect {
  fill: #1E293B;
  stroke-width: 1.5;
  transition: all 0.15s;
}
.node-sub:hover rect {
  fill: #2D3F6B;
  stroke-width: 2.5;
}
.node-sub text {
  fill: #CBD5E1;
  font-size: 11.5px;
  text-anchor: middle;
  pointer-events: none;
}
.node-sub:hover text { fill: white; }

.hint {
  position: absolute;
  top: 10px;
  left: 12px;
  color: #64748B;
  font-size: 11px;
  user-select: none;
  pointer-events: none;
}
.controls {
  position: absolute;
  top: 10px;
  right: 12px;
  display: flex;
  gap: 6px;
}
.ctrl-btn {
  background: #1E293B;
  border: 1px solid #334155;
  color: #CBD5E1;
  border-radius: 6px;
  padding: 5px 10px;
  font-size: 11px;
}
.ctrl-btn:hover { background: #334155; color: white; }
</style>
</head>
<body>
<div id="wrap">
  <div class="hint">💡 Klikni na uzol • Koliesko = zoom • Ťahaj pozadie</div>
  <div class="controls">
    <button class="ctrl-btn" onclick="resetView()">⟲ Reset</button>
  </div>
  <svg id="mm" viewBox="-500 -400 1000 800" preserveAspectRatio="xMidYMid meet">
    <defs>
      <linearGradient id="gradCentral" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#6366F1"/>
        <stop offset="100%" stop-color="#10B981"/>
      </linearGradient>
    </defs>
    <g id="root"></g>
  </svg>
</div>

<script>
const DATA = __DATA__;

// Paletka 6 farieb pre vetvy (pastelovo-sýte)
const COLORS = [
  '#F472B6', // pink
  '#60A5FA', // blue
  '#FBBF24', // amber
  '#34D399', // emerald
  '#A78BFA', // violet
  '#FB923C', // orange
];

const svg  = document.getElementById('mm');
const root = document.getElementById('root');

function ns(tag) { return document.createElementNS('http://www.w3.org/2000/svg', tag); }

function textWidth(txt, fontSize) {
  return Math.max(60, txt.length * fontSize * 0.58 + 24);
}

function makeNode(x, y, label, type, color, onClick) {
  const g = ns('g');
  g.setAttribute('class', 'node-' + type);
  g.setAttribute('transform', `translate(${x}, ${y})`);

  const fs = type === 'central' ? 15 : type === 'topic' ? 12.5 : 11.5;
  const padH = type === 'central' ? 22 : type === 'topic' ? 16 : 12;
  const h = type === 'central' ? 48 : type === 'topic' ? 36 : 28;
  const w = textWidth(label, fs) + (padH - 12);
  const rx = h / 2;

  const rect = ns('rect');
  rect.setAttribute('x', -w/2);
  rect.setAttribute('y', -h/2);
  rect.setAttribute('width', w);
  rect.setAttribute('height', h);
  rect.setAttribute('rx', rx);
  rect.setAttribute('ry', rx);
  if (type === 'topic') {
    rect.setAttribute('fill', color);
    rect.setAttribute('stroke', color);
  } else if (type === 'sub') {
    rect.setAttribute('stroke', color);
  }
  g.appendChild(rect);

  const t = ns('text');
  t.setAttribute('y', fs * 0.35);
  t.textContent = label;
  g.appendChild(t);

  return { g, w, h };
}

function makeLink(x1, y1, x2, y2, color, width) {
  const dx = x2 - x1, dy = y2 - y1;
  // Bezier: kontrolné body pre plynulý ohyb
  const cx1 = x1 + dx * 0.5;
  const cy1 = y1;
  const cx2 = x1 + dx * 0.5;
  const cy2 = y2;
  const path = ns('path');
  path.setAttribute('d', `M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`);
  path.setAttribute('class', 'link');
  path.setAttribute('stroke', color);
  path.setAttribute('stroke-width', width);
  return path;
}

function askTopic(label) {
  // Cez parent window — Streamlit iframe je same-origin
  try {
    const url = new URL(window.top.location.href);
    url.searchParams.set('mm_ask', label);
    window.top.location.href = url.toString();
  } catch(e) {
    // Fallback — otvor vo vlastnom okne
    window.location.href = '?mm_ask=' + encodeURIComponent(label);
  }
}

function render() {
  const N = DATA.branches.length;
  // Polkruhy — vetvy idú vľavo a vpravo od centra
  // Layout: vľavo polovica vetiev, vpravo polovica, rozložené vertikálne
  const halfN = Math.ceil(N / 2);
  const Rtopic = 260;           // vzdialenosť topic uzlov od centra
  const Rsub   = 440;           // vzdialenosť sub uzlov od centra

  // Centrálny uzol
  const centralColor = '#6366F1';
  const central = makeNode(0, 0, DATA.central, 'central', centralColor, null);
  root.appendChild(central.g);

  DATA.branches.forEach((branch, i) => {
    const color = COLORS[i % COLORS.length];
    const isLeft = i % 2 === 1; // striedavo vľavo/vpravo
    const sideIndex = Math.floor(i / 2);
    const sideCount = isLeft
      ? Math.floor(N / 2)
      : Math.ceil(N / 2);

    // Vertikálna pozícia v rámci strany: rozlož rovnomerne
    const spread = 520;
    const yStep  = sideCount > 1 ? spread / (sideCount - 1) : 0;
    const ty     = sideCount > 1 ? (-spread/2) + sideIndex * yStep : 0;
    const tx     = isLeft ? -Rtopic : Rtopic;

    // Link centrum → topic
    const startX = isLeft ? -central.w/2 : central.w/2;
    root.appendChild(makeLink(startX, 0, tx, ty, color, 3));

    // Topic uzol
    const topic = makeNode(tx, ty, branch.topic, 'topic', color,null);
    root.appendChild(topic.g);

    // Sub-uzly pod/nad topic-om
    const subs = branch.subtopics || [];
    const subCount = subs.length;
    const subSpread = 130;
    const subYStep = subCount > 1 ? subSpread / (subCount - 1) : 0;

    subs.forEach((sub, j) => {
      const sy = ty + (subCount > 1 ? (-subSpread/2) + j * subYStep : 0);
      const sx = isLeft ? -Rsub : Rsub;

      // Link topic → sub
      const tStartX = isLeft ? tx - topic.w/2 : tx + topic.w/2;
      root.appendChild(makeLink(tStartX, ty, sx, sy, color, 1.5));

      // Sub uzol
      const subNode = makeNode(sx, sy, sub, 'sub', color,null);
      root.appendChild(subNode.g);
    });
  });
}

// Pan & zoom
let viewBox = { x: -500, y: -400, w: 1000, h: 800 };
const original = { ...viewBox };

function updateViewBox() {
  svg.setAttribute('viewBox',
    `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);
}

svg.addEventListener('wheel', (e) => {
  e.preventDefault();
  const scale = e.deltaY > 0 ? 1.12 : 0.88;
  const pt = svg.createSVGPoint();
  pt.x = e.clientX; pt.y = e.clientY;
  const cursorpt = pt.matrixTransform(svg.getScreenCTM().inverse());
  viewBox.x = cursorpt.x - (cursorpt.x - viewBox.x) * scale;
  viewBox.y = cursorpt.y - (cursorpt.y - viewBox.y) * scale;
  viewBox.w *= scale;
  viewBox.h *= scale;
  updateViewBox();
}, { passive: false });

let isPan = false, panStart = null;
svg.addEventListener('mousedown', (e) => {
  if (e.target.tagName === 'svg' || e.target.tagName === 'path') {
    isPan = true;
    panStart = { x: e.clientX, y: e.clientY, vx: viewBox.x, vy: viewBox.y };
  }
});
window.addEventListener('mousemove', (e) => {
  if (!isPan) return;
  const ctm = svg.getScreenCTM();
  const sx = ctm.a, sy = ctm.d;
  viewBox.x = panStart.vx - (e.clientX - panStart.x) / sx;
  viewBox.y = panStart.vy - (e.clientY - panStart.y) / sy;
  updateViewBox();
});
window.addEventListener('mouseup', () => { isPan = false; });

function resetView() {
  viewBox = { ...original };
  updateViewBox();
}

render();
</script>
</body>
</html>"""
    html = html.replace("__DATA__", data_json)
    components.html(html, height=620, scrolling=False)


# ─── Kartičky ─────────────────────────────────────────────────────────────────
def render_flashcards_tab(nb_id: str):
    import streamlit.components.v1 as components
    import json as _json

    st.markdown("### 🃏 Kartičky")
    

    col1, col2 = st.columns(2)
    with col1:
        count = st.selectbox("Počet kartičiek", [5, 10, 15, 20], index=1)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate = st.button("🔄 Generovať", use_container_width=True, type="primary")

    if generate:
        with st.spinner("Generujem kartičky z dokumentov..."):
            sample = _get_diverse_sample(nb_id, max_chars=12000, seed=None and 0)  # None → nový seed vždy
            # Refresh každý klik
            import time
            sample = _get_diverse_sample(nb_id, max_chars=12000, seed=int(time.time()))

            if not sample:
                st.warning("Žiadne dokumenty nie sú nahrané.")
                return

            prompt = (
                f"Si pedagóg. Z TEXTU nižšie vytvor PRESNE {count} skúšobných kartičiek.\n\n"
                "PRÍSNE PRAVIDLÁ:\n"
                f"• VYTVOR PRESNE {count} kartičiek, nie menej, nie viac.\n"
                "• Vychádzaj VÝHRADNE z TEXTU — nevymýšľaj si.\n"
                "• JAZYK: VŠETKY otázky (front) aj odpovede (back) MUSIA byť v SLOVENČINE.\n"
                "  NEPOUŽÍVAJ angličtinu, iba slovenčinu. Anglické odborné termíny prelož.\n"
                "• front: konkrétna otázka v slovenčine (8–14 slov), musí byť o OBSAHU (definície, "
                "vzorce, rozdiely, postupy, charakteristiky). NIKDY nie o knihe/autorovi/počte kapitol.\n"
                "• back: stručná vecná odpoveď v slovenčine (20–40 slov). Vzorce v LaTeX: "
                "inline $x^2$, display $$...$$.\n"
                "• Každá kartička je o INEJ téme.\n"
                "• PRÍKLADY (iba ilustračné, nekopíruj ak nie sú v texte):\n"
                '  {"front": "Čo je to regresia (regression) podľa dokumentu?",\n'
                '   "back": "Regresia je štatistická metóda na modelovanie vzťahu medzi premennými."}\n'
                '  {"front": "Aký je rozdiel medzi blokovaním (blocking) a náhodným výberom?",\n'
                '   "back": "Blokovanie odstraňuje vplyv rušivých faktorov, náhodný výber zaisťuje reprezentatívnosť."}\n\n'
                "FORMÁT — IBA JSON POLE, nič iné:\n"
                '[{"front": "...?", "back": "..."}, ...]\n\n'
                "TEXT:\n"
                f"{sample}"
            )

            raw = _call_llm_for_tools(prompt, max_tokens=2500, temperature=0.3)

            try:
                m = re.search(r"\[.*\]", raw, re.DOTALL)
                if not m:
                    st.error("Model nevrátil validný JSON.")
                    return
                cards = _json.loads(m.group())

                valid = [
                    c for c in cards
                    if isinstance(c, dict) and "front" in c and "back" in c
                    and len(c["front"].strip()) >= 8 and len(c["back"].strip()) >= 10
                    and "?" in c["front"]
                    # Filter nevhodných patternov
                    and not re.search(r"\b(autor|kniha|kapitol[ya]|softvér|obrázok|strana)\b",
                                      c["front"].lower())
                ]
                if len(valid) < max(3, count // 2):
                    st.error(f"Model vrátil iba {len(valid)} validných kartičiek. Skús znova.")
                    return
                st.session_state.flashcards_result = valid[:count]
            except _json.JSONDecodeError as e:
                st.error(f"JSON chyba: {e}")
                return

    if st.session_state.get("flashcards_result"):
        cards      = st.session_state.flashcards_result
        cards_json = _json.dumps(cards, ensure_ascii=False)
        total      = len(cards)
        cols       = 2
        rows       = (total + cols - 1) // cols
        card_h     = 175
        gap        = 14
        height     = rows * (card_h + gap) + 70

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script>
MathJax = {{
  tex: {{ inlineMath: [['$','$']], displayMath: [['$$','$$']] }},
  startup: {{ typeset: true }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 4px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: {gap}px; }}
.card {{ perspective: 1000px; height: {card_h}px; cursor: pointer; }}
.inner {{ position: relative; width: 100%; height: 100%; transition: transform 0.55s cubic-bezier(.4,0,.2,1); transform-style: preserve-3d; }}
.card.flipped .inner {{ transform: rotateY(180deg); }}
.face {{ position: absolute; width: 100%; height: 100%; backface-visibility: hidden; border-radius: 12px; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 12px 14px; text-align: center; font-size: 0.83rem; line-height: 1.4; overflow: auto; }}
.front {{ background: #1E3A5F; border: 1px solid #3B82F6; color: #BAE6FD; }}
.front .type, .back .type {{ font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.5; margin-bottom: 8px; font-weight: 600; }}
.front .text {{ font-weight: 600; font-size: 0.85rem; }}
.back {{ background: #1a3a2a; border: 1px solid #10B981; color: #A7F3D0; transform: rotateY(180deg); }}
.hint {{ position: absolute; bottom: 7px; right: 10px; font-size: 0.62rem; opacity: 0.3; }}
.counter {{ font-size: 0.72rem; color: #64748B; text-align: right; margin-bottom: 8px; }}
</style>
</head>
<body>
<div class="counter" id="ctr">0 / {total} otočených</div>
<div class="grid" id="grid"></div>
<script>
const cards = {cards_json};
let flipped = 0;
const grid = document.getElementById('grid');
cards.forEach((c) => {{
  const div = document.createElement('div');
  div.className = 'card';
  div.innerHTML = `
    <div class="inner">
      <div class="face front">
        <div class="type">Otázka</div>
        <div class="text">${{c.front}}</div>
        <span class="hint">↺ klikni</span>
      </div>
      <div class="face back">
        <div class="type">Odpoveď</div>
        <div>${{c.back}}</div>
      </div>
    </div>`;
  div.onclick = () => {{
    const was = div.classList.contains('flipped');
    div.classList.toggle('flipped');
    flipped += was ? -1 : 1;
    document.getElementById('ctr').textContent = flipped + ' / {total} otočených';
    if (window.MathJax) MathJax.typesetPromise([div]);
  }};
  grid.appendChild(div);
}});
</script>
</body>
</html>"""
        components.html(html, height=height, scrolling=True)


# ─── Chat panel ───────────────────────────────────────────────────────────────
def render_chat(manager: NotebookManager, nb_id: str):
    import streamlit.components.v1 as components

    nb = manager.get_notebook(nb_id)
    if nb:
        st.markdown(
            f'<div class="app-header">'
            f'<span style="font-size:1.8rem">📚</span>'
            f'<div><h1>Študijný asistent</h1>'
            f'<small style="color:#64748B">Zošit: '
            f'<strong style="color:#818CF8">{nb["name"]}</strong></small>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    history = st.session_state.chat_history

    if not history:
        st.markdown(
            '<div class="info-box" style="text-align:center;padding:20px;margin-bottom:12px;">'
            '🎓 <strong>Študijný asistent ťa tu víta!</strong><br>'
            'Nahraj dokumenty v ľavom paneli a začni klásť otázky.<br>'
            '<small>AI odpovedá <em>výhradne</em> z tvojich nahraných materiálov.</small>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        messages_html = ""
        for msg in history:
            role    = msg["role"]
            content = msg["content"]
            safe = (content
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("\n\n", "</p><p>")
                    .replace("\n", "<br>"))
            safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
            safe = re.sub(r"(?<!\*)\*([^\*]+?)\*(?!\*)", r"<em>\1</em>", safe)

            if role == "user":
                messages_html += f"""
<div class="msg user">
  <div class="label">👤 Ty</div>
  <p>{safe}</p>
</div>"""
            else:
                messages_html += f"""
<div class="msg ai">
  <div class="label">🤖 Study Assistant</div>
  <p>{safe}</p>
</div>"""

        n = len(history)
        height = min(620, max(320, n * 90))

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script>
MathJax = {{
  tex: {{ inlineMath: [['$','$'], ['\\\\(','\\\\)']], displayMath: [['$$','$$']] }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea'] }},
  startup: {{ typeset: true }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
* {{ box-sizing: border-box; margin:0; padding:0; }}
html, body {{ height: 100%; background: #111827; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 14px; color: #F1F5F9; }}
body {{ overflow-y: auto; padding: 8px 6px 12px 6px; scrollbar-width: thin; scrollbar-color: #475569 transparent; }}
body::-webkit-scrollbar {{ width: 4px; }}
body::-webkit-scrollbar-thumb {{ background: #475569; border-radius: 4px; }}
.msg {{ margin-bottom: 10px; }}
.label {{ font-size: 0.68rem; opacity: 0.55; margin-bottom: 4px; padding: 0 4px; }}
.msg.user .label {{ text-align: right; }}
.msg.user > p {{ background: #4F46E5; border-radius: 18px 18px 4px 18px; padding: 10px 14px; margin-left: 20%; color: white; line-height: 1.55; box-shadow: 0 2px 6px rgba(0,0,0,0.3); }}
.msg.ai > p {{ background: #1E3A5F; border: 1px solid #2D4A70; border-radius: 18px 18px 18px 4px; padding: 10px 14px; margin-right: 20%; color: #E2E8F0; line-height: 1.65; box-shadow: 0 2px 6px rgba(0,0,0,0.3); }}
p {{ margin: 0; }} p + p {{ margin-top: 6px; }}
#anchor {{ height: 1px; }}
</style>
</head>
<body>
{messages_html}
<div id="anchor"></div>
<script>document.getElementById('anchor').scrollIntoView();</script>
</body>
</html>"""
        components.html(html, height=height, scrolling=True)

    # Chat input
    if question := st.chat_input(
        placeholder="Opýtaj sa niečo o nahraných dokumentoch...",
        key="chat_input",
    ):
        with st.spinner("🤔 Hľadám odpoveď v dokumentoch..."):
            process_user_message(manager, nb_id, question)
        st.rerun()


# ─── Pravý panel – zdrojové obrázky ──────────────────────────────────────────
def render_source_panel(nb_id: str):
    st.markdown(
        "### 📖 Zdrojové stránky",
        help="Tu uvidíš stránky z dokumentov, z ktorých AI čerpala odpoveď."
    )
    st.markdown("---")

    sources = st.session_state.get("last_sources", [])
    if not sources:
        st.markdown(
            '<div class="info-box" style="text-align:center; padding: 20px;">'
            '🔍<br><strong>Zdrojové stránky</strong><br>'
            '<small>Po položení otázky tu uvidíš stránky dokumentov, '
            'z ktorých AI čerpala odpoveď.</small>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    for src in sources:
        fname    = src["filename"]
        page     = src["page"]
        img_path = src["img_path"]

        st.markdown(
            f'<div class="source-card">'
            f'<h4>📄 {fname} — Strana {page}</h4>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if img_path and Path(img_path).exists():
            st.image(str(img_path), caption=f"{fname} • Strana {page}", use_container_width=True)
        else:
            st.markdown(
                '<div class="warn-box">⚠️ Náhľad stránky nie je dostupný '
                '(podporované iba pre PDF).</div>',
                unsafe_allow_html=True,
            )
        st.markdown("---")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    init_session()
    manager = NotebookManager()

    show_hardware_warning()

    render_sidebar(manager)
    nb_id = st.session_state.active_notebook_id

    # ═══ Handler kliku na myšlienkovú mapu ═══
    mm_ask = st.query_params.get("mm_ask")
    if mm_ask and nb_id:
        question = (
            f"Podrobne vysvetli pojem alebo tému: **{mm_ask}**. "
            f"Uveď definíciu, kľúčové vlastnosti, vzorce (ak existujú) a príklad "
            f"použitia, pokiaľ sú informácie v nahraných dokumentoch."
        )
        if not st.session_state.chat_history:
            st.session_state.chat_history = manager.load_chat_history(nb_id)
        with st.spinner(f"🤔 Hľadám informácie o '{mm_ask}'..."):
            process_user_message(manager, nb_id, question)
        st.session_state.tools_panel_open = False  # prepni na chat
        st.query_params.clear()
        st.rerun()

    if not nb_id:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px;">
            <div style="font-size: 4rem; margin-bottom: 20px;">📚</div>
            <h1 style="background: linear-gradient(90deg, #818CF8, #34D399);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        font-size: 2.5rem; margin-bottom: 12px;">
                Študijný asistent
            </h1>
            <p style="color: #64748B; font-size: 1.1rem; max-width: 500px; margin: 0 auto 30px;">
                Vzdelávacia pomôcka poháňaná umelou inteligenciou.
                Nahraj svoje učebnice, skripty a poznámky — a AI ti pomôže pochopiť látku.
            </p>
            <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
                <div style="background: #1E293B; border: 1px solid #334155;
                            border-radius: 12px; padding: 20px; width: 180px;">
                    <div style="font-size: 2rem">📁</div>
                    <strong style="color: #F1F5F9">1. Zošit</strong>
                    <p style="color: #64748B; font-size: 0.85rem; margin-top: 6px;">
                        Vytvor zošit a nahraj dokumenty
                    </p>
                </div>
                <div style="background: #1E293B; border: 1px solid #334155;
                            border-radius: 12px; padding: 20px; width: 180px;">
                    <div style="font-size: 2rem">💬</div>
                    <strong style="color: #F1F5F9">2. Pýtaj sa</strong>
                    <p style="color: #64748B; font-size: 0.85rem; margin-top: 6px;">
                        Pýtaj sa otázky k dokumentom
                    </p>
                </div>
                <div style="background: #1E293B; border: 1px solid #334155;
                            border-radius: 12px; padding: 20px; width: 180px;">
                    <div style="font-size: 2rem">🔍</div>
                    <strong style="color: #F1F5F9">3.Overuj</strong>
                    <p style="color: #64748B; font-size: 0.85rem; margin-top: 6px;">
                        Over odpovede priamo v zdrojoch
                    </p>
                </div>
                <div style="background: #1E293B; border: 1px solid #334155;
                            border-radius: 12px; padding: 20px; width: 180px;">
                    <div style="font-size: 2rem">🧠</div>
                    <strong style="color: #F1F5F9">4.Nástroje</strong>
                    <p style="color: #64748B; font-size: 0.85rem; margin-top: 6px;">
                        Vytváraj kartičky, kvízy a súhrny
                    </p>
                </div>
            </div>
            <p style="margin-top: 40px; color: #475569; font-size: 0.9rem;">
                👈 Začni vytvorením zošita v ľavom paneli
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    if not st.session_state.chat_history:
        st.session_state.chat_history = manager.load_chat_history(nb_id)

    tools_open = st.session_state.get("tools_panel_open", False)

    toggle_label = "◀ Nástroje" if not tools_open else "▶ Zavrieť"
    if st.button(toggle_label, key="toggle_tools"):
        st.session_state.tools_panel_open = not tools_open
        # Nereznujeme výsledky pri prepínaní, nech si ich užívateľ môže znova pozrieť
        st.rerun()

    if tools_open:
        chat_col, tools_col = st.columns([3, 2], gap="medium")
        with chat_col:
            render_chat(manager, nb_id)
        with tools_col:
            st.markdown("### 🛠️ Nástroje")
            tab1, tab2, tab3 = st.tabs(["📋 Súhrn", "🧠 Myšlienková mapa", "🃏 Kartičky"])
            with tab1:
                render_summary_tab(nb_id)
            with tab2:
                render_mindmap_tab(nb_id)
            with tab3:
                render_flashcards_tab(nb_id)
    else:
        chat_col, source_col = st.columns([3, 2], gap="medium")
        with chat_col:
            render_chat(manager, nb_id)
        with source_col:
            render_source_panel(nb_id)



if __name__ == "__main__":
    main()