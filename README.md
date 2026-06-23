# RAG Aplikácia

Aplikácia implementujúca systém Retrieval Augmented Generation (RAG) v Pythone, vyvinutá ako súčasť diplomovej práce. Aplikácia umožňuje nahrať dokumenty rôznych formátov a klásť k ich obsahu otázky, na ktoré odpovedá lokálne nasadený jazykový model (LLM) na základe relevantného kontextu vyhľadaného v dokumentoch.

## TL;DR

RAG systém s webovým rozhraním (Streamlit), ktorý spracuje PDF/XLSX/DOCX dokumenty, rozdelí ich na chunky (parent-child prístup), vyhľadá relevantný obsah kombináciou BM25 + FAISS s rerankingom a odpoveď generuje lokálny LLM cez Ollama. Vyhľadávací proces je riadený ako stavový graf v LangGraph.

## Hlavné funkcie

- **Spracovanie viacerých formátov dokumentov:** PDF, XLSX, DOCX
- **Dva režimy spracovania PDF:**
  - **Rýchly režim** – extrakcia textu pomocou `pymupdf4llm`
  - **Pomalý, kvalitný režim** – extrakcia pomocou `MinerU` (vyššia presnosť, napr. pri zložitejšom rozložení dokumentu, tabuľkách a podobne)
- **Webové rozhranie** (Streamlit) na nahrávanie dokumentov a komunikáciu s aplikáciou formou otázok a odpovedí
- **Lokálny LLM** – generovanie odpovedí prostredníctvom modelu nasadeného cez Ollama (bez závislosti na externom API)

## Architektúra a spracovanie dát

**1. Chunkovanie**
Po nahratí a extrakcii textu sa dokumenty rozdelia pomocou prístupu **parent-child chunking** – text sa delí na menšie (child) časti vhodné na presné vyhľadávanie, pričom si zároveň udržiava väzbu na väčšie (parent) celky, ktoré poskytujú širší kontext.

**2. Vyhľadávanie (retrieval)**
Vyhľadávanie je implementované ako **stavový graf v LangGraph**. Vyhľadávanie relevantných child chunkov prebieha kombináciou:
- **BM25** (klasické keyword-based vyhľadávanie)
- **FAISS** (vektorové/sémantické vyhľadávanie)

Výsledky z oboch metód sa kombinujú pomocou **RRF (Reciprocal Rank Fusion)**.

**3. Reranking a rozšírenie kontextu**
Po nájdení relevantných child chunkov prebieha reranking výsledkov. Následne sa vybrané child chunky **rozšíria na svoje parent chunky**, čím sa jazykovému modelu poskytne širší a kvalitnejší kontext pre generovanie finálnej odpovede.

**4. Generovanie odpovede**
Na základe vyhľadaného a rozšíreného kontextu generuje odpoveď lokálny LLM model nasadený prostredníctvom **Ollama**.

## Technológie

- **Jazyk:** Python
- **Vektorová databáza:** FAISS
- **Keyword vyhľadávanie:** BM25
- **Orchestrácia / stavový graf:** LangGraph (LangChain)
- **LLM runtime:** Ollama
- **Extrakcia PDF:** pymupdf4llm, MinerU
- **Webové rozhranie:** Streamlit
- **Kontajnerizácia:** Docker
