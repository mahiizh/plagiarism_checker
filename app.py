import re
import time
import difflib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fpdf import FPDF

# ── Config ────────────────────────────────────────────────────────────────────
SBERT_MODEL   = "all-MiniLM-L6-v2"
GROQ_API_KEY  = "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"
SS_API_KEY    = "fdAZIOcioA2M8gg5OM4cs6597lJZ0PUu95AeCEr9"
SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_FIELDS     = "title,abstract,authors,year,isOpenAccess,openAccessPdf,url"
TOP_N_PAPERS  = 10
SCORE_WEIGHTS = {"tfidf": 0.45, "sbert": 0.55}
SS_HEADERS    = {"x-api-key": SS_API_KEY}

# ── Startup ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model() -> SentenceTransformer:
    return SentenceTransformer(SBERT_MODEL)

groq_client = Groq(api_key=GROQ_API_KEY)

# ── Text helpers ──────────────────────────────────────────────────────────────
def extract_pdf(file) -> str:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join(p.get_text() for p in doc)


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def extract_body_text(text: str) -> str:
    words = text.split()
    if len(words) > 600:
        words = words[300:-200]
    elif len(words) > 300:
        words = words[200:]
    return " ".join(words)


# ── Groq keyword extraction ───────────────────────────────────────────────────
def extract_keywords_groq(text: str) -> tuple[str, str]:
    body = extract_body_text(text)
    body = re.sub(r"\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)+\b", "", body)
    body = re.sub(r"\s+", " ", body).strip()

    prompt = (
        "Extract 8 to 10 specific academic topic keywords or keyphrases from the text below. "
        "Focus on domain-specific technical terms, methodologies, and concepts. "
        "Ignore author names, institutions, journal names, and generic words. "
        "Return only a comma-separated list of keywords, nothing else.\n\n"
        f"Text:\n{body[:3000]}"
    )
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100,
        )
        raw = resp.choices[0].message.content.strip()
        keywords = [k.strip().strip("\"'") for k in raw.split(",") if k.strip()]
        query_keywords   = " ".join(keywords[:10])
        display_keywords = ", ".join(keywords[:10])
        return query_keywords, display_keywords
    except Exception as e:
        st.warning(f"Groq keyword extraction failed: {e}. Falling back to first 50 words.")
        words = [w for w in clean(body).split() if len(w) > 4]
        return " ".join(words[:8]), " ".join(words[:8])


# ── Semantic Scholar ──────────────────────────────────────────────────────────
def fetch_papers(query: str) -> list[dict]:
    for attempt in range(3):
        try:
            resp = requests.get(
                SS_SEARCH_URL,
                headers=SS_HEADERS,
                params={"query": query, "limit": TOP_N_PAPERS, "fields": SS_FIELDS},
                timeout=10,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt
                st.warning(f"Rate limited. Retrying in {wait}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json().get("data", [])
        except Exception as e:
            st.error(f"Semantic Scholar API error: {e}")
            return []
    st.error("Rate limit exceeded after retries. Please wait and try again.")
    return []


def fetch_full_text(pdf_url: str) -> str:
    try:
        resp = requests.get(pdf_url, timeout=15)
        resp.raise_for_status()
        doc = fitz.open(stream=resp.content, filetype="pdf")
        return "\n".join(p.get_text() for p in doc)
    except Exception:
        return ""


def get_paper_text(paper: dict) -> tuple[str, str]:
    oa      = paper.get("openAccessPdf") or {}
    pdf_url = oa.get("url", "")
    abstract = paper.get("abstract") or ""
    if pdf_url:
        time.sleep(1)
        full_text = fetch_full_text(pdf_url)
        if full_text.strip():
            return full_text, "full text"
    return abstract, "abstract only"


def get_paper_url(paper: dict) -> str:
    return paper.get("url") or ""


# ── Similarity engines ────────────────────────────────────────────────────────
def tfidf_score(t1: str, t2: str) -> float:
    if not t1.strip() or not t2.strip():
        return 0.0
    try:
        vec = TfidfVectorizer(stop_words=None, token_pattern=r"(?u)\b\w+\b")
        m   = vec.fit_transform([t1, t2])
        return float(cosine_similarity(m[0], m[1])[0][0])
    except ValueError:
        return 0.0


def sbert_doc_score(emb1: np.ndarray, emb2: np.ndarray) -> float:
    dot  = np.dot(emb1, emb2)
    norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    return float(dot / norm) if norm else 0.0


def top_sentence_pairs(
    sents1: list[str], embs1: np.ndarray,
    sents2: list[str], embs2: np.ndarray,
    top_n: int = 5,
) -> list[tuple[str, str, float]]:
    if not sents1 or not sents2 or embs1.size == 0 or embs2.size == 0:
        return []
    sim = cosine_similarity(embs1, embs2)
    pairs = []
    for i, row in enumerate(sim):
        j = int(np.argmax(row))
        pairs.append((sents1[i], sents2[j], float(row[j])))
    return sorted(pairs, key=lambda x: -x[2])[:top_n]


# ── Scoring ───────────────────────────────────────────────────────────────────
def weighted_score(tf: float, sb: float) -> float:
    return SCORE_WEIGHTS["tfidf"] * tf + SCORE_WEIGHTS["sbert"] * sb


def verdict(score: float) -> tuple[str, str]:
    if score >= 0.85:
        return "High Plagiarism", "red"
    elif score >= 0.60:
        return "Moderate Similarity", "orange"
    elif score >= 0.35:
        return "Low Similarity", "goldenrod"
    else:
        return "Likely Original", "green"


# ── UI helpers ────────────────────────────────────────────────────────────────
def scrollable_html(content: str) -> str:
    return (
        f'<div style="height:400px;overflow-y:auto;padding:10px;'
        f'border:1px solid #ddd;border-radius:6px;font-size:0.85rem;line-height:1.7">'
        f'{content}</div>'
    )


def get_important_words(t1: str, t2: str, threshold: float = 0.1) -> set[str]:
    try:
        vec = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b\w+\b")
        tfidf_matrix = vec.fit_transform([t1, t2])
        terms = vec.get_feature_names_out()
        scores_d1 = tfidf_matrix[0].toarray()[0]
        scores_d2 = tfidf_matrix[1].toarray()[0]
        return {
            term for i, term in enumerate(terms)
            if scores_d1[i] > threshold or scores_d2[i] > threshold
        }
    except Exception:
        return set()


def diff_highlight(t1: str, t2: str) -> tuple[str, str]:
    important_words = get_important_words(t1, t2)
    words1 = t1.split()
    words2 = t2.split()
    d      = difflib.SequenceMatcher(None, words1, words2)
    out1, out2 = [], []
    style = "background:#ffd700;color:#1a1a2e;padding:1px 4px;border-radius:3px;font-weight:600"

    for tag, i1, i2, j1, j2 in d.get_opcodes():
        chunk1 = words1[i1:i2]
        chunk2 = words2[j1:j2]
        if tag == "equal":
            for w in chunk1:
                if w.lower() in important_words:
                    out1.append(f'<mark style="{style}">{w}</mark>')
                else:
                    out1.append(f'<span style="color:#aaa">{w}</span>')
            for w in chunk2:
                if w.lower() in important_words:
                    out2.append(f'<mark style="{style}">{w}</mark>')
                else:
                    out2.append(f'<span style="color:#aaa">{w}</span>')
        else:
            for w in chunk1:
                out1.append(f'<span style="color:#ccc">{w}</span>')
            for w in chunk2:
                out2.append(f'<span style="color:#ccc">{w}</span>')

    return " ".join(out1), " ".join(out2)


def sentence_pair_card(s1: str, s2: str, sc: float):
    bg   = "#ffe5e5" if sc > 0.8 else "#fff7e5" if sc > 0.5 else "#f0f0f0"
    font = "#7b0000" if sc > 0.8 else "#7a4f00" if sc > 0.5 else "#333"
    st.markdown(
        f'<div style="background:{bg};color:{font};padding:10px;border-radius:6px;margin-bottom:8px">'
        f'<b>Match: {sc*100:.1f}%</b><br>'
        f'<b>Your doc:</b> {s1}<br>'
        f'<b>Paper:</b> {s2}</div>',
        unsafe_allow_html=True,
    )


def generate_pdf_report(report_text: str):
    report_text = re.sub(r'[^\x00-\x7F]+', '', report_text)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    for line in report_text.split("\n"):
        pdf.cell(0, 8, txt=line, ln=True)
    return pdf.output(dest="S").encode("latin-1")


def doc_input_ui(col, label: str, kp: str) -> str:
    with col:
        st.subheader(label)
        m = st.radio("Input type", ["Paste text", "Upload PDF"], key=f"{kp}_mode", horizontal=True)
        if m == "Paste text":
            return st.text_area("Paste content", height=300, key=f"{kp}_text") or ""
        f = st.file_uploader("Upload PDF", type=["pdf"], key=f"{kp}_pdf")
        if f:
            txt = extract_pdf(f)
            st.success(f"{len(txt.split())} words extracted")
            with st.expander("Preview"):
                st.write(txt[:800] + "…")
            return txt
    return ""


# ── App ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Plagiarism Analyzer", page_icon="🔍", layout="wide")
st.title("🔍 Document Similarity & Plagiarism Analyzer")

with st.spinner("Loading SBERT model…"):
    model = load_model()

mode = st.radio(
    "Select analysis mode",
    ["📄 Doc vs Doc", "🌐 Doc vs Semantic Scholar"],
    horizontal=True,
)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: Doc vs Doc
# ══════════════════════════════════════════════════════════════════════════════
if mode == "📄 Doc vs Doc":

    if "dvd_active_tab" not in st.session_state:
        st.session_state["dvd_active_tab"] = "input"
    if "dvd_run" not in st.session_state:
        st.session_state["dvd_run"] = False

    active = st.session_state["dvd_active_tab"]
    t1, t2, t3 = st.columns(3)
    with t1:
        if st.button("📄 Input",    use_container_width=True,
                     type="primary" if active == "input"    else "secondary"):
            st.session_state["dvd_active_tab"] = "input";    st.rerun()
    with t2:
        if st.button("📊 Analysis", use_container_width=True,
                     type="primary" if active == "analysis" else "secondary"):
            st.session_state["dvd_active_tab"] = "analysis"; st.rerun()
    with t3:
        if st.button("📋 Report",   use_container_width=True,
                     type="primary" if active == "report"   else "secondary"):
            st.session_state["dvd_active_tab"] = "report";   st.rerun()
    st.divider()

    # ── Input ─────────────────────────────────────────────────────────────────
    if active == "input":
        col1, col2 = st.columns(2)
        d1 = doc_input_ui(col1, "Document 1", "d1")
        d2 = doc_input_ui(col2, "Document 2", "d2")
        st.session_state["dvd_d1"] = d1
        st.session_state["dvd_d2"] = d2

        if st.button("▶ Run TF-IDF Analysis", type="primary", disabled=not (d1 and d2)):
            st.session_state["dvd_run"]        = True
            st.session_state["dvd_sbert"]      = None
            st.session_state["dvd_active_tab"] = "analysis"
            st.rerun()

        if not (d1 and d2):
            st.info("Provide both documents to begin.")

    # ── Analysis ──────────────────────────────────────────────────────────────
    elif active == "analysis":
        if not st.session_state.get("dvd_run"):
            st.info("Complete the **Input** tab and click **Run TF-IDF Analysis**.")
        else:
            d1 = st.session_state["dvd_d1"]
            d2 = st.session_state["dvd_d2"]
            c1, c2 = clean(d1), clean(d2)

            with st.spinner("Running TF-IDF…"):
                tf = tfidf_score(c1, c2)
            st.session_state["dvd_tfidf"] = tf

            st.subheader("TF-IDF Result")
            st.metric("TF-IDF Cosine Similarity", f"{tf*100:.1f}%")
            lbl, col = verdict(tf)
            st.markdown(f"**Preliminary Verdict:** <span style='color:{col}'>{lbl}</span>",
                        unsafe_allow_html=True)
            st.divider()

            if st.checkbox("🔬 Run Deep Semantic Analysis (SBERT)", value=False):
                with st.spinner("Running SBERT…"):
                    sents1 = split_sentences(d1)
                    sents2 = split_sentences(d2)
                    embs1  = model.encode(sents1, convert_to_numpy=True) if sents1 else np.array([])
                    embs2  = model.encode(sents2, convert_to_numpy=True) if sents2 else np.array([])
                    de1    = model.encode([c1], convert_to_numpy=True)[0]
                    de2    = model.encode([c2], convert_to_numpy=True)[0]
                    sb     = sbert_doc_score(de1, de2)

                st.session_state.update({
                    "dvd_sbert": sb, "dvd_embs1": embs1,
                    "dvd_embs2": embs2, "dvd_sents1": sents1, "dvd_sents2": sents2,
                })

                final = weighted_score(tf, sb)
                lbl, col = verdict(final)
                m1, m2, m3 = st.columns(3)
                m1.metric("TF-IDF",        f"{tf*100:.1f}%")
                m2.metric("SBERT Semantic", f"{sb*100:.1f}%")
                m3.metric("Weighted Final", f"{final*100:.2f}%")
                st.markdown(f"**Final Verdict:** <span style='color:{col}'>{lbl}</span>",
                            unsafe_allow_html=True)

                chart_df = pd.DataFrame(
                    {"Score (%)": [tf*100, sb*100]},
                    index=["TF-IDF (45%)", "SBERT (55%)"]
                )
                st.bar_chart(chart_df)

                st.subheader("Top Matching Sentence Pairs")
                for s1, s2, sc in top_sentence_pairs(
                    sents1[:50], embs1[:50], sents2[:50], embs2[:50]
                ):
                    sentence_pair_card(s1, s2, sc)

    # ── Report ────────────────────────────────────────────────────────────────
    elif active == "report":
        tf     = st.session_state.get("dvd_tfidf")
        sb     = st.session_state.get("dvd_sbert")
        d1     = st.session_state.get("dvd_d1", "")
        d2     = st.session_state.get("dvd_d2", "")
        sents1 = st.session_state.get("dvd_sents1", [])
        sents2 = st.session_state.get("dvd_sents2", [])
        embs1  = st.session_state.get("dvd_embs1", np.array([]))
        embs2  = st.session_state.get("dvd_embs2", np.array([]))

        if tf is None:
            st.info("Run analysis first.")
        else:
            final = weighted_score(tf, sb) if sb is not None else tf
            lbl, _ = verdict(final)
            st.subheader("📋 Report")
            st.table(pd.DataFrame([
                ("TF-IDF Cosine",  f"{tf*100:.2f}%"),
                ("SBERT Semantic", f"{sb*100:.2f}%" if sb else "Not run"),
                ("Weighted Final", f"{final*100:.2f}%" if sb else "—"),
                ("Verdict",        lbl),
            ], columns=["Metric", "Value"]))

            if d1 and d2:
                st.subheader("🖍 Side-by-Side Diff")
                st.caption("🟡 Yellow = matching words | Dark text for readability")
                h1, h2 = diff_highlight(d1[:3000], d2[:3000])
                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**Document 1**")
                    st.markdown(scrollable_html(h1), unsafe_allow_html=True)
                with cb:
                    st.markdown("**Document 2**")
                    st.markdown(scrollable_html(h2), unsafe_allow_html=True)

            # ── Build PDF report ──────────────────────────────────────────────
            report_lines = [
                "PLAGIARISM REPORT",
                "=" * 50,
                f"TF-IDF Cosine  : {tf*100:.2f}%",
                f"SBERT Semantic : {f'{sb*100:.2f}%' if sb else 'Not run'}",
                f"Final Score    : {final*100:.2f}%",
                f"Verdict        : {lbl}",
            ]

            if sb is not None and sents1 and sents2 and embs1.size > 0 and embs2.size > 0:
                pairs = top_sentence_pairs(sents1[:50], embs1[:50], sents2[:50], embs2[:50])
                if pairs:
                    report_lines += [
                        "",
                        "-" * 50,
                        "TOP MATCHING SENTENCE PAIRS",
                        "-" * 50,
                    ]
                    for idx, (s1, s2, sc) in enumerate(pairs, 1):
                        report_lines += [
                            "",
                            f"Pair {idx}  -  Match: {sc*100:.1f}%",
                            f"  Doc 1 : {s1}",
                            f"  Doc 2 : {s2}",
                        ]

            pdf_data = generate_pdf_report("\n".join(report_lines))
            st.download_button(
                "⬇ Download PDF Report",
                pdf_data,
                file_name="plagiarism_report.pdf",
                mime="application/pdf",
            )


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: Doc vs Semantic Scholar
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.subheader("🌐 Doc vs Semantic Scholar")
    st.caption("Keywords extracted via Groq LLM · Comparison via SBERT · Powered by Semantic Scholar API")

    m = st.radio("Input type", ["Paste text", "Upload PDF"], key="ss_mode", horizontal=True)
    query_text = ""
    if m == "Paste text":
        query_text = st.text_area("Paste document content", height=250, key="ss_text") or ""
    else:
        f = st.file_uploader("Upload PDF", type=["pdf"], key="ss_pdf")
        if f:
            query_text = extract_pdf(f)
            st.success(f"{len(query_text.split())} words extracted")
            with st.expander("Preview"):
                st.write(query_text[:800] + "…")

    if query_text:
        with st.spinner("Extracting keywords via Groq…"):
            query_kw, display_kw = extract_keywords_groq(query_text)
        st.info(f"Keywords: {display_kw}")

    if st.button("▶ Run Analysis", type="primary", disabled=not query_text):

        with st.spinner("Extracting keywords via Groq…"):
            query_kw, display_kw = extract_keywords_groq(query_text)

        with st.spinner(f"Querying Semantic Scholar for: {query_kw}…"):
            papers = fetch_papers(query_kw)

        if not papers:
            st.error("No papers returned. Check your connection or API key.")
            st.stop()

        st.success(f"Fetched {len(papers)} papers from Semantic Scholar.")

        with st.spinner("Encoding your document…"):
            q_clean   = clean(query_text)
            q_sents   = split_sentences(query_text)
            q_embs    = model.encode(q_sents, convert_to_numpy=True) if q_sents else np.array([])
            q_doc_emb = model.encode([q_clean], convert_to_numpy=True)[0]

        results = []
        prog = st.progress(0, text="Fetching and analyzing papers…")
        for i, paper in enumerate(papers):
            text, source = get_paper_text(paper)
            if text.strip():
                p_clean   = clean(text)
                p_sents   = split_sentences(text)
                p_embs    = model.encode(p_sents, convert_to_numpy=True) if p_sents else np.array([])
                p_doc_emb = model.encode([p_clean], convert_to_numpy=True)[0]
                sb        = sbert_doc_score(q_doc_emb, p_doc_emb)
                pairs     = top_sentence_pairs(q_sents[:50], q_embs[:50], p_sents[:50], p_embs[:50])
            else:
                sb, pairs = 0.0, []

            results.append({
                "paper":  paper,
                "source": source,
                "sbert":  sb,
                "pairs":  pairs,
            })
            prog.progress((i + 1) / len(papers), text=f"Analyzed: {paper.get('title','')[:60]}…")

        results.sort(key=lambda x: -x["sbert"])
        st.session_state["ss_results"] = results

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.get("ss_results"):
        results = st.session_state["ss_results"]

        st.subheader("📊 Results — Ranked by SBERT Similarity")

        summary_rows = []
        for r in results:
            paper   = r["paper"]
            authors = paper.get("authors", [])
            author_names = ", ".join(
                a.get("name", "") if isinstance(a, dict) else str(a)
                for a in authors[:3]
            )
            summary_rows.append({
                "Paper":     paper.get("title", "Unknown")[:55],
                "Year":      paper.get("year", "N/A"),
                "Authors":   author_names,
                "SBERT (%)": round(r["sbert"] * 100, 2),
                "Link":      get_paper_url(paper),
            })

        st.data_editor(
            pd.DataFrame(summary_rows),
            column_config={
                "Link": st.column_config.LinkColumn("Link", display_text="🔗 Open")
            },
            use_container_width=True,
            hide_index=True,
            disabled=True,
        )

        chart_df = pd.DataFrame(
            {"SBERT Similarity (%)": [r["sbert"]*100 for r in results]},
            index=[r["paper"].get("title", "")[:45] for r in results]
        )
        st.bar_chart(chart_df)

        for r in results:
            paper      = r["paper"]
            lbl, color = verdict(r["sbert"])
            title      = paper.get("title", "Unknown")
            year       = paper.get("year", "N/A")
            authors    = paper.get("authors", [])
            author_str = ", ".join(
                a.get("name", "") if isinstance(a, dict) else str(a)
                for a in authors[:3]
            )
            ss_url  = get_paper_url(paper)
            oa      = paper.get("openAccessPdf") or {}
            pdf_url = oa.get("url", "")

            with st.expander(f"📄 {title[:70]}  —  {r['sbert']*100:.1f}%  {lbl}"):
                st.markdown(f"**Year:** {year} &nbsp;|&nbsp; **Authors:** {author_str}")
                st.markdown(f"**Text compared against:** {r['source']}")
                if ss_url:
                    st.markdown(f"🔗 [View on Semantic Scholar]({ss_url})")
                if pdf_url:
                    st.markdown(f"📥 [Download PDF]({pdf_url})")
                st.markdown(
                    f"**Verdict:** <span style='color:{color}'>{lbl}</span>",
                    unsafe_allow_html=True,
                )
                st.metric("SBERT Similarity", f"{r['sbert']*100:.1f}%")
                if r["pairs"]:
                    st.markdown("**Top Matching Sentence Pairs:**")
                    for s1, s2, sc in r["pairs"]:
                        sentence_pair_card(s1, s2, sc)

        # ── Build PDF report ──────────────────────────────────────────────────
        report_lines = [
            "SEMANTIC SCHOLAR PLAGIARISM REPORT",
            "=" * 50,
            "",
        ]
        for r in results:
            lbl, _ = verdict(r["sbert"])
            title  = r["paper"].get("title", "Unknown")
            link   = get_paper_url(r["paper"])

            report_lines += [
                f"Paper  : {title}",
                f"SBERT  : {r['sbert']*100:.1f}%  |  {lbl}",
                f"Source : {r['source']}",
                f"Link   : {link}",
            ]

            if r["pairs"]:
                report_lines.append("  Top Matching Sentence Pairs:")
                for idx, (s1, s2, sc) in enumerate(r["pairs"], 1):
                    report_lines += [
                        "",
                        f"  Pair {idx}  -  Match: {sc*100:.1f}%",
                        f"    Your doc : {s1}",
                        f"    Paper    : {s2}",
                    ]

            report_lines.append("-" * 50)

        pdf_data = generate_pdf_report("\n".join(report_lines))
        st.download_button(
            "⬇ Download PDF Report",
            pdf_data,
            file_name="ss_plagiarism_report.pdf",
            mime="application/pdf",
        )
