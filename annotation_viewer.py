# app.py — two-pane image flagger (zero top gap, fixed-shape panes, comments)
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd, streamlit as st
from pathlib import Path
from difflib import get_close_matches
import unicodedata, re
from urllib.parse import urlparse

# ─── paths ────────────────────────────────────────────────────────────────────
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

IMG_DIR  = (ROOT / "images").resolve()
CSV_FILE = (ROOT / "metadata_v2.csv").resolve()

# ─── config ───────────────────────────────────────────────────────────────────
EXTS = {".png",".jpg",".jpeg",".webp",".bmp",".gif",".tif",".tiff",".svg"}
st.set_page_config(page_title="Image Flagger", layout="wide")

# ✂️ Kill extra vertical space above the first content block and inside markdown
st.markdown(
    """
    <style>
    /* remove Streamlit's default top spacing */
    [data-testid="stHeader"] { height: 0px; }
    [data-testid="stAppViewContainer"] > .main .block-container {
        padding-top: 0rem; padding-bottom: 0rem;
    }
    /* remove vertical padding between blocks */
    [data-testid="stVerticalBlock"] { padding-top: 0rem; padding-bottom: 0rem; }
    /* tighten markdown spacing */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown p { margin: 0.1rem 0; }
    .stMarkdown ul { margin: 0.1rem 0; }
    .stMarkdown li { margin: 0.05rem 0; }
    /* fixed-shape pane */
    .pane {
        height: var(--pane-h);
        display: flex;
        flex-direction: column;
        border: 1px solid rgba(0,0,0,.08);
        border-radius: 8px;
        padding: 8px 10px;
        box-sizing: border-box;
        margin: 0;   /* no extra outer gap */
    }
    .pane-header {
        font-weight: 600;
        font-size: 0.95rem;
        line-height: 1.2;
        margin: 0 0 6px 0;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .pane-body {
        flex: 1 1 auto;
        min-height: 0; /* required so overflow works in flex box */
        overflow: auto;     /* scroll INSIDE the pane, not the page */
        padding-right: 4px; /* avoid scrollbar overlaying text */
    }
    .pane-footer {
        flex: 0 0 var(--pane-footer-h);
        display: flex;
        flex-direction: column;
        gap: 6px;
        margin-top: 6px;
    }
    .pane-footer .stButton>button { width: 100%; }
    /* keep links from stretching layout */
    .stMarkdown a { word-break: break-word; overflow-wrap: anywhere; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    AUTO_NEXT = st.checkbox("Auto-advance after marking", value=True)
    DEBUG     = st.checkbox("Debug resolver", value=False)
    # Height of each pane; total is constant and never reflows
    PANE_HEIGHT_PX = st.number_input("Pane height (px)", 600, 3000, 820, 20)
    # Reserve space for comment + buttons + nav; body auto-scrolls
    FOOTER_HEIGHT_PX = st.number_input("Footer reserve (px)", 120, 360, 210, 10)

# inject CSS variables for pane sizing
st.markdown(
    f"""
    <style>
    :root {{
        --pane-h: {int(PANE_HEIGHT_PX)}px;
        --pane-footer-h: {int(FOOTER_HEIGHT_PX)}px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── helpers: normalization + indexing ────────────────────────────────────────
def normkey(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).strip().strip('"').strip("'").lower()
    return "".join(ch for ch in s if ch.isalnum())

def show_or_none(val: str) -> str:
    s = unicodedata.normalize("NFKC", str(val)).strip()
    return s if s else "None"

def show_img(src):
    try:
        st.image(str(src), use_container_width=True)
    except TypeError:
        st.image(str(src))

def bordered_container():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()

@st.cache_data(show_spinner=False)
def build_indexes(img_dir: Path):
    stem_index, name_index = {}, {}
    if not img_dir.exists():
        return stem_index, name_index
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            stem_index.setdefault(normkey(p.stem), p)
            name_index.setdefault(normkey(p.name.rsplit(".",1)[0]), p)
    return stem_index, name_index

STEM_INDEX, NAME_INDEX = build_indexes(IMG_DIR)

@st.cache_data(show_spinner=False)
def meta(csv_path: Path):
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]
    if "map" not in df.columns:
        raise ValueError("metadata.csv must contain a 'map' column")
    for col in ["description","demographics","n","age","tags","paper1","paper2"]:
        if col not in df.columns:
            df[col] = ""
    if "paper" in df.columns:
        take = df["paper1"].str.strip() == ""
        df.loc[take, "paper1"] = df.loc[take, "paper"].fillna("")
    return df

def resolve(raw: str):
    from urllib.parse import urlparse
    diag = {"input": raw, "tried": []}
    if raw is None: return None, diag
    s = unicodedata.normalize("NFKC", str(raw)).strip().strip('"').strip("'")
    if not s: return None, diag
    if s.startswith(("http://","https://","file://")):
        diag["result"] = "url"; return s, diag
    cand = Path(s)
    if cand.is_absolute() and cand.is_file():
        diag["result"] = "abs"; return cand, diag
    diag["tried"].append(f"abs:{cand}")
    if cand.is_file():
        diag["result"] = "rel_cwd"; return cand.resolve(), diag
    diag["tried"].append(f"rel_cwd:{cand}")
    q = (ROOT / s)
    if q.is_file():
        diag["result"] = "rel_root"; return q.resolve(), diag
    diag["tried"].append(f"rel_root:{q}")
    q = (IMG_DIR / s)
    if q.is_file():
        diag["result"] = "rel_imgdir"; return q.resolve(), diag
    diag["tried"].append(f"rel_imgdir:{q}")
    if cand.suffix == "":
        for ext in EXTS:
            for base in (IMG_DIR, ROOT):
                q = (base / (s + ext))
                if q.is_file():
                    diag["result"] = f"added_ext:{ext}"; return q.resolve(), diag
                diag["tried"].append(f"added_ext:{q}")
    key = normkey(cand.stem if cand.suffix else s)
    if key in STEM_INDEX:
        diag["result"] = "stem_index"; diag["key"] = key; return STEM_INDEX[key], diag
    if key in NAME_INDEX:
        diag["result"] = "name_index"; diag["key"] = key; return NAME_INDEX[key], diag
    from difflib import get_close_matches
    keys = list(STEM_INDEX.keys() | NAME_INDEX.keys())
    diag["suggestions"] = get_close_matches(key, keys, n=5, cutoff=0.6)
    return None, diag

# --- robust link parsing for papers (Python 3.13-safe) ---
_URL_RE    = re.compile(r'(https?://[^\s\]\)>,;]+)', re.I)
_DOI_RE    = re.compile(r'(?:doi:\s*|DOI:\s*)?(10\.\d{4,9}/[^\s\]\)>,;]+)')
_ARXIV_RE  = re.compile(r'arxiv:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)', re.I)
_PMID_RE   = re.compile(r'pmid:\s*(\d+)', re.I)
_PMCID_RE  = re.compile(r'pmcid:\s*(PMC\d+)', re.I)

def _dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def parse_links_field(s: str) -> list[str]:
    """Extract links from free text (URLs, DOIs, arXiv, PubMed/PMC)."""
    if not s:
        return []
    text = str(s)

    links = []
    # 1) Explicit URLs
    for m in _URL_RE.findall(text):
        links.append(m.rstrip(').,;]'))

    # 2) DOIs (with/without 'doi:')
    for m in _DOI_RE.findall(text):
        links.append(f"https://doi.org/{m}")

    # 3) arXiv IDs
    for m in _ARXIV_RE.findall(text):
        links.append(f"https://arxiv.org/abs/{m}")

    # 4) PubMed / PMC IDs
    for m in _PMID_RE.findall(text):
        links.append(f"https://pubmed.ncbi.nlm.nih.gov/{m}/")
    for m in _PMCID_RE.findall(text):
        links.append(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{m}/")

    return _dedup(links)


def parse_links_field(s: str) -> list[str]:
    if not s: return []
    text = str(s)
    links = []
    for m in _URL_RE.findall(text):
        links.append(m.rstrip(').,;]'))
    for m in _DOI_RE.findall(text):
        links.append(f"https://doi.org/{m}")
    for m in _ARXIV_RE.findall(text):
        links.append(f"https://arxiv.org/abs/{m}")
    for m in _PMID_RE.findall(text):
        links.append(f"https://pubmed.ncbi.nlm.nih.gov/{m}/")
    for m in _PMCID_RE.findall(text):
        links.append(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{m}/")
    return _dedup(links)

def links_markdown_list(*fields: str, label_style: str = "url", numbered: bool = False) -> str:
    all_links = []
    for f in fields:
        all_links.extend(parse_links_field(f))
    all_links = _dedup(all_links)
    if not all_links: return "None"
    lines = []
    if numbered:
        for k, u in enumerate(all_links, 1):
            label = u if label_style == "url" else (urlparse(u).netloc.replace("www.", "") or u)
            lines.append(f"{k}. [{label}]({u})")
    else:
        for u in all_links:
            label = u if label_style == "url" else (urlparse(u).netloc.replace("www.", "") or u)
            lines.append(f"- [{label}]({u})")
    return "\n".join(lines)

# ─── load data ────────────────────────────────────────────────────────────────
df_meta = meta(CSV_FILE)
N = len(df_meta)
if N == 0: st.stop()

dup = df_meta["map"][df_meta["map"].duplicated()].tolist()
if dup:
    st.warning(f"Duplicate 'map' values found; using first occurrence: {sorted(set(dup))[:5]}{'...' if len(dup)>5 else ''}")

idx_by_map = {}
for i, m in enumerate(df_meta["map"]):
    if m not in idx_by_map:
        idx_by_map[m] = i

# state
st.session_state.setdefault("paneA_idx", 0)
st.session_state.setdefault("paneB_idx", 1 if N > 1 else 0)
st.session_state.setdefault("flags", {})
st.session_state.setdefault("notes", {})

# actions
def set_idx(idx_key: str, new_idx: int):
    st.session_state[idx_key] = int(new_idx) % N

def mark_and_maybe_advance(idx_key: str, i: int, map_id: str, flag: str, auto_next: bool):
    st.session_state["flags"][map_id] = flag
    if auto_next:
        st.session_state[idx_key] = (int(i) + 1) % N

def clear_flag(map_id: str):
    st.session_state["flags"].pop(map_id, None)

def save_comment(map_id: str, key: str):
    st.session_state["notes"][map_id] = st.session_state.get(key, "").strip()

# pane renderer
def render_pane(pane_name: str, idx_key: str):
    i = int(st.session_state[idx_key]) % N
    row = df_meta.iloc[i]
    map_id = row["map"]
    src, diag = resolve(map_id)

    st.markdown(f"<div class='pane'>", unsafe_allow_html=True)
    st.markdown(f"<div class='pane-header'>{pane_name}: <strong>{map_id}</strong></div>", unsafe_allow_html=True)
    st.markdown("<div class='pane-body'>", unsafe_allow_html=True)

    desc = show_or_none(row.get("description", ""))
    demo = show_or_none(row.get("demographics", ""))
    n    = show_or_none(row.get("n", ""))
    age  = show_or_none(row.get("age", ""))
    tags = show_or_none(row.get("tags", ""))

    papers_md = links_markdown_list(row.get("paper1",""), row.get("paper2",""), label_style="url", numbered=True)

    with bordered_container():
        st.markdown(
            f"**Description:** {desc}  \n"
            f"**Demographics:** {demo}  \n"
            f"**N:** {n}  \n"
            f"**Age:** {age}  \n"
            f"**Tags:** {tags}  \n"
        )
        st.markdown("**Papers:**")
        st.markdown(papers_md)

    if src is None:
        st.error("image not found")
        if DEBUG:
            with st.expander("Why not found? (debug)"):
                st.json(diag)
    else:
        show_img(src)

    # avoid captions (they add space variability)
    # st.caption(f"Current flag: {st.session_state['flags'].get(map_id, '—')}")

    st.markdown("</div>", unsafe_allow_html=True)  # end body

    st.markdown("<div class='pane-footer'>", unsafe_allow_html=True)

    ckey = f"{pane_name}_comment_{map_id}"
    st.text_input("Comment (optional)",
                  value=st.session_state["notes"].get(map_id, ""),
                  key=ckey,
                  on_change=save_comment,
                  args=(map_id, ckey))

    b1, b2, b3, _sp = st.columns([2, 2, 2, 6])
    def wide_button(label, **kwargs):
        try:
            return st.button(label, use_container_width=True, **kwargs)
        except TypeError:
            return st.button(label, **kwargs)

    with b1:
        wide_button("✅ OK",  key=f"{pane_name}_ok",
                    on_click=mark_and_maybe_advance,
                    args=(idx_key, i, map_id, "OK", AUTO_NEXT))
    with b2:
        wide_button("❌ Bad", key=f"{pane_name}_bad",
                    on_click=mark_and_maybe_advance,
                    args=(idx_key, i, map_id, "Bad", AUTO_NEXT))
    with b3:
        wide_button("🧹 Clear", key=f"{pane_name}_clear",
                    on_click=clear_flag, args=(map_id,))

    c1, c2, c3 = st.columns([1,1,3])
    with c1:
        st.button("⬅ Prev", key=f"{pane_name}_prev",
                  on_click=set_idx, args=(idx_key, i-1))
    with c2:
        st.button("Next ➡", key=f"{pane_name}_next",
                  on_click=set_idx, args=(idx_key, i+1))
    with c3:
        jump_map = st.selectbox("Jump to map",
                                options=df_meta["map"].tolist(),
                                index=i,
                                key=f"{pane_name}_jump_select")
        st.button("Go", key=f"{pane_name}_jump_go",
                  on_click=set_idx, args=(idx_key, idx_by_map.get(jump_map, i)))

    st.markdown("</div>", unsafe_allow_html=True)  # footer
    st.markdown("</div>", unsafe_allow_html=True)  # pane

# layout (no extra headings above panes)
colA, colB = st.columns(2, gap="large")
with colA: render_pane("Window A", "paneA_idx")
with colB: render_pane("Window B", "paneB_idx")

# annotations/export
st.markdown("### Your annotations")
annot_series   = pd.Series(st.session_state["flags"], name="flag")
comment_series = pd.Series(st.session_state["notes"], name="comment")
out = df_meta.merge(annot_series, how="left", left_on="map", right_index=True)
out = out.merge(comment_series, how="left", left_on="map", right_index=True)
mask = out["flag"].notna() | out["comment"].fillna("").astype(str).str.len().gt(0)
flagged = out[mask].reset_index(drop=True)
st.progress(len(flagged) / max(N,1))
st.data_editor(flagged, num_rows="dynamic", use_container_width=True)
st.download_button("📥 Download CSV",
                   flagged.to_csv(index=False).encode(),
                   "annotations.csv", "text/csv")
