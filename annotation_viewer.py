# app.py — two-pane image flagger (rock-solid, button-based, persistent by map)
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd, streamlit as st
from pathlib import Path
from difflib import get_close_matches
import unicodedata

# ─── paths ────────────────────────────────────────────────────────────────────
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

IMG_DIR  = (ROOT / "images").resolve()
CSV_FILE = (ROOT / "metadata.csv").resolve()

# ─── config ───────────────────────────────────────────────────────────────────
EXTS = {".png",".jpg",".jpeg",".webp",".bmp",".gif",".tif",".tiff",".svg"}
st.set_page_config(page_title="Image Flagger", layout="wide")
with st.sidebar:
    AUTO_NEXT = st.checkbox("Auto-advance after marking", value=True)
    DEBUG     = st.checkbox("Debug resolver", value=False)

# ─── helpers: normalization + indexing ────────────────────────────────────────
def normkey(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).strip().strip('"').strip("'").lower()
    return "".join(ch for ch in s if ch.isalnum())

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
    if "map" not in df.columns:
        raise ValueError("metadata.csv must contain a 'map' column")
    for col in ("description","paper"):
        if col not in df.columns: df[col] = ""
    return df

def resolve(raw: str):
    """Return (src, diag). src is str|Path|None."""
    diag = {"input": raw, "tried": []}
    if raw is None: return None, diag
    s = unicodedata.normalize("NFKC", str(raw)).strip().strip('"').strip("'")
    if not s: return None, diag
    # URL
    if s.startswith(("http://","https://","file://")):
        diag["result"] = "url"; return s, diag
    cand = Path(s)
    # absolute
    if cand.is_absolute() and cand.is_file():
        diag["result"] = "abs"; return cand, diag
    diag["tried"].append(f"abs:{cand}")
    # rel CWD
    if cand.is_file():
        diag["result"] = "rel_cwd"; return cand.resolve(), diag
    diag["tried"].append(f"rel_cwd:{cand}")
    # rel ROOT
    q = (ROOT / s)
    if q.is_file():
        diag["result"] = "rel_root"; return q.resolve(), diag
    diag["tried"].append(f"rel_root:{q}")
    # rel IMG_DIR
    q = (IMG_DIR / s)
    if q.is_file():
        diag["result"] = "rel_imgdir"; return q.resolve(), diag
    diag["tried"].append(f"rel_imgdir:{q}")
    # add ext
    if cand.suffix == "":
        for ext in EXTS:
            for base in (IMG_DIR, ROOT):
                q = (base / (s + ext))
                if q.is_file():
                    diag["result"] = f"added_ext:{ext}"; return q.resolve(), diag
                diag["tried"].append(f"added_ext:{q}")
    # index lookup
    key = normkey(cand.stem if cand.suffix else s)
    if key in STEM_INDEX:
        diag["result"] = "stem_index"; diag["key"] = key; return STEM_INDEX[key], diag
    if key in NAME_INDEX:
        diag["result"] = "name_index"; diag["key"] = key; return NAME_INDEX[key], diag
    # fuzzy
    keys = list(STEM_INDEX.keys() | NAME_INDEX.keys())
    diag["suggestions"] = get_close_matches(key, keys, n=5, cutoff=0.6)
    return None, diag

# ─── load data ────────────────────────────────────────────────────────────────
df_meta = meta(CSV_FILE)
N = len(df_meta)
if N == 0:
    st.stop()

# guard against duplicate 'map' values
dup = df_meta["map"][df_meta["map"].duplicated()].tolist()
if dup:
    st.warning(f"Duplicate 'map' values found; using first occurrence: {sorted(set(dup))[:5]}{'...' if len(dup)>5 else ''}")

# fast lookups
idx_by_map = {}
for i, m in enumerate(df_meta["map"]):
    if m not in idx_by_map:
        idx_by_map[m] = i  # keep first

# ─── state: indices & flags persist across reruns ─────────────────────────────
st.session_state.setdefault("paneA_idx", 0)
st.session_state.setdefault("paneB_idx", 1 if N > 1 else 0)
st.session_state.setdefault("flags", {})  # map -> "OK"/"Bad"

# ─── actions (callbacks) ─────────────────────────────────────────────────────
def set_idx(idx_key: str, new_idx: int):
    st.session_state[idx_key] = int(new_idx) % N

def mark_and_maybe_advance(idx_key: str, i: int, map_id: str, flag: str, auto_next: bool):
    st.session_state["flags"][map_id] = flag
    if auto_next:
        st.session_state[idx_key] = (int(i) + 1) % N

def clear_flag(map_id: str):
    st.session_state["flags"].pop(map_id, None)

# ─── pane renderer ────────────────────────────────────────────────────────────
def render_pane(pane_name: str, idx_key: str):
    i = int(st.session_state[idx_key]) % N
    row = df_meta.iloc[i]
    map_id = row["map"]
    src, diag = resolve(map_id)

    st.markdown(f"### {pane_name}")
    st.write(f"**{map_id}**")
    if row["description"]: st.write(row["description"])
    if row["paper"]:       st.markdown(f"[paper link]({row['paper']})")

    if src is None:
        st.error("image not found")
        if DEBUG:
            with st.expander("Why not found? (debug)"):
                st.code(diag, language="json")
    else:
        st.image(str(src), use_container_width=True)

    st.caption(f"Current flag: {st.session_state['flags'].get(map_id, '—')}")

    # buttons (stable keys per pane)
    b1, b2, b3, _sp = st.columns([1,1,1,4])
    with b1:
        st.button("✅ OK", key=f"{pane_name}_ok",
                  on_click=mark_and_maybe_advance,
                  args=(idx_key, i, map_id, "OK", AUTO_NEXT))
    with b2:
        st.button("❌ Bad", key=f"{pane_name}_bad",
                  on_click=mark_and_maybe_advance,
                  args=(idx_key, i, map_id, "Bad", AUTO_NEXT))
    with b3:
        st.button("🧹 Clear", key=f"{pane_name}_clear",
                  on_click=clear_flag, args=(map_id,))

    st.divider()

    # navigation (stable keys, callback changes index)
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

# ─── layout: two independent panes ────────────────────────────────────────────
colA, colB = st.columns(2, gap="large")
with colA:
    render_pane("Window A", "paneA_idx")
with colB:
    render_pane("Window B", "paneB_idx")

# ─── annotations view/export (by map) ─────────────────────────────────────────
st.subheader("Your annotations")
annot_series = pd.Series(st.session_state["flags"], name="flag")
out = df_meta.merge(annot_series, how="left", left_on="map", right_index=True)
flagged = out[out["flag"].notna()].reset_index(drop=True)

st.progress(len(flagged) / max(N,1))
st.data_editor(flagged, num_rows="dynamic", use_container_width=True)
st.download_button("📥 Download CSV",
                   flagged.to_csv(index=False).encode(),
                   "annotations.csv", "text/csv")
