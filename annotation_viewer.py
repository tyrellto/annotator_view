# app.py ─ a compact, fault-tolerant Streamlit image-flagger
# ──────────────────────────────────────────────────────────
import pandas as pd, streamlit as st
from pathlib import Path
from PIL import Image

# project root = folder containing this file
ROOT     = Path(__file__).resolve().parent
IMG_DIR  = ROOT / "images"           # local images live here
CSV_FILE = ROOT / "metadata.csv"     # must contain column "map"

# ─── pre-scan images once (stem → Path), case-insensitive, any depth/extension
FILE_INDEX = {p.stem.lower(): p                # e.g. "img001": images/pet/IMG001.PNG
              for p in IMG_DIR.rglob("*") if p.is_file()}

# ─── metadata table
@st.cache_data
def meta():
    df = pd.read_csv(CSV_FILE, dtype=str).fillna("")
    if "description" not in df: df["description"] = ""
    if "paper"       not in df: df["paper"]       = ""
    return df
df_meta = meta()

# ─── per-session annotation store
if "annot" not in st.session_state:
    st.session_state.annot = pd.DataFrame(columns=["map","description","paper","flag"])

# ─── resolve helper (url | path | stem lookup) — 5 concise lines
def resolve(img_name:str):
    img_name = img_name.strip()
    if img_name.startswith(("http://","https://")): return img_name      # remote
    p = (IMG_DIR / img_name) if (IMG_DIR / img_name).exists() else None  # explicit rel-path
    return p or FILE_INDEX.get(Path(img_name).stem.lower())              # by stem

st.title("Good / Bad image flagger")

for i,row in df_meta.iterrows():
    path = resolve(row["map"])
    st.subheader(row["map"])
    if not path: st.error("image not found"); st.divider(); continue
    st.image(Image.open(path) if isinstance(path,Path) else path, use_column_width=True)
    st.write(row["description"])
    if row["paper"]: st.markdown(f"[paper link]({row['paper']})")

    choice = st.radio("flag",["✅ OK","❌ Bad","⏭️ skip"],horizontal=True,key=f"k{i}")
    if choice!="⏭️ skip":
        st.session_state.annot.loc[i]=[row["map"],row["description"],row["paper"],choice[2:].strip()]
    st.divider()

out = st.session_state.annot.reset_index(drop=True)
st.data_editor(out, num_rows="dynamic")
st.download_button("📥 download CSV", out.to_csv(index=False).encode(),
                   "annotations.csv","text/csv")
