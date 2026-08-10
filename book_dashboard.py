"""
FAISS Book Search Dashboard — Book Summaries
----------------------------------------------
A professional Streamlit dashboard for the "Book Summaries" FAISS lab.
Deliberately styled differently from the Customer Reviews dashboard
(green/yellow theme, dropdown query picker, KPI cards, extra analytics)
while following the same underlying database-driven pattern.

Run locally:
    streamlit run book_dashboard.py

Deploy: see README (GitHub -> Streamlit Community Cloud).
"""

import random
import time

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="FAISS Book Search Dashboard",
    page_icon="📖",
    layout="wide",
)

# --------------------------------------------------------------------------
# Theme — light green / yellow palette + a professional matplotlib theme
# --------------------------------------------------------------------------
INDEX_COLORS = {
    "KNN (FlatL2)": "#1B5E20",   # deep forest green — ground truth, stands apart
    "IVF": "#66BB6A",            # medium leaf green
    "PQ": "#FBC02D",             # golden yellow
    "IVF+PQ": "#C0CA33",         # lime / olive
    "HNSW": "#AED581",           # light spring green
}
GENRE_COLORS = ["#A5D6A7", "#FFF59D", "#DCE775", "#C5E1A5", "#FDD835"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial", "Helvetica"],
    "axes.edgecolor": "#B0B0B0",
    "axes.labelcolor": "#33421F",
    "axes.labelsize": 10,
    "text.color": "#33421F",
    "xtick.color": "#4A4A4A",
    "ytick.color": "#4A4A4A",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.titlepad": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

st.markdown("""
<style>
.hero-banner {
    background: linear-gradient(90deg, #A8E063 0%, #F9D423 100%);
    padding: 26px 32px;
    border-radius: 14px;
    margin-bottom: 18px;
}
.hero-banner h1 { color: #22350D; margin: 0; font-size: 30px; }
.hero-banner p { color: #33421F; margin: 6px 0 0 0; font-size: 15px; }
div[data-testid="stMetric"] {
    background: #F7FBEF;
    border: 1px solid #D9EBC2;
    border-left: 5px solid #8BC34A;
    border-radius: 10px;
    padding: 10px 14px 4px 14px;
}
</style>
""", unsafe_allow_html=True)


def style_bar_subplot(ax, value_fmt="{:.2f}"):
    """Removes chart junk, adds subtle gridlines behind bars, and labels
    every bar with its value — used across every bar chart in the app."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for bar in ax.patches:
        bar.set_edgecolor("white")
        bar.set_linewidth(1.1)
        height = bar.get_height()
        ax.annotate(
            value_fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center", va="bottom", fontsize=8.5, color="#33421F",
            xytext=(0, 3), textcoords="offset points",
        )
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20, length=0)
    ax.tick_params(axis="y", length=0)
    for label in ax.get_xticklabels():
        label.set_ha("right")


# --------------------------------------------------------------------------
# Step 1 — Database (20 book summaries, 5 genres) — same data as the notebook
# --------------------------------------------------------------------------
BOOK_DATABASE = [
    {"id": 1, "genre": "romance", "title": "Ugly Love", "author": "Colleen Hoover",
     "text": "Ugly Love by Colleen Hoover follows Tate Collins as she falls for her brother's friend Miles Archer, a pilot with strict no-romance rules. The novel alternates between their present-day relationship and Miles's tragic past. As Tate uncovers why he refuses to fall in love again, Colleen Hoover reveals the loss that shaped him."},
    {"id": 2, "genre": "romance", "title": "The Fault in Our Stars", "author": "John Green",
     "text": "In The Fault in Our Stars by John Green, teenage cancer patient Hazel Grace Lancaster meets Augustus Waters at a support group. Their romance takes them from swapped novels to a trip to Amsterdam. John Green follows their bond as illness forces them to confront mortality together."},
    {"id": 3, "genre": "romance", "title": "You Have Reached Sam", "author": "Julie Buxbaum",
     "text": "After her boyfriend Sam dies, Julie keeps calling his old phone number, and he answers. You Have Reached Sam by Julie Buxbaum follows her impossible phone calls with Sam as she works through grief. The novel gradually reveals sides of Sam she never knew while he was alive."},
    {"id": 4, "genre": "romance", "title": "It Ends with Us", "author": "Colleen Hoover",
     "text": "It Ends with Us by Colleen Hoover follows Lily Bloom as she starts a new relationship with surgeon Ryle Kincaid. When patterns from her parents' marriage begin to resurface, Lily must decide whether to break a painful cycle. Flashbacks to her first love, Atlas, run alongside her present-day choices."},
    {"id": 5, "genre": "horror", "title": "Bird Box", "author": "Josh Malerman",
     "text": "In Bird Box by Josh Malerman, an unseen presence drives anyone who looks at it to violence, forcing survivors to move blindfolded. Malorie raises two children in a boarded-up house before attempting a blindfolded river journey to safety. The novel alternates between the outbreak's early days and her desperate escape years later."},
    {"id": 6, "genre": "horror", "title": "Mexican Gothic", "author": "Silvia Moreno-Garcia",
     "text": "Mexican Gothic by Silvia Moreno-Garcia follows Noemi Taboada to a decaying countryside mansion after her cousin sends a disturbing letter. She uncovers a decades-old secret tied to the house's controlling family and its mold-covered walls. Silvia Moreno-Garcia blends gothic atmosphere with body horror as Noemi tries to escape."},
    {"id": 7, "genre": "horror", "title": "The Haunting of Hill House", "author": "Shirley Jackson",
     "text": "The Haunting of Hill House by Shirley Jackson gathers four strangers to investigate a notoriously unsettling mansion. Eleanor Vance, fragile and lonely, becomes increasingly bound to the house as strange events escalate. Shirley Jackson builds dread through atmosphere rather than explicit violence."},
    {"id": 8, "genre": "horror", "title": "Home Before Dark", "author": "Riley Sager",
     "text": "Home Before Dark by Riley Sager follows Maggie Holt, who returns to the house her father wrote about in a bestselling \"true\" horror book. She sets out to separate fact from fiction in his account of their family's brief, frightening stay. Riley Sager splices excerpts from that book within the novel as Maggie digs into what really happened."},
    {"id": 9, "genre": "science fiction", "title": "Project Hail Mary", "author": "Andy Weir",
     "text": "Project Hail Mary by Andy Weir opens with Ryland Grace waking up alone on a spacecraft with no memory of his mission. As his memories return, he realizes he is humanity's last hope against a threat to the sun. Andy Weir follows Grace's efforts to solve the crisis, aided by an unexpected alien ally."},
    {"id": 10, "genre": "science fiction", "title": "The Martian", "author": "Andy Weir",
     "text": "The Martian by Andy Weir follows astronaut Mark Watney after he is stranded alone on Mars, presumed dead by his crew. Using engineering and botany skills, he must find ways to survive until a rescue becomes possible. Andy Weir grounds the story in real science as Watney improvises his way through each setback."},
    {"id": 11, "genre": "science fiction", "title": "Dune", "author": "Frank Herbert",
     "text": "Dune by Frank Herbert follows Paul Atreides after his family takes control of the desert planet Arrakis, source of the galaxy's most valuable resource. Betrayal forces Paul into the desert among the native Fremen, where he begins to embrace a destiny beyond his own control. Frank Herbert builds a vast political and ecological world around Paul's rise."},
    {"id": 12, "genre": "science fiction", "title": "Klara and the Sun", "author": "Kazuo Ishiguro",
     "text": "Klara and the Sun by Kazuo Ishiguro is narrated by Klara, an artificial companion chosen by a young girl named Josie. Klara observes human relationships with careful, outsider curiosity while quietly hoping to help Josie recover from illness. Kazuo Ishiguro uses Klara's perspective to explore devotion, sacrifice, and what it means to be human."},
    {"id": 13, "genre": "thriller", "title": "Never Lie", "author": "Freida McFadden",
     "text": "Never Lie by Freida McFadden follows Tricia and her husband as a snowstorm traps them inside the isolated home of a missing psychiatrist. As they search the house, Tricia begins listening to the psychiatrist's old patient tapes, uncovering secrets tied to the home itself. Freida McFadden alternates between the present search and the psychiatrist's past sessions."},
    {"id": 14, "genre": "thriller", "title": "The Inmate", "author": "Freida McFadden",
     "text": "The Inmate by Freida McFadden follows Brooke Sullivan, a new prison nurse who recognizes an inmate from her past. Her decision to testify against him years earlier now threatens to catch up with her inside the prison's walls. Freida McFadden builds tension around a secret Brooke has hidden from everyone in her new life."},
    {"id": 15, "genre": "thriller", "title": "Verity", "author": "Colleen Hoover",
     "text": "Verity by Colleen Hoover follows struggling writer Lowen Ashleigh, hired to finish a bestselling series after its author, Verity Crawford, is incapacitated. While organizing Verity's office, Lowen finds an unfinished autobiography revealing disturbing secrets. Colleen Hoover leaves readers questioning how much of Verity's confession can be trusted."},
    {"id": 16, "genre": "thriller", "title": "The Silent Patient", "author": "Alex Michaelides",
     "text": "The Silent Patient by Alex Michaelides centers on Alicia Berenson, who stops speaking after allegedly killing her husband. Theo Faber, a criminal psychotherapist, becomes obsessed with uncovering why she went silent. Alex Michaelides builds toward a twist that recontextualizes everything readers thought they knew."},
    {"id": 17, "genre": "memoir", "title": "Educated", "author": "Tara Westover",
     "text": "Educated by Tara Westover recounts her upbringing in a survivalist family in rural Idaho with no formal schooling. She eventually leaves home and pursues a doctorate, navigating the gap between her past and her new life. Tara Westover reflects on family loyalty and the cost of forging her own path."},
    {"id": 18, "genre": "memoir", "title": "Becoming", "author": "Michelle Obama",
     "text": "Becoming by Michelle Obama traces her path from a childhood on Chicago's South Side to her years as First Lady. She reflects candidly on career, marriage, and the personal toll of public life. Michelle Obama's memoir moves chronologically through the experiences that shaped her."},
    {"id": 19, "genre": "memoir", "title": "Wild", "author": "Cheryl Strayed",
     "text": "Wild by Cheryl Strayed follows her solo backpacking journey along the Pacific Crest Trail after the collapse of her marriage and her mother's death. Ill-prepared and grieving, she pushes through physical hardship that mirrors her emotional recovery. Cheryl Strayed weaves memories of loss into the trail's daily challenges."},
    {"id": 20, "genre": "memoir", "title": "Maybe You Should Talk to Someone", "author": "Lori Gottlieb",
     "text": "Maybe You Should Talk to Someone by Lori Gottlieb follows the author, a therapist, as she becomes a patient herself after a painful breakup. The memoir moves between her own therapy sessions and stories from several of her clients. Lori Gottlieb shows how each of their struggles pushes her toward her own breakthroughs."},
]

# 12 preset queries (genre label, query text) — same set used in the notebook
PRESET_QUERIES = [
    ("Romance", "A messy on-and-off romance told through two alternating timelines."),
    ("Romance", "A YA love story about two teenagers dealing with a serious illness."),
    ("Romance", "A story about a girl who keeps calling her deceased boyfriend's old phone number."),
    ("Horror", "A horror novel where the characters must stay blindfolded to survive."),
    ("Horror", "A gothic horror story set in a decaying mansion in Mexico."),
    ("Horror", "A classic haunted house novel focused on psychological dread rather than gore."),
    ("Sci-Fi", "A science fiction book about an astronaut who wakes up with amnesia on a solo space mission."),
    ("Sci-Fi", "An epic science fiction novel about politics and ecology on a desert planet."),
    ("Thriller", "A psychological thriller about a woman trapped in an isolated house during a storm."),
    ("Thriller", "A thriller about a therapist trying to understand a patient who refuses to speak."),
    ("Memoir", "A memoir about growing up without formal schooling and eventually earning a doctorate."),
    ("Author-specific", "Books written by Colleen Hoover."),
]

INDEX_NAMES = ["KNN (FlatL2)", "IVF", "PQ", "IVF+PQ", "HNSW"]
VALID_PQ_M = [4, 8, 12, 16, 24, 32, 48, 64, 96]  # all divide 384 (MiniLM embedding dim)


# --------------------------------------------------------------------------
# Cached resources — model, embeddings, indexes
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading sentence-transformer model (first run only)...")
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner="Encoding the 20 book summaries into embeddings...")
def get_database_and_embeddings():
    df = pd.DataFrame(BOOK_DATABASE)
    model = load_model()
    embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True).astype("float32")
    return df, embeddings


@st.cache_resource(show_spinner="Building FAISS indexes with current parameters...")
def build_indexes(_embeddings, d, nlist, pq_m, pq_nbits, hnsw_m):
    indexes, training_times = {}, {}

    t0 = time.perf_counter()
    idx = faiss.IndexFlatL2(d)
    idx.add(_embeddings)
    training_times["KNN (FlatL2)"] = time.perf_counter() - t0
    indexes["KNN (FlatL2)"] = idx

    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFFlat(quantizer, d, nlist)
    idx.train(_embeddings)
    idx.add(_embeddings)
    idx.nprobe = min(2, nlist)
    training_times["IVF"] = time.perf_counter() - t0
    indexes["IVF"] = idx

    t0 = time.perf_counter()
    idx = faiss.IndexPQ(d, pq_m, pq_nbits)
    idx.train(_embeddings)
    idx.add(_embeddings)
    training_times["PQ"] = time.perf_counter() - t0
    indexes["PQ"] = idx

    t0 = time.perf_counter()
    quantizer2 = faiss.IndexFlatL2(d)
    idx = faiss.IndexIVFPQ(quantizer2, d, nlist, pq_m, pq_nbits)
    idx.train(_embeddings)
    idx.add(_embeddings)
    idx.nprobe = min(2, nlist)
    training_times["IVF+PQ"] = time.perf_counter() - t0
    indexes["IVF+PQ"] = idx

    t0 = time.perf_counter()
    idx = faiss.IndexHNSWFlat(d, hnsw_m)
    idx.add(_embeddings)
    training_times["HNSW"] = time.perf_counter() - t0
    indexes["HNSW"] = idx

    return indexes, training_times


def run_query(query_text, model, indexes, top_k):
    q_vec = model.encode([query_text], convert_to_numpy=True).astype("float32")
    results = {}
    for name, idx in indexes.items():
        t0 = time.perf_counter()
        D, I = idx.search(q_vec, top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results[name] = {"I": I[0], "D": D[0], "latency_ms": elapsed_ms}

    gt_set = set(results["KNN (FlatL2)"]["I"].tolist())
    for name in results:
        approx_set = set(results[name]["I"].tolist())
        overlap = len(gt_set & approx_set)
        results[name]["recall"] = overlap / len(gt_set) if gt_set else 0.0
        results[name]["precision"] = overlap / len(approx_set) if approx_set else 0.0
    return results


# --------------------------------------------------------------------------
# Sidebar — tunable index parameters + extra controls
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Search & Index Settings")
TOP_K = st.sidebar.slider("Top-K (results per query)", min_value=1, max_value=10, value=5,
                           help="How many nearest neighbours each index returns per query.")
NLIST = st.sidebar.slider("IVF nlist (clusters)", min_value=1, max_value=20, value=4,
                           help="Number of inverted-file clusters for IVF and IVF+PQ.")
PQ_M = st.sidebar.select_slider("PQ subquantizers (M)", options=VALID_PQ_M, value=8,
                                 help="Must evenly divide the embedding dimension (384).")
PQ_NBITS = st.sidebar.slider("PQ nbits (bits/subquantizer)", min_value=2, max_value=8, value=4,
                              help="2^nbits centroids per subquantizer.")
HNSW_M = st.sidebar.slider("HNSW M (neighbors/node)", min_value=4, max_value=64, value=16,
                            help="Graph connectivity for the HNSW index.")

st.sidebar.divider()
st.sidebar.subheader("🧩 Extra Options")
selected_indexes = st.sidebar.multiselect(
    "Indexes to compare",
    options=INDEX_NAMES,
    default=INDEX_NAMES,
    help="Choose a subset of indexes to display in results (all 5 are still built and used as ground truth internally).",
)
if not selected_indexes:
    selected_indexes = INDEX_NAMES

st.sidebar.divider()
st.sidebar.caption(
    "Changing any slider automatically rebuilds all 5 indexes with the new "
    "settings (cheap on this 20-book demo database)."
)

# --------------------------------------------------------------------------
# Load model / embeddings / indexes
# --------------------------------------------------------------------------
model = load_model()
df_database, embeddings = get_database_and_embeddings()
d = embeddings.shape[1]
indexes, training_times = build_indexes(embeddings, d, NLIST, PQ_M, PQ_NBITS, HNSW_M)

# --------------------------------------------------------------------------
# Header — hero banner + KPI cards
# --------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
  <h1>📖 FAISS Book Search Dashboard</h1>
  <p>Semantic search & FAISS index benchmarking over a 20-book library across 5 genres</p>
</div>
""", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("📚 Books in Database", len(df_database))
kpi2.metric("🏷️ Genres", df_database["genre"].nunique())
kpi3.metric("🧬 Embedding Dimension", d)
kpi4.metric("🗂️ Indexes Built", len(indexes))

tab_query, tab_benchmark, tab_data = st.tabs(["🔎 Semantic Search", "📊 Benchmark & Analytics", "📖 Book Library"])

# --------------------------------------------------------------------------
# Tab 1 — Search: dropdown of 12 preset queries + random pick + custom prompt
# --------------------------------------------------------------------------
with tab_query:
    if "active_query" not in st.session_state:
        st.session_state.active_query = None

    st.subheader("Choose a query")
    placeholder = "— Select a preset query —"
    query_labels = [placeholder] + [f"{i + 1}. [{g}] {q}" for i, (g, q) in enumerate(PRESET_QUERIES)]
    label_to_text = {f"{i + 1}. [{g}] {q}": q for i, (g, q) in enumerate(PRESET_QUERIES)}

    col_dd, col_rand = st.columns([5, 1])
    with col_dd:
        picked_label = st.selectbox("📋 12 inbuilt queries (grouped by genre)", query_labels)
    with col_rand:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        if st.button("🎲 Random", use_container_width=True):
            picked_label = random.choice(query_labels[1:])
            st.session_state.active_query = label_to_text[picked_label]

    st.subheader("...or type your own prompt")
    with st.form("custom_query_form"):
        custom_query = st.text_input("Custom search prompt", placeholder="e.g. A slow-burn mystery set on a remote island")
        submitted = st.form_submit_button("🔍 Search")
        if submitted:
            if custom_query.strip():
                st.session_state.active_query = custom_query.strip()
            elif picked_label != placeholder:
                st.session_state.active_query = label_to_text[picked_label]

    st.divider()

    if st.session_state.active_query:
        results = run_query(st.session_state.active_query, model, indexes, TOP_K)
        st.markdown(f"**Query:** _{st.session_state.active_query}_")

        cols = st.columns(len(selected_indexes))
        for col, name in zip(cols, selected_indexes):
            r = results[name]
            with col:
                st.markdown(
                    f"<div style='border-left:5px solid {INDEX_COLORS[name]}; padding:6px 10px; "
                    f"background:#F7FBEF; border-radius:6px;'><b>{name}</b></div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"{r['latency_ms']:.4f} ms · Recall {r['recall']:.2f} · Precision {r['precision']:.2f}")
                for i in r["I"]:
                    row = df_database.iloc[int(i)]
                    st.write(f"- **{row['title']}** — {row['author']} _( {row['genre']} )_")

        latencies = [results[name]["latency_ms"] for name in selected_indexes]
        colors = [INDEX_COLORS[name] for name in selected_indexes]

        fig, ax = plt.subplots(figsize=(9, 3.2), dpi=150)
        bars = ax.barh(selected_indexes, latencies, color=colors, edgecolor="white", linewidth=1.1, height=0.62, zorder=3)
        ax.invert_yaxis()
        ax.set_title("Query Latency by Index", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Latency (ms)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color="#E4E4E4", linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for bar, val in zip(bars, latencies):
            ax.annotate(
                f"{val:.4f} ms",
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                ha="left", va="center", fontsize=9, color="#33421F",
                xytext=(6, 0), textcoords="offset points",
            )
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Pick a preset query from the dropdown, hit 🎲 Random, or type your own prompt — then click Search.")

# --------------------------------------------------------------------------
# Tab 2 — Benchmark: run all 12 preset queries, aggregate + extra analytics
# --------------------------------------------------------------------------
with tab_benchmark:
    st.subheader("Run the full 12-query benchmark")
    st.caption(
        "Runs every preset query against all 5 indexes, aggregates timing, and computes "
        "mean Recall@K / Precision@K vs. IndexFlatL2 (ground truth) — the same evaluation "
        "the original notebook performs, plus a couple of extra dashboard-only metrics."
    )

    if st.button("▶️ Run all 12 queries", type="primary"):
        per_index = {name: {"times": [], "recalls": [], "precisions": []} for name in INDEX_NAMES}
        progress = st.progress(0.0, text="Running queries...")

        for qi, (genre, qtext) in enumerate(PRESET_QUERIES):
            results = run_query(qtext, model, indexes, TOP_K)
            for name in INDEX_NAMES:
                per_index[name]["times"].append(results[name]["latency_ms"])
                per_index[name]["recalls"].append(results[name]["recall"])
                per_index[name]["precisions"].append(results[name]["precision"])
            progress.progress((qi + 1) / len(PRESET_QUERIES), text=f"Query {qi + 1}/{len(PRESET_QUERIES)}")

        progress.empty()

        rows = []
        flat_avg_latency = sum(per_index["KNN (FlatL2)"]["times"]) / len(PRESET_QUERIES)
        for name in INDEX_NAMES:
            inference_time_ms = sum(per_index[name]["times"])
            avg_latency_ms = inference_time_ms / len(PRESET_QUERIES)
            rows.append({
                "Index Method": name,
                "Training Time (ms)": round(training_times[name] * 1000, 4),
                "Inference Time (ms)": round(inference_time_ms, 4),
                "Avg Query Latency (ms)": round(avg_latency_ms, 4),
                f"Recall@{TOP_K}": round(float(np.mean(per_index[name]["recalls"])), 3),
                f"Precision@{TOP_K}": round(float(np.mean(per_index[name]["precisions"])), 3),
                "Speedup vs Exact (x)": round(flat_avg_latency / avg_latency_ms, 2) if avg_latency_ms > 0 else float("inf"),
            })

        results_df = pd.DataFrame(rows).set_index("Index Method")
        st.session_state.results_df = results_df

    if "results_df" in st.session_state:
        results_df = st.session_state.results_df

        best_recall = results_df[f"Recall@{TOP_K}"].idxmax()
        fastest = results_df["Avg Query Latency (ms)"].idxmin()
        best_speedup = results_df["Speedup vs Exact (x)"].replace([np.inf], np.nan).idxmax()

        b1, b2, b3 = st.columns(3)
        b1.metric("🏆 Highest Recall", best_recall, f"{results_df.loc[best_recall, f'Recall@{TOP_K}']:.2f}")
        b2.metric("⚡ Fastest Avg Latency", fastest, f"{results_df.loc[fastest, 'Avg Query Latency (ms)']:.4f} ms")
        b3.metric("🚀 Best Speedup vs Exact", best_speedup, f"{results_df.loc[best_speedup, 'Speedup vs Exact (x)']:.2f}x")

        st.dataframe(results_df, use_container_width=True)

        csv = results_df.to_csv().encode("utf-8")
        st.download_button("⬇️ Download results as CSV", csv, "faiss_book_benchmark_results.csv", "text/csv")

        fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), dpi=150)
        fig.suptitle("FAISS Index Performance Comparison — Book Summaries", fontsize=16, fontweight="bold", y=0.99)

        bar_kwargs = dict(kind="bar", width=0.62, zorder=3,
                           color=[INDEX_COLORS[n] for n in results_df.index])

        results_df["Training Time (ms)"].plot(ax=axes[0, 0], **bar_kwargs)
        axes[0, 0].set_title("Training (Build) Time by Index")
        axes[0, 0].set_ylabel("ms")
        style_bar_subplot(axes[0, 0], value_fmt="{:.3f}")

        results_df["Inference Time (ms)"].plot(ax=axes[0, 1], **bar_kwargs)
        axes[0, 1].set_title("Total Inference Time by Index")
        axes[0, 1].set_ylabel("ms")
        style_bar_subplot(axes[0, 1], value_fmt="{:.3f}")

        results_df["Avg Query Latency (ms)"].plot(ax=axes[0, 2], **bar_kwargs)
        axes[0, 2].set_title("Average Query Latency by Index")
        axes[0, 2].set_ylabel("ms")
        style_bar_subplot(axes[0, 2], value_fmt="{:.3f}")

        results_df[f"Recall@{TOP_K}"].plot(ax=axes[1, 0], **bar_kwargs)
        axes[1, 0].set_title(f"Recall@{TOP_K} by Index")
        axes[1, 0].set_ylim(0, 1.15)
        style_bar_subplot(axes[1, 0], value_fmt="{:.2f}")

        results_df[f"Precision@{TOP_K}"].plot(ax=axes[1, 1], **bar_kwargs)
        axes[1, 1].set_title(f"Precision@{TOP_K} by Index")
        axes[1, 1].set_ylim(0, 1.15)
        style_bar_subplot(axes[1, 1], value_fmt="{:.2f}")

        axes[1, 2].axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        st.pyplot(fig)

        st.subheader("Speed vs. Accuracy Trade-off")
        st.caption("Bubble size = index build (training) time. Ideal indexes sit toward the top-left: fast *and* accurate.")
        fig2, ax2 = plt.subplots(figsize=(9, 5), dpi=150)
        for name in INDEX_NAMES:
            x = results_df.loc[name, "Avg Query Latency (ms)"]
            y = results_df.loc[name, f"Recall@{TOP_K}"]
            size = max(training_times[name] * 1000, 0.05) * 350 + 120
            ax2.scatter(x, y, s=size, color=INDEX_COLORS[name], edgecolor="white", linewidth=1.4, alpha=0.9, zorder=3)
            ax2.annotate(name, (x, y), textcoords="offset points", xytext=(10, 6), fontsize=9.5, color="#33421F", fontweight="bold")
        ax2.set_xlabel("Avg Query Latency (ms)")
        ax2.set_ylabel(f"Recall@{TOP_K}")
        ax2.set_ylim(-0.05, 1.2)
        ax2.set_title("Speed vs. Accuracy — All 5 Indexes", fontsize=12, fontweight="bold", pad=12)
        ax2.grid(color="#E4E4E4", linewidth=0.9, zorder=0)
        ax2.set_axisbelow(True)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)
    else:
        st.info("Click **Run all 12 queries** to generate the performance table and charts.")

# --------------------------------------------------------------------------
# Tab 3 — Browse the book database + genre distribution
# --------------------------------------------------------------------------
with tab_data:
    st.subheader("20-book database (5 genres × 4 books)")

    genre_filter = st.multiselect(
        "Filter by genre",
        options=sorted(df_database["genre"].unique()),
        default=sorted(df_database["genre"].unique()),
    )
    filtered = df_database[df_database["genre"].isin(genre_filter)] if genre_filter else df_database

    col_table, col_chart = st.columns([3, 2])
    with col_table:
        st.dataframe(filtered[["id", "title", "author", "genre", "text"]], use_container_width=True, height=520)

    with col_chart:
        genre_counts = df_database["genre"].value_counts()
        fig3, ax3 = plt.subplots(figsize=(5.5, 5.5), dpi=150)
        colors = GENRE_COLORS[: len(genre_counts)]
        wedges, texts, autotexts = ax3.pie(
            genre_counts.values, labels=genre_counts.index, colors=colors,
            autopct="%1.0f%%", startangle=90, pctdistance=0.8,
            wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        )
        for t in texts:
            t.set_fontsize(9.5)
            t.set_color("#33421F")
        for at in autotexts:
            at.set_fontsize(9)
            at.set_color("#22350D")
            at.set_fontweight("bold")
        ax3.set_title("Genre Distribution", fontsize=13, fontweight="bold", pad=14)
        fig3.tight_layout()
        st.pyplot(fig3)
