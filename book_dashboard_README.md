# FAISS Book Search Dashboard — Book Summaries

A separate, differently-themed dashboard for the **Book Summaries** FAISS lab
(green/yellow visual identity, dropdown query picker, KPI cards, extra
analytics) — distinct from the earlier Customer Reviews dashboard.

**What's different from the Customer Reviews dashboard**
- 🎨 Light green / yellow color theme (`INDEX_COLORS`, `GENRE_COLORS`) instead of the blue/orange/red palette
- 🌈 Gradient hero banner + KPI metric cards at the top
- 📋 12 preset queries as a **dropdown** (grouped by genre in the label) instead of buttons, plus a **🎲 Random** button
- 🧩 Sidebar **multiselect** to choose which indexes to display/compare (extra option)
- 📊 Extra **Speed vs. Accuracy trade-off bubble chart** on the benchmark tab
- 🍩 Genre-distribution **donut chart** + genre filter on the library tab
- 🏆 Benchmark tab shows "best recall / fastest / best speedup" KPI callouts and a **Speedup vs. Exact (x)** column

---

## 1. Run locally first

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

> এই ফোল্ডারে ফাইলগুলো `book_dashboard.py`, `book_dashboard_requirements.txt`
> নামে দেওয়া হয়েছে যাতে আগের Customer Reviews dashboard-এর ফাইলের সাথে গুলিয়ে
> না যায়। নতুন GitHub repo বানানোর সময় এগুলো `app.py` আর `requirements.txt`
> নামে rename করে নাও (দুইটা আলাদা repo, তাই একই নাম ব্যবহার করলেও সমস্যা নেই)।

---

## 2. Push to GitHub (as its own repo)

```bash
mkdir faiss-book-dashboard && cd faiss-book-dashboard
# rename book_dashboard.py -> app.py
# rename book_dashboard_requirements.txt -> requirements.txt
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "FAISS book search dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/faiss-book-dashboard.git
git push -u origin main
```

---

## 3. Deploy on Streamlit Community Cloud

1. [share.streamlit.io](https://share.streamlit.io) → GitHub দিয়ে sign in।
2. **"New app"** → repo, branch (`main`), main file **`app.py`** সিলেক্ট করো।
3. **Deploy** চাপো — প্রথমবার build + model download-এ ২–৫ মিনিট লাগতে পারে।
4. `faiss-cpu` build ফেইল করলে Settings → Python version 3.11-এ সেট করো।

---

## 4. Notes

- Same 20-book / 5-genre dataset as the notebook (`romance`, `horror`,
  `science fiction`, `thriller`, `memoir`), baked directly into `app.py` —
  no external dataset file needed.
- Deselecting an index in the sidebar only hides it from the results/benchmark
  display — `IndexFlatL2` is still always built and used internally as the
  Recall@K / Precision@K ground truth.
- On this 20-vector demo database, FAISS prints harmless
  `"please provide at least N training points"` warnings to the server
  console for IVF/PQ/IVF+PQ — expected, doesn't affect correctness, and
  doesn't show in the Streamlit UI.
