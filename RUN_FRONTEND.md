# Run the VectorDB Frontend (Windows)

Open this folder in VS Code terminal:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-ui.txt
streamlit run app.py
```

Then open the URL shown in the terminal, normally `http://localhost:8501`.

The app creates `data/documents.db`, stores documents in SQLite, generates embeddings using `all-MiniLM-L6-v2`, and ranks results with cosine similarity. If the embedding model cannot load, it falls back to TF-IDF so the UI can still run.

For a real semantic-search demo, keep the Sentence Transformers dependency installed and allow the model to download on first run.
