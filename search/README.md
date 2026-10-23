# search

Next step: the query script -- free text in, ranked events out, combining
tag/embedding similarity with graph connectivity.

## Final comparison: small against large embedding models

Six models, 20 queries, exact search. Models: `distilbert`, `minilm`, `bge-small` (small); `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large` (large).

```
docker start findzzer-db
python search/embed_events.py --model minilm               # once per model; local ones download the model first
python search/embed_events.py --model bge-small
python search/embed_events.py --model text-embedding-3-large
python search/embed_events.py --model text-embedding-ada-002
python search/embed_events.py --model text-embedding-3-small
python search/embed_events.py --model distilbert
python search/testing_query.py                             # all six models, exact search, saves data/search_runs/results_<model>_exact.txt
python search/pool_report.py                               # pilot report: pooled events per query, overlap, events still to grade
```

Notes:
- `text-embedding-3-large` has 3,072 dimensions. pgvector stores up to 16,000, but indexes (HNSW, IVFFlat) only up to 2,000, so that column gets no index. The search is exact anyway.
- `bge-small` is trained to read a query with the prefix "Represent this sentence for searching relevant passages: ", so the query gets it and events do not (`QUERY_PREFIX` in `embedders.py`).
- `evaluation/grades_first8.csv` holds the 226 grades already given, for the first 8 queries.

## Other usage

```
python search/embed_events.py                      # once: computes and stores the event embeddings
python search/embed_events.py --model text-embedding-ada-002
python search/embed_events.py --model distilbert   # local, 768 dimensions, downloads the model once
python search/testing_query.py --model distilbert  # runs the test queries, saves data/search_runs/
python search/vector_search.py "free techno night in Skopje"
python search/vector_search.py                    # interactive mode
```

Needs `DATABASE_URL` in `.env`, and `OPENAI_API_KEY` for the OpenAI models.
