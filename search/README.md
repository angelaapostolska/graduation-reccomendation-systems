# search

Next step: the query script -- free text in, ranked events out, combining
tag/embedding similarity with graph connectivity.

## Usage

```
python search/embed_events.py                      # once: computes and stores the event embeddings
python search/vector_search.py "free techno night in Skopje"
python search/vector_search.py                    # interactive mode
```

Needs `OPENAI_API_KEY` and `DATABASE_URL` in `.env`.
