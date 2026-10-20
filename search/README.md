# search

Next step: the query script -- free text in, ranked events out, combining
tag/embedding similarity with graph connectivity.

## Usage

```
python search/embed_events.py                      # once: computes and stores the event embeddings
python search/embed_events.py --model text-embedding-ada-002
python search/embed_events.py --model distilbert   # local, 768 dimensions, downloads the model once
python search/testing_query.py --model distilbert  # runs the test queries, saves data/search_runs/
python search/vector_search.py "free techno night in Skopje"
python search/vector_search.py                    # interactive mode
```

Needs `DATABASE_URL` in `.env`, and `OPENAI_API_KEY` for the OpenAI models.
