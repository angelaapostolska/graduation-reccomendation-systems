# gnn

Path 2: does adding the event ontology (venue, performer, genre, event type) improve search over text embeddings alone?

Each event gets a new vector built from its text vector and its graph neighbours. The query is still embedded with ada-002 and compared by cosine similarity, so the same queries can be run on the text vectors, the training-free graph vectors and the GNN vectors.

## Usage

```
python gnn/build_graph.py            # tables -> data/graph/graph.npz
python gnn/export_embeddings.py      # ada-002 event vectors -> data/graph/event_text.npy
python gnn/smooth.py                 # training-free graph vectors -> event_smooth.npy
python gnn/search.py --variant text
python gnn/search.py --variant smooth
python gnn/train.py                  # hides genre and event-type links, trains, writes event_gnn.npy
python gnn/search.py --variant gnn
```

Results are saved as `data/search_runs/results_<variant>.txt`, in the same format as the earlier model comparison.

`train.py` reports a held-out check (hit@3 of the hidden genre and event-type links) against a text-neighbour baseline and a most-common baseline. The held-out events are chosen by series (same title at the same venue).
