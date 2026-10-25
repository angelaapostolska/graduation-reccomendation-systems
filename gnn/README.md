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
python gnn/train.py                  # hides genre and event-type links, trains with the final (v2) settings, writes event_gnn.npy
python gnn/search.py --variant gnn
```

Results are saved as `data/search_runs/results_<variant>.txt`, in the same format as the earlier model comparison.

`train.py` reports a held-out check (hit@3 of the hidden genre and event-type links) against a text-neighbour baseline and a most-common baseline. The held-out events are chosen by series (same title at the same venue).

## Final settings (v2)

The defaults of `train.py` are the settings used for every reported result: learning rate 0.0003, patience 100, drift penalty 1.0, up to 400 epochs, seed 7. Running `python gnn/train.py` and then `python gnn/search.py --variant gnn` reproduces them.

The first version (v1) used learning rate 0.001, patience 40, drift penalty 0.1 and 300 epochs. It can still be run with `python gnn/train.py --lr 0.001 --patience 40 --drift 0.1 --epochs 300`. On the first 8 test queries v1 scored 0.24 strict P@10 and v2 scored 0.51.
