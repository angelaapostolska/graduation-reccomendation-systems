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

## Stripped-text test

The event texts copy the genre and type tags ("Techno night at Izlet"), so plain text search has the answer in the text. This test removes the ontology's genre and event-type words from every event text, embeds the stripped texts with ada-002, and runs the same graph scripts on them. The tags stay in the graph, so the graph can supply what the text no longer says.

```
python gnn/stripped_text.py --dry-run     # statistics and before/after examples, no API call
python gnn/stripped_text.py               # embeds the stripped texts, saves data/graph_stripped/ (about $0.03)
GRAPH_DIR=data/graph_stripped python gnn/smooth.py
GRAPH_DIR=data/graph_stripped python gnn/train.py
GRAPH_DIR=data/graph_stripped python gnn/search.py --variant text   --out data/search_runs/results_stripped_text.txt
GRAPH_DIR=data/graph_stripped python gnn/search.py --variant smooth --out data/search_runs/results_stripped_smooth.txt
GRAPH_DIR=data/graph_stripped python gnn/search.py --variant gnn    --out data/search_runs/results_stripped_gnn.txt
```

Words that are also venue types, languages or ordinary words (Bar, Restaurant, Caffe Bar, Other, Macedonian, Gig, Vinyl, Retro, Country, Alternative) are not stripped. Queries are not changed.

## Labelled graph (names of the genre and event-type nodes)

In the first graph, a label node such as "Karaoke Night" started as the average of the vectors of the events that carry it, and the name itself was never turned into a vector. As long as the event texts contain the genre words the average still carries their meaning; once the words are stripped from the texts it carries none. In the labelled graph the genre, event-type, venue-type and city nodes start as the ada-002 vector of their own name.

```
python gnn/node_names.py                 # embeds the 242 label names (a fraction of a cent), saves data/graph/node_names.npz
python gnn/smooth.py --named
python gnn/train.py --named
python gnn/search.py --variant smooth_named --out data/search_runs/results_named_smooth.txt
python gnn/search.py --variant gnn_named    --out data/search_runs/results_named_gnn.txt
```
The same four commands work on the stripped texts with `GRAPH_DIR=data/graph_stripped` in front. The named variants write their own files (`_named`), so the earlier results are not overwritten.
