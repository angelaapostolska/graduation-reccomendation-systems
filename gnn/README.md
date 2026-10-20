# gnn

Path 2: a graph neural network on the event ontology, compared with text-embedding search.

Steps:

1. `build_graph.py` exports the 12 tables as a heterogeneous graph (done).
2. Shared evaluation set: test queries and graded relevant events (recall@10, nDCG@10, MRR).
3. Text baselines: name + description, and name + description + all fields.
4. Training task and split, including a controlled cold-start split.
5. Train the heterogeneous GNN (event embeddings from text vectors plus graph neighbours).
6. Retrieval with the fused embeddings, then evaluation against the baselines.
7. Ablations: text only, graph only, combined; with and without synthetic extras.

## Usage

```
python gnn/build_graph.py     # reads data/tables/*.csv, writes data/graph/graph.npz and summary.json
```
