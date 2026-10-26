## Normal texts, all 20 searches
| Method                 | Strict P@10 | Lenient P@10 | nDCG@10 |
|------------------------|-------------|--------------|---------|
| ada-002                | 0.615       | 0.805        | 0.793   |
| 3-small                | 0.645       | 0.825        | 0.834   |
| 3-large                | 0.645       | 0.850        | 0.837   |
| bge-small              | 0.545       | 0.750        | 0.748   |
| MiniLM                 | 0.460       | 0.655        | 0.667   |
| DistilBERT             | 0.455       | 0.630        | 0.631   |
| GNN, blank cards       | 0.480       | 0.690        | 0.668   |
| GNN, fixed graph       | 0.520       | 0.720        | 0.702   |
| Smoothing, fixed graph | 0.580       | 0.825        | 0.803   |

| Comparison (strict P@10)                      | Mean difference | 95% bootstrap CI | Searches wins / losses | Sign test p |
|-----------------------------------------------|-----------------|------------------|------------------------|-------------|
| GNN, fixed graph minus GNN, blank cards       | +0.040          | [-0.020, +0.100] | 8 / 4                  | 0.388       |
| GNN, fixed graph minus ada-002                | -0.095          | [-0.185, -0.010] | 4 / 9                  | 0.267       |
| GNN, blank cards minus ada-002                | -0.135          | [-0.220, -0.050] | 3 / 11                 | 0.057       |
| Smoothing, fixed graph minus ada-002          | -0.035          | [-0.140, +0.065] | 7 / 7                  | 1.000       |
| Smoothing, fixed graph minus GNN, fixed graph | +0.060          | [-0.055, +0.185] | 9 / 6                  | 0.607       |

Ungraded slots: none.

## Stripped texts, the 10 searches that name a genre or type
| Method                          | Strict P@10 | Lenient P@10 | nDCG@10 | Coverage                                    |
|---------------------------------|-------------|--------------|---------|---------------------------------------------|
| Stripped text (ada-002)         | 0.200       | 0.550        | 0.472   | all graded                                  |
| Stripped smoothing, blank cards | 0.230       | 0.510        | 0.428   | 24 ungraded; strict between 0.230 and 0.470 |
| Stripped GNN, blank cards       | 0.190       | 0.390        | 0.380   | 27 ungraded; strict between 0.190 and 0.460 |
| Stripped smoothing, fixed graph | 0.510       | 0.870        | 0.773   | all graded                                  |
| Stripped GNN, fixed graph       | 0.140       | 0.430        | 0.371   | all graded                                  |
| Normal ada-002 (reference)      | 0.610       | 0.880        | 0.866   | all graded                                  |

| Comparison (strict P@10)                                        | Mean difference | 95% bootstrap CI | Searches wins / losses | Sign test p |
|-----------------------------------------------------------------|-----------------|------------------|------------------------|-------------|
| Stripped smoothing, fixed graph minus Stripped text (ada-002)   | +0.310          | [+0.140, +0.510] | 7 / 1                  | 0.070       |
| Stripped GNN, fixed graph minus Stripped text (ada-002)         | -0.060          | [-0.150, +0.010] | 1 / 4                  | 0.375       |
| Stripped GNN, fixed graph minus Stripped smoothing, fixed graph | -0.370          | [-0.580, -0.180] | 1 / 7                  | 0.070       |
| Stripped text (ada-002) minus Normal ada-002 (reference)        | -0.410          | [-0.630, -0.200] | 0 / 7                  | 0.016       |

Strict P@10 per search (stripped text / stripped smoothing, fixed / stripped GNN, fixed / normal ada-002):
  free techno night in Skopje: 0.0 / 0.4 / 0.0 / 0.4
  cheap standup comedy in Skopje: 0.1 / 0.4 / 0.0 / 1.0
  live jazz concert this weekend: 0.1 / 0.4 / 0.0 / 0.8
  hip-hop party with a student discount: 0.0 / 0.4 / 0.1 / 0.3
  live rock band tonight: 0.7 / 1.0 / 0.3 / 1.0
  stand up comedy in English: 0.0 / 0.0 / 0.0 / 0.0
  karaoke night: 0.0 / 1.0 / 0.0 / 1.0
  salsa dancing lessons: 0.0 / 0.0 / 0.0 / 0.0
  underground electronic music: 0.1 / 0.6 / 0.0 / 0.6
  beach party with DJ: 1.0 / 0.9 / 1.0 / 1.0