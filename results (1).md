Test set: 860 messages (same for every model). Latency: mean over 200 single-message runs.

Hardware: CPU: Intel(R) Xeon(R) CPU @ 2.00GHz | logical cores: 2 | torch threads: 1 | Linux-6.6.159+-x86_64-with-glibc2.39

| model                        |   macro_f1 |   accuracy |   size_mb |   latency_mean_ms |   latency_p95_ms |
|:-----------------------------|-----------:|-----------:|----------:|------------------:|-----------------:|
| Rule: edit-distance lexicon  |     0.661  |     0.7779 |      0    |              1.63 |             3.64 |
| Baseline: TF-IDF + LogReg    |     0.7219 |     0.7767 |      0.92 |              0.95 |             1.2  |
| Neural: Word2Vec + FFN       |     0.681  |     0.7535 |      3.74 |              0.28 |             0.41 |
| Transformer: DistilBERT FP32 |     0.7857 |     0.8349 |    267.86 |             66.67 |            94.68 |
| Quantized: DistilBERT INT8   |     0.7549 |     0.8291 |    138.71 |             33.98 |            59.85 |
