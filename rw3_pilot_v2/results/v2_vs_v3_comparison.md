# v2 vs v3 Key Metrics Comparison (Leak Fix Before/After)

| Combination | Metric | v2 | v3 | Delta | Fix |
|---|---|---|---|---|---|
| clinc150 MiniLM | near Cohen's d (best k*) | 1.74 | 1.71 | -0.03 | L1 |
| clinc150 BGE | near Cohen's d (best k*) | 2.03 | 2.02 | -0.01 | L1 |
| banking77 MiniLM | near Cohen's d (best k*) | 1.11 | 1.05 | -0.06 | L1 |
| banking77 BGE | near Cohen's d (best k*) | 1.25 | 1.14 | -0.11 | L1 |
| clinc150 MiniLM | CP near det. improve (pp) | +8.0 | +5.6 | -2.4 | L1 |
| clinc150 BGE | CP near det. improve (pp) | +31.6 | +30.8 | -0.8 | L1 |
| banking77 MiniLM | CP near det. improve (pp) | +10.0 | +8.5 | -1.5 | L1 |
| banking77 BGE | CP near det. improve (pp) | +11.9 | -1.5 | -13.4 | L1 |
| clinc150 MiniLM | graph increment (near AUROC) | +1.4% | +0.8% | -0.6 | L2 |
| clinc150 BGE | graph increment (near AUROC) | +4.2% | +3.7% | -0.5 | L2 |
| banking77 MiniLM | graph increment (near AUROC) | +8.1% | +4.7% | -3.4 | L2 |
| banking77 BGE | graph increment (near AUROC) | +7.3% | +5.1% | -2.2 | L2 |