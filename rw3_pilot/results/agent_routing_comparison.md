# Agent Routing Strategy Comparison

Results averaged over 3 random seeds: [42, 123, 456]

## Performance Metrics

| Strategy | Benign Utility | Safety Rate | Overall Risk | Worst-Group Risk |
|----------|----------------|-------------|--------------|------------------|
| no_guard | 1.000±0.000 | 0.000±0.000 | 0.293±0.025 | 0.399±0.052 |
| always_abstain | 0.000±0.000 | 1.000±0.000 | 0.000±0.000 | 0.000±0.000 |
| heuristic | 0.383±0.041 | 0.645±0.040 | 0.103±0.017 | 0.183±0.040 |
| unconditional_cp | 0.907±0.026 | 0.894±0.009 | 0.048±0.002 | 0.085±0.007 |
| graph_conditional_cp | 0.836±0.027 | 0.974±0.011 | 0.036±0.005 | 0.069±0.014 |

## Key Comparisons

- **Utility preservation**: Graph-CP utility drop = 16.4% (vs No-Guard)
- **Worst-group improvement**: Graph-CP improves worst-group risk by 19.2% (vs Unconditional CP)
