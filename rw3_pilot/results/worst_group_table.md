# Worst-Group Risk Analysis

## By Tool Type

| Group | Risk Rate | Std | N Traces | Success Rate | Avg Max Risk |
|-------|-----------|-----|----------|--------------|---------------|
| banking_client | 0.316 | 0.466 | 266 | 0.827 | 0.650 |
| calendar_client | 0.276 | 0.448 | 315 | 0.924 | 0.435 |
| cloud_drive | 0.302 | 0.460 | 305 | 0.892 | 0.467 |
| email_client | 0.314 | 0.464 | 641 | 0.900 | 0.489 |
| slack | 0.295 | 0.457 | 241 | 0.880 | 0.474 |
| travel_booking | 0.353 | 0.479 | 167 | 0.844 | 0.550 |
| web_browse | 0.246 | 0.434 | 65 | 0.892 | 0.603 |

**Statistical Tests:**
- Kruskal-Wallis H: 4.64, p = 5.91e-01
- Worst group: travel_booking (risk = 0.353)
- Best group: web_browse (risk = 0.246)
- **Worst/Average ratio: 1.18**
- Cohen's d (worst vs best): 0.23

## By Attack Type

| Group | Risk Rate | Std | N Traces | Success Rate | Avg Max Risk |
|-------|-----------|-----|----------|--------------|---------------|
| dos | 0.745 | 0.437 | 153 | 0.843 | 0.691 |
| injection | 0.912 | 0.285 | 147 | 0.762 | 0.930 |
| none | 0.053 | 0.224 | 1398 | 0.928 | 0.349 |
| prompt_manipulation | 0.912 | 0.284 | 148 | 0.750 | 0.914 |
| tool_knowledge | 0.994 | 0.081 | 154 | 0.792 | 0.948 |

**Statistical Tests:**
- Kruskal-Wallis H: 1415.05, p = 3.76e-305
- Worst group: tool_knowledge (risk = 0.994)
- Best group: none (risk = 0.053)
- **Worst/Average ratio: 1.37**
- Cohen's d (worst vs best): 5.59

## By Tool Risk Category

| Group | Risk Rate | Std | N Traces | Success Rate | Avg Max Risk |
|-------|-----------|-----|----------|--------------|---------------|
| high_risk | 0.319 | 0.467 | 498 | 0.841 | 0.611 |
| low_risk | 0.276 | 0.448 | 315 | 0.924 | 0.435 |
| medium_risk | 0.307 | 0.461 | 1187 | 0.894 | 0.480 |

**Statistical Tests:**
- Kruskal-Wallis H: 1.73, p = 4.22e-01
- Worst group: high_risk (risk = 0.319)
- Best group: low_risk (risk = 0.276)
- **Worst/Average ratio: 1.06**
- Cohen's d (worst vs best): 0.09

## By Agent Suite

| Group | Risk Rate | Std | N Traces | Success Rate | Avg Max Risk |
|-------|-----------|-----|----------|--------------|---------------|
| banking | 0.334 | 0.472 | 491 | 0.833 | 0.614 |
| slack | 0.303 | 0.460 | 525 | 0.886 | 0.482 |
| travel | 0.297 | 0.457 | 465 | 0.895 | 0.492 |
| workspace | 0.287 | 0.453 | 519 | 0.927 | 0.439 |

**Statistical Tests:**
- Kruskal-Wallis H: 2.89, p = 4.08e-01
- Worst group: banking (risk = 0.334)
- Best group: workspace (risk = 0.287)
- **Worst/Average ratio: 1.09**
- Cohen's d (worst vs best): 0.10

## By DAG Depth

| Group | Risk Rate | Std | N Traces | Success Rate | Avg Max Risk |
|-------|-----------|-----|----------|--------------|---------------|
| shallow | 0.305 | 0.461 | 2000 | 0.885 | 0.506 |

**Statistical Tests:**
- Kruskal-Wallis H: 0.00, p = 1.00e+00
- Worst group: shallow (risk = 0.305)
- Best group: shallow (risk = 0.305)
- **Worst/Average ratio: 1.00**
- Cohen's d (worst vs best): 0.00

