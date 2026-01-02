# RW1 Preliminary Experiment Report

**Generated**: 2026-01-02 13:28:02

---

## Executive Summary

This report presents the results of preliminary experiments to verify
5 research hypotheses for the RW1 research work on Remote Supervision
Relation Extraction.

**Overall Result**: 2/5 hypotheses passed verification

## Summary Table


| Hypothesis | Description | Primary Metric | Threshold | Value | p-value | Result |
|------------|-------------|----------------|-----------|-------|---------|--------|
| H1 (LDA-LLM) | Distribution Shift vs F1 | Pearson r | > 0.8 | 1.0000 | 0.0000e+00 | ✅ Passed |
| H2 (LLM-RFCRE) | ARS vs Forgetting Rate | Spearman ρ | > 0.5 | -0.0381 | 8.0370e-01 | ❌ Failed |
| H3 (LLM-PUQ) | PDI vs Noise Rate | Pearson r | > 0.5 | 0.7201 | 7.2565e-06 | ✅ Passed |
| H4 (HGT-LC) | Path Length vs FN Rate | Cohen's d | > 0.5 | 0.3748 | 1.6383e-129 | ❌ Failed |
| H5 (PGCDN) | Bag Size vs Reliability | Cohen's d | 0.5-0.8 | 1.2741 | 7.9644e-187 | ❌ Failed |


---

## Detailed Results


### H1: Distribution shift vs F1 drop correlation

**Status**: **PASSED** ✅

#### Statistics

- **pearson_r**: 1.0000
- **p_value**: 0.0000

#### Threshold Criteria

- pearson_r: 0.8
- p_value: 0.05


### H2: ARS vs Forgetting Rate correlation

**Status**: **FAILED** ❌

#### Statistics

- **spearman_rho**: -0.0381
- **p_value**: 0.8037
- **cohens_d**: 0.1232

#### Group Analysis

- **high_ars_mean_fr**: 0.2391
- **low_ars_mean_fr**: 0.2045
- **median_ars**: 0.0392

#### Threshold Criteria

- spearman_rho: 0.5
- cohens_d: 0.5
- p_value: 0.05


### H3: PDI vs Noise Rate correlation

**Status**: **PASSED** ✅

#### Statistics

- **pearson_r**: 0.7201
- **p_value**: 0.0000

#### Group Analysis

- **low_noise_mean_pdi**: 0.0033
- **high_noise_mean_pdi**: 0.0034
- **cohens_d**: 1.1792

#### Threshold Criteria

- pearson_r: 0.5
- p_value: 0.05


### H4: Path Length vs False Negative Rate

**Status**: **FAILED** ❌

#### Statistics

- **cohens_d**: 0.3748
- **t_statistic**: 24.4304
- **p_value**: 0.0000
- **pearson_r**: 0.2557

#### Group Analysis

- **short_paths**:
  - count: 0
  - fn_rate: 0.0000
- **medium_paths**:
  - count: 7982
  - fn_rate: 0.5045
- **long_paths**:
  - count: 9302
  - fn_rate: 0.6849

#### Threshold Criteria

- cohens_d: 0.5


### H5: Bag Size vs Label Reliability

**Status**: **FAILED** ❌

#### Statistics

- **cohens_d**: 1.2741
- **t_statistic**: 33.7976
- **p_value**: 0.0000
- **pearson_r**: 0.7855

#### Group Analysis

- **size_1**:
  - count: 1374
  - reliability: 0.5218
- **size_2**:
  - count: 879
  - reliability: 0.8237
- **size_3plus**:
  - count: 1295
  - reliability: 0.9884

#### Threshold Criteria

- cohens_d: [0.5, 0.8]

---

## Recommendations

### ⚠️ Partial Success - Review and Adjust

**2/5** hypotheses passed verification.

Recommendations:

1. Focus on methods for passed hypotheses
2. Re-evaluate failed hypotheses:
   - H2: Consider alternative verification methods
   - H4: Consider alternative verification methods
   - H5: Consider alternative verification methods
3. Consider adjusting research scope

---

## Methodology Notes

### Statistical Standards

- **Significance level**: p < 0.05
- **Effect size interpretation**: Cohen's d (0.2=small, 0.5=medium, 0.8=large)
- **Correlation interpretation**: |r| (0.3=weak, 0.5=moderate, 0.7=strong)

### Avoiding Circular Reasoning

All experiments were designed to avoid circular reasoning:
- No target labels used in feature computation
- Independent validation methods (e.g., human annotations for H5)
- Synthetic noise injection for controlled experiments (H3)

---

## Appendix: File Locations

### Results

- `results/h1/h1_results.json`
- `results/h2/h2_results.json`
- `results/h3/h3_results.json`
- `results/h4/h4_results.json`
- `results/h5/h5_results.json`

### Figures

- `results/h1/figures/`
- `results/h2/figures/`
- `results/h3/figures/`
- `results/h4/figures/`
- `results/h5/figures/`
