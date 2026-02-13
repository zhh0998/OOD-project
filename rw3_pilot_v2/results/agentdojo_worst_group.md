# AgentDojo Worst-Group Analysis (Real Data)

## Data Source
- **Source**: Real benchmark results from agentdojo repository runs/
- **Total results parsed**: 36679
- **Suite-level metrics**: 312

## Worst-Group Across Suites

| Model | Defense | Attack | n_suites | worst_utility | avg_utility | worst_asr | avg_asr | asr_ratio |
|-------|---------|--------|----------|---------------|-------------|-----------|---------|----------|
| Meta-SecAlign-70B | none | direct | 4 | 0.629 | 0.749 | 1.000 | 0.979 | 1.02 |
| Meta-SecAlign-70B | none | ignore_previous | 4 | 0.643 | 0.727 | 1.000 | 0.973 | 1.03 |
| Meta-SecAlign-70B | none | important_instructions | 4 | 0.643 | 0.735 | 1.000 | 0.959 | 1.04 |
| Meta-SecAlign-70B | none | none | 4 | 0.650 | 0.768 | 0.000 | 0.000 | nan |
| Meta-SecAlign-70B | repeat_user_prompt | direct | 4 | 0.657 | 0.761 | 1.000 | 0.979 | 1.02 |
| Meta-SecAlign-70B | repeat_user_prompt | ignore_previous | 4 | 0.621 | 0.751 | 1.000 | 0.977 | 1.02 |
| Meta-SecAlign-70B | repeat_user_prompt | important_instructions | 4 | 0.643 | 0.737 | 1.000 | 0.961 | 1.04 |
| Meta-SecAlign-70B | repeat_user_prompt | none | 4 | 0.700 | 0.856 | 0.000 | 0.000 | nan |
| claude-3-5-sonnet-20 | none | important_instructions | 4 | 0.446 | 0.528 | 0.917 | 0.634 | 1.45 |
| claude-3-5-sonnet-20 | none | none | 4 | 0.650 | 0.782 | 0.000 | 0.000 | nan |
| claude-3-5-sonnet-20 | none | important_instructions | 4 | 0.505 | 0.687 | 1.000 | 0.985 | 1.01 |
| claude-3-5-sonnet-20 | none | none | 4 | 0.650 | 0.775 | 0.000 | 0.000 | nan |
| claude-3-7-sonnet-20 | none | important_instructions | 4 | 0.700 | 0.762 | 0.993 | 0.922 | 1.08 |
| claude-3-7-sonnet-20 | none | none | 4 | 0.750 | 0.863 | 0.000 | 0.000 | nan |
| claude-3-haiku-20240 | none | important_instructions | 4 | 0.293 | 0.336 | 0.986 | 0.888 | 1.11 |
| claude-3-haiku-20240 | none | none | 4 | 0.350 | 0.400 | 0.000 | 0.000 | nan |
| claude-3-opus-202402 | none | important_instructions | 4 | 0.486 | 0.520 | 0.971 | 0.860 | 1.13 |
| claude-3-opus-202402 | none | none | 4 | 0.562 | 0.674 | 0.000 | 0.000 | nan |
| claude-3-sonnet-2024 | none | important_instructions | 4 | 0.295 | 0.331 | 0.867 | 0.692 | 1.25 |
| claude-3-sonnet-2024 | none | none | 4 | 0.500 | 0.542 | 0.000 | 0.000 | nan |

## Worst-Group Across Attack Types

| Model | Defense | Suite | n_attacks | worst_asr | avg_asr | asr_ratio | worst_attack |
|-------|---------|-------|-----------|-----------|---------|-----------|-------------|
| Meta-SecAlign-70B | none | banking | 4 | 0.924 | 0.684 | 1.35 | direct |
| Meta-SecAlign-70B | none | slack | 4 | 0.990 | 0.726 | 1.36 | direct |
| Meta-SecAlign-70B | none | travel | 4 | 1.000 | 0.750 | 1.33 | important_instructions |
| Meta-SecAlign-70B | none | workspace | 4 | 1.000 | 0.750 | 1.33 | important_instructions |
| Meta-SecAlign-70B | repeat_user_prompt | banking | 4 | 0.917 | 0.686 | 1.34 | direct |
| Meta-SecAlign-70B | repeat_user_prompt | slack | 4 | 1.000 | 0.731 | 1.37 | direct |
| Meta-SecAlign-70B | repeat_user_prompt | travel | 4 | 1.000 | 0.750 | 1.33 | important_instructions |
| Meta-SecAlign-70B | repeat_user_prompt | workspace | 4 | 1.000 | 0.750 | 1.33 | important_instructions |
| claude-3-5-sonnet-20 | none | banking | 2 | 0.917 | 0.458 | 2.00 | important_instructions |
| claude-3-5-sonnet-20 | none | slack | 2 | 0.314 | 0.157 | 2.00 | important_instructions |
| claude-3-5-sonnet-20 | none | travel | 2 | 0.621 | 0.311 | 2.00 | important_instructions |
| claude-3-5-sonnet-20 | none | workspace | 2 | 0.683 | 0.342 | 2.00 | important_instructions |
| claude-3-5-sonnet-20 | none | banking | 2 | 0.979 | 0.490 | 2.00 | important_instructions |
| claude-3-5-sonnet-20 | none | slack | 2 | 0.962 | 0.481 | 2.00 | important_instructions |
| claude-3-5-sonnet-20 | none | travel | 2 | 1.000 | 0.500 | 2.00 | important_instructions |
| claude-3-5-sonnet-20 | none | workspace | 2 | 1.000 | 0.500 | 2.00 | important_instructions |
| claude-3-7-sonnet-20 | none | banking | 2 | 0.958 | 0.479 | 2.00 | important_instructions |
| claude-3-7-sonnet-20 | none | slack | 2 | 0.762 | 0.381 | 2.00 | important_instructions |
| claude-3-7-sonnet-20 | none | travel | 2 | 0.993 | 0.496 | 2.00 | important_instructions |
| claude-3-7-sonnet-20 | none | workspace | 2 | 0.973 | 0.487 | 2.00 | important_instructions |

## Statistical Tests

- **utility_suite_diff_none**: H=12.59, p=0.0056 (significant)
- **asr_suite_diff_none**: H=nan, p=nan (not significant)
- **utility_suite_diff_important_instructions**: H=9.03, p=0.0289 (significant)
- **asr_suite_diff_important_instructions**: H=26.61, p=0.0000 (significant)
- **utility_suite_diff_direct**: H=7.03, p=0.0708 (not significant)
- **asr_suite_diff_direct**: H=14.14, p=0.0027 (significant)
- **utility_suite_diff_ignore_previous**: H=0.97, p=0.8095 (not significant)
- **asr_suite_diff_ignore_previous**: H=9.69, p=0.0214 (significant)

## Key Findings

1. **Worst-group/Average ASR ratio**: max=1.83, median=1.15
2. **Cross-suite differences**: Significant
3. **Utility-Security tradeoff**: Not clearly present
