# AgentDojo Data Structure Analysis

## Overview

- Total User Tasks: 86 (paper reports ~97)
- Total Injection Tasks: 27
- Total Security Test Combinations: 567 (paper reports ~629)
- Attack Types: 16
- Tool Categories: 10
- Models Benchmarked: 29

## Suite Breakdown

| Suite | User Tasks | Injection Tasks | Security Tests |
|-------|------------|-----------------|----------------|
| banking | 16 | 9 | 144 |
| workspace | 33 | 6 | 198 |
| travel | 20 | 7 | 140 |
| slack | 17 | 5 | 85 |

## Attack Types

1. direct
2. ignore_previous
3. system_message
4. injecagent
5. important_instructions
6. important_instructions_no_user_name
7. important_instructions_no_model_name
8. important_instructions_no_names
9. important_instructions_wrong_model_name
10. important_instructions_wrong_user_name
11. tool_knowledge
12. dos
13. swearwords_dos
14. captcha_dos
15. offensive_email_dos
16. felony_dos

## Tool Categories

- calendar_client
- travel_booking_client
- user_account
- banking_client
- web
- cloud_drive_client
- email_client
- slack
- types
- file_reader

## Benchmark Results Summary

Total traces analyzed: 8405

### Safety Rate by Model

| Model | Safety Rate (attacked traces) |
|-------|-------------------------------|
| claude-3-5-sonnet-20240620 | 33.86% |
| claude-3-opus-20240229 | 11.29% |
| gpt-4o-2024-05-13 | 28.58% |

### Safety Rate by Attack Type

| Attack Type | Safety Rate |
|-------------|-------------|
| important_instructions_no_model_name | 46.10% |
| important_instructions_no_names | 45.79% |
| important_instructions_no_user_name | 44.83% |
| captcha_dos | 38.14% |
| offensive_email_dos | 37.11% |
| felony_dos | 36.08% |
| tool_knowledge | 34.50% |
| dos | 32.99% |
| swearwords_dos | 31.96% |
| important_instructions | 30.95% |
| important_instructions_wrong_model_name | 23.69% |
| important_instructions_wrong_user_name | 23.21% |
| injecagent | 5.72% |
| ignore_previous | 5.41% |
| direct | 3.66% |

### Safety Rate by Suite (Tool Domain)

| Suite | Safety Rate |
|-------|-------------|
| banking | 33.19% |
| slack | 59.78% |
| travel | 11.01% |
| workspace | 19.74% |
