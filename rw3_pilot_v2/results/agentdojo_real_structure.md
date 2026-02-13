# AgentDojo Real Structure Analysis

**Data Source**: Real benchmark results from agentdojo repository

## Summary
- **Models evaluated**: 22
- **Suites**: 4
- **Attack types**: 16
- **Defense strategies**: 5
- **Total result files**: 36679

## Models
- Meta-SecAlign-70B
- claude-3-5-sonnet-20240620
- claude-3-5-sonnet-20241022
- claude-3-7-sonnet-20250219
- claude-3-haiku-20240307
- claude-3-opus-20240229
- claude-3-sonnet-20240229
- command-r
- command-r-plus
- gemini-1.5-flash-001
- gemini-1.5-flash-002
- gemini-1.5-pro-001
- gemini-1.5-pro-002
- gemini-2.0-flash-001
- gemini-2.0-flash-exp
- gpt-3.5-turbo-0125
- gpt-4-0125-preview
- gpt-4-turbo-2024-04-09
- gpt-4o-2024-05-13
- gpt-4o-mini-2024-07-18
- meta-llama_Llama-3-70b-chat-hf
- meta-llama_Llama-3.3-70B-Instruct

## Suites
### travel
- User tasks: 20
- Injection tasks: 7
- Attacks tested: {'important_instructions_wrong_model_name', 'important_instructions', 'offensive_email_dos', 'felony_dos', 'dos', 'important_instructions_no_model_name', 'none', 'direct', 'important_instructions_wrong_user_name', 'injecagent', 'ignore_previous', 'tool_knowledge', 'swearwords_dos', 'important_instructions_no_names', 'captcha_dos', 'important_instructions_no_user_name'}

### workspace
- User tasks: 40
- Injection tasks: 14
- Attacks tested: {'important_instructions_wrong_model_name', 'important_instructions', 'offensive_email_dos', 'felony_dos', 'dos', 'important_instructions_no_model_name', 'none', 'direct', 'important_instructions_wrong_user_name', 'injecagent', 'ignore_previous', 'tool_knowledge', 'swearwords_dos', 'important_instructions_no_names', 'captcha_dos', 'important_instructions_no_user_name'}

### slack
- User tasks: 21
- Injection tasks: 5
- Attacks tested: {'important_instructions_wrong_model_name', 'important_instructions', 'offensive_email_dos', 'felony_dos', 'dos', 'important_instructions_no_model_name', 'none', 'direct', 'important_instructions_wrong_user_name', 'injecagent', 'ignore_previous', 'tool_knowledge', 'swearwords_dos', 'important_instructions_no_names', 'captcha_dos', 'important_instructions_no_user_name'}

### banking
- User tasks: 16
- Injection tasks: 9
- Attacks tested: {'important_instructions_wrong_model_name', 'important_instructions', 'offensive_email_dos', 'felony_dos', 'dos', 'important_instructions_no_model_name', 'none', 'direct', 'important_instructions_wrong_user_name', 'injecagent', 'ignore_previous', 'tool_knowledge', 'swearwords_dos', 'important_instructions_no_names', 'captcha_dos', 'important_instructions_no_user_name'}

## Attack Types
- captcha_dos
- direct
- dos
- felony_dos
- ignore_previous
- important_instructions
- important_instructions_no_model_name
- important_instructions_no_names
- important_instructions_no_user_name
- important_instructions_wrong_model_name
- important_instructions_wrong_user_name
- injecagent
- none
- offensive_email_dos
- swearwords_dos
- tool_knowledge

## Defenses
- none
- repeat_user_prompt
- spotlighting_with_delimiting
- tool_filter
- transformers_pi_detector
