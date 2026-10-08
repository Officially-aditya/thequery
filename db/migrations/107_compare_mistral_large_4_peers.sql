-- Replace the standalone score list with a same-evaluator peer comparison.
-- This edits only the Mistral glossary; raw model benchmark evidence remains.
WITH comparison AS (
  SELECT id, replace(body,
    $section_before$## Benchmark profile

The table distinguishes Mistral's launch claims from independent results reported by Artificial Analysis and Vals. Values were reviewed on October 8, 2026. Matching benchmark names alone do not establish matching checkpoints, prompts, tool settings or evaluation harnesses.

| Evaluation | Reported result | Source and conditions |
| --- | ---: | --- |
| DeepSWE v1.1 | 61.7% | Mistral launch report |
| SWE-Atlas Codebase QnA | 59.4% | Mistral launch report |
| Terminal-Bench 4.0 | 28.3%\* | Mistral launch report |
| AutomationBench-AA | 59.9% | Artificial Analysis |
| GDPval-AA v2.1 | 1423.91 Elo | Artificial Analysis |
| AA-Briefcase v1.1 | 1392.53 Elo | Artificial Analysis |
| GDP.pdf | 18.6% | Artificial Analysis, all-pass metric |
| SciCode | 54.2% | Artificial Analysis, evaluation marked under review |
| CritPt | 10.6% | Artificial Analysis, evaluation marked under review |
| Humanity's Last Exam | 35.0% | Artificial Analysis |
| MMMU-Pro | 76.4% | Artificial Analysis |
| AA-Omniscience accuracy | 25.8% | Artificial Analysis |
| AA-Omniscience hallucination rate | 41.9% | Artificial Analysis, lower is better |
| Vals Finance Agent v2 | 54.68% | Vals |
| Harvey's Legal Agent Benchmark | 15.83% | Vals, held-out task pass rate |
| CyberGym-E2E-AA | 82% | Artificial Analysis launch analysis |
| Cybench | 93% | Mistral launch report |

\* **Terminal-Bench 4.0:** Mistral reports 28.3%, Artificial Analysis reports 26.8%, and Vals reports 22.73%. The table shows the highest reported result. Evaluation conditions are not fully specified across these sources.

AA's long-context reasoning result is 0.813 on AA-LCR v1.1. Its Omniscience Index is -5.3 on a scale from -100 to 100. That index measures knowledge reliability and is separate from both accuracy and hallucination rate, so the three Omniscience readings should not be substituted for one another.

A composite index works like an exam average across several subjects. AA's Intelligence Index is 38.38 for this preview, while Vals reports 48.05% on its own Vals Index. The indexes use different constituent evaluations and scoring rules. Vals' legal result is also a task pass rate: a task must satisfy the benchmark's completion criteria. It is not the same measurement as the criterion pass rate used in AA's Harvey LAB-AA implementation. AA does not publish a Harvey LAB-AA result for this model in its profile, and this entry does not fill that gap with the Vals number.

$section_before$,
    $section_after$## Benchmark profile

Mistral Large 4 sits in roughly the same independent capability tier as GLM-5.3-Flash, DeepSeek V4.1 Flash and GPT-6 Luna: their Artificial Analysis Intelligence Index scores range from about 38 to 42. These are multimodal reasoning and agentic alternatives with different sizes, availability and prices. GPT-6 Luna provides a low-cost hosted baseline alongside the two Flash models.

The comparison below uses Artificial Analysis results throughout, reviewed on October 8, 2026. Each benchmark cell shows the best published result among the model's available evaluated settings; a setting below the highest recorded effort is shown in parentheses. The index, cost, speed and latency rows use the configuration with that model's highest Intelligence Index. Bold marks the strongest reported result in a row. Lower is better for cost, answer latency and hallucination rate.

| Metric | **Mistral Large 4 Preview** | [GLM-5.3-Flash](/glossary/glm-5-3-flash) | [DeepSeek V4.1 Flash](/glossary/deepseek-v4-1-flash) | [GPT-6 Luna](/glossary/gpt-6-luna) |
| --- | ---: | ---: | ---: | ---: |
| AA Intelligence Index | 38.4 | **41.8** | 39.5 | 38.1 |
| AA index-task cost (USD, lower) | 1.13 | 0.25 | 0.27 | **0.07** |
| Time to first answer (s, lower) | 18.7 | 42.1 | **10.1** | 111.3 |
| Output speed (tokens/s) | 116.1 | 51.4 | **223.1** | 128.2 |
| Terminal-Bench 4.0 | 26.8%\* | **32.8%** | 26.8% | 12.6% |
| AutomationBench-AA | 59.9% | 60.4% | **68.9%** | 53.2% |
| GDPval-AA v2.1 (Elo) | 1,424 | **1,647** | 1,600 | 1,432 |
| AA-Briefcase v1.1 (Elo) | 1,393 | **1,454** | 1,420 | 1,336 |
| GDP.pdf (all-pass) | 18.6% | 15.4% | 12.8% | **22.8%** |
| SciCode | 54.2% | 51.6% | 51.9% | **54.6%** |
| CritPt | 10.6% | 15.4% | 14.3% | **19.4%** |
| Humanity's Last Exam | 35.0% | **39.9%** | 39.2% | 38.5% |
| AA-LCR v1.1 (0 to 1) | 0.813 | 0.800 | **0.840** | 0.833 |
| MMMU-Pro | 76.4% | Not reported | 77.0% | **79.7%** (xhigh) |
| AA-Omniscience accuracy | 25.8% | 27.5% | **46.4%** | 44.2% (xhigh) |
| AA-Omniscience hallucination rate (lower) | 41.9% | **27.6%** | 96.5% | 76.7% |
| AA-Omniscience Index | -5.3 | **7.5** | -5.3 | 0.7 |

\* **Terminal-Bench 4.0:** This comparison uses AA's 26.8% for Mistral Large 4 so the evaluator is consistent across columns. Mistral reports 28.3%, Artificial Analysis reports 26.8%, and Vals reports 22.73%. Evaluation conditions are not fully specified across those sources.

Costs use standard token rates. AA estimates Mistral's index-task cost at about $0.57 during the launch discount, compared with $1.13 at standard pricing. Speed and latency are API measurements for the evaluated configurations, not guarantees for every request. AA marks SciCode and CritPt under review. Not reported denotes an absent published result. AA-LCR is shown on its 0 to 1 scale, and the Omniscience Index runs from -100 to 100; accuracy and hallucination rate are separate measurements.

**Comparison sources:** [Mistral Large 4](https://artificialanalysis.ai/models/mistral-large-4), [GLM-5.3-Flash](https://artificialanalysis.ai/models/glm-5-3-flash), [DeepSeek V4.1 Flash](https://artificialanalysis.ai/models/deepseek-v4-1-flash) and [GPT-6 Luna](https://artificialanalysis.ai/models/gpt-6-luna).

The tradeoffs are clearer than a standalone list of Mistral scores. AA reports higher GDP.pdf and SciCode scores for Mistral than for both Flash alternatives, while Luna is slightly ahead of Mistral on both. GLM leads the index and the two Elo-rated knowledge-work evaluations, while DeepSeek leads AutomationBench-AA. Mistral's standard index-task cost is substantially higher than the other three at this capability level, and its preview weights are still scheduled for a later release.

Mistral separately reports 61.7% on DeepSWE v1.1, 59.4% on SWE-Atlas Codebase QnA and 93% on Cybench. AA's cyber launch analysis reports 82% on CyberGym-E2E-AA. These additional results describe areas the vendor emphasizes, but they are not substituted for the shared evaluations in the table.

A composite index works like an exam average across several subjects. Vals reports 48.05% on its own Vals Index, which uses different constituent evaluations and scoring from AA's index. Vals' 15.83% result on Harvey's Legal Agent Benchmark is a held-out task pass rate, not the criterion pass rate used in AA's Harvey LAB-AA implementation. AA does not publish a Harvey LAB-AA result for this model in its profile, so that gap remains unfilled.

$section_after$
  ) AS body
  FROM content_items
  WHERE kind = 'glossary'
    AND slug = 'mistral-large-4'
    AND parent_slug = ''
)
UPDATE content_items AS content
SET
  body = comparison.body,
  blocks = jsonb_build_array(jsonb_build_object(
    'id', 'markdown-1', 'type', 'markdown', 'content', comparison.body
  )),
  updated_at = NOW()
FROM comparison
WHERE content.id = comparison.id;
