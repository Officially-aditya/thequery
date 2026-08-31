WITH benchmark_entry AS (
  SELECT $body$
An AI benchmark is a standardized test used to measure and compare the behavior or performance of models and AI systems under defined conditions. It usually combines a task, test data, instructions, an execution setup, a scoring method, and rules for aggregating results. The score is evidence about performance on that test. It is not a universal measurement of intelligence, usefulness, safety, or product quality.

The word is used in two ways. People may call a dataset such as MMLU a benchmark, but a reproducible benchmark result requires more than the questions. Prompt format, few-shot examples, tool access, sampling settings, model version, reasoning effort, time and compute limits, scorer, exclusions, and benchmark version can all change the outcome.

## What makes up an AI benchmark?

| Component | What it specifies | Why it matters |
| --- | --- | --- |
| Construct | The capability or property the evaluator intends to measure | A test can score arithmetic without measuring general reasoning |
| Tasks or scenarios | What the system must do | Multiple choice, code repair, dialogue, tool use, and image understanding test different behavior |
| Test data | Inputs, expected outputs, labels, rubrics, or environments | Quality, representativeness, freshness, and secrecy affect validity |
| Evaluation protocol | Prompting, examples, tools, time limits, retries, and allowed resources | Different protocols can produce different scores from the same model |
| System configuration | Model version, agent harness, temperature, reasoning effort, context limit, and hardware | A result belongs to the tested configuration, not an abstract model name |
| Metric and scorer | How individual outputs become numbers or judgments | Exact match, tests, human preference, and LLM judges capture different things |
| Aggregation | How task-level results become a reported score | A mean can hide weak categories, sample imbalance, and uncertainty |
| Baselines | Random, human, previous-system, or non-AI reference performance | A raw score has little meaning without a comparison point |
| Version and date | The precise dataset, rules, code, and evaluation date | Benchmarks change, models change, and public test sets become contaminated |

## Benchmark vs evaluation vs leaderboard

| Term | Meaning | Relationship |
| --- | --- | --- |
| Benchmark | A standardized test and protocol intended for repeatable comparison | One reusable form of evaluation |
| Evaluation or eval | Any systematic assessment of an AI system for a stated goal | Can include benchmarks, human studies, red teaming, simulations, and production tests |
| Leaderboard | A table that ranks submitted results | Displays benchmark outcomes but is not the benchmark itself |
| Test set | The held-out examples or environments used for scoring | One benchmark component |
| Metric | A rule that converts results into a measurement | Accuracy, pass rate, latency, cost, and win rate are examples |

A product evaluation can be valid without producing a public leaderboard. A leaderboard can also look precise while combining incomparable submissions. The important question is not only “Which model is first?” but “What exact system was tested, under which rules, for which intended use?”

## Common types of AI benchmarks

| Benchmark type | What it tests | Typical examples of evidence |
| --- | --- | --- |
| Knowledge and academic reasoning | Recall and problem solving in defined subjects | Multiple-choice or short-answer accuracy |
| Mathematics and formal reasoning | Correct multi-step solutions | Exact answer, proof verification, or judge score |
| Coding | Function generation, repository repair, or terminal work | Unit tests passed or issues resolved |
| Multimodal | Reasoning across text, images, audio, or video | Task accuracy, rubric score, or grounded response quality |
| Agentic | Planning and acting through tools or environments | End-to-end task completion under budgets and timeouts |
| Human preference | Which answer users or raters prefer | Pairwise win rate or a fitted rating |
| Safety and robustness | Harmful behavior, jailbreak resistance, bias, privacy, or adversarial stability | Attack success, refusal quality, subgroup metrics, or severity ratings |
| Efficiency | Speed and resource use | Latency, throughput, memory, energy, tokens, or cost |
| Domain-specific | Performance in legal, medical, financial, scientific, or company workflows | Expert-reviewed or operational metrics tied to the domain |

No single benchmark covers all of these dimensions. A model can lead a math test and still be expensive, unsafe, weak at tool use, or unsuitable for a specific language or profession.

## Common benchmark metrics

| Metric | What it reports | Important caution |
| --- | --- | --- |
| Accuracy | Share of items answered correctly | Treats every item equally and may hide class or subject differences |
| Exact match | Share matching a reference exactly after stated normalization | Can reject semantically correct wording |
| Precision, recall, and F1 | Quality of predicted labels, spans, or retrieved items | The averaging method and decision threshold matter |
| Pass@1 | Probability or share solved by the first generated answer | Depends on generation and execution rules |
| Pass@k | Chance that at least one of k samples succeeds | Not comparable with pass@1 and improves by spending more samples |
| Resolve rate | Share of end-to-end tasks accepted by a verifier | Measures the complete model-and-harness system |
| Win rate or rating | Pairwise preference against other systems | Depends on opponent mix, rater population, prompt distribution, and statistical model |
| nDCG, MRR, or Recall@k | Ranking or retrieval quality | Requires relevance judgments and a defined cutoff |
| Calibration | Whether confidence matches observed correctness | A capable model can still be poorly calibrated |
| Latency, throughput, and cost | Operational efficiency | Hardware, batching, caching, and provider pricing must be fixed |

A percentage without the metric is incomplete. “80 on a coding benchmark” could mean 80 percent of unit tests, 80 percent of issues resolved, pass@10, or a normalized composite. Those are different claims.

## Model benchmark vs system benchmark

Most modern AI results measure a system rather than isolated model weights. A system can include the model, prompt template, agent harness, tool definitions, retrieval, memory, reasoning effort, sampling parameters, safety filters, runtime, and error-recovery policy.

This distinction is especially important for coding and agentic benchmarks. A model may suggest a correct patch, but the agent must find the files, use tools, run tests, recover from errors, and leave an artifact the verifier accepts. Changing the harness or resource limit can change the score without changing the model.

The honest label is therefore specific, such as “Model X in Agent Y at high reasoning effort on Benchmark Z version 4.0.” Shortening that to “Model X scores 42” discards the conditions that produced the number.

## How an AI benchmark is run

1. **Define the claim.** State the capability, risk, or operational question the test is meant to inform.
2. **Choose representative tasks.** The examples should resemble the population of situations covered by the claim.
3. **Freeze the protocol.** Record prompts, examples, tools, budgets, sampling settings, versions, exclusions, and environment details.
4. **Run enough trials.** Deterministic tasks may need one run, while stochastic models and human ratings require repeated trials and uncertainty estimates.
5. **Score outputs consistently.** Use verified ground truth, executable tests, documented rubrics, blinded raters, or validated judges as appropriate.
6. **Report disaggregated results.** Include category scores, failures, uncertainty, cost, and configuration rather than only one average.
7. **Check external validity.** Confirm that higher benchmark performance predicts better outcomes in the intended real-world setting.

Reproducibility means another evaluator can run the stated procedure and obtain a statistically compatible result. It does not automatically prove that the benchmark measures the right thing.

## Why benchmark scores can mislead

### Data contamination

If test questions, answers, or close variants appear in pretraining, fine-tuning, synthetic-data generation, retrieval corpora, or prompt examples, a model may reproduce known material rather than generalize. Public benchmarks are particularly exposed because their data and solutions spread across the web. Sequestered test sets, temporal cutoffs, canary strings, contamination analysis, and regularly refreshed tasks reduce the risk but rarely prove zero exposure.

### Saturation

When leading systems approach the benchmark's ceiling, small score differences may reflect noise, formatting, or a few disputed items. The test stops separating frontier systems even if it remains useful for smaller models. Harder or refreshed benchmarks are then needed, but changing the tasks creates a new version whose scores should not be mixed with the old one.

### Benchmark gaming and overfitting

Once a benchmark becomes a target, developers can tune prompts, training data, routers, and post-processing specifically for it. That can produce genuine task improvement, memorization, or narrow optimization. A hidden test set helps, but repeated submissions can still leak information through scores.

### Weak construct validity

A test may have a convenient automatic metric but only a weak relationship to the capability named in its headline. Multiple-choice science questions do not fully measure scientific research. Preference votes do not prove factual accuracy. Passing repository tests does not prove maintainability or security.

### Protocol sensitivity

Prompt wording, answer extraction, few-shot examples, context order, temperature, maximum tokens, tool access, and reasoning budget can move results. Scores from two labs are not directly comparable when those conditions differ.

### Judge and rater bias

Human and model judges can favor particular styles, lengths, identities, or answer positions. Human preference is valuable but depends on who voted and which prompts they supplied. LLM-as-a-judge evaluation is scalable but needs calibration against qualified humans and checks for self-preference, verbosity bias, and inconsistency.

### Dataset and scorer errors

Benchmarks can contain ambiguous questions, wrong labels, broken tests, exploitable verifiers, duplicate examples, or unrealistic environments. A model may be penalized for a correct alternative or rewarded for satisfying the checker without solving the intended task.

### Aggregation hides tradeoffs

A composite average can conceal severe weakness in one category, place arbitrary weight on tasks, or combine incompatible units. Rankings can reverse when weights, subsets, or normalization change. Publish the components alongside the headline score.

## Static, dynamic, and human-preference benchmarks

| Design | Benefit | Limitation |
| --- | --- | --- |
| Public static test set | Easy to reproduce and inspect | Vulnerable to contamination, tuning, and saturation |
| Private or sequestered test set | Reduces direct leakage and answer lookup | Harder to audit and depends on trusted evaluation infrastructure |
| Dynamic or live benchmark | Refreshes questions to follow the frontier | Versions change and historical scores may become incomparable |
| Human-preference arena | Uses real prompts and direct user judgments | Prompt population, voter preferences, and model availability shift over time |
| Executable environment | Verifies whether code or an agent completed a task | Infrastructure, tests, resources, and harness behavior become part of the result |

LiveBench, for example, was designed around frequently refreshed questions with objective ground truth to reduce contamination. Chatbot Arena evaluates pairwise human preference on crowdsourced prompts. SWE-bench uses real GitHub issues and repository test environments. Each supplies useful evidence, but they answer different questions.

## Examples of widely used AI benchmarks

| Benchmark or framework | Primary focus | What to remember |
| --- | --- | --- |
| MMLU and MMLU-Pro | Broad academic knowledge and reasoning | Multiple-choice performance is not a general product score |
| GPQA Diamond | Difficult graduate-level science questions | Small, expert-level sets can have wider uncertainty and contamination concerns |
| AIME | Competition mathematics | Exact final answers emphasize mathematical problem solving under a specific prompt protocol |
| HumanEval | Function-level code generation | Pass@k and execution settings must be reported |
| SWE-bench | Resolving repository-level GitHub issues | Measures a model or agent inside an executable software environment |
| MMMU and MMMU-Pro | Multimodal academic reasoning | Requires both perception and subject reasoning |
| BIG-bench | Diverse language-model capabilities across many contributed tasks | Aggregation across heterogeneous tasks needs careful interpretation |
| HELM | Transparent, multi-scenario evaluation framework | Emphasizes standardized prompts, metrics, and broad reporting |
| LiveBench | Refreshed, objectively scored language-model tasks | Designed to reduce contamination through new releases |
| Chatbot Arena or LMArena | Human preference in anonymous pairwise comparisons | A preference leaderboard, not an objective correctness exam |

## How to compare two benchmark claims

Before treating one score as better, check:

- The exact benchmark name, subset, release, and test split
- The exact model checkpoint or API snapshot
- Whether the result measures base, instruct, reasoning, or fine-tuned variants
- Prompt template, number of examples, answer extraction, and system messages
- Temperature, number of samples, pass@k, reasoning effort, and token limit
- Tools, retrieval, web access, agent harness, and human intervention
- Hardware, concurrency, timeout, cost, and resource limits for performance tests
- Metric definition, scorer version, judge model, and tie handling
- Number of trials, confidence interval, and statistical significance
- Contamination controls, exclusions, failed runs, and whether the result is self-reported or independently reproduced

If these details are missing, the score is a marketing claim that still needs verification, not a stable scientific comparison.

## What makes a good benchmark?

A useful benchmark has a clearly stated purpose, representative and correctly labeled tasks, an appropriate difficulty range, validated metrics, meaningful baselines, documented protocols, uncertainty estimates, versioning, contamination controls, and enough transparency for independent scrutiny. It separates current systems without rewarding shortcuts and predicts outcomes that matter outside the test.

No benchmark remains perfect forever. Tasks saturate, public data leaks, products change, and deployment risks evolve. Good benchmark maintenance includes error correction, changelogs, regrading rules, retirement criteria, and new versions that do not silently overwrite old results.

## Bottom line

An AI benchmark is a controlled measurement instrument, not a final verdict. It tells you how a particular model or system performed on particular tasks under particular rules. Use several complementary benchmarks, preserve the full evaluation configuration, report uncertainty and cost, inspect failures, and validate the result on the real work the system is supposed to do.
$body$::text AS body
)
UPDATE content_items AS item
SET
  title = 'Benchmark',
  summary = 'A standardized test and evaluation protocol used to measure and compare AI models or systems on defined tasks, metrics, and operating conditions.',
  body = benchmark_entry.body,
  blocks = jsonb_build_array(
    jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', benchmark_entry.body)
  ),
  sources = jsonb_build_array(
    jsonb_build_object('title', 'NIST AI 800-3: Expanding the AI Evaluation Toolbox with Statistical Models', 'url', 'https://doi.org/10.6028/NIST.AI.800-3'),
    jsonb_build_object('title', 'Holistic Evaluation of Language Models (HELM)', 'url', 'https://arxiv.org/abs/2211.09110'),
    jsonb_build_object('title', 'Beyond the Imitation Game: BIG-bench', 'url', 'https://arxiv.org/abs/2206.04615'),
    jsonb_build_object('title', 'SWE-bench: Can Language Models Resolve Real-World GitHub Issues?', 'url', 'https://arxiv.org/abs/2310.06770'),
    jsonb_build_object('title', 'LiveBench: A Challenging, Contamination-Free LLM Benchmark', 'url', 'https://arxiv.org/abs/2406.19314'),
    jsonb_build_object('title', 'Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference', 'url', 'https://arxiv.org/abs/2403.04132')
  ),
  metadata = COALESCE(item.metadata, '{}'::jsonb) || jsonb_build_object(
    'category', 'Foundations',
    'relatedTerms', jsonb_build_array('gpqa-diamond', 'aime', 'mmmu-pro', 'lmarena', 'artificial-analysis', 'terminal-bench'),
    'analogy', 'An AI benchmark is like a standardized exam with a published syllabus and scoring key: useful for comparison, but not a complete prediction of performance on the job.',
    'seoDescription', 'An AI benchmark is a standardized test for comparing models or systems. Learn about datasets, metrics, leaderboards, contamination, and fair score comparisons.',
    'seoKeywords', jsonb_build_array('what is an AI benchmark', 'AI benchmark explained', 'LLM benchmarks', 'AI benchmark vs evaluation', 'benchmark vs leaderboard', 'how AI benchmarks work', 'AI benchmark metrics', 'compare AI benchmark scores', 'benchmark data contamination', 'benchmark saturation', 'LLM evaluation', 'model vs system benchmark', 'AI leaderboard scores', 'benchmark gaming')
  ),
  published_at = DATE '2026-08-31',
  updated_at = NOW()
FROM benchmark_entry
WHERE item.kind = 'glossary'
  AND item.slug = 'benchmark'
  AND item.parent_slug = '';
