-- Expand the Artificial Analysis glossary entry from a three-paragraph stub into a
-- structured reference that names the platform's actual products and indexes.
--
-- Sources fetched and verified on September 26 2026:
-- - Artificial Analysis home page (https://artificialanalysis.ai/)
-- - Intelligence benchmarking methodology v4.3.2
--   (https://artificialanalysis.ai/methodology/intelligence-benchmarking)
-- - Coding agent benchmarking methodology v1.5
--   (https://artificialanalysis.ai/agents/coding-agents)
-- - Performance / inference benchmarking methodology v2.2.0, dated 2 March 2026
--   (https://artificialanalysis.ai/methodology/performance-benchmarking)
-- - About page (https://artificialanalysis.ai/about)
--
-- Fixes stale text carried over from the original stub: the Intelligence Index is
-- now v4.3.2 with ten evaluations, and GPQA Diamond has moved to the legacy set
-- rather than sitting inside the index.
--
-- Body and blocks stay in sync, sources are attached, and metadata gains SEO
-- keywords plus related terms that all exist as glossary slugs.

WITH artificial_analysis_entry AS (
  SELECT $body$
Artificial Analysis is an independent benchmarking company that measures AI models, inference providers, cloud services, and chips on a common, reproducible footing. It is the reference most industry reporting reaches for when someone asks a blunt question: which model is actually best, and what will it cost to run?

The company was founded by Micah Hill-Smith and George Cameron, runs headquarters in San Francisco with a second office in Melbourne, and describes its own scope as 500+ models benchmarked, 100+ inference providers, 1,000+ endpoints, and more than a trillion evaluation tokens. Independence is the product: a vendor cannot quietly change a score that an outside party publishes, and both sides of a buying decision can point at the same number.

## What Artificial Analysis measures

Four things, in descending order of how often people cite them:

- **Model intelligence.** Composite and per-benchmark capability scores under identical conditions.
- **Serving performance.** Output speed, time to first token, and end-to-end response time for the same model across different providers and hardware.
- **Price and real cost.** Input, cache-hit, cache-write, and output pricing turned into a cost-per-task figure rather than a raw per-million-token rate.
- **Openness.** How much of a model is actually available and how much of its training story is disclosed.

Alongside the measurement, Artificial Analysis ships products: Optima for building custom benchmarks, a Model Recommender that weights intelligence, speed, and cost by your priorities, MicroEvals, a Data Playground, and Image Lab. It also maintains hardware benchmarks covering NVIDIA and AMD GPUs, Google TPUs, AWS Trainium, Cerebras WSE, and SambaNova RDU.

## Artificial Analysis leaderboard

The main leaderboard is the LLM Leaderboard, and it is deliberately a set of ranked charts rather than a single winner column. The same 600-odd model entries get re-sorted by Intelligence Index, output tokens per second, cost per task, and openness, and each chart carries filters for open weights versus proprietary, reasoning versus non-reasoning, text-only versus multimodal, and country of origin.

The same page also plots the axes against each other: intelligence against cost per task, intelligence against time per task, and intelligence against output tokens per task, each with a Pareto line and a highlighted most-attractive quadrant. That is the practical way to use the data. The fastest model is rarely the right model, and the strongest model is rarely the right model at a given budget. The quadrant view answers the question you actually have, which is where the efficient frontier sits.

Frontier intelligence is also charted over time, with all reasoning and effort variants of a release grouped together, so a capability jump is visible as a step rather than smeared across variants.

## Artificial Analysis Intelligence Index

The Artificial Analysis Intelligence Index is the headline number, currently v4.3.2. It aggregates ten evaluations into a single score across four weighted categories, with the weighting tilted toward agentic work because that is where models actually differentiate now:

| Category | Weight |
| --- | ---: |
| Agents | 30% |
| General | 30% |
| Coding | 20% |
| Scientific Reasoning | 20% |

The ten constituent evaluations and their share of the index:

| Evaluation | Category | Weight | What it measures |
| --- | --- | ---: | --- |
| AA-Omniscience | General | 15% | Knowledge accuracy (10%) plus non-hallucination rate (5%) |
| AA-Briefcase v1.1 | Agents | 15% | Multi-week knowledge work graded as Elo |
| GDPval-AA v2.1 | Agents | 10% | Economically valuable professional deliverables, Elo |
| Terminal-Bench 4.0 | Coding | 10% | Terminal-based agentic task execution |
| SciCode | Coding | 10% | Scientific Python with sub-problem scoring |
| GDP.pdf | General | 10% | Long PDF document reasoning, all-pass |
| Humanity's Last Exam | Scientific Reasoning | 10% | Frontier knowledge and reasoning |
| CritPt | Scientific Reasoning | 10% | Physics and critical-thinking problems |
| AutomationBench-AA | Agents | 5% | SaaS workflow automation with a guardrail |
| AA-LCR v1.1 | General | 5% | Long context retrieval |

Two structural details matter. First, the index is a weighted average, not a mean, so a strong coding score cannot fully paper over weak long-context retrieval. Second, the Elo-based evaluations are anchored and then frozen: GDPval-AA pins DeepSeek V4.1 Flash (max) at 1600, AA-Briefcase pins GPT-5.5 (medium) at 1000, and each model's Elo is converted with clamp((Elo - 500) / 2000) at the time it is added. Freezing the contribution keeps the index comparable across months even as the underlying Elo scale drifts.

Artificial Analysis states a 95% confidence interval of under plus or minus 1% for the index itself, based on more than ten repeats on some models, while noting that individual evaluations are noisier. The index is primarily text-based and English-language; image inputs, speech, and multilingual performance are tracked separately.

GPQA Diamond, which older write-ups often list as part of the index, is now in the legacy evaluation set, alongside MMLU-Pro, AIME 2025, LiveCodeBench, MATH-500, tau-squared-Bench Telecom, and Terminal-Bench 2.1. If you are reading a model card that quotes an AA Intelligence Index score, check the version number before comparing it to a current one.

## Artificial Analysis Coding Agent Index

The Coding Agent Index, currently v1.5, answers a different question from the Intelligence Index: not what can the model do, but what happens when you point a real coding agent at real software work. It is an equal-weight average of pass@1 across three benchmarks, 303 tasks in total, with three attempts per task:

| Benchmark | Tasks | Source | What it measures |
| --- | ---: | --- | --- |
| DeepSWE v1.1 | 113 | Datacurve | Long-horizon software engineering patches, verified by a program checker |
| Terminal-Bench 4.0 | 66 | Laude Institute | Agentic terminal use, test-suite pass or fail |
| SWE-Atlas-QnA | 124 | Scale AI | Repository question answering, task resolve rate |

Scoring averages the three attempts per task and then averages across tasks. Time-limit overruns and safety refusals score zero rather than being excluded. The index is charted against cost per task and against execution time, where execution time means active agent wall time and excludes environment startup, judge time, and harness overhead.

The important framing detail: each row is an agent variant, meaning a model plus a harness, not a bare model. Anthropic plus Cognition and OpenAI plus Cognition appear as distinct entries, which is exactly the point. The same model behaves differently under a different scaffold, so quoting a bare model score for an agentic comparison is a category error. Recent versions also track safety refusal rate (blocked versus fallback) and reward hacking rate, the latter on Terminal-Bench 4.0 with a Claude Code harness.

## Capability indexes

A single intelligence number hides which kind of smart a model has. The Capability Indexes split performance into six professional domains, Finance and Accounting, Strategy and Operations, Legal, Healthcare and Medical, Engineering, and Economics, each built from the relevant subset of the index evaluations. The Finance and Accounting Index, for example, combines AA-Omniscience, GDPval-AA v2.1, AA-Briefcase v1.1, Humanity's Last Exam, AutomationBench-AA, AA-LCR v1.1, and GDP.pdf.

There is also a separate Artificial Analysis Multilingual Index built on Global-MMLU-Lite across sixteen languages, including English, Chinese, Hindi, Spanish, French, Arabic, Japanese, German, and Korean. A model can lead the aggregate index while trailing badly in a language that matters to your users, and this is the chart that catches it.

## Artificial Analysis Openness Index

The Openness Index scores how open a model actually is across four components: transparency of pre-training data, transparency of post-training data, transparency of methodology, and model availability, up to a maximum of 18. Availability is not a yes or no; a model labelled Commercial Use Restricted scores differently from one released under a non-commercial licence.

This matters because open-weight and open-source are routinely used interchangeably and are not the same claim. The Openness Index keeps the distinction measurable, and it can be plotted directly against the Intelligence Index to show whether the capability you want is actually reachable under a licence you can accept.

## Price and cost benchmarks

Raw price per million tokens hides most of what you will actually pay. Artificial Analysis reports cost per Intelligence Index task, computed from input, cache-hit, cache-write, reasoning, and answer token prices, divided by task count and weighted by each evaluation's index weight. Two models with identical sticker prices can land in different places once reasoning tokens and cache behaviour are counted.

Related views include the total cost to run the whole index, the blended price, and stacked blended pricing that separates cache-hit, input, and output rates. The same data is available as a price against intelligence scatter, which is usually more useful than a table.

## Speed and latency benchmarks

The performance methodology, currently v2.2.0, tests standardized workloads of 1k, 10k, and 100k input tokens plus a vision workload, at a single prompt and at ten parallel prompts. The 10k workload is the site default, which replaced the old 1k default when the prompt set was refreshed.

The reported metrics are time to first token, time to first answer token for reasoning models, output speed measured after the first token, total response time for 100 output tokens, end-to-end response time, cache hit rate, and average reasoning tokens. Model figures are the median over the trailing 72 hours, or 14 days for the 100k workload, so a leaderboard screenshot is a snapshot of a rolling window rather than a fixed result. Token counting uses the o200k_base tokenizer so the same text is measured identically across models, temperature is 0 for non-reasoning models and 0.6 for reasoning models with top_p at 1, and tests run from Google Cloud us-central1-a. Output speed for reasoning models is computed from the last 80% of answer chunks so thinking time does not flatter the number.

Providers also sign integrity terms: they may not fingerprint Artificial Analysis traffic, route it to dedicated hardware, serve a different quantization, or use unrepresentative concurrency. That is a meaningful constraint, because a speed leaderboard built on preferential serving would be worthless.

## Provider and endpoint benchmarking

Speed is a property of the endpoint, not just the model. The provider section compares endpoints on the same model, and the Endpoint Accuracy Index v1.0 measures how much accuracy a given endpoint preserves by re-running BFCL v4-500, HLE-250, and AA-LCR-25 against it, expressed as a percentage of a self-hosted reference endpoint at 100%. Below-reference results point to quantization, sampling defaults, or other endpoint-side configuration, and the speed-versus-price chart is the fastest way to spot a provider that is fast and accurate or fast and quietly degraded.

## Image, video and speech arenas

Beyond text, Artificial Analysis runs blind-preference arenas for images, video, and voice. The Image Arena and Video Arena produce Elo rankings with 95% confidence intervals for text-to-image, image editing, text-to-video, image-to-video, and video editing. Speech results come from a Voice Arena preference Elo, a controlled voice arena, and the AA-WER indexes for streaming and non-streaming transcription, plus a speech-to-speech index.

Arenas measure preference, which is a genuinely different signal from pass@1. Users like answers that are better in ways a rubric does not capture, but preference also rewards style, so an Elo lead is not a capability guarantee.

## How to use the numbers

A few habits make the data much more useful. Always pin the index version, because v4.3.2 is not comparable to v4.0 and the composition changes between versions. Read the category breakdown rather than the headline when your workload is narrow, since coding-heavy and long-context-heavy systems can rank very differently on the same total. For agentic work, compare agent variants including harness, and read cost per task rather than price per token. For anything latency-sensitive, remember the figures are a rolling median from one cloud region.

The limitations are real too. The Intelligence Index is English and text-first. Several constituent evaluations are private, so exact question sets are not reproducible by a third party even though the methodology is published. Tokenization differences, incomplete quantization disclosure, and time-to-first-token sensitivity to server location are all acknowledged by Artificial Analysis itself.

## Bottom line

Artificial Analysis is best understood as the shared measuring instrument for the model market, not as a verdict. Its value is that one independent party applies one disclosed method to every model and every endpoint, so a score can be quoted without arguing about whose harness produced it. Use the Intelligence Index for a general capability read, the Coding Agent Index for agentic work, the capability indexes for domain fit, and the cost and speed charts to find where your budget actually meets the frontier.
$body$::text AS body
)
UPDATE content_items AS item
SET
  body = artificial_analysis_entry.body,
  blocks = jsonb_build_array(
    jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', artificial_analysis_entry.body)
  ),
  sources = jsonb_build_array(
    jsonb_build_object('title', 'Artificial Analysis: Independent Analysis of AI Models and API Providers', 'url', 'https://artificialanalysis.ai/'),
    jsonb_build_object('title', 'Artificial Analysis Intelligence Benchmarking Methodology v4.3.2', 'url', 'https://artificialanalysis.ai/methodology/intelligence-benchmarking'),
    jsonb_build_object('title', 'Artificial Analysis Coding Agent Index and Methodology v1.5', 'url', 'https://artificialanalysis.ai/agents/coding-agents'),
    jsonb_build_object('title', 'Artificial Analysis Performance Benchmarking Methodology v2.2.0', 'url', 'https://artificialanalysis.ai/methodology/performance-benchmarking'),
    jsonb_build_object('title', 'Artificial Analysis About: the independent benchmarking company for AI', 'url', 'https://artificialanalysis.ai/about'),
    jsonb_build_object('title', 'Artificial Analysis LLM Leaderboard', 'url', 'https://artificialanalysis.ai/leaderboards/models')
  ),
  metadata = jsonb_set(
    jsonb_set(
      COALESCE(item.metadata, '{}'::jsonb),
      '{seoKeywords}',
      jsonb_build_array(
        'what is Artificial Analysis',
        'Artificial Analysis Intelligence Index',
        'Artificial Analysis Coding Agent Index',
        'Artificial Analysis leaderboard',
        'Artificial Analysis Openness Index',
        'Artificial Analysis cost per task',
        'Artificial Analysis output speed benchmark',
        'Artificial Analysis Terminal-Bench 4.0',
        'Artificial Analysis AA-Omniscience',
        'Artificial Analysis Endpoint Accuracy Index',
        'AI model benchmark methodology',
        'artificialanalysis.ai'
      ),
      true
    ),
    '{relatedTerms}',
    jsonb_build_array(
      'benchmark',
      'large-language-model',
      'inference',
      'agentic-ai',
      'agentic-workflows',
      'hallucination',
      'terminal-bench',
      'gpqa-diamond',
      'lmarena',
      'open-weight-model',
      'quantization',
      'context-window',
      'throughput',
      'api'
    ),
    true
  ),
  updated_at = NOW()
FROM artificial_analysis_entry
WHERE item.kind = 'glossary'
  AND item.slug = 'artificial-analysis'
  AND item.parent_slug = '';
