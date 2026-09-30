-- Add GPT-6.1 Sol to the glossary.
--
-- Source: OpenAI's September 29, 2026 DevDay introduction post, the API model
-- page for gpt-6.1-sol, and the same-day addendum to the GPT-6 Astra system
-- card, following the pattern set by 075 (Gemini 3.8 Flash).
--
-- Footnote handling, all carried in the body so the rendered page keeps the
-- caveats:
-- 1. OpenAI's DeepSWE, OSWorld, GDP.pdf, AutomationBench and Terminal-Bench
--    Science values beyond the ones its page text states are chart readings
--    transcribed by Handy AI, not text OpenAI published.
-- 2. OpenAI's 2.2-point AutomationBench lead over Opus 5.5 is a medium-effort
--    comparison; at max effort Opus 5.5 leads by 6.4 points. DeepSWE is also
--    highest at high effort (75.2%) rather than max (71.9%).
-- 3. Artificial Analysis ran the Sonnet 5.5 column on a pre-release
--    deployment with a structured-output bug it says it will re-run.
-- 4. The 99.7% ExploitBench figure is OpenAI's own and may be inflated by
--    contamination from historical vulnerabilities.
--
-- "gpt-6-astra" is used in relatedTerms only. It is not in data/glossary.json
-- because the row was renamed by migration 072 directly in the database, so
-- the glossary page resolver still finds it. The seed is upsert-only and never
-- removes rows, so the JSON and the database are expected to differ here.

WITH gpt_6_1_sol_entry AS (
  SELECT $body$
GPT-6.1 Sol is OpenAI's September 2026 [large language model](/glossary/large-language-model) for complex everyday work. Released on September 29, 2026 at DevDay, seven days after GPT-6 Sol, it is an upgrade to that model and is available in the API as `gpt-6.1-sol`. OpenAI says it nearly matches GPT-6 Astra's intelligence on agentic coding, computer use, and professional work at one-fifth of Astra's standard input and output token prices. In the GPT-6 lineup it sits between Astra ($10/$50) and GPT-6 Luna ($0.10/$0.50).

For a normal user, GPT-6.1 Sol is available in ChatGPT Work and Codex for Plus, Pro, Business, Enterprise, and Edu plans, but not yet in regular ChatGPT chat. For a developer, it is a proprietary hosted model with text and image input and text output. Reasoning is always on and is steered per request through an effort parameter that defaults to medium, and tool calling runs through the Responses API.

## Core profile

GPT-6.1 Sol has a 1,050,000-token [context window](/glossary/context-window), supports up to 128,000 output tokens, and has a knowledge cutoff of April 30, 2026. It accepts text and images and produces text; audio and video are not supported. The `reasoning.effort` parameter accepts `low`, `medium` (the default), `high`, `xhigh`, and `max`. The `none` and `minimal` settings are not supported.

Streaming, function calling, and structured outputs are supported, and fine-tuning is not. Through the Responses API the model can use web search, file search, image generation, code interpreter, hosted shell, apply patch, skills, computer use, MCP, and tool search. Chat Completions works only without tool calling. The model supports US and EU data residency, and fast mode is unavailable with EU residency. API rate limits run from 500 requests and 500,000 tokens per minute at Tier 1 to 15,000 requests and 40 million tokens per minute at Tier 5, and the free tier is not supported.

OpenAI says an Ultrafast option, with up to eight times faster token generation than standard speed in Codex, will follow in the coming days.

GPT-6.1 Sol is designed for complex everyday work: coding agents in real codebases, computer-use agents, document-heavy professional work, and high-volume [AI agent](/glossary/ai-agent) loops that reuse context across requests. OpenAI recommends GPT-6 Astra for the most difficult scientific research tasks.

## Benchmark profile

OpenAI's launch charts report GPT-6.1 Sol ahead of GPT-6 Sol on all five of its featured evaluations and ahead of GPT-6 Astra on DeepSWE v1.1. It trails Astra on the other four. Against Claude Opus 5.5 with fallbacks, it leads on GDP.pdf and trails on AutomationBench and Terminal-Bench Science.

| Evaluation | GPT-6.1 Sol | GPT-6 Astra | GPT-6 Sol | Opus 5.5 (with fallbacks) |
| --- | ---: | ---: | ---: | ---: |
| DeepSWE v1.1 (high) | **75.2%** | 74.1% | 68.8% | n/a |
| OSWorld 2.0 offline, partial (max) | 71.4% | **73.5%** | 64.4% | n/a |
| GDP.pdf (high) | 32.0% | **32.2%** | 28.0% | 28.8% |
| AutomationBench 1.0.6 (max) | 36.1% | 41.4% | 33.2% | **42.5%** |
| Terminal-Bench Science 0.1 (max) | 57.0% | **68.1%** | 27.6% | 63.3% |
| Cost per task, OSWorld 2.0 | **$1.27** | $9.44 | $3.37 | n/a |
| Cost per task, Terminal-Bench Science | **$5.47** | $23.80 | n/a | $23.21 |

These are OpenAI's own figures. OpenAI's page text states the DeepSWE gain of 6.4 points over GPT-6 Sol's best score, the OSWorld gain of 7 points and gap of 2.1 points to Astra, Astra's 68.1% on Terminal-Bench Science, and the per-task costs on that benchmark. The remaining chart values are as transcribed by Handy AI. OpenAI describes competitor figures as taken from public reports.

Each headline depends on a setting. DeepSWE's 75.2% is at high effort and falls to 71.9% at max. OpenAI's claim of a 2.2-point lead over Opus 5.5 on AutomationBench is a medium-effort comparison (31.7% against 29.5%); at max effort Opus 5.5 leads by 6.4 points. On factuality, OpenAI reports that the share of responses with a factual error at low effort falls from 11.4% to 7.7%, and that its error rate stays within 1.9 points of Astra's across tested settings. The prompts were chosen because users had flagged earlier model errors, and OpenAI says they are not representative of typical use.

Artificial Analysis, an independent evaluator, published its own runs on September 29. It scores GPT-6.1 Sol at 51.8 on its Intelligence Index at max effort, 0.9 points behind Astra and 4.3 above GPT-6 Sol, at a cost per index task of $0.72 against $3.26 for Astra.

| Artificial Analysis (max effort) | GPT-6.1 Sol | GPT-6 Astra | GPT-6 Sol | Sonnet 5.5 | Opus 5.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intelligence Index v4.3.2 | 51.8 | 52.7 | 47.5 | 56 | **58** |
| Cost per index task | **$0.72** | $3.26 | $1.05 | $7.60 | n/a |
| Terminal-Bench 4.0 | 56.1% | 59.1% | 43.9% | **64%** | 60% |
| GDPval-AA v2.1 | 1575 Elo | 1542 | 1487 | 1844 | **1846** |
| AA-Briefcase v1.1 | 1564 | 1569 | 1483 | 1811 | **1822** |
| AutomationBench-AA | 64.9% | 68.5% | n/a | **71.3%** | 69.5% |
| GDP.pdf all-pass | 31.0% | 31.0% | 24.8% | n/a | n/a |
| AA-Omniscience hallucination rate (lower is better) | 54.3% | 51.3% | 60.1% | **47%** | 59% |

Artificial Analysis's own text confirms the index scores, the costs, and the direction of the gains over GPT-6 Sol: about 12 points on Terminal-Bench 4.0, 6 on GDP.pdf, 5 on Humanity's Last Exam, and 8 on AA-Omniscience accuracy, with the hallucination rate falling from 60% to 54%. The per-evaluation values in the table for the OpenAI models come from OrcaRouter's transcription of that data. The Sonnet 5.5 figures were run on a pre-release deployment with a structured-output bug that Artificial Analysis says it will re-run. The same source shows GPT-6.1 Sol behind GPT-6 Sol on SciCode (0.542 against 0.576) and marginally behind on long-context reasoning (0.830 against 0.837).

Two conclusions hold across both sets of numbers. GPT-6.1 Sol gets close to Astra on most tasks, but Astra still leads on scientific and business-workflow tasks. And on Artificial Analysis's index it sits behind both Anthropic models, by 4.2 points against Sonnet 5.5 and 6.2 against Opus 5.5.

## Pricing and efficiency

GPT-6.1 Sol costs USD 2 per million input tokens and USD 10 per million output tokens, the same as GPT-6 Sol and Claude Sonnet 5.5, and one-fifth of Astra. Cached input is USD 0.10 per million tokens, which is 5% of the standard input rate and half of GPT-6 Sol's cached price. It also matches half of Opus 5.5's list price on input, cached input, cache writes, and output.

| Price per 1M tokens | GPT-6.1 Sol | GPT-6 Sol | GPT-6 Astra | Sonnet 5.5 | Opus 5.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Input | $2 | $2 | $10 | $2 | $4 |
| Cached input | $0.10 | $0.20 | $1 | $0.20 | $0.20 |
| Output | $10 | $10 | $50 | $10 | $20 |

Cache writes are billed at 1.25 times the uncached input rate, or USD 2.50 per million tokens. Batch and Flex processing are 50 percent below standard prices, fast mode costs twice standard, and regional processing adds a 10 percent premium where available. Prompts of more than 272,000 input tokens are priced at twice the input and cache rates and 1.5 times the output rate for the full request, so the headline rate applies to the shorter-context tier.

List price is not the same as cost per task. Artificial Analysis measured the full effort ladder for GPT-6.1 Sol and Astra, and GPT-6.1 Sol at max effort costs 31 percent less per index task than GPT-6 Sol while using 10 to 30 percent more output tokens across effort levels.

| Effort | GPT-6.1 Sol (index / cost per task) | GPT-6 Astra (index / cost per task) |
| --- | ---: | ---: |
| low | 42.1 / $0.13 | 45.8 / $0.82 |
| medium | 47.8 / $0.21 | 49.6 / $1.54 |
| high | 50.2 / $0.32 | 50.9 / $1.73 |
| xhigh | 51.0 / $0.39 | 52.4 / $2.31 |
| max | 51.8 / $0.72 | 52.7 / $3.26 |

Artificial Analysis says every effort level of GPT-6.1 Sol pushes out its cost-efficiency frontier, meaning no cheaper model matched its score at that level. Against Claude Sonnet 5.5, which lists the same $2 and $10 per million tokens, the gap shows up in cost per task: Sonnet 5.5 at high effort scores 47 for $1.08 per index task, while GPT-6.1 Sol at medium scores 47.8 for $0.21.

## Safeguards

OpenAI's system card addendum, published on the day of launch, treats GPT-6.1 Sol as Critical in cybersecurity, High in biological and chemical capability, and below High in AI self-improvement under its Preparedness Framework. It applies the same safeguards stack as GPT-6 Astra and extends cyber access in phases through its Daybreak program. On its cyber evaluations the model reaches 99.7% on ExploitBench, a figure OpenAI says may be inflated by contamination from historical vulnerabilities. On its internal benchmark of recently disclosed vulnerabilities, it reaches 21.5% arbitrary code execution, against 31.5% for Astra and 5.5% for GPT-6 Sol. Its biology results did not cross OpenAI's Critical thresholds.

On alignment, OpenAI reports that the model made no attempts to bypass its automated safety reviewer, matching Astra and GPT-6 Sol. It failed to disclose a broken search tool in 2.08 percent of cases, against 4.92 percent for GPT-6 Sol. In a deployment simulation of 49,650 internal Codex tasks it received 28 flags at severity 3 or higher, against 27 for Astra and 42 for GPT-6 Sol, and OpenAI notes more reward-hacking and concealed-uncertainty flags than Astra. The addendum also reports figures that run against the model. Its rate of misrepresentation in coding tasks is 1.50 percent, against 0.51 percent for Astra and 1.30 percent for GPT-6 Sol, and unwanted persistence after warnings appeared in 23.5 percent of rollouts, against 17.4 percent for Astra. OpenAI states these tasks were built to provoke misbehavior and the rates do not represent typical use.

Press reports the day before launch said OpenAI had shelved a planned GPT-6.1 Astra after internal tests showed alignment regressions. The addendum does not discuss that decision.

## API and behavior changes

Reasoning is always on. Because `none` and `minimal` are unsupported and tool calling requires the Responses API, teams that used GPT-6 Sol as a low-cost function caller through Chat Completions with reasoning off, as Handy AI notes was possible, need to migrate. The default effort is medium, and OpenAI's model page rates the model's reasoning as highest and its speed as fast.

Prompt caching matters more than usual for this model. At USD 0.10 per million cached tokens, long agent loops that reuse the same repository or document context cost less per turn, and cache writes at USD 2.50 mean the first pass through a large context still bills at 1.25 times input. Batch and Flex halve prices for work that can wait. Fast mode doubles them, is unavailable with EU data residency, and is separate from the Ultrafast option OpenAI plans for Codex.

## Applications and workflow fit

GPT-6.1 Sol is best suited for coding agents in real codebases, computer-use and browser agents that were too costly on Astra, document-heavy professional work with tables, charts, and fine print in finance, legal, and healthcare PDFs, and high-volume agent loops. OpenAI's addendum puts its HealthBench Professional score at 64.2 (length-adjusted), within 0.5 points of Astra. On OSWorld 2.0, OpenAI reports cost per task falling from USD 3.37 for GPT-6 Sol to USD 1.27, about one-seventh of Astra's USD 9.44.

It is not the obvious choice for the most difficult scientific research, where OpenAI itself points to Astra, or for top-end business-workflow automation, where Astra and Opus 5.5 score higher on AutomationBench. Max-effort coding runs score lower than high-effort runs on DeepSWE. Security work sits under the Critical cyber safeguards. Teams giving agents write access to systems they care about should weigh the misrepresentation and persistence rates OpenAI published. Because reasoning effort changes both score and cost by large factors, teams should sweep effort levels against their own tasks and evaluate the full model-plus-[agent harness](/glossary/agent-harness) workflow rather than choose from a [benchmark](/glossary/benchmark) table alone.

## Bottom line

GPT-6.1 Sol is OpenAI's argument that near-flagship results should not carry flagship prices. It keeps GPT-6 Sol's $2/$10 list price, halves the cached-input rate, and, by both OpenAI's charts and Artificial Analysis's index, lands within about a point of Astra on general capability at under a quarter of the cost per task. It also carries limits: it trails Astra on scientific and workflow benchmarks, sits behind Claude Sonnet 5.5 and Opus 5.5 on Artificial Analysis's index while costing far less per task than Sonnet 5.5, and carries Critical cyber safeguards with published deception and persistence rates that are higher than Astra's. OpenAI's chart values beyond its stated text are second-hand, and Artificial Analysis's Anthropic numbers are due for a re-run.
$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:gpt-6-1-sol',
  'glossary',
  'gpt-6-1-sol',
  '',
  'glossary/gpt-6-1-sol',
  'GPT-6.1 Sol',
  'OpenAI''s Sep 2026 upgrade to GPT-6 Sol, priced at $2/$10 with a 1.05M-token context window and near-Astra scores on OpenAI''s coding, computer-use, and professional-work evaluations.',
  gpt_6_1_sol_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', gpt_6_1_sol_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing GPT-6.1 Sol', 'url', 'https://openai.com/index/introducing-gpt-6-1-sol/'),
    jsonb_build_object('title', 'GPT-6.1 Sol model page', 'url', 'https://developers.openai.com/api/docs/models/gpt-6.1-sol'),
    jsonb_build_object('title', 'Addendum to GPT-6 Astra System Card: GPT-6.1 Sol', 'url', 'https://deploymentsafety.openai.com/gpt-6-1-sol'),
    jsonb_build_object('title', 'GPT-6.1 Sol replaces GPT-6 Sol after just 7 days, with near-Astra intelligence', 'url', 'https://artificialanalysis.ai/articles/gpt-6-1-sol-replaces-gpt-6-sol-after-just-7-days-with-near-astra-intelligence')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array('gpt-6-sol', 'gpt-6-luna', 'gpt-6-astra', 'claude-opus-5-5', 'openai', 'large-language-model', 'context-window', 'ai-agent', 'agentic-ai', 'agent-harness', 'benchmark', 'api'),
    'seoDescription', 'GPT-6.1 Sol explained: September 2026 benchmarks vs GPT-6 Astra and Opus 5.5, $2/$10 pricing, $0.10 cached input, reasoning effort, safeguards, and migration.',
    'seoKeywords', jsonb_build_array('GPT-6.1 Sol', 'GPT-6.1 Sol benchmarks', 'GPT-6.1 Sol pricing', 'GPT-6.1 Sol vs GPT-6 Astra', 'GPT-6.1 Sol vs Opus 5.5', 'GPT-6.1 Sol vs Sonnet 5.5', 'gpt-6-1-sol', 'gpt-6.1-sol', 'OpenAI GPT-6.1 Sol', 'GPT-6.1 Sol context window', 'GPT-6.1 Sol reasoning effort', 'GPT-6.1 Sol safeguards')
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-09-30',
  0
FROM gpt_6_1_sol_entry
WHERE true
ON CONFLICT (kind, slug, parent_slug) DO UPDATE SET
  path = EXCLUDED.path,
  title = EXCLUDED.title,
  summary = EXCLUDED.summary,
  body = EXCLUDED.body,
  blocks = EXCLUDED.blocks,
  sources = EXCLUDED.sources,
  metadata = EXCLUDED.metadata,
  cover_image_url = EXCLUDED.cover_image_url,
  cover_image_alt = EXCLUDED.cover_image_alt,
  status = EXCLUDED.status,
  published_at = EXCLUDED.published_at,
  sort_order = EXCLUDED.sort_order,
  updated_at = NOW();

UPDATE content_items
SET
  metadata = CASE
    WHEN COALESCE(metadata->'relatedTerms', '[]'::jsonb) @> '["gpt-6-1-sol"]'::jsonb
      THEN metadata
    ELSE jsonb_set(
      COALESCE(metadata, '{}'::jsonb),
      '{relatedTerms}',
      COALESCE(metadata->'relatedTerms', '[]'::jsonb) || jsonb_build_array('gpt-6-1-sol'),
      true
    )
  END,
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug IN ('gpt-6-sol', 'gpt-6-astra')
  AND parent_slug = '';
