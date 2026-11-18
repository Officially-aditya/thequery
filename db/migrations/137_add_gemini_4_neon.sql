-- Add Gemini 4 Neon to the glossary and model catalog.
--
-- Source: Google DeepMind's November 18, 2026 launch announcement, the
-- gemini-4-neon API documentation, and independent benchmark evaluations
-- from Artificial Analysis and Vals AI.
--
-- Footnotes and caveats carried in the body so the rendered model card retains
-- full technical nuance:
-- 1. Fourth model in the Gemini 4 generation, bridging Hydrogen's speed with Carbon's reasoning.
-- 2. Artificial Analysis scores Neon at 61 on its Intelligence Index v5.0.1 (leads class).
-- 3. Terminal-Bench-Science 0.1 at 70.5% leads class ahead of GPT-6.2 Sol (~68%) and Claude Opus 5.5 (63.3%).
-- 4. Pricing is $3.00/1M input and $15.00/1M output, with context caching at $0.15/1M (95% discount).
-- 5. Context window is 1M unified tokens with up to 64K default output (1M with Long Decode Continuation).

WITH gemini_4_neon_entry AS (
  SELECT $body$Gemini 4 Neon is Google DeepMind's November 2026 mid-tier frontier [large language model](/glossary/large-language-model) and the fourth model in the Gemini 4 generation, joining [Gemini 4 Argon](/glossary/gemini-4-argon), [Gemini 4 Carbon](/glossary/gemini-4-carbon), and [Gemini 4 Hydrogen](/glossary/gemini-4-hydrogen). Released on November 18, 2026 as the API model `gemini-4-neon`, it is engineered to bridge high-velocity developer execution with heavyweight multi-step reasoning, delivering near-Carbon reasoning quality at Flash-tier speeds and economics.

For developers and enterprises, Gemini 4 Neon is available through Google AI Studio, Google Cloud Vertex AI, and the Gemini API. For subscribers, it powers advanced features within Google AI Pro and Google AI Ultra across the Gemini App. The model accepts text and multimodal inputs (including high-resolution images, video up to one hour, audio up to 9.5 hours, and technical PDF blueprints up to 1,000 pages) and returns text, structured JSON, and tool invocations. Configurable thinking levels (`low`, `medium`, `high` default, and `xhigh`) allow developers to explicitly balance deliberation depth against execution latency.

## Core profile

| Specification | Gemini 4 Neon |
| --- | --- |
| Developer | Google DeepMind |
| Release date | November 18, 2026 |
| API model ID | `gemini-4-neon` |
| Context window | 1,048,576 tokens (1M unified tokens) |
| Maximum output | 65,536 tokens (up to 1,048,576 via Long Decode Continuation) |
| Knowledge cutoff | March 2026 |
| Input modalities | Text, images, video, audio, and documents (PDF) |
| Output modalities | Text (including code, structured JSON, tool calls) |
| Reasoning / thinking | Configurable thinking levels (`low`, `medium`, `high` default, `xhigh`), plus non-thinking mode |
| Standard pricing (1M) | \$3.00 base input / \$15.00 base output |
| Prompt cache reads | \$0.15 per million tokens (95% discount) |
| Cache writes / storage | Included in standard input / tiered hourly storage for persistent caches |
| Batch / flex discount | 50% discount on standard rates (\$1.50 input / \$7.50 output) |
| Tool integration | Function calling, structured outputs, code execution, web search grounding, MCP |
| Availability | Gemini API, Google AI Studio, Vertex AI, Google AI Pro / AI Ultra in Gemini App |
| Downloadable weights | Proprietary; hosted cloud API |

The unified 1M-token [context window](/glossary/context-window) bounds the total volume of conversational history, reference libraries, repository files, audio recordings, or video frames retained in working memory during an [inference](/glossary/inference) pass. Output capacity defaults to 65,536 tokens per request, with Long Decode Continuation enabling recursive generation up to the full 1M-token ceiling across connected Interactions API sessions.

## Lineup positioning and generation changes

Within Google DeepMind's Gemini 4 hierarchy, Gemini 4 Neon anchors the high-performance developer tier:

- **Versus Gemini 4 Carbon:** Carbon is Google's heaviest flagship, scoring 59 on the Artificial Analysis Intelligence Index v5.0 and excelling at multi-day autonomous software repair and deep cybersecurity validation under Fairwind. Neon provides near-Carbon reasoning quality—scoring 61 on the Artificial Analysis Intelligence Index v5.0.1 and leading Terminal-Bench-Science 0.1 at 70.5%—at Flash-tier speed and lower cost (\$3.00/\$15.00 per 1M tokens), with immediate general availability.
- **Versus Gemini 4 Hydrogen:** [Gemini 4 Hydrogen](/glossary/gemini-4-hydrogen) is the ultra-lightweight entry point at \$0.30/\$1.20 per million tokens. While Hydrogen focuses on sub-second time-to-first-token latency and raw throughput (184 tokens/second), Neon provides significantly higher reasoning depth, outperforming Hydrogen on Terminal-Bench (65.2% vs 58.4%), SWE-sweep (39.0% vs 7.8%), and Agents' Last Exam (75.8% vs 71.2%).
- **Versus Gemini 4 Argon:** Argon served as the initial preview for Gemini 4's long-horizon capabilities under restricted access. Neon surpasses Argon across benchmarks (Artificial Analysis Index 61 vs 50, Terminal-Bench 65.2% vs 57.1%) and replaces it as Google's production mid-tier model for open developer adoption.
- **Versus GPT-6.2 Sol & Claude Opus 5.5:** Neon competes directly with OpenAI's [GPT-6.2 Sol](/glossary/gpt-6-2-sol) (\$2.00/\$10.00) and Anthropic's [Claude Opus 5.5](/glossary/claude-opus-5-5) (\$5.00/\$25.00). On Terminal-Bench-Science 0.1, Neon leads at 70.5% (versus Sol's ~68% and Opus 5.5's 63.3%). On SWE-sweep under the provider harness, Neon reaches 39.0%, and on the Artificial Analysis Intelligence Index v5.0.1, Neon achieves a score of 61 (ahead of Opus 5.5 at 59 and Sol at 57).

## Benchmark profile

Google DeepMind's launch evaluation and independent benchmarks establish Gemini 4 Neon as a leading mid-tier reasoning model across coding, science, mathematics, and agentic workflows:

| Evaluation | Gemini 4 Neon | Gemini 4 Carbon | Gemini 4 Hydrogen | GPT-6.2 Sol | Claude Opus 5.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AA Intelligence Index v5.0.1 | **61** | 60 | 52 | 57 | 59 |
| Terminal-Bench-Science 0.1 | **70.5%** | 64.8% | ~58% | ~68% | 63.3% |
| Terminal-Bench 4.1 (adjusted) | **65.2%** | 65.2% (v4.0) | 58.4% (v4.0) | 61.5% | 65.8% |
| Agents' Last Exam v1.1 | **75.8%** | 77.4% | 71.2% | ~73% | 71.6% |
| SWE-sweep (provider harness) | **39.0%** | 9.6% | 7.8% | 8.3% | 10.1% |
| CursorBench 4.1 | **65.4%** | — | — | — | 58.1% |
| RiemannBench | **84.0%** | 80.0% | — | — | 69.6% |
| HLE-Diamond (science) | **71.3%** | — | — | — | — |
| FrontierCode 2.0 (Cognition) | 46.2% | 41.1% | — | **49.3%** (max) | 47.3% |
| Vals Index v2.2 | **70.9%** | 70.4% | — | — | 70.2% |

Evaluation conditions provide essential operational context:

On Artificial Analysis's Intelligence Index v5.0.1, Gemini 4 Neon posts a score of 61, ranking first in its class ahead of Claude Opus 5.5 (59) and GPT-6.2 Sol (57). On Terminal-Bench-Science 0.1, Neon achieves 70.5% at its default high thinking setting, outperforming Sol (~68%) and Opus 5.5 (63.3%).

In software engineering evaluations, Neon marks a significant advance in repository-scale problem solving. On SWE-sweep under the provider harness, Neon resolves 39.0% of tasks (and 12.8% on the mini-SWE harness). On CursorBench 4.1, Neon achieves 65.4%, outpacing Claude Opus 5.5 (58.1%). On advanced mathematical proofs (RiemannBench), Neon leads at 84.0% compared to Carbon's 80.0%.

On Cognition's FrontierCode 2.0, Neon achieves 46.2%, trailing GPT-6.2 Sol's maximum reasoning effort score (49.3%) while outperforming earlier generation frontier systems.

## Pricing and efficiency

Google DeepMind lists the following standard API pricing for Gemini 4 Neon, effective November 18, 2026:

| Token Type | Rate (per 1M tokens) |
| --- | ---: |
| Standard input | \$3.00 |
| Context cache reads | \$0.15 |
| Standard output (incl. thinking) | \$15.00 |
| Batch / flex input | \$1.50 |
| Batch / flex output | \$7.50 |

Context caching economics provide significant operational savings. Reading an 800,000-token repository or document corpus repeatedly costs \$0.12 per turn at \$0.15 per million tokens, delivering a 95% discount over uncached input (\$2.40). For non-interactive asynchronous workflows, Google's 50% Batch API discount lowers base rates to \$1.50 input and \$7.50 output per million tokens. There is no free API tier for `gemini-4-neon`, but the model is bundled for subscribers in Google AI Pro and AI Ultra.

## Safeguards and safety profile

Gemini 4 Neon inherits Google DeepMind's Gemini 4 defense-in-depth safety and alignment stack:

- **Agentic safety and honesty:** Evaluated on Google's agentic safety benchmark suite, Neon records a 0.09% misaligned action rate. On agent honesty evaluations, the model achieves a 99.2% pass rate for accurate claims and transparent capability limitations.
- **Indirect prompt-injection (IPI) resilience:** Built with enhanced input verification layers, Neon demonstrates state-of-the-art resilience against adversarial prompt injections embedded inside untrusted web pages, retrieved search snippets, code comments, and corporate documents.
- **Fairwind Program governance:** While high-risk offensive cyber vulnerability exploitation capabilities are restricted to vetted security defenders under the Fairwind Program, defensive auditing, patch validation, and vulnerability explanation are fully available to general API developers.
- **Enterprise data privacy:** Fully covered by Google Cloud Vertex AI's commitments, including zero data retention options and intellectual property indemnification.

## API and behavior changes

Developers integrating `gemini-4-neon` should account for several key API behavioral characteristics:

- **Thinking controls:** Reasoning is configurable via the `thinking_config` parameter. Developers can set thinking effort to `low`, `medium`, `high` (default), or `xhigh`. For latency-critical conversational applications, non-thinking mode can be explicitly enabled (`mode: "off"`).
- **Long Decode Continuation:** Output generation beyond single-request HTTP limits is supported via Long Decode, allowing agents to produce up to 1,048,576 tokens without conversational state fragmentation.
- **Multimodal ingestion:** Full omnimodal support allows native streaming and ingestion of text, high-resolution images, video (up to 1 hour), audio (up to 9.5 hours), and dense PDF documents (up to 1,000 pages) in a single request.
- **Tool integration and MCP:** Native support for function calling, structured outputs with strict JSON Schema verification, Google Search grounding, Python code execution, and Model Context Protocol (MCP) server endpoints.

## Applications and workflow fit

Gemini 4 Neon is best suited for:

- **Repository-scale software engineering:** Autonomous coding agents executing multi-file refactoring, terminal debugging, and test suite generation where near-Carbon quality is required without Carbon's latency overhead.
- **Computational science and mathematics:** Research workflows analyzing complex scientific literature, executing algorithmic proofs, and validating experimental datasets (70.5% on Terminal-Bench-Science, 84.0% on RiemannBench).
- **Multimodal document analysis:** Processing multi-hundred-page technical blueprints, financial reports with embedded charts, and recorded technical lectures.
- **High-throughput enterprise agents:** Multi-turn autonomous agent loops that repeatedly query massive knowledge bases utilizing \$0.15 context caching reads.

It is less suited for:

- Ultra-low-latency real-time voice or customer chat triage requiring sub-300ms time-to-first-token (better served by [Gemini 4 Hydrogen](/glossary/gemini-4-hydrogen)).
- Multi-week autonomous zero-day exploit validation and specialized hardware-level firmware remediation, which remains restricted to [Gemini 4 Carbon](/glossary/gemini-4-carbon) under Fairwind.

## Bottom line

Gemini 4 Neon establishes an optimal balance of reasoning depth, coding capability, and operational cost within the Gemini 4 generation. Combining a 1M-token unified context window, adjustable thinking levels, an Artificial Analysis Intelligence Index score of 61, 70.5% on Terminal-Bench-Science 0.1, and \$3.00/\$15.00 token pricing with \$0.15 cached reads, it delivers near-Carbon reasoning quality at accessible production economics.$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:gemini-4-neon',
  'glossary',
  'gemini-4-neon',
  '',
  'glossary/gemini-4-neon',
  'Gemini 4 Neon',
  'Google DeepMind''s Nov 2026 mid-tier frontier model in the Gemini 4 family, featuring 1M context, near-Carbon reasoning, and $3.00/$15.00 pricing with $0.15 cache reads.',
  gemini_4_neon_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', gemini_4_neon_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing Gemini 4 Neon', 'url', 'https://blog.google/technology/ai/gemini-4-neon/'),
    jsonb_build_object('title', 'Gemini 4 Neon documentation', 'url', 'https://ai.google.dev/gemini-api/docs/models/gemini-4-neon'),
    jsonb_build_object('title', 'Gemini 4 Neon model evaluation', 'url', 'https://storage.googleapis.com/deepmind-media/gemini/gemini_4_neon_evaluation.pdf'),
    jsonb_build_object('title', 'Google Cloud Vertex AI pricing', 'url', 'https://cloud.google.com/vertex-ai/pricing')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array(
      'gemini-4-carbon',
      'gemini-4-argon',
      'gemini-4-hydrogen',
      'gemini-3-8-flash',
      'google-deepmind',
      'large-language-model',
      'ai-agent',
      'agentic-ai',
      'agent-harness',
      'context-window',
      'benchmark',
      'api'
    ),
    'seoDescription', 'Gemini 4 Neon model card: Google DeepMind''s Nov 2026 mid-tier frontier model, 1M context window, benchmark results, pricing, and availability.',
    'seoKeywords', jsonb_build_array(
      'Gemini 4 Neon',
      'Gemini 4 Neon benchmarks',
      'Gemini 4 Neon pricing',
      'Gemini 4 Neon model card',
      'Gemini 4 Neon context window',
      'gemini-4-neon',
      'Google DeepMind Gemini 4 Neon',
      'Gemini 4 Neon API availability',
      'Gemini 4 Neon vs Gemini 4 Carbon',
      'Gemini 4 Neon vs GPT-6.2 Sol'
    )
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-11-18',
  0
FROM gemini_4_neon_entry
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
    WHEN COALESCE(metadata->'relatedTerms', '[]'::jsonb) @> '["gemini-4-neon"]'::jsonb
      THEN metadata
    ELSE jsonb_set(
      COALESCE(metadata, '{}'::jsonb),
      '{relatedTerms}',
      COALESCE(metadata->'relatedTerms', '[]'::jsonb) || jsonb_build_array('gemini-4-neon'),
      true
    )
  END,
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug IN ('gemini-4-carbon', 'gemini-4-argon', 'gemini-4-hydrogen', 'google-deepmind')
  AND parent_slug = '';

INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at, updated_at
)
VALUES (
  'gemini-4-neon',
  'Gemini 4 Neon',
  'Google DeepMind',
  DATE '2026-11-18',
  DATE '2026-11-18',
  'proprietary',
  'Gemini 4',
  $specs${
  "Developer": "Google DeepMind",
  "Release date": "2026-11-18",
  "API model ID": "gemini-4-neon",
  "Context window": "1M tokens",
  "Max output": "64K tokens (1M with Long Decode Continuation)",
  "Knowledge cutoff": "Mar 2026",
  "Reasoning / effort": "Configurable thinking levels (low, medium, high default, xhigh), plus non-thinking mode",
  "Input / 1M tokens": "$3.00",
  "Cached input / 1M": "$0.15",
  "Cache write / 1M": "Included in standard input",
  "Output / 1M tokens": "$15.00",
  "Batch / flex discount": "50% off standard rates ($1.50 input / $7.50 output)",
  "Text input": "Yes",
  "Image / vision input": "Yes",
  "Audio input": "Yes",
  "Video input": "Yes",
  "File / document input": "Yes (PDF, docs)",
  "Text output": "Yes",
  "Image output": "No native image output",
  "Audio output": "No",
  "Video output": "No",
  "Tool / function calling": "Yes, function calling, structured outputs, code execution, web search, MCP",
  "Computer use": "Supported via agentic tool calling",
  "Terminal-Bench 4.1": "65.2%",
  "Terminal-Bench": "65.2% (Terminal-Bench 4.1; cheating-adjusted)",
  "Terminal-Bench-Science 0.1": "70.5%",
  "SWE-sweep": "39.0%",
  "Agents' Last Exam": "75.8%",
  "RiemannBench": "84.0%",
  "FrontierCode 2.0": "46.2%",
  "API access": "Yes",
  "Product access": "Gemini API + Google AI Studio + Google Cloud Vertex AI + Google AI Pro/Ultra in Gemini App",
  "Weights / license": "Proprietary",
  "Primary focus": "Mid-tier frontier reasoning, repository-level coding, STEM problem-solving, and cost-efficient agentic workflows",
  "Long-horizon work": "Built for persistent agent loops that reuse large repository or document context with $0.15 cached reads",
  "Safety / approvals": "Production safeguards with 0.09% misaligned action rate, 99.2% honesty pass rate, and Fairwind cyber defense governance"
}$specs$::jsonb,
  $sources$[
  {
    "title": "Introducing Gemini 4 Neon",
    "url": "https://blog.google/technology/ai/gemini-4-neon/"
  },
  {
    "title": "Gemini 4 Neon documentation",
    "url": "https://ai.google.dev/gemini-api/docs/models/gemini-4-neon"
  },
  {
    "title": "Gemini 4 Neon model evaluation",
    "url": "https://storage.googleapis.com/deepmind-media/gemini/gemini_4_neon_evaluation.pdf"
  },
  {
    "title": "Google Cloud Vertex AI pricing",
    "url": "https://cloud.google.com/vertex-ai/pricing"
  }
]$sources$::jsonb,
  'Google DeepMind''s November 2026 mid-tier frontier model in the Gemini 4 generation. 1M unified context window, 64K default output (1M with Long Decode Continuation), $3.00/$15.00 per 1M tokens with $0.15 context cache reads. Scores 61 on Artificial Analysis Intelligence Index v5.0.1, 70.5% on Terminal-Bench-Science 0.1, and 39.0% on SWE-sweep.',
  '{"launch_stage": "ga", "verification_pass": "gemini-4-neon-2026-11-18", "default_effort": "high"}'::jsonb,
  NOW(),
  NOW()
)
ON CONFLICT (slug) DO UPDATE SET
  name = EXCLUDED.name,
  developer = EXCLUDED.developer,
  release_date = EXCLUDED.release_date,
  ga_date = EXCLUDED.ga_date,
  access = EXCLUDED.access,
  family = EXCLUDED.family,
  comparison_data = EXCLUDED.comparison_data,
  sources = EXCLUDED.sources,
  notes = EXCLUDED.notes,
  metadata = COALESCE(models.metadata, '{}'::jsonb) || EXCLUDED.metadata,
  verified_at = NOW(),
  updated_at = NOW();

INSERT INTO model_benchmarks (
  id, model_slug, category, benchmark_name, benchmark_version,
  score_numeric, score_display, score_unit, tools, reasoning_effort,
  harness, evaluator, evaluation_date, source, notes
)
SELECT
  evidence.id,
  evidence.model_slug,
  evidence.category,
  evidence.benchmark_name,
  evidence.benchmark_version,
  evidence.score_numeric,
  evidence.score_display,
  evidence.score_unit,
  evidence.tools,
  evidence.reasoning_effort,
  evidence.harness,
  evidence.evaluator,
  evidence.evaluation_date::date,
  evidence.source,
  evidence.notes
FROM jsonb_to_recordset($benchmarks$[
  {
    "id": "tq-20261118-gemini4neon-terminalbenchscience",
    "model_slug": "gemini-4-neon",
    "category": "coding",
    "benchmark_name": "Terminal-Bench-Science 0.1",
    "benchmark_version": null,
    "score_numeric": 70.5,
    "score_display": "70.5%",
    "score_unit": "percent",
    "tools": true,
    "reasoning_effort": "high",
    "harness": null,
    "evaluator": "Google DeepMind",
    "evaluation_date": "2026-11-18",
    "source": "https://blog.google/technology/ai/gemini-4-neon/",
    "notes": "Self-reported launch result at high thinking level."
  },
  {
    "id": "tq-20261118-gemini4neon-terminalbench41",
    "model_slug": "gemini-4-neon",
    "category": "coding",
    "benchmark_name": "Terminal-Bench 4.1",
    "benchmark_version": null,
    "score_numeric": 65.2,
    "score_display": "65.2%",
    "score_unit": "percent",
    "tools": true,
    "reasoning_effort": "high",
    "harness": null,
    "evaluator": "Google DeepMind",
    "evaluation_date": "2026-11-18",
    "source": "https://blog.google/technology/ai/gemini-4-neon/",
    "notes": "Cheating-adjusted Terminal-Bench 4.1 evaluation."
  },
  {
    "id": "tq-20261118-gemini4neon-swesweep",
    "model_slug": "gemini-4-neon",
    "category": "coding",
    "benchmark_name": "SWE-sweep",
    "benchmark_version": null,
    "score_numeric": 39,
    "score_display": "39.0%",
    "score_unit": "percent",
    "tools": true,
    "reasoning_effort": "high",
    "harness": "provider",
    "evaluator": "Google DeepMind",
    "evaluation_date": "2026-11-18",
    "source": "https://blog.google/technology/ai/gemini-4-neon/",
    "notes": "Provider harness at high thinking level."
  },
  {
    "id": "tq-20261118-gemini4neon-agentslastexam",
    "model_slug": "gemini-4-neon",
    "category": "agentic_computer_use",
    "benchmark_name": "Agents' Last Exam",
    "benchmark_version": "v1.1",
    "score_numeric": 75.8,
    "score_display": "75.8%",
    "score_unit": "percent",
    "tools": true,
    "reasoning_effort": "high",
    "harness": null,
    "evaluator": "Google DeepMind",
    "evaluation_date": "2026-11-18",
    "source": "https://blog.google/technology/ai/gemini-4-neon/",
    "notes": "Agents' Last Exam v1.1 evaluation."
  },
  {
    "id": "tq-20261118-gemini4neon-riemannbench",
    "model_slug": "gemini-4-neon",
    "category": "math_reasoning",
    "benchmark_name": "RiemannBench",
    "benchmark_version": null,
    "score_numeric": 84,
    "score_display": "84.0%",
    "score_unit": "percent",
    "tools": false,
    "reasoning_effort": "high",
    "harness": null,
    "evaluator": "Google DeepMind",
    "evaluation_date": "2026-11-18",
    "source": "https://blog.google/technology/ai/gemini-4-neon/",
    "notes": "Advanced mathematical reasoning evaluation."
  },
  {
    "id": "tq-20261118-gemini4neon-frontiercode20",
    "model_slug": "gemini-4-neon",
    "category": "coding",
    "benchmark_name": "FrontierCode 2.0",
    "benchmark_version": null,
    "score_numeric": 46.2,
    "score_display": "46.2%",
    "score_unit": "percent",
    "tools": true,
    "reasoning_effort": "high",
    "harness": "Cognition",
    "evaluator": "Google DeepMind",
    "evaluation_date": "2026-11-18",
    "source": "https://blog.google/technology/ai/gemini-4-neon/",
    "notes": "Cognition FrontierCode 2.0 evaluation."
  }
]$benchmarks$::jsonb) AS evidence(
  id TEXT, model_slug TEXT, category TEXT, benchmark_name TEXT, benchmark_version TEXT,
  score_numeric DOUBLE PRECISION, score_display TEXT, score_unit TEXT, tools BOOLEAN,
  reasoning_effort TEXT, harness TEXT, evaluator TEXT, evaluation_date TEXT,
  source TEXT, notes TEXT
)
ON CONFLICT (id) DO UPDATE SET
  model_slug = EXCLUDED.model_slug,
  category = EXCLUDED.category,
  benchmark_name = EXCLUDED.benchmark_name,
  benchmark_version = EXCLUDED.benchmark_version,
  score_numeric = EXCLUDED.score_numeric,
  score_display = EXCLUDED.score_display,
  score_unit = EXCLUDED.score_unit,
  tools = EXCLUDED.tools,
  reasoning_effort = EXCLUDED.reasoning_effort,
  harness = EXCLUDED.harness,
  evaluator = EXCLUDED.evaluator,
  evaluation_date = EXCLUDED.evaluation_date,
  source = EXCLUDED.source,
  notes = EXCLUDED.notes,
  updated_at = NOW();
