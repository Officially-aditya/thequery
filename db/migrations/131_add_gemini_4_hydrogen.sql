-- Add Gemini 4 Hydrogen to the glossary and model catalog.
--
-- Source: Google DeepMind's November 15, 2026 launch announcement, the
-- gemini-4-hydrogen API documentation, and independent benchmark evaluations
-- from Artificial Analysis and Vals AI.
--
-- Footnotes and caveats carried in the body so the rendered model card retains
-- full technical nuance:
-- 1. Unlike Argon and Carbon which began under Fairwind cyber-defender access,
--    Hydrogen is generally available to developers on day one.
-- 2. Artificial Analysis scores Hydrogen at 52 on its Intelligence Index v5.0,
--    with an index task cost of $0.28 and output throughput of 184 tokens/second.
-- 3. Pricing is $0.30/1M input and $1.20/1M output, with prompt caching at $0.015/1M.
-- 4. Context window is 1M tokens with up to 128K max output (1M with Long Decode Continuation).

WITH gemini_4_hydrogen_entry AS (
  SELECT $body$Gemini 4 Hydrogen is Google DeepMind's November 2026 lightweight frontier [large language model](/glossary/large-language-model) and the third model in the Gemini 4 generation, following [Gemini 4 Argon](/glossary/gemini-4-argon) and [Gemini 4 Carbon](/glossary/gemini-4-carbon). Named after the lightest and most abundant element, Hydrogen is engineered for high-throughput developer workflows, real-time [AI agent](/glossary/ai-agent) loops, and cost-efficient multimodal reasoning. It packages the architectural advancements of Gemini 4 into a high-velocity execution engine featuring sub-second time-to-first-token latency and developer-friendly token economics.

Unlike Argon and Carbon, which initially launched under restricted access for vetted cybersecurity defenders through Google's Fairwind Program, Gemini 4 Hydrogen is generally available to developers on day one through Google AI Studio, Vertex AI, and the Gemini API as `gemini-4-hydrogen`.

## Core profile

| Specification | Gemini 4 Hydrogen |
| --- | --- |
| Developer | Google DeepMind |
| Release date | November 15, 2026 |
| API model ID | `gemini-4-hydrogen` |
| Context window | 1,000,000 tokens |
| Maximum output | 128,000 tokens (up to 1,000,000 via Long Decode Continuation) |
| Knowledge cutoff | October 2026 |
| Input modalities | Text, images, video, audio, and documents (PDF) |
| Output modalities | Text (including code, structured JSON, tool calls) |
| Reasoning / thinking | Configurable thinking levels (`low`, `medium` default, `high`), plus non-thinking mode |
| Output speed | 184 tokens per second (Artificial Analysis measurement) |
| Standard pricing (1M) | $0.30 base input / $1.20 base output |
| Prompt cache reads | $0.015 per million tokens (95% discount) |
| Tool integration | Function calling, structured outputs, code execution, web search, MCP |
| Availability | Gemini API, Google AI Studio, Google Cloud Vertex AI |
| Weights / license | Proprietary; hosted cloud API |

The one-million-token [context window](/glossary/context-window) allows Hydrogen to ingest entire technical manuals, multi-hour video streams, or large codebases in a single forward pass. Operating at 184 tokens per second, Hydrogen generates responses more than twice as fast as Gemini 4 Argon (82 tps) and Gemini 4 Carbon (99.8 tps), making it suited for latency-sensitive applications where heavy frontier flagships introduce noticeable delays.

## Lineup positioning and generation changes

Within Google's late-2026 ecosystem, Gemini 4 Hydrogen fulfills the high-velocity frontier role:

- **Versus Gemini 4 Carbon:** Carbon is Google's heaviest flagship, scoring 59 on the Artificial Analysis Intelligence Index v5.0 and excelling at multi-day autonomous software repair and deep cybersecurity defense. Hydrogen trades peak depth for extreme throughput, scoring 52 on the index while running nearly twice as fast and costing less than one-tenth of Carbon's $4/$20 regular rate.
- **Versus Gemini 4 Argon:** Argon served as the initial restricted preview for Gemini 4's long-horizon capabilities. Hydrogen matches Argon's 1M-token context window while surpassing Argon on the Artificial Analysis Intelligence Index (52 vs 50) and offering day-one general API availability without waiting for Fairwind vetting.
- **Versus Gemini 3.8 Flash:** [Gemini 3.8 Flash](/glossary/gemini-3-8-flash) established strong coding capabilities at $0.75/$3.75 introductory pricing. Hydrogen represents a generational architectural shift, delivering higher reasoning fidelity, stronger indirect prompt-injection resilience, and cheaper token economics ($0.30/$1.20) with prompt caching discounts.
- **Versus GPT-6.1 Terra & Claude Haiku 5.5:** Hydrogen competes with OpenAI's mid-tier GPT-6.1 Terra ($1/$5) and Anthropic's Claude Haiku 5.5 ($0.50/$2.50). Hydrogen undercuts both on base token pricing while providing native omnimodal audio and video ingestion that GPT-6.1 Terra lacks.

## Benchmark profile

Independent and developer evaluations position Gemini 4 Hydrogen as a high-efficiency performer across coding, agentic reasoning, multimodal comprehension, and long-context retrieval.

| Evaluation | Gemini 4 Hydrogen | Gemini 4 Carbon | Gemini 4 Argon | GPT-6.1 Terra | Claude Haiku 5.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AA Intelligence Index v5.0 | 52 | **59** | 50 | 48 | 43 |
| Terminal-Bench 4.0 | 58.4% | **65.2%** | 57.1% | 50.3% | 39.2% |
| SWE-sweep | 7.8% | **9.6%** | 4.3% | 5.2% | 3.1% |
| Agents' Last Exam v1.1 | 71.2% | **77.4%** | 69.0% | 68.4% | 58.6% |
| AutomationBench | 46.8% | **53.9%** | 51.3% | 40.2% | 34.5% |
| GraphWalks (256K-1M) | 88.4% | **96.2%** | 84.2% | 74.0% | 68.2% |
| CharXiv (chart reasoning) | **89.5%** | 88.2% | 85.0% | 82.5% | 76.4% |
| Cost per index task | **$0.28** | $7.76 | $7.68 | $0.40 | $0.35 |

Evaluation conditions provide essential nuance:

On Artificial Analysis's Intelligence Index v5.0, Hydrogen posts a score of 52 at a cost of $0.28 per completed task, establishing one of the most efficient cost-to-intelligence ratios in the frontier catalog. On Terminal-Bench 4.0, Hydrogen achieves 58.4% at its high thinking level, placing ahead of Argon (57.1%) and GPT-6.1 Terra (50.3%) while trailing Carbon's 65.2%.

In visual document and chart reasoning, Hydrogen leads the comparison set on CharXiv at 89.5%, reflecting DeepMind's continued strength in native multimodal tokenization. On GraphWalks across long contexts (256,000 to 1,000,000 tokens), Hydrogen retains an 88.4% BFS F1 retrieval score, showing significantly less context rot than comparable mid-tier models.

As with all agentic evaluations, scores reflect the interaction between the model, the [agent harness](/glossary/agent-harness), execution timeouts, and prompt scaffolding.

## Pricing and efficiency

Gemini 4 Hydrogen introduces competitive pricing for the Gemini 4 generation:

| Token Type | Rate (per 1M tokens) |
| --- | ---: |
| Standard input | $0.30 |
| Prompt cache reads | $0.015 |
| Standard output | $1.20 |
| Cost per index task | $0.28 |
| Output throughput | 184 tokens/second |
| Time-to-first-token (TTFT) | ~320 ms |

At $0.015 per million cached tokens (a 95% discount), long-running agent workflows that repeatedly supply repository structures, user manuals, or persistent system prompts incur negligible recurring input costs. Batch processing reduces standard token rates by an additional 50% ($0.15 input / $0.60 output per million tokens) for non-interactive workloads.

High throughput (184 tokens per second) and low time-to-first-token (approximately 320 milliseconds) allow Hydrogen to serve conversational agents, real-time IDE code completion, and tight multi-turn tool loops without noticeable latency stalls.

## Safeguards and safety profile

Gemini 4 Hydrogen inherits Google DeepMind's Gemini 4 defense-in-depth safety architecture:

- **Misuse and cyber defense:** While Carbon is deployed as a specialized offensive validation and patching engine behind Fairwind, Hydrogen includes foundational cybersecurity guardrails that refuse malware creation and exploit compilation while supporting secure code scanning, defensive auditing, and vulnerability explanation.
- **Indirect prompt-injection resilience:** On Gray Swan's prompt-injection benchmarks, Hydrogen achieves a 0.2% attack success rate, maintaining high robustness against adversarial instructions hidden within untrusted web pages, documents, and emails.
- **Alignment and hallucination control:** Grounded retrieval and source attribution are prioritized; when queried outside its verified knowledge base, Hydrogen demonstrates a 32% lower unsupported-claim rate compared to Gemini 3.8 Flash.
- **Enterprise compliance:** Available with zero data retention configurations and covered by Google Cloud's intellectual property indemnification commitments.

## API and behavior changes

Developers integrating `gemini-4-hydrogen` should note the following API considerations:

- **Thinking controls:** Reasoning is configurable via the `thinking_config` block. Developers can set thinking effort to `low`, `medium` (default), or `high`. An explicit `mode: "off"` setting enables standard non-thinking generation for ultra-low latency conversational interfaces.
- **Long Decode Continuation:** For complex reasoning or expansive generation exceeding single-request limits, the Gemini API supports pausing and resuming execution across multiple HTTP connections without losing state.
- **Strict schema enforcement:** Tool calling and structured outputs support JSON Schema with deterministic type checking. Native tool support includes web search grounding, Python code interpreter, and MCP server bridges.
- **Multimodal file ingestion:** Documents up to 1,000 pages, audio files up to 9.5 hours, and video files up to 1 hour can be ingested directly without separate pre-processing or external transcription pipelines.

## Applications and workflow fit

Gemini 4 Hydrogen is best suited for:

- **Real-time agentic tool loops:** Multi-step autonomous agents where low per-step latency and cheap prompt caching outweigh the need for exhaustive multi-day reasoning.
- **Multimodal document intelligence:** Extracting structured tables, diagrams, and cross-references from large collections of contracts, research papers, and technical blueprints.
- **Interactive developer tooling:** IDE plugins, code explanation, inline refactoring, and test generation where instant response times are required.
- **Customer support and triage:** High-volume conversational agents requiring context awareness over extensive user histories without exceeding budget limits.

It is less suited for:

- Multi-day autonomous codebase refactoring and advanced kernel-level bug remediation, where [Gemini 4 Carbon](/glossary/gemini-4-carbon) or [Claude Fable 5.2](/glossary/claude-fable-5-2) provide deeper recursive problem-solving.
- Specialized zero-day exploit analysis, which remains restricted to Fairwind partners using Carbon.

## Bottom line

Gemini 4 Hydrogen successfully brings the Gemini 4 generation into production-ready economics. With a 1M-token context window, 184 tokens/second throughput, a 52 score on the Artificial Analysis Intelligence Index v5.0, and $0.30/$1.20 token pricing with $0.015 cached reads, it offers one of the most compelling performance-to-cost balances available for modern agentic and multimodal applications.$body$::text AS body,
    'Google DeepMind''s Nov 2026 lightweight frontier model in the Gemini 4 generation, engineered for high-throughput agent loops, 1M context, sub-second latency, and $0.30/$1.20 pricing.'::text AS summary
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:gemini-4-hydrogen',
  'glossary',
  'gemini-4-hydrogen',
  '',
  'glossary/gemini-4-hydrogen',
  'Gemini 4 Hydrogen',
  gemini_4_hydrogen_entry.summary,
  gemini_4_hydrogen_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', gemini_4_hydrogen_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing Gemini 4 Hydrogen', 'url', 'https://blog.google/technology/ai/gemini-4-hydrogen/'),
    jsonb_build_object('title', 'Gemini 4 Hydrogen documentation', 'url', 'https://ai.google.dev/gemini-api/docs/models/gemini-4-hydrogen'),
    jsonb_build_object('title', 'Gemini 4 Hydrogen model evaluation', 'url', 'https://storage.googleapis.com/deepmind-media/gemini/gemini_4_hydrogen_evaluation.pdf'),
    jsonb_build_object('title', 'Google Cloud Vertex AI pricing', 'url', 'https://cloud.google.com/vertex-ai/pricing')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array(
      'gemini-4-carbon',
      'gemini-4-argon',
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
    'seoDescription', 'Gemini 4 Hydrogen model card: Google DeepMind''s Nov 2026 lightweight frontier model, 1M context window, benchmark results, pricing, and availability.',
    'seoKeywords', jsonb_build_array(
      'Gemini 4 Hydrogen',
      'Gemini 4 Hydrogen benchmarks',
      'Gemini 4 Hydrogen pricing',
      'Gemini 4 Hydrogen model card',
      'Gemini 4 Hydrogen context window',
      'gemini-4-hydrogen',
      'Google DeepMind Gemini 4 Hydrogen',
      'Gemini 4 Hydrogen API availability',
      'Gemini 4 Hydrogen vs Gemini 4 Carbon',
      'Gemini 4 Hydrogen vs Claude Haiku 5.5'
    )
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-11-15',
  0
FROM gemini_4_hydrogen_entry
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
    WHEN COALESCE(metadata->'relatedTerms', '[]'::jsonb) @> '["gemini-4-hydrogen"]'::jsonb
      THEN metadata
    ELSE jsonb_set(
      COALESCE(metadata, '{}'::jsonb),
      '{relatedTerms}',
      COALESCE(metadata->'relatedTerms', '[]'::jsonb) || jsonb_build_array('gemini-4-hydrogen'),
      true
    )
  END,
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug IN ('gemini-4-carbon', 'gemini-4-argon', 'gemini-3-8-flash', 'google-deepmind')
  AND parent_slug = '';

INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at, updated_at
)
VALUES (
  'gemini-4-hydrogen',
  'Gemini 4 Hydrogen',
  'Google DeepMind',
  DATE '2026-11-15',
  DATE '2026-11-15',
  'proprietary',
  'Gemini 4',
  $specs${
    "Developer": "Google DeepMind",
    "Release date": "2026-11-15",
    "API model ID": "gemini-4-hydrogen",
    "Context window": "1M tokens",
    "Max output": "128K tokens (1M with Long Decode Continuation)",
    "Knowledge cutoff": "Oct 2026",
    "Reasoning / effort": "Configurable thinking levels (low, medium default, high), plus non-thinking mode",
    "Input / 1M tokens": "$0.30",
    "Cached input / 1M": "$0.015",
    "Cache write / 1M": "Included in standard input",
    "Output / 1M tokens": "$1.20",
    "Batch / flex discount": "50% off standard rates ($0.15 input / $0.60 output)",
    "Text input": "Yes",
    "Image / vision input": "Yes",
    "Audio input": "Yes",
    "Video input": "Yes",
    "File / document input": "Yes (PDF, docs)",
    "Text output": "Yes",
    "Image output": "No native image output",
    "Audio output": "No",
    "Video output": "No",
    "Tool / function calling": "Yes, function calling, structured outputs, code interpreter, web search, MCP",
    "Computer use": "Supported via agentic tool calling",
    "Terminal-Bench 4.0": "58.4%",
    "Terminal-Bench": "58.4% (Terminal-Bench 4.0; high thinking)",
    "SWE-sweep": "7.8%",
    "Agents' Last Exam": "71.2%",
    "AutomationBench": "46.8%",
    "MCP / tool-use benchmark": "AutomationBench 46.8%",
    "API access": "Yes",
    "Product access": "Gemini API + Google AI Studio + Google Cloud Vertex AI",
    "Weights / license": "Proprietary",
    "Primary focus": "High-throughput agent loops, multimodal document reasoning, low-latency developer tooling, and cost-efficient frontier intelligence",
    "Long-horizon work": "Built for persistent agent loops that reuse large repository or document context with $0.015 cached reads",
    "Safety / approvals": "Production safeguards with cyber defense baselines and Gray Swan prompt injection resilience"
  }$specs$::jsonb,
  $sources$[
    {"title": "Introducing Gemini 4 Hydrogen", "url": "https://blog.google/technology/ai/gemini-4-hydrogen/"},
    {"title": "Gemini 4 Hydrogen documentation", "url": "https://ai.google.dev/gemini-api/docs/models/gemini-4-hydrogen"},
    {"title": "Gemini 4 Hydrogen model evaluation", "url": "https://storage.googleapis.com/deepmind-media/gemini/gemini_4_hydrogen_evaluation.pdf"},
    {"title": "Google Cloud Vertex AI pricing", "url": "https://cloud.google.com/vertex-ai/pricing"}
  ]$sources$::jsonb,
  'Google DeepMind''s November 2026 lightweight frontier model in the Gemini 4 generation. 1M context window, 128K output (1M with Long Decode Continuation), 184 tokens/sec throughput, $0.30/$1.20 per 1M tokens with $0.015 cached input. Scores 52 on Artificial Analysis Intelligence Index v5.0 at $0.28 per index task, 58.4% on Terminal-Bench 4.0, and 7.8% on SWE-sweep.',
  '{"launch_stage": "ga", "verification_pass": "gemini-4-hydrogen-2026-11-15", "default_effort": "medium"}'::jsonb,
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
  {"id":"tq-20261115-gemini4hydrogen-terminalbench40","model_slug":"gemini-4-hydrogen","category":"coding","benchmark_name":"Terminal-Bench 4.0","benchmark_version":null,"score_numeric":58.4,"score_display":"58.4%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-11-15","source":"https://blog.google/technology/ai/gemini-4-hydrogen/","notes":"Self-reported launch result at high thinking level."},
  {"id":"tq-20261115-gemini4hydrogen-swesweep","model_slug":"gemini-4-hydrogen","category":"coding","benchmark_name":"SWE-sweep","benchmark_version":null,"score_numeric":7.8,"score_display":"7.8%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-11-15","source":"https://blog.google/technology/ai/gemini-4-hydrogen/","notes":"Launch evaluation at high thinking level."},
  {"id":"tq-20261115-gemini4hydrogen-charxiv","model_slug":"gemini-4-hydrogen","category":"multimodal","benchmark_name":"CharXiv","benchmark_version":null,"score_numeric":89.5,"score_display":"89.5%","score_unit":"percent","tools":false,"reasoning_effort":"medium","harness":null,"evaluator":"Google DeepMind","evaluation_date":"2026-11-15","source":"https://blog.google/technology/ai/gemini-4-hydrogen/","notes":"Chart reasoning without tools."}
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
