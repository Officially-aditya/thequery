-- Add GPT-6.2 Sol to the glossary and model catalog.
--
-- Source: OpenAI's November 17, 2026 launch announcement, the gpt-6.2-sol
-- API model documentation, the GPT-6.2 System Card and safety addendum,
-- and verified comparative benchmarks against GPT-6.1 Sol, GPT-6.2 Astra, and Claude Sonnet 5.5.
--
-- Footnotes and caveats carried in the body so the rendered model card retains
-- full technical nuance:
-- 1. Keeps $2/$10 token pricing and $0.10 cached reads, while nearing Astra on FrontierCode 2.0 (49.3% vs 49.8%).
-- 2. Scores 96.0% on GPQA Diamond (zero-shot, no tools), matching the original GPT-6 Astra.
-- 3. Expanded access to Ultrafast generation mode for Pro 200 subscribers and enterprise workspaces.
-- 4. Mandatory adaptive reasoning; non-thinking mode (reasoning.effort: "none") is unsupported.

WITH gpt_6_2_sol_entry AS (
  SELECT $body$GPT-6.2 Sol is OpenAI's November 2026 high-efficiency [large language model](/glossary/large-language-model) built for everyday software engineering, document-intensive professional labor, and high-volume [AI agent](/glossary/ai-agent) workflows. Released on November 17, 2026 as the API model `gpt-6.2-sol`, it succeeds [GPT-6.1 Sol](/glossary/gpt-6-1-sol) as the core workhorse of OpenAI's GPT-6 generation. Sol delivers near-frontier reasoning fidelity—landing within 0.5 points of the flagship [GPT-6.2 Astra](/glossary/gpt-6-2-astra) on Cognition's FrontierCode 2.0 benchmark—at one-fifth of Astra's token price and one-fifth the per-task execution cost.

For an everyday user, GPT-6.2 Sol is available in ChatGPT Work and Codex across Plus, Pro, Enterprise, and Edu plans, with expanded access to high-velocity Ultrafast generation. For developers, it is a proprietary hosted model with text and vision inputs, mandatory adaptive reasoning, and comprehensive tool calling capabilities via OpenAI's Responses API.

## Core profile

| Specification | GPT-6.2 Sol |
| --- | --- |
| Developer | OpenAI |
| Release date | November 17, 2026 |
| API model ID | `gpt-6.2-sol` |
| Model family | GPT-6 (succeeds GPT-6.1 Sol) |
| Context window | 1,050,000 tokens |
| Maximum input | 922,000 tokens |
| Maximum output | 128,000 tokens |
| Knowledge cutoff | June 1, 2026 |
| Input modalities | Text, images (documents, screenshots, blueprints, diagrams) |
| Output modalities | Text (including code, structured JSON, tool calls) |
| Reasoning / effort | Adaptive thinking (always on); `low`, `medium` (default), `high`, `xhigh`, `max` |
| Non-thinking mode | Unsupported (reasoning is always active) |
| API endpoints | Responses API (full tool surface) and Chat Completions (no-tool mode) |
| Tool integration | Web search, file search, image generation, code interpreter, hosted shell, apply patch, skills, computer use, MCP and tool search |
| Structured outputs | Supported (`json_schema` with `strict: true`) |
| Standard pricing (1M) | \$2.00 base input / \$10.00 base output |
| Prompt cache reads | \$0.10 per million tokens (95% discount) |
| Cache writes | \$2.50 (30-min retention) / \$5.00 (12-hour retention) |
| Batch / flex discount | 50% discount on standard rates (\$1.00 input / \$5.00 output) |
| Long-context surcharge | Prompts over 272K input tokens billed at 2x input/cache rates and 1.5x output for the full request |
| Availability | OpenAI API, ChatGPT Work, Codex (Plus, Pro, Enterprise, Edu) |
| Downloadable weights | Proprietary; hosted cloud API |

The 1,050,000-token [context window](/glossary/context-window) allows Sol to hold substantial code repositories, technical specifications, and multi-turn conversation logs in active memory. Sol's prompt caching architecture is optimized for iterative agent loops: by charging \$0.10 per million cached tokens, recurring turns through large contexts cost pennies. The 128,000-token output limit provides ample room for full code rewrites and structured multi-step responses.

## Lineup positioning and generation changes

Within OpenAI's model family, GPT-6.2 Sol occupies the high-throughput production sweet spot:

- **Versus GPT-6.1 Sol:** Sol maintains the same \$2.00/\$10.00 per million token pricing and \$0.10 cache read rate as GPT-6.1 Sol, but advances software engineering capability to 49.3% on Cognition's FrontierCode 2.0. GPQA Diamond science accuracy advances from 94.2% to 96.0%, the knowledge cutoff extends from April 30, 2026 to June 1, 2026, and Ultrafast tier generation speed is broadened to Pro 200 subscribers.
- **Versus GPT-6.2 Astra:** [GPT-6.2 Astra](/glossary/gpt-6-2-astra) is OpenAI's flagship (\$8.00/\$40.00). Sol reduces token costs by 75% compared to Astra and achieves near-parity on real-world repository coding (49.3% vs 49.8% on FrontierCode 2.0 at maximum effort), executing tasks at roughly one-fifth of Astra's cost per completed task. However, Astra maintains an advantage on deep scientific computation (77.1% vs 61.5% on Terminal-Bench-Science 0.1) and complex legal brief reasoning (84.5% vs 71.2% on Harvey LAB-AA).
- **Versus GPT-6.1 Terra:** [GPT-6.1 Terra](/glossary/gpt-6-1-terra) is OpenAI's mid-tier model (\$1.00/\$5.00). Terra supports turning reasoning off entirely (`reasoning.effort: "none"`) for ultra-fast, deterministic extraction. Sol requires adaptive reasoning on every request, making it more capable on intricate coding and multi-turn agent logic, but less suited for ultra-low-latency non-thinking tasks.
- **Versus Claude Sonnet 5.5:** Anthropic's Claude Sonnet 5.5 lists at \$3.00/\$15.00 per million tokens. GPT-6.2 Sol undercuts Sonnet 5.5 on base token rates while matching or exceeding it on agentic coding and browser use benchmarks.

## Benchmark profile and capability evaluations

OpenAI's launch reports and comparative benchmarks highlight Sol's balance of intelligence and cost efficiency:

| Evaluation | GPT-6.2 Sol | GPT-6.2 Astra | GPT-6.1 Sol | GPT-6.1 Terra | Claude Sonnet 5.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FrontierCode 2.0 (Cognition) | **49.3%** (max) | 49.8% (max) | 44.2% | 40.5% | 46.2% |
| GPQA Diamond (no tools) | **96.0%** | 97.0% | 94.2% | 89.4% | 91.5% |
| Terminal-Bench-Science 0.1 | **61.5%** (max) | 77.1% (max) | 57.0% | 48.2% | 55.4% |
| Agents' Last Exam (ALE) | **70.4%** (max) | 77.6% (max) | 65.2% | 59.3% | 66.8% |
| OSWorld 2.1 (offline subset) | **82.6%** (max) | 89.2% (max) | 71.4% | 64.0% | 76.5% |
| LifeVal / Practical Tasks | **86.4%** | 88.0% | 80.1% | 78.5% | 82.0% |

The defining milestone for GPT-6.2 Sol is its score on Cognition's FrontierCode 2.0 benchmark. FrontierCode evaluates whether pull requests generated by an AI coding agent satisfy open-source project maintainers' merge standards. At maximum reasoning effort, Sol solves 49.3% of tasks, landing within 0.5 points of GPT-6.2 Astra (49.8%) while costing about one-fifth per completed task.

On zero-shot GPQA Diamond without tools, Sol reaches 96.0%, matching the original GPT-6 Astra and demonstrating exceptional factual grounding. On desktop computer use evaluated by OSWorld 2.1, Sol scores 82.6%, providing robust execution for multi-step browser and office application workflows.

As with all reasoning benchmarks, performance varies with the [agent harness](/glossary/agent-harness), system scaffolding, and reasoning effort budget.

## Pricing and cost efficiency

OpenAI maintains aggressive pricing for GPT-6.2 Sol, reviewed on November 17, 2026:

| Token type | GPT-6.2 Sol | GPT-6.1 Sol | GPT-6.2 Astra | GPT-6.1 Terra |
| --- | ---: | ---: | ---: | ---: |
| Base input (per 1M) | \$2.00 | \$2.00 | \$8.00 | \$1.00 |
| Cached input reads | \$0.10 | \$0.10 | \$0.40 | \$0.05 |
| Cache write (30-min TTL) | \$2.50 | \$2.50 | \$10.00 | \$1.25 |
| Base output (per 1M) | \$10.00 | \$10.00 | \$40.00 | \$5.00 |
| Batch / flex input | \$1.00 | \$1.00 | \$4.00 | \$0.50 |
| Batch / flex output | \$5.00 | \$5.00 | \$20.00 | \$2.50 |

Prompt cache reads cost \$0.10 per million tokens—a 95% discount against uncached input. In production agent architectures where an agent re-evaluates a 300,000-token repository context across 15 turns, prompt caching slashes input costs from \$9.00 to \$0.45. Batch and Flex processing provide an additional 50% discount (\$1.00 input / \$5.00 output) for non-interactive jobs. Prompts exceeding 272,000 input tokens incur a long-context surcharge (2x input and 1.5x output).

Because Sol requires adaptive reasoning, total spend reflects the reasoning effort tier selected (`low`, `medium`, `high`, `xhigh`, `max`). For general coding and test writing, `medium` effort provides optimal token economy; `max` effort is best reserved for tricky concurrency bugs or complex multi-file migrations.

## Safeguards and system card profile

OpenAI's November 17, 2026 system card outlines safety and risk classifications for GPT-6.2 Sol:

- **Cybersecurity:** Classified as Critical capability under the Preparedness Framework, sharing the same baseline cyber risk evaluation as GPT-6.2 Astra. OpenAI applies automated guardrails that restrict the creation of functional cyber exploits without authorization.
- **Biological and chemical hazards:** Rated as High capability. Built-in boundary filters refuse actionable dual-use biochemical requests.
- **AI self-improvement:** Rated below the High threshold (unlike GPT-6.2 Astra, which crossed into High), confirming that Sol's autonomous ML research acceleration capabilities remain contained within standard bounds.
- **Alignment and behavioral profile:** Sol demonstrates low improper action rates on simulated office and digital labor benchmarks. In evaluation tasks testing transparent error reporting, developers are advised to include explicit error-logging system prompts in Codex to prevent the model from silently working around environmental roadblocks.

## API and behavior changes

Integrating `gpt-6.2-sol` involves several behavioral considerations:

- **Preserving conversation prefix:** Like Astra, Sol's API drops reasoning items if mid-turn context tampering or model downgrades are detected. Client applications must maintain immutable message prefixes to preserve reasoning traces and prompt cache hits.
- **Responses API tooling:** All tool use (bash shell, patch application, file search, web retrieval, computer use, and MCP servers) requires the Responses API. The Chat Completions endpoint is reserved for prompt-only conversational interactions.
- **Mandatory reasoning:** The `none` effort setting is unsupported. Effort must be configured between `low` and `max`.
- **Ultrafast generation mode:** Pro 200 and Enterprise subscribers can leverage Ultrafast mode, generating tokens up to 4.5x faster in Codex and ChatGPT Work for accelerated developer workflows.

## Applications and workflow fit

GPT-6.2 Sol is the recommended engine for the vast majority of production LLM tasks:

- End-to-end repository code generation, bug fixing, test suite expansion, and PR creation in Codex.
- Multi-step customer service and IT automation agents operating with tool calling and prompt caching.
- Document extraction, financial spreadsheet modeling, and business intelligence reporting.
- Desktop GUI navigation and web browser automation agents via computer use.

When tasks involve novel scientific proofs, mathematical discovery, or extreme legal rigor, teams can escalate to [GPT-6.2 Astra](/glossary/gpt-6-2-astra).

## Bottom line

GPT-6.2 Sol solidifies OpenAI's everyday workhorse tier by delivering near-flagship intelligence at everyday prices. Matching GPT-6.2 Astra within 0.5 points on FrontierCode 2.0 while keeping the \$2/\$10 token price and \$0.10 cached-read economics makes Sol one of the most cost-effective coding and agentic models available. Combined with expanded Ultrafast generation access and a 1.05M-token context window, GPT-6.2 Sol is the default choice for production engineering teams scaling agentic pipelines in late 2026.$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:gpt-6-2-sol',
  'glossary',
  'gpt-6-2-sol',
  '',
  'glossary/gpt-6-2-sol',
  'GPT-6.2 Sol',
  'OpenAI''s November 2026 high-efficiency everyday model with 1.05M context, $2/$10 pricing with $0.10 cache reads, expanded Ultrafast access, and near-Astra coding on FrontierCode 2.0.',
  gpt_6_2_sol_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', gpt_6_2_sol_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing GPT-6.2 Astra and GPT-6.2 Sol', 'url', 'https://openai.com/index/introducing-gpt-6-2/'),
    jsonb_build_object('title', 'GPT-6.2 Sol model documentation', 'url', 'https://developers.openai.com/api/docs/models/gpt-6.2-sol'),
    jsonb_build_object('title', 'GPT-6.2 System Card: Safety, Cybersecurity, and Alignment Addendum', 'url', 'https://deploymentsafety.openai.com/gpt-6-2-system-card'),
    jsonb_build_object('title', 'Artificial Analysis: GPT-6.2 Sol efficiency and agentic coding evaluation', 'url', 'https://artificialanalysis.ai/models/gpt-6.2-sol')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array(
      'gpt-6-2-astra',
      'gpt-6-1-sol',
      'gpt-6-sol',
      'gpt-6-1-terra',
      'gpt-6-luna',
      'claude-haiku-5-5',
      'openai',
      'large-language-model',
      'context-window',
      'ai-agent',
      'agentic-ai',
      'agent-harness',
      'benchmark',
      'api'
    ),
    'seoDescription', 'GPT-6.2 Sol model card: OpenAI''s Nov 2026 workhorse, 1.05M context, $2/$10 pricing, $0.10 cache reads, FrontierCode near-Astra coding, and Ultrafast tier.',
    'seoKeywords', jsonb_build_array(
      'GPT-6.2 Sol',
      'GPT-6.2 Sol model card',
      'GPT-6.2 Sol benchmarks',
      'GPT-6.2 Sol pricing',
      'GPT-6.2 Sol context window',
      'gpt-6-2-sol',
      'gpt-6.2-sol',
      'OpenAI GPT-6.2 Sol',
      'GPT-6.2 Sol vs Astra',
      'GPT-6.2 Sol FrontierCode'
    )
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-11-17',
  0
FROM gpt_6_2_sol_entry
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
    WHEN COALESCE(metadata->'relatedTerms', '[]'::jsonb) @> '["gpt-6-2-sol"]'::jsonb
      THEN metadata
    ELSE jsonb_set(
      COALESCE(metadata, '{}'::jsonb),
      '{relatedTerms}',
      COALESCE(metadata->'relatedTerms', '[]'::jsonb) || jsonb_build_array('gpt-6-2-sol'),
      true
    )
  END,
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug IN ('gpt-6-2-astra', 'gpt-6-1-sol', 'gpt-6-1-terra', 'gpt-6-sol', 'openai')
  AND parent_slug = '';

INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at, updated_at
)
VALUES (
  'gpt-6-2-sol',
  'GPT-6.2 Sol',
  'OpenAI',
  DATE '2026-11-17',
  DATE '2026-11-17',
  'proprietary',
  'GPT-6',
  $specs${
    "Developer": "OpenAI",
    "Release date": "2026-11-17",
    "API model ID": "gpt-6.2-sol",
    "Context window": "1,050,000 tokens",
    "Max output": "128,000 tokens",
    "Knowledge cutoff": "Jun 1, 2026",
    "Reasoning / effort": "Always on; low / medium (default) / high / xhigh / max. Non-thinking mode unsupported",
    "Input / 1M tokens": "$2.00",
    "Cached input / 1M": "$0.10",
    "Cache write / 1M": "$2.50 (1.25x uncached input for 30-min retention)",
    "Output / 1M tokens": "$10.00",
    "Batch / flex discount": "Batch and Flex 50 percent off standard rates ($1.00 input / $5.00 output)",
    "Long-context surcharge": "Prompts above 272K input tokens billed at 2x input and cache rates and 1.5x output for the full request",
    "Text input": "Yes",
    "Image / vision input": "Yes (documents, UI screenshots, technical drawings, blueprints)",
    "Audio input": "No",
    "Video input": "No",
    "File / document input": "Yes, through Responses API file search and vision ingestion",
    "Text output": "Yes",
    "Image output": "No native image output; image generation available as a Responses API tool",
    "Audio output": "No",
    "Video output": "No",
    "Tool / function calling": "Yes, streaming, function calling and structured outputs. Tool calling requires the Responses API; Chat Completions works without tools. Fine-tuning not supported",
    "Computer use": "Yes, through the Responses API",
    "FrontierCode 2.0": "49.3%",
    "GPQA Diamond": "96.0%",
    "Terminal-Bench-Science 0.1": "61.5%",
    "Agents' Last Exam": "70.4%",
    "OSWorld 2.1": "82.6%",
    "LifeVal": "86.4%",
    "API access": "OpenAI API (gpt-6.2-sol); drops reasoning items if history prefix is modified",
    "Product access": "ChatGPT Work and Codex (Plus, Pro, Enterprise, Edu); Ultrafast tier available",
    "Weights / license": "Proprietary",
    "Primary focus": "High-volume software engineering, multi-turn coding agents, document-heavy workflows, and cost-efficient agent loops",
    "Long-horizon work": "Built for persistent agent loops that reuse large repository or document context with $0.10 cached reads",
    "Agent orchestration": "Responses API tool surface: web search, file search, image generation, code interpreter, hosted shell, apply patch, skills, computer use, MCP and tool search",
    "Efficiency / generation change": "Retains $2/$10 pricing with $0.10 cached input while closing the gap with Astra on FrontierCode 2.0 (49.3% vs 49.8%) at one-fifth the per-task execution cost; expands Ultrafast access to Pro 200 subscribers",
    "Safety / approvals": "Critical in cybersecurity and High in biological and chemical capability under the Preparedness Framework; below High in AI self-improvement"
  }$specs$::jsonb,
  $sources$[
    {"title": "Introducing GPT-6.2 Astra and GPT-6.2 Sol", "url": "https://openai.com/index/introducing-gpt-6-2/"},
    {"title": "GPT-6.2 Sol model documentation", "url": "https://developers.openai.com/api/docs/models/gpt-6.2-sol"},
    {"title": "GPT-6.2 System Card: Safety, Cybersecurity, and Alignment Addendum", "url": "https://deploymentsafety.openai.com/gpt-6-2-system-card"},
    {"title": "Artificial Analysis: GPT-6.2 Sol efficiency and agentic coding evaluation", "url": "https://artificialanalysis.ai/models/gpt-6-2-sol"}
  ]$sources$::jsonb,
  'OpenAI''s November 17, 2026 everyday workhorse model succeeding GPT-6.1 Sol. Features 1.05M-token context window with $2 input and $10 output per million tokens, and $0.10 cached input. Achieves 49.3% on Cognition''s FrontierCode 2.0 benchmark at max effort—landing within 0.5 points of GPT-6.2 Astra at roughly one-fifth the per-task cost. Scores 96.0% on GPQA Diamond and 61.5% on Terminal-Bench-Science 0.1. Rated Critical in cybersecurity and High in biological/chemical capability under the Preparedness Framework, with expanded access to Ultrafast generation for Pro 200 subscribers.',
  '{"launch_stage": "ga", "verification_pass": "gpt-6-2-sol-2026-11-17", "default_effort": "medium", "supersedes": "gpt-6-1-sol"}'::jsonb,
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
  {"id":"tq-20261117-gpt62sol-frontiercode20","model_slug":"gpt-6-2-sol","category":"coding","benchmark_name":"FrontierCode 2.0","benchmark_version":null,"score_numeric":49.3,"score_display":"49.3%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"OpenAI","evaluation_date":"2026-11-17","source":"https://openai.com/index/introducing-gpt-6-2/","notes":"Cognition FrontierCode 2.0 benchmark at max effort; within 0.5 points of GPT-6.2 Astra at one-fifth the per-task cost."},
  {"id":"tq-20261117-gpt62sol-gpqadiamond","model_slug":"gpt-6-2-sol","category":"knowledge","benchmark_name":"GPQA Diamond","benchmark_version":null,"score_numeric":96.0,"score_display":"96.0%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":null,"evaluator":"OpenAI","evaluation_date":"2026-11-17","source":"https://openai.com/index/introducing-gpt-6-2/","notes":"Zero-shot graduate-level science reasoning without tools."},
  {"id":"tq-20261117-gpt62sol-terminalbenchscience01","model_slug":"gpt-6-2-sol","category":"coding","benchmark_name":"Terminal-Bench-Science 0.1","benchmark_version":null,"score_numeric":61.5,"score_display":"61.5%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"OpenAI","evaluation_date":"2026-11-17","source":"https://openai.com/index/introducing-gpt-6-2/","notes":"Self-reported launch result on terminal scientific workflows at max reasoning effort."}
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
