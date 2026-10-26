-- Add GPT-6.1 Terra to the glossary and model catalog.
--
-- Source: OpenAI's October 26, 2026 launch announcement, the gpt-6.1-terra API
-- model documentation, the October 26 addendum to the GPT-6 system card, and
-- Artificial Analysis's independent evaluation profile.
--
-- Notes and caveats carried in the body so the rendered model card retains
-- full technical nuance:
-- 1. Unlike GPT-6.1 Sol and Astra where reasoning cannot be disabled, Terra
--    supports reasoning.effort="none" for low-latency, deterministic pipelines.
-- 2. OpenAI's launch evaluations report 50.3% on Terminal-Bench 4.0 at max
--    effort (more than double GPT-5.6 Terra's 21.5%), 74.2% on OSWorld 2.1,
--    and 34.8% on Agents' Last Exam.
-- 3. Artificial Analysis reports an Intelligence Index of 48.1 at max effort,
--    with an index task cost of $0.40 and an output speed of 87 tokens/sec.
-- 4. AA-Omniscience hallucination rate falls from 88.0% to 38.0%, a 50-point
--    drop in guessing when context is missing.
-- 5. System card addendum rates Terra Medium in cybersecurity, Low in biology,
--    and below High in AI self-improvement, with 98.8% prompt injection defense.

WITH gpt_6_1_terra_entry AS (
  SELECT $body$GPT-6.1 Terra is OpenAI's October 2026 [large language model](/glossary/large-language-model) designed for high-volume workflows, visual reasoning, desktop computer use, and natural writing. Released on October 26, 2026, as the direct successor to GPT-5.6 Terra, it is available in the API as `gpt-6.1-terra`. In OpenAI's GPT-6 lineup, it sits between the economy-tier [GPT-6 Luna](/glossary/gpt-6-luna) (\$0.10/\$0.50) and the high-intelligence [GPT-6.1 Sol](/glossary/gpt-6-1-sol) (\$2/\$10), cutting Sol's token prices in half while approaching flagship visual and agentic capabilities.

For an everyday user, GPT-6.1 Terra is available in ChatGPT Work and Codex across Plus, Pro, Business, Enterprise, and Edu plans, and is integrated into GitHub Copilot for terminal and coding tasks. For a developer, it is a proprietary hosted model accepting text and image inputs and producing text outputs. Unlike [GPT-6.1 Sol](/glossary/gpt-6-1-sol) and [GPT-6 Astra](/glossary/gpt-6-astra), where reasoning is mandatory, GPT-6.1 Terra supports turning reasoning off entirely via `reasoning.effort: "none"`, providing a low-latency, deterministic execution mode for routine pipelines.

## Key specifications

| Specification | GPT-6.1 Terra |
| --- | --- |
| Developer | OpenAI |
| Release date | October 26, 2026 |
| API model ID | `gpt-6.1-terra` |
| Model family | GPT-6 (succeeds GPT-5.6 Terra) |
| Context window | 1,050,000 tokens |
| Maximum input | 922,000 tokens |
| Maximum output | 128,000 tokens |
| Knowledge cutoff | June 1, 2026 |
| Input modalities | Text, images (documents, screenshots, blueprints, diagrams) |
| Output modalities | Text (including code, structured JSON, tool calls) |
| Reasoning effort | `none`, `low`, `medium` (default), `high`, `xhigh`, `max` |
| Non-thinking mode | Supported (`reasoning.effort: "none"`) |
| API endpoints | Chat Completions (no-tool mode) and Responses API (full tool surface) |
| Tool integration | Web search, file search, code interpreter, computer use, MCP |
| Structured outputs | Supported (`json_schema` with `strict: true`) |
| Availability | OpenAI API, ChatGPT Work, Codex, GitHub Copilot, Amazon Bedrock |
| Downloadable weights | Proprietary; no public model weights |

Think of the [context window](/glossary/context-window) as an analyst's working desk: it bounds the total volume of conversation history, instructions, and reference material that can be held in working memory during an [inference](/glossary/inference) pass. A 1.05M-token window does not guarantee that every buried detail across a 900-page document corpus will be retrieved with identical precision, and long-context reasoning should be evaluated against retrieval quality. Similarly, the 128,000-token maximum output is a request ceiling rather than an expected response length.

## Lineup positioning and generation changes

GPT-6.1 Terra replaces GPT-5.6 Terra and repositions OpenAI's mid-tier offering along several structural axes:

- **Versus GPT-5.6 Terra:** Standard API token pricing is cut in half, dropping from \$2.00 to \$1.00 per million input tokens and from \$10.00 to \$5.00 per million output tokens. Reasoning capabilities improve substantially: on Terminal-Bench 4.0, Terra's pass rate jumps from 21.5% to 50.3% at maximum effort, while hallucination guessing rates on AA-Omniscience drop by 50 percentage points.
- **Versus GPT-6.1 Sol:** Sol remains OpenAI's recommendation for repository-scale software engineering and multi-step codebase refactoring, leading Terra on the Artificial Analysis Coding Agent Index (54 vs 48) and Terminal-Bench (56.1% vs 50.3%). However, Terra costs 50% less per token, executes agentic tasks over three times faster, and allows developers to disable reasoning when raw speed is paramount.
- **Versus GPT-6 Luna:** Luna is an economy text-only model priced at \$0.10/\$0.50 per million tokens, built for classification, shallow routing, and basic extraction. Terra introduces multimodal understanding, complex computer-use execution, and a 1.05M-token context window that Luna cannot match.
- **Versus GPT-6 Astra:** Astra represents OpenAI's research frontier (\$10/\$50 per million tokens) for novel mathematical and scientific problems. On day-to-day office tasks, form filling, and UI interaction evaluated in Agents' Last Exam, Terra nears Astra's performance at approximately one-tenth of the token cost.

## Pricing and cost efficiency

OpenAI lists flat rates for GPT-6.1 Terra across its standard context tier. Prices below are in US dollars per million tokens, reviewed on October 26, 2026.

| Token type | GPT-6.1 Terra | GPT-5.6 Terra | GPT-6.1 Sol | GPT-6 Luna | GPT-6 Astra |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard input | $1.00 | $2.00 | $2.00 | $0.10 | $10.00 |
| Cached input reads | $0.05 | $0.20 | $0.10 | $0.01 | $1.00 |
| Cache writes | $1.25 | $2.50 | $2.50 | $0.125 | $12.50 |
| Standard output | $5.00 | $10.00 | $10.00 | $0.50 | $50.00 |

Cached input reads cost \$0.05 per million tokens, representing a 95% discount compared to uncached input. Cache writes are billed at 1.25 times the uncached input rate (\$1.25 per million tokens) for 30-minute retention. For asynchronous and high-volume workloads, OpenAI offers Batch and Flex processing at a 50% discount (\$0.50 input / \$2.50 output per million tokens).

List price does not directly dictate end-to-end task cost. On the Artificial Analysis benchmark indexing suite, GPT-6.1 Terra records an average cost per completed task of \$0.40 at maximum effort, compared to \$0.72 for GPT-6.1 Sol, \$1.40 for GPT-5.6 Terra, and \$3.26 for GPT-6 Astra. Output throughput is measured between 83 and 91 tokens per second depending on reasoning effort, enabling fast iterative loops in tool-driven agents.

## Benchmark profile and independent evaluations

The following figures come from OpenAI's launch materials and represent vendor-reported evaluations across coding, agentic computer use, and domain-specific knowledge.

| Evaluation | GPT-6.1 Terra | GPT-5.6 Terra | GPT-6.1 Sol | GPT-6 Astra | What it measures |
| --- | ---: | ---: | ---: | ---: | --- |
| Terminal-Bench 4.0 (max) | **50.3%** | 21.5% | 56.1% | 59.1% | Shell execution and CLI problem-solving in agent environments |
| OSWorld 2.1 offline (max) | **74.2%** | 58.0% | 71.4% (v2.0) | 73.5% (v2.0) | Cross-application desktop UI navigation and computer use |
| Agents' Last Exam (medium) | **34.8%** | 19.8% | n/a | 35.2% | Professional computer tasks and technical artifact generation |
| MMMU-Pro (vision) | **62.4%** | 46.1% | 68.5% | 71.2% | Multimodal college-level multidisciplinary reasoning |
| DualEntry Accounting | **78.2%** | 61.0% | 82.4% | 85.0% | Financial document extraction and ledger entry reconciliation |

Independent evaluator Artificial Analysis published its comprehensive profile on October 26, 2026, measuring intelligence, throughput, and error rates.

| Measurement | GPT-6.1 Terra (max) | GPT-5.6 Terra | GPT-6.1 Sol (max) | GPT-6 Astra (max) |
| --- | ---: | ---: | ---: | ---: |
| Intelligence Index v4.3.2 | 48.1 | 42.0 | 51.8 | 52.7 |
| Coding Agent Index v2.0 | 48.0 | 34.0 | 54.0 | 57.0 |
| Cost per index task | **$0.40** | $1.40 | $0.72 | $3.26 |
| Output generation speed | 87 tps | 62 tps | 72 tps | 45 tps |
| AA-Omniscience hallucination rate (lower is better) | **38.0%** | 88.0% | 54.3% | 51.3% |

These numbers reflect specific configurations. On Terminal-Bench 4.0, score differences between 50.3% and 56.1% depend heavily on the [agent harness](/glossary/agent-harness), sandbox timeout budgets, and permission architectures. On DualEntry Accounting, Terra completes tasks in approximately 16 seconds per case, achieving high throughput but exhibiting minor degradation on complex accounts payable adjustments compared to Sol and Astra. The dramatic drop in AA-Omniscience hallucination rate (from 88% down to 38%) reflects calibrated abstention when contextual data is missing; it does not eliminate domain-specific hallucinations during ungrounded generation.

## API behavior and reasoning configuration

API callers can interact with `gpt-6.1-terra` through two endpoints:

- **Responses API:** Required for full [tool calling](/glossary/tool-calling), including web search, file search, code interpreter, MCP server connections, and computer use.
- **Chat Completions:** Supported for conversational and prompt-response workloads without tool calls.

The `reasoning.effort` parameter supports `none`, `low`, `medium` (default), `high`, `xhigh`, and `max`. When set to `none`, the model behaves as a standard non-reasoning generator: sampling parameters such as `temperature` and `top_p` operate conventionally, latency is minimized, and no reasoning tokens are billed. When any thinking tier is active, OpenAI recommends leaving `temperature` and `top_p` at their default values, and reasoning tokens are returned in the response stream before the final output.

## Safeguards and system card profile

OpenAI's October 26, 2026 system card addendum evaluates GPT-6.1 Terra under its Preparedness Framework:

- **Cybersecurity:** Rated Medium, contrasting with the Critical designation assigned to GPT-6.1 Sol and GPT-6 Astra. While Terra can interpret scripts and inspect CLI environments, it does not demonstrate autonomous zero-day vulnerability synthesis capabilities.
- **Biological and chemical hazards:** Rated Low.
- **AI self-improvement:** Rated below High.
- **Prompt injection and operational safety:** The system card reports 98.8% resistance to indirect prompt injections in adversarial document and web browsing tests, alongside improved boundary compliance during file system operations.

Unlike Sol and Astra, which require gated access programs for certain advanced cyber features, Terra is deployed broadly across commercial tiers without restricted cybersecurity tiers.

## Workflow fit and bottom line

GPT-6.1 Terra is tailored for high-volume enterprise automation where raw per-token cost, multimodal ingestion, and execution speed matter more than bleeding-edge mathematical reasoning. It is particularly effective for:

- Parsing multimodal business documents (invoices, receipts, engineering blueprints, architectural schematics).
- Desktop automation and computer-use subagents handling routine form completion and UI interactions.
- High-throughput coding assistance, linting, and terminal explanation within GitHub Copilot.
- Multi-step customer support and back-office agents that benefit from prompt caching across turns.

For deep algorithmic work, full-repository refactoring, or frontier scientific discovery, [GPT-6.1 Sol](/glossary/gpt-6-1-sol) and [GPT-6 Astra](/glossary/gpt-6-astra) remain the appropriate models. For basic text classification where vision is unnecessary, [GPT-6 Luna](/glossary/gpt-6-luna) provides an even cheaper alternative. By halving GPT-5.6 Terra's pricing while doubling terminal task performance and introducing non-thinking flexibility, GPT-6.1 Terra establishes a practical benchmark for cost-efficient multimodal intelligence.
$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:gpt-6-1-terra',
  'glossary',
  'gpt-6-1-terra',
  '',
  'glossary/gpt-6-1-terra',
  'GPT-6.1 Terra',
  'OpenAI''s October 2026 mid-tier model with 1.05M context, text/image input, optional reasoning, and $1/$5 per million token pricing for high-volume workflows.',
  gpt_6_1_terra_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', gpt_6_1_terra_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing GPT-6.1 Terra', 'url', 'https://openai.com/index/introducing-gpt-6-1-terra/'),
    jsonb_build_object('title', 'GPT-6.1 Terra model documentation', 'url', 'https://developers.openai.com/api/docs/models/gpt-6.1-terra'),
    jsonb_build_object('title', 'Addendum to GPT-6 System Card: GPT-6.1 Terra', 'url', 'https://deploymentsafety.openai.com/gpt-6-1-terra'),
    jsonb_build_object('title', 'Artificial Analysis: GPT-6.1 Terra benchmark and efficiency evaluation', 'url', 'https://artificialanalysis.ai/models/gpt-6-1-terra')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array('gpt-6-1-sol', 'gpt-6-luna', 'gpt-6-astra', 'gpt-5-6', 'openai', 'large-language-model', 'context-window', 'ai-agent', 'agentic-ai', 'agent-harness', 'benchmark', 'api'),
    'seoDescription', 'GPT-6.1 Terra model card: 1.05M context, $1/$5 API pricing, $0.05 cache reads, explicit reasoning with none mode, visual understanding and benchmarks.',
    'seoKeywords', jsonb_build_array('GPT-6.1 Terra', 'GPT-6.1 Terra model card', 'GPT-6.1 Terra benchmarks', 'GPT-6.1 Terra pricing', 'gpt-6-1-terra', 'gpt-6.1-terra', 'OpenAI GPT-6.1 Terra', 'GPT-6.1 Terra vs Sol', 'GPT-6.1 Terra context window', 'GPT-6.1 Terra reasoning effort')
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-10-26',
  0
FROM gpt_6_1_terra_entry
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
    WHEN COALESCE(metadata->'relatedTerms', '[]'::jsonb) @> '["gpt-6-1-terra"]'::jsonb
      THEN metadata
    ELSE jsonb_set(
      COALESCE(metadata, '{}'::jsonb),
      '{relatedTerms}',
      COALESCE(metadata->'relatedTerms', '[]'::jsonb) || jsonb_build_array('gpt-6-1-terra'),
      true
    )
  END,
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug IN ('gpt-6-1-sol', 'gpt-6-luna', 'gpt-6-sol')
  AND parent_slug = '';

INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at
)
VALUES (
  'gpt-6-1-terra',
  'GPT-6.1 Terra',
  'OpenAI',
  DATE '2026-10-26',
  DATE '2026-10-26',
  'proprietary',
  'GPT-6',
  $specs${
    "Developer": "OpenAI",
    "Release date": "2026-10-26",
    "API model ID": "gpt-6.1-terra",
    "Context window": "1,050,000 tokens",
    "Max output": "128,000 tokens",
    "Knowledge cutoff": "Jun 1, 2026",
    "Reasoning / effort": "none / low / medium (default) / high / xhigh / max. Supports non-thinking mode (none) for low latency",
    "Input / 1M tokens": "$1.00",
    "Cached input / 1M": "$0.05",
    "Cache write / 1M": "$1.25 (1.25x uncached input for 30-min retention)",
    "Output / 1M tokens": "$5.00",
    "Batch / flex discount": "Batch and Flex 50 percent off standard; fast mode unavailable; regional processing where available",
    "Long-context surcharge": "Standard tiered structure applies for large context volumes above 272K",
    "Text input": "Yes",
    "Image / vision input": "Yes (documents, UI screenshots, technical drawings, blueprints)",
    "Audio input": "No",
    "Video input": "No",
    "File / document input": "Yes, through Responses API file search and vision ingestion",
    "Text output": "Yes",
    "Image output": "Yes, through the Responses API image generation tool",
    "Audio output": "No",
    "Video output": "No",
    "Tool / function calling": "Yes, streaming, function calling and structured outputs. Tool calling requires the Responses API; Chat Completions works without tools. Fine-tuning not supported",
    "Computer use": "Yes, through the Responses API",
    "API access": "Yes, OpenAI API, Amazon Bedrock, OpenRouter; US and EU data residency supported",
    "Product access": "ChatGPT Work, Codex (Plus, Pro, Business, Enterprise, Edu), GitHub Copilot",
    "Weights / license": "Proprietary",
    "Primary focus": "High-volume workflows, visual reasoning, computer-use subagents, and cost-efficient agent loops",
    "Long-horizon work": "Built for agent loops that reuse repository or document context across requests with $0.05 cached reads",
    "Agent orchestration": "Responses API tool surface: web search, file search, image generation, code interpreter, computer use, MCP and tool search",
    "Efficiency / generation change": "50 percent cheaper than GPT-5.6 Terra and GPT-6.1 Sol ($1/$5 per 1M tokens); over 3x faster task execution than flagships; supports non-thinking mode (none)",
    "Safety / approvals": "Medium in cybersecurity, Low in biological and chemical capability, below High in AI self-improvement under the Preparedness Framework; 98.8 percent prompt injection resistance"
  }$specs$::jsonb,
  $sources$[
    {"title": "Introducing GPT-6.1 Terra", "url": "https://openai.com/index/introducing-gpt-6-1-terra/"},
    {"title": "GPT-6.1 Terra model documentation", "url": "https://developers.openai.com/api/docs/models/gpt-6.1-terra"},
    {"title": "Addendum to GPT-6 System Card: GPT-6.1 Terra", "url": "https://deploymentsafety.openai.com/gpt-6-1-terra"},
    {"title": "Artificial Analysis: GPT-6.1 Terra benchmark and efficiency evaluation", "url": "https://artificialanalysis.ai/models/gpt-6-1-terra"}
  ]$sources$::jsonb,
  'Released October 26, 2026 as the successor to GPT-5.6 Terra. In the GPT-6 lineup, it sits between GPT-6 Luna ($0.10/$0.50) and GPT-6.1 Sol ($2.00/$10.00). Offers 1.05M-token context window with $1 input and $5 output per million tokens, and $0.05 cached input. Unlike Sol and Astra, reasoning can be disabled with reasoning.effort: "none". Strong multimodal and computer-use capabilities with 74.2% on OSWorld 2.1 offline and 50.3% on Terminal-Bench 4.0 at max effort. Artificial Analysis Intelligence Index score of 48.1 at max effort, delivering $0.40 cost per completed task.',
  '{"verification_pass": "gpt-6-1-terra-2026-10-26", "default_effort": "medium", "catalog_status": "current", "supersedes": "gpt-5-6-terra"}'::jsonb,
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
FROM jsonb_to_recordset($benchmarks$
[
  {"id":"tq-20261026-gpt61terra-terminalbench40","model_slug":"gpt-6-1-terra","category":"coding","benchmark_name":"Terminal-Bench 4.0","benchmark_version":null,"score_numeric":50.3,"score_display":"50.3%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"Claude Code","evaluator":"OpenAI","evaluation_date":"2026-10-26","source":"https://openai.com/index/introducing-gpt-6-1-terra/","notes":"Self-reported launch result at max effort. More than double GPT-5.6 Terra's 21.5% and closely trailing GPT-6.1 Sol's 56.1%."},
  {"id":"tq-20261026-gpt61terra-osworld21","model_slug":"gpt-6-1-terra","category":"agentic_computer_use","benchmark_name":"OSWorld 2.1","benchmark_version":"offline","score_numeric":74.2,"score_display":"74.2%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"offline desktop computer-use","evaluator":"OpenAI","evaluation_date":"2026-10-26","source":"https://openai.com/index/introducing-gpt-6-1-terra/","notes":"Self-reported offline computer-use evaluation score at max reasoning effort."},
  {"id":"tq-20261026-gpt61terra-ale-medium","model_slug":"gpt-6-1-terra","category":"professional","benchmark_name":"Agents' Last Exam","benchmark_version":null,"score_numeric":34.8,"score_display":"34.8%","score_unit":"percent","tools":true,"reasoning_effort":"medium","harness":"desktop workflow and 3D tools","evaluator":"OpenAI","evaluation_date":"2026-10-26","source":"https://openai.com/index/introducing-gpt-6-1-terra/","notes":"Self-reported result at medium effort, gaining 15 percentage points over GPT-5.6 Terra and approaching GPT-6 Astra's 35.2%."},
  {"id":"tq-20261026-gpt61terra-mmmupro","model_slug":"gpt-6-1-terra","category":"multimodal","benchmark_name":"MMMU-Pro","benchmark_version":null,"score_numeric":62.4,"score_display":"62.4%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"OpenAI","evaluation_date":"2026-10-26","source":"https://openai.com/index/introducing-gpt-6-1-terra/","notes":"Multimodal reasoning benchmark testing visual and multidisciplinary problem solving at max effort."},
  {"id":"tq-20261026-gpt61terra-dualentry","model_slug":"gpt-6-1-terra","category":"professional","benchmark_name":"DualEntry Accounting","benchmark_version":null,"score_numeric":78.2,"score_display":"78.2%","score_unit":"percent","tools":true,"reasoning_effort":"medium","harness":"financial agent harness","evaluator":"OpenAI","evaluation_date":"2026-10-26","source":"https://openai.com/index/introducing-gpt-6-1-terra/","notes":"Financial document extraction and reconciliation benchmark, completing tasks in ~16s."},
  {"id":"tq-20261026-gpt61terra-omniscience","model_slug":"gpt-6-1-terra","category":"knowledge","benchmark_name":"AA-Omniscience","benchmark_version":null,"score_numeric":38.0,"score_display":"38.0%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-10-26","source":"https://artificialanalysis.ai/models/gpt-6-1-terra","notes":"Hallucination rate, so lower is better. Represents a 50-point drop in guessing when context is missing compared to GPT-5.6 Terra's 88.0%."}
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
