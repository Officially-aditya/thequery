-- Add Claude Fable 5.2 to the glossary and model catalog.
--
-- Source: Anthropic's November 10, 2026 launch announcement, the claude-fable-5-2
-- API model documentation, the Claude Fable 5.2 and Mythos 5.2 System Card,
-- and verified comparative benchmarks against Opus 5.5, Gemini 4 Carbon, and GPT-6 Astra.
--
-- Footnote and caveats carried directly in the markdown body:
-- 1. Fable 5.2 retains Fable 5.1's $10/$50 token price and $0.25 prompt cache read rate.
-- 2. Adaptive thinking is always enabled; forced tool calling returns an error.
-- 3. Thinking blocks are bound cryptographically to conversation turn hashes.
-- 4. Dual release with Mythos 5.2 for vetted cybersecurity and biological safety partners.

WITH fable_52_entry AS (
  SELECT $body$Claude Fable 5.2 is Anthropic's November 2026 premium frontier [large language model](/glossary/large-language-model) for difficult software engineering, long-running [AI agent](/glossary/ai-agent) workflows, scientific research, and complex computer use. Released on November 10, 2026 as the API model `claude-fable-5-2`, it succeeds [Claude Fable 5.1](/glossary/claude-fable-51) at the apex of Anthropic's capability tier, sitting above [Claude Opus 5.5](/glossary/claude-opus-5-5) as the designated escalation model for tasks where cheaper models fail quietly or unforced tool errors derail multi-hour agent runs.

For a normal user, Fable 5.2 is available on Claude.ai across Pro, Team, and Enterprise plans, as well as in Claude Code and Claude Cowork. For a developer, it is a proprietary hosted model accepting text and image inputs and returning text. Adaptive thinking is always enabled and steered per request through an effort parameter that defaults to high in Claude Code and medium in Claude.ai, with native support for tool calling, MCP server integration, and desktop computer use.

## Core profile

| Specification | Claude Fable 5.2 |
| --- | --- |
| Developer | Anthropic |
| Release date | November 10, 2026 |
| API model ID | `claude-fable-5-2` |
| Context window | 1,000,000 tokens |
| Maximum output | 128,000 tokens (up to 300,000 tokens in Batch API) |
| Knowledge cutoff | September 2026 |
| Input modalities | Text, images (documents, screenshots, diagrams) |
| Output modalities | Text (code, structured JSON, tool calls) |
| Reasoning / effort | Adaptive thinking (always on); `low`, `medium`, `high` (default for Claude Code), `max` |
| Pricing (1M tokens) | $10.00 base input / $50.00 base output |
| Prompt cache reads | $0.25 per million tokens (97.5% discount) |
| Cache writes | $12.50 (5-minute TTL) / $20.00 (1-hour TTL) |
| Batch discount | 50% discount on input and output ($5 / $25) |
| Tool integration | Function calling, strict structured outputs, MCP servers, computer use |
| Availability | Claude API, Claude.ai, Claude Code, Amazon Bedrock, Google Cloud, Microsoft Foundry |
| Weights / license | Proprietary; hosted API only |

The one-million-token [context window](/glossary/context-window) allows Fable 5.2 to ingest entire codebases, multi-volume technical documentation, or extensive session histories without aggressive summarization. The 128,000-token output limit provides ample head-room for full-file code rewrites and comprehensive research syntheses. Like Fable 5.1 and Opus 5.5, thinking tokens are billed as standard output tokens, making test-time compute budgeting an explicit design consideration.

## Lineup positioning and generation changes

In Anthropic's late-2026 model hierarchy, Fable 5.2 serves as the flagship intelligence ceiling:

- **Versus Claude Fable 5.1:** Base input and output token prices remain identical at $10 and $50 per million tokens, and cache reads remain $0.25 per million tokens. The primary advancements are reasoning density, fewer unforced tool errors during deep agent recursions, and superior recovery from syntax or environmental failures. Anthropic reports a 12.4-point gain on Terminal-Bench 4.0 and an 10.9-point gain on Terminal-Bench-Science 0.1.
- **Versus Claude Opus 5.5:** Opus 5.5 remains Anthropic's workhorse flagship at $4/$20 per million tokens. Fable 5.2 is designed for the subset of problems that outstrip Opus 5.5: whole-repository refactoring spanning dozens of files, novel algorithmic problem solving, automated root-cause analysis in legacy software, and high-consequence corporate decision-making. Where Opus 5.5 aims for maximum cost efficiency on day-long tasks, Fable 5.2 prioritizes task completion probability on frontier-grade difficulty.
- **Versus Gemini 4 Carbon & GPT-6 Astra:** Fable 5.2 competes directly with Google DeepMind's Gemini 4 Carbon and OpenAI's GPT-6 Astra for frontier coding and reasoning. Unlike Carbon, which launched with restricted access under the Fairwind Program, Fable 5.2 is generally available across the Claude API and major hyperscalers on day one.

## Benchmark profile

Anthropic's launch scorecard positions Fable 5.2 ahead of Fable 5.1 and Claude Opus 5.5 across major software engineering, computer use, and domain-expert reasoning benchmarks, while remaining competitive with Gemini 4 Carbon and GPT-6 Astra.

| Evaluation | Claude Fable 5.2 | Claude Opus 5.5 | Claude Fable 5.1 | Gemini 4 Carbon | GPT-6 Astra |
| --- | ---: | ---: | ---: | ---: | ---: |
| Terminal-Bench 4.0 | **68.2%** (max) | 66.4% (xhigh) | 55.8% | 65.2% | 59.1% |
| Terminal-Bench-Science 0.1 | 63.5% | 58.7% | 52.6% | **64.8%** | 63.3% |
| FrontierCode v1.1 Main | **58.2%** | 54.4% | 50.3% | 51.0% | 53.8% |
| SWE-sweep | **11.2%** | 3.6% | 2.9% | 9.6% | 9.5% |
| Humanity's Last Exam (tools) | **72.8%** | 67.7% | 65.6% | 71.6% | 73.1% |
| OSWorld 2.1 (offline subset) | **88.5%** | 87.2% | 80.7% | 81.5% | 80.9% |
| AutomationBench | 45.2% | 40.0% | 31.4% | **53.9%** | 41.4% |
| GDPval-AA v2.1 | **1912 Elo** | 1846 Elo | 1735 Elo | 1810 Elo | 1824 Elo |

These numbers reflect Anthropic's reported figures alongside public leaderboard baselines. Several evaluation subtleties apply:

Fable 5.2 scores use adaptive thinking at max effort for Terminal-Bench 4.0 and high effort for FrontierCode. On Terminal-Bench 4.0, Anthropic notes that the gap between Fable 5.2 (68.2%) and Opus 5.5 (66.4%) widens substantially in multi-repo test harnesses where dependencies must be discovered without pre-configured environments. On SWE-sweep, Fable 5.2 reaches 11.2%, edging out Gemini 4 Carbon's 9.6% and Astra's 9.5%. On AutomationBench, Zapier's benchmark for automated workflows, Fable 5.2 advances to 45.2% but trails Gemini 4 Carbon (53.9%), where Google's model demonstrates superior multimodal document parsing across messy business forms.

As with earlier Fable models, benchmark measurements evaluate the model, the [agent harness](/glossary/agent-harness), and the runtime sandbox together. Differences in tool schemas, bash timeouts, and verification loops can swing pass rates by several points.

## Pricing and cache economics

Claude Fable 5.2 retains Anthropic's premium pricing tier:

| Token Type | Standard Rate (per 1M) | Batch API (50% Off) |
| --- | ---: | ---: |
| Base input | $10.00 | $5.00 |
| Prompt cache reads | $0.25 | $0.125 |
| 5-minute cache writes | $12.50 | $6.25 |
| 1-hour cache writes | $20.00 | $10.00 |
| Base output | $50.00 | $25.00 |

Prompt caching is central to making Fable 5.2 economically viable in production. At $0.25 per million cached tokens, reading a 500,000-token repository context costs just $0.125 per turn, compared to $5.00 for an uncached pass. For iterative agent loops where an agent explores, tests, and edits code across 20 turns, prompt caching reduces the cumulative input bill by more than 90%.

Because adaptive thinking produces output tokens at $50 per million, total task cost is driven by reasoning depth. Anthropic recommends using `effort: "medium"` or `effort: "high"` for general coding and reserving `effort: "max"` for architectural overhauls, formal proofs, or elusive race conditions.

## Safeguards and Mythos 5.2

Fable 5.2 shares its underlying foundation with Claude Mythos 5.2. As with the 5.1 release, Anthropic maintains a dual-tier deployment architecture:

- **Public production safeguards:** Claude Fable 5.2 ships with general availability safeguards that monitor tool usage, prevent destructive host operations, and refuse harmful requests in cybersecurity, biological hazards, and chemical weapons. Requests involving offensive vulnerability exploitation are either refused or routed to restricted verifier pipelines.
- **Mythos 5.2 restricted tier:** High-risk offensive cyber validation and biochemical exploration capabilities are isolated within Mythos 5.2, accessible only to verified security researchers and government agencies through Anthropic's Cyber Verification Program.
- **Behavioral audit & alignment:** In Anthropic's automated behavioral evaluation across 2,500 deceptive scenarios, Fable 5.2 demonstrated an 88% reduction in covert goal drift and zero observed attempts to circumvent boundary checks.
- **Preserved thinking:** To counter [model distillation](/glossary/model-distillation) and protect proprietary reasoning traces, thinking blocks remain cryptographically bound to the generating conversation. Editing historical context invalidates downstream thinking signatures.
- **Content provenance:** All completions include statistical watermarking compliant with the EU AI Act and international provenance standards.

## API and behavior changes

Developers migrating to `claude-fable-5-2` should account for several behavioral characteristics:

- **Adaptive thinking enforcement:** Thinking cannot be disabled (`effort: "none"` is unsupported). API requests that specify `tool_choice: { type: "any" }` or attempt forced tool execution return an error; tool usage must be resolved autonomously by the model.
- **Thinking block continuity:** Thinking blocks are strictly bound to previous turn hashes. Injecting modified assistant messages or tampering with turn histories causes subsequent completion calls to fail verification.
- **Tool call serialization:** Unlike earlier Claude releases that frequently bundled multiple parallel tool calls into a single turn, Fable 5.2 favors deliberate, sequential execution on complex coding tasks, verifying intermediate output before issuing secondary modifications.
- **Inter-turn progress narration:** In long agentic tasks, progress updates between tool invocations are emitted inside thinking blocks. Front-end interfaces must configure streaming display parsers accordingly to show user-visible status updates.

## Applications and workflow fit

Claude Fable 5.2 is optimal for:

- **Complex multi-file refactoring:** Migrating frameworks, converting large codebases between programming languages, and resolving cross-package dependency cycles.
- **Autonomous coding agents:** Powering persistent agent harnesses like Claude Code where high accuracy on the first attempt outweighs raw token cost.
- **Deep root-cause analysis:** Investigating intermittent concurrency bugs, memory leaks, or compiler failures that require inspecting dozens of source files and execution logs.
- **Regulatory, legal, and financial auditing:** Synthesizing massive document collections where silent hallucinations carry catastrophic liability.

It is less suited for:

- High-throughput conversational chatbots, simple text classification, and single-turn summarization where Claude Sonnet 5.5 or Haiku 5.5 deliver equivalent functional utility at 80% to 95% lower cost.
- Latency-critical pipelines requiring sub-second time-to-first-token responses.

## Bottom line

Claude Fable 5.2 establishes Anthropic's new high-water mark for difficult reasoning and autonomous coding. By combining frontier benchmark scores—including 68.2% on Terminal-Bench 4.0 and 11.2% on SWE-sweep—with $0.25 prompt caching and day-one availability across major clouds, it offers a formidable escalation engine for long-horizon agent systems. Teams adopting it should enforce prompt caching, align reasoning effort with task complexity, and reserve its $50/M output capacity for challenges where cheaper models reliably fail.$body$::text AS body,
    'Anthropic''s Nov 2026 premium frontier model for autonomous coding, complex multi-step reasoning, and long-horizon agent workflows, with 1M context and $10/$50 pricing.'::text AS summary
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:claude-fable-5-2',
  'glossary',
  'claude-fable-5-2',
  '',
  'glossary/claude-fable-5-2',
  'Claude Fable 5.2',
  fable_52_entry.summary,
  fable_52_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', fable_52_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Claude Fable 5.2 model overview', 'url', 'https://platform.claude.com/docs/en/models/fable-5-2/overview'),
    jsonb_build_object('title', 'Anthropic introduces Claude Fable 5.2', 'url', 'https://www.anthropic.com/news/claude-fable-5-2'),
    jsonb_build_object('title', 'Claude Fable 5.2 and Mythos 5.2 System Card', 'url', 'https://www.anthropic.com/claude-fable-5-2-system-card'),
    jsonb_build_object('title', 'Claude Platform Pricing', 'url', 'https://platform.claude.com/docs/en/about-claude/pricing')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array(
      'claude-fable-51',
      'claude-fable-5',
      'claude-opus-5-5',
      'claude-sonnet-5-5',
      'anthropic',
      'claude-code',
      'large-language-model',
      'ai-agent',
      'agentic-ai',
      'agent-harness',
      'context-window',
      'benchmark',
      'api',
      'model-distillation'
    ),
    'seoDescription', 'Claude Fable 5.2 model card: Anthropic''s Nov 2026 frontier model, 1M context window, benchmarks, $10/$50 token pricing, safeguards, and availability.',
    'seoKeywords', jsonb_build_array(
      'Claude Fable 5.2',
      'Claude Fable 5.2 benchmarks',
      'Claude Fable 5.2 pricing',
      'Claude Fable 5.2 model card',
      'Claude Fable 5.2 vs Opus 5.5',
      'Claude Fable 5.2 context window',
      'claude-fable-5-2',
      'Anthropic Claude Fable 5.2',
      'Claude Fable 5.2 API'
    )
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-11-10',
  0
FROM fable_52_entry
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
    WHEN COALESCE(metadata->'relatedTerms', '[]'::jsonb) @> '["claude-fable-5-2"]'::jsonb
      THEN metadata
    ELSE jsonb_set(
      COALESCE(metadata, '{}'::jsonb),
      '{relatedTerms}',
      COALESCE(metadata->'relatedTerms', '[]'::jsonb) || jsonb_build_array('claude-fable-5-2'),
      true
    )
  END,
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug IN ('claude-fable-51', 'claude-fable-5', 'claude-opus-5-5', 'anthropic')
  AND parent_slug = '';

INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at, updated_at
)
VALUES (
  'claude-fable-5-2',
  'Claude Fable 5.2',
  'Anthropic',
  DATE '2026-11-10',
  DATE '2026-11-10',
  'proprietary',
  'Claude Fable',
  $specs${
    "Developer": "Anthropic",
    "Release date": "2026-11-10",
    "API model ID": "claude-fable-5-2",
    "Context window": "1M tokens",
    "Max output": "128K tokens (300K in Batch API)",
    "Knowledge cutoff": "Sep 2026",
    "Reasoning / effort": "Adaptive thinking (always on); default high in Claude Code, medium in Claude.ai; per-message effort",
    "Input / 1M tokens": "$10",
    "Cached input / 1M": "$0.25",
    "Cache write / 1M": "$12.50 (5m) / $20 (1h)",
    "Output / 1M tokens": "$50",
    "Batch / flex discount": "Batch API: 50% off input and output",
    "Text input": "Yes",
    "Image / vision input": "Yes",
    "Audio input": "No",
    "Video input": "No",
    "Text output": "Yes",
    "Image output": "No",
    "Audio output": "No",
    "Video output": "No",
    "Tool / function calling": "Yes, function calling and strict structured outputs. Forced tool choice rejected",
    "Computer use": "Yes",
    "Terminal-Bench 4.0": "68.2%",
    "Terminal-Bench": "68.2% (Terminal-Bench 4.0; max effort)",
    "Terminal-Bench-Science 0.1": "63.5%",
    "FrontierCode v1.1": "58.2%",
    "SWE-sweep": "11.2%",
    "Humanity's Last Exam": "72.8% (with tools)",
    "HLE-Verified": "72.8% (with tools)",
    "OSWorld 2.0": "88.5% (offline subset)",
    "OSWorld": "88.5% (OSWorld 2.1 offline subset)",
    "AutomationBench": "45.2%",
    "MCP / tool-use benchmark": "AutomationBench 45.2%",
    "API access": "Yes",
    "Product access": "Claude API + Claude.ai + Claude Code + Amazon Bedrock + Google Cloud + Microsoft Foundry",
    "Weights / license": "Proprietary",
    "Primary focus": "Peak frontier software engineering, multi-hour agent loops, deep reasoning, and high-consequence enterprise workflows",
    "Long-horizon work": "Built for persistent agent loops that reuse repository or document context across requests with $0.25 cached reads",
    "Safety / approvals": "Production safeguards with Life Sciences & Cyber Verification Programs; shared foundation with restricted Mythos 5.2"
  }$specs$::jsonb,
  $sources$[
    {"title": "Claude Fable 5.2 model overview", "url": "https://platform.claude.com/docs/en/models/fable-5-2/overview"},
    {"title": "Anthropic introduces Claude Fable 5.2", "url": "https://www.anthropic.com/news/claude-fable-5-2"},
    {"title": "Claude Fable 5.2 and Mythos 5.2 System Card", "url": "https://www.anthropic.com/claude-fable-5-2-system-card"},
    {"title": "Claude Platform Pricing", "url": "https://platform.claude.com/docs/en/about-claude/pricing"}
  ]$sources$::jsonb,
  'Anthropic''s November 2026 premium frontier model succeeding Fable 5.1. 1M context window, 128K output (300K in Batch), always-on adaptive thinking, $10/$50 per 1M tokens with $0.25 prompt cache reads. Scores 68.2% on Terminal-Bench 4.0 at max effort, 63.5% on Terminal-Bench-Science 0.1, 58.2% on FrontierCode v1.1, 11.2% on SWE-sweep, and 72.8% on Humanity''s Last Exam with tools.',
  '{"launch_stage": "ga", "verification_pass": "claude-fable-5-2-2026-11-10", "default_effort": "high"}'::jsonb,
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
  {"id":"tq-20261110-fable52-terminalbench40","model_slug":"claude-fable-5-2","category":"coding","benchmark_name":"Terminal-Bench 4.0","benchmark_version":null,"score_numeric":68.2,"score_display":"68.2%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"Anthropic","evaluation_date":"2026-11-10","source":"https://www.anthropic.com/claude-fable-5-2-system-card","notes":"Launch evaluation at max effort."},
  {"id":"tq-20261110-fable52-terminalbenchscience","model_slug":"claude-fable-5-2","category":"math_reasoning","benchmark_name":"Terminal-Bench Science 0.1","benchmark_version":null,"score_numeric":63.5,"score_display":"63.5%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"Anthropic","evaluation_date":"2026-11-10","source":"https://www.anthropic.com/claude-fable-5-2-system-card","notes":"Launch evaluation at max effort."},
  {"id":"tq-20261110-fable52-hle-tools","model_slug":"claude-fable-5-2","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":72.8,"score_display":"72.8%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":null,"evaluator":"Anthropic","evaluation_date":"2026-11-10","source":"https://www.anthropic.com/claude-fable-5-2-system-card","notes":"With tools at high effort."}
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
