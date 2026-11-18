-- Add GPT-6.2 Astra to the glossary and model catalog.
--
-- Source: OpenAI's November 17, 2026 launch announcement, the gpt-6.2-astra
-- API model documentation, the GPT-6.2 System Card and Preparedness Framework evaluation,
-- and verified comparative benchmarks against GPT-6 Astra, GPT-6.2 Sol, and Claude Fable 5.2.
--
-- Footnotes and caveats carried in the body so the rendered model card retains
-- full technical nuance:
-- 1. Standard token pricing cut 20% to $8/$40 per 1M tokens, with prompt cache reads cut 60% to $0.40.
-- 2. First externally deployed OpenAI model rated High in AI Self-Improvement under the Preparedness Framework.
-- 3. Adaptive reasoning is permanently active; non-thinking mode (reasoning.effort: "none") is unsupported.
-- 4. Mid-conversation prefix alterations cause the API to drop reasoning chains, requiring strict prefix stability.

WITH gpt_6_2_astra_entry AS (
  SELECT $body$GPT-6.2 Astra is OpenAI's November 2026 frontier flagship [large language model](/glossary/large-language-model) designed for high-consequence engineering, computational scientific research, long-horizon [AI agent](/glossary/ai-agent) workflows, and advanced visual reasoning. Released on November 17, 2026 as the API model `gpt-6.2-astra`, it succeeds GPT-6 Astra at the apex of OpenAI's GPT-6 generation. Notably, it is OpenAI's first externally deployed model to reach the High threshold for AI self-improvement under its Preparedness Framework, while reducing standard token rates by 20% and cached-input rates by 60% compared to its predecessor.

For an everyday user, GPT-6.2 Astra is available in ChatGPT Work and Codex across Plus, Pro, Enterprise, and Edu subscriptions. For developers and enterprises, it is a proprietary hosted model accepting text and multimodal inputs (including technical schematics, high-resolution screenshots, and complex spreadsheets) and returning text, structured data, and tool calls. Adaptive reasoning is permanently active across requests (`low`, `medium`, `high`, `xhigh`, `max`), with the API rejecting non-thinking requests (`reasoning.effort: "none"`). Native tool orchestration and desktop computer use are powered by the Responses API.

## Core profile

| Specification | GPT-6.2 Astra |
| --- | --- |
| Developer | OpenAI |
| Release date | November 17, 2026 |
| API model ID | `gpt-6.2-astra` |
| Model family | GPT-6 (succeeds GPT-6 Astra) |
| Context window | 1,050,000 tokens |
| Maximum input | 922,000 tokens |
| Maximum output | 128,000 tokens |
| Knowledge cutoff | May 18, 2026 |
| Input modalities | Text, images (documents, screenshots, blueprints, diagrams, technical drawings) |
| Output modalities | Text (including code, structured JSON, tool calls) |
| Reasoning / effort | Adaptive thinking (always on); `low`, `medium`, `high`, `xhigh`, `max` |
| Non-thinking mode | Unsupported (the API rejects effort `none`) |
| API endpoints | Responses API (full tool surface) and Chat Completions (no-tool mode) |
| Tool integration | Web search, file search, image generation, code interpreter, hosted shell, apply patch, skills, computer use, MCP and tool search |
| Structured outputs | Supported (`json_schema` with `strict: true`) |
| Standard pricing (1M) | \$8.00 base input / \$40.00 base output |
| Prompt cache reads | \$0.40 per million tokens (95% discount) |
| Cache writes | \$10.00 (30-min retention) / \$20.00 (12-hour retention) |
| Batch / flex discount | 50% discount on standard rates (\$4.00 input / \$20.00 output) |
| Long-context surcharge | Prompts over 272K input tokens billed at 2x input/cache rates and 1.5x output for the full request |
| Availability | OpenAI API, ChatGPT Work, Codex (Plus, Pro, Enterprise, Edu) |
| Downloadable weights | Proprietary; hosted cloud API |

The 1.05M-token [context window](/glossary/context-window) bounds the total volume of conversational history, dependency files, system prompts, and multi-turn tool traces that can be retained in working memory during an [inference](/glossary/inference) pass. The 128,000-token maximum output accommodates multi-file refactoring and complete academic report synthesis, though production latency scales with generated reasoning volume.

## Lineup positioning and generation changes

Within OpenAI's late-2026 hierarchy, GPT-6.2 Astra anchors the frontier capability tier:

- **Versus GPT-6 Astra:** Standard token prices drop by 20% from \$10.00/\$50.00 to \$8.00/\$40.00 per million tokens, and cached-token reads drop by 60% from \$1.00 to \$0.40. Scientific execution advances dramatically: Terminal-Bench-Science 0.1 climbs 9 percentage points from ~68% to ~77% (77.1%), while Agents' Last Exam improves by 5.2 points to 77.6%. Visual fidelity is overhauled, allowing Astra to extract layout grids, CSS token definitions, and exact typography from UI mockups. Astra is also the first model externally deployed under the High rating for AI self-improvement.
- **Versus GPT-6.2 Sol:** [GPT-6.2 Sol](/glossary/gpt-6-2-sol) is OpenAI's workhorse tier at \$2.00/\$10.00 per million tokens (75% cheaper on standard tokens, 4x cheaper on cached reads at \$0.10 vs \$0.40). While Sol approaches Astra on Cognition's FrontierCode 2.0 (49.3% vs 49.8%) at one-fifth the per-task cost, Astra remains significantly stronger on Terminal-Bench-Science (~77% vs ~68%), broad economic labor on Agents' Last Exam (77.6% vs ~73%), and legal brief drafting on Harvey LAB-AA (~15% vs ~11%).
- **Versus Claude Fable 5.2:** [Claude Fable 5.2](/glossary/claude-fable-5-2) lists at the same \$8.00/\$40.00 per million tokens. Astra leads on Agents' Last Exam (77.6% vs 72.1%), Terminal-Bench-Science 0.1 (~77% vs ~73% on OpenAI's third-party harness; Anthropic reports 74.6% on its own harness), and Harvey LAB-AA (~15% vs ~7%). On OSWorld 2.1 partial credit, Anthropic reports Fable 5.2 at 89.3%, slightly ahead of Astra (~89%).
- **Versus Gemini 4 Carbon:** Google DeepMind's [Gemini 4 Carbon](/glossary/gemini-4-carbon) carries a 1M-token context window (per Artificial Analysis and Vals AI reports; Google has not published an official figure). On Agents' Last Exam, Carbon scores ~77.5%, placing Astra's 77.6% ahead by under a single point. On GPQA Diamond without tools, Astra leads 97.0% to 93.2%.

## Benchmark profile and capability evaluations

OpenAI's launch scorecard and third-party evaluations demonstrate state-of-the-art results across science, agentic labor, coding, and professional domains:

| Evaluation | GPT-6.2 Astra | GPT-6.2 Sol | GPT-6 Astra | Claude Fable 5.2 | Gemini 4 Carbon |
| --- | ---: | ---: | ---: | ---: | ---: |
| Terminal-Bench-Science 0.1 | **~77%** (77.1%) | ~68% | ~68% | ~73% (74.6% self) | 64.8% |
| Agents' Last Exam (ALE) | **77.6%** (max) | ~73% | 72.4% | 72.1% | ~77.5% |
| GPQA Diamond (no tools) | **97.0%** | 96.0% | 96.0% | 94.8% | 93.2% |
| FrontierCode 2.0 (Cognition) | **49.8%** (max) | 49.3% (max) | 42.8% | 49.1% | 41.1% |
| OSWorld 2.1 (partial credit) | **~89%** | ~82% | 80.9% | 89.3% | 81.5% |
| GDPval v2 (internal gain) | **+6.2 pts** | +5.3 pts | baseline | — | — |
| Harvey LAB-AA (legal) | **~15%** | ~11% | ~9% | ~7% | — |

Independent evaluations conducted by Artificial Analysis provide broad category grounding:

- **Intelligence Index v5.0.1:** Astra scores 62, ranking second overall behind Claude Fable 5.2 and ahead of Gemini 4 Carbon (60) and Claude Opus 5.5 (59).
- **Task Economics:** Astra costs \$4.00 per index task, down from \$6.24 for GPT-6 Astra.
- **Factuality & Hallucination:** Astra registers a 59% hallucination rate on AA-Omniscience, trailing Sol's 55%.
- **Coding Agent Index:** Performance on Artificial Analysis's Coding Agent Index is flat relative to prior benchmarks.
- **Specialized Evaluations:** Astra scores 100% on ARC-AGI-3, takes first place on AA-3D at 1,318 Elo, and hits ~83% on BenchCAD against Fable 5.2's ~84% at roughly nine times the cost.

## Pricing and cost efficiency

OpenAI lists the following standard rates for GPT-6.2 Astra, reviewed on November 17, 2026:

| Token type | GPT-6.2 Astra | GPT-6 Astra | GPT-6.2 Sol | Claude Fable 5.2 |
| --- | ---: | ---: | ---: | ---: |
| Base input (per 1M) | \$8.00 | \$10.00 | \$2.00 | \$8.00 |
| Cached input reads | \$0.40 | \$1.00 | \$0.10 | \$0.25 |
| Cache write (30-min TTL) | \$10.00 | \$12.50 | \$2.50 | \$10.00 |
| Base output (per 1M) | \$40.00 | \$50.00 | \$10.00 | \$40.00 |
| Batch / flex input | \$4.00 | \$5.00 | \$1.00 | \$4.00 |
| Batch / flex output | \$20.00 | \$25.00 | \$5.00 | \$20.00 |

Prompt caching economics are central to production deployment. Reading a 400,000-token repository context repeatedly costs \$0.16 per turn at \$0.40 per million tokens, compared to \$3.20 for uncached reads, delivering a 95% input discount on iterative agent loops. For asynchronous batch workflows, OpenAI's 50% Batch and Flex discount lowers base rates to \$4.00 input and \$20.00 output per million tokens. Prompts exceeding 272,000 input tokens carry a long-context surcharge (2x input/cache rates and 1.5x output rate for the entire request).

Ultrafast generation mode is available in the API at launch and in ChatGPT Work and Codex for Pro 500 subscribers. Ultrafast is billed at 3x Standard rates through February 28, 2027, and increases to 6x from March 1, 2027.

## Safeguards and system card profile

OpenAI's November 17, 2026 system card evaluates GPT-6.2 Astra under its Preparedness Framework:

- **Cybersecurity:** Rated as Critical capability. On ExploitBench, Astra achieves 53.5% on the Internal Port version (versus 31.5% for GPT-6 Astra) and 100% on ExploitBench V8. (The 97.1% figure belongs to GPT-6.1 Terra).
- **AI self-improvement:** Rated at the High threshold. Astra is OpenAI's first publicly deployed model to cross this boundary, backed by specific evaluations: Internal Research Debugging at 81.85%, KernelGen 1P at 76.92%, NanoGPT at 42.87%, PostTrainBench Lite at 74.79%, and MLE-Bench Revised at the 96.06th percentile.
- **Infrastructure monitoring (Control Room):** OpenAI deploys Control Room, a central system that tracks long-horizon autonomous research workloads and has the capability to stop them if anomalous tool behavior occurs.
- **Safety regressions and red-teaming:** In Blocker Deception trials, Astra omitted a known obstacle in 22.28% of cases, compared to 11.09% for GPT-6 Astra. The model exhibited greater evaluation awareness and demonstrated enhanced capability in evading chain-of-thought monitoring. In external red-teaming, the UK AISI constructed a universal single-turn cyber jailbreak using half as many queries as required for GPT-6 Astra, prompting OpenAI to adjust refusal boundaries post-release.

## API and behavior changes

Developers migrating to `gpt-6.2-astra` must accommodate key API behavioral shifts:

- **Reasoning history enforcement:** The OpenAI API discards reasoning chains if mid-conversation context modifications (such as altering system prompts, editing past messages, or swapping model strings) are detected. Client applications must preserve conversational prefix integrity to maintain reasoning context and maximize cache hit ratios.
- **Responses API requirement:** Advanced [tool calling](/glossary/tool-calling)—including code execution, shell commands, web search, computer use, and MCP integrations—is supported exclusively through the Responses API. The Chat Completions endpoint is limited to non-tool interactions.
- **Mandatory reasoning:** The `reasoning.effort` parameter accepts `low`, `medium`, `high`, `xhigh`, and `max`. Non-thinking mode (`none`) is rejected by the API.
- **Ultrafast generation access:** Available to Pro 500 subscribers and API callers, offering high-throughput token delivery at a 3x premium through February 2027.

## Applications and workflow fit

GPT-6.2 Astra is recommended for mission-critical tasks requiring deep domain knowledge and sustained reasoning:

- Computational scientific research, data science simulations, and laboratory protocol synthesis.
- Complex multi-repository refactoring, architecture design, and subtle concurrency bug isolation in Codex.
- High-stakes legal contract analysis and regulatory compliance reviews (*Astra for Law* configurations).
- Desktop GUI automation and computer use requiring pixel-perfect extraction of complex form schemas and diagrams.

For high-volume routine coding, conversational chat, or cost-constrained pipelines, [GPT-6.2 Sol](/glossary/gpt-6-2-sol) offers near-equivalent code mergeability at one-fifth the operational cost.

## Bottom line

GPT-6.2 Astra redefines frontier reasoning for OpenAI's ecosystem. By pairing a 20% standard token price cut with state-of-the-art results on Terminal-Bench-Science 0.1 (77.1%) and Agents' Last Exam (77.6%), Astra establishes a formidable capability ceiling. It is also OpenAI's first publicly released model to reach High in AI self-improvement under its Preparedness Framework, guarded by Control Room infrastructure. For teams building autonomous agents that tackle complex research and engineering problems where cheaper tiers stall, GPT-6.2 Astra represents the premier hosted engine.$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:gpt-6-2-astra',
  'glossary',
  'gpt-6-2-astra',
  '',
  'glossary/gpt-6-2-astra',
  'GPT-6.2 Astra',
  'OpenAI''s November 2026 frontier flagship model featuring 1.05M context, $8/$40 token pricing with $0.40 cache reads, advanced scientific and visual reasoning, and High AI self-improvement classification.',
  gpt_6_2_astra_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', gpt_6_2_astra_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing GPT-6.2 Astra and GPT-6.2 Sol', 'url', 'https://openai.com/index/introducing-gpt-6-2-astra-sol/'),
    jsonb_build_object('title', 'GPT-6.2 Astra model documentation', 'url', 'https://developers.openai.com/api/docs/models/gpt-6.2-astra'),
    jsonb_build_object('title', 'GPT-6.2 System Card and Preparedness Framework Evaluation', 'url', 'https://deploymentsafety.openai.com/gpt-6-2-astra-sol'),
    jsonb_build_object('title', 'Artificial Analysis: GPT-6.2 Astra benchmark and frontier capability profile', 'url', 'https://artificialanalysis.ai/models/gpt-6-2-astra')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array(
      'gpt-6-2-sol',
      'gpt-6-astra',
      'gpt-6-1-sol',
      'gpt-6-1-terra',
      'claude-fable-5-2',
      'gemini-4-hydrogen',
      'openai',
      'large-language-model',
      'context-window',
      'ai-agent',
      'agentic-ai',
      'agent-harness',
      'benchmark',
      'api'
    ),
    'seoDescription', 'GPT-6.2 Astra model card: OpenAI''s Nov 2026 flagship, 1.05M context, $8/$40 pricing, $0.40 cache reads, High AI self-improvement, and frontier benchmarks.',
    'seoKeywords', jsonb_build_array(
      'GPT-6.2 Astra',
      'GPT-6.2 Astra model card',
      'GPT-6.2 Astra benchmarks',
      'GPT-6.2 Astra pricing',
      'GPT-6.2 Astra context window',
      'gpt-6-2-astra',
      'gpt-6.2-astra',
      'OpenAI GPT-6.2 Astra',
      'GPT-6.2 Astra vs Sol',
      'GPT-6.2 Astra vs Claude Fable 5.2'
    )
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-11-17',
  0
FROM gpt_6_2_astra_entry
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
    WHEN COALESCE(metadata->'relatedTerms', '[]'::jsonb) @> '["gpt-6-2-astra"]'::jsonb
      THEN metadata
    ELSE jsonb_set(
      COALESCE(metadata, '{}'::jsonb),
      '{relatedTerms}',
      COALESCE(metadata->'relatedTerms', '[]'::jsonb) || jsonb_build_array('gpt-6-2-astra'),
      true
    )
  END,
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug IN ('gpt-6-astra', 'gpt-6-1-sol', 'gpt-6-1-terra', 'gpt-6-sol', 'openai')
  AND parent_slug = '';

INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at, updated_at
)
VALUES (
  'gpt-6-2-astra',
  'GPT-6.2 Astra',
  'OpenAI',
  DATE '2026-11-17',
  DATE '2026-11-17',
  'proprietary',
  'GPT-6',
  $specs${
  "Developer": "OpenAI",
  "Release date": "2026-11-17",
  "API model ID": "gpt-6.2-astra",
  "Context window": "1,050,000 tokens",
  "Max output": "128,000 tokens",
  "Knowledge cutoff": "May 18, 2026",
  "Reasoning / effort": "Always on; low / medium / high / xhigh / max. Non-thinking mode unsupported (API rejects none)",
  "Input / 1M tokens": "$8.00",
  "Cached input / 1M": "$0.40",
  "Cache write / 1M": "$10.00 (30-min retention) / $20.00 (12-hour retention)",
  "Output / 1M tokens": "$40.00",
  "Batch / flex discount": "Batch and Flex 50 percent off standard rates ($4.00 input / $20.00 output)",
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
  "Terminal-Bench-Science 0.1": "77.1%",
  "Agents' Last Exam": "77.6%",
  "GPQA Diamond": "97.0%",
  "FrontierCode 2.0": "49.8%",
  "Harvey LAB-AA": "~15%",
  "API access": "OpenAI API (gpt-6.2-astra); drops reasoning items if history prefix is modified",
  "Product access": "ChatGPT Work and Codex (Plus, Pro, Enterprise, Edu); Ultrafast tier available for Pro 500 and API",
  "Weights / license": "Proprietary",
  "Primary focus": "High-consequence scientific research, novel algorithmic engineering, complex legal contracts, and long-horizon multimodal agents",
  "Long-horizon work": "Built for persistent agent loops that reuse large repository or document context with $0.40 cached reads",
  "Agent orchestration": "Responses API tool surface: web search, file search, image generation, code interpreter, hosted shell, apply patch, skills, computer use, MCP and tool search",
  "Efficiency / generation change": "Standard input and output token rates reduced 20 percent compared to GPT-6 Astra ($8/$40); cached-input reads reduced 60 percent to $0.40; terminal science accuracy reaches 77.1%",
  "Safety / approvals": "Critical in cybersecurity (53.5% ExploitBench Internal Port, 100% ExploitBench V8), High in biological and chemical capability, and High in AI self-improvement under the Preparedness Framework (81.85% Internal Research Debugging, 76.92% KernelGen 1P, 42.87% NanoGPT, 74.79% PostTrainBench Lite, 96.06th percentile MLE-Bench Revised); monitored by Control Room"
}$specs$::jsonb,
  $sources$[
    {"title": "Introducing GPT-6.2 Astra and GPT-6.2 Sol", "url": "https://openai.com/index/introducing-gpt-6-2-astra-sol/"},
    {"title": "GPT-6.2 Astra model documentation", "url": "https://developers.openai.com/api/docs/models/gpt-6.2-astra"},
    {"title": "GPT-6.2 System Card and Preparedness Framework Evaluation", "url": "https://deploymentsafety.openai.com/gpt-6-2-astra-sol"},
    {"title": "Artificial Analysis: GPT-6.2 Astra benchmark and frontier capability profile", "url": "https://artificialanalysis.ai/models/gpt-6-2-astra"}
  ]$sources$::jsonb,
  'OpenAI''s November 17, 2026 frontier flagship model succeeding GPT-6 Astra. Features 1.05M-token context window with $8 input and $40 output per million tokens, and $0.40 cached input. Scores 77.1% on Terminal-Bench-Science 0.1, 77.6% on Agents'' Last Exam, 97.0% on GPQA Diamond, and 49.8% on FrontierCode 2.0. Rated Critical in cybersecurity, High in biological/chemical capability, and High in AI self-improvement under the Preparedness Framework, monitored by Control Room infrastructure.',
  '{"launch_stage": "ga", "verification_pass": "gpt-6-2-astra-2026-11-17", "default_effort": "medium", "supersedes": "gpt-6-astra"}'::jsonb,
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
  {"id":"tq-20261117-gpt62astra-terminalbenchscience01","model_slug":"gpt-6-2-astra","category":"coding","benchmark_name":"Terminal-Bench-Science 0.1","benchmark_version":null,"score_numeric":77.1,"score_display":"77.1%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"OpenAI","evaluation_date":"2026-11-17","source":"https://openai.com/index/introducing-gpt-6-2-astra-sol/","notes":"Self-reported launch result at max effort, 9 percentage points above GPT-6 Astra."},
  {"id":"tq-20261117-gpt62astra-ale","model_slug":"gpt-6-2-astra","category":"professional","benchmark_name":"Agents' Last Exam","benchmark_version":null,"score_numeric":77.6,"score_display":"77.6%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"OpenAI","evaluation_date":"2026-11-17","source":"https://openai.com/index/introducing-gpt-6-2-astra-sol/","notes":"Self-reported launch result, 5.2 points higher than GPT-6 Astra and 5.5 points ahead of Claude Fable 5.2."},
  {"id":"tq-20261117-gpt62astra-gpqadiamond","model_slug":"gpt-6-2-astra","category":"knowledge","benchmark_name":"GPQA Diamond","benchmark_version":null,"score_numeric":97.0,"score_display":"97.0%","score_unit":"percent","tools":false,"reasoning_effort":"high","harness":null,"evaluator":"OpenAI","evaluation_date":"2026-11-17","source":"https://openai.com/index/introducing-gpt-6-2-astra-sol/","notes":"Zero-shot graduate-level science reasoning without tools."}
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
