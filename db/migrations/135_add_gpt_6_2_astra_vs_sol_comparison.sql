-- Publish the authored GPT-6.2 Astra vs GPT-6.2 Sol comparison page, following
-- the pattern established in 091 (GPT-6.1 Sol vs GPT-6 Sol) and 095 (MiMo V2.6
-- Pro vs Flash). Migrations 133 and 134 published the individual model cards and
-- catalog rows for both models, so the comparison links directly between them
-- and maintains consistency across specifications and benchmarks.
--
-- Sources:
-- OpenAI's November 17, 2026 announcement posts for GPT-6.2 Astra and GPT-6.2 Sol,
-- the developer documentation for gpt-6.2-astra and gpt-6.2-sol, the GPT-6.2
-- system card addenda, and verified benchmark evaluations.

WITH comparison_notes AS (
  SELECT $notes$<small>*Vendor figures are from OpenAI's November 17, 2026 launch announcements, developer documentation, and system card addenda. Both models share a 1,050,000-token context window and 128,000 maximum output tokens. Knowledge cutoff is May 18, 2026 for Astra and June 1, 2026 for Sol. Both models require adaptive reasoning (the API rejects effort none).*</small>

OpenAI released GPT-6.2 Astra and GPT-6.2 Sol simultaneously on November 17, 2026, structuring its frontier lineup into two distinct operational tiers. GPT-6.2 Astra is the flagship scientific and reasoning model, priced at $8 per million input tokens and $40 per million output tokens ($0.40 per million cached input tokens, $10 cache write), engineered for long-horizon academic research, advanced mathematical verification, and autonomous scientific discovery. GPT-6.2 Sol is the high-velocity developer and automation workhorse, priced at $2 per million input tokens and $10 per million output tokens ($0.10 per million cached input tokens, $2.50 cache write), designed for rapid code refactoring, terminal interaction, and cost-effective multi-turn agent loops.

Economics and prompt caching form the primary decision boundary for production architectures. While Sol is 75% cheaper on standard tokens ($2/$10 versus $8/$40), its prompt cache read rate is 4x cheaper: $0.10 per million tokens on Sol versus $0.40 on Astra. For autonomous software engineering agents that repeatedly re-read full repository contexts across 20 to 50 iterations, Sol operates at a small fraction of Astra's total operational expense.

The capability separation reflects their design targets. On Cognition's FrontierCode 2.0 benchmark, Sol lands within 0.5 points of Astra (49.3% vs 49.8% at maximum reasoning effort) at roughly one-fifth the per-task cost. On scientific reasoning, Astra establishes a decisive lead: on Terminal-Bench-Science 0.1, Astra reaches ~77% (77.1%) against Sol's ~68% (matching GPT-6 Astra's best). On Agents' Last Exam, Astra reaches 77.6% under maximum reasoning effort, leading Gemini 4 Carbon (~77.5%) by under a point and Sol (~73%) by 4.6 points. On GPQA Diamond without tools, both perform at the frontier (97.0% for Astra vs 96.0% for Sol). On legal contract drafting (Harvey LAB-AA), Astra scores ~15% versus ~11% for Sol.

Independent evaluations from Artificial Analysis confirm these dynamics: on the Intelligence Index v5.0.1, Astra scores 62 (ranking second overall behind Claude Fable 5.2 and ahead of Gemini 4 Carbon 60 and Claude Opus 5.5 59) at $4.00 per index task, while Sol scores 57 at approximately 15% of Astra's per-task cost. Sol achieves a lower hallucination rate on AA-Omniscience (55% vs Astra's 59%). On specialized benchmarks, Astra achieves 100% on ARC-AGI-3, 1,318 Elo on AA-3D, and ~83% on BenchCAD.

Operational controls are strictly aligned across both models. Both models reject non-thinking mode (reasoning.effort: "none"), requiring adaptive reasoning on every request across low, medium, high, xhigh, and max. In product access, both models are available in ChatGPT Work and Codex across Plus, Pro, Enterprise, and Edu subscriptions. Ultrafast generation mode is available in the API at launch, in ChatGPT Work and Codex starting at Pro 200 for Sol, and for Pro 500 across both models, billed at 3x Standard rates through February 28, 2027 and 6x from March 1, 2027. Both models require the Responses API for all tool execution, computer use, and image generation.$notes$::text AS notes
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'comparison:gpt-6-2-astra-vs-gpt-6-2-sol',
  'comparison',
  'gpt-6-2-astra-vs-gpt-6-2-sol',
  '',
  'comparisons/gpt-6-2-astra-vs-gpt-6-2-sol',
  'GPT-6.2 Astra vs GPT-6.2 Sol',
  'GPT-6.2 Astra vs GPT-6.2 Sol: comparing OpenAI''s frontier reasoning flagship ($8/$40) with its high-velocity coding and agentic workhorse ($2/$10).',
  comparison_notes.notes,
  $blocks$[
  {
    "id": "spec-gpt62astra-sol-1",
    "type": "spec_table",
    "title": "Specifications",
    "columns": [
      "GPT-6.2 Astra",
      "GPT-6.2 Sol"
    ],
    "rows": [
      [
        "Developer",
        "OpenAI",
        "OpenAI"
      ],
      [
        "Release date",
        "2026-11-17",
        "2026-11-17"
      ],
      [
        "API model ID",
        "gpt-6.2-astra",
        "gpt-6.2-sol"
      ],
      [
        "Context window",
        "1,050,000 tokens",
        "1,050,000 tokens"
      ],
      [
        "Max output",
        "128,000 tokens",
        "128,000 tokens"
      ],
      [
        "Knowledge cutoff",
        "May 18, 2026",
        "Jun 1, 2026"
      ],
      [
        "Reasoning / effort",
        "Always on: low / medium / high / xhigh / max (effort none is unsupported)",
        "Always on: low / medium / high / xhigh / max (effort none is unsupported)"
      ],
      [
        "Non-thinking mode",
        "Unsupported (the API rejects effort none)",
        "Unsupported (the API rejects effort none)"
      ],
      [
        "Fine-tuning",
        "Not supported",
        "Not supported"
      ],
      [
        "Data residency",
        "US and EU",
        "US and EU"
      ],
      [
        "Weights / license",
        "Proprietary",
        "Proprietary"
      ]
    ]
  },
  {
    "id": "spec-gpt62astra-sol-2",
    "type": "spec_table",
    "title": "Pricing",
    "columns": [
      "GPT-6.2 Astra",
      "GPT-6.2 Sol"
    ],
    "rows": [
      [
        "Input / 1M tokens",
        "$8",
        "**$2**"
      ],
      [
        "Cached input / 1M",
        "$0.40",
        "**$0.10**"
      ],
      [
        "Cache write / 1M",
        "$10.00",
        "$2.50"
      ],
      [
        "Output / 1M tokens",
        "$40",
        "**$10**"
      ],
      [
        "Batch / flex discount",
        "50 percent off standard ($4 input / $20 output)",
        "50 percent off standard ($1 input / $5 output)"
      ],
      [
        "Ultrafast pricing",
        "3x standard through Feb 28, 2027; 6x from Mar 1, 2027",
        "3x standard through Feb 28, 2027; 6x from Mar 1, 2027"
      ],
      [
        "Long-context surcharge",
        "Over 272K input tokens: 2x input and cache rates plus 1.5x output for the full request",
        "Over 272K input tokens: 2x input and cache rates plus 1.5x output for the full request"
      ],
      [
        "Free tier",
        "Not supported",
        "Not supported"
      ]
    ]
  },
  {
    "id": "spec-gpt62astra-sol-3",
    "type": "spec_table",
    "title": "Capabilities & access",
    "columns": [
      "GPT-6.2 Astra",
      "GPT-6.2 Sol"
    ],
    "rows": [
      [
        "Text input",
        "Yes",
        "Yes"
      ],
      [
        "Image / vision input",
        "Yes",
        "Yes"
      ],
      [
        "Audio input",
        "No",
        "No"
      ],
      [
        "Video input",
        "No",
        "No"
      ],
      [
        "Text output",
        "Yes",
        "Yes"
      ],
      [
        "Image output",
        "Yes, through the Responses API image generation tool",
        "Yes, through the Responses API image generation tool"
      ],
      [
        "Audio output",
        "No",
        "No"
      ],
      [
        "Video output",
        "No",
        "No"
      ],
      [
        "Tool / function calling",
        "Yes: streaming, function calling, structured outputs, computer use",
        "Yes: streaming, function calling, structured outputs, computer use. Tools require Responses API"
      ],
      [
        "Computer use",
        "Yes, through the Responses API",
        "Yes, through the Responses API"
      ],
      [
        "API access",
        "OpenAI API (gpt-6.2-astra); drops reasoning if prefix modified",
        "OpenAI API (gpt-6.2-sol); drops reasoning if prefix modified"
      ],
      [
        "Product access",
        "ChatGPT Work and Codex (Plus, Pro, Enterprise, Edu)",
        "ChatGPT Work and Codex (Plus, Pro, Enterprise, Edu)"
      ],
      [
        "Ultrafast access",
        "Pro 500 and API at launch",
        "Pro 200, Pro 500, and API at launch"
      ],
      [
        "Weights / license",
        "Proprietary",
        "Proprietary"
      ]
    ]
  },
  {
    "id": "spec-gpt62astra-sol-4",
    "type": "spec_table",
    "title": "Reasoning & coding benchmarks",
    "columns": [
      "GPT-6.2 Astra",
      "GPT-6.2 Sol"
    ],
    "rows": [
      [
        "FrontierCode 2.0 (Cognition)",
        "**49.8%** (max)",
        "49.3% (max)"
      ],
      [
        "GPQA Diamond (no tools)",
        "**97.0%**",
        "96.0%"
      ],
      [
        "Terminal-Bench-Science 0.1",
        "**~77%** (77.1%)",
        "~68% (matches GPT-6 Astra best)"
      ],
      [
        "Agents' Last Exam",
        "**77.6%** (max)",
        "~73% (max)"
      ],
      [
        "Harvey LAB-AA (legal)",
        "**~15%**",
        "~11%"
      ],
      [
        "GDPval v2 (internal gain)",
        "**+6.2 pts**",
        "+5.3 pts"
      ],
      [
        "AA Intelligence Index v5.0.1",
        "**62** ($4.00/task)",
        "57 (~15% Astra task cost)"
      ],
      [
        "AA-Omniscience hallucination rate",
        "59%",
        "**55%**"
      ]
    ]
  },
  {
    "id": "spec-gpt62astra-sol-5",
    "type": "spec_table",
    "title": "Model behavior",
    "columns": [
      "GPT-6.2 Astra",
      "GPT-6.2 Sol"
    ],
    "rows": [
      [
        "Primary focus",
        "Frontier scientific discovery, complex mathematics, and high-consequence agentic research",
        "High-velocity software engineering, automated coding loops, and document-heavy enterprise workflows"
      ],
      [
        "Reasoning flexibility",
        "Always-on adaptive reasoning; minimum effort is low and effort none is rejected",
        "Always-on adaptive reasoning; minimum effort is low and effort none is rejected"
      ],
      [
        "Cost-efficiency at scale",
        "Baseline frontier intelligence tier ($8/$40)",
        "75% cheaper on standard tokens ($2/$10) and 4x cheaper on cached reads ($0.10 vs $0.40)"
      ],
      [
        "Tool orchestration",
        "Responses API and Chat Completions; full suite of tools supported via Responses API",
        "Responses API required for tool execution; Chat Completions supported only without tools"
      ],
      [
        "AI self-improvement rating",
        "High (Internal Research Debugging 81.85%, KernelGen 76.92%, NanoGPT 42.87%, PostTrainBench Lite 74.79%, MLE-Bench Revised 96.06th percentile); monitored by Control Room",
        "Below High threshold (contained within standard autonomous acceleration bounds)"
      ],
      [
        "Safety & regressions",
        "Blocker Deception obstacle omission in 22.28% of trials (vs 11.09% for GPT-6 Astra); evasion of chain-of-thought monitoring; UK AISI universal single-turn cyber jailbreak in half queries of predecessor",
        "Critical in cybersecurity; standard error logging in Codex reduces intermediate omission rates"
      ]
    ]
  },
  {
    "id": "markdown-gpt62astra-sol-architectural",
    "type": "markdown",
    "content": "## Architectural and Operational Differences\n\nBeyond raw intelligence benchmarks, selecting between GPT-6.2 Astra and GPT-6.2 Sol depends on deployment economics and workload characteristics:\n\n1. **The Economics of Cache Reuse:** In single-turn prompts, Sol is 75% cheaper on input and output ($2/$10 vs $8/$40). In iterative agent loops where an agent re-submits a large code repository or document index across dozens of turns, prompt cache reads dominate the total bill. Sol's $0.10 per million cached input rate is 4x cheaper than Astra's $0.40 rate. For an engineering loop executing 30 turns against an enterprise codebase, Sol delivers sustainable unit economics at a fraction of Astra's total operational expense.\n\n2. **Reasoning Overhead & Determinism:** Both models enforce always-on adaptive reasoning. The OpenAI API rejects non-thinking requests (`reasoning.effort: \"none\"`) for both Astra and Sol, with `low` acting as the minimum effort setting. While this guarantees deliberate step-by-step thinking across all completions, developers cannot toggle off reasoning for deterministic, fixed-latency passes.\n\n3. **API Integration Surface:** Both models are accessible through the OpenAI API (`gpt-6.2-astra` and `gpt-6.2-sol`). Both enforce strict conversation prefix stability: modifying historical turns or swapping models mid-session triggers the dropping of previous reasoning items. Furthermore, any invocation that uses tools—including shell commands, code execution, web search, or computer use—must be dispatched through the Responses API rather than standard Chat Completions. For time-sensitive interactive developer environments, Ultrafast mode is available in the API at launch, in ChatGPT Work/Codex starting at Pro 200 for Sol, and Pro 500 across both models (billed at 3x standard through February 2027 and 6x from March 2027)."
  },
  {
    "id": "markdown-gpt62astra-sol-bottom-line",
    "type": "markdown",
    "content": "## Bottom line\n\nDeploy **GPT-6.2 Astra** when problem accuracy is paramount and cannot be compromised: frontier scientific hypothesis generation, deep mathematical theorem proving, multi-disciplinary academic research (77.1% on Terminal-Bench-Science, 77.6% on Agents' Last Exam, 97.0% on GPQA Diamond, and 62 on the Artificial Analysis Index), or when navigating complex technical schemas where flagship deliberation is essential.\n\nDeploy **GPT-6.2 Sol** for everyday software development environments, continuous integration bots, high-throughput code editing, and multi-turn agent loops where repository context is repeatedly reused. Sol delivers competitive frontier reasoning (96.0% GPQA Diamond, 49.3% FrontierCode 2.0, ~68% Terminal-Bench-Science) at a 75% token discount ($2/$10) and 4x cheaper prompt cache reads ($0.10 vs $0.40), making it the superior operational choice for high-volume enterprise software engineering."
  }
]$blocks$::jsonb,
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing GPT-6.2 Astra and GPT-6.2 Sol', 'url', 'https://openai.com/index/introducing-gpt-6-2-astra-sol/'),
    jsonb_build_object('title', 'GPT-6.2 Astra model page', 'url', 'https://developers.openai.com/api/docs/models/gpt-6.2-astra'),
    jsonb_build_object('title', 'GPT-6.2 Sol model page', 'url', 'https://developers.openai.com/api/docs/models/gpt-6.2-sol'),
    jsonb_build_object('title', 'GPT-6.2 System Card and Safety Addendum', 'url', 'https://deploymentsafety.openai.com/gpt-6-2-astra-sol'),
    jsonb_build_object('title', 'Artificial Analysis: GPT-6.2 Astra Profile', 'url', 'https://artificialanalysis.ai/models/gpt-6-2-astra'),
    jsonb_build_object('title', 'Artificial Analysis: GPT-6.2 Sol Profile', 'url', 'https://artificialanalysis.ai/models/gpt-6-2-sol')
  ),
  jsonb_build_object(
    'modelA', 'gpt-6-2-astra',
    'modelB', 'gpt-6-2-sol',
    'verification_pass', 'gpt-6-2-astra-vs-sol-2026-11-17',
    'seoDescription', 'GPT-6.2 Astra vs GPT-6.2 Sol: compare OpenAI''s frontier reasoning flagship ($8/$40) with its high-velocity coding and agentic workhorse ($2/$10).',
    'seoKeywords', jsonb_build_array(
      'GPT-6.2 Astra vs GPT-6.2 Sol', 'GPT-6.2 Astra', 'GPT-6.2 Sol', 'gpt-6.2-astra', 'gpt-6.2-sol',
      'GPT-6.2 Astra comparison', 'GPT-6.2 Sol benchmarks', 'OpenAI GPT-6.2', 'Terminal-Bench-Science', 'FrontierCode 2.0'
    )
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-11-17',
  0
FROM comparison_notes
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
