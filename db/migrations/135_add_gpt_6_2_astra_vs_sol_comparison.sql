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
  SELECT $notes$
<small>*Vendor figures are from OpenAI's November 17, 2026 launch announcements, developer documentation, and system card addenda. Both models share a 1,050,000-token context window and 128,000 maximum output tokens with an August 2026 knowledge cutoff. Pricing and reasoning configurations differ substantially across the pair.*</small>

OpenAI released GPT-6.2 Astra and GPT-6.2 Sol simultaneously on November 17, 2026, structuring its frontier lineup into two distinct operational tiers. GPT-6.2 Astra is the flagship scientific and reasoning model, priced at $5 per million input tokens and $20 per million output tokens ($1.25 per million cached input tokens), engineered for long-horizon academic research, advanced mathematical verification, and autonomous scientific discovery. GPT-6.2 Sol is the high-velocity developer and automation workhorse, priced at $2 per million input tokens and $10 per million output tokens ($0.10 per million cached input tokens), designed for rapid code refactoring, terminal interaction, and cost-effective multi-turn agent loops.

The capability separation reflects their design targets. On scientific and complex multi-step reasoning evaluations, Astra establishes a commanding lead: on Terminal-Bench-Science 0.1, Astra scores 77.1% against Sol's 61.5%, a 15.6-point advantage that underscores Astra's capacity for deep domain problem solving. On Agents' Last Exam, Astra reaches 77.6% under maximum reasoning effort, whereas Sol does not report an equivalent score. On GPQA Diamond, both models perform at the frontier with Astra slightly ahead at 97.0% compared to Sol's 96.0%. Sol, conversely, focuses its compute on software engineering execution, recording 49.3% on FrontierCode 2.0.

Economics and prompt caching form the primary decision boundary for production architectures. While Sol is 60% cheaper on uncached input ($2 versus $5) and 50% cheaper on output ($10 versus $20), its prompt cache read rate is where the gap becomes transformative: $0.10 per million tokens on Sol versus $1.25 on Astra, representing a 12.5x cost advantage for cached context. For autonomous software engineering agents that repeatedly re-read full repository contexts across 20 to 50 iterations, Sol operates at a small fraction of Astra's total operational expense.

Operational controls also diverge. Astra preserves the complete reasoning effort spectrum from `none` to `max`. Setting effort to `none` allows teams to invoke Astra deterministically with zero reasoning token overhead for latency-critical or predictable classification workflows. In contrast, Sol enforces always-on reasoning with a minimum effort setting of `low` (`none` and `minimal` are unsupported), requiring reasoning tokens on every completion. Furthermore, Sol strictly requires the Responses API for executing tools, computer use, and image generation, with Chat Completions supported solely for tool-free turns.
$notes$::text AS notes
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
  'GPT-6.2 Astra vs GPT-6.2 Sol: comparing OpenAI''s frontier scientific reasoning flagship ($5/$20) with its high-throughput agentic workhorse ($2/$10).',
  comparison_notes.notes,
  $blocks$
  [
    {"id": "spec-gpt62astra-sol-1", "type": "spec_table", "title": "Specifications",
     "columns": ["GPT-6.2 Astra", "GPT-6.2 Sol"],
     "rows": [
       ["Developer", "OpenAI", "OpenAI"],
       ["Release date", "2026-11-17", "2026-11-17"],
       ["API model ID", "gpt-6.2-astra", "gpt-6.2-sol"],
       ["Context window", "1,050,000 tokens", "1,050,000 tokens"],
       ["Max output", "128,000 tokens", "128,000 tokens"],
       ["Knowledge cutoff", "Aug 2026", "Aug 2026"],
       ["Reasoning / effort", "none / low / medium (default) / high / xhigh / max", "Always on: low / medium (default) / high / xhigh / max. none and minimal are not supported"],
       ["Fine-tuning", "Not supported", "Not supported"],
       ["Data residency", "US and EU", "US and EU"],
       ["Weights / license", "Proprietary", "Proprietary"]
     ]},
    {"id": "spec-gpt62astra-sol-2", "type": "spec_table", "title": "Pricing",
     "columns": ["GPT-6.2 Astra", "GPT-6.2 Sol"],
     "rows": [
       ["Input / 1M tokens", "$5", "**$2**"],
       ["Cached input / 1M", "$1.25", "**$0.10**"],
       ["Cache write / 1M", "$6.25", "$2.50"],
       ["Output / 1M tokens", "$20", "**$10**"],
       ["Batch / flex discount", "50 percent off standard", "50 percent off standard"],
       ["Long-context surcharge", "Over 272K input tokens: 2x input and cache rates plus 1.5x output for the full request", "Over 272K input tokens: 2x input and cache rates plus 1.5x output for the full request"],
       ["Free tier", "Not supported", "Not supported"]
     ]},
    {"id": "spec-gpt62astra-sol-3", "type": "spec_table", "title": "Capabilities & access",
     "columns": ["GPT-6.2 Astra", "GPT-6.2 Sol"],
     "rows": [
       ["Text input", "Yes", "Yes"],
       ["Image / vision input", "Yes", "Yes"],
       ["Audio input", "No", "No"],
       ["Video input", "No", "No"],
       ["Text output", "Yes", "Yes"],
       ["Image output", "Yes, through the Responses API image generation tool", "Yes, through the Responses API image generation tool"],
       ["Audio output", "No", "No"],
       ["Video output", "No", "No"],
       ["Tool / function calling", "Yes: streaming, function calling, structured outputs, computer use", "Yes: streaming, function calling, structured outputs, computer use. Tools require Responses API"],
       ["Computer use", "Yes, through the Responses API", "Yes, through the Responses API"],
       ["API access", "OpenAI API; Microsoft Azure", "OpenAI API; Microsoft Azure"],
       ["Product access", "ChatGPT Pro, Team, Enterprise, Edu", "ChatGPT Work and Codex (Plus, Pro, Business, Enterprise, Edu)"],
       ["Weights / license", "Proprietary", "Proprietary"]
     ]},
    {"id": "spec-gpt62astra-sol-4", "type": "spec_table", "title": "Reasoning & coding benchmarks",
     "columns": ["GPT-6.2 Astra", "GPT-6.2 Sol"],
     "rows": [
       ["Terminal-Bench-Science 0.1", "**77.1%** (verified)", "61.5% (max reasoning)"],
       ["GPQA Diamond", "**97.0%** (max reasoning)", "96.0% (max reasoning)"],
       ["Agents' Last Exam", "**77.6%** (max reasoning)", "Not reported"],
       ["FrontierCode 2.0", "Not reported", "**49.3%**"]
     ]},
    {"id": "spec-gpt62astra-sol-5", "type": "spec_table", "title": "Model behavior",
     "columns": ["GPT-6.2 Astra", "GPT-6.2 Sol"],
     "rows": [
       ["Primary focus", "Frontier scientific reasoning, autonomous long-horizon research, advanced mathematics, and complex multi-step reasoning", "High-throughput software engineering, autonomous coding agents, document-heavy workflows, and high-frequency multi-turn agent loops"],
       ["Reasoning flexibility", "Full continuum from none to max; zero-reasoning execution supported for deterministic tasks", "Always-on reasoning; low is minimum effort. Reasoning tokens are billed on every call"],
       ["Cost-efficiency at scale", "Premium tier suited for high-stakes analysis where accuracy supersedes token spend", "High-efficiency tier; 12.5x cheaper cached input enables cost-effective multi-turn agent loops"],
       ["Tool orchestration", "Responses API and Chat Completions; full suite of tools supported", "Responses API required for tool execution; Chat Completions supported only without tools"]
     ]},
    {"id": "markdown-gpt62astra-sol-architectural", "type": "markdown",
     "content": "## Architectural and Operational Differences\n\nBeyond raw intelligence metrics, the choice between GPT-6.2 Astra and GPT-6.2 Sol hinges on economic architecture and deployment constraints:\n\n1. **The Economics of Cache Reuse:** In single-turn requests, Sol is 2.5x cheaper on input and 2x cheaper on output. However, in agentic environments where an agent repeatedly scans a large codebase or document collection, prompt caching dominates the total bill. Sol's $0.10 per million cached input rate is 12.5x cheaper than Astra's $1.25 rate. Over a 30-turn agent session with 100K cached tokens per turn, Sol incurs negligible cache read costs while Astra accrues substantial per-turn overhead.\n\n2. **Reasoning Overhead & Determinism:** Astra allows developers to set `reasoning_effort: none`, which bypasses the internal thinking phase entirely. This is crucial for workflows that require deterministic responses, minimal time-to-first-token (TTFT), or fixed per-token cost profiles. Sol removes `none` and `minimal`, enforcing `low` as the minimum reasoning setting. While this guarantees reasoning depth on every prompt, it introduces reasoning latency and token expenditure that cannot be toggled off.\n\n3. **API Integration Surface:** While Astra supports tool execution across both standard Chat Completions and the newer Responses API, Sol requires the Responses API for all function calling, computer use, and web tools. Callers looking to drop Sol into legacy Chat Completions endpoints with tool schemas must migrate their client implementations to Responses."},
    {"id": "markdown-gpt62astra-sol-bottom-line", "type": "markdown",
     "content": "## Bottom line\n\nDeploy **GPT-6.2 Astra** when problem accuracy is paramount and cannot be compromised: frontier scientific hypothesis generation, deep mathematical theorem proving, multi-disciplinary academic research (77.1% on Terminal-Bench-Science and 77.6% on Agents' Last Exam), or when deterministic zero-reasoning completions are strictly required.\n\nDeploy **GPT-6.2 Sol** for developer automation, continuous integration bots, high-throughput code editing, and multi-turn agent loops where repository context is repeatedly reused. Sol delivers competitive frontier reasoning (96.0% GPQA Diamond, 49.3% FrontierCode 2.0) at a fraction of the cost, making it the superior operational choice for high-volume enterprise software engineering."}
  ]$blocks$::jsonb,
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing GPT-6.2 Astra', 'url', 'https://openai.com/index/introducing-gpt-6-2-astra/'),
    jsonb_build_object('title', 'Introducing GPT-6.2 Sol', 'url', 'https://openai.com/index/introducing-gpt-6-2-sol/'),
    jsonb_build_object('title', 'GPT-6.2 Astra model page', 'url', 'https://developers.openai.com/api/docs/models/gpt-6.2-astra'),
    jsonb_build_object('title', 'GPT-6.2 Sol model page', 'url', 'https://developers.openai.com/api/docs/models/gpt-6.2-sol')
  ),
  jsonb_build_object(
    'modelA', 'gpt-6-2-astra',
    'modelB', 'gpt-6-2-sol',
    'verification_pass', 'gpt-6-2-astra-vs-sol-2026-11-17',
    'seoDescription', 'GPT-6.2 Astra vs GPT-6.2 Sol: compare OpenAI''s frontier reasoning flagship ($5/$20) with its high-velocity coding and agentic workhorse ($2/$10).',
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
