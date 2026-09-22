-- Add Claude Opus 5.5 to the model catalog, record the Opus 5.5 launch scorecard
-- as normalized benchmark evidence for both Opus 5.5 and Fable 5.1,
-- and publish the authored Fable 5.1 vs Opus 5.5 comparison page.
--
-- Source: the Anthropic introduction article for Claude Opus 5.5
-- (September 22, 2026) as the model card, which reports Opus 5.5 against
-- Fable 5.1, Opus 5, GPT-6 Astra, and GPT-5.6 Sol side by side.
-- Opus 5.5 headline results use adaptive thinking at max effort except
-- Terminal-Bench 4.0 at xhigh, and every row carries its qualifier.

-- 1. Model catalog entry for Claude Opus 5.5.
INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at
)
SELECT
  'claude-opus-5-5',
  'Claude Opus 5.5',
  'Anthropic',
  DATE '2026-09-22',
  DATE '2026-09-22',
  'proprietary',
  'Claude Opus 5',
  '{
    "Developer": "Anthropic",
    "Release date": "2026-09-22",
    "API model ID": "claude-opus-5-5",
    "Context window": "1M tokens",
    "Max output": "128K tokens (300K on Batch API beta)",
    "Knowledge cutoff": "Jun 2026",
    "Reasoning / effort": "Adaptive thinking (always on), medium default",
    "Input / 1M tokens": "$4",
    "Cached input / 1M": "$0.20",
    "Cache write / 1M": "$5 (5 min) / $8 (1 hr)",
    "Output / 1M tokens": "$20",
    "Batch / flex discount": "Batch API: 50% off input and output",
    "Long-context surcharge": "None - standard pricing through 1M context",
    "Text input": "Yes",
    "Image / vision input": "Yes",
    "Audio input": "No native audio input documented",
    "Video input": "No native video input documented",
    "Text output": "Yes",
    "Image output": "No",
    "Audio output": "No",
    "Video output": "No",
    "Tool / function calling": "Yes",
    "Computer use": "Yes",
    "API access": "Yes",
    "Product access": "Claude + Claude Code + Claude API + cloud partners",
    "Weights / license": "Proprietary"
  }'::jsonb,
  '[
    {"title": "Claude Opus 5.5", "url": "https://www.anthropic.com/claude-opus-5-5"},
    {"title": "Claude Opus 5.5 overview", "url": "https://platform.claude.com/docs/en/models/opus-5-5/overview"},
    {"title": "What is new in Claude Opus 5.5", "url": "https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5"}
  ]'::jsonb,
  'First release in the Claude 5.5 family. Same 1M context as Opus 5 with lower per-token prices, 60 percent cheaper cache reads, and fewer tokens per task for about 40 percent lower typical cost. Adaptive thinking is always on with a medium default, and thinking cannot be disabled.',
  '{
    "verification_pass": "opus-5-5-comparison-2026-09-22",
    "default_effort": "medium",
    "catalog_status": "current"
  }'::jsonb,
  NOW()
WHERE NOT EXISTS (SELECT 1 FROM models WHERE slug = 'claude-opus-5-5')
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

-- 2. Benchmark evidence from the Opus 5.5 launch scorecard (September 22, 2026).
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
  {"id":"tq-20260922-opus55-terminalbench40","model_slug":"claude-opus-5-5","category":"coding","benchmark_name":"Terminal-Bench","benchmark_version":"4.0","score_numeric":66.4,"score_display":"66.4%","score_unit":"percent","tools":null,"reasoning_effort":"xhigh","harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Self-reported launch result at xhigh effort. Fable 5.1 comparison value is 55.8 percent."},
  {"id":"tq-20260922-opus55-frontiercode11main","model_slug":"claude-opus-5-5","category":"coding","benchmark_name":"FrontierCode 1.1 Main","benchmark_version":null,"score_numeric":54.4,"score_display":"54.4%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Self-reported launch result at max effort. Fable 5.1 comparison value is 50.3 percent."},
  {"id":"tq-20260922-opus55-cursorbench40","model_slug":"claude-opus-5-5","category":"coding","benchmark_name":"CursorBench","benchmark_version":"4.0","score_numeric":57.8,"score_display":"57.8%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Self-reported launch result at max effort. Fable 5.1 comparison value is 51.8 percent."},
  {"id":"tq-20260922-opus55-gdpvalaa21","model_slug":"claude-opus-5-5","category":"agentic_computer_use","benchmark_name":"GDPval-AA","benchmark_version":"v2.1","score_numeric":1846.0,"score_display":"1846 Elo","score_unit":"Elo","tools":null,"reasoning_effort":"max","harness":"Artificial Analysis","evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Vendor-reported launch result at max effort. Fable 5.1 comparison value is 1735 Elo."},
  {"id":"tq-20260922-opus55-automationbench","model_slug":"claude-opus-5-5","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":null,"score_numeric":40.0,"score_display":"40.0%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"Zapier","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Run and reported by Zapier during early access with no fallback models, so safeguard interventions counted as failures."},
  {"id":"tq-20260922-opus55-hle-tools","model_slug":"claude-opus-5-5","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":67.7,"score_display":"67.7%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Self-reported launch result with tools at max effort. Fable 5.1 comparison value is 65.6 percent."},
  {"id":"tq-20260922-opus55-terminalbenchscience01","model_slug":"claude-opus-5-5","category":"coding","benchmark_name":"Terminal-Bench Science 0.1","benchmark_version":null,"score_numeric":58.7,"score_display":"58.7%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Self-reported launch result at max effort. Fable 5.1 comparison value is 52.6 percent."},
  {"id":"tq-20260922-opus55-osworld20-partial","model_slug":"claude-opus-5-5","category":"agentic_computer_use","benchmark_name":"OSWorld 2.0","benchmark_version":"partial","score_numeric":81.8,"score_display":"81.8%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Self-reported partial-credit result at max effort. Fable 5.1 comparison value is 80.7 percent."},
  {"id":"tq-20260922-opus55-chartography","model_slug":"claude-opus-5-5","category":"multimodal","benchmark_name":"Chartography","benchmark_version":null,"score_numeric":89.0,"score_display":"89.0%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Self-reported visual chart result with tools at max effort. Fable 5.1 comparison value is 88.4 percent."},
  {"id":"tq-20260922-fable51-terminalbench40","model_slug":"claude-fable-5-1","category":"coding","benchmark_name":"Terminal-Bench","benchmark_version":"4.0","score_numeric":55.8,"score_display":"55.8%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Fable 5.1 comparison value in the Opus 5.5 launch table."},
  {"id":"tq-20260922-fable51-frontiercode11main","model_slug":"claude-fable-5-1","category":"coding","benchmark_name":"FrontierCode 1.1 Main","benchmark_version":null,"score_numeric":50.3,"score_display":"50.3%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Fable 5.1 comparison value in the Opus 5.5 launch table."},
  {"id":"tq-20260922-fable51-cursorbench40","model_slug":"claude-fable-5-1","category":"coding","benchmark_name":"CursorBench","benchmark_version":"4.0","score_numeric":51.8,"score_display":"51.8%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Fable 5.1 comparison value in the Opus 5.5 launch table."},
  {"id":"tq-20260922-fable51-gdpvalaa21","model_slug":"claude-fable-5-1","category":"agentic_computer_use","benchmark_name":"GDPval-AA","benchmark_version":"v2.1","score_numeric":1735.0,"score_display":"1735 Elo","score_unit":"Elo","tools":null,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Fable 5.1 comparison value in the Opus 5.5 launch table."},
  {"id":"tq-20260922-fable51-automationbench","model_slug":"claude-fable-5-1","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":null,"score_numeric":31.4,"score_display":"31.4%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Zapier","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Fable 5.1 comparison value in the Opus 5.5 launch table."},
  {"id":"tq-20260922-fable51-hle-tools","model_slug":"claude-fable-5-1","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":65.6,"score_display":"65.6%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Fable 5.1 comparison value in the Opus 5.5 launch table, with tools."},
  {"id":"tq-20260922-fable51-terminalbenchscience01","model_slug":"claude-fable-5-1","category":"coding","benchmark_name":"Terminal-Bench Science 0.1","benchmark_version":null,"score_numeric":52.6,"score_display":"52.6%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Fable 5.1 comparison value in the Opus 5.5 launch table."},
  {"id":"tq-20260922-fable51-osworld20-partial","model_slug":"claude-fable-5-1","category":"agentic_computer_use","benchmark_name":"OSWorld 2.0","benchmark_version":"partial","score_numeric":80.7,"score_display":"80.7%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Fable 5.1 partial-credit comparison value in the Opus 5.5 launch table."},
  {"id":"tq-20260922-fable51-chartography","model_slug":"claude-fable-5-1","category":"multimodal","benchmark_name":"Chartography","benchmark_version":null,"score_numeric":88.4,"score_display":"88.4%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-22","source":"https://www.anthropic.com/claude-opus-5-5","notes":"Fable 5.1 comparison value in the Opus 5.5 launch table, with tools."}
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

-- 3. Authored comparison page: Claude Fable 5.1 vs Claude Opus 5.5.
WITH comparison_notes AS (
  SELECT $notes$
<small>*Scores are vendor-reported from the Opus 5.5 launch table (September 22, 2026). Opus 5.5 headline results use adaptive thinking at max effort except Terminal-Bench 4.0 at xhigh, while Fable 5.1 rows carry no stated effort, so same-column deltas mix effort levels. AutomationBench was run by Zapier with no fallback models, and Opus 5.5 ran with production safeguards enabled with transparent fallback to Opus 4.8 and Opus 5 where they intervened.*</small>

Fable 5.1 is the September 2026 premium model for difficult coding, long-running agent work, research, computer use, and professional knowledge tasks. Opus 5.5 is the first release in the new Claude 5.5 family for the same long-horizon work, positioned at Fable 5.1 level on most tasks while costing substantially less to run. Neither release publishes parameter counts or architecture details, so the comparison is best judged on behavior, benchmarks, and bills.

List price is where the two diverge most. Fable 5.1 bills USD 10 input and USD 50 output per million tokens with USD 0.25 cache reads, while Opus 5.5 bills USD 4 input and USD 20 output with USD 0.20 cache reads. Same 1M context window, same 128K max output, same June 2026 knowledge cutoff, and the same always-on adaptive thinking with per-message effort control on both models.

The launch table shows Opus 5.5 ahead on every shared row, with the clearest gaps on Terminal-Bench 4.0 (66.4 vs 55.8), CursorBench 4.0 (57.8 vs 51.8), and GDPval-AA v2.1 (1846 vs 1735 Elo). Anthropic itself cautions that the real-world gap feels narrower than these margins suggest, and Astra still leads both models on Terminal-Bench-Science 0.1 and AutomationBench in the same table.

Equal capability claims do not mean equal bills. Opus 5.5 costs 40 percent less than Opus 5 on typical workloads at default settings, and at default medium effort it beats or matches flagship rivals for a fifth to two-fifths of the cost per task. Pick Fable 5.1 when its premium tier buys a measurable win on your own workload, and pick Opus 5.5 when Fable-level quality at Opus prices is the requirement.
$notes$::text AS notes
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'comparison:fable-5-1-vs-opus-5-5',
  'comparison',
  'fable-5-1-vs-opus-5-5',
  '',
  'comparisons/fable-5-1-vs-opus-5-5',
  'Claude Fable 5.1 vs Claude Opus 5.5',
  'Claude Opus 5.5 vs Fable 5.1: Sep 22, 2026 launch with the same 1M context at less than half the price - benchmark gains, effort defaults, and safeguards.',
  comparison_notes.notes,
  '[
    {"id": "spec-fable51-opus55-1", "type": "spec_table", "title": "Specifications",
     "columns": ["Claude Fable 5.1", "Claude Opus 5.5"],
     "rows": [
       ["Developer", "Anthropic", "Anthropic"],
       ["Release date", "2026-09-01", "2026-09-22"],
       ["API model ID", "claude-fable-5-1", "claude-opus-5-5"],
       ["Context window", "1M tokens", "1M tokens"],
       ["Max output", "128K tokens", "128K tokens (300K on Batch API beta)"],
       ["Knowledge cutoff", "Jun 2026", "Jun 2026"],
       ["Reasoning / effort", "Adaptive thinking (always on), default high, per-message effort", "Adaptive thinking (always on), medium default, per-message effort"]
     ]},
    {"id": "spec-fable51-opus55-2", "type": "spec_table", "title": "Pricing",
     "columns": ["Claude Fable 5.1", "Claude Opus 5.5"],
     "rows": [
       ["Input / 1M tokens", "$10", "$4"],
       ["Cached input / 1M", "$0.25", "$0.20"],
       ["Cache write / 1M", "$12.50 (5m) / $20 (1h)", "$5 (5m) / $8 (1h)"],
       ["Output / 1M tokens", "$50", "$20"],
       ["Batch / flex discount", "Batch API: 50% off input and output", "Batch API: 50% off input and output"],
       ["Long-context surcharge", "None - standard pricing through 1M context", "None - standard pricing through 1M context"]
     ]},
    {"id": "spec-fable51-opus55-3", "type": "spec_table", "title": "Capabilities & access",
     "columns": ["Claude Fable 5.1", "Claude Opus 5.5"],
     "rows": [
       ["Text input", "Yes", "Yes"],
       ["Image / vision input", "Yes", "Yes"],
       ["Audio input", "No native audio input documented", "No native audio input documented"],
       ["Video input", "No native video input documented", "No native video input documented"],
       ["Text output", "Yes", "Yes"],
       ["Image output", "No", "No"],
       ["Audio output", "No", "No"],
       ["Video output", "No", "No"],
       ["Tool / function calling", "Yes", "Yes"],
       ["Computer use", "Yes", "Yes"],
       ["API access", "Yes", "Yes"],
       ["Product access", "Claude + Claude Code + Claude API + cloud partners", "Claude + Claude Code + Claude API + cloud partners"],
       ["Weights / license", "Proprietary", "Proprietary"]
     ]},
    {"id": "spec-fable51-opus55-4", "type": "spec_table", "title": "Model behavior",
     "columns": ["Claude Fable 5.1", "Claude Opus 5.5"],
     "rows": [
       ["Primary focus", "Premium model for difficult coding, long agents, research, and knowledge work", "Long-running agentic coding and knowledge work at Fable-level quality"],
       ["Long-horizon work", "Multi-file changes and debugging sessions that run for hours", "Codebase-wide migrations and audits, including overnight unattended runs"],
       ["Agent orchestration", "Claude Code agent loops with per-message effort control", "Same harness with clearer delegation and self-verification loops"],
       ["User collaboration", "Denser prose in some long runs", "Most-important-first writing with less jargon and better style-rule following"],
       ["Efficiency / generation change", "Baseline premium verbosity", "40% lower typical cost than Opus 5 with 30%+ faster output and fewer tokens per task"],
       ["Safety / approvals", "Cyber, biology, and distillation safeguards with transparent fallback", "Same safeguard class as Fable 5.1 with preserved thinking and verification programs"]
     ]},
    {"id": "spec-fable51-opus55-5", "type": "spec_table", "title": "Coding",
     "columns": ["Claude Fable 5.1", "Claude Opus 5.5"],
     "rows": [
       ["Terminal-Bench 4.0", "55.8%", "**66.4%** (xhigh)"],
       ["FrontierCode 1.1 Main", "50.3%", "**54.4%**"],
       ["CursorBench 4.0", "51.8%", "**57.8%**"],
       ["Terminal-Bench Science 0.1", "52.6%", "**58.7%**"]
     ]},
    {"id": "spec-fable51-opus55-6", "type": "spec_table", "title": "Knowledge",
     "columns": ["Claude Fable 5.1", "Claude Opus 5.5"],
     "rows": [
       ["Humanity''s Last Exam", "65.6% with tools", "**67.7%** with tools"]
     ]},
    {"id": "spec-fable51-opus55-7", "type": "spec_table", "title": "Agentic & computer use",
     "columns": ["Claude Fable 5.1", "Claude Opus 5.5"],
     "rows": [
       ["GDPval-AA v2.1", "1735 Elo", "**1846 Elo**"],
       ["AutomationBench", "31.4%", "**40.0%**"],
       ["OSWorld 2.0", "80.7% partial", "**81.8%** partial"]
     ]},
    {"id": "spec-fable51-opus55-8", "type": "spec_table", "title": "Multimodal",
     "columns": ["Claude Fable 5.1", "Claude Opus 5.5"],
     "rows": [
       ["Chartography", "88.4% with tools", "**89.0%** with tools"]
     ]},
    {"id": "markdown-fable51-opus55-bottom-line", "type": "markdown",
     "content": "## Bottom line\n\nOpus 5.5 leads the shared launch-table rows while billing less than half the Fable 5.1 rate: same 1M context, same cutoff, same always-on thinking, with a cheaper default effort and clearer writing. The vendor deltas carry cross-effort caveats and Anthropic says the lived gap is narrower than the margins. Default to Opus 5.5 for long agent work and escalate to Fable 5.1 only where your own evals show the premium tier earning its price."}
  ]'::jsonb,
  jsonb_build_array(
    jsonb_build_object('title', 'Claude Opus 5.5', 'url', 'https://www.anthropic.com/claude-opus-5-5'),
    jsonb_build_object('title', 'Claude Opus 5.5 overview', 'url', 'https://platform.claude.com/docs/en/models/opus-5-5/overview'),
    jsonb_build_object('title', 'What is new in Claude Opus 5.5', 'url', 'https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5'),
    jsonb_build_object('title', 'Claude Opus 5.5 system card', 'url', 'https://www.anthropic.com/claude-opus-5-5-system-card')
  ),
  jsonb_build_object(
    'modelA', 'claude-fable-5-1',
    'modelB', 'claude-opus-5-5',
    'verification_pass', 'opus-5-5-comparison-2026-09-22'
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-09-22',
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
