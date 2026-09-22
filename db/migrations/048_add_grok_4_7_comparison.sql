-- Add Grok 4.7 to the model catalog, record the Grok 4.7 launch scorecard
-- as normalized benchmark evidence for both Grok 4.7 and Grok 4.6,
-- and publish the authored Grok 4.6 vs Grok 4.7 comparison page.
--
-- Source: xAI's Grok 4.7 launch table (September 21, 2026), which reports
-- both models side by side, plus the independent Artificial Analysis
-- readings cited in the Grok 4.7 glossary entry. Effort and harness differ
-- across rows, so every row carries its qualifier in the notes.

-- 1. Model catalog entry for Grok 4.7.
INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at
)
SELECT
  'grok-4-7',
  'Grok 4.7',
  'xAI',
  DATE '2026-09-21',
  DATE '2026-09-21',
  'proprietary',
  'Grok 4',
  '{
    "Developer": "xAI",
    "Release date": "2026-09-21",
    "API model ID": "grok-4.7",
    "Context window": "500K tokens",
    "Max output": "No fixed output limit documented",
    "Knowledge cutoff": "Jun 2026 pretraining, supplemental training through Aug 2026",
    "Reasoning / effort": "low / medium / high / xhigh, high default",
    "Input / 1M tokens": "$2 <=200K / $4 >200K",
    "Cached input / 1M": "$0.50 <=200K / $1 >200K",
    "Output / 1M tokens": "$6 <=200K / $12 >200K",
    "Long-context surcharge": ">=200K prompt tokens: every token bills at the higher tier",
    "Text input": "Yes",
    "Image / vision input": "Yes",
    "Audio input": "No native audio input documented",
    "Video input": "No native video input documented",
    "Text output": "Yes",
    "Image output": "No",
    "Audio output": "No",
    "Video output": "No",
    "Tool / function calling": "Yes, function calling, web search, X search, code execution",
    "Computer use": "Via Grok Build and agent environments, with native Grok Bot harness understanding",
    "API access": "Yes",
    "Product access": "Grok Build + Cursor + xAI API + partner gateways",
    "Weights / license": "Proprietary"
  }'::jsonb,
  '[
    {"title": "Introducing Grok 4.7", "url": "https://x.ai/news/grok-4-7"},
    {"title": "Grok 4.7 model documentation", "url": "https://docs.x.ai/developers/models/grok-4-7"},
    {"title": "xAI pricing", "url": "https://docs.x.ai/developers/pricing"}
  ]'::jsonb,
  'New, larger base model followed by a longer reinforcement-learning run over harder multi-hour tasks, with stronger self-verification and native Grok Bot harness understanding. Same 500K context window and same $2/$6 list price as Grok 4.6. Fast serving variant doubles output speed at twice the price on Cursor and Grok Build.',
  '{
    "verification_pass": "grok-4-7-comparison-2026-09-22",
    "default_effort": "high",
    "catalog_status": "current"
  }'::jsonb,
  NOW()
WHERE NOT EXISTS (SELECT 1 FROM models WHERE slug = 'grok-4-7')
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

-- 2. Benchmark evidence from the Grok 4.7 launch scorecard (September 21, 2026).
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
  {"id":"tq-20260921-grok47-cursorbench40","model_slug":"grok-4-7","category":"coding","benchmark_name":"CursorBench","benchmark_version":"4.0","score_numeric":46.3,"score_display":"46.3%","score_unit":"percent","tools":null,"reasoning_effort":"xhigh","harness":null,"evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Self-reported launch result at xhigh. Grok 4.6 comparison value is 40.4% at high."},
  {"id":"tq-20260921-grok47-deepswe11","model_slug":"grok-4-7","category":"coding","benchmark_name":"DeepSWE","benchmark_version":"1.1","score_numeric":71.0,"score_display":"71.0%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Self-reported launch result at high. Grok 4.6 comparison value is 65.2%."},
  {"id":"tq-20260921-grok47-terminalbench40","model_slug":"grok-4-7","category":"coding","benchmark_name":"Terminal-Bench","benchmark_version":"4.0","score_numeric":38.0,"score_display":"38.0%","score_unit":"percent","tools":null,"reasoning_effort":"xhigh","harness":"Grok Build","evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Self-reported launch result on the Grok Build harness. Independent mini-swe-agent reading is 26%."},
  {"id":"tq-20260921-grok47-terminalbench40-aa","model_slug":"grok-4-7","category":"coding","benchmark_name":"Terminal-Bench","benchmark_version":"4.0","score_numeric":26.0,"score_display":"26%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":"mini-swe-agent","evaluator":"Artificial Analysis","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Independent harness reading cited in launch coverage, versus 38% self-reported on Grok Build."},
  {"id":"tq-20260921-grok47-eebench","model_slug":"grok-4-7","category":"coding","benchmark_name":"EEBench","benchmark_version":null,"score_numeric":66.0,"score_display":"66.0%","score_unit":"percent","tools":null,"reasoning_effort":"xhigh","harness":null,"evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Self-reported electrical-engineering agent result at xhigh. Grok 4.6 comparison value is 60.0%."},
  {"id":"tq-20260921-grok47-aabriefcase11","model_slug":"grok-4-7","category":"agentic_computer_use","benchmark_name":"AA-Briefcase","benchmark_version":"v1.1","score_numeric":1657.0,"score_display":"1657 Elo","score_unit":"Elo","tools":null,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Artificial Analysis evaluation reported by xAI. Grok 4.6 comparison value is 1546 Elo."},
  {"id":"tq-20260921-grok47-harvey","model_slug":"grok-4-7","category":"professional","benchmark_name":"Harvey Legal Agent","benchmark_version":null,"score_numeric":19.6,"score_display":"19.6%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Self-reported legal agent result. Grok 4.6 comparison value is 15.8%."},
  {"id":"tq-20260921-grok47-healthbench","model_slug":"grok-4-7","category":"professional","benchmark_name":"HealthBench","benchmark_version":"Professional","score_numeric":56.7,"score_display":"56.7%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Self-reported launch result. Grok 4.6 comparison value is 48.5%."},
  {"id":"tq-20260921-grok47-swemarathon11","model_slug":"grok-4-7","category":"coding","benchmark_name":"SWE Marathon","benchmark_version":"v1.1","score_numeric":46.0,"score_display":"46.0%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Long-agent-streak evaluation at high effort. Grok 4.6 comparison value is 31.9% at high."},
  {"id":"tq-20260921-grok47-coding-agent-index","model_slug":"grok-4-7","category":"coding","benchmark_name":"Coding Agent Index","benchmark_version":null,"score_numeric":56.0,"score_display":"56","score_unit":"points","tools":null,"reasoning_effort":null,"harness":"Grok Build","evaluator":"Artificial Analysis","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Independent Grok Build pairing score, nine points above Grok 4.6."},
  {"id":"tq-20260921-grok46-cursorbench40","model_slug":"grok-4-6","category":"coding","benchmark_name":"CursorBench","benchmark_version":"4.0","score_numeric":40.4,"score_display":"40.4%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Grok 4.6 comparison value in the Grok 4.7 launch table, at high effort."},
  {"id":"tq-20260921-grok46-deepswe11","model_slug":"grok-4-6","category":"coding","benchmark_name":"DeepSWE","benchmark_version":"1.1","score_numeric":65.2,"score_display":"65.2%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Grok 4.6 comparison value in the Grok 4.7 launch table. Retained alongside the earlier 65.9% high-effort launch value."},
  {"id":"tq-20260921-grok46-terminalbench40","model_slug":"grok-4-6","category":"coding","benchmark_name":"Terminal-Bench","benchmark_version":"4.0","score_numeric":20.3,"score_display":"20.3%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Grok 4.6 comparison value in the Grok 4.7 launch table."},
  {"id":"tq-20260921-grok46-eebench","model_slug":"grok-4-6","category":"coding","benchmark_name":"EEBench","benchmark_version":null,"score_numeric":60.0,"score_display":"60.0%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Grok 4.6 comparison value in the Grok 4.7 launch table."},
  {"id":"tq-20260921-grok46-aabriefcase11","model_slug":"grok-4-6","category":"agentic_computer_use","benchmark_name":"AA-Briefcase","benchmark_version":"v1.1","score_numeric":1546.0,"score_display":"1546 Elo","score_unit":"Elo","tools":null,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Grok 4.6 comparison value in the Grok 4.7 launch table."},
  {"id":"tq-20260921-grok46-harvey","model_slug":"grok-4-6","category":"professional","benchmark_name":"Harvey Legal Agent","benchmark_version":null,"score_numeric":15.8,"score_display":"15.8%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Grok 4.6 comparison value in the Grok 4.7 launch table."},
  {"id":"tq-20260921-grok46-healthbench","model_slug":"grok-4-6","category":"professional","benchmark_name":"HealthBench","benchmark_version":"Professional","score_numeric":48.5,"score_display":"48.5%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Grok 4.6 comparison value in the Grok 4.7 launch table."},
  {"id":"tq-20260921-grok46-swemarathon11","model_slug":"grok-4-6","category":"coding","benchmark_name":"SWE Marathon","benchmark_version":"v1.1","score_numeric":31.9,"score_display":"31.9%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Grok 4.6 comparison value in the Grok 4.7 launch table, at high effort."},
  {"id":"tq-20260921-grok46-coding-agent-index","model_slug":"grok-4-6","category":"coding","benchmark_name":"Coding Agent Index","benchmark_version":null,"score_numeric":47.0,"score_display":"47","score_unit":"points","tools":null,"reasoning_effort":null,"harness":"Grok Build","evaluator":"xAI","evaluation_date":"2026-09-21","source":"https://x.ai/news/grok-4-7","notes":"Derived baseline: xAI reports 56 for the Grok 4.7 pairing, nine points above Grok 4.6."}
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

-- 3. Authored comparison page: Grok 4.6 vs Grok 4.7.
WITH comparison_notes AS (
  SELECT $notes$
<small>*Scores are vendor-reported from the Grok 4.7 launch table (September 21, 2026). Reasoning effort differs across rows: Grok 4.7 is shown at xhigh wherever tagged, Grok 4.6 at high or untagged, so same-column deltas mix effort levels. Terminal-Bench 4.0 uses the Grok Build harness, where the independent mini-swe-agent reading for Grok 4.7 is 12 points lower.*</small>

Grok 4.6 was a post-training upgrade of Grok 4.5: a longer supplemental training run, regenerated fine-tuning trajectories, and new reinforcement learning on the same weights. Grok 4.7 is a new, larger base model followed by a longer reinforcement-learning run over harder tasks that take many hours to complete, with stronger self-verification and native understanding of the Grok Bot harness. No parameter count or architecture details were published for either release, so the upgrade is best judged on behavior and benchmarks.

List price and context are unchanged: $2 input and $6 output per million tokens under 200K prompt tokens, doubling at or above 200K, with the same 500K context window and the same low, medium, high, and xhigh effort levels. A fast serving variant doubles output speed at twice the price on Cursor and Grok Build for both models.

The launch table shows steady progress rather than a sweep. Grok 4.7 improves every shared row, with the largest gains on Terminal-Bench 4.0 (38.0 vs 20.3) and SWE-Marathon v1.1 (46.0 vs 31.9 at high effort), and leads on EEBench and the Harvey legal agent benchmark. The same table has Grok 4.7 trailing Fable 5.1 on CursorBench 4.0, Terminal-Bench 4.0, and HealthBench Professional, so the 4.6-to-4.7 delta is a same-vendor step forward, not a frontier takeover.

Equal per-token prices do not mean equal bills. Independent testing measured roughly 81,000 output tokens per task for Grok 4.7 against 38,000 for Grok 4.6, about 2.5x the per-task API cost at twice the wall-clock time for the nine-point Coding Agent Index gain. Upgrade for longer horizons and better self-checking, and evaluate on elapsed job time and total tokens per completed task rather than the $2/$6 sticker.
$notes$::text AS notes
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'comparison:grok-4-6-vs-grok-4-7',
  'comparison',
  'grok-4-6-vs-grok-4-7',
  '',
  'comparisons/grok-4-6-vs-grok-4-7',
  'Grok 4.6 vs Grok 4.7',
  'Grok 4.7 vs 4.6: same $2/$6 price and 500K context — what the Sept 21, 2026 larger-base-model upgrade gains on benchmarks and long agent runs.',
  comparison_notes.notes,
  '[
    {"id": "spec-grok46-47-1", "type": "spec_table", "title": "Specifications",
     "columns": ["Grok 4.6", "Grok 4.7"],
     "rows": [
       ["Developer", "xAI", "xAI"],
       ["Release date", "2026-08-12", "2026-09-21"],
       ["API model ID", "grok-4.6", "grok-4.7"],
       ["Context window", "500K tokens", "500K tokens"],
       ["Max output", "No fixed output limit documented", "No fixed output limit documented"],
       ["Knowledge cutoff", "Feb 1, 2026", "Jun 2026, plus supplemental training through Aug 2026"],
       ["Reasoning / effort", "low / medium / high / xhigh, high default", "low / medium / high / xhigh, high default"]
     ]},
    {"id": "spec-grok46-47-2", "type": "spec_table", "title": "Pricing",
     "columns": ["Grok 4.6", "Grok 4.7"],
     "rows": [
       ["Input / 1M tokens", "$2 <=200K / $4 >200K", "$2 <=200K / $4 >200K"],
       ["Cached input / 1M", "$0.50 <=200K / $1 >200K", "$0.50 <=200K / $1 >200K"],
       ["Output / 1M tokens", "$6 <=200K / $12 >200K", "$6 <=200K / $12 >200K"],
       ["Long-context surcharge", ">=200K prompt tokens: every token bills at the higher tier", ">=200K prompt tokens: every token bills at the higher tier"]
     ]},
    {"id": "spec-grok46-47-3", "type": "spec_table", "title": "Capabilities & access",
     "columns": ["Grok 4.6", "Grok 4.7"],
     "rows": [
       ["Text input", "Yes", "Yes"],
       ["Image / vision input", "Yes", "Yes"],
       ["Audio input", "No native audio input documented", "No native audio input documented"],
       ["Video input", "No native video input documented", "No native video input documented"],
       ["Text output", "Yes", "Yes"],
       ["Image output", "No", "No"],
       ["Audio output", "No", "No"],
       ["Video output", "No", "No"],
       ["Tool / function calling", "Yes, function calling, web search, X search, code execution", "Yes, function calling, web search, X search, code execution"],
       ["Computer use", "Via Grok Build and agent environments", "Via Grok Build and agent environments, with native Grok Bot understanding"],
       ["API access", "Yes", "Yes"],
       ["Product access", "Grok Build + Cursor + xAI API + partner gateways", "Grok Build + Cursor + xAI API + partner gateways"],
       ["Weights / license", "Proprietary", "Proprietary"]
     ]},
    {"id": "spec-grok46-47-4", "type": "spec_table", "title": "Model behavior",
     "columns": ["Grok 4.6", "Grok 4.7"],
     "rows": [
       ["Primary focus", "Post-training upgrade of Grok 4.5 for longer agentic runs", "New, larger base model plus longer RL for multi-hour tasks"],
       ["Long-horizon work", "Supplemental training run with regenerated SFT trajectories and RL over coding and agent environments", "Longer RL run over a harder multi-hour task mix, with stronger self-verification"],
       ["Agent orchestration", "Grok Build and Cursor agent loops, with context compaction", "Same harnesses, plus native understanding of the Grok Bot harness"],
       ["Efficiency / generation change", "Baseline verbosity", "About 81K output tokens per task vs about 38K (independent test): roughly 2.5x per-task cost at 2x wall-clock time"],
       ["Safety / approvals", "Standard xAI safeguards", "New safeguard stack: LatchBio 62.4%, HackerBench v0.3 lets through 3.3% of risky prompts"]
     ]},
    {"id": "spec-grok46-47-5", "type": "spec_table", "title": "Coding",
     "columns": ["Grok 4.6", "Grok 4.7"],
     "rows": [
       ["CursorBench 4.0", "40.4% (high)", "**46.3%** (xhigh)"],
       ["DeepSWE v1.1", "65.2%", "**71.0%** (high)"],
       ["Terminal-Bench 4.0 (Grok Build)", "20.3%", "**38.0%** (xhigh)"],
       ["Terminal-Bench 4.0 (mini-swe-agent, independent)", "Not published", "26%"],
       ["SWE-Marathon v1.1", "31.9% (high)", "**46.0%** (high)"],
       ["EEBench", "60.0%", "**66.0%** (xhigh)"]
     ]},
    {"id": "spec-grok46-47-6", "type": "spec_table", "title": "Knowledge",
     "columns": ["Grok 4.6", "Grok 4.7"],
     "rows": [
       ["Harvey Legal Agent", "15.8%", "**19.6%**"],
       ["HealthBench Professional", "48.5%", "**56.7%**"]
     ]},
    {"id": "spec-grok46-47-7", "type": "spec_table", "title": "Agentic & computer use",
     "columns": ["Grok 4.6", "Grok 4.7"],
     "rows": [
       ["AA-Briefcase v1.1", "1546 Elo", "**1657 Elo**"],
       ["Coding Agent Index (Grok Build, independent)", "47", "**56**"]
     ]},
    {"id": "markdown-grok46-47-bottom-line", "type": "markdown",
     "content": "## Bottom line\n\nGrok 4.7 is a same-price, same-context upgrade that buys longer horizons and better self-checking: every shared launch-table row improves, with the clearest gains on Terminal-Bench 4.0 and SWE-Marathon v1.1. The vendor deltas carry cross-effort and harness caveats, and higher verbosity means per-task costs rise even as per-token prices stay flat. Upgrade when long agent runs are the workload, and measure the win in completed jobs rather than leaderboard points."}
  ]'::jsonb,
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing Grok 4.7', 'url', 'https://x.ai/news/grok-4-7'),
    jsonb_build_object('title', 'Grok 4.7 model documentation', 'url', 'https://docs.x.ai/developers/models/grok-4-7'),
    jsonb_build_object('title', 'Grok 4.7 Benchmarks, Pricing and Context Window', 'url', 'https://llm-stats.com/models/grok-4-7'),
    jsonb_build_object('title', 'Introducing Grok 4.6', 'url', 'https://x.ai/news/grok-4-6')
  ),
  jsonb_build_object(
    'modelA', 'grok-4-6',
    'modelB', 'grok-4-7',
    'verification_pass', 'grok-4-7-comparison-2026-09-22'
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-09-21',
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
