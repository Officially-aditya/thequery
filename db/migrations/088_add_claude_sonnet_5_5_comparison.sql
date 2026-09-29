-- Add Claude Sonnet 5.5 to the model catalog and record the Sonnet 5.5 launch
-- scorecard as normalized benchmark evidence, following the pattern set by
-- 048 (Grok 4.7) and 052 (Opus 5.5).
--
-- Source: Anthropic's introduction article for Claude Sonnet 5.5
-- (September 28, 2026) as the model card, which reports Sonnet 5.5 against
-- Sonnet 5, Opus 5.5 and GPT-6 Sol side by side, plus the Claude Platform
-- model overview and the what's-new page for specifications, pricing,
-- availability and the five breaking API changes.
--
-- Footnote handling, all carried in the row notes so the rendered
-- comparison keeps the vendor's own caveats:
-- 1. Opus 5.5's Terminal-Bench 4.0 value is its highest, at Xhigh effort.
-- 2. Sonnet 5.5 scores lower at Max than at Xhigh on FrontierCode, so both
--    readings are stored as separate rows rather than collapsed.
-- 3. GDPval-AA and AA-Briefcase were run by Artificial Analysis on a
--    pre-release Sonnet 5.5 deployment that had a structured-output bug
--    Anthropic says has since been fixed.
-- 4. OpenAI fixed an image-understanding bug in GPT-6 Sol, so the
--    Artificial Analysis and Surge AI figures cited for Sol may predate it.
--
-- The four "—" rows for GPT-6 Sol (Terminal-Bench 4.0, CursorBench 4.0,
-- Humanity's Last Exam, OSWorld 2.1) are not reported by OpenAI and are
-- deliberately left unseeded. Opus 5.5's FrontierCode 1.1 Main value of
-- 49.3% is already stored from the Sol launch table, so it is not repeated.

-- 0. Label correction: the Opus 5.5 launch table reports the computer-use
-- row as "OSWorld 2.1", not 2.0. Migration 052 stored both Anthropic rows
-- under the older spelling, which would render a duplicate OSWorld row in
-- every database-generated comparison against the new Sonnet 5.5 rows.
-- Same class of canonicalization as 070: use the label the vendor printed.
UPDATE model_benchmarks SET
  benchmark_name = 'OSWorld 2.1',
  updated_at = NOW()
WHERE id IN ('tq-20260922-opus55-osworld20-partial', 'tq-20260922-fable51-osworld20-partial');

-- 1. Model catalog entry for Claude Sonnet 5.5.
INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at
)
SELECT
  'claude-sonnet-5-5',
  'Claude Sonnet 5.5',
  'Anthropic',
  DATE '2026-09-28',
  DATE '2026-09-28',
  'proprietary',
  'Claude Sonnet 5',
  '{
    "Developer": "Anthropic",
    "Release date": "2026-09-28",
    "API model ID": "claude-sonnet-5-5",
    "Context window": "1M tokens",
    "Max output": "128K tokens (300K on Batch API beta)",
    "Knowledge cutoff": "Jun 2026",
    "Reasoning / effort": "Adaptive thinking on by default; low/medium/high/xhigh/max, high default on the Claude API and medium in Claude Code and the Claude apps",
    "Input / 1M tokens": "$2",
    "Cached input / 1M": "$0.20",
    "Cache write / 1M": "$2.50 (5 min) / $4 (1 hr)",
    "Output / 1M tokens": "$10",
    "Batch / flex discount": "Batch API: 50% off input and output",
    "Long-context surcharge": "None - standard pricing through 1M context",
    "Text input": "Yes",
    "Image / vision input": "Yes",
    "Audio input": "No native audio input documented",
    "Video input": "No native video input documented",
    "File / document input": "Yes - PDF support and the Files API",
    "Text output": "Yes",
    "Image output": "No",
    "Audio output": "No",
    "Video output": "No",
    "Tool / function calling": "Yes - forced tool use returns an error, use auto with strict tools",
    "Computer use": "Yes - computer_toolset_20260801 on the Claude API and Google Cloud",
    "API access": "Yes",
    "Product access": "Claude + Claude Code + Claude apps + Claude API + cloud partners",
    "Weights / license": "Proprietary",
    "Primary focus": "Well-scoped everyday work: fixing bugs, writing code, and producing polished documents, slides and spreadsheets",
    "Long-horizon work": "Strong on long-horizon and image-understanding work, and the first Sonnet model to beat Pokemon Red working only from screenshots",
    "Agent orchestration": "Batches tool calls together more often than Sonnet 5, leading to fewer steps and lower cost per task",
    "User collaboration": "Writes more clearly than the previous generation, and early testers preferred it as a collaboration partner",
    "Efficiency / generation change": "Generates output 30%+ faster than Sonnet 5 at the same $2/$10 list price, for up to 30% less cost per task",
    "Safety / approvals": "First Sonnet with cyber safeguards that fall back to Sonnet 5, the same biology safeguards as Sonnet 5, plus reasoning-extraction classifiers and expanded preserved thinking"
  }'::jsonb,
  '[
    {"title": "Claude Sonnet 5.5", "url": "https://www.anthropic.com/claude-sonnet-5-5"},
    {"title": "Claude Sonnet 5.5 overview", "url": "https://platform.claude.com/docs/en/models/sonnet-5-5/overview"},
    {"title": "What is new in Claude Sonnet 5.5", "url": "https://platform.claude.com/docs/en/models/sonnet-5-5/whats-new-sonnet-5-5"},
    {"title": "Claude Sonnet 5.5 system card", "url": "https://www.anthropic.com/document/claude-sonnet-5-5-system-card"}
  ]'::jsonb,
  'Second release in the Claude 5.5 family, following Opus 5.5. Same $2/$10 list price as Sonnet 5 and half the Opus 5.5 rate, with a 1M-token context window and the same Jun 2026 cutoff. The savings come from token and tool-call efficiency rather than a lower rate, so teams should re-run their effort sweep because effort levels are recalibrated and the API default is now high. Five breaking changes affect code migrating from Sonnet 5.',
  '{
    "verification_pass": "sonnet-5-5-comparison-2026-09-28",
    "default_effort": "high",
    "catalog_status": "current"
  }'::jsonb,
  NOW()
WHERE NOT EXISTS (SELECT 1 FROM models WHERE slug = 'claude-sonnet-5-5')
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

-- 2. Benchmark evidence from the Sonnet 5.5 launch scorecard
-- (September 28, 2026).
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
  {"id":"tq-20260928-sonnet55-terminalbench40","model_slug":"claude-sonnet-5-5","category":"coding","benchmark_name":"Terminal-Bench 4.0","benchmark_version":null,"score_numeric":70.6,"score_display":"70.6%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Self-reported launch headline. The launch table does not state an effort for this row. Sonnet 5 comparison value is 10.3 percent and Opus 5.5 is 66.4 percent at Xhigh, its highest. GPT-6 Sol did not report this benchmark publicly."},
  {"id":"tq-20260928-sonnet55-frontiercode11main-xhigh","model_slug":"claude-sonnet-5-5","category":"coding","benchmark_name":"FrontierCode 1.1 Main","benchmark_version":null,"score_numeric":52.1,"score_display":"52.1%","score_unit":"percent","tools":null,"reasoning_effort":"xhigh","harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Self-reported launch result at Xhigh effort, the higher of the two readings Anthropic publishes. At this setting Sonnet 5.5 matches GPT-6 Sol best score for about a fifth of the cost per task."},
  {"id":"tq-20260928-sonnet55-frontiercode11main-max","model_slug":"claude-sonnet-5-5","category":"coding","benchmark_name":"FrontierCode 1.1 Main","benchmark_version":null,"score_numeric":46.2,"score_display":"46.2%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Self-reported launch result at Max effort, kept as a separate row because Anthropic states the score is lower at Max than at Xhigh: at Max the model more often ran Claude Code code-review skill, which split review across subagents, and in two examined cases that caused a timeout or out-of-scope edits."},
  {"id":"tq-20260928-sonnet55-cursorbench40","model_slug":"claude-sonnet-5-5","category":"coding","benchmark_name":"CursorBench","benchmark_version":"4.0","score_numeric":55.5,"score_display":"55.5%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":"ambiguous multi-file tasks from real Cursor sessions","evaluator":"Anthropic","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Self-reported launch headline, second only to Opus 5.5 at 57.8 percent. At Low effort Sonnet 5.5 exceeds Sonnet 5 best score for less than a tenth of the cost per task. GPT-6 Sol did not report this benchmark publicly."},
  {"id":"tq-20260928-sonnet55-gdpvalaa21","model_slug":"claude-sonnet-5-5","category":"agentic_computer_use","benchmark_name":"GDPval-AA","benchmark_version":"v2.1","score_numeric":1844,"score_display":"1844 Elo","score_unit":"Elo","tools":null,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Artificial Analysis ran this on a pre-release Sonnet 5.5 deployment that had a bug affecting structured outputs, which Anthropic says has since been fixed and expects the effect to be small and to understate the model. Two points below Opus 5.5 at 1846 Elo, about 400 above Sonnet 5 at 1449."},
  {"id":"tq-20260928-sonnet55-aabriefcase11","model_slug":"claude-sonnet-5-5","category":"agentic_computer_use","benchmark_name":"AA-Briefcase","benchmark_version":"v1.1","score_numeric":1811,"score_display":"1811 Elo","score_unit":"Elo","tools":null,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"New long-horizon knowledge-work benchmark, run by Artificial Analysis on the same pre-release deployment as GDPval-AA v2.1. At Medium effort Sonnet 5.5 beats Sonnet 5 best score for about one ninth of the cost per task. Opus 5.5 comparison value is 1822 Elo."},
  {"id":"tq-20260928-sonnet55-hle-tools","model_slug":"claude-sonnet-5-5","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":64.5,"score_display":"64.5%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Self-reported launch result with tools. Sonnet 5 comparison value is 54.9 percent and Opus 5.5 is 67.7 percent. GPT-6 Sol did not report this benchmark publicly."},
  {"id":"tq-20260928-sonnet55-osworld21-partial","model_slug":"claude-sonnet-5-5","category":"agentic_computer_use","benchmark_name":"OSWorld 2.1","benchmark_version":"partial","score_numeric":80.1,"score_display":"80.1%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Self-reported partial-credit launch result. Sonnet 5 comparison value is 57.0 percent and Opus 5.5 is 81.8 percent. GPT-6 Sol did not report this benchmark publicly."},
  {"id":"tq-20260928-sonnet55-chartography-notools","model_slug":"claude-sonnet-5-5","category":"multimodal","benchmark_name":"Chartography","benchmark_version":null,"score_numeric":61.6,"score_display":"61.6%","score_unit":"percent","tools":false,"reasoning_effort":null,"harness":"Surge AI","evaluator":"Anthropic","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Visual chart recognition, no tools. This is the no-tools reading and is kept distinct from the with-tools Chartography rows already stored for Opus 5.5 and Fable 5.1. Sonnet 5 comparison value is 15.6 percent and Opus 5.5 is 64.4 percent."},
  {"id":"tq-20260928-sonnet5-terminalbench40","model_slug":"claude-sonnet-5","category":"coding","benchmark_name":"Terminal-Bench 4.0","benchmark_version":null,"score_numeric":10.3,"score_display":"10.3%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Sonnet 5 comparison value in the Sonnet 5.5 launch table. The launch page does not explain this low result."},
  {"id":"tq-20260928-sonnet5-frontiercode11main","model_slug":"claude-sonnet-5","category":"coding","benchmark_name":"FrontierCode 1.1 Main","benchmark_version":null,"score_numeric":42.4,"score_display":"42.4%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Sonnet 5 comparison value in the Sonnet 5.5 launch table, stated without an effort qualifier."},
  {"id":"tq-20260928-sonnet5-cursorbench40","model_slug":"claude-sonnet-5","category":"coding","benchmark_name":"CursorBench","benchmark_version":"4.0","score_numeric":34.1,"score_display":"34.1%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":"ambiguous multi-file tasks from real Cursor sessions","evaluator":"Anthropic","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Sonnet 5 comparison value in the Sonnet 5.5 launch table."},
  {"id":"tq-20260928-sonnet5-gdpvalaa21","model_slug":"claude-sonnet-5","category":"agentic_computer_use","benchmark_name":"GDPval-AA","benchmark_version":"v2.1","score_numeric":1449,"score_display":"1449 Elo","score_unit":"Elo","tools":null,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Sonnet 5 comparison value in the Sonnet 5.5 launch table, read from the same Artificial Analysis run as the Sonnet 5.5 row. Distinct from the earlier GDPval-AA v2 value of 1618 Elo stored for Sonnet 5 at launch."},
  {"id":"tq-20260928-sonnet5-aabriefcase11","model_slug":"claude-sonnet-5","category":"agentic_computer_use","benchmark_name":"AA-Briefcase","benchmark_version":"v1.1","score_numeric":1359,"score_display":"1359 Elo","score_unit":"Elo","tools":null,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Sonnet 5 comparison value in the Sonnet 5.5 launch table, read from the same Artificial Analysis run as the Sonnet 5.5 row."},
  {"id":"tq-20260928-sonnet5-hle-tools","model_slug":"claude-sonnet-5","category":"knowledge","benchmark_name":"Humanity's Last Exam","benchmark_version":null,"score_numeric":54.9,"score_display":"54.9%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Sonnet 5 comparison value in the Sonnet 5.5 launch table, with tools."},
  {"id":"tq-20260928-sonnet5-osworld21-partial","model_slug":"claude-sonnet-5","category":"agentic_computer_use","benchmark_name":"OSWorld 2.1","benchmark_version":"partial","score_numeric":57.0,"score_display":"57.0%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Anthropic","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Sonnet 5 comparison value in the Sonnet 5.5 launch table, partial credit."},
  {"id":"tq-20260928-sonnet5-chartography-notools","model_slug":"claude-sonnet-5","category":"multimodal","benchmark_name":"Chartography","benchmark_version":null,"score_numeric":15.6,"score_display":"15.6%","score_unit":"percent","tools":false,"reasoning_effort":null,"harness":"Surge AI","evaluator":"Anthropic","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Sonnet 5 comparison value in the Sonnet 5.5 launch table, no tools. The launch page does not explain this low result."},
  {"id":"tq-20260928-opus55-aabriefcase11","model_slug":"claude-opus-5-5","category":"agentic_computer_use","benchmark_name":"AA-Briefcase","benchmark_version":"v1.1","score_numeric":1822,"score_display":"1822 Elo","score_unit":"Elo","tools":null,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Opus 5.5 comparison value in the Sonnet 5.5 launch table, read from the same Artificial Analysis run as the Sonnet 5.5 row."},
  {"id":"tq-20260928-opus55-chartography-notools","model_slug":"claude-opus-5-5","category":"multimodal","benchmark_name":"Chartography","benchmark_version":null,"score_numeric":64.4,"score_display":"64.4%","score_unit":"percent","tools":false,"reasoning_effort":null,"harness":"Surge AI","evaluator":"Anthropic","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"Opus 5.5 no-tools Chartography comparison value in the Sonnet 5.5 launch table. Kept distinct from the 89.0 percent with-tools reading already stored from the Opus 5.5 launch table."},
  {"id":"tq-20260928-gpt6sol-gdpvalaa21","model_slug":"gpt-6-sol","category":"agentic_computer_use","benchmark_name":"GDPval-AA","benchmark_version":"v2.1","score_numeric":1487,"score_display":"1487 Elo","score_unit":"Elo","tools":null,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"OpenAI","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"GPT-6 Sol comparison value as cited by Anthropic. OpenAI fixed a bug that degraded image understanding in GPT-6 Sol, and Anthropic notes the Artificial Analysis figures for Sol may not yet reflect the fixed version."},
  {"id":"tq-20260928-gpt6sol-aabriefcase11","model_slug":"gpt-6-sol","category":"agentic_computer_use","benchmark_name":"AA-Briefcase","benchmark_version":"v1.1","score_numeric":1483,"score_display":"1483 Elo","score_unit":"Elo","tools":null,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"OpenAI","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"GPT-6 Sol comparison value as cited by Anthropic, with the same caveat about the GPT-6 Sol image-understanding fix."},
  {"id":"tq-20260928-gpt6sol-chartography-notools","model_slug":"gpt-6-sol","category":"multimodal","benchmark_name":"Chartography","benchmark_version":null,"score_numeric":53.6,"score_display":"53.6%","score_unit":"percent","tools":false,"reasoning_effort":null,"harness":"Surge AI","evaluator":"OpenAI","evaluation_date":"2026-09-28","source":"https://www.anthropic.com/claude-sonnet-5-5","notes":"GPT-6 Sol comparison value as cited by Anthropic, no tools. Anthropic states the Chartography figures for Sol may predate the OpenAI image-understanding fix, and that internal testing suggests the bug did not affect the result."}
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
