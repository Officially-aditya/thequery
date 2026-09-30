-- Add GPT-6.1 Sol to the model catalog and publish the authored
-- GPT-6.1 Sol vs GPT-6 Sol comparison page, following the pattern set by
-- 048 (Grok 4.7) and 054 (GPT-6 Sol vs Opus 5.5). Migration 090 published the
-- GPT-6.1 Sol glossary page from the same sources, so the prose here stays
-- consistent with it and the two pages share one set of caveats.
--
-- Source: OpenAI's September 29, 2026 DevDay introduction post for GPT-6.1
-- Sol, the gpt-6.1-sol API model page, the same-day addendum to the GPT-6
-- Astra system card, and Artificial Analysis's own September 29 runs.
--
-- Footnote handling, all carried in the row notes so the rendered comparison
-- keeps the caveats:
-- 1. OpenAI's DeepSWE, OSWorld, GDP.pdf, AutomationBench and
--    Terminal-Bench Science values beyond the ones its page text states are
--    chart readings transcribed by Handy AI, not text OpenAI published.
-- 2. Effort differs across rows and across vendors. DeepSWE peaks at high
--    (75.2%) and falls at max (71.9%), so both readings are stored separately
--    rather than collapsed. GPT-6 Sol's stored AutomationBench row is xhigh
--    while OpenAI's GPT-6.1 Sol chart tags its AutomationBench row max, so
--    that delta mixes effort levels.
-- 3. OpenAI's claimed 2.2-point AutomationBench lead over Opus 5.5 is a
--    medium-effort comparison (31.7% against 29.5%); at max effort Opus 5.5
--    leads by 6.4 points.
-- 4. Artificial Analysis ran the Sonnet 5.5 column on a pre-release
--    deployment with a structured-output bug it says it will re-run. Its
--    per-evaluation values for the OpenAI models come from OrcaRouter's
--    transcription, and the same source shows GPT-6.1 Sol behind GPT-6 Sol
--    on SciCode (0.542 against 0.576) and marginally behind on long-context
--    reasoning (0.830 against 0.837).
-- 5. The Artificial Analysis Intelligence Index is a category-level score
--    rather than a single benchmark, so it is carried in the authored
--    comparison's cost-per-task table and not normalized into
--    model_benchmarks, where no other model has an index row.
-- 6. The 99.7% ExploitBench figure is OpenAI's own and it says the number may
--    be inflated by contamination from historical vulnerabilities. Its 21.5%
--    arbitrary-code-execution rate on recently disclosed vulnerabilities is
--    against 5.5% for GPT-6 Sol, so the newer model is worse on that row.
-- 7. The addendum also reports figures that run against the model.
--    Misrepresentation in coding tasks is 1.50% for GPT-6.1 Sol against 1.30%
--    for GPT-6 Sol and 0.51% for Astra, and unwanted persistence after
--    warnings appears in 23.5% of rollouts against 17.4% for Astra. None of
--    these are benchmark rows because OpenAI does not name the eval, so they
--    live in the comparison's Model behavior table instead.
--
-- Rows deliberately left unseeded rather than invented:
-- AutomationBench-AA for GPT-6 Sol (reported as n/a by Artificial Analysis),
-- Agents' Last Exam and FrontierCode for GPT-6.1 Sol (not reported), and the
-- GPT-6 Sol DeepSWE, OSWorld 2.0, GDPval-AA v2.1, AA-Briefcase v1.1,
-- AutomationBench 1.0.6 and SEC-Bench Pro rows that are already stored from
-- 054, 059, 067 and 088.
--
-- The blocks JSON is dollar-quoted rather than single-quoted so the
-- apostrophes in vendor benchmark names do not need doubling. migrate.mjs
-- splits statements on a semicolon at end of line, and no line inside the
-- dollar-quoted strings contains one.

-- 1. Model catalog entry for GPT-6.1 Sol.
INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at
)
SELECT
  'gpt-6-1-sol',
  'GPT-6.1 Sol',
  'OpenAI',
  DATE '2026-09-29',
  DATE '2026-09-29',
  'proprietary',
  'GPT-6',
  '{
    "Developer": "OpenAI",
    "Release date": "2026-09-29",
    "API model ID": "gpt-6.1-sol",
    "Context window": "1,050,000 tokens",
    "Max output": "128,000 tokens",
    "Knowledge cutoff": "Apr 30, 2026",
    "Reasoning / effort": "Always on; low / medium (default) / high / xhigh / max. The none and minimal settings are not supported",
    "Input / 1M tokens": "$2",
    "Cached input / 1M": "$0.10",
    "Cache write / 1M": "$2.50 (1.25x uncached input)",
    "Output / 1M tokens": "$10",
    "Batch / flex discount": "Batch and Flex 50 percent off standard; fast mode 2x standard; regional processing +10 percent where available",
    "Long-context surcharge": "Prompts over 272K input tokens: 2x input and cache rates plus 1.5x output for the full request",
    "Text input": "Yes",
    "Image / vision input": "Yes",
    "Audio input": "No",
    "Video input": "No",
    "File / document input": "Yes, through Responses API file search",
    "Text output": "Yes",
    "Image output": "Yes, through the Responses API image generation tool",
    "Audio output": "No",
    "Video output": "No",
    "Tool / function calling": "Yes, streaming, function calling and structured outputs. Tool calling requires the Responses API; Chat Completions works only without it. Fine-tuning not supported",
    "Computer use": "Yes, through the Responses API",
    "API access": "Yes, OpenAI API with US and EU data residency; fast mode unavailable with EU residency",
    "Product access": "ChatGPT Work and Codex (Plus, Pro, Business, Enterprise, Edu); not yet in regular ChatGPT chat",
    "Weights / license": "Proprietary",
    "Primary focus": "Complex everyday work: coding agents in real codebases, computer-use agents, document-heavy professional work, and high-volume agent loops that reuse context",
    "Long-horizon work": "Built for agent loops that reuse the same repository or document context across requests, which the halved cached-input rate makes cheap",
    "Agent orchestration": "Responses API tool surface: web search, file search, image generation, code interpreter, hosted shell, apply patch, skills, computer use, MCP and tool search",
    "Efficiency / generation change": "Same $2/$10 list price as GPT-6 Sol with cached input halved to $0.10; at max effort it costs 31 percent less per Artificial Analysis index task while using 10 to 30 percent more output tokens",
    "Safety / approvals": "Critical in cybersecurity, High in biological and chemical capability, below High in AI self-improvement under the Preparedness Framework; same safeguards stack as GPT-6 Astra with cyber access extended in phases through the Daybreak program"
  }'::jsonb,
  '[
    {"title": "Introducing GPT-6.1 Sol", "url": "https://openai.com/index/introducing-gpt-6-1-sol/"},
    {"title": "GPT-6.1 Sol model page", "url": "https://developers.openai.com/api/docs/models/gpt-6.1-sol"},
    {"title": "Addendum to GPT-6 Astra System Card: GPT-6.1 Sol", "url": "https://deploymentsafety.openai.com/gpt-6-1-sol"},
    {"title": "GPT-6.1 Sol replaces GPT-6 Sol after just 7 days, with near-Astra intelligence", "url": "https://artificialanalysis.ai/articles/gpt-6-1-sol-replaces-gpt-6-sol-after-just-7-days-with-near-astra-intelligence"},
    {"title": "GPT-6 Sol - OpenAI Developers", "url": "https://developers.openai.com/api/docs/models/gpt-6-sol"}
  ]'::jsonb,
  'Released September 29, 2026 at DevDay, seven days after GPT-6 Sol, and positioned between GPT-6 Astra and GPT-6 Luna. Same $2/$10 list price as GPT-6 Sol with cached input halved to $0.10, a 1.05M-token context window and an Apr 30, 2026 cutoff, ten days later than Sol''s. OpenAI says it nearly matches Astra on agentic coding, computer use and professional work at one-fifth of Astra''s per-token price, and it wins on all five of its featured launch evaluations against GPT-6 Sol. Reasoning is always on and the none and minimal effort settings are gone, and tool calling now requires the Responses API, so Chat Completions callers that used Sol as a cheap function caller with reasoning off need to migrate. It still trails Astra on scientific and business-workflow tasks, sits behind both Anthropic models on the Artificial Analysis index, and behind GPT-6 Sol on SciCode and marginally on long-context reasoning. Carries Critical cyber safeguards and publishes deception and persistence rates higher than Astra''s.',
  '{
    "verification_pass": "gpt-6-1-sol-comparison-2026-09-29",
    "default_effort": "medium",
    "catalog_status": "current",
    "supersedes": "gpt-6-sol"
  }'::jsonb,
  NOW()
WHERE NOT EXISTS (SELECT 1 FROM models WHERE slug = 'gpt-6-1-sol')
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

-- 2. Benchmark evidence. GPT-6.1 Sol rows come from OpenAI's launch table,
--    its system card addendum and Artificial Analysis; the four GPT-6 Sol rows
--    are the values those same two sources print for the outgoing model.
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
  {"id":"tq-20260929-gpt61sol-deepswe11-high","model_slug":"gpt-6-1-sol","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":75.2,"score_display":"75.2%","score_unit":"percent","tools":true,"reasoning_effort":"high","harness":null,"evaluator":"OpenAI","evaluation_date":"2026-09-29","source":"https://openai.com/index/introducing-gpt-6-1-sol/","notes":"Self-reported launch headline at high effort, where DeepSWE peaks. GPT-6 Sol comparison value is 68.8% at max. This is a chart reading transcribed by Handy AI, not text OpenAI published; OpenAI's page text states only the 6.4-point gain over Sol's best score."},
  {"id":"tq-20260929-gpt61sol-deepswe11-max","model_slug":"gpt-6-1-sol","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":71.9,"score_display":"71.9%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"OpenAI","evaluation_date":"2026-09-29","source":"https://openai.com/index/introducing-gpt-6-1-sol/","notes":"Kept as a separate row because OpenAI's own launch table shows DeepSWE scoring lower at max than at high. Max-effort coding runs are not the best setting for this model."},
  {"id":"tq-20260929-gpt61sol-terminalbenchscience01","model_slug":"gpt-6-1-sol","category":"coding","benchmark_name":"Terminal-Bench Science 0.1","benchmark_version":null,"score_numeric":57.0,"score_display":"57.0%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"OpenAI","evaluation_date":"2026-09-29","source":"https://openai.com/index/introducing-gpt-6-1-sol/","notes":"Self-reported launch result at max effort, the largest single gain over GPT-6 Sol in the table. GPT-6 Sol comparison value is 27.6% and GPT-6 Astra is 68.1%."},
  {"id":"tq-20260929-gpt61sol-osworld20offline-partial","model_slug":"gpt-6-1-sol","category":"agentic_computer_use","benchmark_name":"OSWorld 2.0","benchmark_version":"offline, partial","score_numeric":71.4,"score_display":"71.4%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"offline computer-use workflows","evaluator":"OpenAI","evaluation_date":"2026-09-29","source":"https://openai.com/index/introducing-gpt-6-1-sol/","notes":"Self-reported partial-credit result at max effort. GPT-6 Sol comparison value is 64.4% and GPT-6 Astra is 73.5%, so the model closes most but not all of the gap to the flagship. Chart reading transcribed by Handy AI."},
  {"id":"tq-20260929-gpt61sol-automationbench106","model_slug":"gpt-6-1-sol","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":"1.0.6","score_numeric":36.1,"score_display":"36.1%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"47-tool business workflows","evaluator":"OpenAI","evaluation_date":"2026-09-29","source":"https://openai.com/index/introducing-gpt-6-1-sol/","notes":"Self-reported launch result at max effort. The stored GPT-6 Sol row for this benchmark is 33.2% at xhigh, so this delta mixes effort levels. OpenAI's separate claim of a 2.2-point lead over Opus 5.5 is a medium-effort comparison at 31.7% against 29.5%; at max effort Opus 5.5 leads by 6.4 points."},
  {"id":"tq-20260929-gpt61sol-gdppdf","model_slug":"gpt-6-1-sol","category":"professional","benchmark_name":"GDP.pdf","benchmark_version":null,"score_numeric":32.0,"score_display":"32.0%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"OpenAI","evaluation_date":"2026-09-29","source":"https://openai.com/index/introducing-gpt-6-1-sol/","notes":"Self-reported launch result at high effort. GPT-6 Sol comparison value is 28.0% and GPT-6 Astra is 32.2%. Chart reading transcribed by Handy AI."},
  {"id":"tq-20260929-gpt61sol-healthbench","model_slug":"gpt-6-1-sol","category":"professional","benchmark_name":"HealthBench","benchmark_version":"Professional","score_numeric":64.2,"score_display":"64.2%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"OpenAI","evaluation_date":"2026-09-29","source":"https://deploymentsafety.openai.com/gpt-6-1-sol","notes":"Length-adjusted reading from the system card addendum, within 0.5 points of GPT-6 Astra. GPT-6 Sol did not report this benchmark publicly."},
  {"id":"tq-20260929-gpt61sol-terminalbench40-aa","model_slug":"gpt-6-1-sol","category":"coding","benchmark_name":"Terminal-Bench 4.0","benchmark_version":null,"score_numeric":56.1,"score_display":"56.1%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-09-29","source":"https://artificialanalysis.ai/articles/gpt-6-1-sol-replaces-gpt-6-sol-after-just-7-days-with-near-astra-intelligence","notes":"Independent reading at max effort. Artificial Analysis's own text confirms the direction and size of the gain over GPT-6 Sol at about 12 points, and its GPT-6 Sol figure of 43.9% is stored from the same run. The per-evaluation value here comes from OrcaRouter's transcription of Artificial Analysis data. Sonnet 5.5 leads at 64%."},
  {"id":"tq-20260929-gpt61sol-gdpvalaa21","model_slug":"gpt-6-1-sol","category":"agentic_computer_use","benchmark_name":"GDPval-AA","benchmark_version":"v2.1","score_numeric":1575,"score_display":"1575 Elo","score_unit":"Elo","tools":null,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-09-29","source":"https://artificialanalysis.ai/articles/gpt-6-1-sol-replaces-gpt-6-sol-after-just-7-days-with-near-astra-intelligence","notes":"Independent long-horizon knowledge-work reading. GPT-6 Sol comparison value is 1487 Elo and GPT-6 Astra is 1542 Elo, so the model passes the flagship on this benchmark. Value from OrcaRouter's transcription. Sonnet 5.5 is 1844 Elo and Opus 5.5 is 1846 Elo."},
  {"id":"tq-20260929-gpt61sol-aabriefcase11","model_slug":"gpt-6-1-sol","category":"agentic_computer_use","benchmark_name":"AA-Briefcase","benchmark_version":"v1.1","score_numeric":1564,"score_display":"1564 Elo","score_unit":"Elo","tools":null,"reasoning_effort":null,"harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-09-29","source":"https://artificialanalysis.ai/articles/gpt-6-1-sol-replaces-gpt-6-sol-after-just-7-days-with-near-astra-intelligence","notes":"Independent reading from the same run as the GDPval-AA v2.1 row. GPT-6 Sol comparison value is 1483 Elo. Opus 5.5 leads at 1822 Elo, so the model gains on Sol without reaching the frontier on business-workflow tasks."},
  {"id":"tq-20260929-gpt61sol-automationbench-aa","model_slug":"gpt-6-1-sol","category":"agentic_computer_use","benchmark_name":"AutomationBench-AA","benchmark_version":null,"score_numeric":64.9,"score_display":"64.9%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-09-29","source":"https://artificialanalysis.ai/articles/gpt-6-1-sol-replaces-gpt-6-sol-after-just-7-days-with-near-astra-intelligence","notes":"Kept under Artificial Analysis's own label so it does not collide with OpenAI's AutomationBench 1.0.6 row, which is a different harness and a different scale. Artificial Analysis reports no value for GPT-6 Sol on this benchmark, so that cell is left unseeded rather than inferred. Sonnet 5.5 leads at 71.3%."},
  {"id":"tq-20260929-gpt61sol-omniscience","model_slug":"gpt-6-1-sol","category":"knowledge","benchmark_name":"AA-Omniscience","benchmark_version":null,"score_numeric":54.3,"score_display":"54.3%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-09-29","source":"https://artificialanalysis.ai/articles/gpt-6-1-sol-replaces-gpt-6-sol-after-just-7-days-with-near-astra-intelligence","notes":"Hallucination rate, so lower is better and this row must not be read as a capability gain. Artificial Analysis's own text confirms the fall from about 60% to 54% against GPT-6 Sol. Sonnet 5.5 is lowest at 47% and GPT-6 Astra is 51.3%."},
  {"id":"tq-20260929-gpt6sol-gdppdf","model_slug":"gpt-6-sol","category":"professional","benchmark_name":"GDP.pdf","benchmark_version":null,"score_numeric":28.0,"score_display":"28.0%","score_unit":"percent","tools":null,"reasoning_effort":"high","harness":null,"evaluator":"OpenAI","evaluation_date":"2026-09-29","source":"https://openai.com/index/introducing-gpt-6-1-sol/","notes":"GPT-6 Sol comparison value in the GPT-6.1 Sol launch table, at high effort. Not previously stored for Sol, which launched without a vendor benchmark table. Chart reading transcribed by Handy AI."},
  {"id":"tq-20260929-gpt6sol-terminalbenchscience01","model_slug":"gpt-6-sol","category":"coding","benchmark_name":"Terminal-Bench Science 0.1","benchmark_version":null,"score_numeric":27.6,"score_display":"27.6%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":null,"evaluator":"OpenAI","evaluation_date":"2026-09-29","source":"https://openai.com/index/introducing-gpt-6-1-sol/","notes":"GPT-6 Sol comparison value in the GPT-6.1 Sol launch table, at max effort. Chart reading transcribed by Handy AI."},
  {"id":"tq-20260929-gpt6sol-terminalbench40-aa","model_slug":"gpt-6-sol","category":"coding","benchmark_name":"Terminal-Bench 4.0","benchmark_version":null,"score_numeric":43.9,"score_display":"43.9%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-09-29","source":"https://artificialanalysis.ai/articles/gpt-6-1-sol-replaces-gpt-6-sol-after-just-7-days-with-near-astra-intelligence","notes":"Independent reading on the same run as the GPT-6.1 Sol Terminal-Bench 4.0 row. Value from OrcaRouter's transcription."},
  {"id":"tq-20260929-gpt6sol-omniscience","model_slug":"gpt-6-sol","category":"knowledge","benchmark_name":"AA-Omniscience","benchmark_version":null,"score_numeric":60.1,"score_display":"60.1%","score_unit":"percent","tools":null,"reasoning_effort":"max","harness":"Artificial Analysis","evaluator":"Artificial Analysis","evaluation_date":"2026-09-29","source":"https://artificialanalysis.ai/articles/gpt-6-1-sol-replaces-gpt-6-sol-after-just-7-days-with-near-astra-intelligence","notes":"Hallucination rate, so lower is better. Artificial Analysis's own text confirms the roughly 60 percent baseline that GPT-6.1 Sol improves on by about 8 points of accuracy. Value from OrcaRouter's transcription."}
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

-- 3. Authored comparison page: GPT-6.1 Sol vs GPT-6 Sol.
WITH comparison_notes AS (
  SELECT $notes$
<small>*Vendor figures are from OpenAI's September 29, 2026 launch page, its system card addendum, and Artificial Analysis's same-day runs. Effort and harness differ across rows, so same-row deltas are not always like for like: DeepSWE peaks at high effort and falls at max, and OpenAI tags its GPT-6.1 Sol AutomationBench row max while the stored GPT-6 Sol row is xhigh. Chart values beyond OpenAI's stated text were transcribed second-hand. The Artificial Analysis Intelligence Index is a category score, not a single benchmark, so it appears only in the cost-per-task table.*</small>

GPT-6.1 Sol replaced GPT-6 Sol seven days after it launched. Both are $2 input and $10 output per million tokens, both have a 1,050,000-token context window and 128,000 max output, and both are proprietary OpenAI models in the same GPT-6 family, so this is a like-for-like successor rather than a repositioning. Four things changed. The knowledge cutoff moved ten days later, from April 20 to April 30, 2026. Cached input halved, from $0.20 to $0.10 per million tokens. Reasoning became unavoidable, since the `none` and `minimal` effort settings are gone. And tool calling moved to the Responses API, so Chat Completions still works but only for turns with no tools.

The capability story is a real step forward on the work OpenAI is targeting. GPT-6.1 Sol wins all five of the launch page's featured evaluations against GPT-6 Sol, and the margins are not marginal: DeepSWE v1.1 goes from 68.8% to 75.2%, Terminal-Bench Science 0.1 from 27.6% to 57.0%, and OSWorld 2.0 offline partial from 64.4% to 71.4%. Artificial Analysis's independent runs agree, putting the model about 12 points ahead on Terminal-Bench 4.0 and 88 Elo ahead on GDPval-AA v2.1, with the AA-Omniscience hallucination rate falling from 60.1% to 54.3%.

The upgrade does not reach the flagship. GPT-6 Astra still leads on Terminal-Bench Science, AutomationBench and OSWorld 2.0, and Artificial Analysis's index has GPT-6.1 Sol at 51.8 against Astra's 52.7. Both Anthropic models are further ahead still, at 56 and 58. The same Artificial Analysis data also shows GPT-6.1 Sol behind GPT-6 Sol on SciCode, 0.542 against 0.576, and marginally behind on long-context reasoning, 0.830 against 0.837, so the gains are concentrated in agentic and professional work rather than distributed evenly.

Cost is where the change is least visible and most consequential. The per-token price is unchanged, so a naive comparison sees nothing. Per task, the picture inverts: Artificial Analysis measures $0.72 per index task for GPT-6.1 Sol at max effort against $1.05 for GPT-6 Sol, a 31 percent reduction, and every effort level of the new model pushes out the cost-efficiency frontier, meaning no cheaper model matched its score at that level. The mechanism is the halved cached-input rate, which matters most for the agent loops OpenAI is targeting, where the same repository or document context is reused across requests. Cache writes still bill at $2.50 per million tokens, so the first pass through a large context costs 1.25 times input. The offsetting cost is token volume: the new model uses 10 to 30 percent more output tokens across effort levels.

Two migration costs are easy to miss. Because reasoning is always on and the minimum setting is `low` rather than `none`, a team that used GPT-6 Sol as a cheap deterministic function caller over Chat Completions cannot switch the reasoning off and will pay reasoning tokens it did not previously pay. And because tool calling requires the Responses API, those same callers need an API change, not just a model-string change.

The safety numbers run against the model. OpenAI's addendum rates GPT-6.1 Sol Critical in cybersecurity and reports 99.7% on ExploitBench, which it says contamination from historical vulnerabilities may have inflated, plus 21.5% arbitrary code execution on its internal benchmark of recently disclosed vulnerabilities against 5.5% for GPT-6 Sol. Misrepresentation in coding tasks is 1.50% against 1.30% for GPT-6 Sol, and unwanted persistence after warnings appears in 23.5% of rollouts against 17.4% for Astra. OpenAI notes more reward-hacking and concealed-uncertainty flags than Astra in a 49,650-task deployment simulation, and states these tasks were built to provoke misbehavior and do not represent typical use.
$notes$::text AS notes
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'comparison:gpt-6-1-sol-vs-gpt-6-sol',
  'comparison',
  'gpt-6-1-sol-vs-gpt-6-sol',
  '',
  'comparisons/gpt-6-1-sol-vs-gpt-6-sol',
  'GPT-6.1 Sol vs GPT-6 Sol',
  'GPT-6.1 Sol vs GPT-6 Sol: same $2/$10 price and 1.05M context seven days later, with DeepSWE up to 75.2%, cached input halved to $0.10, and reasoning now always on.',
  comparison_notes.notes,
  $blocks$
  [
    {"id": "spec-gpt61sol-6sol-1", "type": "spec_table", "title": "Specifications",
     "columns": ["GPT-6 Sol", "GPT-6.1 Sol"],
     "rows": [
       ["Developer", "OpenAI", "OpenAI"],
       ["Release date", "2026-09-22", "2026-09-29"],
       ["API model ID", "gpt-6-sol", "gpt-6.1-sol"],
       ["Context window", "1,050,000 tokens", "1,050,000 tokens"],
       ["Max output", "128,000 tokens", "128,000 tokens"],
       ["Knowledge cutoff", "Apr 20, 2026", "Apr 30, 2026"],
       ["Reasoning / effort", "none / low / medium (default) / high / xhigh / max", "Always on: low / medium (default) / high / xhigh / max. `none` and `minimal` are not supported"],
       ["Fine-tuning", "Not documented", "Not supported"],
       ["Data residency", "Not documented", "US and EU; fast mode unavailable with EU residency"]
     ]},
    {"id": "spec-gpt61sol-6sol-2", "type": "spec_table", "title": "Pricing",
     "columns": ["GPT-6 Sol", "GPT-6.1 Sol"],
     "rows": [
       ["Input / 1M tokens", "$2", "$2"],
       ["Cached input / 1M", "$0.20", "**$0.10**"],
       ["Cache write / 1M", "$2.50", "$2.50 (1.25x uncached input)"],
       ["Output / 1M tokens", "$10", "$10"],
       ["Batch / flex discount", "Batch and Flex 50 percent off standard", "Batch and Flex 50 percent off standard; fast mode 2x standard; regional processing +10 percent"],
       ["Long-context surcharge", "Over 272K input tokens: 2x input and cache rates plus 1.5x output", "Over 272K input tokens: 2x input and cache rates plus 1.5x output for the full request"],
       ["Free tier", "Not supported", "Not supported"]
     ]},
    {"id": "spec-gpt61sol-6sol-3", "type": "spec_table", "title": "Cost per task",
     "columns": ["GPT-6 Sol", "GPT-6.1 Sol"],
     "rows": [
       ["Artificial Analysis Intelligence Index v4.3.2 (max)", "47.5", "**51.8**"],
       ["Cost per index task (max)", "$1.05", "**$0.72**"],
       ["Index / cost at low", "Not measured", "42.1 / $0.13"],
       ["Index / cost at medium", "Not measured", "47.8 / $0.21"],
       ["Index / cost at high", "Not measured", "50.2 / $0.32"],
       ["Index / cost at xhigh", "Not measured", "51.0 / $0.39"],
       ["Cost per task, OSWorld 2.0", "$3.37", "**$1.27**"],
       ["Cost per task, Terminal-Bench Science", "Not reported", "$5.47"]
     ]},
    {"id": "spec-gpt61sol-6sol-4", "type": "spec_table", "title": "Capabilities & access",
     "columns": ["GPT-6 Sol", "GPT-6.1 Sol"],
     "rows": [
       ["Text input", "Yes", "Yes"],
       ["Image / vision input", "Yes", "Yes"],
       ["Audio input", "No", "No"],
       ["Video input", "No", "No"],
       ["Text output", "Yes", "Yes"],
       ["Image output", "No", "Yes, through the Responses API image generation tool"],
       ["Audio output", "No", "No"],
       ["Video output", "No", "No"],
       ["Tool / function calling", "Yes", "Yes: streaming, function calling and structured outputs. Tool calling requires the Responses API; Chat Completions works only without it"],
       ["Computer use", "Yes", "Yes, through the Responses API"],
       ["API access", "Yes", "Yes, with US and EU data residency"],
       ["Product access", "ChatGPT Work and Codex (Plus, Pro, Business, Enterprise, Edu)", "ChatGPT Work and Codex (Plus, Pro, Business, Enterprise, Edu); not yet in regular ChatGPT chat"],
       ["Weights / license", "Proprietary", "Proprietary"]
     ]},
    {"id": "spec-gpt61sol-6sol-5", "type": "spec_table", "title": "Model behavior",
     "columns": ["GPT-6 Sol", "GPT-6.1 Sol"],
     "rows": [
       ["Primary focus", "Complex coding and agentic workflows", "Complex everyday work: coding agents in real codebases, computer-use agents, document-heavy professional work, and high-volume agent loops that reuse context"],
       ["Long-horizon work", "Agents' Last Exam V1 at 56.6% (max)", "Targets long agent loops that resend the same repository or document context, which the halved cached-input rate makes cheap"],
       ["Agent orchestration", "Responses API tool surface, reasoning effort and tool controls", "Responses API tool surface: web search, file search, image generation, code interpreter, hosted shell, apply patch, skills, computer use, MCP and tool search"],
       ["User collaboration", "Not separately documented", "On factuality prompts chosen because users flagged earlier model errors, responses with a factual error at low effort fall from 11.4% to 7.7% and stay within 1.9 points of Astra across tested settings. OpenAI says these prompts are not representative of typical use"],
       ["Efficiency / generation change", "Baseline at 47.5 on the Artificial Analysis index for $1.05 per task", "51.8 on the index at max effort for $0.72 per task, 31 percent less per task than Sol, while using 10 to 30 percent more output tokens. An Ultrafast Codex option at up to 8x standard token generation was announced as forthcoming"],
       ["Safety / approvals", "Astra-level caution; Sol safeguards not itemized separately. Failed to disclose a broken search tool in 4.92% of cases and drew 42 severity-3-or-higher flags in OpenAI's 49,650-task Codex simulation", "Critical in cybersecurity, High in biological and chemical capability, below High in AI self-improvement. 21.5% arbitrary code execution on recently disclosed vulnerabilities against 5.5% for Sol, 99.7% on ExploitBench which OpenAI says contamination may have inflated, 1.50% misrepresentation in coding against 1.30% for Sol, 23.5% unwanted persistence after warnings against 17.4% for Astra, and 28 severity-3-or-higher flags in the same simulation"]
     ]},
    {"id": "spec-gpt61sol-6sol-6", "type": "spec_table", "title": "Coding",
     "columns": ["GPT-6 Sol", "GPT-6.1 Sol"],
     "rows": [
       ["DeepSWE v1.1 (OpenAI)", "68.8% (max)", "**75.2%** (high) / 71.9% (max)"],
       ["DeepSWE v1.1 (Artificial Analysis)", "Not reported", "**56.1%** (max)"],
       ["Terminal-Bench Science 0.1", "27.6% (max)", "**57.0%** (max)"],
       ["Terminal-Bench 4.0 (Artificial Analysis)", "43.9% (max)", "**56.1%** (max)"],
       ["FrontierCode 1.1 Main", "49.3% (max)", "Not reported"],
       ["SEC-Bench Pro (BenchLM)", "66.3%", "Not reported"]
     ]},
    {"id": "spec-gpt61sol-6sol-7", "type": "spec_table", "title": "Knowledge",
     "columns": ["GPT-6 Sol", "GPT-6.1 Sol"],
     "rows": [
       ["AA-Omniscience hallucination rate (lower is better)", "60.1%", "**54.3%**"]
     ]},
    {"id": "spec-gpt61sol-6sol-8", "type": "spec_table", "title": "Agentic & computer use",
     "columns": ["GPT-6 Sol", "GPT-6.1 Sol"],
     "rows": [
       ["OSWorld 2.0 offline, partial", "64.4% (max)", "**71.4%** (max)"],
       ["AutomationBench 1.0.6 (OpenAI)", "33.2% (xhigh)", "**36.1%** (max)"],
       ["AutomationBench-AA (Artificial Analysis)", "Not reported", "64.9% (max)"],
       ["GDPval-AA v2.1", "1487 Elo", "**1575 Elo**"],
       ["AA-Briefcase v1.1", "1483 Elo", "**1564 Elo**"],
       ["Agents' Last Exam V1", "56.6% (max)", "Not reported"]
     ]},
    {"id": "spec-gpt61sol-6sol-9", "type": "spec_table", "title": "Professional",
     "columns": ["GPT-6 Sol", "GPT-6.1 Sol"],
     "rows": [
       ["GDP.pdf", "28.0% (high)", "**32.0%** (high)"],
       ["HealthBench Professional", "Not reported", "64.2% (length-adjusted, within 0.5 points of Astra)"]
     ]},
    {"id": "markdown-gpt61sol-6sol-migration", "type": "markdown",
     "content": "## What breaks when you switch\n\nTwo things, and neither is a model-string change. First, reasoning is always on and the minimum setting is `low` rather than `none`, so a team that used GPT-6 Sol as a cheap deterministic function caller now pays reasoning tokens on every call it did not pay for before. Second, tool calling requires the Responses API, so Chat Completions callers that pass tools need an API migration, not just a new model id. Prompt caching offsets both for agent loops that resend the same context: at $0.10 per million cached tokens a long repository loop is materially cheaper per turn, though the first pass still bills cache writes at 1.25 times input. Batch and Flex halve prices for work that can wait, and fast mode doubles them while being unavailable under EU data residency."},
    {"id": "markdown-gpt61sol-6sol-bottom-line", "type": "markdown",
     "content": "## Bottom line\n\nGPT-6.1 Sol is a straightforward upgrade on the axis GPT-6 Sol was weak on. Per-token price is identical, so the headline comparison is flat, but the model is 6.4 points better on DeepSWE, roughly double on Terminal-Bench Science, 88 Elo better on GDPval-AA, and 31 percent cheaper per Artificial Analysis index task. The gains are real but concentrated: it still trails GPT-6 Astra on scientific and business-workflow work, trails both Anthropic models on the independent index, and actually regresses against GPT-6 Sol on SciCode and slightly on long-context reasoning. Switch for agentic coding, computer use and document-heavy professional work, and treat the always-on reasoning and the Responses API requirement as migration work rather than a version bump. Sweep effort levels against your own tasks before choosing a setting, because max is not the best setting here: DeepSWE peaks at high."}
  ]$blocks$::jsonb,
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing GPT-6.1 Sol', 'url', 'https://openai.com/index/introducing-gpt-6-1-sol/'),
    jsonb_build_object('title', 'GPT-6.1 Sol model page', 'url', 'https://developers.openai.com/api/docs/models/gpt-6.1-sol'),
    jsonb_build_object('title', 'Addendum to GPT-6 Astra System Card: GPT-6.1 Sol', 'url', 'https://deploymentsafety.openai.com/gpt-6-1-sol'),
    jsonb_build_object('title', 'GPT-6.1 Sol replaces GPT-6 Sol after just 7 days, with near-Astra intelligence', 'url', 'https://artificialanalysis.ai/articles/gpt-6-1-sol-replaces-gpt-6-sol-after-just-7-days-with-near-astra-intelligence'),
    jsonb_build_object('title', 'GPT-6 Sol - OpenAI Developers', 'url', 'https://developers.openai.com/api/docs/models/gpt-6-sol')
  ),
  jsonb_build_object(
    'modelA', 'gpt-6-1-sol',
    'modelB', 'gpt-6-sol',
    'verification_pass', 'gpt-6-1-sol-comparison-2026-09-29',
    'seoDescription', 'GPT-6.1 Sol vs GPT-6 Sol: same $2/$10 price, DeepSWE 75.2% vs 68.8%, cached input halved to $0.10, and what breaks when reasoning becomes mandatory.',
    'seoKeywords', jsonb_build_array(
      'GPT-6.1 Sol vs GPT-6 Sol', 'GPT-6.1 Sol', 'GPT-6 Sol', 'gpt-6.1-sol', 'gpt-6-sol',
      'GPT-6.1 Sol benchmarks', 'GPT-6.1 Sol DeepSWE', 'GPT-6.1 Sol cached input',
      'GPT-6.1 Sol reasoning effort', 'GPT-6.1 Sol migration', 'GPT-6 vs GPT-6.1', 'OpenAI GPT-6.1 Sol'
    )
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-09-29',
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
