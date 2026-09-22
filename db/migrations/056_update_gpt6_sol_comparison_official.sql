-- Update the GPT-6 Sol vs Claude Opus 5.5 comparison with the official
-- OpenAI GPT-6 Sol and Luna launch post plus catalog refreshes for both models
--
-- Source: the official OpenAI launch post for GPT-6 Sol and Luna
-- (https://openai.com/index/introducing-gpt-6-sol-and-luna/)
-- which publishes pricing, availability, caching, alignment and vendor-reported
-- scores for AutomationBench 1.0.6, Agents Last Exam V1, DeepSWE v1.1,
-- FrontierCode 1.1 Main (qualitative), OSWorld 2.0 offline and factuality
-- Opus 5.5 scores stay vendor-reported from the Anthropic launch table
-- Cross-vendor AutomationBench and OSWorld figures come from different runs
-- and subsets so they are recorded as non-comparable runs

-- 1. Refresh GPT-6 Sol catalog entry with official launch details.
UPDATE models SET
  comparison_data = '{
    "Developer": "OpenAI",
    "Release date": "2026-09-22",
    "API model ID": "gpt-6-sol",
    "Input / 1M tokens": "$2",
    "Output / 1M tokens": "$10",
    "Cached input": "90 percent discount on cached reads",
    "API access": "Yes",
    "Product access": "ChatGPT Work and Codex (Plus, Pro, Business, Enterprise, Edu)",
    "Primary focus": "Lower-cost GPT-6 for professional work, coding, automation and computer use",
    "Weights / license": "Proprietary"
  }'::jsonb,
  sources = '[
    {"title": "Introducing GPT-6 Sol and Luna - OpenAI", "url": "https://openai.com/index/introducing-gpt-6-sol-and-luna/"},
    {"title": "OpenAI expands GPT-6 lineup with cheaper Sol and Luna models - Reuters", "url": "https://www.reuters.com/technology/openai-expands-gpt-6-lineup-with-cheaper-sol-luna-models-2026-09-22"}
  ]'::jsonb,
  notes = $notes$Launched September 22, 2026 alongside GPT-6 Luna as lower-cost GPT-6 options below the Astra flagship. API model gpt-6-sol in ChatGPT Work and Codex for Plus, Pro, Business, Enterprise and Edu with gradual rollout and not yet in Chat. Priced at half the GPT-5.6 Sol promotional rate with a 90 percent discount on cached input reads and cache-preserving reasoning effort and tool controls. Vendor-reported highlights include AutomationBench 1.0.6 at 33.2 percent xhigh for $0.27 per task, Agents Last Exam V1 at 56.4 percent max, DeepSWE v1.1 at 68.8 percent max within 1.1 points of Fable 5 at about 80 percent lower cost, FrontierCode 1.1 Main matching Fable 5.1 xhigh with scores undisclosed, OSWorld 2.0 offline at 60.5 percent xhigh and an internal factuality eval that halves predecessor mistakes while nearing Astra reliability.$notes$,
  metadata = COALESCE(metadata, '{}'::jsonb) || '{"verification_pass": "gpt6-sol-official-2026-09-22", "catalog_status": "current", "launch_disclosure": "official_launch_post_with_vendor_benchmarks"}'::jsonb,
  verified_at = NOW(),
  updated_at = NOW()
WHERE slug = 'gpt-6-sol';

-- 2. Refresh GPT-6 Luna catalog entry with official launch details.
UPDATE models SET
  comparison_data = '{
    "Developer": "OpenAI",
    "Release date": "2026-09-22",
    "API model ID": "gpt-6-luna",
    "Input / 1M tokens": "$0.10",
    "Output / 1M tokens": "$0.50",
    "Cached input": "90 percent discount on cached reads",
    "API access": "Yes",
    "Product access": "ChatGPT Work and Codex plus Luna on desktop for Free and Go users",
    "Primary focus": "Cheapest GPT-6 option for high-volume professional work, coding, automation and computer use",
    "Weights / license": "Proprietary"
  }'::jsonb,
  sources = '[
    {"title": "Introducing GPT-6 Sol and Luna - OpenAI", "url": "https://openai.com/index/introducing-gpt-6-sol-and-luna/"},
    {"title": "OpenAI expands GPT-6 lineup with cheaper Sol and Luna models - Reuters", "url": "https://www.reuters.com/technology/openai-expands-gpt-6-lineup-with-cheaper-sol-luna-models-2026-09-22"}
  ]'::jsonb,
  notes = $notes$Launched September 22, 2026 alongside GPT-6 Sol as the cheapest GPT-6 tier with API model gpt-6-luna. Vendor-reported highlights include DeepSWE v1.1 at 66.6 percent max described as comparable to Opus 5 and Fable 5 at medium effort with 93 to 96 percent lower cost per task, AutomationBench gains of 5.4 points at high effort with 58 percent lower cost per task, OSWorld 2.0 offline above GPT-5.6 Sol medium at one tenth the cost and an internal factuality eval that matches GPT-5.6 Sol at about one hundredth the cost at higher effort.$notes$,
  metadata = COALESCE(metadata, '{}'::jsonb) || '{"verification_pass": "gpt6-luna-official-2026-09-22", "catalog_status": "current", "launch_disclosure": "official_launch_post_with_vendor_benchmarks"}'::jsonb,
  verified_at = NOW(),
  updated_at = NOW()
WHERE slug = 'gpt-6-luna';

-- 3. Benchmark evidence from the official OpenAI GPT-6 Sol and Luna post.
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
  {"id":"tq-20260922-gpt6sol-automationbench106","model_slug":"gpt-6-sol","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":"1.0.6","score_numeric":33.2,"score_display":"33.2%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"47-tool business workflows","evaluator":"OpenAI","evaluation_date":"2026-09-22","source":"https://openai.com/index/introducing-gpt-6-sol-and-luna/","notes":"Self-reported at xhigh effort for $0.27 per task, above Astra low at 30.3 percent and Opus 5 max at 26.9 percent in the same table with Fable 5.1 with Opus 5 fallback at 31.4 percent and fallback cost excluded"},
  {"id":"tq-20260922-gpt6sol-agentslastexamv1","model_slug":"gpt-6-sol","category":"professional","benchmark_name":"Agents Last Exam","benchmark_version":"V1","score_numeric":56.4,"score_display":"56.4%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"long-horizon professional workflows","evaluator":"OpenAI","evaluation_date":"2026-09-22","source":"https://openai.com/index/introducing-gpt-6-sol-and-luna/","notes":"Self-reported at max effort, above the Opus 5 highest score in the same table at 60 percent lower cost per task"},
  {"id":"tq-20260922-gpt6sol-deepswe11","model_slug":"gpt-6-sol","category":"coding","benchmark_name":"DeepSWE","benchmark_version":"1.1","score_numeric":68.8,"score_display":"68.8%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"long-horizon software engineering tasks","evaluator":"OpenAI","evaluation_date":"2026-09-22","source":"https://openai.com/index/introducing-gpt-6-sol-and-luna/","notes":"Self-reported at max effort, within 1.1 points of Fable 5 at 69.9 percent xhigh at about 80 percent lower cost per task"},
  {"id":"tq-20260922-gpt6luna-deepswe11","model_slug":"gpt-6-luna","category":"coding","benchmark_name":"DeepSWE","benchmark_version":"1.1","score_numeric":66.6,"score_display":"66.6%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"long-horizon software engineering tasks","evaluator":"OpenAI","evaluation_date":"2026-09-22","source":"https://openai.com/index/introducing-gpt-6-sol-and-luna/","notes":"Self-reported at max effort, described as comparable to Opus 5 and Fable 5 at medium effort with 93 to 96 percent lower cost per task"},
  {"id":"tq-20260922-gpt6sol-osworld20offline","model_slug":"gpt-6-sol","category":"agentic_computer_use","benchmark_name":"OSWorld 2.0","benchmark_version":"offline","score_numeric":60.5,"score_display":"60.5%","score_unit":"percent","tools":true,"reasoning_effort":"xhigh","harness":"offline computer-use workflows","evaluator":"OpenAI","evaluation_date":"2026-09-22","source":"https://openai.com/index/introducing-gpt-6-sol-and-luna/","notes":"Self-reported partial reward on the offline set v2026.08.08 at xhigh effort, against Opus 5 medium at 60.3 percent in the same table at about 80 percent lower cost per task"}
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

-- 4. Updated comparison page: GPT-6 Sol vs Claude Opus 5.5.
WITH comparison_notes AS (
  SELECT $notes$
<small>*Updated with the official OpenAI GPT-6 Sol and Luna launch post (September 22, 2026). Sol scores below are vendor-reported at stated effort levels with cost-per-task framing. Opus 5.5 scores are vendor-reported from the Anthropic launch table at max effort except Terminal-Bench 4.0 at xhigh. Cross-vendor AutomationBench and OSWorld figures come from different runs and subsets so they are not head-to-head wins.*</small>

OpenAI and Anthropic shipped on the same day from opposite directions. OpenAI expanded the GPT-6 lineup downward with Sol at USD 2 input and USD 10 output per million tokens, half the GPT-5.6 Sol promotional rate, while holding Astra as the flagship for demanding projects. Anthropic moved the Opus line upward in capability while cutting its price, with Opus 5.5 at USD 4 input and USD 20 output performing at Fable 5.1 level on most work for roughly 60 percent lower operating cost than Opus 5.

The official OpenAI post replaces the launch-day pricing-only picture with API model IDs, availability, caching, alignment and vendor-reported scores. Sol and Luna run as gpt-6-sol and gpt-6-luna in ChatGPT Work and Codex for Plus, Pro, Business, Enterprise and Edu, with Luna also on desktop for Free and Go users and neither model in Chat yet. Caching brings higher hit rates by default with a 90 percent discount on cached reads, plus reasoning-effort and tool-availability changes that preserve cache reuse and explicit breakpoints for prefix control. GitHub reports over 50 percent fewer prompt tokens needing fresh processing across billions of requests. Alignment builds on Astra with lower misleading coding claims than GPT-5.6 counterparts, measured on deliberately challenging cases rather than typical use.

On professional work OpenAI reports Sol xhigh at 33.2 percent on AutomationBench 1.0.6 for USD 0.27 per task, above Astra low at 30.3 percent at 3.9 times the cost and Opus 5 max at 26.9 percent at 11.1 times the cost, with Fable 5.1 plus Opus 5 fallback at 31.4 percent and fallback cost excluded. Luna at high effort improves 5.4 points over its predecessor at 58 percent lower cost per task. On Agents Last Exam V1 Sol max reaches 56.4 percent, above the Opus 5 highest score at 60 percent lower cost. Factuality comes from an internal eval built on de-identified conversations where users had flagged mistakes, with Sol making about half as many mistakes as its predecessor and nearing Astra reliability while Luna at higher effort matches GPT-5.6 Sol at about one hundredth the cost.

Coding and computer use follow the same cost-per-task framing. On DeepSWE v1.1 Sol max scores 68.8 percent, within 1.1 points of Fable 5 at 69.9 percent xhigh at about 80 percent lower cost, while Luna max at 66.6 percent is described as comparable to Opus 5 and Fable 5 at medium effort with 93 to 96 percent lower cost. On FrontierCode 1.1 Main Sol matches Fable 5.1 xhigh at much lower cost after a substantial gain over GPT-5.6 Sol, with scores undisclosed. On OSWorld 2.0 offline Sol xhigh reaches 60.5 percent against Opus 5 medium at 60.3 percent at about 80 percent lower cost on the partial-reward offline set, while Luna max exceeds GPT-5.6 Sol medium at one tenth the cost. These OSWorld and AutomationBench runs differ from the Anthropic launch runs for Opus 5.5, so treat the side-by-side cells as separate vendor reports rather than a single leaderboard.

Default to Opus 5.5 where the workload needs the broader verified launch table across Terminal-Bench 4.0, FrontierCode, CursorBench, Humanity's Last Exam with tools, GDPval-AA and OSWorld partial with third-party graders behind the headline rows. Default to GPT-6 Sol where per-task cost dominates and the workload matches the officially reported AutomationBench, DeepSWE and OSWorld offline patterns. Revisit the remaining blank Sol cells if OpenAI publishes Terminal-Bench, CursorBench, Humanity's Last Exam or GDPval numbers for Sol itself.
$notes$::text AS notes
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'comparison:gpt-6-sol-vs-opus-5-5',
  'comparison',
  'gpt-6-sol-vs-opus-5-5',
  '',
  'comparisons/gpt-6-sol-vs-opus-5-5',
  'GPT-6 Sol vs Claude Opus 5.5',
  'GPT-6 Sol vs Claude Opus 5.5: official Sep 22, 2026 OpenAI benchmarks put Sol at half the Opus per-token price with AutomationBench, DeepSWE and OSWorld cost-per-task wins against Opus 5.5 verified coding evals.',
  comparison_notes.notes,
  $blocks$[
    {"id": "spec-gpt6sol-opus55-1", "type": "spec_table", "title": "Specifications",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Developer", "OpenAI", "Anthropic"],
       ["Release date", "2026-09-22", "2026-09-22"],
       ["API model ID", "gpt-6-sol", "claude-opus-5-5"],
       ["Context window", "Not disclosed in OpenAI launch", "1M tokens"],
       ["Max output", "Not disclosed in OpenAI launch", "128K tokens (300K on Batch API beta)"],
       ["Knowledge cutoff", "Not disclosed in OpenAI launch", "Jun 2026"],
       ["Reasoning / effort", "Adjustable effort (xhigh, high, max and low reported) with cache-preserving controls", "Adaptive thinking (always on), medium default, per-message effort"]
     ]},
    {"id": "spec-gpt6sol-opus55-2", "type": "spec_table", "title": "Pricing",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Input / 1M tokens", "**$2**", "$4"],
       ["Cached input / 1M", "90 percent discount on cached reads", "$0.20"],
       ["Cache write / 1M", "Not disclosed in OpenAI launch", "$5 (5m) / $8 (1h)"],
       ["Output / 1M tokens", "**$10**", "$20"],
       ["Batch / flex discount", "Not disclosed in OpenAI launch", "Batch API: 50 percent off input and output"],
       ["Long-context surcharge", "Not disclosed in OpenAI launch", "None - standard pricing through 1M context"],
       ["Generation change", "50 percent cheaper than GPT-5.6 Sol promotional pricing", "20 percent below Opus 5 per token with 40 percent lower typical cost"]
     ]},
    {"id": "spec-gpt6sol-opus55-3", "type": "spec_table", "title": "Capabilities & access",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Text input", "Yes", "Yes"],
       ["Image / vision input", "Not itemized in OpenAI launch", "Yes"],
       ["Audio input", "Not itemized in OpenAI launch", "No native audio input documented"],
       ["Video input", "Not itemized in OpenAI launch", "No native video input documented"],
       ["Text output", "Yes", "Yes"],
       ["Image output", "Not itemized in OpenAI launch", "No"],
       ["Audio output", "Not itemized in OpenAI launch", "No"],
       ["Video output", "Not itemized in OpenAI launch", "No"],
       ["Tool / function calling", "Yes - 47-tool AutomationBench workflows plus coding and computer-use agents", "Yes"],
       ["Computer use", "Yes - OSWorld 2.0 offline 60.5 percent at xhigh effort", "Yes"],
       ["API access", "Yes - gpt-6-sol", "Yes"],
       ["Product access", "ChatGPT Work plus Codex for Plus, Pro, Business, Enterprise and Edu with gradual rollout - not yet in Chat", "Claude plus Claude Code plus Claude API plus cloud partners"],
       ["Weights / license", "Proprietary", "Proprietary"]
     ]},
    {"id": "spec-gpt6sol-opus55-4", "type": "spec_table", "title": "Model behavior",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Primary focus", "Lower-cost GPT-6 for professional work, coding, automation and computer use", "Long-running agentic coding and knowledge work at Fable-level quality"],
       ["AutomationBench 1.0.6 (OpenAI-reported)", "33.2 percent xhigh at $0.27 per task - above Astra low 30.3 percent at 3.9x cost and Opus 5 max 26.9 percent at 11.1x cost", "40.0 percent max on Zapier early-access run - different run from OpenAI 1.0.6 table so not directly comparable"],
       ["Agents Last Exam V1 (OpenAI-reported)", "56.4 percent max - above Opus 5 highest at 60 percent lower cost per task", "Not reported in Opus 5.5 launch table"],
       ["Factuality (OpenAI internal eval)", "Sol halves predecessor mistakes and nears Astra reliability - Luna matches GPT-5.6 Sol at about one hundredth the cost", "Not reported in Opus 5.5 launch table"],
       ["Efficiency story", "Lower token prices plus higher cache hits with 90 percent cached-read discount and GitHub over 50 percent fewer fresh tokens", "40 percent lower typical cost than Opus 5 with 30 percent plus faster output and fewer tokens per task"],
       ["Safety and alignment", "Builds on Astra alignment with lower misleading coding claims than GPT-5.6 counterparts", "Fable-5.1-class safeguards with verification programs and preserved thinking"],
       ["Collaboration style", "Astra communication style - clearer, less jargon, fewer low-value details, slightly shorter", "Most-important-first writing with less jargon and better style-rule following"]
     ]},
    {"id": "spec-gpt6sol-opus55-5", "type": "spec_table", "title": "Coding",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["DeepSWE v1.1 (OpenAI-reported)", "**68.8 percent** max - within 1.1 points of Fable 5 69.9 percent xhigh at about 80 percent lower cost", "Not reported in Opus 5.5 launch table"],
       ["FrontierCode 1.1 Main (OpenAI-reported)", "Matches Fable 5.1 xhigh at much lower cost and up substantially over GPT-5.6 Sol - scores not disclosed", "**54.4 percent** max"],
       ["Terminal-Bench 4.0", "Not reported in OpenAI launch", "**66.4 percent** (xhigh)"],
       ["CursorBench 4.0", "Not reported in OpenAI launch", "**57.8 percent**"],
       ["Terminal-Bench Science 0.1", "Not reported in OpenAI launch", "**58.7 percent**"]
     ]},
    {"id": "spec-gpt6sol-opus55-6", "type": "spec_table", "title": "Knowledge",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Humanity's Last Exam", "Not reported in OpenAI launch - see internal factuality eval in behavior", "**67.7 percent** with tools"]
     ]},
    {"id": "spec-gpt6sol-opus55-7", "type": "spec_table", "title": "Agentic & computer use",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["AutomationBench", "OpenAI 1.0.6 run - **33.2 percent** xhigh at $0.27 (Fable 5.1 with Opus 5 fallback 31.4 percent at over 8.9x cost, fallback cost excluded)", "**40.0 percent** max on Zapier early-access run with no fallbacks where safeguard interventions counted as failures"],
       ["OSWorld 2.0", "Offline subset partial reward v2026.08.08 - **60.5 percent** xhigh vs Opus 5 medium 60.3 percent at about 80 percent lower cost", "**81.8 percent** partial max on different subset - not directly comparable"],
       ["GDPval-AA v2.1", "Not reported in OpenAI launch", "**1846 Elo**"],
       ["Caching for agents", "Higher hit rates by default plus effort and tool changes that preserve cache plus explicit breakpoints", "Cache reads at $0.20 with 60 percent cut from Opus 5 for long sessions that reread context"]
     ]},
    {"id": "markdown-gpt6sol-opus55-bottom-line", "type": "markdown",
     "content": "## Bottom line\n\nSol now carries official vendor-reported numbers on selected professional, coding and computer-use tasks with cost-per-task framing, while Opus 5.5 carries the broader third-party-graded launch table. Pick Sol where per-task cost dominates and the work matches the AutomationBench, DeepSWE and OSWorld offline patterns in the OpenAI post. Pick Opus 5.5 where verified multi-hour coding across Terminal-Bench, FrontierCode, CursorBench, Humanity's Last Exam with tools, GDPval and OSWorld partial is the requirement. Treat cross-vendor AutomationBench and OSWorld cells as separate runs and keep effort-level and harness caveats in view."}
  ]$blocks$::jsonb,
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing GPT-6 Sol and Luna - OpenAI', 'url', 'https://openai.com/index/introducing-gpt-6-sol-and-luna/'),
    jsonb_build_object('title', 'OpenAI expands GPT-6 lineup with cheaper Sol and Luna models - Reuters', 'url', 'https://www.reuters.com/technology/openai-expands-gpt-6-lineup-with-cheaper-sol-luna-models-2026-09-22'),
    jsonb_build_object('title', 'Claude Opus 5.5', 'url', 'https://www.anthropic.com/claude-opus-5-5'),
    jsonb_build_object('title', 'Claude Opus 5.5 overview', 'url', 'https://platform.claude.com/docs/en/models/opus-5-5/overview'),
    jsonb_build_object('title', 'What is new in Claude Opus 5.5', 'url', 'https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5')
  ),
  jsonb_build_object(
    'modelA', 'gpt-6-sol',
    'modelB', 'claude-opus-5-5',
    'verification_pass', 'gpt6-sol-official-2026-09-22'
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
