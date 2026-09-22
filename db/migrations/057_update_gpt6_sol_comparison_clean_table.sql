-- Update the GPT-6 Sol vs Claude Opus 5.5 comparison with the clean
-- three-model benchmark scorecard (GPT-6 Sol, GPT-6 Luna, Claude Opus 5.5)
--
-- Source: the official OpenAI GPT-6 Sol and Luna launch post
-- (https://openai.com/index/introducing-gpt-6-sol-and-luna/)
-- Opus 5.5 scores stay vendor-reported from the Anthropic launch table
-- Cross-vendor AutomationBench and OSWorld figures come from different runs
-- and subsets so they are recorded as non-comparable runs
--
-- Why a comparison_table block: spec_table blocks render exactly two model
-- columns in the UI and normalizeBlocks truncates spec columns to two, so the
-- three-model scorecard uses the comparison_table block type which supports
-- N columns while the spec tables stay two-column Sol vs Opus 5.5

-- 1. Benchmark evidence matching the clean scorecard.
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
  {"id":"tq-20260922-gpt6sol-agentslastexamv1","model_slug":"gpt-6-sol","category":"professional","benchmark_name":"Agents Last Exam","benchmark_version":"V1","score_numeric":56.6,"score_display":"56.6%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"long-horizon professional workflows","evaluator":"OpenAI","evaluation_date":"2026-09-22","source":"https://openai.com/index/introducing-gpt-6-sol-and-luna/","notes":"Self-reported at max effort, above the Opus 5 highest score in the same table at 60 percent lower cost per task. Launch text states 56.4 percent and the chart reads 56.6 percent, scorecard uses the chart value"},
  {"id":"tq-20260922-gpt6luna-automationbench106","model_slug":"gpt-6-luna","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":"1.0.6","score_numeric":20.7,"score_display":"20.7%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"47-tool business workflows","evaluator":"OpenAI","evaluation_date":"2026-09-22","source":"https://openai.com/index/introducing-gpt-6-sol-and-luna/","notes":"Self-reported max-effort chart reading. Launch text describes Luna gains of 5.4 points at high effort with 58 percent lower cost per task"},
  {"id":"tq-20260922-gpt6luna-agentslastexamv1","model_slug":"gpt-6-luna","category":"professional","benchmark_name":"Agents Last Exam","benchmark_version":"V1","score_numeric":50.9,"score_display":"50.9%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"long-horizon professional workflows","evaluator":"OpenAI","evaluation_date":"2026-09-22","source":"https://openai.com/index/introducing-gpt-6-sol-and-luna/","notes":"Self-reported max-effort chart reading"},
  {"id":"tq-20260922-gpt6sol-frontiercode11main","model_slug":"gpt-6-sol","category":"coding","benchmark_name":"FrontierCode","benchmark_version":"1.1 Main","score_numeric":49.3,"score_display":"49.3%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"merge-ready code changes in real codebases","evaluator":"OpenAI","evaluation_date":"2026-09-22","source":"https://openai.com/index/introducing-gpt-6-sol-and-luna/","notes":"Self-reported max-effort chart reading, described as matching Fable 5.1 xhigh at much lower cost with a substantial gain over GPT-5.6 Sol"},
  {"id":"tq-20260922-gpt6luna-frontiercode11main","model_slug":"gpt-6-luna","category":"coding","benchmark_name":"FrontierCode","benchmark_version":"1.1 Main","score_numeric":42.4,"score_display":"42.4%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"merge-ready code changes in real codebases","evaluator":"OpenAI","evaluation_date":"2026-09-22","source":"https://openai.com/index/introducing-gpt-6-sol-and-luna/","notes":"Self-reported max-effort chart reading"},
  {"id":"tq-20260922-gpt6sol-osworld20-max","model_slug":"gpt-6-sol","category":"agentic_computer_use","benchmark_name":"OSWorld 2.0","benchmark_version":null,"score_numeric":64.4,"score_display":"64.4%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"long-horizon computer-use workflows","evaluator":"OpenAI","evaluation_date":"2026-09-22","source":"https://openai.com/index/introducing-gpt-6-sol-and-luna/","notes":"Self-reported max-effort chart reading. Launch text separately reports 60.5 percent xhigh on the offline partial-reward subset v2026.08.08 against Opus 5 medium at 60.3 percent"},
  {"id":"tq-20260922-gpt6luna-osworld20-max","model_slug":"gpt-6-luna","category":"agentic_computer_use","benchmark_name":"OSWorld 2.0","benchmark_version":null,"score_numeric":52.7,"score_display":"52.7%","score_unit":"percent","tools":true,"reasoning_effort":"max","harness":"long-horizon computer-use workflows","evaluator":"OpenAI","evaluation_date":"2026-09-22","source":"https://openai.com/index/introducing-gpt-6-sol-and-luna/","notes":"Self-reported max-effort chart reading. Launch text separately describes Luna max exceeding GPT-5.6 Sol medium at one tenth the cost"}
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

-- 2. Updated comparison page with the three-model scorecard on top.
WITH comparison_notes AS (
  SELECT $notes$
<small>*Benchmark scorecard with max-effort chart values for GPT-6 Sol and Luna next to Claude Opus 5.5 launch scores. Sol and Luna figures are vendor-reported by OpenAI at stated effort levels and Opus 5.5 figures are vendor-reported by Anthropic. Cross-vendor AutomationBench and OSWorld figures come from different runs and subsets so they are not head-to-head wins.*</small>

The scorecard adds GPT-6 Luna next to Sol and Opus 5.5 across five benchmarks. On AutomationBench 1.0.6 Sol reads 33.2 percent xhigh against Luna 20.7 percent max and Opus 5.5 at 40.0 percent on the separate Zapier early-access run. On Agents Last Exam V1 Sol reads 56.6 percent max with Luna at 50.9 percent max and no Opus 5.5 launch figure. The launch text states 56.4 percent for Sol while the chart reads 56.6 percent and the scorecard uses the chart value.

On FrontierCode 1.1 Main Sol reads 49.3 percent max with Luna at 42.4 percent max against Opus 5.5 at 54.4 percent max. This replaces the earlier undisclosed qualitative match with chart values and keeps the cost framing of Sol matching Fable 5.1 xhigh at much lower cost. On DeepSWE v1.1 Sol holds 68.8 percent max with Luna at 66.6 percent max and no Opus 5.5 launch figure, within 1.1 points of Fable 5 at about 80 percent lower cost per task.

On OSWorld 2.0 Sol reads 64.4 percent max with Luna at 52.7 percent max against Opus 5.5 at 81.8 percent partial on a different subset. The launch text separately reports Sol xhigh at 60.5 percent on the offline partial-reward subset v2026.08.08 against Opus 5 medium at 60.3 percent, and both readings are now stored as benchmark evidence. Treat every cross-vendor AutomationBench and OSWorld cell as a separate vendor report rather than a single leaderboard.

Pricing is unchanged from the official post. Sol runs as gpt-6-sol at USD 2 input and USD 10 output per million tokens in ChatGPT Work and Codex for Plus, Pro, Business, Enterprise and Edu with a 90 percent discount on cached reads. Luna runs as gpt-6-luna at USD 0.10 input and USD 0.50 output with the same cached-read discount and Luna on desktop for Free and Go users. Opus 5.5 runs at USD 4 input and USD 20 output with USD 0.20 cache reads.

Default to Opus 5.5 where the workload needs the broader verified launch table across Terminal-Bench 4.0, FrontierCode, CursorBench, Humanity's Last Exam with tools, GDPval-AA and OSWorld partial with third-party graders behind the headline rows. Default to GPT-6 Sol where per-task cost dominates and the workload matches the officially reported AutomationBench, DeepSWE and OSWorld patterns. Default to GPT-6 Luna where high-volume inference dominates and the 50.9 percent Agents Last Exam, 42.4 percent FrontierCode and 52.7 percent OSWorld max-effort readings clear the quality bar.
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
  'GPT-6 Sol vs Claude Opus 5.5 with a three-model scorecard adding GPT-6 Luna: AutomationBench, Agents Last Exam, FrontierCode, DeepSWE and OSWorld chart values against Opus 5.5 verified launch evals.',
  comparison_notes.notes,
  $blocks$[
    {"id": "scorecard-gpt6sol-luna-opus55", "type": "comparison_table", "title": "Benchmark scorecard",
     "caption": "GPT-6 Sol and Luna scores are vendor-reported by OpenAI at stated effort levels. Opus 5.5 scores are vendor-reported by Anthropic. AutomationBench and OSWorld runs differ across vendors so side-by-side cells are separate reports rather than a single leaderboard.",
     "columns": ["Category", "Benchmark", "GPT-6 Sol", "GPT-6 Luna", "Claude Opus 5.5"],
     "rows": [
       ["Professional work", "AutomationBench 1.0.6", "33.2% (xhigh)", "20.7% (max)", "40.0%"],
       ["Professional work", "Agents Last Exam V1", "56.6% (max)", "50.9% (max)", "Not reported in Opus 5.5 launch table"],
       ["Coding", "FrontierCode 1.1 Main", "49.3% (max)", "42.4% (max)", "54.4%"],
       ["Coding", "DeepSWE v1.1", "68.8% (max)", "66.6% (max)", "Not reported in Opus 5.5 launch table"],
       ["Computer use", "OSWorld 2.0", "64.4% (max)", "52.7% (max)", "81.8%"]
     ],
     "sourceNote": "OpenAI GPT-6 Sol and Luna launch post and Anthropic Opus 5.5 launch table. Cross-vendor AutomationBench and OSWorld figures come from different runs and subsets."},
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
       ["Computer use", "Yes - OSWorld 2.0 max-effort chart reading 64.4 percent", "Yes"],
       ["API access", "Yes - gpt-6-sol", "Yes"],
       ["Product access", "ChatGPT Work plus Codex for Plus, Pro, Business, Enterprise and Edu with gradual rollout - not yet in Chat", "Claude plus Claude Code plus Claude API plus cloud partners"],
       ["Weights / license", "Proprietary", "Proprietary"]
     ]},
    {"id": "spec-gpt6sol-opus55-4", "type": "spec_table", "title": "Model behavior",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Primary focus", "Lower-cost GPT-6 for professional work, coding, automation and computer use", "Long-running agentic coding and knowledge work at Fable-level quality"],
       ["AutomationBench 1.0.6 (OpenAI-reported)", "33.2 percent xhigh at $0.27 per task - above Astra low 30.3 percent at 3.9x cost and Opus 5 max 26.9 percent at 11.1x cost (Luna 20.7 percent max, see scorecard)", "40.0 percent max on Zapier early-access run - different run from OpenAI 1.0.6 table so not directly comparable"],
       ["Agents Last Exam V1 (OpenAI-reported)", "56.6 percent max chart value - above Opus 5 highest at 60 percent lower cost per task (Luna 50.9 percent max, see scorecard)", "Not reported in Opus 5.5 launch table"],
       ["Factuality (OpenAI internal eval)", "Sol halves predecessor mistakes and nears Astra reliability - Luna matches GPT-5.6 Sol at about one hundredth the cost", "Not reported in Opus 5.5 launch table"],
       ["Efficiency story", "Lower token prices plus higher cache hits with 90 percent cached-read discount and GitHub over 50 percent fewer fresh tokens", "40 percent lower typical cost than Opus 5 with 30 percent plus faster output and fewer tokens per task"],
       ["Safety and alignment", "Builds on Astra alignment with lower misleading coding claims than GPT-5.6 counterparts", "Fable-5.1-class safeguards with verification programs and preserved thinking"],
       ["Collaboration style", "Astra communication style - clearer, less jargon, fewer low-value details, slightly shorter", "Most-important-first writing with less jargon and better style-rule following"]
     ]},
    {"id": "spec-gpt6sol-opus55-5", "type": "spec_table", "title": "Coding",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["DeepSWE v1.1 (OpenAI-reported)", "**68.8 percent** max - within 1.1 points of Fable 5 69.9 percent xhigh at about 80 percent lower cost (Luna **66.6 percent** max, see scorecard)", "Not reported in Opus 5.5 launch table"],
       ["FrontierCode 1.1 Main (OpenAI-reported)", "**49.3 percent** max chart value - matches Fable 5.1 xhigh at much lower cost and up substantially over GPT-5.6 Sol (Luna **42.4 percent** max, see scorecard)", "**54.4 percent** max"],
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
       ["OSWorld 2.0", "Max-effort chart reading **64.4 percent** (Luna **52.7 percent** max, see scorecard) - launch text separately reports 60.5 percent xhigh on the offline partial-reward subset v2026.08.08 vs Opus 5 medium 60.3 percent at about 80 percent lower cost", "**81.8 percent** partial max on different subset - not directly comparable"],
       ["GDPval-AA v2.1", "Not reported in OpenAI launch", "**1846 Elo**"],
       ["Caching for agents", "Higher hit rates by default plus effort and tool changes that preserve cache plus explicit breakpoints", "Cache reads at $0.20 with 60 percent cut from Opus 5 for long sessions that reread context"]
     ]},
    {"id": "markdown-gpt6sol-opus55-bottom-line", "type": "markdown",
     "content": "## Bottom line\n\nThe scorecard now shows all three models side by side with chart values for Sol and Luna. Pick Sol where per-task cost dominates and the work matches the AutomationBench, DeepSWE and OSWorld patterns in the OpenAI post. Pick Luna where high-volume inference dominates and the 50.9 percent Agents Last Exam, 42.4 percent FrontierCode and 52.7 percent OSWorld max-effort readings clear the quality bar. Pick Opus 5.5 where verified multi-hour coding across Terminal-Bench, FrontierCode, CursorBench, Humanity's Last Exam with tools, GDPval and OSWorld partial is the requirement. Treat cross-vendor AutomationBench and OSWorld cells as separate runs and keep effort-level and harness caveats in view."}
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
    'verification_pass', 'gpt6-sol-clean-table-2026-09-22'
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
