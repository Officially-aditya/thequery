-- Enrich the GPT-6 Sol vs Claude Opus 5.5 comparison with the missing
-- official model-docs specs, keeping Sol vs Opus 5.5 two-column only.
--
-- Source: official OpenAI model compare + GPT-6 Sol detail page
-- (https://developers.openai.com/api/docs/models/compare?model=gpt-6-sol)
-- (https://developers.openai.com/api/docs/models/gpt-6-sol)
-- Opus 5.5 scores stay vendor-reported from the Anthropic launch table.
-- Scorecard stays 5 rows x 2 model columns (no Luna column).

WITH comparison_notes AS (
  SELECT $notes$
<small>*Benchmark scorecard with max-effort chart values for GPT-6 Sol next to Claude Opus 5.5 launch scores. Sol figures are vendor-reported by OpenAI at stated effort levels and Opus 5.5 figures are vendor-reported by Anthropic. Cross-vendor AutomationBench and OSWorld figures come from different runs and subsets so they are not head-to-head wins. Sol specs below are filled from the official OpenAI model docs.*</small>

The scorecard compares Sol and Opus 5.5 across five benchmarks. On AutomationBench 1.0.6 Sol reads 33.2 percent xhigh against Opus 5.5 at 40.0 percent on the separate Zapier early-access run. On Agents Last Exam V1 Sol reads 56.6 percent max with no Opus 5.5 launch figure. The launch text states 56.4 percent for Sol while the chart reads 56.6 percent and the scorecard uses the chart value.

On FrontierCode 1.1 Main Sol reads 49.3 percent max against Opus 5.5 at 54.4 percent max. This replaces the earlier undisclosed qualitative match with chart values and keeps the cost framing of Sol matching Fable 5.1 xhigh at much lower cost. On DeepSWE v1.1 Sol holds 68.8 percent max with no Opus 5.5 launch figure, within 1.1 points of Fable 5 at about 80 percent lower cost per task.

On OSWorld 2.0 Sol reads 64.4 percent max against Opus 5.5 at 81.8 percent partial on a different subset. The launch text separately reports Sol xhigh at 60.5 percent on the offline partial-reward subset v2026.08.08 against Opus 5 medium at 60.3 percent, and both readings are now stored as benchmark evidence. Treat every cross-vendor AutomationBench and OSWorld cell as a separate vendor report rather than a single leaderboard.

Specs are now filled from the official OpenAI model docs. Sol runs as gpt-6-sol at USD 2 input, USD 0.20 cached input, USD 2.50 cache writes and USD 10 output per million tokens, with a 1,050,000-token context window, 128,000 max output tokens and an Apr 20, 2026 knowledge cutoff. Input is Text and Image, output is Text only, and the Responses API supports web search, file search, image generation, code interpreter, hosted shell, apply patch, skills, computer use, MCP and tool search. Opus 5.5 runs at USD 4 input and USD 20 output with USD 0.20 cache reads.

Default to Opus 5.5 where the workload needs the broader verified launch table across Terminal-Bench 4.0, FrontierCode, CursorBench, Humanity's Last Exam with tools, GDPval-AA and OSWorld partial with third-party graders behind the headline rows. Default to GPT-6 Sol where per-task cost dominates and the workload matches the officially reported AutomationBench, DeepSWE and OSWorld patterns.
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
  'GPT-6 Sol vs Claude Opus 5.5: AutomationBench, Agents Last Exam, FrontierCode, DeepSWE and OSWorld chart values against Opus 5.5 verified launch evals.',
  comparison_notes.notes,
  $blocks$[
    {"id": "scorecard-gpt6sol-opus55", "type": "comparison_table", "title": "Benchmark scorecard",
     "caption": "GPT-6 Sol scores are vendor-reported by OpenAI at stated effort levels. Opus 5.5 scores are vendor-reported by Anthropic. AutomationBench and OSWorld runs differ across vendors so side-by-side cells are separate reports rather than a single leaderboard.",
     "columns": ["Category", "Benchmark", "GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Professional work", "AutomationBench 1.0.6", "33.2% (xhigh)", "40.0%"],
       ["Professional work", "Agents Last Exam V1", "56.6% (max)", "Not reported in Opus 5.5 launch table"],
       ["Coding", "FrontierCode 1.1 Main", "49.3% (max)", "54.4%"],
       ["Coding", "DeepSWE v1.1", "68.8% (max)", "Not reported in Opus 5.5 launch table"],
       ["Computer use", "OSWorld 2.0", "64.4% (max)", "81.8%"]
     ],
     "sourceNote": "OpenAI GPT-6 Sol launch post, official OpenAI model docs and Anthropic Opus 5.5 launch table. Cross-vendor AutomationBench and OSWorld figures come from different runs and subsets."},
    {"id": "spec-gpt6sol-opus55-1", "type": "spec_table", "title": "Specifications",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Developer", "OpenAI", "Anthropic"],
       ["Release date", "2026-09-22", "2026-09-22"],
       ["API model ID", "gpt-6-sol", "claude-opus-5-5"],
       ["Context window", "1,050,000 tokens", "1M tokens"],
       ["Max output", "128,000 tokens", "128K tokens (300K on Batch API beta)"],
       ["Knowledge cutoff", "Apr 20, 2026", "Jun 2026"],
       ["Reasoning / effort", "Adjustable effort - none, low, medium (default), high, xhigh, max - with cache-preserving controls", "Adaptive thinking (always on), medium default, per-message effort"]
     ]},
    {"id": "spec-gpt6sol-opus55-2", "type": "spec_table", "title": "Pricing",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Input / 1M tokens", "**$2**", "$4"],
       ["Cached input / 1M", "**$0.20** (10 percent of input rate)", "$0.20"],
       ["Cache write / 1M", "**$2.50** (1.25x input rate)", "$5 (5m) / $8 (1h)"],
       ["Output / 1M tokens", "**$10**", "$20"],
       ["Batch / flex discount", "Batch and Flex 50 percent off Standard rates - Fast mode 2x applicable rates", "Batch API: 50 percent off input and output"],
       ["Long-context surcharge", "Prompts over 272K input tokens: 2x input and cache rates plus 1.5x output - regional processing +10 percent where available", "None - standard pricing through 1M context"],
       ["Generation change", "50 percent cheaper than GPT-5.6 Sol promotional pricing", "20 percent below Opus 5 per token with 40 percent lower typical cost"]
     ]},
    {"id": "spec-gpt6sol-opus55-3", "type": "spec_table", "title": "Capabilities & access",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Text input", "Yes - Text input and output", "Yes"],
       ["Image / vision input", "Yes - Image input only", "Yes"],
       ["Audio input", "No - Not supported", "No native audio input documented"],
       ["Video input", "No - Not supported", "No native video input documented"],
       ["Text output", "Yes - Text output", "Yes"],
       ["Image output", "No - input only, no image output", "No"],
       ["Audio output", "No - Not supported", "No"],
       ["Video output", "No - Not supported", "No"],
       ["Tool / function calling", "Yes - Function calling plus Structured outputs plus Responses API tools (web search, file search, code interpreter, hosted shell, apply patch, skills, MCP, tool search)", "Yes"],
       ["Computer use", "Yes - Supported tool plus OSWorld 2.0 max-effort chart reading 64.4 percent", "Yes"],
       ["API access", "Yes - gpt-6-sol via v1/chat/completions, v1/responses, v1/batch", "Yes"],
       ["Product access", "ChatGPT Work plus Codex for Plus, Pro, Business, Enterprise and Edu with gradual rollout - not yet in Chat", "Claude plus Claude Code plus Claude API plus cloud partners"],
       ["Streaming", "Supported", "Yes"],
       ["Fine-tuning", "Not supported", "No documented fine-tuning"],
       ["Endpoints", "v1/chat/completions, v1/responses, v1/batch (plus Realtime, Assistants, Embeddings per model page)", "Claude API plus Bedrock, Vertex and Foundry endpoints"],
       ["Weights / license", "Proprietary", "Proprietary"]
     ]},
    {"id": "spec-gpt6sol-opus55-4", "type": "spec_table", "title": "Model behavior",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Primary focus", "Built to power complex coding and agentic workflows - lower-cost GPT-6 for professional work, coding, automation and computer use", "Long-running agentic coding and knowledge work at Fable-level quality"],
       ["Reasoning / speed label", "Highest reasoning, Fast speed per model page", "Adaptive thinking with 30 percent plus faster output than Opus 5"],
       ["AutomationBench 1.0.6 (OpenAI-reported)", "33.2 percent xhigh at $0.27 per task - above Astra low 30.3 percent at 3.9x cost and Opus 5 max 26.9 percent at 11.1x cost", "40.0 percent max on Zapier early-access run - different run from OpenAI 1.0.6 table so not directly comparable"],
       ["Agents Last Exam V1 (OpenAI-reported)", "56.6 percent max chart value - above Opus 5 highest at 60 percent lower cost per task", "Not reported in Opus 5.5 launch table"],
       ["Factuality (OpenAI internal eval)", "Sol halves predecessor mistakes and nears Astra reliability", "Not reported in Opus 5.5 launch table"],
       ["Efficiency story", "Lower token prices plus higher cache hits with 90 percent cached-read discount and GitHub over 50 percent fewer fresh tokens", "40 percent lower typical cost than Opus 5 with 30 percent plus faster output and fewer tokens per task"],
       ["Safety and alignment", "Builds on Astra alignment with lower misleading coding claims than GPT-5.6 counterparts", "Fable-5.1-class safeguards with verification programs and preserved thinking"],
       ["Collaboration style", "Astra communication style - clearer, less jargon, fewer low-value details, slightly shorter", "Most-important-first writing with less jargon and better style-rule following"]
     ]},
    {"id": "spec-gpt6sol-opus55-5", "type": "spec_table", "title": "Coding",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["DeepSWE v1.1 (OpenAI-reported)", "**68.8 percent** max - within 1.1 points of Fable 5 69.9 percent xhigh at about 80 percent lower cost", "Not reported in Opus 5.5 launch table"],
       ["FrontierCode 1.1 Main (OpenAI-reported)", "**49.3 percent** max chart value - matches Fable 5.1 xhigh at much lower cost and up substantially over GPT-5.6 Sol", "**54.4 percent** max"],
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
       ["OSWorld 2.0", "Max-effort chart reading **64.4 percent** - launch text separately reports 60.5 percent xhigh on the offline partial-reward subset v2026.08.08 vs Opus 5 medium 60.3 percent at about 80 percent lower cost", "**81.8 percent** partial max on different subset - not directly comparable"],
       ["GDPval-AA v2.1", "Not reported in OpenAI launch", "**1846 Elo**"],
       ["Caching for agents", "Higher hit rates by default plus effort and tool changes that preserve cache plus explicit breakpoints - cached reads at $0.20", "Cache reads at $0.20 with 60 percent cut from Opus 5 for long sessions that reread context"]
     ]},
    {"id": "markdown-gpt6sol-opus55-bottom-line", "type": "markdown",
     "content": "## Bottom line\n\nThe scorecard shows Sol and Opus 5.5 side by side with chart values for Sol. Pick Sol where per-task cost dominates and the work matches the AutomationBench, DeepSWE and OSWorld patterns in the OpenAI post. Pick Opus 5.5 where verified multi-hour coding across Terminal-Bench, FrontierCode, CursorBench, Humanity's Last Exam with tools, GDPval and OSWorld partial is the requirement. Treat cross-vendor AutomationBench and OSWorld cells as separate runs and keep effort-level and harness caveats in view."}
  ]$blocks$::jsonb,
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing GPT-6 Sol and Luna - OpenAI', 'url', 'https://openai.com/index/introducing-gpt-6-sol-and-luna/'),
    jsonb_build_object('title', 'GPT-6 Sol - OpenAI Developers', 'url', 'https://developers.openai.com/api/docs/models/gpt-6-sol'),
    jsonb_build_object('title', 'Compare models - GPT-6 Sol - OpenAI Developers', 'url', 'https://developers.openai.com/api/docs/models/compare?model=gpt-6-sol'),
    jsonb_build_object('title', 'OpenAI expands GPT-6 lineup with cheaper Sol and Luna models - Reuters', 'url', 'https://www.reuters.com/technology/openai-expands-gpt-6-lineup-with-cheaper-sol-luna-models-2026-09-22'),
    jsonb_build_object('title', 'Claude Opus 5.5', 'url', 'https://www.anthropic.com/claude-opus-5-5'),
    jsonb_build_object('title', 'Claude Opus 5.5 overview', 'url', 'https://platform.claude.com/docs/en/models/opus-5-5/overview'),
    jsonb_build_object('title', 'What is new in Claude Opus 5.5', 'url', 'https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5')
  ),
  jsonb_build_object(
    'modelA', 'gpt-6-sol',
    'modelB', 'claude-opus-5-5',
    'verification_pass', 'gpt6-sol-official-specs-2026-09-22'
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
