-- Add GPT-6 Sol and GPT-6 Luna to the model catalog and publish the
-- authored GPT-6 Sol vs Claude Opus 5.5 comparison page.
--
-- Source: Reuters reporting on the September 22, 2026 launch
-- (https://www.reuters.com/technology/openai-expands-gpt-6-lineup-with-cheaper-sol-luna-models-2026-09-22).
-- Launch-day disclosure covers pricing and positioning only. OpenAI published
-- no benchmark table for Sol or Luna, so this migration records no benchmark
-- evidence for either model and the comparison states that explicitly.

-- 1. Model catalog entries for GPT-6 Sol and GPT-6 Luna.
INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at
)
SELECT
  entry.slug,
  entry.name,
  'OpenAI',
  DATE '2026-09-22',
  DATE '2026-09-22',
  'proprietary',
  'GPT-6',
  entry.comparison_data,
  entry.sources,
  entry.notes,
  '{"verification_pass": "gpt6-sol-luna-launch-2026-09-22", "catalog_status": "current", "launch_disclosure": "pricing_and_positioning_only"}'::jsonb,
  NOW()
FROM (VALUES (
  'gpt-6-sol',
  'GPT-6 Sol',
  '{
    "Developer": "OpenAI",
    "Release date": "2026-09-22",
    "Input / 1M tokens": "$2",
    "Output / 1M tokens": "$10",
    "Primary focus": "Lower-cost GPT-6 option for professional work, coding, automation, and computer-use tasks",
    "Weights / license": "Proprietary"
  }'::jsonb,
  '[{"title": "OpenAI expands GPT-6 lineup with cheaper Sol and Luna models", "url": "https://www.reuters.com/technology/openai-expands-gpt-6-lineup-with-cheaper-sol-luna-models-2026-09-22"}]'::jsonb,
  'Launched September 22, 2026 alongside GPT-6 Luna as lower-cost GPT-6 options below the Astra flagship. Priced at half the GPT-5.6 Sol promotional rate. OpenAI says Sol and Luna were trained with similar methods to Astra with gains in reasoning, factual reliability, coding, computer use, and alignment, while Astra remains the most capable model for demanding projects. No vendor benchmark table published at launch.'
), (
  'gpt-6-luna',
  'GPT-6 Luna',
  '{
    "Developer": "OpenAI",
    "Release date": "2026-09-22",
    "Input / 1M tokens": "$0.10",
    "Output / 1M tokens": "$0.50",
    "Primary focus": "Cheapest GPT-6 option for high-volume professional work, coding, automation, and computer-use tasks",
    "Weights / license": "Proprietary"
  }'::jsonb,
  '[{"title": "OpenAI expands GPT-6 lineup with cheaper Sol and Luna models", "url": "https://www.reuters.com/technology/openai-expands-gpt-6-lineup-with-cheaper-sol-luna-models-2026-09-22"}]'::jsonb,
  'Launched September 22, 2026 alongside GPT-6 Sol as the cheapest GPT-6 tier. OpenAI attributes the pricing to caching and inference gains and says the savings are passed directly to users. Positioned below Sol the way Luna sat below Sol in the GPT-5.6 family. No vendor benchmark table published at launch.'
)) AS entry(slug, name, comparison_data, sources, notes)
WHERE NOT EXISTS (SELECT 1 FROM models WHERE slug = entry.slug)
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

-- 2. Authored comparison page: GPT-6 Sol vs Claude Opus 5.5.
WITH comparison_notes AS (
  SELECT $notes$
<small>*Launch-day comparison (September 22, 2026). OpenAI published pricing and positioning for GPT-6 Sol but no benchmark table, so every Sol benchmark cell below reads as undisclosed rather than zero. Opus 5.5 scores are vendor-reported from the Anthropic launch table, with Opus 5.5 at max effort except Terminal-Bench 4.0 at xhigh.*</small>

OpenAI and Anthropic shipped on the same day from opposite directions. OpenAI expanded the GPT-6 lineup downward with Sol at USD 2 input and USD 10 output per million tokens, half the GPT-5.6 Sol promotional rate, while holding Astra as the flagship for demanding projects. Anthropic moved the Opus line upward in capability while cutting its price, with Opus 5.5 at USD 4 input and USD 20 output performing at Fable 5.1 level on most work for roughly 60 percent lower operating cost than Opus 5.

Per-token price is the one clean win in this comparison and it belongs to Sol. At list rates Sol bills half of Opus 5.5 on both input and output, and Luna undercuts both at USD 0.10 input and USD 0.50 output for high-volume work. What that discount does not yet include is evidence. OpenAI says Sol and Luna were trained with similar methods to Astra with improvements in reasoning, factual reliability, coding, computer use, and alignment, but the company published no Sol scores, no context window, and no effort controls at launch.

The nearest measured GPT-6 reference is Astra, not Sol. In the Anthropic launch table Astra reads 57.9 percent on Terminal-Bench 4.0 and 53.3 percent on FrontierCode 1.1 Main against 66.4 and 54.4 for Opus 5.5, with Astra ahead instead on Terminal-Bench-Science 0.1. OpenAI explicitly positions Sol below Astra, so Astra numbers are a ceiling for Sol expectations rather than a proxy for them.

Default to Opus 5.5 where the workload needs verified long-horizon coding with published evals and third-party graders behind it. Default to GPT-6 Sol where per-token cost dominates and the workload matches the stated positioning around professional work, coding, automation, and computer use. Revisit the benchmark rows as soon as OpenAI publishes Sol evals, since this page will update when they land.
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
  'GPT-6 Sol vs Claude Opus 5.5: same-day Sep 22, 2026 launches with Sol at half the Opus per-token price and no published benchmarks yet against Opus 5.5 verified long-agent coding evals.',
  comparison_notes.notes,
  '[
    {"id": "spec-gpt6sol-opus55-1", "type": "spec_table", "title": "Specifications",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Developer", "OpenAI", "Anthropic"],
       ["Release date", "2026-09-22", "2026-09-22"],
       ["API model ID", "Not disclosed at launch", "claude-opus-5-5"],
       ["Context window", "Not disclosed at launch", "1M tokens"],
       ["Max output", "Not disclosed at launch", "128K tokens (300K on Batch API beta)"],
       ["Knowledge cutoff", "Not disclosed at launch", "Jun 2026"],
       ["Reasoning / effort", "Same training methods as Astra (per OpenAI), effort controls not disclosed", "Adaptive thinking (always on), medium default, per-message effort"]
     ]},
    {"id": "spec-gpt6sol-opus55-2", "type": "spec_table", "title": "Pricing",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Input / 1M tokens", "**$2**", "$4"],
       ["Cached input / 1M", "Not disclosed at launch", "$0.20"],
       ["Cache write / 1M", "Not disclosed at launch", "$5 (5m) / $8 (1h)"],
       ["Output / 1M tokens", "**$10**", "$20"],
       ["Batch / flex discount", "Not disclosed at launch", "Batch API: 50% off input and output"],
       ["Long-context surcharge", "Not disclosed at launch", "None - standard pricing through 1M context"]
     ]},
    {"id": "spec-gpt6sol-opus55-3", "type": "spec_table", "title": "Capabilities & access",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Text input", "Not itemized at launch", "Yes"],
       ["Image / vision input", "Not itemized at launch", "Yes"],
       ["Audio input", "Not itemized at launch", "No native audio input documented"],
       ["Video input", "Not itemized at launch", "No native video input documented"],
       ["Text output", "Not itemized at launch", "Yes"],
       ["Image output", "Not itemized at launch", "No"],
       ["Audio output", "Not itemized at launch", "No"],
       ["Video output", "Not itemized at launch", "No"],
       ["Tool / function calling", "Positioned for coding, automation, and computer-use tasks, API details not disclosed", "Yes"],
       ["Computer use", "Positioned for computer-use tasks, API details not disclosed", "Yes"],
       ["API access", "Priced per token, availability details not disclosed", "Yes"],
       ["Product access", "Lower-cost GPT-6 tier below Astra", "Claude + Claude Code + Claude API + cloud partners"],
       ["Weights / license", "Proprietary", "Proprietary"]
     ]},
    {"id": "spec-gpt6sol-opus55-4", "type": "spec_table", "title": "Model behavior",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Primary focus", "Lower-cost GPT-6 for professional work, coding, automation, and computer use", "Long-running agentic coding and knowledge work at Fable-level quality"],
       ["Long-horizon work", "No published long-horizon evals yet", "Codebase-wide migrations and audits spanning hours, including overnight unattended runs"],
       ["Efficiency / generation change", "Half the GPT-5.6 Sol promotional rate via caching and inference gains (per OpenAI)", "40% lower typical cost than Opus 5 with 30%+ faster output and fewer tokens per task"],
       ["Safety / approvals", "Astra-level caution applies, Sol safeguards not itemized", "Fable-5.1-class safeguards with verification programs and preserved thinking"]
     ]},
    {"id": "spec-gpt6sol-opus55-5", "type": "spec_table", "title": "Coding",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Terminal-Bench 4.0", "Not published at launch", "**66.4%** (xhigh)"],
       ["FrontierCode 1.1 Main", "Not published at launch", "**54.4%**"],
       ["CursorBench 4.0", "Not published at launch", "**57.8%**"],
       ["Terminal-Bench Science 0.1", "Not published at launch", "**58.7%**"]
     ]},
    {"id": "spec-gpt6sol-opus55-6", "type": "spec_table", "title": "Knowledge",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["Humanity''s Last Exam", "Not published at launch", "**67.7%** with tools"]
     ]},
    {"id": "spec-gpt6sol-opus55-7", "type": "spec_table", "title": "Agentic & computer use",
     "columns": ["GPT-6 Sol", "Claude Opus 5.5"],
     "rows": [
       ["GDPval-AA v2.1", "Not published at launch", "**1846 Elo**"],
       ["AutomationBench", "Not published at launch", "**40.0%**"],
       ["OSWorld 2.0", "Not published at launch", "**81.8%** partial"]
     ]},
    {"id": "markdown-gpt6sol-opus55-bottom-line", "type": "markdown",
     "content": "## Bottom line\n\nSol wins on list price at half the Opus 5.5 per-token rate with no published benchmarks to judge it by. Opus 5.5 carries verified long-agent coding evals with third-party graders behind the headline rows. Pick Sol where cost per token dominates and the work matches its stated positioning, pick Opus 5.5 where verified multi-hour coding performance is the requirement, and treat Astra scores as a ceiling for Sol expectations until OpenAI publishes Sol evals."}
  ]'::jsonb,
  jsonb_build_array(
    jsonb_build_object('title', 'OpenAI expands GPT-6 lineup with cheaper Sol and Luna models', 'url', 'https://www.reuters.com/technology/openai-expands-gpt-6-lineup-with-cheaper-sol-luna-models-2026-09-22'),
    jsonb_build_object('title', 'Claude Opus 5.5', 'url', 'https://www.anthropic.com/claude-opus-5-5'),
    jsonb_build_object('title', 'Claude Opus 5.5 overview', 'url', 'https://platform.claude.com/docs/en/models/opus-5-5/overview'),
    jsonb_build_object('title', 'What is new in Claude Opus 5.5', 'url', 'https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5')
  ),
  jsonb_build_object(
    'modelA', 'gpt-6-sol',
    'modelB', 'claude-opus-5-5',
    'verification_pass', 'gpt6-sol-luna-launch-2026-09-22'
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
