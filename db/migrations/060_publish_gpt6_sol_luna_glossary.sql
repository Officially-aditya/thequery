-- Publish glossary entries for GPT-6 Sol and GPT-6 Luna.
--
-- Source: OpenAI launch announcement "Introducing GPT-6 Sol and Luna"
-- (https://openai.com/index/introducing-gpt-6-sol-and-luna/), plus the
-- vendor-reported benchmark scorecard recorded in data/glossary.json.
-- data/glossary.json is only a seed source; the live site reads glossary
-- rows from the content_items table, so these entries ship as a migration
-- like the 047 Grok 4.7 and 051 Opus 5.5 glossary publishes.

-- 1. Glossary entry for GPT-6 Sol.
WITH sol_entry AS (
  SELECT $solbody$
GPT-6 Sol is OpenAI's September 2026 production-focused [large language model](/glossary/large-language-model), released on September 22, 2026 alongside [GPT-6 Luna](/glossary/gpt-6-luna) as the lower-cost extension of the GPT-6 family below the Astra flagship. It is positioned for sustained professional use — coding, automation, and computer use — rather than as a lightweight fallback, with adjustable reasoning effort (up to xhigh/max) that trades latency and spend for more deliberate reasoning.

Sol costs $2 per million input tokens and $10 per million output tokens, a 50% cut against GPT-5.6 Sol ($4/$20). GPT-6 increases cache hit rates, offers a 90% discount on cached input-token reads, and lets developers modify reasoning effort and tool availability without invalidating cache reuse — efficiency gains aimed squarely at [AI agent](/glossary/ai-agent) workflows that resend the same repository, instructions, or accumulated context across dozens of calls.

## Benchmark profile

OpenAI's vendor-reported launch figures frame Sol as a cost-efficient production model rather than simply a weaker Astra. On AutomationBench, OpenAI reports Sol at xhigh effort scoring 33.2% at a reported $0.27 per task, versus 30.3% for GPT-6 Astra at low effort at 3.9 times the cost. On DeepSWE v1.1 at maximum effort, Sol scores 68.8%, within 1.1 percentage points of Claude Fable 5's highest reported score at roughly 80% lower per-task cost. These are vendor-reported economics and should not be treated as universal production benchmarks.

## Benchmark table

| Category | Benchmark | GPT-6 Sol | GPT-6 Luna | Claude Opus 5.5 |
| --- | --- | ---: | ---: | ---: |
| Professional work | AutomationBench 1.0.6 | 33.2% (xhigh) | 20.7% (max) | 40.0% |
| Professional work | Agents' Last Exam V1 | 56.6% (max) | 50.9% (max) | Not reported in Opus 5.5 launch table |
| Coding | FrontierCode 1.1 Main | 49.3% (max) | 42.4% (max) | 54.4% |
| Coding | DeepSWE v1.1 | 68.8% (max) | 66.6% (max) | Not reported in Opus 5.5 launch table |
| Computer use | OSWorld 2.0 | 64.4% (max) | 52.7% (max) | 81.8% |

Scores for Sol and Luna are OpenAI-reported chart readings at stated effort levels; Opus 5.5 figures are Anthropic-reported. Cross-vendor cells reflect different runs and subsets, so they are not strict head-to-head wins — see the [GPT-6 Sol vs Claude Opus 5.5 comparison](/comparisons/gpt-6-sol-vs-opus-5-5) for the full scorecard with caveats.
$solbody$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:gpt-6-sol',
  'glossary',
  'gpt-6-sol',
  '',
  'glossary/gpt-6-sol',
  'GPT-6 Sol',
  'OpenAI''s Sep 2026 production-focused GPT-6 model at $2/$10 per million tokens, positioned for professional work, coding, automation, and computer use below the Astra flagship.',
  sol_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', sol_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing GPT-6 Sol and Luna', 'url', 'https://openai.com/index/introducing-gpt-6-sol-and-luna/'),
    jsonb_build_object('title', 'GPT-6 Sol vs Claude Opus 5.5 comparison', 'url', '/comparisons/gpt-6-sol-vs-opus-5-5'),
    jsonb_build_object('title', 'GPT-6 Sol and Luna Are Here. OpenAI Is Cutting the Cost of Intelligence.', 'url', '/articles/gpt-6-sol-luna-cutting-cost-intelligence')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array('gpt-6-luna', 'gpt-5-6', 'openai', 'large-language-model', 'ai-agent', 'agentic-ai', 'api', 'context-window', 'benchmark', 'inference', 'claude-opus-5-5'),
    'analogy', 'The long-haul work truck of the GPT-6 family: not the flagship showpiece, but the model engineered to do serious professional work all day at a per-mile cost that makes scale practical.',
    'seoDescription', 'GPT-6 Sol explained: September 2026 launch pricing ($2/$10), benchmark scorecard vs Luna and Opus 5.5, caching, effort levels, and production use cases.',
    'seoKeywords', jsonb_build_array('GPT-6 Sol', 'GPT-6 Sol benchmarks', 'GPT-6 Sol pricing', 'GPT-6 Sol vs Opus 5.5', 'GPT-6 Sol vs Luna', 'gpt-6-sol', 'OpenAI GPT-6 Sol')
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-09-22',
  0
FROM sol_entry
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

-- 2. Glossary entry for GPT-6 Luna.
WITH luna_entry AS (
  SELECT $lunabody$
GPT-6 Luna is OpenAI's September 2026 high-volume [large language model](/glossary/large-language-model), released on September 22, 2026 alongside [GPT-6 Sol](/glossary/gpt-6-sol) as the lower-cost extension of the GPT-6 family below the Astra flagship. Where Sol targets sustained professional work, Luna targets the much larger class of high-volume workloads — search products, support systems, coding workflows, and [AI agents](/glossary/ai-agent) making thousands or millions of calls — where per-call inference cost dominates the economics.

Luna costs $0.10 per million input tokens and $0.50 per million output tokens. Against GPT-5.6 Luna ($0.20/$1.20), input pricing is down 50% and output pricing is down about 58%. Like Sol, it benefits from GPT-6 inference and caching improvements, including a 90% discount on cached input-token reads and the ability to modify reasoning effort and tool availability without invalidating cache reuse.

## Benchmark profile

OpenAI's vendor-reported launch figures position Luna as a model whose capability floor has moved downward in price rather than upward in absolute performance. OpenAI says Luna at higher effort can match GPT-5.6 Sol on its internal factuality evaluation at roughly one hundredth of Sol's cost, and on OSWorld 2.0 offline it can exceed GPT-5.6 Sol at medium effort at one tenth of its cost. These are vendor-reported comparisons, not independent benchmarks.

## Benchmark table

| Category | Benchmark | GPT-6 Sol | GPT-6 Luna | Claude Opus 5.5 |
| --- | --- | ---: | ---: | ---: |
| Professional work | AutomationBench 1.0.6 | 33.2% (xhigh) | 20.7% (max) | 40.0% |
| Professional work | Agents' Last Exam V1 | 56.6% (max) | 50.9% (max) | Not reported in Opus 5.5 launch table |
| Coding | FrontierCode 1.1 Main | 49.3% (max) | 42.4% (max) | 54.4% |
| Coding | DeepSWE v1.1 | 68.8% (max) | 66.6% (max) | Not reported in Opus 5.5 launch table |
| Computer use | OSWorld 2.0 | 64.4% (max) | 52.7% (max) | 81.8% |

Scores for Sol and Luna are OpenAI-reported chart readings at stated effort levels; Opus 5.5 figures are Anthropic-reported. Cross-vendor cells reflect different runs and subsets, so they are not strict head-to-head wins — see the [GPT-6 Sol vs Claude Opus 5.5 comparison](/comparisons/gpt-6-sol-vs-opus-5-5) for the full scorecard with caveats.

The significance of Luna is not that it replaces Sol. It is that the capability floor is moving downward in cost: work that previously needed a mid-tier model can now run on Luna at a fraction of the operating cost.
$lunabody$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:gpt-6-luna',
  'glossary',
  'gpt-6-luna',
  '',
  'glossary/gpt-6-luna',
  'GPT-6 Luna',
  'OpenAI''s Sep 2026 high-volume GPT-6 model at $0.10/$0.50 per million tokens, for workloads where inference cost matters more than squeezing out the last increment of capability.',
  luna_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', luna_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing GPT-6 Sol and Luna', 'url', 'https://openai.com/index/introducing-gpt-6-sol-and-luna/'),
    jsonb_build_object('title', 'GPT-6 Sol vs Claude Opus 5.5 comparison', 'url', '/comparisons/gpt-6-sol-vs-opus-5-5'),
    jsonb_build_object('title', 'GPT-6 Sol and Luna Are Here. OpenAI Is Cutting the Cost of Intelligence.', 'url', '/articles/gpt-6-sol-luna-cutting-cost-intelligence')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array('gpt-6-sol', 'gpt-5-6', 'openai', 'large-language-model', 'ai-agent', 'agentic-ai', 'api', 'context-window', 'benchmark', 'inference', 'claude-opus-5-5'),
    'analogy', 'The city-wide delivery fleet to Sol''s long-haul truck: each individual run carries less, but the per-mile cost is so low that constant delivery at scale becomes economical.',
    'seoDescription', 'GPT-6 Luna explained: September 2026 launch pricing ($0.10/$0.50), benchmark scorecard vs Sol and Opus 5.5, caching, and high-volume use cases.',
    'seoKeywords', jsonb_build_array('GPT-6 Luna', 'GPT-6 Luna benchmarks', 'GPT-6 Luna pricing', 'GPT-6 Luna vs Sol', 'GPT-6 Luna vs Opus 5.5', 'gpt-6-luna', 'OpenAI GPT-6 Luna')
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-09-22',
  0
FROM luna_entry
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
