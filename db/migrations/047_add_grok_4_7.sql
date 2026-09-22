-- Add Grok 4.7 glossary entry and cross-link it from Grok 4.6 and Grok 4.5.

-- 1. Glossary entry for Grok 4.7.
WITH grok47_entry AS (
  SELECT $grok47$
Grok 4.7 is xAI's September 2026 frontier [large language model](/glossary/large-language-model) for software engineering, long-running [AI agents](/glossary/ai-agent), and professional knowledge work. Released on September 21, 2026 as the API model `grok-4.7`, it succeeds [Grok 4.6](/glossary/grok-4-6) at the same list price and the same 500,000-token [context window](/glossary/context-window). Official launch pages brand the company as SpaceXAI; the developer, API, and model lineage remain xAI's Grok family.

For a normal user, Grok 4.7 is the model to reach for in Cursor and Grok Build when a task is difficult or runs long: xAI says it stays on hard problems longer, checks its own work more carefully, and ships with the company's best-calibrated safeguards to date. For a developer, it is a proprietary hosted model with text and image input, text output, function calling, web search, X search, and code execution, plus four reasoning-effort levels - low, medium, high (the default), and xhigh - controlled through the `reasoning.effort` parameter.

## What changed from Grok 4.6

Grok 4.7 is built on a new, larger base model rather than another supplemental training run of the 4.6 weights, followed by a longer [reinforcement learning](/glossary/reinforcement-learning) run over a harder mix of tasks that take many hours to complete. xAI also claims stronger self-verification and native understanding of the Grok Bot harness. No parameter count, architecture details, or training corpus were published, so the upgrade is best judged on behavior and benchmarks rather than specifications.

## Key specifications

| Specification | Grok 4.7 |
| --- | --- |
| Release date | September 21, 2026 |
| API model ID | `grok-4.7` |
| Context window | 500,000 tokens |
| Pretraining cutoff | June 2026, with supplemental training through August 2026 |
| Reasoning effort | Low / medium / high (default) / xhigh |
| Inputs | Text and images |
| Output | Text; no fixed output limit documented |
| Tools | Function calling, web search, X search, code execution |
| Availability | Grok API, Cursor, Grok Build, third-party harnesses, routers, and clouds |
| License | Proprietary; weights not released |

The 500K [context window](/glossary/context-window) and the effort levels carry over from [Grok 4.6](/glossary/grok-4-6), so existing integrations move over without reworking prompts or context budgets. What changes is how the model spends long runs: more steps, more self-checks, and - as independent measurements show - substantially more output tokens per task.

## Official benchmark comparison

These are xAI's self-reported launch results. Effort settings differ across rows (xhigh for Grok 4.7 against high for Grok 4.6 in places), and the Terminal-Bench 4.0 score uses the Grok Build harness, so cross-vendor rows are vendor-reported snapshots rather than identical configurations.

| Benchmark | Grok 4.7 | Grok 4.6 | GPT-5.6 Sol | Fable 5.1 |
| --- | ---: | ---: | ---: | ---: |
| CursorBench 4.0 | 46.3% (xhigh) | 40.4% (high) | 41.7% | 51.8% |
| DeepSWE v1.1 | 71.0% (high) | 65.2% | 72.7% | 70.0% |
| Terminal-Bench 4.0 | 38.0% (xhigh) | 20.3% | 37.3% | 57.9% |
| EEBench | 66.0% (xhigh) | 60.0% | 39.4% | 56.4% |
| AA Briefcase v1.1 | 1,657 | 1,546 | 1,487 | 1,678 |
| Harvey Legal Agent | 19.6% | 15.8% | 2.5% | 6.7% |
| HealthBench Professional | 56.7% | 48.5% | 60.5% | 62.1% |

The pattern is steady progress rather than a sweep. Grok 4.7 leads xAI's own table on EEBench and the legal agent [benchmark](/glossary/benchmark), closes ground on DeepSWE v1.1, and posts its largest gain on Terminal-Bench 4.0 - while trailing Fable 5.1 on CursorBench 4.0, Terminal-Bench 4.0, and HealthBench Professional. SWE-Marathon v1.1, a long-agent-streak evaluation, jumps from 31.9% to 46.0% at high effort, which is the more informative delta if your workload looks like sustained agent runs rather than single-issue patches.

## Independent measurements

Artificial Analysis scored Grok 4.7 at 46 on its Intelligence Index (v4.3.2), mid-pack against 53 each for Fable 5.1 and GPT-6, and noted its two highest reasoning levels perform about the same. Its Coding Agent Index pairing of Grok Build with Grok 4.7 scored 56, up nine points from Grok 4.6 - fourth among native-harness pairings behind Fable 5.1, GPT-6 Astra, and Opus 5. The same Terminal-Bench 4.0 reads 26% under the independent mini-swe-agent harness versus 38% self-reported on Grok Build: twelve points explained by harness, not capability. Hallucination rate improved to 29% from 34% at roughly unchanged accuracy.

## Pricing

| Lane | Input / cached input / output per 1M tokens |
| --- | --- |
| Standard (under 200K prompt tokens) | $2 / $0.50 / $6 |
| Long context (200K prompt tokens and above) | $4 / $1 / $12 |

List price matches [Grok 4.6](/glossary/grok-4-6), and a fast serving variant doubles output speed at twice the price on Cursor and Grok Build. But equal per-token prices do not guarantee equal bills: cross the 200K prompt-token threshold and every token in the request bills at the higher tier, and independent testing measured roughly 81,000 output tokens per task against 38,000 for Grok 4.6 - about 2.5x the per-task API cost at twice the wall-clock time for the nine-point Coding Agent Index gain.

## Safeguards

Grok 4.7 ships with a new safeguard stack that xAI calls its strongest on refusals and jailbreak resistance. It tops LatchBio's biosafety [benchmark](/glossary/benchmark) at 62.4% and posts the highest safety score on HackerBench v0.3, letting through only 3.3% of risky dual-use cyber prompts while rarely blocking legitimate security work. Select cybersecurity partners get invite-only access to its red-team capabilities for defense research.

## When to use Grok 4.7

Grok 4.7 fits long-horizon coding-agent runs, multi-hour office and terminal work, electrical-engineering agent tasks, and professional knowledge work where persistence and self-verification matter more than raw leaderboard position. It is not the right pick when independent top scores are the requirement: Fable 5.1 and GPT-6 lead the composite indices, and the cheaper DeepSeek V4.1 Flash edges it on independent Terminal-Bench 4.0 while billing far less per task.

## Bottom line

Grok 4.7 is a same-price, same-context upgrade that buys longer horizons and better self-checking instead of a new price tier. The vendor-reported deltas are real but carry cross-effort and harness caveats, and the model's verbosity means per-task costs rise even as per-token prices stay flat. Evaluate it on elapsed job time and total tokens per completed task, not on the $2/$6 sticker.
$grok47$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:grok-4-7',
  'glossary',
  'grok-4-7',
  '',
  'glossary/grok-4-7',
  'Grok 4.7',
  'xAI''s September 2026 frontier model for coding, long-running agents, and knowledge work, with a 500K-token context window at the same $2/$6 price as Grok 4.6.',
  grok47_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', grok47_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing Grok 4.7', 'url', 'https://x.ai/news/grok-4-7'),
    jsonb_build_object('title', 'Grok 4.7 model documentation', 'url', 'https://docs.x.ai/developers/models/grok-4-7'),
    jsonb_build_object('title', 'Grok 4.7 Benchmarks, Pricing and Context Window', 'url', 'https://llm-stats.com/models/grok-4-7'),
    jsonb_build_object('title', 'xAI launches Grok 4.7 at bargain prices, but benchmarks reveal a wide gap', 'url', 'https://the-decoder.com/xai-launches-grok-4-7-at-bargain-prices-but-benchmarks-reveal-a-wide-gap-to-claude-and-gpt-6/')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array('grok-4-6', 'grok-4-5', 'xai', 'large-language-model', 'ai-agent', 'agentic-ai', 'agent-harness', 'context-window', 'benchmark', 'api', 'reinforcement-learning', 'inference-time-compute', 'cursor'),
    'analogy', 'A same-price engine upgrade that runs longer shifts and double-checks its work: cheaper per token than the frontier, but the longer shifts show up on the invoice.',
    'seoDescription', 'Grok 4.7 explained: September 2026 coding benchmarks, 500K context, $2/$6 API pricing, reasoning levels, safeguards, and Grok 4.6 comparison.',
    'seoKeywords', jsonb_build_array('Grok 4.7', 'Grok 4.7 benchmarks', 'Grok 4.7 pricing', 'Grok 4.7 API', 'Grok 4.7 context window', 'Grok 4.7 vs Grok 4.6', 'Grok 4.7 CursorBench', 'Grok 4.7 DeepSWE', 'Grok 4.7 Terminal-Bench', 'SpaceXAI Grok 4.7')
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-09-21',
  0
FROM grok47_entry
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

-- 2. Cross-link the new glossary slug from the existing Grok 4.x entries.
UPDATE content_items
SET
  metadata = CASE
    WHEN COALESCE(metadata->'relatedTerms', '[]'::jsonb) @> '["grok-4-7"]'::jsonb
      THEN metadata
    ELSE jsonb_set(
      COALESCE(metadata, '{}'::jsonb),
      '{relatedTerms}',
      COALESCE(metadata->'relatedTerms', '[]'::jsonb) || jsonb_build_array('grok-4-7'),
      true
    )
  END,
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug IN ('grok-4-6', 'grok-4-5')
  AND parent_slug = '';
