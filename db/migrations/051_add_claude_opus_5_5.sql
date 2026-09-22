-- Add Claude Opus 5.5 glossary entry and cross-link it from the existing
-- Claude Opus and Fable entries.
--
-- Source: the Anthropic introduction article for Claude Opus 5.5
-- (September 22, 2026) as the model card, plus the Opus 5.5 platform
-- overview for specifications, pricing, availability, and API changes.

-- 1. Glossary entry for Claude Opus 5.5.
WITH opus55_entry AS (
  SELECT $opus55$
Claude Opus 5.5 is Anthropic's September 2026 flagship [large language model](/glossary/large-language-model) for long-running agentic coding and professional knowledge work. Released on September 22, 2026 as the API model `claude-opus-5-5`, it is the first release in the Claude 5.5 family, and Anthropic positions it at the level of [Claude Fable 5](/glossary/claude-fable-5) on most work at roughly 60 percent lower operating cost than [Claude Opus 5](/glossary/claude-opus-5).

For a normal user, Opus 5.5 is the model to reach for when work spans hours rather than minutes: codebase-wide migrations, long debugging sessions, multi-step research, and document production where a cheaper model quietly going wrong would cost more than the token rate. For a developer, it is a proprietary hosted model with text and image input, text output, tool calling, and computer use, plus adaptive thinking that is always on and steered per request through an effort parameter that defaults to medium.

## Core profile

Opus 5.5 has a one-million-token [context window](/glossary/context-window), supports up to 128,000 output tokens on the synchronous API and up to 300,000 output tokens on the Batch API beta, and uses adaptive thinking on every request. Its reliable knowledge cutoff and training-data cutoff are both June 2026.

The model is available through the Claude API as `claude-opus-5-5`, as well as Claude's products, Amazon Bedrock, Google Cloud, Microsoft Foundry, and Claude Platform on AWS. Anthropic lists medium as the default effort level. Active status is committed with retirement not sooner than September 22, 2027.

Opus 5.5 is designed for tasks that last longer than one answer: multi-file software changes, migrations that run overnight, audits across large codebases, financial and legal analysis, and [AI agent](/glossary/ai-agent) workflows that must maintain state across many tool calls.

## Benchmark profile

Anthropic's launch table reports Opus 5.5 ahead of Fable 5.1, Opus 5, GPT-6 Astra, and GPT-5.6 Sol on nearly every shared row, with Terminal-Bench-Science 0.1 going to Astra instead. Anthropic adds that at these capability levels benchmark margins have become a less reliable guide to real-world differences, and that the gap between Opus 5.5 and Fable 5.1 feels narrower in daily use than the scores suggest.

| Evaluation | Opus 5.5 | Fable 5.1 | Opus 5 |
| --- | ---: | ---: | ---: |
| Terminal-Bench 4.0 | **66.4%** (xhigh) | 55.8% | 52.3% |
| FrontierCode v1.1 Main | **54.4%** | 50.3% | 48.0% |
| CursorBench 4.0 | **57.8%** | 51.8% | 46.6% |
| GDPval-AA v2.1 | **1846 Elo** | 1735 | 1708 |
| AutomationBench | **40.0%** | 31.4% | 26.9% |
| Humanity's Last Exam, with tools | **67.7%** | 65.6% | 63.6% |
| Terminal-Bench-Science 0.1 | **58.7%** | 52.6% | 29.0% |
| OSWorld 2.0, partial | **81.8%** | 80.7% | 74.0% |
| Chartography, with tools | **89.0%** | 88.4% | 83.4% |

Unless otherwise noted, Opus 5.5 results use adaptive thinking at max effort, with Terminal-Bench 4.0 at xhigh effort and the Astra figures as reported by OpenAI. AutomationBench was run and reported by Zapier without fallback models, so safeguard interventions counted as failures and the score understates production behavior. Opus 5.5 ran with production safeguards enabled, and when they intervened cybersecurity tasks fell back to Opus 4.8 while biology and frontier-model tasks fell back to Opus 5, which likely reduces the reported scores.

The efficiency story sits underneath the accuracy story. At default medium effort Opus 5.5 beats GPT-6 Astra's top FrontierCode score for about a fifth of the cost per task, matches Astra on Terminal-Bench 4.0 for about 40 percent of the cost, and beats GPT-5.6 Sol on CursorBench by 11 points for about a third of the cost. One early tester translated HAProxy from C to Rust in 9.5 hours against 12 for Fable 5.1 at 51 percent lower cost, with both rewrites passing nearly all regression tests.

## Pricing and efficiency

Opus 5.5 costs USD 4 per million base input tokens and USD 20 per million output tokens, which is 20 percent below Opus 5 per token. Five-minute cache writes cost USD 5 per million tokens and one-hour cache writes cost USD 8.

| Token type | Opus 5.5 | Opus 5 |
| --- | ---: | ---: |
| 1M input | $4 | $5 |
| 1M output | $20 | $25 |
| 1M cache read | $0.20 | $0.50 |
| 5m / 1h cache write | $5 / $8 | $6.25 / $10 |

The important change is the cache-read rate. A cache read costs USD 0.20 per million tokens, down 60 percent from Opus 5. Long agent sessions repeatedly read the same growing prefix, so cache reads can dominate their input bill. Anthropic's tests show default settings costing 40 percent less than Opus 5 on typical workloads, with output generated more than 30 percent faster and fewer tokens spent per task.

Fast mode is available as a research preview on Claude Code and the Claude Platform with up to 2.5x output speed, priced separately at USD 8 per million input tokens and USD 40 per million output tokens. Subscription plans also get higher five-hour usage limits plus a savable rate-limit reset.

## Safeguards

Opus 5.5 scored better than any recent Claude model on Anthropic's automated behavioral audit across nearly 2,000 scenarios, including the behaviors behind recent cybersecurity incidents such as motivated reasoning, sandbox escape attempts, and acting on a belief of being in a simulation. In a containment-boundary evaluation it attempted to circumvent boundaries around 85 percent less often than Opus 5 or Mythos 5.1, and every attempt was low severity and self-reported.

Because Opus 5.5 is comparable to Mythos 5.1 in biology and cybersecurity, it ships with a similar class of safeguards to Fable 5.1. Most cybersecurity work reroutes transparently to Opus 4.8, biology research goes through the Life Sciences Verification Program, and verified practitioners get expanded cyber access through the Cyber Verification Program. On Gray Swan's prompt-injection [benchmark](/glossary/benchmark) it ties Fable 5.1 for the lowest attack success rate of any model tested.

Opus 5.5 launches with preserved thinking, the anti-[distillation](/glossary/model-distillation) safeguard introduced with Fable 5.1 that stops API users from editing Claude's prior context to extract its reasoning. Like previous Opus models it is available with zero data retention, and it carries watermarking measures for EU AI Act compliance.

## API and behavior changes

Adaptive thinking is always on and cannot be disabled, so forced tool use returns an error and thinking effort is controlled through the effort parameter instead. Thinking blocks are bound to the model and conversation prefix that produced them, and editing an earlier message, tool definition, or system prompt can invalidate later thinking. Text produced between tool calls now arrives inside thinking blocks that read empty at the default display setting, so streaming progress UIs need an explicit display value to keep narrating between calls. On the Claude API and Google Cloud the earlier computer use tool generation is no longer accepted.

## Applications and workflow fit

Opus 5.5 is best suited for repository-scale coding, long migrations and audits, complex refactors, research that follows evidence across several steps, financial and legal analysis, and enterprise agent tasks where a cheaper model's quiet mistake would cost more than the Opus token rate. Early testers report large efficiency wins: a 680,000-line migration finished in under a day, a 200,000-line audit completed in under three hours against 20 hours on Opus 5 with 2.5x the tokens, and success cutting web-app load times 39 out of 40 attempts where Opus 5 made smaller changes that altered app behavior.

It is also a stronger writing partner than Opus 5 by most tester accounts, putting the most important information first with less jargon and better adherence to style rules, which makes long-session work easier to check. That readability is a safety benefit as well as a practical one for teams running Claude unattended.

Opus 5.5 is not the obvious choice for short chat, simple classification, or latency-sensitive requests where Sonnet or Haiku answer faster and cheaper. Teams should evaluate the complete model-plus-harness workflow rather than choose it from the [benchmark](/glossary/benchmark) table alone, since safeguard fallbacks materially affect what the public model completes on high-risk tasks.

## Bottom line

Claude Opus 5.5 is the new leading Claude model for agentic coding, computer use, and knowledge work, with the unusual property that the upgrade also cuts the bill: same 1M context as Opus 5, lower per-token prices, much cheaper cache reads, and fewer tokens per task. The launch table shows broad gains over Fable 5.1 with the usual vendor-reported caveats around effort levels, harnesses, and safeguard fallbacks. Sonnet 5.5 and Haiku 5.5 follow in the coming weeks with many of the same improvements.
$opus55$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:claude-opus-5-5',
  'glossary',
  'claude-opus-5-5',
  '',
  'glossary/claude-opus-5-5',
  'Claude Opus 5.5',
  'Anthropic''s Sep 2026 Opus flagship for long agentic coding and knowledge work, matching Fable 5.1 at 60 percent lower cost with a 1M-token context window.',
  opus55_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', opus55_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Claude Opus 5.5', 'url', 'https://www.anthropic.com/claude-opus-5-5'),
    jsonb_build_object('title', 'Claude Opus 5.5 overview', 'url', 'https://platform.claude.com/docs/en/models/opus-5-5/overview'),
    jsonb_build_object('title', 'What is new in Claude Opus 5.5', 'url', 'https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5'),
    jsonb_build_object('title', 'Claude Opus 5.5 system card', 'url', 'https://www.anthropic.com/claude-opus-5-5-system-card')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array('claude-opus-5', 'claude-fable-5', 'anthropic', 'claude-code', 'large-language-model', 'ai-agent', 'agentic-ai', 'agent-harness', 'context-window', 'benchmark', 'api', 'model-distillation'),
    'analogy', 'The same senior engineer as Fable 5.1 with a cheaper timesheet: equal care on long jobs, lower fees for rereading the file, and clearer status updates along the way.',
    'seoDescription', 'Claude Opus 5.5 explained: September 2026 launch benchmarks, 1M context, $4/$20 API pricing, adaptive thinking, safeguards, Fable 5.1 comparison.',
    'seoKeywords', jsonb_build_array('Claude Opus 5.5', 'Claude Opus 5.5 benchmarks', 'Claude Opus 5.5 pricing', 'Claude Opus 5.5 vs Fable 5.1', 'Claude Opus 5.5 API', 'Claude Opus 5.5 context window', 'claude-opus-5-5', 'Opus 5.5 Terminal-Bench', 'Opus 5.5 safeguards', 'Anthropic Opus 5.5')
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-09-22',
  0
FROM opus55_entry
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

-- 2. Cross-link the new glossary slug from the existing Claude entries.
UPDATE content_items
SET
  metadata = CASE
    WHEN COALESCE(metadata->'relatedTerms', '[]'::jsonb) @> '["claude-opus-5-5"]'::jsonb
      THEN metadata
    ELSE jsonb_set(
      COALESCE(metadata, '{}'::jsonb),
      '{relatedTerms}',
      COALESCE(metadata->'relatedTerms', '[]'::jsonb) || jsonb_build_array('claude-opus-5-5'),
      true
    )
  END,
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug IN ('claude-opus-5', 'claude-fable-5', 'claude-fable-51')
  AND parent_slug = '';
