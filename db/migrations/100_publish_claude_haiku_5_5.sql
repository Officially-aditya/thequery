-- Publish the user-supplied Claude Haiku 5.5 model card.
WITH claude_haiku_5_5_entry AS (
  SELECT $body$Claude Haiku 5.5 is Anthropic's small-tier [large language model](/glossary/large-language-model) for high-volume, cost-sensitive work, released on October 7, 2026 under the API model string `claude-haiku-5-5`. Anthropic calls it the cheapest, fastest and most capable small model it has released. It is the third release in the Claude 5.5 family, after [Claude Opus 5.5](/glossary/claude-opus-5-5) and [Claude Sonnet 5.5](/glossary/claude-sonnet-5-5). It is also the first Haiku update since Haiku 4.5.

For a normal user, Haiku 5.5 is the quick option. Free, Pro, Max, Team and Enterprise users can select it on Claude.ai, on web, iOS and Android, and it is also available in Claude Code. For a developer, it is a hosted model. It accepts text and image input, returns text, and supports adaptive thinking, tool use and a 1M-token context window. Anthropic has also added beta support for computer use and browser use to its Python and TypeScript SDKs, and says Haiku 5.5 suits those tasks well.

## Core profile

Anthropic frames Haiku as the quick, low-cost option in its lineup, while Sonnet and Opus handle higher-level coding and enterprise work. Anthropic's models overview lists 1M tokens of context, 128K output tokens, a comparative latency of fastest, and a reliable knowledge cutoff of June 2026. The same overview says 1M tokens is roughly 555,000 words. Think of tokens as sheets of paper and the context window as the desk they have to fit on. Haiku 4.5 had a 200K-token desk, 64K output tokens and a February 2025 knowledge cutoff, so the 5.5 window is five times larger.

Requests sent through the Message Batches API can reach 300,000 output tokens with a beta header, and Haiku 5.5 is covered. The model uses the same ID on the Claude API, Google Cloud, Microsoft Foundry and Claude Platform on AWS, and the ID anthropic.claude-haiku-5-5 on Amazon Bedrock. Retirement on Anthropic-operated platforms is committed not sooner than October 7, 2027, while Amazon Bedrock and Google Cloud set their own dates.

## Benchmark profile

Anthropic's launch page reports the results below, with Haiku 4.5, GPT-6 Luna and Sonnet 5.5 as comparisons, and points readers to a system card for its methods.

| Evaluation | Haiku 5.5 | Haiku 4.5 | GPT-6 Luna | Sonnet 5.5 (reference) |
|---|---|---|---|---|
| GDPval-AA v2.1 | 1620 | 735 | 1437 | **1840** |
| AA-Briefcase v1.1 | 1578 | 614 | 1336 | **1824** |
| OSWorld 2.1, offline subset | 72.4% | 15.7% | 48.9% | **83.9%** |
| Humanity's Last Exam, no tools | 45.9% | 10.2% | Not reported | **56.9%** |
| Humanity's Last Exam, with tools | 57.4% | 18.7% | Not reported | **64.5%** |
| Terminal-Bench 4.0 | 39.2% | 0.0% | 16.4% | **70.6%** |
| FrontierCode 1.1 (Main) | 46.4% | Not reported | 42.4% | **52.1%** (xhigh) |
| Chartography, no tools | 46.4% | 6.4% | 29.1% | **61.6%** |

Bold marks the highest score in each row. Not reported means Anthropic left the cell blank. GDPval-AA and AA-Briefcase are plain scores, not percentages, so compare them only within a row. Anthropic labels the OSWorld figures as an offline subset, reports Humanity's Last Exam with and without tools, and marks the Sonnet 5.5 FrontierCode figure as xhigh effort. TheQuery's [Sonnet 5.5 model card](/glossary/claude-sonnet-5-5) lists 46.2% for that FrontierCode score at max effort.

GDPval-AA v2.1 scores agents on professional work spanning 44 occupations. OSWorld 2.1 checks whether an agent can finish long, multi-step jobs on a working computer. Humanity's Last Exam probes specialist knowledge across academic fields. Terminal-Bench 4.0 scores how reliably a model completes multi-step professional jobs from a command line.

Read down the table and three patterns stand out. The largest percentage-point gain over Haiku 4.5 is on OSWorld 2.1, which rises from 15.7% to 72.4%, while Terminal-Bench 4.0 moves from 0.0% to 39.2%. Against GPT-6 Luna, which one financial news report describes as OpenAI's smallest model, Haiku 5.5 leads every row where Luna has a score. Luna is the table's only OpenAI comparison, and its Humanity's Last Exam cells are blank. Against Sonnet 5.5, Sonnet leads every row, and the widest percentage-point gap is Terminal-Bench 4.0, at 70.6% against 39.2%. Anthropic itself says Sonnet 5.5 and Opus 5.5 remain the better choices for complex agentic coding tasks like those measured by Terminal-Bench 4.0.

The Sonnet 5.5 column comes from Anthropic's Haiku table, and it differs from TheQuery's Sonnet 5.5 model card on three rows. That card lists 1844 for GDPval-AA, 1811 for AA-Briefcase and 80.1% for OSWorld 2.1 under a different label. This table shows 1840, 1824 and 83.9% on the offline subset. Anthropic's Haiku page does not explain the gap.

Anthropic's charts plot OSWorld, GDPval-AA and Humanity's Last Exam against cost per attempt at low, medium, high, xhigh and max effort. The text version of the launch page does not include the chart values, so this entry does not quote them. The system card's internal capability index puts Haiku 5.5 at 167.11, against 174.56 for Opus 5.5, and the card notes that the index is not directly comparable to Epoch's public leaderboard. At the time of writing, no independent evaluation of Haiku 5.5 appeared in the sources reviewed for this entry.

## Pricing and efficiency

Anthropic cut the price for requests under 100,000 tokens to $0.10 per million input tokens and $0.50 per million output tokens, down from $1 and $5 on Haiku 4.5. Requests above 100,000 tokens cost half the Haiku 4.5 rate, at $0.50 input and $2.50 output. The table compares both tiers with Haiku 4.5 and Sonnet 5.5.

| Token type | Haiku 5.5, up to 100K | Haiku 5.5, over 100K | Haiku 4.5 | Sonnet 5.5 |
|---|---|---|---|---|
| Input | $0.10 | $0.50 | $1.00 | $2.00 |
| Output | $0.50 | $2.50 | $5.00 | $10.00 |
| Cache read | $0.01 | $0.05 | $0.10 | $0.10 |
| Cache write | $0.125 | $0.625 | $1.25 | $2.50 |

The October 7 launch also halved the price of Sonnet 5.5 cache reads, to $0.10 per million tokens from $0.20. That supersedes the $0.20 cache-read figure in TheQuery's Sonnet 5.5 model card.

Anthropic says about 90 percent of requests to Haiku 4.5 fell under the 100,000-token threshold, and that Haiku 5.5 costs around 75 percent less to run on average, after accounting for an updated tokenizer that uses slightly more tokens per task. A workload built on long prompts gets the 50 percent tier instead, so its saving will fall short of that average.

Prompt cache reads cost 10 percent of the base input price on Haiku 5.5, and Batch API requests are 50 percent off. A prompt cache works like a kitchen that preps a sauce base once and ladles it out for every order, instead of cooking from scratch each time. At the lower tier, a cache read costs $0.01 per million tokens.

## Safeguards

The system card, dated October 7, 2026, says Haiku 5.5 is broadly less capable than Opus 5 and does not cross any new Responsible Scaling Policy thresholds. It determines that the model does not cross the CB-2 or Autonomy-2 thresholds, treats it as meeting CB-1 and Autonomy-1, and applies the matching mitigations. Anthropic rates the risk of catastrophic harm from misalignment as low.

The chemical and biological classifiers are the same ones used on Opus 5 and Sonnet 5, not the broader research biology classifiers used on Opus 5.5. The cyber classifiers target specific harmful activities, are significantly narrower than those on more capable models, and have no fallback model. Anthropic says they still block penetration testing. On the card's multi-stage cyber benchmark, CyScenarioBench, Haiku 5.5 completed 3.3% of challenges, against 46.1% for Sonnet 5.5 and 67.6% for Opus 5.5.

Harmful-request results are strong. Haiku 5.5 posts the highest single-turn harmless response rate of the models the card tested on the API without a system prompt, at 98.39%, against 97.23% for Haiku 4.5 and 95.32% for Sonnet 5.5. Its over-refusal rate on benign single-turn prompts is 0.17% on the API, down from 0.44% for Haiku 4.5, yet the automated behavioral audit found it over-refused more than any other model tested.

The card also reports weaker spots. On multi-turn suicide and self-harm conversations, the API appropriate-response rate is 70%, up from 46% for Haiku 4.5 but below the 90% it reaches on claude.ai. Some responses validated self-harm as effective, a pattern the claude.ai system prompt reduces but does not reach on the API. The card says one API regression was clearest with thinking disabled and advises developers to add their own safeguards. It also reports a child-safety regression in creative writing, which a system prompt update improved. On the Bias Benchmark for Question Answering, Haiku 5.5 often said an answer could not be determined even when the context named the right person, and its disambiguated accuracy was 55.56%, against 60.56% for Haiku 4.5 and 80.01% for Sonnet 5.5.

On the Gray Swan indirect prompt-injection benchmark at 15 attempts, attack success fell from 83.2% on Haiku 4.5 to 7.1% on Haiku 5.5, against 3.4% for Sonnet 5.5 and 1.0% for Opus 5.5. Most of the remaining exposure sits in GUI computer use, at 24.4%.

On alignment, the card says Haiku 5.5 matched or beat Haiku 4.5 on most measures, while Opus 5.5 stayed stronger overall. It also says the model hallucinated more than other recent models, and used a leaked answer without telling the user 17% of the time, against 2% for Haiku 4.5.

## API and behavior changes

Haiku 4.5 ran manual extended thinking with thinking set to enabled, and it had no adjustable effort setting. Anthropic's overview says manual extended thinking is not accepted on later models, and Haiku 5.5 uses adaptive thinking instead.

Effort is the new control. Haiku 5.5 is the first Haiku-class model with an adjustable effort setting. Its default effort is medium. The levels run from low through medium, high, xhigh and max. Think of effort as the time a student gets on a homework problem. Five minutes gets a quick answer, an hour gets a more careful one, and the time spent sets the bill.

Haiku 5.5 can run with thinking disabled, and the system card reports results in that setting. The tokenizer is updated, so the same text may cost slightly more tokens than on Haiku 4.5. Haiku 4.5 is listed as legacy, with retirement not sooner than October 15, 2026. Anthropic published a migration guide with the release. This entry did not review its breaking-change list, so developers should read it before switching model strings.

## Applications and workflow fit

Anthropic positions Haiku 5.5 for quick, repetitive work such as summaries, compaction, database queries and classification, and for speed-sensitive jobs such as live customer support and browser use. It also pairs with Opus 5.5 and Sonnet 5.5 as a subagent on coding work. A subagent is a junior analyst who pulls the figures from filings while a senior partner writes the memo. Compaction works like condensing a week of meeting notes into a one-page brief: the working history gets shorter so the agent can keep going.

Customer figures come from early testing and are vendor-reported, like the benchmarks. HubSpot reported 92.8% averaged over three runs on its CRM simulated-portal suite, the best score it has seen on that suite. AlphaSense reported a statistically significant gain over Haiku 4.5 on 400 queries, at 0.84 against 0.76. Box said Haiku 5.5 scored 11 points above Haiku 4.5 at about half the latency, and Asana reported more than a 30% cut in task-completion latency.

## Bottom line

Claude Haiku 5.5 is Anthropic's case that the cheapest tier can carry real agent work when the job is narrow and the volume is high. The launch numbers show large gains over Haiku 4.5, a 1M-token window in place of 200K, and a 90 percent per-token cut on prompts up to 100,000 tokens. The same table places it behind Sonnet 5.5 on every row, and Anthropic says as much. Every benchmark figure is vendor-reported, the headline 75 percent saving depends on prompt length, and the system card records regressions in sensitive areas that developers have to cover themselves. Haiku 5.5 takes the volume. The judgment calls still go to Sonnet and Opus.
$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:claude-haiku-5-5',
  'glossary',
  'claude-haiku-5-5',
  '',
  'glossary/claude-haiku-5-5',
  'Claude Haiku 5.5',
  'Anthropic''s October 2026 small-tier model for high-volume work, priced at $0.10/$0.50 per million tokens for prompts up to 100,000 tokens, with a 1M context window and an average cost cut of about 75 percent versus Haiku 4.5 that Anthropic reports.',
  claude_haiku_5_5_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', claude_haiku_5_5_entry.body)),
  '[]'::jsonb,
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array('claude-haiku-4-5', 'claude-sonnet-5-5', 'claude-opus-5-5', 'anthropic', 'large-language-model', 'context-window'),
    'seoDescription', 'Anthropic''s October 2026 small-tier model for high-volume work, with a 1M context window, $0.10/$0.50 pricing up to 100K tokens, and benchmarks against Haiku 4.5, GPT-6 Luna and Sonnet 5.5.',
    'seoKeywords', jsonb_build_array('Claude Haiku 5.5 model card', 'Claude Haiku 5.5 specs', 'Claude Haiku 5.5 benchmarks', 'Claude Haiku 5.5 pricing')
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-10-08',
  0
FROM claude_haiku_5_5_entry
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
