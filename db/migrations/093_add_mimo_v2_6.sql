-- Add Xiaomi's MiMo V2.6 series and the MiMo V2.6 Pro flagship to the
-- glossary, following the pattern set by 090 (GPT-6.1 Sol).
--
-- Sources: Xiaomi's September 22, 2026 MiMo-V2.6 announcement post, the
-- mimo.mi.com model pages for Pro and Flash, the MiMo-V2.6-Pro-RL and
-- MiMo-V2.6-Flash-RL model cards on Hugging Face, the accompanying technical
-- report, and the MiMo-V2.6-Pro page on Artificial Analysis.
--
-- Two terms rather than one, because the series framing and the flagship
-- answer different questions. The series page covers the lineup, the
-- reinforcement learning run, and the release model. The Pro page covers
-- architecture, benchmarks, pricing, and the cyber numbers, which are too
-- large to sit inside a series overview without being understated.
--
-- Footnote handling, all carried in the body so the rendered pages keep the
-- caveats:
-- 1. Every benchmark in both tables is Xiaomi's own transcription. Most of the
--    compared labs do not publish the underlying figures, and Xiaomi leaves
--    several cells blank, so cross-vendor cells are vendor-reported.
-- 2. MiMo Code Bench, MiMo Visual Coding, and MiMo Cyber Bench are Xiaomi
--    in-house benchmarks and cannot be reproduced by anyone else.
-- 3. The announcement post is dated September 22, 2026; Artificial Analysis
--    records the release as September 21, 2026. Both are stated in the body.
-- 4. The post's prose and its own per-step table disagree slightly on the
--    DeepSWE endpoints (48.8 to 65.68 and 58.4 to 72.57 in prose, 48.7 and
--    65.7 in the table). The body notes this rather than silently picking one.
-- 5. The prose sentence giving 25% and 12% pass-rate gains does not map each
--    figure to a specific model, so the body does not attribute them.
-- 6. The MIT license and the Apache-2.0 RL datasets are not stated on either
--    model page; they come from the Hugging Face repositories and are
--    attributed there in the body.
-- 7. Xiaomi's claim of "the strongest open-source model to date" is reported
--    as Xiaomi's claim. Artificial Analysis independently puts Pro at 46,
--    ahead of GLM-5.3 at 45 and Kimi K3 at 44, and also describes the model as
--    slow and verbose, generating 140M tokens across its index run.
--
-- "xiaomi" is used in prose but not in relatedTerms, because there is no
-- Xiaomi term in the glossary. Both terms cross-link to each other directly
-- in relatedTerms rather than via a separate UPDATE, since they are inserted
-- in the same migration.

WITH mimo_v2_6_entry AS (
  SELECT $body$
MiMo V2.6 is Xiaomi's September 2026 model series: two [large language models](/glossary/large-language-model) built on the same multimodal architecture, plus a faster serving variant of the flagship, released together with the weights, the technical report, the reinforcement learning code, and the training environments behind them. [MiMo V2.6 Pro](/glossary/mimo-v2-6-pro) is the capability flagship at 1.02 trillion total parameters, MiMo V2.6 Flash is the 309-billion model aimed at the cost end, and MiMo V2.6 Pro UltraSpeed serves the Pro weights at up to 20 times the standard output speed for ten times the price. The two models are not equally multimodal in practice. Xiaomi markets the series as omnimodal, but Artificial Analysis's specification table lists Pro as accepting text, image, speech and video input while listing Flash as accepting text and image only.

Two dates are in circulation. Xiaomi's announcement post is dated September 22, 2026, and Artificial Analysis records the release as September 21, 2026. The weights ship as `XiaomiMiMo/MiMo-V2.6-Pro-RL` and `XiaomiMiMo/MiMo-V2.6-Flash-RL` on Hugging Face under the MIT License, with the RL datasets and environments released under Apache-2.0. Publishing the training stack alongside the weights is the unusual part. Most labs release one or the other, and the best-resourced labs release neither.

## What is in the series

| Model | Total / active parameters | Role | API model ID |
| --- | --- | --- | --- |
| MiMo V2.6 Pro | 1.02T / 42B | Flagship, most capable to date | `mimo-v2.6-pro` |
| MiMo V2.6 Flash | 309B / 15B | Intelligence-to-cost balance | `mimo-v2.6-flash` |
| MiMo V2.6 Pro UltraSpeed | 1.02T / 42B | Same weights, up to 20x output speed | `mimo-v2.6-pro-ultraspeed` |

Xiaomi also published MiMo-V2.6-Distill-Qwen-9B, a small [distilled](/glossary/model-distillation) model that served as the starting point for reinforcement learning rather than as a shipping SKU. The `-RL` suffix on the Hugging Face repositories names the training stage the checkpoint ended in, not a distinct model. API model IDs are lowercase and dotted, the endpoint is `https://api.xiaomimimo.com/v1`, and it speaks both the OpenAI and Anthropic protocols, so migrating an existing project is a base URL and model name change.

## The training run is the actual story

Xiaomi framed the release as a step along what it calls the RSI path, and the reinforcement learning run is where the engineering content is. The company says each model completed 30 [RL](/glossary/reinforcement-learning) steps in under six days across roughly 750,000 trajectories, at a reported cost of about $0.85 million for Flash and $2.62 million for Pro. The production run was [streamed live](https://mimo.xiaomi.com/rl) as it happened, and the full 30-step curves were published for DeepSWE v1.1, AutomationBench v1.0.6, and MiMo Visual Coding.

Three axes were scaled. Batches reached 1,568 samples per update on a fully asynchronous architecture, training at up to 1M context length and 3.5 to 3.7 billion tokens per step. The task suite spanned coding, general agents, visual, and cyber work, deliberately mixed across several [agent harnesses](/glossary/agent-harness) so that gains in one capability reinforced the others. And the grader was given more compute, which produced the most transferable idea in the release.

### Groupwise agentic grading

A binary pass/fail reward cannot rank two solutions that both pass, so a correct rollout carries no information about whether it was better than the solution sitting next to it. That is a real ceiling on self-improvement: if the model is only ever told it succeeded, nothing in the signal pushes it toward the shorter or cleaner path.

Xiaomi's answer operates within each group of rollouts. Groupwise Reward Synthesis builds task-specific rubrics offline by contrasting rollouts against each other, then fuses rubric quality with test outcomes. Groupwise Advantage Redistribution ranks the passing trajectories online and moves advantage toward the higher-quality ones. Because the reference is the policy's own samples rather than an external gold answer, the loop closes on itself, which is exactly the condition under which reward hacking becomes the central risk. The release is candid about that, documenting a four-layer defense spanning reward design, adversarial evaluation, anomaly detection, and cross-checking between verifiers, and freezing the router as the run scaled up to suppress training drift.

Self-improvement against your own distribution is a research bet rather than a settled result. It can efficiently optimize for whatever the verifier is able to recognize, and the verifier here was trained by the same process. That is the question worth keeping in mind when reading the headline index score.

## Benchmarks

Xiaomi's published comparison, against its own previous generation and the closed frontier:

| Benchmark | MiMo V2.6 Pro | MiMo V2.6 Flash | MiMo V2.5 Pro | Claude Opus 5 | GPT 6 Astra | DeepSeek V4.1 Flash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSWE v1.1 | 71.9 | 67.9 | 19.0 | 74.0 | 74.0 | 74.2 |
| ProgramBench | 26.5 | 26.0 | 12.5 | 37.0 | n/a | 20.3 |
| MiMo Code Bench (in-house) | 63.2 | 61.2 | 40.4 | 68.6 | 61.4 | 60.2 |
| GDPval 2.1 (AA, Elo) | 1673 | n/a | 1107 | 1708 | 1542 | 1600 |
| Toolathlon-verified | 76.9 | 73.6 | 49.1 | 80.6 | n/a | n/a |
| AutomationBench v1.0.6 | 53.1 | 52.3 | 16.0 | 50.3 | 52.0 | 54.8 |
| Agents' Last Exam | 31.6 | 27.6 | 13.2 | 31.6 | 34.2 | 31.8 |
| Terminal-Bench 4.0 | 34.9 | 28.8 | 1.5 | 49.0 | 59.6 | 26.8 |
| JobBench | 62.0 | 61.2 | 25.0 | 65.7 | n/a | 45.8 |
| MiMo Visual Coding (in-house) | 72.3 | 71.5 | n/a | 70.0 | 82.2 | 70.6 |
| CyberGym | 94.0 | 95.1 | 40.0 | n/a | n/a | 88.1 |
| ExploitGym | 17.8 | 6.0 | 0.1 | 22.1 | 42.4 | 15.3 |

Three of these rows are Xiaomi's own in-house benchmarks that nobody outside the company can reproduce. The competitor columns are Xiaomi's transcription, and most of those labs do not publish the underlying numbers, so the cross-vendor cells are vendor-reported. The generation-on-generation jump is the clearest signal in the table: V2.5-Pro scores 19.0 on DeepSWE v1.1, 16.0 on AutomationBench, and 1.5 on Terminal-Bench 4.0, against 71.9, 53.1, and 34.9 for V2.6-Pro. Those are not incremental post-training gains.

Xiaomi's own curves are published per step, and they are not monotonic. Pro's DeepSWE score moves from 58.4 at step 1 to 72.6 at step 30 with dips in between, and the same is true of every curve. Note also that the prose and the step table disagree slightly: the text reports the DeepSWE rise as 48.8 to 65.68 and 58.4 to 72.57, while the table records 48.7 and 65.7 for the same two endpoints. Neither reading changes the conclusion, but the numbers are not perfectly consistent within the post.

The cyber rows are the ones to read twice, and they are covered on the [Pro page](/glossary/mimo-v2-6-pro) because the series framing understates them.

Artificial Analysis, running independently, scores MiMo V2.6 Pro at 46 on Intelligence Index v4.3.2, the highest open-weights result, ahead of GLM-5.3 at 45 and Kimi K3 at 44. Xiaomi cites 46.32; the gap to 46 is rounding on the same measurement. It scores Flash at 38, at $0.13 and $0.06 of index cost per task respectively. Both figures come with a speed warning that matters for cost modelling: Pro generates 41.1 tokens per second with a 4.24-second time to first token, and while Flash is faster at 55 tokens per second, it is the more verbose of the two, spending 240M output tokens on its index run against Pro's 140M and a 140M median for the comparison set. That runs against Xiaomi's own description of Flash as the token-efficient end of the series.

## Pricing

API pricing is unchanged from the V2.5 series, which is the commercial point. Better scores at last quarter's prices is what moves the intelligence-versus-cost frontier outward.

| USD per 1M tokens | Input (cache hit) | Input (cache miss) | Output |
| --- | ---: | ---: | ---: |
| MiMo V2.6 Flash | $0.0028 | $0.14 | $0.28 |
| MiMo V2.6 Pro | $0.0036 | $0.435 | $0.87 |
| MiMo V2.6 Pro UltraSpeed | $0.036 | $4.35 | $8.70 |

The cached-input rate is what makes these models viable for long [agent](/glossary/ai-agent) loops. At $0.0036 per million cached tokens, a coding agent that re-reads the same repository every turn pays close to nothing for context it has already seen, and cache writes are free for a limited time. The roughly 120x spread between cache hit and cache miss on Pro is the widest relative gap at this performance level, so caching discipline matters more here than with most models.

For self-hosting, MIT weights plus published SGLang and vLLM recipes make both models unusually straightforward to run yourself. That is a real advantage over any closed frontier model, and it comes with the serving cost of a 1.02T [mixture of experts](/glossary/mixture-of-experts) with 42B active parameters.

## Availability

MiMo V2.6 Pro and Flash are available through Xiaomi's AI Studio, MiMo Code, MiMo Desktop, the MiMo API platform, and OpenRouter. MiMo Desktop left early access with its first official release, with both models built in. A Token Plan subscription covers Pro, Flash, and the V2.5 suite at better unit economics than pay-as-you-go at high volume, and a MiMo Claw client is offered on a limited-time free trial.

## Bottom line

MiMo V2.6 is a well-executed open release: MIT weights, the RL code, the training environments, the technical report, live-streamed training curves, and a top independent intelligence index score at prices that were not raised. The groupwise grading work is the most interesting idea to come out of it.

Three limits are worth carrying forward. The absolute frontier still belongs to others, with DeepSWE, Terminal-Bench 4.0, ProgramBench, and ExploitGym all going to competitors, in one case by 25 points. Three headline benchmarks are Xiaomi's own and unreproducible, and most competitor cells are second-hand. And no safety evaluation accompanied the release, which for MIT-licensed weights scoring 94.0 on CyberGym is not a minor omission.
$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:mimo-v2-6',
  'glossary',
  'mimo-v2-6',
  '',
  'glossary/mimo-v2-6',
  'MiMo V2.6',
  'Xiaomi''s September 2026 open-weight model series: a 1.02T-parameter omnimodal flagship, a 309B Flash variant, and MIT-licensed weights released with the RL code and training environments.',
  mimo_v2_6_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', mimo_v2_6_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing MiMo-V2.6 series', 'url', 'https://mimo.xiaomi.com/mimo-v2-6'),
    jsonb_build_object('title', 'MiMo-V2.6-Pro model page', 'url', 'https://mimo.mi.com/models/en-US/mimo-v2.6-pro'),
    jsonb_build_object('title', 'MiMo-V2.6-Flash model page', 'url', 'https://mimo.mi.com/models/en-US/mimo-v2.6-flash'),
    jsonb_build_object('title', 'MiMo-V2.6-Pro on Artificial Analysis', 'url', 'https://artificialanalysis.ai/models/mimo-v2-6-pro')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array('mimo-v2-6-pro', 'large-language-model', 'open-weight-model', 'open-source', 'mixture-of-experts', 'reinforcement-learning', 'agent-harness', 'agentic-ai', 'ai-agent', 'context-window', 'artificial-analysis', 'benchmark', 'kimi-k3', 'glm-5-3', 'api'),
    'seoDescription', 'MiMo V2.6 explained: Xiaomi''s Sept 2026 open-weight series, 1.02T/42B Pro and 309B Flash, MIT weights, agentic RL grading, benchmarks, and pricing.',
    'seoKeywords', jsonb_build_array('MiMo V2.6', 'Xiaomi MiMo V2.6', 'MiMo V2.6 Pro', 'MiMo V2.6 Flash', 'MiMo V2.6 benchmarks', 'MiMo V2.6 pricing', 'MiMo V2.6 open weights', 'mimo-v2-6', 'Xiaomi MiMo model', 'MiMo V2.6 reinforcement learning', 'MiMo groupwise reward synthesis', 'MiMo V2.6 vs Kimi K3')
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-09-30',
  0
FROM mimo_v2_6_entry
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


WITH mimo_v2_6_pro_entry AS (
  SELECT $body$
MiMo V2.6 Pro is Xiaomi's flagship [large language model](/glossary/large-language-model), released on September 22, 2026 as the most capable model the company has shipped and the highest-scoring open-weights model on Artificial Analysis's Intelligence Index. It is a sparse [mixture of experts](/glossary/mixture-of-experts) with 1.02 trillion total parameters and 42 billion active, natively omnimodal across text, image, video, and audio, which Artificial Analysis lists as text, image, speech and video input, and MIT licensed on Hugging Face as `XiaomiMiMo/MiMo-V2.6-Pro-RL`. It is part of the [MiMo V2.6 series](/glossary/mimo-v2-6).

The interesting part of the specification is what did not change. V2.5-Pro also ran 42 billion active parameters and cost the same $0.0036/$0.435/$0.87 per million tokens. Pro is about 2 percent larger in total parameters at an identical active compute cost and an identical price. Whatever Xiaomi achieved here came out of training rather than out of more compute per token.

## Core profile

| Specification | MiMo V2.6 Pro |
| --- | --- |
| Developer | Xiaomi MiMo |
| Release date | September 22, 2026 |
| Architecture | Sparse MoE, hybrid sliding-window and global attention |
| Total / active parameters | 1.02T / 42B |
| [Context window](/glossary/context-window) | 1,000,000 tokens |
| Maximum output | 128,000 tokens |
| Inputs | Text, image, video, audio |
| Output | Text |
| Rate limits | 100 requests/minute, 10M tokens/minute |
| Capabilities | Deep thinking, [tool calling](/glossary/tool-calling), streaming, web search, structured output, context caching |
| Weights | `XiaomiMiMo/MiMo-V2.6-Pro-RL`, MIT License |
| API model ID | `mimo-v2.6-pro` |

The 100 requests-per-minute ceiling is worth planning around. It is low for a model at this capability level, and it is a provider choice rather than a property of the model.

## Architecture

| Component | Configuration |
| --- | --- |
| Layers (total / sliding-window / global) | 70 / 60 / 10 |
| Hidden size | 6144 |
| Attention heads, Q / KV (both types) | 128 / 8 |
| Head dimensions, QK / V | 192 / 128 |
| Sliding window size | 128 |
| Routed experts (total / activated) | 384 / 8 |
| Maximum context | 1M tokens |
| Multi-token prediction drafter | 5 sliding-window layers, window 1024 |
| Vision encoder | 681M MiMo ViT, 28 layers, hidden 1280 |
| Audio encoder | 308M AudioTokenizer plus a 127M audio patch encoder |

The backbone is mostly local. Sixty of seventy layers use a sliding window of 128 tokens and ten use global attention. The first Transformer block uses global attention with a dense feed-forward network, and the remaining blocks interleave local and global attention with sparse MoE feed-forward layers and no shared experts. Activating 8 of 384 routed experts per token is what keeps a 1.02T model inside a 42B inference cost, and the narrow global-attention share is what makes a 1M [context window](/glossary/context-window) affordable rather than quadratic.

The [speculative decoder](/glossary/speculative-decoding) is a five-layer sliding-window multi-token-prediction drafter that proposes seven subsequent tokens per forward pass for parallel verification. That is a serving optimization rather than a separate model, and it is the same mechanism the UltraSpeed SKU pushes much harder. Flash is the same design scaled down to 48 layers, 39 sliding-window and 9 global, with 256 routed experts, 309B total, and 15B active.

## Benchmarks

Xiaomi's published results, with its own unreproducible benchmarks marked:

| Benchmark | MiMo V2.6 Pro | MiMo V2.6 Flash | Claude Opus 5 | GPT 5.6 Sol | GPT 6 Astra | DeepSeek V4.1 Flash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSWE v1.1 | 71.9 | 67.9 | 74.0 | n/a | 74.0 | 74.2 |
| ProgramBench | 26.5 | 26.0 | 37.0 | 25.0 | n/a | 20.3 |
| MiMo Code Bench (in-house) | 63.2 | 61.2 | 68.6 | 59.3 | 61.4 | 60.2 |
| GDPval 2.1 (AA, Elo) | 1673 | n/a | 1708 | 1588 | 1542 | 1600 |
| Toolathlon-verified | 76.9 | 73.6 | 80.6 | 74.9 | n/a | n/a |
| AutomationBench v1.0.6 | 53.1 | 52.3 | 50.3 | 45.8 | 52.0 | 54.8 |
| Agents' Last Exam | 31.6 | 27.6 | 31.6 | 30.8 | 34.2 | 31.8 |
| Terminal-Bench 4.0 | 34.9 | 28.8 | 49.0 | 39.9 | 59.6 | 26.8 |
| OSWorld-Verified | 82.0 | 80.8 | 83.4 | 83.0 | n/a | n/a |
| JobBench | 62.0 | 61.2 | 65.7 | 45.4 | n/a | 45.8 |
| MiMo Visual Coding (in-house) | 72.3 | 71.5 | 70.0 | 73.4 | 82.2 | 70.6 |
| CyberGym | 94.0 | 95.1 | n/a | n/a | n/a | 88.1 |
| ExploitGym | 17.8 | 6.0 | 22.1 | 30.3 | 42.4 | 15.3 |
| ExploitBench | 47.9 | 25.3 | 70.0 | 78.5 | 100.0 | n/a |
| SEC Bench Pro | 66.3 | 47.5 | n/a | 79.1 | 85.4 | 62.8 |
| MiMo Cyber Bench (in-house) | 81.7 | 77.2 | n/a | n/a | n/a | 62.7 |

Read this as the vendor's own transcription. Most of those competitors do not publish the underlying numbers, and Xiaomi has left several cells blank while filling others, so the cross-vendor rows are vendor-reported rather than verified. Three rows cannot be reproduced by anyone outside Xiaomi at all.

Where Pro leads the set: AutomationBench v1.0.6 at 53.1, ahead of Claude Opus 5 and effectively level with GPT 6 Astra. Agents' Last Exam at 31.6, tied with Opus 5. OSWorld-Verified at 82.0, close to the best available. And GDPval 2.1 at 1673 Elo, ahead of both GPT 6 Astra and GPT 5.6 Sol on knowledge work.

Where it trails, in one case badly. Terminal-Bench 4.0 is 34.9 against Astra's 59.6, a 25-point gap on the benchmark that tracks general agentic capability most closely. ProgramBench is 26.5 against Opus 5's 37.0. ExploitGym is 17.8 against Astra's 42.4, and ExploitBench is 47.9 against Astra's 100.0. Xiaomi's own overview chart leads with DeepSWE v1.1, and Pro's 71.9 there places it behind DeepSeek V4.1 Flash and behind both Anthropic and OpenAI flagships.

Artificial Analysis, running independently, scores Pro at 46 on Intelligence Index v4.3.2, first among open-weights models ahead of GLM-5.3 at 45 and Kimi K3 at 44. Two observations from that same page matter for cost modelling. The model is slow, and it is verbose, generating 140M tokens across its index run, well above the median of the comparison set. Artificial Analysis measures it at 41.1 output tokens per second and 4.24 seconds to first token, both at the poor end of its open-weights size class, and puts the cost of one index task at $0.13. On a per-token basis Pro is cheap, but a verbose model bills more per completed task than its unit price suggests.

## The cyber numbers

Pro scores 94.0 on CyberGym, 66.3 on SEC Bench Pro, and 81.7 on Xiaomi's in-house MiMo Cyber Bench. On ExploitBench it reaches 47.9, against 100.0 for GPT 6 Astra and 70.0 for Claude Opus 5. Flash scores higher than Pro on CyberGym at 95.1 and far lower on ExploitBench at 25.3, which is itself a reminder that these benchmarks measure specific, narrow capabilities rather than a single cyber level.

These are vulnerability discovery and exploitation evaluations, and they place Pro in a band that better-resourced labs describe with dedicated safeguards. OpenAI classifies a model Critical in cybersecurity capability and gates access to it in phases through its Daybreak program. Anthropic publishes per-model cyber capability evaluations in its system cards. Pro carries no equivalent framing, because no safety evaluation was published alongside it.

That combination is the fact to sit with. These are MIT-licensed weights anyone can download, carrying strong published vulnerability-discovery numbers, with no accompanying risk assessment, deployment guidance, or refusal-behavior evaluation to go with them. The technical report and the RL environments are genuinely open, and the training suite explicitly included cyber tasks. None of that implies intent, and this is a legitimate research artifact. But publishing everything under a permissive license and publishing a capability evaluation are two different decisions, and only the first one was made.

For planning purposes, treat the cyber numbers as a property of the weights rather than of the product. API access routes through Xiaomi's platform, which can apply whatever controls it chooses. Self-hosting the MIT weights applies none of them.

## Pricing

| USD per 1M tokens | Price |
| --- | ---: |
| Input (cache hit) | $0.0036 |
| Input (cache miss) | $0.435 |
| Output | $0.87 |

Cache writes are free for a limited time. China-market pricing is roughly 7x higher in CNY terms at ¥0.025 cache hit, ¥3 cache miss, and ¥6 output. Pricing is identical to V2.5, so the intelligence-versus-cost curve moved without a price change.

The cached-input rate is what makes this practical for [agent](/glossary/ai-agent) loops that re-read the same repository or document set on every turn. A 120x discount against a cache miss means context reuse dominates the bill, and the same workload that caches well and one that does not can differ by two orders of magnitude. On this model, caching discipline matters more than which frontier model you picked.

UltraSpeed serves the same weights at up to 20x output speed for $0.036/$4.35/$8.70, exactly ten times the price. It is a latency purchase rather than a value one.

## Where it fits

Pro is well matched to long-horizon coding agents in real repositories, multimodal document, video, and audio analysis where 1M context and native non-text input matter together, high-volume agent loops that benefit from cheap cached context, and self-hosted deployments where MIT weights and published SGLang and vLLM recipes matter more than holding the highest absolute score. Xiaomi's demonstrations include a materials research co-pilot that searched literature, called open-source simulation tools, set up its own environment, and shortlisted MOF candidates for PFAS capture; a Lean 4 formalization of the main theorem in Li and Yorke's Period Three Implies Chaos, over 6,000 lines, verified by Lean's kernel with no unfinished placeholders and no Lean-specific post-training; and Blender-native 3D scene generation from text or reference images.

It is not the obvious choice where frontier agentic performance is the requirement and Terminal-Bench 4.0 is your proxy, because Astra leads by 25 points. It is a poor default for security-sensitive autonomous agents, for the reason above. And for the hardest research and top-end business workflow automation, the composite index places GPT 6 Astra, Claude Opus 5.5, and Claude Sonnet 5.5 above it, at multiples of the price.

## Bottom line

MiMo V2.6 Pro is the best open-weights model available, and the reason is not the architecture. It is that Xiaomi moved a 42-billion-active-parameter model to 46 on the Artificial Analysis index without changing the active compute or the price, published the reinforcement learning code and environments that did it, and licensed the result under MIT.

The gaps are real. It trails the frontier by 25 points on Terminal-Bench 4.0 and by 25 on ExploitGym, it is slow and verbose, and its 100 requests-per-minute ceiling is tight for production traffic. The benchmark table mixes Xiaomi's unreproducible in-house results with second-hand competitor figures. Most consequentially, the strongest capability signal in the release, high vulnerability discovery scores on downloadable MIT weights, arrived with no published safety evaluation, and that is the one thing to weigh before deploying it.
$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:mimo-v2-6-pro',
  'glossary',
  'mimo-v2-6-pro',
  '',
  'glossary/mimo-v2-6-pro',
  'MiMo V2.6 Pro',
  'Xiaomi''s September 2026 open-weight flagship: a 1.02T-parameter sparse MoE with 42B active, 1M context, native video and audio input, and the top Artificial Analysis score among open models.',
  mimo_v2_6_pro_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', mimo_v2_6_pro_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'MiMo-V2.6-Pro model page', 'url', 'https://mimo.mi.com/models/en-US/mimo-v2.6-pro'),
    jsonb_build_object('title', 'Introducing MiMo-V2.6 series', 'url', 'https://mimo.xiaomi.com/mimo-v2-6'),
    jsonb_build_object('title', 'MiMo-V2.6-Pro on Artificial Analysis', 'url', 'https://artificialanalysis.ai/models/mimo-v2-6-pro')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array('mimo-v2-6', 'large-language-model', 'mixture-of-experts', 'sparse-model', 'mixture-of-experts-routing', 'speculative-decoding', 'context-window', 'agent-harness', 'ai-agent', 'agentic-ai', 'inference-time-compute', 'artificial-analysis', 'open-weight-model', 'benchmark', 'kimi-k3', 'gpt-6-1-sol', 'claude-opus-5', 'api'),
    'seoDescription', 'MiMo V2.6 Pro explained: 1.02T/42B sparse MoE, 1M context, omnimodal input, Artificial Analysis index 46, MIT weights, and cyber benchmark scores.',
    'seoKeywords', jsonb_build_array('MiMo V2.6 Pro', 'Xiaomi MiMo V2.6 Pro', 'MiMo V2.6 Pro benchmarks', 'MiMo V2.6 Pro pricing', 'MiMo V2.6 Pro parameters', 'MiMo V2.6 Pro context window', 'mimo-v2-6-pro', 'mimo-v2.6-pro', 'MiMo V2.6 Pro vs GPT 6 Astra', 'MiMo V2.6 Pro vs Claude Opus 5', 'MiMo V2.6 Pro MIT license', 'MiMo V2.6 Pro CyberGym')
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-09-30',
  0
FROM mimo_v2_6_pro_entry
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
