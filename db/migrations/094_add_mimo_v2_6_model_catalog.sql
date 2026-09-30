-- Add Xiaomi's MiMo V2.6 series to the model catalog, following the pattern set
-- by 091 (GPT-6.1 Sol). Migration 093 published the MiMo V2.6 and MiMo V2.6 Pro
-- glossary pages from the same sources, so the specs and caveats here stay
-- consistent with them and the catalog row, the glossary page and any future
-- comparison page share one set of footnotes.
--
-- Sources: Xiaomi's September 22, 2026 MiMo-V2.6 announcement post, the
-- mimo.mi.com model pages for Pro and Flash, the MiMo-V2.6-Pro-RL and
-- MiMo-V2.6-Flash-RL model cards on Hugging Face with the accompanying
-- technical report, the MiMo-V2.6-RL-oss training environments and datasets,
-- and the MiMo-V2.6-Pro page on Artificial Analysis.
--
-- Three catalog rows, not one. The catalog is keyed at SKU level, the same way
-- it carries gpt-5-4 and gpt-5-4-mini separately, and the series ships two
-- distinct models plus a serving variant with its own price.
--
-- MiMo-V2.6-Distill-Qwen-9B is deliberately not seeded. Xiaomi published it as
-- the starting point for the reinforcement learning run rather than as a
-- shipping SKU, so it has no API model ID, no price and no benchmarks. The
-- glossary series page says the same thing.
--
-- Footnote handling, all carried in the row notes so a rendered comparison
-- keeps the caveats:
-- NOTE: no WHERE NOT EXISTS guards on these upserts. A guard makes the
-- INSERT a no-op when the row already exists, which silently defeats the
-- ON CONFLICT DO UPDATE clause below and leaves content corrections
-- unapplied on any re-run.
-- 1. Every benchmark value here is Xiaomi's own transcription. Most of the
--    compared labs do not publish the underlying figures and Xiaomi leaves
--    several cells blank, so cross-vendor cells are vendor-reported. The
--    evaluator is recorded as Xiaomi rather than as the lab whose model it is.
-- 2. MiMo Code Bench, MiMo Visual Coding and MiMo Cyber Bench are Xiaomi
--    in-house benchmarks. Nobody outside the company can reproduce them, so
--    they are named without an "(in-house)" suffix here because the benchmark
--    name itself carries the vendor, and the row notes say so.
-- 3. Two release dates are in circulation. The announcement post is dated
--    September 22, 2026 and Artificial Analysis records September 21, 2026.
--    The catalog stores the post date and the notes carry both.
-- 4. The post's prose and its own per-step table disagree on the DeepSWE
--    endpoints (48.8 to 65.68 and 58.4 to 72.57 in prose, 48.7 and 65.7 in the
--    table). The final figures below are the settled V2.6 values, not the
--    training-curve endpoints, so the discrepancy does not affect them.
-- 5. Xiaomi's transcription of Terminal-Bench 4.0 diverges materially from the
--    vendors' own published values for the same models. It lists GPT 6 Astra at
--    59.6 against the 57.9 stored for Astra from OpenAI, and DeepSeek V4.1
--    Flash at 26.8 against the 31.2 stored from DeepSeek. The MiMo rows store
--    Xiaomi's own numbers and the competitor figures live in the notes only.
--    Competitor rows are not inserted from this table, because that would
--    silently replace vendor-sourced values with second-hand ones. A comparison
--    page joining MiMo against Astra on Terminal-Bench 4.0 will therefore show
--    Xiaomi's 34.9 for Pro against OpenAI's 57.9 for Astra, and the row notes
--    say why those two numbers are not the same measurement.
-- 6. Flash has no GDPval 2.1 figure in Xiaomi's table, so that row is left
--    unseeded for Flash rather than inferred from Pro.
-- 7. The Artificial Analysis Intelligence Index reading of 46 for Pro is a
--    category-level score rather than a single benchmark, so it is carried in
--    the model notes and not normalized into model_benchmarks, where no other
--    model has an index row. This follows 091's treatment of the same figure.
-- 8. The MIT license and the Apache-2.0 RL datasets are not stated on either
--    model page. They come from the Hugging Face repositories and the release
--    notes, and are attributed there.
-- 9. access is open_source rather than open_weights. The catalog already draws
--    that line: unmodified OSI-approved licenses such as Apache-2.0 and MIT are
--    stored as open_source, while Modified MIT on Kimi is stored as open_weights
--    because a modified license is not open source. MiMo is unmodified MIT.
-- 10. UltraSpeed serves the Pro weights at up to 20x output speed for ten times
--    the price. It is seeded with pricing and no benchmark rows, because it has
--    no scores of its own. Its metadata points at the Pro slug rather than
--    duplicating 16 evidence rows for identical weights.
-- 11. No safety evaluation was published with this release. Pro carries
--    strong vulnerability discovery and exploitation scores on downloadable
--    MIT weights, and unlike OpenAI and Anthropic there is no published cyber
--    capability classification, refusal-behavior evaluation or deployment
--    guidance. This is recorded as an observation about what was and was not
--    published, in the model notes and the cyber benchmark rows.
-- 12. Context windows are stated by Xiaomi for all three SKUs. Pro and Flash
--    each carry "Context Window 1M tokens" on their own model pages under
--    Performance, and the Open Platform model list gives all three a 1M context
--    and a 128K maximum output. UltraSpeed's is the same 1M and 128K, which is
--    consistent with it serving the Pro weights.
--
-- 13. UltraSpeed is served on custom terms rather than the shared platform
--    limits. The Open Platform model list leaves its capability column blank,
--    where Pro and Flash are listed with full-modal understanding, text
--    generation, deep thinking, streaming, function call, structured output
--    and web search. It publishes no RPM or TPM figure, offering "customized
--    services available, please contact us" instead of the 100 RPM and 10M TPM
--    that Pro and Flash carry. It does not support the Batch API, which the
--    pricing page states explicitly, and it is not covered by the Token Plan,
--    which covers Pro, Flash and the V2.5 suite. Its modality support is not
--    documented either way. What is published is its context window, its
--    maximum output and its price. The capability rows below therefore record
--    what the documentation says and mark the rest as unstated rather than
--    copying Pro's feature list onto it, since the identical weights do not
--    imply that the serving tier exposes the same features.
--
-- The benchmarks JSON is dollar-quoted rather than single-quoted so the
-- apostrophe in Agents' Last Exam does not need doubling. migrate.mjs splits
-- statements on a semicolon at end of line, and no line inside the dollar-quoted
-- strings contains one.

-- 1. MiMo V2.6 Pro, the capability flagship.
INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at
)
SELECT
  'mimo-v2-6-pro',
  'MiMo V2.6 Pro',
  'Xiaomi MiMo',
  DATE '2026-09-22',
  DATE '2026-09-22',
  'open_source',
  'MiMo V2.6',
  '{
    "Developer": "Xiaomi MiMo",
    "Release date": "2026-09-22",
    "API model ID": "mimo-v2.6-pro",
    "Parameters": "1.02T total, 42B active, sparse MoE with 384 routed experts and 8 activated per token",
    "Context window": "1,000,000 tokens",
    "Max output": "128,000 tokens",
    "Architecture": "70 layers, 60 sliding-window and 10 global, hidden size 6144; 128 query heads and 8 key-value heads per attention type; a 5-layer sliding-window multi-token-prediction drafter proposing 7 tokens per pass",
    "Vision encoder": "681M MiMo ViT, 28 layers, hidden 1280",
    "Audio encoder": "308M AudioTokenizer plus a 127M audio patch encoder",
    "Reasoning / effort": "Deep thinking, on by default",
    "Input / 1M tokens": "$0.435 (cache miss)",
    "Cached input / 1M": "$0.0036 (cache hit, a roughly 120x discount against a miss; cache writes are free for a limited time)",
    "Output / 1M tokens": "$0.87",
    "Text input": "Yes",
    "Image / vision input": "Yes, native",
    "Audio input": "Yes, native",
    "Video input": "Yes, native. Artificial Analysis lists Pro as accepting text, image, speech and video input",
    "File / document input": "Yes, as text, image, audio or video",
    "Text output": "Yes",
    "Image output": "No",
    "Audio output": "No",
    "Video output": "No",
    "Tool / function calling": "Yes, with web search, structured output and context caching; streaming supported",
    "Computer use": "Yes, through tool calling and web search",
    "API access": "Yes, https://api.xiaomimimo.com/v1, speaking both the OpenAI and Anthropic protocols, so migrating an existing project is a base URL and model name change. Rate limits are 100 requests/minute and 10M tokens/minute",
    "Product access": "Xiaomi AI Studio, MiMo Code, MiMo Desktop, the MiMo API platform, and OpenRouter. A Token Plan subscription covers Pro, Flash and the V2.5 suite",
    "Weights / license": "Open source. XiaomiMiMo/MiMo-V2.6-Pro-RL on Hugging Face under the MIT License, with the RL datasets and environments under Apache-2.0 and published SGLang and vLLM recipes",
    "Primary focus": "Long-horizon coding agents in real repositories, natively omnimodal analysis of documents, video and audio, and high-volume agent loops that reuse cached context",
    "Training / post-training": "30 reinforcement learning steps in under six days across roughly 750,000 trajectories, at a reported cost of about $2.62 million. The production run was streamed live and the full 30-step curves were published for DeepSWE v1.1, AutomationBench v1.0.6 and MiMo Visual Coding. Xiaomi''s central idea is groupwise reward synthesis, a second learned grader for tasks where a single program-level check cannot tell a good solution from a lucky one",
    "Efficiency / generation change": "Same per-token price as MiMo V2.5 Pro at identical 42B active parameters, so the gain is training rather than more compute per token. Artificial Analysis scores it 46 on Intelligence Index v4.3.2, the highest open-weights result, ahead of GLM-5.3 at 45 and Kimi K3 at 44, while describing it as slow and verbose with 140M tokens generated across its index run",
    "Safety / approvals": "No safety evaluation was published with the release. There is no cyber capability classification, no refusal-behavior evaluation and no deployment guidance, although the model scores 94.0 on CyberGym and 66.3 on SEC Bench Pro on downloadable MIT weights and the RL environments explicitly included cyber tasks. OpenAI classes comparable capability Critical and gates access in phases; Anthropic publishes per-model cyber evaluations",
    "Known limits": "Trails the frontier by 25 points on Terminal-Bench 4.0 and by 25 on ExploitGym, is slow and verbose, and has a 100 requests-per-minute ceiling that is tight for production traffic. Three of its headline benchmarks are in-house and unreproducible, and most competitor figures in its table are second-hand"
  }'::jsonb,
  '[
    {"title": "MiMo-V2.6-Pro model page", "url": "https://mimo.mi.com/models/en-US/mimo-v2.6-pro"},
    {"title": "Introducing MiMo-V2.6 series", "url": "https://mimo.xiaomi.com/mimo-v2-6"},
    {"title": "MiMo-V2.6-Pro on Artificial Analysis", "url": "https://artificialanalysis.ai/models/mimo-v2-6-pro"}
  ]'::jsonb,
  'Released September 22, 2026 by Xiaomi''s post, which Artificial Analysis records as September 21, 2026. Unmodified MIT weights with the reinforcement learning code, training environments and serving recipes published alongside them, which is the unusual part of the release rather than the architecture. The commercial point is that the per-token price is unchanged from MiMo V2.5 Pro at identical 42B active parameters, so the move to 46 on the Artificial Analysis index came from training and not from more compute per token. It is the highest-scoring open-weights model on that index. It is also a poor default for security-sensitive autonomous agents, because the cyber capability numbers are strong, the weights are downloadable by anyone, and no safety evaluation was published with them. Benchmark values are Xiaomi''s own transcription of a table that mixes unreproducible in-house results with second-hand competitor figures, and Xiaomi''s Terminal-Bench 4.0 figures for competitors diverge from what those vendors published themselves.',
  '{
    "verification_pass": "mimo-v2-6-catalog-2026-09-22",
    "catalog_status": "current",
    "glossary_slug": "mimo-v2-6-pro",
    "series_slug": "mimo-v2-6",
    "lower_cost_sibling": "mimo-v2-6-flash",
    "serving_variant": "mimo-v2-6-pro-ultraspeed",
    "independent_intelligence_index": 46,
    "intelligence_index_version": "v4.3.2",
    "artificial_analysis_output_tokens_per_second": 41.1,
    "artificial_analysis_time_to_first_token_seconds": 4.24,
    "artificial_analysis_index_cost_per_task_usd": 0.13,
    "artificial_analysis_index_output_tokens": "140M",
    "safety_evaluation_published": false,
    "license": "MIT"
  }'::jsonb,
  NOW()
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

-- 2. MiMo V2.6 Flash, the cost-end model in the same series.
INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at
)
SELECT
  'mimo-v2-6-flash',
  'MiMo V2.6 Flash',
  'Xiaomi MiMo',
  DATE '2026-09-22',
  DATE '2026-09-22',
  'open_source',
  'MiMo V2.6',
  '{
    "Developer": "Xiaomi MiMo",
    "Release date": "2026-09-22",
    "API model ID": "mimo-v2.6-flash",
    "Parameters": "309B total, 15B active, sparse MoE with 256 routed experts",
    "Context window": "1,000,000 tokens",
    "Architecture": "The MiMo V2.6 design scaled down: 48 layers, 39 sliding-window and 9 global, with 256 routed experts, 309B total and 15B active",
    "Reasoning / effort": "Deep thinking, on by default",
    "Input / 1M tokens": "$0.14 (cache miss)",
    "Cached input / 1M": "$0.0028 (cache hit)",
    "Output / 1M tokens": "$0.28",
    "Text input": "Yes",
    "Image / vision input": "Yes, native",
    "Audio input": "No. Artificial Analysis lists Flash as accepting text and image input only, against text, image, speech and video for Pro, so the series'' omnimodal framing does not carry over to Flash",
    "Video input": "No, on the same reading. Xiaomi''s series-level marketing describes the architecture as multimodal; its own Flash model page does not claim audio or video input",
    "Text output": "Yes",
    "Tool / function calling": "Yes, with web search, structured output and context caching; streaming supported",
    "API access": "Yes, https://api.xiaomimimo.com/v1, speaking both the OpenAI and Anthropic protocols",
    "Product access": "Xiaomi AI Studio, MiMo Code, MiMo Desktop, the MiMo API platform, and OpenRouter",
    "Weights / license": "Open source. XiaomiMiMo/MiMo-V2.6-Flash-RL on Hugging Face under the MIT License, with the RL datasets and environments under Apache-2.0",
    "Primary focus": "The intelligence-to-cost end of the V2.6 series, at 15B active parameters and roughly a fifth of Pro''s price",
    "Training / post-training": "30 reinforcement learning steps in under six days at a reported cost of about $0.85 million, on the same groupwise reward synthesis approach as Pro, with the same live-streamed and per-step published curves",
    "Safety / approvals": "No safety evaluation was published with the release. Flash scores 95.1 on CyberGym, higher than Pro''s 94.0, while scoring far lower on ExploitBench at 25.3 against Pro''s 47.9"
  }'::jsonb,
  '[
    {"title": "MiMo-V2.6-Flash model page", "url": "https://mimo.mi.com/models/en-US/mimo-v2.6-flash"},
    {"title": "Introducing MiMo-V2.6 series", "url": "https://mimo.xiaomi.com/mimo-v2-6"},
    {"title": "MiMo-V2.6-Flash on Artificial Analysis", "url": "https://artificialanalysis.ai/models/mimo-v2-6-flash"}
  ]'::jsonb,
  'Released September 22, 2026 alongside Pro, at unchanged V2.5-series pricing and 15B active parameters. Flash trails Pro on most of the shared benchmarks, but it is cheaper and the two are not ordered the same way everywhere: Flash scores higher on CyberGym at 95.1 against 94.0 while scoring much lower on ExploitBench at 25.3 against 47.9, which is a reminder that these are narrow capability measurements rather than a single cyber level. Flash has no GDPval 2.1 figure in Xiaomi''s table, so no such row is seeded. Its 1M context window and 128K maximum output are stated on Xiaomi''s own model page and repeated in the Open Platform model list. All benchmark values are Xiaomi''s own transcription and three of them are in-house benchmarks nobody else can reproduce.',
  '{
    "verification_pass": "mimo-v2-6-catalog-2026-09-22",
    "catalog_status": "current",
    "glossary_slug": "mimo-v2-6",
    "series_slug": "mimo-v2-6",
    "capability_sibling": "mimo-v2-6-pro",
    "serving_variant": "mimo-v2-6-pro-ultraspeed",
    "independent_intelligence_index": 38,
    "intelligence_index_version": "v4.3.2",
    "artificial_analysis_output_tokens_per_second": 55,
    "artificial_analysis_index_cost_per_task_usd": 0.06,
    "artificial_analysis_index_output_tokens": "240M",
    "input_modalities": "text, image",
    "safety_evaluation_published": false,
    "license": "MIT"
  }'::jsonb,
  NOW()
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

-- 3. MiMo V2.6 Pro UltraSpeed, a serving variant of the Pro weights at ten
--    times the price. Seeded for the comparison picker and for pricing, with no
--    benchmark rows because the weights and therefore the scores are Pro's.
INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at
)
SELECT
  'mimo-v2-6-pro-ultraspeed',
  'MiMo V2.6 Pro UltraSpeed',
  'Xiaomi MiMo',
  DATE '2026-09-22',
  DATE '2026-09-22',
  'open_source',
  'MiMo V2.6',
  '{
    "Developer": "Xiaomi MiMo",
    "Release date": "2026-09-22",
    "API model ID": "mimo-v2.6-pro-ultraspeed",
    "Parameters": "1.02T total, 42B active. Identical weights to MiMo V2.6 Pro",
    "Context window": "1,000,000 tokens",
    "Reasoning / effort": "Deep thinking, on by default",
    "Input / 1M tokens": "$4.35 (cache miss)",
    "Cached input / 1M": "$0.036 (cache hit)",
    "Output / 1M tokens": "$8.70",
    "Serving change": "Up to 20x the standard output speed, pushing the same multi-token-prediction speculative decoding much harder than the standard Pro endpoint. Treat it as not production-ready: Xiaomi documents almost none of its capability surface, it has no Batch API, no published rate limits and no Token Plan coverage, and Artificial Analysis has no evaluation page for it at all, unlike Pro and Flash",
    "Independent evaluation": "None. Artificial Analysis scores and profiles Pro and Flash but has no page for this SKU, so every performance claim about it traces to Xiaomi",
    "Text input": "Unstated. Xiaomi publishes no capability list for this SKU, and identical weights do not imply the serving tier exposes the same features",
    "Image / vision input": "Unstated, see the text input note",
    "Audio input": "Unstated, see the text input note",
    "Video input": "Unstated, see the text input note",
    "Text output": "Unstated, see the text input note",
    "Tool / function calling": "Unstated. Pro and Flash are documented with streaming, function call, structured output and web search; this SKU''s capability column is blank",
    "Context window / max output": "1M tokens / 128K tokens, per the Open Platform model list",
    "Batch API": "No. The pricing page states this SKU does not support the Batch API, where Pro and Flash are billed at half the real-time rate",
    "Rate limits": "Not published. The Open Platform lists custom terms and a contact link instead of the 100 RPM and 10M TPM that Pro and Flash carry",
    "Token Plan coverage": "No. The Token Plan covers mimo-v2.6-pro, mimo-v2.6-flash and the V2.5 suite",
    "API access": "Yes, https://api.xiaomimimo.com/v1, speaking both the OpenAI and Anthropic protocols",
    "Product access": "Xiaomi MiMo Desktop, which ships with the UltraSpeed mode built in, and the Xiaomi MiMo API platform. MiMo Desktop is the access path the documentation actually pairs this SKU with",
    "Weights / license": "Open source. The XiaomiMiMo/MiMo-V2.6-Pro-RL weights under the MIT License; the variant is a serving configuration rather than a separate checkpoint"
  }'::jsonb,
  '[
    {"title": "Xiaomi MiMo Open Platform model list, which records this SKU''s context window and limits and the absence of a published capability list", "url": "https://mimo.mi.com/docs/en-US/quick-start/summary/model"},
    {"title": "Xiaomi MiMo API pricing, which states this SKU does not support the Batch API", "url": "https://mimo.mi.com/docs/en-US/price/pay-as-you-go"},
    {"title": "Introducing MiMo-V2.6 series", "url": "https://mimo.xiaomi.com/mimo-v2-6"}
  ]'::jsonb,
  'A latency purchase rather than a value one, and a narrower one than the identical weights suggest. The UltraSpeed endpoint serves the MiMo V2.6 Pro weights at up to 20x the standard output speed for exactly ten times the price in every direction, $0.036 cache hit, $4.35 cache miss and $8.70 output. Because the weights are identical, no benchmark rows are seeded: duplicating Pro''s evidence under a second slug would make the same scores look like two independent measurements. Its metadata points at mimo-v2-6-pro. The catch is that Xiaomi documents far less about it than about the two named models. The Open Platform model list leaves its capability column blank rather than listing full-modal understanding, text generation, deep thinking, streaming, function call, structured output and web search, and publishes no RPM or TPM ceiling, offering customized services and a contact form instead of the 100 RPM and 10M TPM on Pro and Flash. It does not support the Batch API, so the half-rate batch path that makes Pro''s agent loops affordable is not available here, and it is not covered by the Token Plan. Xiaomi pairs it with MiMo Desktop, where it ships as the built-in UltraSpeed mode, rather than with the general API and third-party surfaces where Pro and Flash are listed. It is worth choosing when output latency is the binding constraint and paying ten times the per-token cost is cheaper than the wait, and worth skipping whenever the standard endpoint is fast enough, since at equal prices this buys nothing but time.',
  '{
    "verification_pass": "mimo-v2-6-catalog-2026-09-22",
    "catalog_status": "current",
    "glossary_slug": "mimo-v2-6",
    "series_slug": "mimo-v2-6",
    "serving_variant_of": "mimo-v2-6-pro",
    "benchmarks_inherited_from": "mimo-v2-6-pro",
    "license": "MIT"
  }'::jsonb,
  NOW()
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

-- 4. Benchmark evidence. Every value is Xiaomi's own transcription of the table
--    on the MiMo-V2.6-Pro model page, with the evaluator recorded as Xiaomi
--    rather than as the lab whose model a competitor column describes. Three
--    benchmark names are Xiaomi in-house and are marked as such in the row
--    notes, because nobody outside the company can reproduce them.
--
--    Benchmark names and categories follow the existing catalog so a future
--    comparison page joins rows instead of splitting them: DeepSWE v1.1,
--    Terminal-Bench 4.0, CyberGym and SEC-Bench Pro under coding, Toolathlon
--    with version Verified, AutomationBench with version 1.0.6, Agents' Last
--    Exam, OSWorld-Verified, ExploitBench and GDPval-AA with version v2.1 under
--    agentic_computer_use, and JobBench under professional. ProgramBench,
--    ExploitGym and the three MiMo in-house benchmarks are new to the catalog.
--
--    Percentages are stored as score_numeric with score_display carrying the
--    percent sign, matching the dominant convention and 091. The GDPval 2.1 row
--    is Elo, following 091's GDPval-AA v2.1 rows.
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
  {"id":"tq-20260922-mimov26pro-deepswe11","model_slug":"mimo-v2-6-pro","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":71.9,"score_display":"71.9%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Xiaomi's transcription of its own model page table. DeepSeek V4.1 Flash leads at 74.2 in the same table, and Xiaomi's overview chart leads with this benchmark. Flash 67.9, MiMo V2.5 Pro 19.0. The post's prose and its per-step table disagree slightly on the training-curve endpoints (48.8 to 65.68 and 58.4 to 72.57 in prose, 48.7 and 65.7 in the table); the settled value here is unaffected. 091 stored a separate GPT-6.1 Sol reading of 75.2 at high effort and 71.9 at max, so this value is numerically identical to the GPT-6.1 Sol max-effort row while coming from a different evaluator and harness."},
  {"id":"tq-20260922-mimov26pro-programbench","model_slug":"mimo-v2-6-pro","category":"coding","benchmark_name":"ProgramBench","benchmark_version":null,"score_numeric":26.5,"score_display":"26.5%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"New to the catalog. Xiaomi's transcription; Claude Opus 5 leads at 37.0 in the same table, so the model trails the frontier here. Flash 26.0."},
  {"id":"tq-20260922-mimov26pro-mimocodebench","model_slug":"mimo-v2-6-pro","category":"coding","benchmark_name":"MiMo Code Bench","benchmark_version":null,"score_numeric":63.2,"score_display":"63.2%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Xiaomi in-house benchmark, so nobody outside the company can reproduce it. The name carries the vendor, which is why it is stored without an (in-house) suffix. Flash 61.2, MiMo V2.5 Pro 40.4."},
  {"id":"tq-20260922-mimov26pro-gdpvalaa21","model_slug":"mimo-v2-6-pro","category":"agentic_computer_use","benchmark_name":"GDPval-AA","benchmark_version":"v2.1","score_numeric":1673,"score_display":"1673 Elo","score_unit":"Elo","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Elo on Artificial Analysis's GDPval 2.1, as transcribed by Xiaomi. This is the benchmark the model leads the open-weights field on, ahead of GPT 6 Astra at 1542 in the same table. Xiaomi transcribes GPT 6 Astra at 1542 and GPT-6.1 Sol is stored at 1575 from Artificial Analysis directly, and Claude Opus 5 at 1708, so this is a mid-frontier knowledge-work reading rather than a lead. Flash reports n/a in Xiaomi's table, so no Flash row is seeded. 091 stored GPT-6 Astra's GDPval-AA v2.1 from OpenAI's own figures, which is the value a comparison page will join against."},
  {"id":"tq-20260922-mimov26pro-toolathlon-verified","model_slug":"mimo-v2-6-pro","category":"agentic_computer_use","benchmark_name":"Toolathlon","benchmark_version":"Verified","score_numeric":76.9,"score_display":"76.9%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Xiaomi's transcription. Claude Opus 5 leads at 80.6 in the same table. Flash 73.6, MiMo V2.5 Pro 49.1."},
  {"id":"tq-20260922-mimov26pro-automationbench106","model_slug":"mimo-v2-6-pro","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":"1.0.6","score_numeric":53.1,"score_display":"53.1%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"One of the three benchmarks Xiaomi published full 30-step training curves for. Effectively level with GPT 6 Astra at 52.0 in the same table, and the stored GPT-6.1 Sol row for this benchmark is 36.1 at max effort. Flash 52.3, MiMo V2.5 Pro 16.0. The version is recorded as 1.0.6 without a v prefix, matching 091 and the other recent migrations."},
  {"id":"tq-20260922-mimov26pro-agentslastexam","model_slug":"mimo-v2-6-pro","category":"agentic_computer_use","benchmark_name":"Agents' Last Exam","benchmark_version":null,"score_numeric":31.6,"score_display":"31.6%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Xiaomi's transcription. Tied with Claude Opus 5 at 31.6 in the same table, with GPT 6 Astra at 34.2. Flash 27.6."},
  {"id":"tq-20260922-mimov26pro-terminalbench40","model_slug":"mimo-v2-6-pro","category":"coding","benchmark_name":"Terminal-Bench 4.0","benchmark_version":null,"score_numeric":34.9,"score_display":"34.9%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"The largest gap in the table, and the one that matters most: Xiaomi transcribes GPT 6 Astra at 59.6, a 25-point lead, on the benchmark that tracks general agentic capability most closely. Note that Xiaomi's Astra figure of 59.6 is not the same measurement as the 57.9 stored for Astra from OpenAI's own numbers, and Xiaomi's DeepSeek V4.1 Flash figure of 26.8 is against the 31.2 stored from DeepSeek. Those competitor values are deliberately not inserted from this table, since that would replace vendor-sourced numbers with second-hand ones, so a comparison page will show this 34.9 against OpenAI's 57.9 for Astra. Flash 28.8, MiMo V2.5 Pro 1.5."},
  {"id":"tq-20260922-mimov26pro-osworldverified","model_slug":"mimo-v2-6-pro","category":"agentic_computer_use","benchmark_name":"OSWorld-Verified","benchmark_version":null,"score_numeric":82.0,"score_display":"82.0%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Stored under the OSWorld-Verified name already in the catalog rather than Xiaomi's label, so a comparison page joins against the 22 existing rows. 091 separately stores GPT-6.1 Sol on OSWorld 2.0 with version offline, partial, which is a different and harder protocol and is deliberately not merged. Claude Opus 5 leads at 83.4 in the same table. Flash 80.8."},
  {"id":"tq-20260922-mimov26pro-jobbench","model_slug":"mimo-v2-6-pro","category":"professional","benchmark_name":"JobBench","benchmark_version":null,"score_numeric":62.0,"score_display":"62.0%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Xiaomi's transcription. Claude Opus 5 leads at 65.7 in the same table. Flash 61.2, MiMo V2.5 Pro 25.0."},
  {"id":"tq-20260922-mimov26pro-mimovisualcoding","model_slug":"mimo-v2-6-pro","category":"multimodal","benchmark_name":"MiMo Visual Coding","benchmark_version":null,"score_numeric":72.3,"score_display":"72.3%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Xiaomi in-house benchmark, one of the three with published 30-step training curves, so nobody outside the company can reproduce it. GPT 6 Astra leads at 82.2 in the same table. Flash 71.5."},
  {"id":"tq-20260922-mimov26pro-cybergym","model_slug":"mimo-v2-6-pro","category":"coding","benchmark_name":"CyberGym","benchmark_version":null,"score_numeric":94.0,"score_display":"94.0%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Vulnerability discovery, and the number that makes the missing safety evaluation consequential: these are downloadable MIT weights, so anyone can run them. No cyber capability classification, refusal-behavior evaluation or deployment guidance was published with the release, unlike OpenAI, which classes comparable capability Critical and gates access in phases through its Daybreak program, and Anthropic, which publishes per-model cyber evaluations. Flash scores higher at 95.1, which is a reminder that these are narrow capability measurements rather than a single cyber level. Stored as coding to match the 3 existing CyberGym rows so a joined comparison does not split them across categories."},
  {"id":"tq-20260922-mimov26pro-exploitgym","model_slug":"mimo-v2-6-pro","category":"coding","benchmark_name":"ExploitGym","benchmark_version":null,"score_numeric":17.8,"score_display":"17.8%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"New to the catalog. Exploitation rather than discovery, and the second-largest gap in the table: GPT 6 Astra leads at 42.4, another 25-point margin. Flash 6.0, MiMo V2.5 Pro 0.1."},
  {"id":"tq-20260922-mimov26pro-exploitbench","model_slug":"mimo-v2-6-pro","category":"agentic_computer_use","benchmark_name":"ExploitBench","benchmark_version":null,"score_numeric":47.9,"score_display":"47.9%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Stored under agentic_computer_use to match the single existing ExploitBench row, which is GPT 6 Astra's 100, so a joined comparison keeps one row group. The gap here is the widest in the table. Flash 25.3."},
  {"id":"tq-20260922-mimov26pro-secbenchpro","model_slug":"mimo-v2-6-pro","category":"coding","benchmark_name":"SEC-Bench Pro","benchmark_version":null,"score_numeric":66.3,"score_display":"66.3%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Vulnerability discovery on MIT weights with no published safety evaluation, alongside the CyberGym row. Stored as coding to match the 8 existing SEC-Bench Pro rows. GPT 6 Astra transcribes at 85.4 in the same table. Flash 47.5."},
  {"id":"tq-20260922-mimov26pro-mimocyberbench","model_slug":"mimo-v2-6-pro","category":"coding","benchmark_name":"MiMo Cyber Bench","benchmark_version":null,"score_numeric":81.7,"score_display":"81.7%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Xiaomi in-house cyber benchmark, so it cannot be reproduced and cannot be compared against anything outside the table. Flash 77.2."},
  {"id":"tq-20260922-mimov26flash-deepswe11","model_slug":"mimo-v2-6-flash","category":"coding","benchmark_name":"DeepSWE v1.1","benchmark_version":null,"score_numeric":67.9,"score_display":"67.9%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Flash column of the same Xiaomi table that supplies the Pro rows. Trails Pro at 71.9 and leads nothing: DeepSeek V4.1 Flash is at 74.2 in the same table. Xiaomi's own overview chart leads with this benchmark."},
  {"id":"tq-20260922-mimov26flash-programbench","model_slug":"mimo-v2-6-flash","category":"coding","benchmark_name":"ProgramBench","benchmark_version":null,"score_numeric":26.0,"score_display":"26.0%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Flash column of the same Xiaomi table. Against Pro 26.5 and Claude Opus 5 37.0."},
  {"id":"tq-20260922-mimov26flash-mimocodebench","model_slug":"mimo-v2-6-flash","category":"coding","benchmark_name":"MiMo Code Bench","benchmark_version":null,"score_numeric":61.2,"score_display":"61.2%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Flash column of the same Xiaomi table. Xiaomi in-house benchmark, not reproducible outside the company. Against Pro 63.2."},
  {"id":"tq-20260922-mimov26flash-toolathlon-verified","model_slug":"mimo-v2-6-flash","category":"agentic_computer_use","benchmark_name":"Toolathlon","benchmark_version":"Verified","score_numeric":73.6,"score_display":"73.6%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Flash column of the same Xiaomi table. Against Pro 76.9 and Claude Opus 5 80.6."},
  {"id":"tq-20260922-mimov26flash-automationbench106","model_slug":"mimo-v2-6-flash","category":"agentic_computer_use","benchmark_name":"AutomationBench","benchmark_version":"1.0.6","score_numeric":52.3,"score_display":"52.3%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Flash column of the same Xiaomi table. Against Pro 53.1, so Flash loses on the benchmark that tracks business workflow automation despite costing roughly a fifth as much, which is the clearest example of the two models not being ordered the same way everywhere."},
  {"id":"tq-20260922-mimov26flash-agentslastexam","model_slug":"mimo-v2-6-flash","category":"agentic_computer_use","benchmark_name":"Agents' Last Exam","benchmark_version":null,"score_numeric":27.6,"score_display":"27.6%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Flash column of the same Xiaomi table. Against Pro 31.6."},
  {"id":"tq-20260922-mimov26flash-terminalbench40","model_slug":"mimo-v2-6-flash","category":"coding","benchmark_name":"Terminal-Bench 4.0","benchmark_version":null,"score_numeric":28.8,"score_display":"28.8%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Flash column of the same Xiaomi table. Against Pro 34.9 and MiMo V2.5 Pro 1.5, so the generation-on-generation gain on this benchmark is the largest anywhere in the series. Xiaomi's competitor transcriptions on this benchmark diverge from the vendors' own published values, as the Pro row notes."},
  {"id":"tq-20260922-mimov26flash-osworldverified","model_slug":"mimo-v2-6-flash","category":"agentic_computer_use","benchmark_name":"OSWorld-Verified","benchmark_version":null,"score_numeric":80.8,"score_display":"80.8%","score_unit":"percent","tools":true,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Flash column of the same Xiaomi table. Against Pro 82.0 and Claude Opus 5 83.4. Stored under the catalog's existing OSWorld-Verified name and left distinct from the OSWorld 2.0 offline partial rows 091 seeded for the OpenAI models."},
  {"id":"tq-20260922-mimov26flash-jobbench","model_slug":"mimo-v2-6-flash","category":"professional","benchmark_name":"JobBench","benchmark_version":null,"score_numeric":61.2,"score_display":"61.2%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Flash column of the same Xiaomi table. Against Pro 62.0 and Claude Opus 5 65.7."},
  {"id":"tq-20260922-mimov26flash-mimovisualcoding","model_slug":"mimo-v2-6-flash","category":"multimodal","benchmark_name":"MiMo Visual Coding","benchmark_version":null,"score_numeric":71.5,"score_display":"71.5%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Flash column of the same Xiaomi table. Xiaomi in-house benchmark, not reproducible outside the company. Against Pro 72.3."},
  {"id":"tq-20260922-mimov26flash-cybergym","model_slug":"mimo-v2-6-flash","category":"coding","benchmark_name":"CyberGym","benchmark_version":null,"score_numeric":95.1,"score_display":"95.1%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Flash scores higher than Pro on this row, at 95.1 against 94.0, and far lower on ExploitBench. That is the clearest evidence that these narrow cyber benchmarks do not measure one underlying capability, and it is why the model-level cyber gap between the two SKUs should not be read as a ranking. Downloadable MIT weights with no published safety evaluation, as on the Pro row."},
  {"id":"tq-20260922-mimov26flash-exploitgym","model_slug":"mimo-v2-6-flash","category":"coding","benchmark_name":"ExploitGym","benchmark_version":null,"score_numeric":6.0,"score_display":"6.0%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Flash column of the same Xiaomi table. Against Pro 17.8 and MiMo V2.5 Pro 0.1. Exploitation rather than discovery, and the row where Flash's drop from Pro is largest in relative terms."},
  {"id":"tq-20260922-mimov26flash-exploitbench","model_slug":"mimo-v2-6-flash","category":"agentic_computer_use","benchmark_name":"ExploitBench","benchmark_version":null,"score_numeric":25.3,"score_display":"25.3%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Flash column of the same Xiaomi table. Against Pro 47.9, while scoring higher than Pro on CyberGym, which is the clearest evidence that the two cyber rows measure different things."},
  {"id":"tq-20260922-mimov26flash-secbenchpro","model_slug":"mimo-v2-6-flash","category":"coding","benchmark_name":"SEC-Bench Pro","benchmark_version":null,"score_numeric":47.5,"score_display":"47.5%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Flash column of the same Xiaomi table. Against Pro 66.3."},
  {"id":"tq-20260922-mimov26flash-mimocyberbench","model_slug":"mimo-v2-6-flash","category":"coding","benchmark_name":"MiMo Cyber Bench","benchmark_version":null,"score_numeric":77.2,"score_display":"77.2%","score_unit":"percent","tools":null,"reasoning_effort":null,"harness":null,"evaluator":"Xiaomi","evaluation_date":"2026-09-22","source":"https://mimo.mi.com/models/en-US/mimo-v2.6-pro","notes":"Flash column of the same Xiaomi table. Xiaomi in-house cyber benchmark, not reproducible and not comparable outside the table. Against Pro 81.7."}
]
$benchmarks$::jsonb) AS evidence(
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
