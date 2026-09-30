-- Publish the authored MiMo V2.6 Pro vs MiMo V2.6 Flash comparison page,
-- following the pattern set by 091 (GPT-6.1 Sol vs GPT-6 Sol). Migration 093
-- published the series and Pro glossary pages and 094 added the catalog rows,
-- all from the same sources, so the prose here stays consistent with them and
-- the three surfaces share one set of caveats.
--
-- Sources: Xiaomi's September 22, 2026 MiMo-V2.6 announcement post, the
-- mimo.mi.com model pages for Pro and Flash, the Xiaomi MiMo Open Platform
-- model list and pay-as-you-go pricing pages, the MiMo-V2.6-Pro-RL and
-- MiMo-V2.6-Flash-RL model cards with the technical report, the
-- MiMo-V2.6-RL-oss training environments and datasets, and MiMo-V2.6 Pro on
-- Artificial Analysis.
--
-- Pro and Flash rather than Pro and UltraSpeed, because UltraSpeed has no
-- scores of its own. It serves the Pro weights, so a Pro-versus-UltraSpeed
-- page could only restate one model's benchmarks against a price list, and
-- 094 seeds no benchmark rows for it for that reason. The UltraSpeed facts
-- that matter for a buyer appear in this page's pricing and access tables.
--
-- Footnote handling, all carried in the row notes so the rendered comparison
-- keeps the caveats:
-- 1. Every benchmark value on this page is Xiaomi's own transcription from a
--    single table in the announcement post. The two models are compared
--    against each other rather than against third parties here, so the
--    second-hand competitor cells that complicate the glossary pages do not
--    enter into it.
-- 2. MiMo Code Bench and MiMo Visual Coding are Xiaomi in-house benchmarks.
--    Nobody outside the company can reproduce them, and they are the two rows
--    where the two models look closest, so the gap they show should not be
--    read as a small measured difference so much as an unrepeatable one.
-- 3. Flash has no GDPval 2.1 figure in Xiaomi's table and no Artificial
--    Analysis Intelligence Index score. Neither is inferred from Pro.
-- 4. The Artificial Analysis Intelligence Index reading of 46 for Pro is a
--    category-level score rather than a single benchmark, so it is carried in
--    the Model behavior table and not normalized into model_benchmarks, where
--    no other model has an index row. This follows 091's treatment.
-- 5. The announcement post's prose and its own per-step table disagree on the
--    DeepSWE training-curve endpoints (48.8 to 65.68 and 58.4 to 72.57 in
--    prose, 48.7 and 65.7 in the table). The step-table figures are used in the
--    Model behavior table and the discrepancy is noted there. These are
--    training-curve endpoints and are distinct from the settled benchmark
--    scores in the benchmark tables.
-- 6. Flash and Pro run the same reinforcement learning recipe and the same
--    published curve methodology. Flash's training-cost figure is about $0.85
--    million against Pro's $2.62 million, so the capability gap is partly a
--    compute gap within one recipe rather than two different approaches.
-- 7. The pass-rate improvements the post reports are 25% for Flash and 12% for
--    Pro, which is the opposite ordering to the benchmark gap between them.
--    The post does not map those percentages to named models in a way that can
--    be stated unambiguously, so they are reported as the post's figures
--    without asserting which model each belongs to.
-- 8. No safety evaluation was published with the release for either model.
--    Flash scores higher than Pro on CyberGym, 95.1 against 94.0, and far lower
--    on ExploitBench, 25.3 against 47.9. That is recorded as a property of the
--    benchmarks rather than read as Flash being safer or Pro being more
--    dangerous, because the two measure different capabilities.
-- 9. UltraSpeed is Pro's weights on a custom serving tier, and it is included
--    in the pricing and access tables because it changes what the top of the
--    range offers. Its capability set is undocumented on the Open Platform,
--    it has no published rate limits, no Batch API and no Token Plan coverage.
--    That is stated in the access table rather than assumed from Pro.
-- 10. Both models are 1M context and 128K maximum output, as stated on their
--    own model pages and repeated in the Open Platform model list.
--
-- The blocks JSON is dollar-quoted rather than single-quoted so the
-- apostrophes in vendor benchmark names do not need doubling. migrate.mjs
-- splits statements on a semicolon at end of line, and no line inside the
-- dollar-quoted strings contains one.

-- 1. Authored comparison page: MiMo V2.6 Pro vs MiMo V2.6 Flash.
WITH comparison_notes AS (
  SELECT $notes$
<small>All benchmark values are Xiaomi's own transcription of a single table in its September 22, 2026 announcement post. MiMo Code Bench and MiMo Visual Coding are Xiaomi in-house benchmarks that nobody outside the company can reproduce, and they are two of the rows where the two models sit closest, so treat that gap as unrepeatable rather than precisely measured. Flash has no GDPval 2.1 figure in Xiaomi's table and none is inferred from Pro. The Artificial Analysis Intelligence Index, cost per task, speed and verbosity figures are measured independently by Artificial Analysis rather than transcribed from Xiaomi; the index itself is a category score, not a single benchmark, so it appears only in the model behavior table. Both models are MIT licensed with no safety evaluation published alongside them.</small>

MiMo V2.6 Pro and MiMo V2.6 Flash are the same model trained under the same recipe at two compute budgets, released together. That is the shape of this comparison, and it is unusual in a catalog full of unrelated models. Both took 30 reinforcement learning steps in under six days on the same fully asynchronous architecture, across roughly 750,000 trajectories, using the same groupwise reward synthesis approach and publishing their per-step training curves the same way. The training cost is where the budget difference shows: about $2.62 million for Pro against $0.85 million for Flash. So this is not one architecture at two sizes. It is one recipe, scaled, and the capability gap between them is largely a compute gap within a single method.

The practical gap is narrower than a 3x price difference suggests. Pro leads on most shared benchmarks, but usually by a few points rather than by a margin. DeepSWE v1.1 is 71.9 against 67.9, Toolathlon-verified 76.9 against 73.6, OSWorld-Verified 82.0 against 80.8, AutomationBench v1.0.6 53.1 against 52.3. On three of the sixteen shared rows Flash is level or ahead: CyberGym 95.1 against 94.0, and GDPval 2.1 where Flash simply has no reported figure. On JobBench the gap is 62.0 to 61.2, less than a point. The exceptions where Pro pulls away are ExploitBench at 47.9 against 25.3 and SEC Bench Pro at 66.3 against 47.5.

Price is where the two are not close. Pro is $0.435 per million input tokens on a cache miss against Flash's $0.14, and $0.87 against $0.28 on output, so roughly 3.1x in both directions. Neither price changed from the V2.5 series, which is the actual commercial claim of the release: better scores at last quarter's prices. For a workload whose quality bar Flash clears, that is a straightforward 3x saving rather than a close call, and the batch rates halve both again.

The cached-input rates tell a more interesting story than the headline prices. Pro's is $0.0036 against a $0.435 miss, a discount of roughly 120x. Flash's is $0.0028 against $0.14, about 50x. The cheaper model is also the one whose cache discount does less work, so an agent loop that re-reads the same repository every turn narrows the real price gap well below 3x. That is the case where Pro's premium buys something specific rather than paying for general superiority: Flash at high cache-hit rates and Pro at low ones can land in the same place, and the crossover depends entirely on the reuse pattern of the workload.

Two limits apply to both. No safety evaluation was published with the release, which matters more here than it would for a closed model, because both are downloadable under an unmodified MIT license. Flash scores higher than Pro on CyberGym, and the Flash versus Pro ordering reverses on ExploitBench by more than 20 points, so these narrow vulnerability benchmarks should not be read as a single cyber capability level. And three of the sixteen shared benchmark rows are Xiaomi's own in-house results, so no third party has reproduced the bulk of this table.
$notes$::text AS notes
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'comparison:mimo-v2-6-pro-vs-mimo-v2-6-flash',
  'comparison',
  'mimo-v2-6-pro-vs-mimo-v2-6-flash',
  '',
  'comparisons/mimo-v2-6-pro-vs-mimo-v2-6-flash',
  'MiMo V2.6 Pro vs MiMo V2.6 Flash',
  'MiMo V2.6 Pro vs MiMo V2.6 Flash: the same RL recipe at two compute budgets, 1.02T/42B against 309B/15B, 3x the price, and a narrower capability gap than that implies.',
  comparison_notes.notes,
  $blocks$
  [
    {"id": "spec-mimo26pro-26flash-1", "type": "spec_table", "title": "Specifications",
     "columns": ["MiMo V2.6 Flash", "MiMo V2.6 Pro"],
     "rows": [
       ["Developer", "Xiaomi MiMo", "Xiaomi MiMo"],
       ["Release date", "2026-09-22", "2026-09-22"],
       ["API model ID", "mimo-v2.6-flash", "mimo-v2.6-pro"],
       ["Total / active parameters", "309B / 15B", "1.02T / 42B"],
       ["Architecture", "Sparse MoE, 48 layers (39 sliding-window, 9 global), 256 routed experts", "Sparse MoE, 70 layers (60 sliding-window, 10 global), 384 routed experts, 8 activated per token"],
       ["Context window", "1,000,000 tokens", "1,000,000 tokens"],
       ["Max output", "128,000 tokens", "128,000 tokens"],
       ["Reasoning / effort", "Deep thinking, on by default", "Deep thinking, on by default"],
       ["Vision encoder", "Not separately documented", "681M MiMo ViT, 28 layers, hidden 1280"],
       ["Audio encoder", "Not separately documented", "308M AudioTokenizer plus a 127M audio patch encoder"],
       ["Speculative decoding", "Not separately documented", "5-layer sliding-window multi-token-prediction drafter, 7 tokens per pass"],
       ["Weights / license", "Open source. XiaomiMiMo/MiMo-V2.6-Flash-RL, MIT License; RL datasets and environments Apache-2.0", "Open source. XiaomiMiMo/MiMo-V2.6-Pro-RL, MIT License; RL datasets and environments Apache-2.0"],
       ["Rate limits", "100 RPM, 10M TPM", "100 RPM, 10M TPM"]
     ]},
    {"id": "spec-mimo26pro-26flash-2", "type": "spec_table", "title": "Pricing",
     "columns": ["MiMo V2.6 Flash", "MiMo V2.6 Pro"],
     "rows": [
       ["Input / 1M tokens (cache miss)", "**$0.14**", "$0.435"],
       ["Cached input / 1M", "**$0.0028**", "$0.0036"],
       ["Cache miss-to-hit discount", "~50x", "~120x"],
       ["Output / 1M tokens", "**$0.28**", "$0.87"],
       ["Batch API input, cache miss", "**$0.07**", "$0.2175"],
       ["Batch API output", "**$0.14**", "$0.435"],
       ["China pricing, cache miss / output", "¥1 / ¥2", "¥3 / ¥6"],
       ["Cache write", "Free for a limited time", "Free for a limited time"],
       ["Price change vs V2.5 series", "None", "None"]
     ]},
    {"id": "spec-mimo26pro-26flash-3", "type": "spec_table", "title": "Capabilities & access",
     "columns": ["MiMo V2.6 Flash", "MiMo V2.6 Pro"],
     "rows": [
       ["Text input", "Yes", "Yes"],
       ["Image / vision input", "Yes, native", "Yes, native"],
       ["Audio input", "No. Artificial Analysis lists Flash as text and image only", "Yes, native. Artificial Analysis lists text, image, speech and video"],
       ["Video input", "No, on the same reading", "Yes, native. Artificial Analysis lists text, image, speech and video"],
       ["Text output", "Yes", "Yes"],
       ["Tool / function calling", "Yes, with streaming, structured output and web search", "Yes, with streaming, structured output and web search"],
       ["Context caching", "Yes", "Yes"],
       ["Batch API", "Yes, at half the real-time rate", "Yes, at half the real-time rate"],
       ["Token Plan coverage", "Yes", "Yes"],
       ["MiMo-V2.6-Pro-UltraSpeed variant", "None", "Yes, at up to 20x output speed for 10x the price. It should be treated as not production-ready: Xiaomi leaves its capability column blank, it has no Batch API, no published rate limits, no Token Plan coverage, and Artificial Analysis has no evaluation page for it, so nothing about it is independently verified"],
       ["API access", "https://api.xiaomimimo.com/v1, both OpenAI and Anthropic protocols", "https://api.xiaomimimo.com/v1, both OpenAI and Anthropic protocols"],
       ["Product access", "Xiaomi AI Studio, MiMo Code, MiMo Desktop, the MiMo API platform, and OpenRouter", "Xiaomi AI Studio, MiMo Code, MiMo Desktop, the MiMo API platform, and OpenRouter"],
       ["Weights / license", "Open source, unmodified MIT", "Open source, unmodified MIT"]
     ]},
    {"id": "spec-mimo26pro-26flash-4", "type": "spec_table", "title": "Model behavior",
     "columns": ["MiMo V2.6 Flash", "MiMo V2.6 Pro"],
     "rows": [
       ["Primary focus", "High-frequency calls and large-scale professional office tasks; the best balance in the series", "The most capable model Xiaomi has shipped, for complex projects, long-horizon work, cybersecurity and research"],
       ["RL steps / wall clock", "30 steps in under 6 days", "30 steps in under 6 days"],
       ["Reported RL training cost", "~$0.85M", "~$2.62M"],
       ["Trajectories", "~750,000 across the series", "~750,000 across the series"],
       ["DeepSWE training curve (step 1 to 30)", "48.8 to 65.7, about 17 points", "58.4 to 72.6, about 14 points. The post's prose reports 48.8 to 65.68 and 58.4 to 72.57 against 48.7 and 65.7 in its own step table"],
       ["Training config", "1,568 samples per update, 1M context, 3.5 to 3.7B tokens per step, fully asynchronous", "Same configuration and same groupwise reward synthesis approach"],
       ["Artificial Analysis Intelligence Index v4.3.2", "**38**, at $0.06 per index task", "**46**, at $0.13 per index task and the highest open-weights result, ahead of GLM-5.3 at 45 and Kimi K3 at 44. Xiaomi cites 46.32; the gap is rounding"],
       ["Generation profile", "Described by Xiaomi as the token-efficient end of the series, but Artificial Analysis measures it as the more verbose of the two: 240M output tokens across its index run against a 140M median. It is the faster of the two at 55 tokens per second", "Artificial Analysis describes it as slow and verbose, generating 140M tokens across its index run at 41.1 tokens per second and a 4.24-second time to first token"],
       ["Pass-rate improvement over training", "The post reports 25% and 12% across the two models without mapping either figure unambiguously to one of them", "Same"],
       ["Safety / approvals", "No safety evaluation published. CyberGym 95.1, ExploitBench 25.3, SEC Bench Pro 47.5", "No safety evaluation published. CyberGym 94.0, ExploitBench 47.9, SEC Bench Pro 66.3"]
     ]},
    {"id": "spec-mimo26pro-26flash-5", "type": "spec_table", "title": "Coding",
     "columns": ["MiMo V2.6 Flash", "MiMo V2.6 Pro"],
     "rows": [
       ["DeepSWE v1.1", "67.9%", "**71.9%**"],
       ["ProgramBench", "26.0%", "**26.5%**"],
       ["MiMo Code Bench (in-house, unreproducible)", "61.2%", "**63.2%**"],
       ["Terminal-Bench 4.0", "28.8%", "**34.9%**"]
     ]},
    {"id": "spec-mimo26pro-26flash-6", "type": "spec_table", "title": "Knowledge",
     "columns": ["MiMo V2.6 Flash", "MiMo V2.6 Pro"],
     "rows": [
       ["GDPval 2.1 (AA, Elo)", "Not reported", "**1673 Elo**"]
     ]},
    {"id": "spec-mimo26pro-26flash-7", "type": "spec_table", "title": "Agentic & computer use",
     "columns": ["MiMo V2.6 Flash", "MiMo V2.6 Pro"],
     "rows": [
       ["Toolathlon-verified", "73.6%", "**76.9%**"],
       ["AutomationBench v1.0.6", "52.3%", "**53.1%**"],
       ["Agents' Last Exam", "27.6%", "**31.6%**"],
       ["OSWorld-Verified", "80.8%", "**82.0%**"]
     ]},
    {"id": "spec-mimo26pro-26flash-8", "type": "spec_table", "title": "Professional",
     "columns": ["MiMo V2.6 Flash", "MiMo V2.6 Pro"],
     "rows": [
       ["JobBench", "61.2%", "**62.0%**"]
     ]},
    {"id": "spec-mimo26pro-26flash-9", "type": "spec_table", "title": "Multimodal",
     "columns": ["MiMo V2.6 Flash", "MiMo V2.6 Pro"],
     "rows": [
       ["MiMo Visual Coding (in-house, unreproducible)", "71.5%", "**72.3%**"]
     ]},
    {"id": "spec-mimo26pro-26flash-10", "type": "spec_table", "title": "Cyber capability",
     "columns": ["MiMo V2.6 Flash", "MiMo V2.6 Pro"],
     "rows": [
       ["CyberGym", "**95.1%**", "94.0%"],
       ["ExploitGym", "6.0%", "**17.8%**"],
       ["ExploitBench", "25.3%", "**47.9%**"],
       ["SEC Bench Pro", "47.5%", "**66.3%**"],
       ["MiMo Cyber Bench (in-house, unreproducible)", "77.2%", "**81.7%**"]
     ]},
    {"id": "markdown-mimo26pro-26flash-switch", "type": "markdown",
     "content": "## What changes when you switch\n\nSwitching Flash to Pro is a model-string change, not a migration. Both speak the OpenAI and Anthropic protocols from the same base URL, both are 1M context and 128K output, both cap at 100 RPM and 10M TPM, and both are covered by the same Token Plan. Neither is deprecated by the other. `mimo-v2.5-pro` and `mimo-v2.5` are deprecated at 10:00 Beijing time on October 21, 2026, so the real migration question in the series is leaving V2.5 rather than choosing within V2.6.\n\nWhat does change is the bill and the latency profile. Pro costs roughly 3.1x Flash per million input and output tokens on a cache miss, and the batch path halves both, so batch inference is the cheapest way to buy Pro's scores. The cached-input rates behave differently: Pro's $0.0036 against a $0.435 miss is about a 120x discount, while Flash's $0.0028 against $0.14 is about 50x. An agent loop that re-sends the same repository or document context every turn leans on that discount, and at high cache-hit rates the effective gap between the two narrows well below 3x. Low-reuse workloads pay the full 3x.\n\nOne caution on the comparison table itself. Both models are MIT licensed with the reinforcement learning code, training environments and serving recipes published alongside them, which is unusual and genuinely useful. But no safety evaluation accompanied the release for either. On downloadable weights, Flash scores higher than Pro on CyberGym and far lower on ExploitBench, a reversal of more than 20 points, so these narrow vulnerability benchmarks do not describe one shared cyber level. Treat them as properties of the weights rather than of the hosted product, and apply whatever controls your own deployment needs."},
    {"id": "markdown-mimo26pro-26flash-bottom-line", "type": "markdown",
     "content": "## Bottom line\n\nMiMo V2.6 Pro and MiMo V2.6 Flash are the same reinforcement learning recipe at two compute budgets, which is the most useful thing to know about the choice. Pro costs about 3x as much and leads on most shared benchmarks, but usually by a few points: DeepSWE 71.9 against 67.9, Toolathlon 76.9 against 73.6, OSWorld 82.0 against 80.8. JobBench is under a point apart, and Flash leads on CyberGym.\n\nFor most production workloads Flash is the right default. It clears the bar for high-frequency office and document work at a third of the price, and Xiaomi states its advantage plainly as token efficiency. Buy Pro where its specific strengths matter rather than as a general upgrade: long-horizon coding agents in real repositories, vulnerability discovery and security research, multimodal analysis where audio and video input alongside a 1M window matters together, and the workloads Artificial Analysis measured at 140M output tokens, where Flash's efficiency may or may not survive contact with the harder tasks. The honest answer for most teams is to run Flash in production and evaluate Pro on the specific tasks where its benchmarks actually separate.\n\nTwo caveats carry across both. Three of the shared benchmark rows are Xiaomi in-house results that no third party can reproduce, and two of those three are among the closest rows on the page, so the narrow margins there are the least trustworthy numbers in the table. And no safety evaluation was published for either model, which matters more than it would for a closed one because both are downloadable under an unmodified MIT license."}
  ]$blocks$::jsonb,
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing MiMo-V2.6 series', 'url', 'https://mimo.xiaomi.com/mimo-v2-6'),
    jsonb_build_object('title', 'MiMo-V2.6-Pro on Artificial Analysis', 'url', 'https://artificialanalysis.ai/models/mimo-v2-6-pro'),
    jsonb_build_object('title', 'MiMo-V2.6-Flash on Artificial Analysis', 'url', 'https://artificialanalysis.ai/models/mimo-v2-6-flash')
  ),
  jsonb_build_object(
    'modelA', 'mimo-v2-6-pro',
    'modelB', 'mimo-v2-6-flash',
    'verification_pass', 'mimo-v2-6-comparison-2026-09-22',
    'seoDescription', 'MiMo V2.6 Pro vs MiMo V2.6 Flash: the same RL recipe at two compute budgets, 1.02T/42B against 309B/15B, 3x the price, and a narrower gap than that.',
    'seoKeywords', jsonb_build_array(
      'MiMo V2.6 Pro vs MiMo V2.6 Flash', 'MiMo V2.6 Pro', 'MiMo V2.6 Flash',
      'mimo-v2.6-pro', 'mimo-v2.6-flash', 'MiMo V2.6 comparison',
      'MiMo V2.6 benchmarks', 'MiMo V2.6 DeepSWE', 'MiMo V2.6 pricing',
      'MiMo V2.6 UltraSpeed', 'Xiaomi MiMo V2.6', 'MiMo V2.6 Pro parameters'
    )
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
