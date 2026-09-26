WITH gemini_38_flash_entry AS (
  SELECT $body$
Gemini 3.8 Flash is Google's September 2, 2026 generally available multimodal model for long-horizon software engineering, autonomous agents, and professional knowledge work. It arrived only three weeks after [Gemini 3.7 Flash](/glossary/gemini-3-7-flash) and is Google's third Flash release in six weeks. Unlike most version bumps, 3.8 Flash is not a new architecture: Google states it is based on 3.7 Flash, and the gains come from post-training plus long-running agentic loops that recursively evaluate and refine the underlying model.

The headline claim is capability at unchanged Flash economics. Google describes 3.8 Flash as "our best reasoning and coding model yet, at the same speed and low cost of 3.7," and says it often approaches the performance of higher-cost frontier models. The launch post also credits rigorous training in cybersecurity as one of the drivers behind the shared coding and reasoning gains.

## Key specifications

| Specification | Gemini 3.8 Flash |
| --- | --- |
| Developer | Google DeepMind |
| Release status | Generally available (GA) |
| Release date | September 2, 2026 |
| API model ID | `gemini-3.8-flash` |
| Inputs | Text, images, video, audio, and PDF |
| Output | Text |
| Context window | 1,048,576 tokens |
| Maximum output | 65,536 tokens |
| Thinking levels | `low`, `medium`, `high`; `medium` is the default |
| Based on | Gemini 3.7 Flash |
| Knowledge cutoff | March 2026 |
| Primary focus | Long-horizon coding, autonomous agents, complex enterprise workflows |

The knowledge cutoff deserves attention. The model card states the cutoff is March 2026, but also warns that in some domains users may still see the model's knowledge limited to January 2025, in line with the wider Gemini 3 family. For time-sensitive factual work, treat grounding and retrieval as required rather than optional.

## What changed from 3.7 Flash

Google describes the change in design terms rather than architectural ones: "3.8 Flash works harder." On complex tasks the model shows greater diligence, executing extra reasoning steps and calling tools iteratively, and it will sometimes spend more tokens to maximize performance, especially at higher effort levels. That is a real cost characteristic, not a footnote: an agent that reasons longer on every request can quietly erase the savings from the low per-token price.

Google attributes the improvement to long-running agentic loops designed to recursively evaluate and refine the underlying models, plus intensive post-training in cybersecurity. Because the model is built on 3.7 Flash, Google defers most architecture, hardware, and training-dataset detail to the 3.7 Flash model card.

## Reported benchmarks

Results below are from Google's September 2026 model card, alongside the same vendor's results for competing models.

| Benchmark | Gemini 3.8 Flash | Gemini 3.7 Flash | Claude Opus 5 | Claude Sonnet 5 | GPT-5.6 Sol |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSWE v1.1 (long-horizon software engineering) | 73.7% | 65.3% | 74.0% | 53.8% | 72.7% |
| Terminal-bench 2.1 (agentic terminal coding) | 89.4% | 85.8% | 89.1% | 80.4% | 88.8% |
| Terminal-bench 4.0 (general agent capability) | 19.1% | 11.2% | 51.8% | 12.4% | 37.3% |
| GDPVal-AA v2 (knowledge work, Elo) | 1545 | 1482 | 1824 | 1584 | 1710 |
| Vals Finance Agent v2 | 61.4% | 59.0% | 58.6% | 53.9% | 53.8% |
| Harvey's Legal Agent Benchmark (all pass) | 10.0% | 8.8% | 6.7% | 5.0% | 2.5% |
| HLE-Verified | 54.9% | 53.6% | 54.4% | 31.0% | 54.5% |
| OSWorld-2.0 (agentic computer use) | 59.0% | 50.6% | 75.4% | 42.6% | 62.6% |
| GDP.PDF (expert PDF comprehension) | 35.0% | 34.0% | 37.0% | 28.0% | 40.0% |
| CharXiv (chart reasoning, no tools) | 86.2% | 84.5% | 83.7% | 70.1% | 85.8% |
| LVBench (long video, static) | 87.1% | 85.4% | 75.4% | 68.5% | 82.1% |
| BioMysteryBench (human difficult) | 56.5% | 43.5% | 49.4% | 34.1% | 44.7% |
| LABBench2 (real-world research tasks) | 86.2% | 82.1% | 84.2% | 80.1% | 82.1% |

Read these as a shape, not a ranking. On terminal coding the model is within half a point of Claude Opus 5 while costing roughly a seventh as much per input token. On general agent capability, however, 3.8 Flash scores 19.1% on Terminal-bench 4.0 against Opus 5 at 51.8%, and on OSWorld-2.0 it trails Opus 5 by more than sixteen points. The wins are concentrated in long-horizon engineering and professional-domain analysis; the losses are in the broad, tool-heavy autonomy category where the frontier models still lead comfortably.

All figures are vendor-reported. Harness, effort level, context management, tool access, and fallback behavior materially change agentic scores, and LVBench in particular is reported in two modes: Google lists 87.8% with agentic video understanding and 87.1% static. Cross-vendor comparison is also uneven, since each lab chooses its own prompts and scaffolding.

## Cost, and the price cliff to plan around

Gemini 3.8 Flash launched at the same introductory price as 3.7 Flash: $0.75 per million input tokens and $3.75 per million output tokens, no caching. That introductory rate expires on December 31, 2026. Starting January 1, 2027, the standard rate of $1.50 per million input tokens and $7.50 per million output tokens applies, which is exactly a doubling.

This is the single most consequential operational detail in the release. A product whose unit economics worked at $0.75/$3.75 in the fourth quarter of 2026 does not automatically keep working in the first quarter of 2027, and the same introductory pricing applies to 3.6 Flash and 3.7 Flash. Budget against the 2027 rate, or keep an escape hatch to a cheaper model for the long tail of traffic.

Google's own guidance is to lower the effort level when compute efficiency matters more than quality, or to stay on 3.7 Flash, which remains fully supported for efficiency-first workloads.

## API details and migrating from 3.7 Flash

Migrating is a one-line model ID change, but one parameter has to be handled properly. Google's migration guide directs you to replace the integer `thinking_budget` configuration with the string enum `thinking_level`, set to `low`, `medium`, or `high`, with `medium` as the default. `MINIMAL` is not supported on this model. Code that relied on `thinking_budget` to cap reasoning will silently lose that control if the parameter is simply dropped, so the swap is required rather than optional.

3.8 Flash keeps the full tool suite carried by 3.7 Flash: caching, code execution, file search, function calling, Search and Maps [grounding](/glossary/grounding), structured output, URL context, and preview computer use. It also supports interactive video understanding, with dedicated emphasis on long-horizon software engineering, autonomous agents, and interactive video understanding across its 1M-token [context window](/glossary/context-window).

## Availability

3.8 Flash is broadly available: the Gemini API and Google AI Studio, Android Studio, Stitch for UI generation, [Google Antigravity](/glossary/muse-spark) for agent-first development, Gemini Enterprise for enterprises, and the Gemini app, AI Mode in Google Search, and Gemini in Sheets for Google AI Pro and Ultra subscribers. Google's Antigravity agent in Gemini Managed Agents now uses 3.8 Flash by default, as does the Antigravity SDK.

The launch also included Gemini 3.8 Flash Cyber, Google's most capable cybersecurity model, released to trusted defenders through the Fairwind Program rather than sold openly. It shares 3.8 Flash's underlying intelligence but ships a more permissive set of mitigations for security use cases. Google reports it on the Pareto frontier for CWE-Bench at 47.2% pass@1 against a leading frontier model at 47.8%, and notes the Chrome Security team found 2.6 times more correct patches from 3.8 Flash Cyber than from the best commercial models that are much larger. Treat these as vendor claims pending independent evaluation.

## Safety and known limitations

Google reports that 3.8 Flash performs similarly to 3.7 Flash on safety and tone, with low unjustified refusal rates, and that safety performance across non-English languages regressed slightly. On tone and instruction following the automated evaluation improved modestly. The model ships with safeguards in CBRN and cyber-offense domains under the Frontier Safety Framework, and Google reports a significant leap in prompt-injection robustness as measured by Gray Swan.

Google's Frontier Safety assessment found no meaningful new capabilities relative to 3.7 Flash, and concluded 3.8 Flash is unlikely to reach any Tracked or Critical Capability Levels. Known limitations remain the standard ones: [hallucination](/glossary/hallucination), occasional slowness or timeouts, and the tendency to consume more tokens at higher effort levels.

## When to use it

Gemini 3.8 Flash is a strong default for production agents where long-horizon software engineering, document and PDF work, financial or legal analysis, and long video understanding all matter more than peak autonomy, and where per-token cost decides whether the workflow is viable at all. The 1M-token context and GA status make it a practical default for retrieval-augmented systems and repository-scale work.

It is a weaker choice for tasks that need genuinely open-ended computer use or general agentic autonomy across arbitrary tools, where Claude Opus 5 remains far ahead, and for knowledge-work scoring on the GDPVal-AA Elo scale. For pure efficiency, 3.7 Flash is still available and supported.

## Bottom line

Gemini 3.8 Flash is Google's most capable Flash model to date, and its pitch is unusually concrete: frontier-adjacent coding and professional reasoning at roughly $0.75 per million input tokens. The catch is that the model spends more tokens to get there, and the introductory price doubles on January 1, 2027. Choose it for long-horizon engineering and knowledge workflows where the economics hold, measure total task cost rather than token price, and plan the 2027 pricing change into your forecasts.
$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:gemini-3-8-flash',
  'glossary',
  'gemini-3-8-flash',
  '',
  'glossary/gemini-3-8-flash',
  'Gemini 3.8 Flash',
  'Google''s September 2026 GA multimodal model for long-horizon coding and autonomous agents, with a 1M-token context window, 64K output, and Flash-level pricing.',
  gemini_38_flash_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', gemini_38_flash_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing Gemini 3.8 Flash and 3.8 Flash Cyber', 'url', 'https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/'),
    jsonb_build_object('title', 'Gemini 3.8 Flash Model Card', 'url', 'https://deepmind.google/models/model-cards/gemini-3-8-flash/'),
    jsonb_build_object('title', 'What''s new in Gemini 3.8 Flash', 'url', 'https://ai.google.dev/gemini-api/docs/latest-model'),
    jsonb_build_object('title', 'Gemini 3.8 Flash API Model Specifications', 'url', 'https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash'),
    jsonb_build_object('title', 'Developer''s guide to Gemini 3.8 Flash', 'url', 'https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-3-8-flash'),
    jsonb_build_object('title', 'Gemini API Pricing', 'url', 'https://ai.google.dev/gemini-api/docs/pricing')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array('gemini-3-7-flash', 'gemini', 'gemini-3-1-pro', 'large-language-model', 'context-window', 'reasoning', 'agentic-ai', 'tool-calling', 'computer-vision', 'grounding', 'benchmark', 'inference', 'kv-cache', 'hallucination', 'api'),
    'analogy', 'A senior engineer who takes longer on the hard problems: the same desk and the same hourly rate as the last hire, but now they read the whole spec, check their work, and fix it before handing it back.',
    'seoDescription', 'Gemini 3.8 Flash explained: GA status, 1M context, thinking levels, migration from 3.7 Flash, official benchmarks, the January 2027 price change, and availability.',
    'seoKeywords', jsonb_build_array('what is Gemini 3.8 Flash', 'Gemini 3.8 Flash benchmarks', 'Gemini 3.8 Flash pricing', 'Gemini 3.8 Flash API', 'Gemini 3.8 Flash context window', 'Gemini 3.8 Flash vs Gemini 3.7 Flash', 'Gemini 3.8 Flash vs Claude Opus 5', 'Gemini 3.8 Flash vs GPT-5.6 Sol', 'gemini-3.8-flash model ID', 'Gemini 3.8 Flash thinking level', 'Gemini 3.8 Flash Cyber', 'Google multimodal coding model')
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-09-02',
  0
FROM gemini_38_flash_entry
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

UPDATE content_items
SET
  metadata = CASE
    WHEN COALESCE(metadata->'relatedTerms', '[]'::jsonb) @> '["gemini-3-8-flash"]'::jsonb
      THEN metadata
    ELSE jsonb_set(
      COALESCE(metadata, '{}'::jsonb),
      '{relatedTerms}',
      COALESCE(metadata->'relatedTerms', '[]'::jsonb) || jsonb_build_array('gemini-3-8-flash'),
      true
    )
  END,
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug = 'gemini-3-7-flash'
  AND parent_slug = '';
