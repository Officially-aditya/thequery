-- Publish the MiniMax M3.1 Flash glossary model card from the official launch,
-- invocation guide and pricing page, with separately attributed AA measurements.
WITH minimax_m31_flash_entry AS (
  SELECT $body$MiniMax M3.1 Flash is MiniMax's October 2026 model for coding, tool use, computer operation and multi-step agent work. It follows [MiniMax M3](/glossary/minimax-m3), with a 1M-token [context window](/glossary/context-window), multimodal input and lower published token prices. It is available through the MiniMax API, M Plan and MiniMax Code.

The official API model ID is `MiniMax-M3.1-Flash`. MiniMax's documentation lists text, image and video input; Artificial Analysis lists text and image input for its evaluated profile and text output. Thinking is required, with five reasoning effort levels and `max` as the default. MiniMax says model weights will be released later.

## Key specifications

| Specification | MiniMax M3.1 Flash |
| --- | --- |
| Developer | MiniMax |
| Release | October 2026 |
| API model ID | `MiniMax-M3.1-Flash` |
| Context window | 1,000,000 tokens |
| Input | Text, images and video in MiniMax's documentation |
| Output | Text |
| Reasoning effort | `low`, `medium`, `high`, `xhigh`, `max`; default `max` |
| Non-thinking mode | Not supported |
| API formats | Anthropic-compatible, OpenAI Chat Completions and OpenAI Responses |
| Availability | MiniMax API, M Plan and MiniMax Code |
| Downloadable weights | Announced for later in the launch post |
| Parameter count, knowledge cutoff and maximum output | Not specified in the launch post or model invocation guide reviewed here |

The context window limits the material available during a request, including a long conversation, documents or code. A 1M context window does not establish a 1M output limit or guarantee that every detail will be used accurately.

## Pricing

MiniMax's launch post and current pricing page list flat rates for M3.1 Flash. Prices below are in US dollars per million tokens, reviewed on October 23, 2026.

| Token type | Price per million tokens |
| --- | ---: |
| Input | $0.15 |
| Cached input reads | $0.015 |
| Output | $0.60 |

Cached input reads cost 90% less than ordinary input at these rates. The M3.1 Flash row does not list the prompt-length price tiers shown separately for MiniMax M3.

For example, 100,000 uncached input tokens and 10,000 output tokens cost $0.021 at the listed rates: $0.015 for input plus $0.006 for output. This is a token-price calculation, rather than a measured cost for completing a particular task. Additional attempts, longer reasoning and tool charges can change the total.

## Official benchmark results

The following figures come from [MiniMax's launch announcement](https://www.minimax.io/blog/minimax-m31f). They are vendor-reported results. The announcement does not provide a complete reproducible configuration for each result, so these numbers should retain their source and benchmark version when compared with other models.

| Benchmark | MiniMax-reported result | Task area |
| --- | ---: | --- |
| BankerToolBench | 76.2 points | Financial workflows, planning and chained tool calls |
| DeepSWE v1.1 | 73.7% | Software engineering |
| OSWorld V2 | 61.7 points | Computer operation and cross-application tasks |
| Agents' Last Exam (ALE) | 29.6 points | Professional work and complex computer tasks |
| CyberGym | 82% | Cybersecurity vulnerability tasks |

MiniMax also highlights AutomationBench-AA performance without giving a numerical score in the announcement's text. Its cybersecurity results measure capability; they do not establish how the model handles every unsafe request or every production security task.

## Independent measurements

[Artificial Analysis](https://artificialanalysis.ai/models/minimax-m3-1-flash) publishes a separate evaluation profile. Its public summary showed these readings when reviewed on October 23, 2026.

| Measurement | Artificial Analysis reading |
| --- | ---: |
| Intelligence Index v4.3.2 | 43 |
| Output generation speed | 92.5 tokens per second |
| Weighted cost per Intelligence Index task | $0.22 |
| Output tokens across the Intelligence Index evaluation | About 180M |

The Intelligence Index is a composite score, rather than a percentage on a single benchmark. The task cost describes Artificial Analysis's evaluation mix, and its token total describes the full evaluation rather than one response. The profile identifies a reasoning model but does not state an effort setting in its summary.

MiniMax reports output speeds above 200 tokens per second and a median above 170, alongside nearly three times faster online inference and 70% lower overall usage cost than its predecessor on agent tasks. Those are vendor claims. Artificial Analysis's 92.5-token-per-second measurement comes from a separate evaluation, and neither output-speed reading alone measures how long a complete agent task takes.

## API behavior and workflow fit

MiniMax recommends its Anthropic-compatible API at `https://api.minimax.io/anthropic`; the OpenAI-compatible base URL is `https://api.minimax.io/v1`. For thinking depth, use `output_config.effort` with the Anthropic format, `reasoning_effort` with Chat Completions, or `reasoning.effort` with Responses.

Thinking cannot be disabled. The documentation says `thinking: {"type": "disabled"}` and an effort of `none` return an HTTP 400 error. Lowering the effort level is the supported way to reduce reasoning latency and token use. With OpenAI Chat Completions, reasoning is returned separately in `reasoning_content`, while `content` contains the final answer.

The model is positioned for coding agents, business automation, visual understanding and computer-use workflows. Its published prices make repeated tool-driven sessions worth evaluating, while its required thinking and measured verbosity make total task cost and completion time important checks.

## Weights and availability

MiniMax's invocation guide describes M3.1 Flash as open-weight, but the launch announcement explicitly says the weights will be released later. The reviewed sources establish hosted access and an announced future weight release; they do not establish a downloadable checkpoint or its final license.

MiniMax also says an early version operated anonymously as Space Bunny and processed more than 200 trillion tokens. That is a vendor-reported account of prior use, rather than an independent benchmark or the model's parameter count.
$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:minimax-m3-1-flash',
  'glossary',
  'minimax-m3-1-flash',
  '',
  'glossary/minimax-m3-1-flash',
  'MiniMax M3.1 Flash',
  'MiniMax''s October 2026 model for coding and agent work, with 1M context, text/image/video input, required reasoning and $0.15/$0.60 input/output pricing per million tokens. Weights are announced for later.',
  minimax_m31_flash_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', minimax_m31_flash_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'MiniMax M3.1 Flash launch announcement', 'url', 'https://www.minimax.io/blog/minimax-m31f'),
    jsonb_build_object('title', 'MiniMax model invocation and thinking guide', 'url', 'https://platform.minimax.io/docs/guides/text-generation'),
    jsonb_build_object('title', 'MiniMax API pricing', 'url', 'https://platform.minimax.io/docs/pricing/overview'),
    jsonb_build_object('title', 'Artificial Analysis: MiniMax M3.1 Flash', 'url', 'https://artificialanalysis.ai/models/minimax-m3-1-flash')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array('minimax-m3', 'context-window', 'tool-calling', 'reasoning', 'benchmark', 'open-weight-model'),
    'seoDescription', 'MiniMax M3.1 Flash model card: 1M context, multimodal input, reasoning effort, API pricing, official benchmarks, independent measurements and weight availability.',
    'seoKeywords', jsonb_build_array('MiniMax M3.1 Flash', 'MiniMax-M3.1-Flash', 'MiniMax M3.1 Flash model card', 'MiniMax M3.1 Flash benchmarks', 'MiniMax M3.1 Flash pricing', 'MiniMax M3.1 Flash API', 'Space Bunny'),
    'verification_pass', 'minimax-m31-flash-2026-10-23'
  ),
  NULL,
  NULL,
  'published',
  NOW(),
  0
FROM minimax_m31_flash_entry
WHERE true
ON CONFLICT (kind, slug, parent_slug) DO UPDATE SET
  path = EXCLUDED.path,
  title = EXCLUDED.title,
  summary = EXCLUDED.summary,
  body = EXCLUDED.body,
  blocks = EXCLUDED.blocks,
  sources = EXCLUDED.sources,
  metadata = EXCLUDED.metadata,
  status = EXCLUDED.status,
  published_at = EXCLUDED.published_at,
  sort_order = EXCLUDED.sort_order,
  updated_at = NOW();
