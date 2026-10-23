-- Kimi's October 23 launch, API quickstart and pricing, verified October 24.
-- Hosted access is live. Do not classify the promised November 4 weights as
-- already downloadable, or inherit K3's exact parameter count or license.
WITH entry AS (
  SELECT $body$Kimi K3.1 Pro is Moonshot AI's October 2026 model for coding, 3D creation, financial research and multi-step agent work. It is a post-trained version of [Kimi K3](/glossary/kimi-k3), with a 1M-token [context window](/glossary/context-window) and native visual understanding. Moonshot describes it as a 3T-class model, meaning the roughly three-trillion-parameter size category, without specifying a separate exact parameter count in the launch announcement.

The model is available through Kimi, Kimi Work, Kimi Code and the Kimi API. The API ID is `kimi-k3.1-pro`. A second serving option, `kimi-k3.1-pro-highspeed`, is advertised as offering the same intelligence at two to three times the speed. That speed claim comes from Moonshot, rather than an independent measurement reproduced here.

**Weight-release status:** Moonshot says full model weights will be released by November 4, 2026. As checked on October 24, the launch establishes hosted access and a future open-weight release. It does not establish a downloadable K3.1 Pro checkpoint or its final license.

## Key specifications

| Specification | Kimi K3.1 Pro |
| --- | --- |
| Developer | Moonshot AI |
| Launch | October 23, 2026 |
| API model ID | `kimi-k3.1-pro` |
| Highspeed API ID | `kimi-k3.1-pro-highspeed` |
| Model family | Kimi K3.1; post-trained from Kimi K3 |
| Size | 3T class; exact total and active counts not separately specified in the reviewed launch sources |
| Context window | 1,048,576 tokens in the API pricing documentation |
| Completion-token setting | `max_completion_tokens` defaults to 131,072 and accepts values up to 1,048,576 in the quickstart |
| Input | Text, images and video |
| Output | Text, including code and tool calls |
| Reasoning effort | Always on; quickstart lists `low`, `high`, `max`, default `max` |
| Tool calling | Supported, including `tool_choice="required"` |
| Structured output | JSON Schema with `strict: true` |
| Knowledge cutoff | Not specified in the reviewed model sources |
| Downloadable weights and license | Weights promised by November 4, 2026; final K3.1 Pro license not established here |

Think of the context window as the model's working desk: it limits how much conversation, source material and generated content can be in reach during a request. A large desk does not guarantee accurate use of every document. The maximum completion setting is a request limit, not a promise of a million-token answer on top of a full million-token input.

The model generates text and can operate external software through tools. Its ability to script Blender or build a game does not mean the base API emits native video or 3D files. The surrounding software creates those artifacts.

## API pricing

The [official price list](https://platform.kimi.ai/docs/pricing/chat) gives the following US-dollar rates per million tokens, checked October 24, 2026. API usage is billed separately from Kimi membership and Kimi Code subscriptions.

| Token type | K3.1 Pro | K3.1 Pro Highspeed |
| --- | ---: | ---: |
| Uncached input | $2.00 | $4.00 |
| Output | $10.00 | $20.00 |
| Cached input read | $0.20 | $0.20 |
| Cache write, five-minute lifetime | $2.00 | $4.00 |
| Cache write, one-hour lifetime | $4.00 | $8.00 |

Moonshot documents flat pricing across context lengths. Context caching works like keeping a frequently consulted file open: a repeated prompt prefix can be reused at the cheaper read rate. Writes and reads have separate billing rules, and the default cache lifetime is five minutes. Highspeed doubles the base input and output rates while retaining the same cache-read rate.

At the base rates, 100,000 uncached input tokens plus 10,000 output tokens cost $0.30: $0.20 for input and $0.10 for output. This calculation excludes cache-write charges, tools, taxes and retries. It is a fixed-token example, not a measured cost for completing an agent task.

The current Batch API guide lists K2.7 Code and K2.6 as supported, excludes K3, and does not list K3.1 Pro. Its batch discount should not be assumed to apply to the new model.

## Reported benchmark results

These results come from [Moonshot's launch announcement](https://www.kimi.ai/blog/kimi-k3-1-pro). They are vendor-reported. Unless otherwise stated, Moonshot uses maximum reasoning effort, temperature 1.0 and top-p 0.95. A benchmark is a test course with particular rules; changing the tools, time budget or scoring changes what the score means.

| Evaluation | K3.1 Pro result | What the result measures |
| --- | ---: | --- |
| Terminal-Bench 4.0 | 65.7% | Terminal tasks using the Claude Code harness |
| Kimi Code Bench 3.0 Long-horizon | 61.2% | Moonshot's internal extended coding-task evaluation |
| BenchCAD, with tools | 92.7% | Voxel overlap on a 1,000-case subset, not a percentage of fully correct CAD models |
| BenchCAD, without tools | 72.8% | Single-shot generation under a different tool setting |
| OfficeQA Pro v2 | 64.4 | Retrieval and numerical reasoning over financial documents |
| Frontier Finance | 58.6 | Financial workflows using the official finance agent v2 harness |
| Kimi Hallu Bench Hard, with tools | 8.3% claim-level hallucination rate | Internal factual-accuracy stress test; lower is better |

For Terminal-Bench, Moonshot reports K3.1 Pro at 65.7% and Claude Opus 5.5 at 66.4%. The small gap should not be treated as a decisive ranking: Anthropic reports a standard error of 2.6 percentage points for its own Opus result, and the vendors use different reasoning settings. The [K3.1 Pro versus Opus 5.5 comparison](/comparisons/kimi-k3-1-pro-vs-opus-5-5) explains the pricing and evaluation differences.

BenchCAD measures overlap between a generated shape and the reference shape, like comparing how closely two silhouettes align in three dimensions. Moonshot evaluates open models on a 1,000-case sample from a 17,900-case dataset and imports the Claude results from Anthropic's system card. Those figures do not establish a new controlled head-to-head run.

Moonshot's hallucination test deliberately selects difficult prompts. Its 8.3% claim-level error rate is not the probability that an ordinary answer is wrong. The internal coding and factuality tests also lack independent reproduction in the sources used here.

## API behavior

The quickstart uses the OpenAI-compatible Chat Completions endpoint at `https://api.moonshot.ai/v1`. A successful account top-up of at least $1 unlocks flagship-model API access. Rate limits depend on the account tier.

Reasoning cannot be disabled. Lowering `reasoning_effort` is the documented way to reduce the amount of deliberation, much like giving an analyst less time to review a problem. Streaming separates reasoning in `reasoning_content` from the final answer in `content`. For continued conversations and tool calls, return the complete assistant message unchanged.

The model-specific quickstart fixes `temperature=1.0`, `top_p=0.95`, `n=1` and both penalty values at zero. It advises omitting those settings. For images, it documents base64 or uploaded `ms://` file references, rather than public image URLs. Its video example uses an uploaded video file.

## Training, safeguards and availability

Moonshot attributes the improvements over K3 to more post-training computation and reinforcement learning across hundreds of task domains. Reinforcement learning is practice with feedback: successful task outcomes guide later behavior. The launch does not provide a separately verified exact architecture, training-cost figure or knowledge cutoff for K3.1 Pro.

Moonshot says it removed cyber-related data from two training stages to reduce offensive capabilities. On its internal xdev-bench-hard test, the final checkpoint scores 81.0% at pass@3, compared with 90.5% without those measures. Pass@3 allows up to three attempts, like giving a candidate three tries at the same test. This is a vendor capability evaluation, not proof of safe behavior in every deployment.

Kimi Work requires version 3.1.0 or later, and Kimi Code exposes the model through `/model`. Enterprise administrators must enable K3.1 Pro because access is off by default at launch. Moonshot says zero data retention is available for API customers.

For coding, visual tool workflows and document-heavy research, K3.1 Pro offers a concrete lower-price candidate to evaluate against proprietary peers. For self-hosting, the next decision point is the promised weight release and its actual license.
$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources,
  metadata, status, published_at, sort_order
)
SELECT
  'glossary:kimi-k3-1-pro', 'glossary', 'kimi-k3-1-pro', '',
  'glossary/kimi-k3-1-pro', 'Kimi K3.1 Pro',
  'Moonshot AI''s October 2026 model with 1M context, text/image/video input and $2/$10 API pricing. Includes benchmarks, Highspeed pricing and the planned November 4 weight release.',
  entry.body,
  jsonb_build_array(jsonb_build_object('id', 'kimi-k31-card', 'type', 'markdown', 'content', entry.body)),
  $sources$[
    {"title":"Introducing Kimi K3.1 Pro","url":"https://www.kimi.ai/blog/kimi-k3-1-pro"},
    {"title":"Kimi K3.1 Pro API quickstart","url":"https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart"},
    {"title":"Kimi API model pricing","url":"https://platform.kimi.ai/docs/pricing/chat"},
    {"title":"Kimi Batch API supported models","url":"https://platform.kimi.ai/docs/guide/use-batch-api"},
    {"title":"Claude Opus 5.5 benchmark methodology","url":"https://www.anthropic.com/claude-opus-5-5"}
  ]$sources$::jsonb,
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array('kimi-k3', 'claude-opus-5-5', 'context-window', 'tool-calling', 'reasoning', 'benchmark'),
    'seoDescription', 'Kimi K3.1 Pro model card: 1M context, $2/$10 API pricing, Highspeed rates, coding and finance benchmarks, reasoning limits and planned open weights.',
    'seoKeywords', jsonb_build_array('Kimi K3.1 Pro', 'kimi-k3.1-pro', 'Kimi 3.1 Pro', 'Kimi K3.1 Pro Highspeed', 'Kimi K3.1 Pro vs Opus 5.5'),
    'verification_pass', 'kimi-k31-pro-2026-10-24'
  ),
  'published', NOW(), 0
FROM entry
WHERE true
ON CONFLICT (kind, slug, parent_slug) DO UPDATE SET
  title = EXCLUDED.title, summary = EXCLUDED.summary, body = EXCLUDED.body,
  blocks = EXCLUDED.blocks, sources = EXCLUDED.sources, metadata = EXCLUDED.metadata,
  status = EXCLUDED.status, updated_at = NOW();

INSERT INTO models (
  slug, name, developer, release_date, access, family,
  comparison_data, sources, notes, metadata, verified_at
)
VALUES (
  'kimi-k3-1-pro', 'Kimi K3.1 Pro', 'Moonshot AI', DATE '2026-10-23',
  'proprietary', 'Kimi K3.1',
  $specs${
    "Developer":"Moonshot AI",
    "Release date":"2026-10-23",
    "API model ID":"kimi-k3.1-pro",
    "Context window":"1,048,576 tokens",
    "Max output":"max_completion_tokens: 131,072 default; up to 1,048,576 (subject to context budget)",
    "Knowledge cutoff":"Not specified in reviewed sources",
    "Reasoning / effort":"Always on; low / high / max; max default in quickstart",
    "Input / 1M tokens":"$2.00",
    "Cached input / 1M":"$0.20",
    "Cache write / 1M":"$2.00 (5 min) / $4.00 (1 hr)",
    "Output / 1M tokens":"$10.00",
    "Batch / flex discount":"K3.1 Pro not listed as supported in the current Batch API guide",
    "Long-context surcharge":"None; flat pricing across context lengths",
    "Text input":"Yes",
    "Image / vision input":"Yes",
    "Video input":"Yes",
    "Audio input":"No native audio input documented in reviewed sources",
    "File / document input":"Via file extraction and text or image input",
    "Text output":"Yes",
    "Image output":"No native image output documented; external tools can create assets",
    "Audio output":"No native audio output documented",
    "Video output":"No native video output documented; external tools can render video",
    "Tool / function calling":"Yes; tool_choice=required supported",
    "Computer use":"Evaluated with Kimi Agent; requires a tool harness",
    "API access":"Kimi API; flagship access after a minimum $1 top-up",
    "Product access":"Kimi, Kimi Work 3.1.0+, Kimi Code, Kimi API",
    "Weights / license":"Hosted at launch; full weights promised by November 4, 2026; final license not verified",
    "Primary focus":"Coding, 3D creation, financial research and multi-step agent work",
    "Efficiency / generation change":"Highspeed variant: vendor-claimed 2-3x speed at $4 input / $20 output per 1M tokens",
    "Safety / approvals":"Vendor reports cyber-related training-data filtering; enterprise access off by default"
  }$specs$::jsonb,
  $sources$[
    {"title":"Introducing Kimi K3.1 Pro","url":"https://www.kimi.ai/blog/kimi-k3-1-pro"},
    {"title":"Kimi K3.1 Pro API quickstart","url":"https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart"},
    {"title":"Kimi API model pricing","url":"https://platform.kimi.ai/docs/pricing/chat"},
    {"title":"Kimi Batch API supported models","url":"https://platform.kimi.ai/docs/guide/use-batch-api"}
  ]$sources$::jsonb,
  'Verified October 24, 2026. Hosted-only access at verification; open weights are promised by November 4. Benchmark figures are Moonshot-reported, generally at max effort. BenchCAD uses a 1,000-case sample. Internal scores are not independently reproduced. Exact parameter count, knowledge cutoff and final license are not inferred from K3.',
  '{"catalog_status":"current","glossary_slug":"kimi-k3-1-pro","verification_pass":"kimi-k31-pro-2026-10-24","weights_status":"announced_not_released","weights_promised_by":"2026-11-04"}'::jsonb,
  NOW()
)
ON CONFLICT (slug) DO UPDATE SET
  name = EXCLUDED.name, developer = EXCLUDED.developer, release_date = EXCLUDED.release_date,
  access = EXCLUDED.access, family = EXCLUDED.family, comparison_data = EXCLUDED.comparison_data,
  sources = EXCLUDED.sources, notes = EXCLUDED.notes, metadata = EXCLUDED.metadata,
  verified_at = EXCLUDED.verified_at, updated_at = NOW();

INSERT INTO model_benchmarks (
  id, model_slug, category, benchmark_name, benchmark_version, score_numeric,
  score_display, score_unit, tools, reasoning_effort, harness, evaluator,
  evaluation_date, source, notes
)
VALUES
  ('tq-20261023-kimik31pro-terminal40', 'kimi-k3-1-pro', 'coding', 'Terminal-Bench 4.0', '4.0',
   65.7, '65.7%', 'percent', true, 'max', 'Claude Code', 'Moonshot AI', DATE '2026-10-23',
   'https://www.kimi.ai/blog/kimi-k3-1-pro', 'Vendor-reported launch result; temperature 1.0, top-p 0.95. Evaluation date records publication.'),
  ('tq-20261023-kimik31pro-codebench3', 'kimi-k3-1-pro', 'coding', 'Kimi Code Bench 3.0 Long-horizon', '3.0',
   61.2, '61.2% (Moonshot internal)', 'percent', true, 'max', 'Claude Code', 'Moonshot AI', DATE '2026-10-23',
   'https://www.kimi.ai/blog/kimi-k3-1-pro', 'Internal benchmark without independent reproduction in reviewed sources.'),
  ('tq-20261023-kimik31pro-benchcad', 'kimi-k3-1-pro', 'multimodal', 'BenchCAD', NULL,
   92.7, '92.7% voxel IoU (1,000-case sample)', 'percent', true, 'max', 'Claude Code', 'Moonshot AI', DATE '2026-10-23',
   'https://www.kimi.ai/blog/kimi-k3-1-pro', 'Voxel overlap, not task pass rate. Open models evaluated on 1,000 cases sampled from 17,900.'),
  ('tq-20261023-kimik31pro-officeqa2', 'kimi-k3-1-pro', 'professional', 'OfficeQA Pro v2', '2',
   64.4, '64.4 (Moonshot-reported)', 'points', true, 'max', 'Claude Code', 'Moonshot AI', DATE '2026-10-23',
   'https://www.kimi.ai/blog/kimi-k3-1-pro', 'Entire PDF corpus supplied as rendered images. Runs retrieving reference answers online receive zero.'),
  ('tq-20261023-kimik31pro-frontierfinance', 'kimi-k3-1-pro', 'professional', 'Frontier Finance', NULL,
   58.6, '58.6 (Moonshot-reported)', 'points', true, 'max', 'Official finance agent v2', 'Moonshot AI', DATE '2026-10-23',
   'https://www.kimi.ai/blog/kimi-k3-1-pro', 'Launch chart result using the official finance agent v2 harness.')
ON CONFLICT (id) DO UPDATE SET
  score_numeric = EXCLUDED.score_numeric, score_display = EXCLUDED.score_display,
  score_unit = EXCLUDED.score_unit, tools = EXCLUDED.tools,
  reasoning_effort = EXCLUDED.reasoning_effort, harness = EXCLUDED.harness,
  evaluator = EXCLUDED.evaluator, evaluation_date = EXCLUDED.evaluation_date,
  source = EXCLUDED.source, notes = EXCLUDED.notes;
