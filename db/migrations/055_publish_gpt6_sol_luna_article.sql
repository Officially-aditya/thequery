-- Publish article: GPT-6 Sol and Luna Are Here. OpenAI Is Cutting the Cost of Intelligence.
WITH article_body AS (
  SELECT $body$OpenAI has expanded its GPT-6 lineup with GPT-6 Sol and GPT-6 Luna, and the timing is hard to ignore. Anthropic launched Claude Opus 5.5 earlier on September 22, and OpenAI followed shortly after with two lower-cost GPT-6 models. The headline is straightforward: Sol is now $2 per million input tokens and $10 per million output tokens, while Luna costs $0.10 per million input tokens and $0.50 per million output tokens.

The more interesting part is what OpenAI says it did to get there. Sol and Luna use techniques developed for GPT-6 Astra, including improvements in reasoning, factuality, coding, computer use and alignment, while improvements to caching and inference reduce the cost of serving them. OpenAI is therefore trying to push the frontier in two directions at once: better models and cheaper inference.

That makes GPT-6 Sol and Luna less interesting as two new model names and more interesting as evidence of where OpenAI thinks the next phase of the model market is going.

## The Price Cut Is the Story

GPT-5.6 Sol currently costs $4 per million input tokens and $20 per million output tokens, while GPT-5.6 Luna costs $0.20 per million input tokens and $1.20 per million output tokens. GPT-6 Sol cuts both prices by 50%. Luna's input price is also down 50%, while its output price falls from $1.20 to $0.50, a reduction of about 58%.

| Model        | Input / 1M tokens | Output / 1M tokens |
| ------------ | ----------------: | -----------------: |
| GPT-5.6 Sol  |             $4.00 |             $20.00 |
| GPT-6 Sol    |             $2.00 |             $10.00 |
| GPT-5.6 Luna |             $0.20 |              $1.20 |
| GPT-6 Luna   |             $0.10 |              $0.50 |

The important comparison is with the already discounted GPT-5.6 pricing, not the original launch rates. OpenAI is therefore making a substantial cut without dropping Sol and Luna into a completely different capability generation.

It is also positioning Sol for professional work, coding, automation and computer use rather than treating it as a lightweight fallback. Luna is aimed at the much larger class of high-volume workloads where inference cost matters more than squeezing out the last increment of capability.

## The Efficiency Is Under the Hood

OpenAI attributes the lower prices primarily to improvements in inference and caching. GPT-6 also increases cache hit rates, offers a 90% discount on cached input-token reads, and allows developers to modify reasoning effort and tool availability without invalidating cache reuse.

That matters particularly for agents, where the same repository, instructions or accumulated context may be sent through dozens of model calls. Reducing the amount of context that has to be processed from scratch can materially lower the cost of an entire workflow.

OpenAI says GitHub has seen the share of prompt tokens requiring fresh processing fall by more than 50% across billions of requests after several months of these caching improvements. That is an OpenAI-reported result rather than an independent benchmark, but it illustrates why inference efficiency is becoming part of the model product itself.

The important shift is therefore from **price per token to cost per completed task**. If a model needs fewer tokens, fewer retries and less fresh context processing to reach the same result, its effective cost can fall faster than the headline rate suggests.

## Sol Is the Production-Focused GPT-6

The model-to-model comparison OpenAI is emphasizing reflects that strategy. On AutomationBench, OpenAI reports GPT-6 Sol at xhigh effort scoring 33.2% at a reported $0.27 per task, versus 30.3% for GPT-6 Astra at low effort at 3.9 times the cost.

On DeepSWE v1.1, OpenAI reports GPT-6 Sol at maximum effort scoring 68.8%, within 1.1 percentage points of Claude Fable 5's highest reported score while costing approximately 80% less per task. GPT-6 Luna reaches 66.6% at maximum effort, which OpenAI says is comparable to Opus 5 and Fable 5 at medium effort.

These are vendor-reported comparisons, so the exact economics should not be treated as universal production benchmarks. The broader point is that Sol is not being positioned simply as “Astra, but worse.” It is being positioned as the GPT-6 model for workloads where capability and operating cost both matter.

## Luna May Be the Bigger Deal

Sol is the obvious model to watch because it sits closer to the frontier. Luna, however, could have a larger effect on actual usage because its pricing makes continuous inference far easier to justify.

OpenAI says GPT-6 Luna at higher effort can match GPT-5.6 Sol on its internal factuality evaluation at roughly one hundredth of Sol's cost. On OSWorld 2.0 offline, OpenAI says Luna at maximum effort can exceed GPT-5.6 Sol at medium effort at one tenth of its cost.

The significance is not that Luna replaces Sol. It is that the capability floor is moving downward.

That matters for products making thousands or millions of model calls. A modest reduction in per-call cost becomes enormous when the model sits inside a search product, support system, coding workflow or agent that runs continuously.

## The Timing With Opus 5.5 Matters

The same-day sequence is important.

Anthropic launched Claude Opus 5.5 on September 22, positioning it as a more efficient frontier model with 40% lower operating cost than Opus 5. OpenAI then released Sol and Luna, immediately putting significantly cheaper GPT-6 options into the market.

The contrast is revealing. Anthropic is pushing efficiency into its flagship class, while OpenAI is using the GPT-6 lineup to push efficiency much further down the pricing ladder.

That changes the competitive question. The important metric for developers is no longer simply which company's most expensive model performs best. It is increasingly how much useful work they can get from each dollar of inference.

For an agent making hundreds of calls, invoking tools and repeatedly processing the same context, that can matter more than a benchmark advantage measured on a single task.

## OpenAI Is Also Keeping the Tiered Strategy

GPT-5.6 introduced Sol, Terra and Luna as distinct capability tiers. GPT-6 is taking a slightly different route. Astra launched first, and Sol and Luna now extend the family downward, leaving a three-model GPT-6 lineup rather than a direct one-for-one replacement of GPT-5.6's structure.

The roles are fairly clear: Astra for the hardest workloads, Sol for sustained professional use, and Luna for high-volume inference.

What has changed is the economics between those tiers. OpenAI can take techniques developed for the flagship model and move them into substantially cheaper models instead of forcing developers to choose between maximum capability and low-cost inference.

That makes the lineup easier to route around in production. Developers can reserve Astra for tasks that genuinely need it, use Sol for most serious workloads, and push simpler or higher-volume work down to Luna.

## GPT-6 Is Becoming an Efficiency Story

GPT-6 Astra was the capability announcement. Sol and Luna make the broader strategy easier to see.

OpenAI is not only trying to make its models more capable; it is trying to make increasingly capable models cheap enough to sit inside software products at scale. The combination of lower inference costs, better caching and more efficient model behavior is what makes the new pricing possible.

That is especially important as AI moves from chat interfaces into agents. Once a model is making repeated tool calls, editing files, browsing the web or working through a long-running task, the economics of the whole workflow matter more than the price of a single response.

The frontier is still moving upward. With Sol and Luna, OpenAI is also trying to move the cost of accessing that frontier downward.

And coming immediately after Claude Opus 5.5, the message is difficult to miss: the next phase of the model race is not just about building a more capable model. It is about making that capability economical enough to run everywhere.$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'article:gpt-6-sol-luna-cutting-cost-intelligence',
  'article',
  'gpt-6-sol-luna-cutting-cost-intelligence',
  '',
  'articles/gpt-6-sol-luna-cutting-cost-intelligence',
  'GPT-6 Sol and Luna Are Here. OpenAI Is Cutting the Cost of Intelligence.',
  'OpenAI launched GPT-6 Sol at $2/$10 and Luna at $0.10/$0.50 per million tokens, halving GPT-5.6 prices with inference and caching gains. What the benchmarks, the tiered lineup, and the same-day arrival with Claude Opus 5.5 signal about the cost-per-task race.',
  article_body.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', article_body.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Introducing GPT-6 Sol and Luna - OpenAI', 'url', 'https://openai.com/index/introducing-gpt-6-sol-and-luna'),
    jsonb_build_object('title', 'OpenAI expands GPT-6 lineup with cheaper Sol and Luna models - Reuters', 'url', 'https://www.reuters.com/technology/openai-expands-gpt-6-lineup-with-cheaper-sol-luna-models-2026-09-22'),
    jsonb_build_object('title', 'OpenAI launches GPT-6 Sol and Luna - TechCrunch', 'url', 'https://techcrunch.com/2026/09/22/openai-launches-gpt-6-sol-and-luna'),
    jsonb_build_object('title', 'Introducing Claude Opus 5.5 - Anthropic', 'url', 'https://www.anthropic.com/claude-opus-5-5')
  ),
  '{}'::jsonb,
  NULL,
  NULL,
  'published',
  DATE '2026-09-22',
  0
FROM article_body
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
