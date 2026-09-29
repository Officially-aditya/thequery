-- Add Terminal-Bench 4.0 Accuracy vs. Cost chart to the article:
-- Why You Should Look at Cost Per Successful Task Instead Of Price Per Token
WITH article_body AS (
  SELECT $body$LLM prices vary a lot. There are cheap models as well as expensive models, and both have their benefits and drawbacks, but today we're covering how standard per-token pricing is often misleading.

After the release of Sonnet 5.5, a debate started trending on X (formerly Twitter) about whether devs should use Sonnet 5.5 rather than Opus 5.5, following the release of Artificial Analysis's Cost per Task report, in which Sonnet 5.5 became the second table topper, falling just behind Fable 5.1.

## Cost Per Task

Artificial Analysis defines cost per task as "The weighted-average cost (USD) to complete one Artificial Analysis Intelligence Index task." That means that models at identical token pricing can use more reasoning tokens than other models. Some models spend more time reasoning than actually completing the task at hand. This depicts the actual price the user has to pay, even when two models share the same token price.

Take Sonnet 5.5 and GPT 6 Sol, for instance. Both cost $2/$10, but their Artificial Analysis cost-per-task rankings differ hugely. GPT 6 Sol stands at $1.99 and Sonnet 5.5 at $7.60. This huge difference comes mostly from the varying reasoning tokens spent by each model.

![Terminal-Bench 4.0: Accuracy vs. cost](/claude-sonnet-5-5-terminal-bench-4.png)
<small>Source: Anthropic, Introducing Claude Sonnet 5.5</small>

## Benchmarks Depict Only Half the Equation

You must have seen a trend of changing benchmark styles in recent model releases. That is because earlier, the comparison tables only depicted the models at their best effort, neglecting the reasoning tokens spent by the model. So, after o3's ARC-AGI results in December 2024, ARC Prize made efficiency to be documented along with cost, making it a requirement. That's why the newer benchmarks carry detailed charts for measuring both accuracy and efficiency.

## Progress Is Largely a Cost Story

Earlier, we used to spend a lot compared to what we're spending today for the same task. Epoch estimates that the cost of achieving a given performance level has fallen about 47% per quarter, or 13× per year, since 2023. Their example: o3 reached a high GPQA Diamond score (about 81% raw) for roughly 30 cents per question, and GPT-5.6 Luna matched it about 18 months later for $0.0004, a 725-fold drop. That's a complete overhaul of the cost metric within a few years.

## Accuracy Alone Is Not Enough

We've also noticed that some models cost much more than the baseline for negligible accuracy gains. Recent models have shown a different pattern, with xhigh effort achieving better results than max effort. We believe the reason for that is simply that models spend more time complicating things at max effort. We believe AI labs should optimise for cost per successful task rather than building complex thinking machines.$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'article:why-you-should-look-at-cost-per-successful-task-instead-of-price-per-token',
  'article',
  'why-you-should-look-at-cost-per-successful-task-instead-of-price-per-token',
  '',
  'articles/why-you-should-look-at-cost-per-successful-task-instead-of-price-per-token',
  'Why You Should Look at Cost Per Successful Task Instead Of Price Per Token',
  'Per-token pricing hides what models really cost. Artificial Analysis''s cost-per-task data shows why cost per successful task matters most.',
  article_body.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', article_body.body)),
  '[]'::jsonb,
  '{"manualGlossaryLinks": false}'::jsonb,
  NULL,
  NULL,
  'published',
  DATE '2026-09-28',
  0
FROM article_body
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
