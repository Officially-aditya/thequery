-- Publish the approved article on AI subscription subsidies sitting on the plan.
WITH article_body AS (
  SELECT $subsidy_body$On June 10, 2026, SemiAnalysis reported that a $200 Claude plan could burn through about $8,000 of tokens at API prices, and a $200 ChatGPT plan about $14,000. On October 5, it ran the test again. The ChatGPT plan came back at about $2,100 on one model and $2,900 on another. The Claude plan came back at about $2,500 on one model and $11,700 on another.

Two price tags. Six numbers. That is not sloppy measurement. It is the finding. A flat-rate plan has no fixed worth, and every figure about its subsidy is a snapshot of one plan, one model and one workload on one date.

The subsidy is real. It is also narrower than we argued on March 15, 2026. It sits on flat-rate plans worked hard by agents, not on every token a lab serves, and the evidence has moved enough to say so plainly, including where our own earlier piece overreached.

## Why Six Numbers Describe Two Plans

SemiAnalysis’s method explains the spread. A subscription does not publish a token allowance. It shows a usage meter from 0 to 100 percent, across a five-hour window and a weekly window. To turn that meter into dollars, SemiAnalysis ran prompts built to burn one kind of token at a time, such as input or output, watched how far the meter moved, and priced the inferred capacity at each lab’s public list prices. The result is called API-equivalent value. Picture an all-you-can-eat buffet where someone totals everything you ate at the à la carte menu. The total tells you what the meal would have cost on the menu. It does not tell you what the kitchen paid to cook it.

The June 10 test, as reported, put a maxed-out $200 Claude plan near $8,000 and a $200 ChatGPT plan near $14,000, against a common belief of about $2,000. The October 5 report describes a different market. OpenAI had halved the allowance on its $200 plan at the end of September. Anthropic’s top plan returned a figure that depends almost entirely on which model it is spent on.

| Plan (all $200 a month) | Model | API-equivalent value at full use |
| --- | --- | --- |
| Claude Max 20x (renamed Ultra on October 30) | Opus 5.5 | $11,726 |
| Claude Max 20x (renamed Ultra on October 30) | Fable 5.1 | $2,485 |
| ChatGPT Pro 200 | GPT-6.1 Sol | $2,084 |
| ChatGPT Pro 200 | GPT-6 Astra | $2,897 |

Source: SemiAnalysis, October 5, 2026. The Fable and Astra figures are in the public part of the report. The Opus and Sol figures come from secondary summaries of its paywalled charts, which agree with each other, so treat them as reported rather than verified.

On the mid-tier models both labs sell as the daily driver, SemiAnalysis found Claude plans worth about five times the API-equivalent value of the ChatGPT equivalents, and says the gap survives a switch to raw token counts: 2.9 billion tokens a month on Claude Pro against 1 billion on ChatGPT’s $20 plan, per The Register’s reading. On the flagships the two are close, $2,485 of Fable 5.1 against $2,897 of Astra, and Fable can draw on only half of the Claude weekly limit. The same Claude plan swings by a factor of 4.7 depending on the model. SemiAnalysis’s own lesson is that a plan’s worth only exists as a combination of plan, model and workload.

## List Price Is Not Cost

The buffet problem is where subsidy claims go wrong. API-equivalent value is a list price, and a list price contains margin. SemiAnalysis assumes Anthropic earns a 92 percent gross margin on API sales. That is its estimate, not a reported figure. At that margin, serving $11,726 of Opus 5.5 at list costs roughly $938. Against $200 paid, a user who maxes the plan every month costs Anthropic about $740 more than they pay, which is the minus 369 percent margin SemiAnalysis reports. Run the same arithmetic on Fable 5.1 and $2,485 of list value costs about $199. That is break-even, and it matches the report’s 1 percent.

Those are the numbers at full use. Almost nobody uses a plan at full use. At a more typical 20 percent utilization, SemiAnalysis models the margins flipping: 6 percent on Opus 5.5 and 80 percent on Fable 5.1. This is how a gym works. The business depends on most members not showing up, and the member who arrives every morning is the problem. An agent is a member who never leaves.

So a 59x gap between list value and price is not a 59x loss. At the extreme it is a loss of about $740 a month on one heavy user. At typical use it is a profit. The reason labs care anyway is concentration. SemiAnalysis’s rough model has subscriptions at about 10 percent of Anthropic’s revenue and over 40 percent of its inference compute, which is how a small revenue line can move a whole margin.

## Where Our March Piece Overreached

On March 15, 2026, we published The VC Subsidy Behind Cheap AI Will Not Last. It said that every time you send a complex query to Claude, GPT-5 or Gemini, “the AI lab is losing money on the transaction,” and it called that a fact rather than speculation. It framed venture capital as the money underwriting the gap between what AI costs to run and what labs charge, on the model of the $4 Uber ride.

That sentence was too broad. What holds up is that AI feels cheap because someone else is paying for something. What does not hold up is the claim that the something is the token on every query.

The strongest case against it comes from Hadley Wickham, who on November 3, 2026 priced out serving Kimi K3, an open model he uses as a proxy for a frontier one. By his calculation, a server costing about $23,000 a month, at the throughput SemiAnalysis measured and 75 percent utilization, serves tokens for roughly $0.25 per million. Hosting providers sell the same model at a blended $1.13 per million on a typical agentic coding mix. He flags his own limits: the analysis is crude, it leans on Claude for some cost inputs, and an open model is a proxy for closed ones. His conclusion is that per-token API prices are mostly honest measures of cost and that the subsidies sit on subscription plans.

Anthropic’s own filing points the same way, loosely. Its S-1 reports $16.29 billion of revenue and $2.38 billion of cost of revenue for the first half of 2026, which CNBC calculated as an implied gross margin of about 85 percent, up from 12 percent a year earlier. The filing also warns that margins will fluctuate as new compute arrives. A blended figure cannot say which channel carries the margin. It does make “every query loses money” hard to defend for the company as a whole.

The old argument keeps one real leg. Wickham counts the cost of serving a token, not the cost of building the model that serves it. Training is the cost of building the factory. A per-token price can cover a shift’s running cost and still leave the factory unpaid, and investors are paying that bill. Wickham concedes that you can call this a subsidy, and the concession is fair. What changes is where we point the word. The subsidy on the token is arguable and mostly a training-cost story. The subsidy on the plan is measured.

## Labs Are Cutting the Subsidy Without Touching the Price

Neither lab raised the sticker price of a plan. Both changed what the sticker buys, through the credit system behind the meter. That system works like a prepaid calling card where each destination burns minutes at its own rate. Every model and token type burns plan credits at a rate of its own, and nothing forces those rates to track API prices, so a lab can reshape a plan’s worth by editing a number the subscriber never sees. SemiAnalysis caught one provider doing a small version of exactly that: an A/B test that gave one older account about 20 percent lower limits than two identical ones. The provider called the test “extremely tiny” and said it was balancing when users hit limits. The byproduct was proof that limits can move without an announcement.

OpenAI took the blunt route. It cut the allowance on its $200 plan roughly in half at the end of September, let existing subscribers keep the old allowance until October 29, added a $500 tier, and removed the relative-usage labels, the 5x and 20x multiples, from its pricing page. SemiAnalysis found the $500 plan offers 21 percent more Astra than the old $200 plan, and that after the cut the $100, $200 and $500 tiers all give the same tokens per dollar. Heavy users used to get a better deal per dollar from OpenAI. They no longer do.

Anthropic’s lever is quieter. Its tiers already gave the same value per dollar, so there was no volume discount to remove. Instead its top model, Fable 5.1, draws on only half of the weekly limit, and a plan’s value falls as the model gets more expensive. SemiAnalysis’s read is that Anthropic would likely already have software-like margins on subscriptions if everyone only used Fable. When Opus 5.5 cut list prices by 20 percent on input and output, Anthropic raised plan limits by about 20 percent on Max and 50 percent on Pro. That was not enough to offset the cut, so API-equivalent value fell.

Competition explains why the cuts are cautious. SemiAnalysis credits OpenAI’s generous limit resets with a surge in Codex adoption, and says that forced Anthropic to walk back planned subscription cuts more than once. A subsidy is hard to remove when the rival is still handing it out.

## Enterprises Pay the Meter, and It Shows

Flat plans are built for individuals and small teams. Large companies buy by the token, and they have started to cap it. Bloomberg reported on November 4, 2026 that Apple set a hard cap of $200 a day on engineers’ AI token use and may cut non-engineers to $25 a day, according to Digiday’s account. Agencies have set limits too. PMG caps staff at $50 a day through its internal portal.

Put $200 a day beside $200 a month. They are not like for like. One is a metered ceiling on corporate work, the other a flat plan with weekly limits. They sit on opposite sides of the same line, and the line is who pays list price.

Dhruv Amin, co-founder of agent company Skydive, told The Register that labs subsidize prosumer plans as “part of their marketing funnel.” People learn the tools at $200 a month, bring them to work, and stop fitting on a plan. Skydive says it cut its own AI costs by more than 75 percent by routing workloads to cheaper open models. That figure comes from a company selling the routing product, and it has not been independently checked.

## What the Numbers Still Can’t Tell Us

Most of SemiAnalysis’s method sits behind a paywall, and we read the public part. The agentic workload mix is its own September usage. The 92 percent API margin is an assumption. Each of those moves the result, and none can be checked from outside.

The revenue split does not match. SemiAnalysis modeled subscriptions at about 10 percent of Anthropic’s revenue. Three weeks later, Anthropic’s amended S-1 reported subscription-based revenue at 21 percent for the nine months to September 30, 2026, and CNBC’s read of the original filing put it near 20 percent for the first half. The filing’s category may be broader than consumer plans, and we cannot tell from the public material. If the larger share is right and the 42 percent compute estimate holds, the skew between revenue and compute shrinks from about four to one to about two to one.

A rough cross-check, ours and fragile, points the same direction. Take 42 percent of the filing’s first-half cost of revenue, about $1 billion, against about $3.3 billion of first-half subscription revenue. That implies a subscription margin near 70 percent, which does not look like a heavy subsidy. The periods and definitions do not line up cleanly, so the number is not the point. The point is that company accounting and outside modeling have not been reconciled in public, and a heavy-user subsidy can coexist with a profitable plan line because light users cover it.

API-equivalent value also rewards expensive models. Artificial Analysis puts the cost of its benchmark task at $0.72 on GPT-6.1 Sol and $5.98 on Opus 5.5, according to The Register. A plan that returns five times the API-equivalent dollars is not returning five times the finished work. SemiAnalysis answers that the industry lacks reliable data on token efficiency and disputes that the Artificial Analysis tasks resemble real work. We cannot settle that here, and neither side has published the comparison that would.

## The Subsidy Moved

The subsidy did not vanish, and it was never everywhere. It sits in a narrow band: flat-rate plans, heavy agent use, expensive models. Labs are closing that band by shrinking limits and steering users toward cheaper models instead of raising prices, so the sticker stays put while the plan underneath it changes. Anyone building a cost model on a $200 plan is building on a number that changed between June and October without the price moving. Anyone building on API prices is building on something much closer to cost, with the training bill left out.

The token was never the gift. The flat fee was.

**Previously on TheQuery:** [The VC Subsidy Behind Cheap AI Will Not Last](https://www.thequery.in/articles/ai-vc-subsidy-inference-costs-engineers) and [Why You Should Look at Cost Per Successful Task Instead Of Price Per Token](https://www.thequery.in/articles/why-you-should-look-at-cost-per-successful-task-instead-of-price-per-token)$subsidy_body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'article:ai-subscription-subsidy-plan-not-token',
  'article',
  'ai-subscription-subsidy-plan-not-token',
  '',
  'articles/ai-subscription-subsidy-plan-not-token',
  'AI Subscription Subsidy Sits on the Plan, Not the Token',
  'A flat-rate plan has no fixed worth. The subsidy sits on subscription plans worked hard by agents rather than every token served, as labs quietly cut limits without touching sticker prices.',
  article_body.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', article_body.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Anthropic Subscriptions Offer 5x+ More Value Than OpenAI - SemiAnalysis', 'url', 'https://newsletter.semianalysis.com/p/anthropic-subscriptions-offer-5x'),
    jsonb_build_object('title', 'Anthropic Claude subscription plan provides more value than OpenAI''s, study says - The Register', 'url', 'https://www.theregister.com/ai-and-ml/2026/10/06/anthropic-claude-subscription-plan-provides-more-value-than-openais-study-says/5301470'),
    jsonb_build_object('title', 'How much do tokens really cost? - Hadley Wickham, AI for data science', 'url', 'https://substack.hadley.nz/p/how-much-do-tokens-really-cost')
  ),
  '{}'::jsonb,
  NULL,
  NULL,
  'published',
  DATE '2026-11-15',
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
