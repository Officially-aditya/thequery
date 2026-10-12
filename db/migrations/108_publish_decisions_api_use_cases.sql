-- Publish the user-supplied Decisions API article with verified sources and related links.
WITH article_body AS (
  SELECT $body$Diogo Almeida spent his OpenAI years teaching language models to talk to people. By his own account, that work became the research behind ChatGPT. On September 15, 2026, his startup TypeSafe AI released a model that cannot write a sentence.

The model is called Jev. You hand it a piece of text and a question, and it hands back a probability for each possible answer. OpenAI's own documentation shows the shape: feed in "I was charged twice for my order," ask which department should handle it, and get back billing at 0.95 with a confidence of 0.93. That response is labeled illustrative, but it is the whole product in one line.

Then everyone else showed up. OpenAI previewed a Decisions API on September 29 and opened a public beta on October 6. Cloudflare, Perplexity and Amazon each released a decision model on October 1. Microsoft followed on October 9. The Register counts more than 100 such models in a Hugging Face index. This piece covers what a decisions API does, where it is being used, and where the evidence stops. The short version: the use cases are real and narrow, and nearly every performance number comes from the company selling the model.

## What a decisions API actually returns

A normal language model answers with prose. A decisions API answers with a typed value. You send a block of state, which can be a sentence, a JSON object or an image, plus one or more named questions. You get back one answer per question, each with probabilities attached. Think of a triage nurse at a hospital front desk. She reads your complaint, points you to a department, and tells you how sure she is. She does not write you a letter.

There are three question types, and the vendors mostly agree on them. A yes or no question returns the probability that a condition is true. OpenAI calls this a predicate, while TypeSafe and Perplexity call it a noul. A choice question picks one option from a fixed list and returns a probability for every option. A score question rates the input against ordered levels. OpenAI's documentation gives the arithmetic: if a bug report has a 0.1 chance of being cosmetic, 0.7 of having a workaround and 0.2 of being fully blocked, the score is 1.1, a number that can fall between levels.

The probabilities matter more than the answer. Microsoft's framing is that a 90 percent prediction should be right about nine times out of ten. That property is called calibration, and it works like a weather forecaster whose 70 percent days really do rain seven times in ten. Your code can act automatically above a threshold, and send everything below it to a person or a bigger model.

## Six launches in 24 days

Jev came first, on September 15, in early access. OpenAI showed its Decisions API at DevDay on September 29, then opened the public beta on October 6. Cloudflare, Perplexity and Amazon all shipped on October 1. Microsoft announced Microsoft-Decision-1 on October 9.

One naming correction is worth making. Microsoft's entry is a model, not an API with a new name, and it is sold through Microsoft Foundry and OpenRouter. Microsoft also once marketed a bundle called Decision inside Azure Cognitive Services, covering Personalizer, Anomaly Detector and Content Moderator. Those services are retired or retiring, with Personalizer and Anomaly Detector scheduled to end on October 1, 2026. They share a name with the new model and nothing else.

| Product | Launched | Built on | Access | Input price per 1M tokens |
| --- | --- | --- | --- | --- |
| TypeSafe Jev | Sep 15 | TypeSafe's own architecture | Hosted API, early access | \$0.042 |
| OpenAI Decisions API | Sep 29 preview, Oct 6 beta | GPT-6 Luna | Hosted API only | \$0.10 |
| Perplexity Decisions API | Oct 1 (v1.1 on Oct 6) | Qwen3.8-27B | Hosted API and open weights | \$0.04, then \$0.02 for v1.1 |
| Cloudflare Clef | Oct 1 | Qwen3.8-27B | Workers AI and open weights | Not stated in launch post |
| Amazon Strands Decider 2B | Oct 1 | Qwen3.5-2B | Open weights, no paid API | Self-hosted |
| Microsoft-Decision-1 | Oct 9 | Qwen3.5-9B | Foundry and OpenRouter | \$0.042 |

Two patterns sit in that table. First, output tokens are free at every vendor that charges, because the output is a handful of numbers. Second, four of the six launches post-train Alibaba's Qwen models, a fact The Register put in a headline about Microsoft. Microsoft says it will rebase Decision-1 on its own models and OpenAI's soon. Microsoft also prices Decision-1 at \$0.042 per million input tokens, the same rate TypeSafe set for Jev three and a half weeks earlier.

## Sorting at volume: routing, labeling and triage

The oldest use case is also the safest one. Support tickets get routed to a team. Customer feedback gets sorted into a fixed set of themes. Database rows get labeled. OpenAI's own examples are a complaint router and a severity scorer. Snowflake put a decision model behind a SQL function in private preview, aimed at running one decision across thousands of rows.

The best named deployment evidence comes from the vendors themselves, which is a limit worth keeping in view. Microsoft says its Xbox Research team used Decision-1 to sort more than 10,000 pieces of survey, Steam and social media feedback into researcher-defined themes, and found it competitive with GPT-6 Sol on quality while running over 14 times faster and 200 times cheaper. Cloudflare says its threat intelligence team uses Clef to classify website domains. In one test it returned a full set of categories in 2.2 seconds, against 4.7 seconds for gpt-oss-120b, which returned only two. Both are internal tests reported by the company that ran them.

## Agent control and verification

The use case drawing the most attention is the gate in front of an agent's next move. Microsoft lists agent controls as its first suggestion: look at a proposed step and decide whether to continue, stop, retry or hand off to a model, tool or person. LangChain's beta integration for OpenAI Decisions ships two pieces of middleware built on this idea. One routes a request to either GPT-6 Luna or the larger GPT-6 Sol. The other asks whether a tool call is risky, and blocks it when the probability is 0.5 or higher. Its documentation example wraps a tool named delete_all_backups.

The second use is judging. Microsoft suggests grading an AI response and deciding to accept, revise or reject it. TypeSafe's own blog lists scoring and guardrails for prompts, reasoning traces and outputs as a core use. The cost logic is simple. A check that was too expensive to run on every action becomes cheap enough to run on all of them. Microsoft's latency arithmetic shows why speed matters here: adding 100 milliseconds to each of 20 sequential decisions adds two seconds. It is like twenty traffic lights in a row, where a few seconds of red at each one turns into a late arrival.

The best independent test of this role came from Cisco on October 5. Its researchers compared Jev against a 1-billion-parameter classifier they trained on their own 24-category safety taxonomy. The trained classifier won. At a strict false-positive budget of 0.5 percent on Cisco's multi-turn set, it caught 61 percent of unsafe content to Jev's 34 percent. Against a prompted 31-billion-parameter judge, Jev's overall AUC was higher on seven of eight benchmarks and tied on the eighth, though the judge caught more at strict budgets on four datasets. Cisco's conclusion was that a decision model lands about as accurate as a big judge without the parsing, and works best beside a trained classifier. Cisco also notes its reference labels came from an LLM, not people.

## Real time, screens and robots

Speed opens a different set of applications, and the demos are the most vivid part of the launches. TypeSafe showed a Doom bot making ten decisions per second for roughly \$7 an hour. Its own blog adds that the demo reads structured game state, not images, and that a non-AI bot could play better. OpenAI showed frames from a driving game feeding a lane-change decision, and a Hugging Face desk robot told to follow an apple, turning its camera toward the fruit. It also paired a voice model with a decision that picks which facial expression an animated character should use. Perplexity's CEO posted a chess self-play demo in which the model sees only a screenshot and picks each move in about 140 milliseconds.

Microsoft's list reaches further: choosing the next interface action from a screenshot, selecting among predefined robot actions, and screening candidate hypotheses in scientific work. Microsoft Discovery reports that Decision-1 scored 46 times more consistently than an LLM-based score while running about three times faster. The common thread is reflex rather than reasoning. A goalkeeper does not deliberate about where the ball is going. These demos suit that reflex, and none of them shows a decision model carrying a high-stakes action alone.

## Where the evidence stops

Start with who graded whom. TypeSafe's workflow evals use the average of GPT-6 Astra and Fable 5.1 as the reference answer, so they measure agreement with two frontier models, not ground truth. TypeSafe also says its team built the workflows and that its headline of 193.6 times faster and 444.6 times cheaper is on the high end of real-world gains. Perplexity reports 85.71 percent against Jev's 84.51 on an 11-benchmark panel of its own, but The New Stack notes Jev wins six of the eleven. Microsoft says Decision-1 scored highest on accuracy across 36 benchmarks. Its appendix of models benchmarked omits Perplexity's decider and Cloudflare's Clef, an editor's note says Jev numbers were added after publication, and The Register reports it ranked second on confidence. Cloudflare says Clef leads the Jev Decision Index. Snowflake says its model tops 72 models on version 0.2.1 of that index. Each entrant is first on a leaderboard it chose. OpenAI, by The New Stack's account, showed no benchmark results at all.

Then there is the claim that these models cannot hallucinate. The Register called that a claim about format, since a well-typed answer can still be wrong. A research team at the University of Southern California and collaborators tested how wrong. In JevOut, posted September 24, short and natural-sounding context additions redirected between 61.4 and 73.2 percent of each system's initially correct decisions toward a wrong option fixed in advance. For Jev, 45.5 percent of decisions that started at 0.9 confidence or higher were flipped. The attack was optimized using the output probabilities, which is a point against exposing them. It is like a con artist who keeps adjusting the story based on how the guard flinches. The authors say the rates are not a universal failure probability, and a neutral sentence changed only 2.2 percent of Jev's decisions. It still describes the exact setup in the tool-gating example above.

Smaller gotchas exist too. A library that wraps Perplexity's API warns that identical requests can differ in the second decimal place, so thresholds need margin. OpenRouter's documentation warns that the Workers AI route for Clef reads roughly the first 2,000 tokens of text and drops the rest without an error, while Cloudflare advertises a 64,000-token window. OpenAI's own guidance is to set thresholds from labeled examples from your application.

## Choosing one, or none

Pick a decisions API when the answer space is closed, the volume is high, the latency matters and a wrong answer is cheap or reviewable. OpenAI's documentation draws the line itself: use Structured Outputs when you need extracted fields or written explanations, and function calling when you need a tool call with arguments. A decision model returns scores, not reasons. For a regulated or high-stakes check where you already hold labeled data, Cisco's result says a trained classifier still wins.

The practical choice is between renting and owning. OpenAI's model is hosted only, which The New Stack called the more consequential difference. In exchange it offers zero data retention and HIPAA use for eligible customers, and data residency in the United States and Europe. Perplexity, Cloudflare and Amazon publish open weights under permissive terms, which is the difference between eating at a restaurant and owning the recipe. The call shape is also settling. Cloudflare calls Clef Jev-API compatible, Upstage serves its Solar Decide model through the same schema, and Ollama added a /v1/systemone endpoint based on TypeSafe's API. The New Stack's headline says it plainly: OpenAI has not signed on.

The API returns the probability. You still own the threshold.

**Previously on TheQuery:** [Jev Claims Zero Hallucination. The Fine Print Disagrees.](https://thequery.in/articles/jev-zero-hallucination-fine-print-disagrees) and [GPT-6 Sol and Luna Are Here. OpenAI Is Cutting the Cost of Intelligence.](https://thequery.in/articles/gpt-6-sol-luna-cutting-cost-intelligence)$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'article:decisions-api-use-cases-self-graded-scores',
  'article',
  'decisions-api-use-cases-self-graded-scores',
  '',
  'articles/decisions-api-use-cases-self-graded-scores',
  'Decisions API Use Cases Are Real. Scores Are Self-Graded.',
  'Six decision-model launches promise fast, cheap routing and agent checks. The use cases are real, but vendor benchmarks and confidence scores need scrutiny.',
  article_body.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', article_body.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Decisions guide - OpenAI', 'url', 'https://developers.openai.com/api/docs/guides/decisions'),
    jsonb_build_object('title', 'Introducing Microsoft-Decision-1 - Microsoft Command Line', 'url', 'https://commandline.microsoft.com/microsoft-decision-1-model-foundry/'),
    jsonb_build_object('title', 'Introducing System One Models & Jev - TypeSafe AI', 'url', 'https://typesafe.ai/blog/introducing-system-one-models-and-jev'),
    jsonb_build_object('title', 'Introducing Clef - Cloudflare Blog', 'url', 'https://blog.cloudflare.com/clef-decision-models/'),
    jsonb_build_object('title', 'Jev: Decision Models on Trial - Cisco Blogs', 'url', 'https://blogs.cisco.com/ai/jev-decision-models-on-trial'),
    jsonb_build_object('title', 'JevOut: Natural Context Can Flip Decision Models - USC and collaborators', 'url', 'https://arxiv.org/abs/2609.30243')
  ),
  '{}'::jsonb,
  NULL,
  NULL,
  'published',
  DATE '2026-10-12',
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
