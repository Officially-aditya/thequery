-- Publish article: Jev Claims Zero Hallucination. The Fine Print Disagrees.
WITH article_body AS (
  SELECT $body$On September 17, a developer wired an AI agent into Minecraft and had it defeat the Ender Dragon in under nine minutes. The total bill for that run came in under a dollar. A separate demo had the same underlying model playing 50 rounds of a browser game simultaneously, and another ran it against Doom, making roughly ten decisions a second for an hour for about seven dollars. None of this was a chatbot. It was Jev, the first public model from a two-year-old stealth startup called TypeSafe AI, and the reason it's cheap enough to do any of this is that it never writes a sentence.

TypeSafe came out of stealth on September 15 with a $40 million seed round led by DCVC and a founder, Diogo Almeida, whose résumé includes real time at OpenAI working on the technique that shaped ChatGPT. That part checks out. What doesn't fully check out, once you start pulling on it, is nearly every headline number attached to the launch: the founder's own credit line, the flagship "zero hallucination" claim, and the benchmark that's supposed to prove Jev is 200 times faster than the models it's built to replace.

None of that makes Jev fake. It makes it a real, reasonably clever product wearing a launch pitch several sizes too big for it.

## What Jev Actually Does

Jev is still built on a transformer, the same underlying architecture as GPT and Claude, but TypeSafe is explicit that it isn't a large language model in the conversational sense. It has no chat window, can't write an email, and won't explain itself. Instead, you feed it a piece of unstructured context, such as an app's current state or a chunk of conversation history, along with a small, bounded question. It answers by picking one option from a predefined list, assigning a score on a scale, or returning a yes/no with a probability, and it attaches a confidence level to whichever it picks.

That's a fundamentally different shape of answer than a chatbot gives. Asking a chatbot to classify something is like asking someone to write you a paragraph and then combing through it yourself for the one word you actually needed. Jev skips the paragraph and hands you the word directly, in 70 to 500 milliseconds, at $0.042 per million input tokens with output tokens free. TypeSafe says that adds up to 20 to 200 times the speed and up to roughly 400 times lower cost than routing the same decision through a standard chatbot.

TypeSafe trains Jev with a method it calls Reinforcement Learning for Calibrated Decisions, or RLCD, which it positions as a deliberate departure from RLHF, the technique that shaped how today's chatbots behave. The pitch is that RLHF optimizes a model to produce text people rate highly, while RLCD optimizes a model to produce a decision that matches a predefined, verifiable outcome. Whether that method is actually new is the first place the launch starts to strain, and it isn't the last.

## The Founder's Résumé, Checked Against the Paper Trail

TypeSafe's own team page states that Almeida "co-invented RLHF and InstructGPT, the methods that lead to ChatGPT and GPT4." Almeida is indeed listed as the fourth of twenty authors on the InstructGPT paper, published and peer-reviewed at NeurIPS 2022, a genuine and significant credit. But RLHF itself predates that paper by five years. It comes from a 2017 paper by a different team entirely, Christiano, Leike, Amodei, and colleagues, with no Almeida listed anywhere on it.

Developer Fabio Akita ran this down directly after a string of people recommended TypeSafe to him unprompted, which he says triggered his own skepticism before he'd looked at anything. Checking publication dates against the company's own claims, he concluded plainly that Almeida "helped apply RLHF to train InstructGPT" but did not invent RLHF, calling it résumé inflation you can prove with a calendar. The credential underneath is real. The specific sentence describing it is not.

## Real Money, Self-Graded Benchmark

The funding is not in dispute. DCVC led a $40 million seed round, and Forbes independently reported a $200 million valuation based on a source with direct knowledge of the deal, not just TypeSafe's own announcement. That's a serious number from a serious investor, not vaporware.

The headline performance claim is a different story. TypeSafe's launch post benchmarks Jev against what it calls "the smartest models," almost certainly OpenAI's new flagship Astra and Anthropic's Claude Fable 5.1, and reports Jev sitting on the Pareto frontier by nearly two orders of magnitude. TypeSafe ran that comparison itself. As of this writing, no independent lab has reproduced the headline 193.6x speed and 444.6x cost figures. More tellingly, TypeSafe's own post discloses that its reference answer key for grading accuracy is the average of Astra's and Fable's answers, and admits outright that this choice "biases answers towards OpenAI and Anthropic's models." Grading a test using the average of two other students' answer sheets instead of the actual answer key is a strange way to prove you outperformed the class.

## Zero Hallucination Is a Schema Guarantee, Not an Accuracy Guarantee

The most quoted line from the launch is that Jev hallucinates zero percent of the time. TypeSafe's own post explains what that number actually measures: schema matching is guaranteed by construction, so the model can only ever return one of the predefined options, never a fifth category that doesn't exist. The company states plainly that this figure is not empirical, it's a property of the design.

That's a real and useful guarantee, but it's a narrower one than the headline implies. A multiple-choice test that only lets you bubble in one of four ovals guarantees you'll never write a fifth answer in the margin. It says nothing about whether you bubbled in the correct oval. Jev can still pick the wrong category with high confidence. TypeSafe's own limitations documentation, for what it's worth, admits as much: it lists weak numeric counting, date-comparison errors, degradation when irrelevant context is present, and vulnerability to adversarial manipulation as known issues in the current model.

## What Independent Testers Actually Found

Two outside tests give an actual read on accuracy rather than TypeSafe's own framing, and they don't tell a single clean story.

| Test | Jev | Comparison model | Jev's edge | Jev's tradeoff |
|---|---|---|---|---|
| Every newsletter, proofreading passages | 0.35 sec, caught 6 of 7 planted errors | Claude Fable 5.1: 8.83 sec, caught 7 of 7 | About 25x faster | Missed the one error Fable caught |
| Phishing dataset (2,000 emails), single verdict | 62.6% accuracy | Claude Haiku 4.5: 81.3% accuracy | Speed and cost | 18.7-point accuracy gap |
| Same phishing dataset, five decomposed signal questions combined by code | 95% accuracy | Claude's composed version: 93.2% accuracy | Beat Claude once broken into sub-questions | Only works if the workflow is redesigned around it |

The pattern across both tests is consistent: asked for one verdict in one shot, Jev is fast, cheap, and measurably worse than the model it's meant to replace. Asked a handful of narrower, decomposed questions that code then combines into a decision, Jev edges ahead. That's a real finding, and it's a more honest description of what Jev is good at than "faster and more accurate than an LLM" would suggest.

## A Training Method Wearing Somebody Else's Name

RLCD is the acronym doing the most work in TypeSafe's pitch, and it turns out to already belong to someone else. A peer-reviewed 2023 paper titled Reinforcement Learning from Contrastive Distillation, with code published on Facebook Research's GitHub by a team that includes Meta-affiliated authors, uses the exact same abbreviation for a completely unrelated technique. TypeSafe's version has no published paper, no architecture description, and nothing beyond a paragraph in a blog post. It isn't the same method borrowing a similar name by coincidence. It's the same three letters, attached to two unrelated things, with only one of them backed by anything a reader can go check.

## The Verdict

Strip away the launch-day framing and the underlying idea is genuinely reasonable: plenty of production systems today send a full prompt to a chatbot, wait several seconds for a paragraph, then regex a label out of it, and that pattern really is slow and wasteful for a decision that only ever has a few possible answers. Building a cheap, fast, narrow decision layer underneath the expensive reasoning models is a legitimate architectural pattern, and Jev is a legitimate, working implementation of it. Akita, after doing the actual checking rather than reacting to the pitch, landed on calling it a small product with a pitch way bigger than what it actually delivers. That's a fair summary of a launch where the funding is confirmed, the founder's underlying credential is confirmed, and almost everything layered on top of those two facts needed a second look before it held up.

Diogo Almeida helped build RLHF. TypeSafe's own website just says he invented it.$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'article:jev-zero-hallucination-fine-print-disagrees',
  'article',
  'jev-zero-hallucination-fine-print-disagrees',
  '',
  'articles/jev-zero-hallucination-fine-print-disagrees',
  'Jev Claims Zero Hallucination. The Fine Print Disagrees.',
  'TypeSafe''s Jev promises zero hallucination, 200x speed, and 400x lower cost. Checking the founder''s credit line, the self-graded benchmark, and independent tests shows what holds up.',
  article_body.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', article_body.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Why Things Like TypeSafe AI Don''t Interest Me - Fabio Akita', 'url', 'https://akitaonrails.com/en/2026/09/16/why-things-like-typesafe-ai-dont-interest-me/'),
    jsonb_build_object('title', 'Introducing System One Models & Jev - TypeSafe AI', 'url', 'https://typesafe.ai/blog/introducing-system-one-models-and-jev'),
    jsonb_build_object('title', 'TypeSafe AI Jev Funding Puts a 445x Cost Claim Under Scrutiny - Remio', 'url', 'https://www.remio.ai/post/typesafe-ai-jev-funding-puts-a-445-cost-claim-under-scrutiny'),
    jsonb_build_object('title', 'A new kind of AI model from a ChatGPT inventor is thrilling developers - TechCrunch', 'url', 'https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/'),
    jsonb_build_object('title', 'Meet Jev: A New Kind of AI Model From One of ChatGPT''s Co-Creators - TechSpot', 'url', 'https://www.techspot.com/article/3172-meet-jev/')
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
