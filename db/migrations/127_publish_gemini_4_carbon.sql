-- Add Gemini 4 Carbon as a published glossary model card.
-- Content and references are transcribed from the user-provided model card.

WITH gemini_4_carbon_entry AS (
  SELECT
    $gemini_carbon_model_card$
Gemini 4 Carbon is Google DeepMind's October 2026 frontier [large language model](https://www.thequery.in/glossary/large-language-model) and the second model of the Gemini 4 generation, succeeding Gemini 4 Argon. Announced on October 28, 2026, it is described by Google as its strongest model, with state-of-the-art cybersecurity and long-horizon capabilities. Argon was announced on September 30 and reached developers on October 13, so Carbon follows its predecessor's general release by about two weeks.

Carbon is not generally available. Starting the week of the announcement, Google made it available to a subset of cyber defenders and said it will phase in additional trusted partners through its Fairwind Program, while it gathers feedback and iterates on guardrails. Google gave no date for developers, businesses, or consumers, saying only that it will come as soon as possible, and said it is taking part in the US government's voluntary pre-release testing process. For a normal user that means waiting. For a developer it means there is no public API model ID, and Artificial Analysis lists the model as having no publicly available API.

## Core profile

| Attribute | Gemini 4 Carbon |
| --- | --- |
| Model ID | Not published |
| Announced | October 28, 2026 |
| Access | Fairwind Program (cyber defenders first) |
| Context window | 1M tokens (reported by Artificial Analysis and Vals AI) |
| Maximum output | 1M tokens (reported by Vals AI) |
| Input / output | Text, image, video, and speech in; text out |
| Knowledge cutoff | Not published |
| Output speed | 99.8 tokens per second (Argon: 82.0), per Artificial Analysis |

Google has not published a technical report, architecture details, a parameter count, or an API specification for Carbon. The context and output limits above come from outside evaluators, who matched Argon's 1M-token output limit, and have not been confirmed by Google. Artificial Analysis measures Carbon at its High reasoning setting.

Carbon is designed for harder, longer engineering problems than quick, isolated code completions. Google says its internal teams have watched Carbon run for days, persisting through dead ends to root-cause and patch subtle bugs across large codebases and low-level systems. Google also reports gains in agentic and scientific tasks, full-stack web development, and multimodal understanding, and positions the model for sustained [AI agent](https://www.thequery.in/glossary/ai-agent) workflows and the full cybersecurity defense lifecycle of vulnerability detection, validation, and remediation.

## Benchmark profile

Google's launch table compares Carbon with Gemini 4 Argon, GPT-6 Astra, Claude Opus 5.5, and Claude Fable 5.1 across 17 benchmarks. Carbon leads 15 and improves on Argon in every row. Claude Opus 5.5 leads OSWorld-2.1, and GPT-6 Astra leads FrontierSWE 2, where Carbon is third behind Astra and Opus 5.5. Claude Sonnet 5.5, GPT-6.1 Sol, and GPT-6.1 Terra, all released in the preceding month, are not in Google's comparison set.

| Benchmark (Google-reported) | Gemini 4 Carbon | Gemini 4 Argon | GPT-6 Astra | Claude Opus 5.5 | Claude Fable 5.1 |
| --- | --- | --- | --- | --- | --- |
| AA Intelligence Index v5.0 | **59** | 50 | 57 | 58 | 49 |
| Vals Index v2.2 | **70.4%** | 69.2% | 65.0% | 70.2% | n/a |
| SWE-sweep | **9.6%** | 4.3% | 9.5% | 3.6% | 2.9% |
| FrontierSWE 2 | 61.2% | 55.0% | **65.5%** | 62.3% | 56.3% |
| Terminal-Bench 4.0 (AA) | **65.2%** | 57.1% | 59.1% | 59.6% | 52.0% |
| Agents' Last Exam v1.1 | **77.4%** | 69.0% | 73.1% | 71.6% | n/a |
| OSWorld-2.1 (offline subset) | 81.5% | 74.7% | 80.9% | **87.2%** | 76.8% |
| AutomationBench (Zapier) | **53.9%** | 51.3% | 41.4% | 42.5% | 31.4% |
| Vending-Bench 2 (ending balance) | **$16,484** | $13,718 | $15,515 | $9,235 | $5,422 |
| Chartography | **72.5%** | 71.6% | 71.0% | 66.3% | 46.2% |
| Blueprint-Bench 2 | **58.6%** | 54.4% | 49.7% | 51.2% | 41.9% |
| GraphWalks, 256K to 1M (BFS F1) | **96.2%** | 84.2% | 71.8% | 66.8% | 65.0% |
| RiemannBench | **80.0%** | 76.0% | 72.0% | 69.6% | 65.6% |
| Terminal-Bench-Science 0.1 (AA) | **64.8%** | 52.9% | 63.3% | 59.0% | 43.3% |
| CWE-bench | **71.0%** | 68.0% | 68.0% | 67.0% | 58.0% |
| ExploitGym | **54.8%** | 42.0% | 42.4% | 34.5% | n/a |
| SEC-bench Pro | **88.2%** | 83.4% | 85.4% | n/a | n/a |

The table reproduces Google's launch graphic as transcribed by Apiyi, cross-checked against VentureBeat, OfficeChai, and DataCamp for the rows those outlets report. n/a means Google reported no score for that model. Argon's figures here differ from its own September launch table on several rows, for example 52.9% rather than 57.6% on Terminal-Bench-Science, because this table uses Artificial Analysis's runs for the two Terminal-Bench rows and a newer Vals Index version.

Google's methodology page, as summarized by Apiyi, limits how far the table can be read. Carbon and Argon scores are pass@1 at the highest thinking setting in the Gemini API, with no majority voting or parallel test-time compute. Unless otherwise noted, scores for non-Gemini models are the numbers their developers published, so the table was not produced by one evaluator running every model in the same environment. OSWorld-2.1 uses an offline subset of 82 tasks, GraphWalks uses 200 questions from the 256K to 1M range, SEC-bench Pro averages three attempts per question, and ExploitGym uses a six-hour timeout per question with mitigations disabled. The Artificial Analysis rows come from Artificial Analysis, FrontierSWE from Proximal's leaderboard, Vending-Bench 2 and Blueprint-Bench 2 from Andon Labs, and CWE-bench from Collinear AI. Some leads are narrow: 0.1 points over Astra on SWE-sweep, 0.2 over Opus 5.5 on the Vals Index, and one point on the headline index.

Artificial Analysis, an independent evaluator, rebuilt its Intelligence Index as version 5.0 on October 27, with roughly 60 percent of it drawn from private tests, so scores cannot be compared with its earlier versions.

| Artificial Analysis Intelligence Index v5.0 | Score | Cost per index task (list price) |
| --- | --- | --- |
| Gemini 4 Carbon (High) | **59** | $7.76 |
| Claude Opus 5.5 | 58 | $10.04 |
| GPT-6 Astra | 57 | $6.25 |
| GPT-6.1 Sol | 55 | n/a |
| Claude Sonnet 5.5 | 54 | n/a |
| GPT-6.1 Terra | 51 | n/a |
| Gemini 4 Argon | 50 | $7.68 |
| Claude Fable 5.1 | 49 | n/a |

Carbon takes first place in four of the index's nine evaluations: HLE-Diamond, Terminal-Bench-Science, AA-Omniscience, and AutomationBench-AA. It is not first in coding or in knowledge work. On Artificial Analysis's Coding Agent Index it scores 59 using Antigravity CLI, behind Claude Code with Opus 5.5 at 62 and above Argon's 53. Artificial Analysis's own Terminal-Bench 4.0 reading for Carbon is 64%, slightly below Google's 65.2%, and on FrontierCode v1.1 it scores 51% against 56% for Opus 5.5. On AA-Briefcase, a knowledge-work test, Carbon scores 1,609 Elo, up 126 points on Argon, but behind Grok 4.7, Muse Wonder, and Kimi K3.1 Pro. Vals AI places Carbon about half a point behind Claude Sonnet 5.5 on its index.

Factual reliability is mixed. On AA-Omniscience, Carbon scores 54, the highest recorded, and answers 58 percent of questions correctly against Argon's 48 percent. Its hallucination rate rises from 4 percent to 11 percent, which is still the second lowest of any model Artificial Analysis tracks, behind only Argon. Carbon also rises 89 points to fourth place, at 1,768, on Arena's WebDev leaderboard. Andon Labs reports that Carbon takes first place on Vending-Bench 2, the first time a Google model has led it in about a year, and ties the human baseline for 3D spatial understanding on Blueprint-Bench 2.

## Pricing and efficiency

Carbon launches at an introductory price of USD 2 per million input tokens and USD 10 per million output tokens, with cached input 95 percent below the input rate, which is about USD 0.10. After the introductory period, the price moves to USD 4 and USD 20, with cached input at about USD 0.20. Google has not announced when Carbon's introductory period ends, while Argon's ends on November 30, 2026.

| Price per 1M tokens | Carbon (intro) | Carbon (regular) | Claude Opus 5.5 | GPT-6 Astra | GPT-6.1 Sol |
| --- | --- | --- | --- | --- | --- |
| Input | $2 | $4 | $4 | $10 | $2 |
| Cached input | $0.10 | $0.20 | $0.20 | $1 | $0.10 |
| Output | $10 | $20 | $20 | $50 | $10 |
| Cost per index task | $3.88 | $7.76 | $10.04 | $6.25 | n/a |

Artificial Analysis calculates cost per task from list prices and prices the introductory rate at $3.88. At the regular rate Carbon costs about 23 percent less per index task than Opus 5.5 and about 1.2 times as much as GPT-6 Astra, even though Astra's token prices are far higher, because Carbon writes more. Carbon needs about 123,000 output tokens per task, 29 percent fewer than Opus 5.5's 173,000, but Artificial Analysis calls it verbose against the field, at 200 million output tokens to run the whole index against a median of 64 million, and ranks it 49th of 51 models for verbosity. Budgeting on the introductory price is optimistic, because the price doubles once the introductory period ends.

## Safeguards

Google says Carbon uses the same multi-layered stack as Gemini 4 Argon, combining model training, real-time misuse defenses, prompt-injection resilience, and chain-of-thought monitoring, and points to Argon's Safety and Security report for the model's alignment propensities. For Carbon, Google reports further reductions in problem behavior. Successful cheating on Terminal-Bench 4.0 fell to 0.0 percent and attempts to cheat to 0.4 percent while accuracy improved, and in live coding environments with Google's internal teams, high-severity misalignment alerts dropped by almost 80 percent. Google says unsupported claims are roughly halved against Argon, and that Carbon reduces overreach by 26 percent and under-service by 29 percent. It also says Carbon retrieves sources and surfaces discrepancies to the user before acting under ambiguity. On Gray Swan's indirect prompt-injection benchmark, the attack success rate is 0.0 percent, against 0.7 percent for Argon.

The cyber results are the reason for the staged rollout. Google reports Carbon at 88.2% on SEC-bench Pro for finding vulnerabilities, 54.8% on ExploitGym for exploitation, and 71.0% on CWE-bench for patching, with more than double Argon's success rate at generating secure code in internal benchmarks. The Chrome team reports Carbon produced high-quality patches more than ten times better than another lab's large frontier model, which Google does not name, and Wiz reports Carbon leading its web penetration testing benchmarks. These security benchmarks were run without guardrails, and none of Google's headline security figures or its multi-day-run claims have been reproduced outside Google.

Fairwind, per Apiyi's summary of its terms, has more than 650 partners, including national cyber authorities and critical infrastructure operators in healthcare, energy, telecommunications, and finance. Participants apply through their Google Cloud customer team, are vetted on security history, need phishing-resistant multi-factor authentication, must log employee access, and may not resell access. Google's internal unsupported-claims result and Artificial Analysis's higher hallucination rate come from different tests and are not directly comparable, but together they leave open how much of Carbon's honesty improvement will show up in other deployments.

## API and behavior changes

No public API details exist for Carbon. Google's announcement gave no model ID, and Gemini API model lists checked on launch day showed `gemini-4-argon` but no Carbon entry. Argon is available now on the Gemini API and Vertex AI, and Google has not said whether Carbon will follow the same path. The one concrete change is pricing and access policy: Carbon matches Argon's $2/$10 introductory and $4/$20 regular rates, but unlike Argon at its general release, it starts behind a vetted program with a no-resale rule. Teams planning around Carbon should build and test on Argon, budget at the regular price, and wait for official documentation before assuming limits, reasoning settings, or tool support.

## Applications and workflow fit

Carbon is best suited to long-horizon engineering and bug root-causing in large codebases and low-level systems, defensive security work across vulnerability detection, validation, and patching, long-context retrieval and reasoning, where its GraphWalks lead is the widest in the table, and multimodal and spatial tasks. Google's own teams report multi-day runs, and Wiz and the Chrome team report gains in penetration testing and patch quality, though those are partner and internal claims.

It is not the obvious choice for computer-use agents, where Opus 5.5 leads OSWorld-2.1 by about six points, or for terminal-style agentic coding, where Claude Code with Opus 5.5 leads Artificial Analysis's Coding Agent Index by three points and Astra and Opus 5.5 both beat Carbon on FrontierSWE 2. It is also a weaker fit for cost-sensitive loops, since Astra is cheaper per task at list price, and for work where a model should decline rather than guess, since its hallucination rate is higher than Argon's. Because access is limited, most teams cannot use it at all yet. As with other agent models, evaluating the full model-plus-[agent harness](https://www.thequery.in/glossary/agent-harness) workflow matters more than any single [benchmark](https://www.thequery.in/glossary/benchmark) row.

## Bottom line

Gemini 4 Carbon puts Google back at the top of the independent leaderboard two weeks after Argon reached customers: first on Artificial Analysis's rebuilt index at 59, ahead of every row of Google's own comparison but two, and strongest on long-context, cyber, and sustained engineering tasks. It also has limits. Its lead on the index is one point, it trails Opus 5.5 on computer use and agentic coding, it is verbose and priced to double, and its hallucination rate is higher than Argon's. The headline security numbers and the multi-day-run claims are Google's own, and the model has no public API, model ID, or release date, only a vetted group of cyber defenders.

## References & Resources

- [Gemini 4 Carbon: our next frontier model](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-carbon/) (Google)
- [Google unveils Gemini 4 Carbon](https://venturebeat.com/technology/google-unveils-gemini-4-carbon-retaking-the-worlds-no-1-spot-on-updated-harder-benchmarks-but-access-is-limited) (VentureBeat)
- [Gemini 4 Carbon: Google's Newest World Model](https://www.datacamp.com/blog/gemini-4-carbon) (DataCamp)
- [Gemini 4 Carbon Release Analysis](https://help.apiyi.com/en/gemini-4-carbon-release-analysis-en.html) (Apiyi)
- [Gemini 4 Carbon Ranks No. 1 on Artificial Analysis Index, Ahead of Claude Opus 5.5](https://www.trendingtopics.eu/gemini-4-carbon-dethrones-claude-opus-5-5/) (Trending Topics)

## Related Terms

[Claude Opus 5.5](https://www.thequery.in/glossary/claude-opus-5-5) [Claude Fable 5](https://www.thequery.in/glossary/claude-fable-5) [Anthropic](https://www.thequery.in/glossary/anthropic) [Large Language Model](https://www.thequery.in/glossary/large-language-model) [AI Agent](https://www.thequery.in/glossary/ai-agent) [Agentic AI](https://www.thequery.in/glossary/agentic-ai) [Agent Harness](https://www.thequery.in/glossary/agent-harness) [Context Window](https://www.thequery.in/glossary/context-window) [Benchmark](https://www.thequery.in/glossary/benchmark) [API](https://www.thequery.in/glossary/api)

Last updated: October 30, 2026
    $gemini_carbon_model_card$::text AS body,
    'Google DeepMind''s Oct 2026 frontier model, ranked first on the Artificial Analysis Intelligence Index v5.0 at 59, built for cyber defense and multi-day engineering work, with introductory pricing of $2/$10 and access limited to Fairwind cyber defenders.'::text AS summary
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:gemini-4-carbon', 'glossary', 'gemini-4-carbon', '', 'glossary/gemini-4-carbon',
  'Gemini 4 Carbon', summary, body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', body)),
  '[{"title":"Gemini 4 Carbon: our next frontier model","url":"https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-carbon/"},{"title":"Google unveils Gemini 4 Carbon","url":"https://venturebeat.com/technology/google-unveils-gemini-4-carbon-retaking-the-worlds-no-1-spot-on-updated-harder-benchmarks-but-access-is-limited"},{"title":"Gemini 4 Carbon: Google''s Newest World Model","url":"https://www.datacamp.com/blog/gemini-4-carbon"},{"title":"Gemini 4 Carbon Release Analysis","url":"https://help.apiyi.com/en/gemini-4-carbon-release-analysis-en.html"},{"title":"Gemini 4 Carbon Ranks No. 1 on Artificial Analysis Index, Ahead of Claude Opus 5.5","url":"https://www.trendingtopics.eu/gemini-4-carbon-dethrones-claude-opus-5-5/"}]'::jsonb,
  '{"category":"Models & Architectures","relatedTerms":["claude-opus-5-5","claude-fable-5","anthropic","large-language-model","ai-agent","agentic-ai","agent-harness","context-window","benchmark","api"],"seoDescription":"Gemini 4 Carbon model card: Google DeepMind''s October 2026 model, benchmark results, $2/$10 introductory pricing, cyber safeguards, availability, and independent evaluations.","seoKeywords":["Gemini 4 Carbon","Gemini 4 Carbon benchmarks","Gemini 4 Carbon pricing","Gemini 4 Carbon vs Claude Opus 5.5","Gemini 4 Carbon Artificial Analysis Index","Gemini 4 Carbon Fairwind","gemini-4-carbon","Google DeepMind Gemini 4 Carbon","Gemini 4 Carbon API availability","Gemini 4 Carbon cyber security benchmarks"]}'::jsonb,
  NULL, NULL,
  'published', DATE '2026-10-30', 0
FROM gemini_4_carbon_entry
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
