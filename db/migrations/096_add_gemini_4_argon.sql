-- Add Gemini 4 Argon to the glossary.
--
-- Source: Google's September 30, 2026 launch post for Gemini 4 Argon, the
-- accompanying DeepMind model evaluation document, VentureBeat's coverage of
-- the launch table, and the Artificial Analysis, Vals AI, Arena, Gray Swan, and
-- Wiz figures those launch-day writeups report. Following the pattern set by
-- 090 (GPT-6.1 Sol) and 093 (MiMo V2.6 Pro).
--
-- Footnote handling, all carried in the body so the rendered page keeps the
-- caveats:
-- 1. Google's launch table is an image. Every n/a cell here means the value
--    could not be read in the coverage available, not that Google left the
--    cell blank, and the body says so.
-- 2. Google's own evaluation document states that Argon's scores are pass@1 at
--    its highest thinking setting, while the rival columns are the providers'
--    self-reported numbers or public leaderboards at maximum reasoning, falling
--    back to the best available. Several Argon results were also run by Google
--    with its own harness choices: mini-swe-agent for DeepSWE, a 6x longer
--    verifier timeout for Terminal-Bench Science, and 1 frame per second for
--    LVBench against 300 to 800 frames for rivals, because of API limits.
-- 3. The Vals Index, Vals Finance Agent, and Harvey figures come from Vals AI,
--    AutomationBench from Zapier's public leaderboard, and FrontierSWE from
--    Proximal's, per Google's document.
-- 4. The Astra and GPT-6.1 Sol columns of the Artificial Analysis table come
--    from OrcaRouter's transcription of Artificial Analysis data, and the Sonnet
--    5.5 figures were run on a pre-release deployment with a bug Artificial
--    Analysis says it will re-run.
-- 5. Artificial Analysis's own text confirms the Argon, Sonnet 5.5, and Opus 5.5
--    index scores and the hallucination rate, as reported by The Decoder.
-- 6. Google's introductory price has no announced end date, and the regular
--    cached rate is implied by the stated 95 percent discount rather than
--    published directly.
-- 7. No model card, knowledge cutoff, API model ID, or quota documentation
--    existed at launch. The body records that absence rather than filling it.
--
-- "gemini-3-8-flash", "claude-sonnet-5-5", "gpt-6-astra", and "gpt-6-1-sol" are
-- used in relatedTerms only. Those rows were renamed or added directly in the
-- database by migrations 072 and later rather than in data/glossary.json, so
-- the seed JSON and the database are expected to differ here.

WITH gemini_4_argon_entry AS (
  SELECT $body$
Gemini 4 Argon is Google's DeepMind September 2026 frontier [large language model](/glossary/large-language-model) and the first model of the Gemini 4 generation. Announced on September 30, 2026, it is built for long-horizon software engineering, enterprise knowledge work such as legal and finance, and cybersecurity defense. It is Google's first frontier model since Gemini 3.1 Pro, after the planned Gemini 3.5 Pro was cancelled and Google shifted its focus to Gemini 3.8 Flash.

Argon is not generally available yet. It is rolling out to trusted cyber defenders through Google's Fairwind Program and to Google's internal teams, both without cyber guardrails, while Google takes part in the US government's voluntary pre-release access process. Google says broader availability for developers, enterprises, and consumers will come as soon as possible, starting with paid API customers and Google AI Ultra subscribers, and has given no date. For a normal user that means waiting. For a developer it means there is no public API model ID yet.

## Core profile

Google's launch post states an output limit of 1 million tokens, up from 64,000 on earlier Gemini models, and does not state a separate input limit. Artificial Analysis and Vals AI both report a 1M-token input [context window](/glossary/context-window), and Vals ran its tests with output capped at 262,000 tokens. Argon accepts text, images, video, and audio as input and produces text only. Google reports its own results at the highest thinking setting, which Artificial Analysis calls High, the usual top tier for Gemini models.

Google is adding a Long Decode Continuation feature to the Gemini API. It pauses a long response and resumes it through follow-up requests so that extended reasoning does not hit a timeout. The model's knowledge cutoff and API model ID were not published at the time of writing, and no full model card existed beyond Google's blog post and its model evaluation document.

Argon is designed for complex work that has to hold together over many steps: coding and large codebase migrations, finance, legal, and tax research, business process automation, long-video and chart analysis, defensive security, and writing. Google says thousands of its own employees already use it for specialized coding, deeper research, and writing quality.

## Benchmark profile

Google's launch table compares Argon with GPT-6 Astra, Claude Fable 5.1, and Claude Opus 5.5 across 18 benchmarks. By VentureBeat's count of that table, Argon leads outright on 12 and ties for first on one, GPT-6 Astra leads outright on three and ties on one, and Opus 5.5 leads outright on two. Google does not claim a clean sweep. [Claude Sonnet 5.5](/glossary/claude-sonnet-5-5) and [GPT-6.1 Sol](/glossary/gpt-6-1-sol), released days earlier, are not in Google's comparison set.

| Benchmark (Google-reported) | Gemini 4 Argon | GPT-6 Astra | Claude Opus 5.5 | Claude Fable 5.1 |
| --- | ---: | ---: | ---: | ---: |
| Vals Index | **68.9%** | 63.1% | 67.0% | 65.8% |
| Vals Finance Agent v2 | **65.4%** | 53.5% | 58.6% | 58.9% |
| Harvey Legal Agent Benchmark | **19.6%** | 5.4% | 3.8% | n/a |
| AutomationBench (private set) | **51.3%** | 41.4% | 42.5% | n/a |
| DeepSWE v1.1 | **77.9%** | 74.1% | 74.2% | 67.4% |
| FrontierSWE v2 | 55.0% | **65.5%** | n/a | n/a |
| Terminal-Bench 4.0 | 57.4% | n/a | **66.4%** | n/a |
| Terminal-Bench Science 0.1 | 57.6% | **68.1%** | n/a | n/a |
| PostTrainBench v1.1 | 45.3% | n/a | **49.3%** | n/a |
| GraphWalks, 256K to 1M tokens | **84.2%** | 71.8% | 66.8% | n/a |
| LVBench (long video) | **91.7%** | 87.5% | 83.7% | n/a |
| CWE-bench v1 (vulnerability remediation) | 68% (tie) | 68% (tie) | 67% | n/a |
| Gray Swan IPI attack success (lower is better) | **0.7%** | 8.5% | 1.0% | 1.0% |

Google's table is an image, so n/a here means a value that could not be read in the coverage available, not that Google left the cell empty. Two of the gaps are on the table's largest deficits. Astra leads Argon by 10.5 points on both FrontierSWE v2 and Terminal-Bench Science 0.1, Opus 5.5 leads on Terminal-Bench 4.0 by 9 points, and Opus 5.5 also leads PostTrainBench.

The methodology matters as much as the numbers. Google's evaluation document says all Argon scores are pass@1 at the highest thinking setting, and that rival results are the providers' self-reported numbers or official public leaderboards at maximum reasoning settings, falling back to the best available. Google computed Argon's DeepSWE, Terminal-Bench 4.0, Terminal-Bench Science, PostTrainBench, GraphWalks, LVBench, and OSWorld 2.0 results itself, in some cases with its own harness choices: DeepSWE used a mini-swe-agent harness, Terminal-Bench Science used a 6x longer verifier timeout, and LVBench sampled video at 1 frame per second for Gemini against 300 to 800 frames for the other models because of API limits. The Vals Index, Finance Agent, and Harvey results come from Vals AI, AutomationBench from Zapier's public leaderboard, and FrontierSWE from Proximal's.

Independent runs published on launch day are more mixed. Artificial Analysis scores Argon at 53 on its Intelligence Index at High effort, level with GPT-6 Astra and Claude Fable 5.1, one point ahead of GPT-6.1 Sol, and behind [Claude Sonnet 5.5](/glossary/claude-sonnet-5-5) at 56 and [Claude Opus 5.5](/glossary/claude-opus-5-5) at 58. That is a 23-point jump over Gemini 3.1 Pro Preview.

| Artificial Analysis | Gemini 4 Argon (High) | GPT-6 Astra | GPT-6.1 Sol | Sonnet 5.5 | Opus 5.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intelligence Index | 53 | 52.7 | 51.8 | 56 | **58** |
| Cost per index task | $1.99 intro, $3.98 regular | $3.26 | **$0.72** | $7.60 | n/a |
| AutomationBench-AA | **77.5%** | 68.5% | 64.9% | 71.3% | 69.5% |
| Terminal-Bench 4.0 | 57% | 59.1% | 56.1% | **64%** | 60% |
| AA-Omniscience hallucination rate (lower is better) | **15%** | 51.3% | 54.3% | 47% | 59% |
| AA-Omniscience accuracy | 50% | 63% | 62.1% | 54% | **66%** |

The Argon figures and the Sonnet 5.5 and Opus 5.5 index scores are from Artificial Analysis as reported by The Decoder. The Astra and GPT-6.1 Sol columns come from OrcaRouter's transcription of Artificial Analysis data in earlier coverage, and Artificial Analysis ran Sonnet 5.5 on a pre-release deployment with a bug it plans to re-run.

Two results stand out. Argon takes first place on Artificial Analysis's agentic AutomationBench-AA, about six points ahead of Sonnet 5.5, and shows the lowest hallucination rate Artificial Analysis has measured among leading models, because it declines to answer more often than it guesses. The trade-off is accuracy: at 50 percent it sits 13 points below Astra and 5 below Gemini 3.1 Pro Preview.

Vals AI ranks Argon first on its Vals Index at 68.9 percent, the first time a Gemini model has topped it, and says Argon fully completed about seven times as many Harvey legal tasks as Sonnet 5.5. AlphaSignal's summary of Vals data adds that Argon used about a quarter of Sonnet 5.5's output tokens per task there. Arena ranks Argon (High) first on its Text Arena at 1,525 points, 20 ahead of [Claude Opus 4.6](/glossary/claude-opus-4-6) (High), and eighth on Code Arena WebDev at 1,679.

## Pricing and efficiency

Google set an introductory price of USD 2 per million input tokens and USD 10 per million output tokens, with cached input 95 percent below the input rate, or about USD 0.10. After the introductory period, which has no announced end date, the price moves to USD 4 and USD 20. Google does not publish cache-write prices, and the regular cached rate of USD 0.20 is implied by the 95 percent discount.

| Price per 1M tokens | Argon (intro) | Argon (regular) | GPT-6 Astra | GPT-6.1 Sol | Opus 5.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Input | $2 | $4 | $10 | $2 | $4 |
| Cached input | $0.10 | $0.20 | $1 | $0.10 | $0.20 |
| Output | $10 | $20 | $50 | $10 | $20 |

At the introductory rate Argon costs one-fifth of Astra and half of Opus 5.5 per token. After it, Argon matches Opus 5.5's list price and stays below Astra. [Claude Sonnet 5.5](/glossary/claude-sonnet-5-5) matches the introductory input and output rates at $2 and $10, with cached input at $0.20.

List price is not cost per task. Artificial Analysis puts Argon at USD 1.99 per index task at the introductory price, about 60 percent of Astra's cost and roughly 2.7 times GPT-6.1 Sol's. At the regular price it rises to USD 3.98, about 22 percent above Astra. The reason is token use: Argon averages 62,000 output tokens per task against 27,000 for Astra, so the price advantage comes from lower token rates, not from efficiency. Vals's runs, summarized by AlphaSignal, show the opposite pattern against Sonnet 5.5, so the answer depends on the workload. A full 1M-token output costs USD 10 at the introductory rate and USD 20 after.

## Safeguards

Google describes four areas it is strengthening before broad availability. For misuse, the model is designed to refuse harmful cyber and chemical, biological, radiological, and nuclear requests while preserving legitimate dual-use research, under Google's Frontier Safety Framework. Google is improving monitoring of the model's internal activations and says its safeguards were tested by internal and external red teams.

For [prompt injection](/glossary/prompt-injection), Google calls Argon its most resilient model yet. On Gray Swan's indirect prompt-injection benchmark the attack success rate is 0.7 percent, against 1.0 percent for Opus 5.5 and Fable 5.1, 8.5 percent for GPT-6 Astra, 27.0 percent for GPT-6 Sol, 31.5 percent for GLM 5.3, and 51.8 percent for Grok 4.8. For misalignment, Google deploys monitors that watch Argon's [chain of thought](/glossary/chain-of-thought) and actions and stop execution when necessary, using a similar system to watch its training runs while avoiding feeding the findings back into training. Google also urges the rest of the industry to preserve reasoning transparency. For agent environments, it is isolating and sealing sandboxes before high-risk training or evaluation begins.

The cyber capabilities are the point of the staged rollout. Google says Argon can autonomously find, validate, and patch critical software vulnerabilities, ties for first on CWE-bench v1 at 68 percent, and outperforms Gemini 3.8 Flash Cyber on Google's internal vulnerability benchmark and on Wiz's black-box penetration testing benchmark. Wiz, using Argon through its Scan for Good initiative, reports a critical vulnerability exposing sensitive personal information in healthcare software used by hospitals worldwide. Bloomberg reported ahead of launch that some Google staff worried Argon is not as strong as offerings from Anthropic or OpenAI.

## API and behavior changes

Almost nothing about API behavior is documented yet. Reviewers checking the public Gemini API catalog on launch day found no Argon entry, no model ID, and no quotas or region details. What is known is the 1M output limit, the Long Decode Continuation feature for resuming long generations, the 95 percent cached-input discount, and the text, image, video, and audio input support with text-only output. Teams planning around Argon should wait for official documentation rather than build against assumed parameters or a guessed endpoint.

## Applications and workflow fit

Google's examples are internal. Argon helped quantum computing researchers cut the spacetime resources, measured as qubits times gates, of a bottleneck subroutine, beating a published baseline by 40 percent in minutes. A team of Argon agents analyzed fleet-wide profiling data and found memory optimizations that Google says will free more than 300 TiB once rolled out, with estimated total savings of 500 TiB to 1 PiB. Argon agents are migrating C and C++ code to Rust at scales from tens of thousands of lines in libraries such as re2 and libgav1 to more than 800,000 lines in the Fuchsia Zircon kernel, with automated and manual audits before production. In libgav1, they replaced 32,000 lines of SIMD code and produced a memory-safe decoder that runs 2.7 times faster than the earlier Rust port with identical video output.

Argon looks best suited to long-horizon coding and migrations, finance, legal, and tax agents, business process automation, long-video and chart analysis, defensive security work, and long-form writing, where Arena ranks it first. It is a weaker fit where the evidence points elsewhere: terminal-style agents and ML engineering tasks, where Opus 5.5 leads Google's own table, science-terminal and FrontierSWE tasks, where Astra leads, factual recall, where accuracy is 50 percent, and web development, where it ranks eighth on Code Arena. Cost-sensitive agent loops should test token use first. The Decoder also notes that Google's surrounding software, the Gemini app, still lags Claude Cowork and ChatGPT Work. Teams should evaluate the full model-plus-[agent harness](/glossary/agent-harness) workflow rather than choose from a [benchmark](/glossary/benchmark) table alone.

## Bottom line

Gemini 4 Argon puts Google back in the frontier conversation after seven months without a flagship. It is the first Gemini model to top the Vals Index, it leads most rows of Google's own table, it posts the strongest agentic score and lowest hallucination rate on Artificial Analysis's tests, and it lists at half of Opus 5.5's price during the introductory period. It also carries limits. Independent testing puts it level with Astra and behind Sonnet 5.5 and Opus 5.5, its per-task cost rises with heavy token use, and many headline scores are Google-run or vendor-reported with different harnesses. The model has no public model ID, knowledge cutoff, or release date, and the first people able to use it are cyber defenders.
$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:gemini-4-argon',
  'glossary',
  'gemini-4-argon',
  '',
  'glossary/gemini-4-argon',
  'Gemini 4 Argon',
  'Google DeepMind''s Sep 2026 frontier model for long-horizon coding, enterprise knowledge work, and cyber defense, with a 1M-token output limit and introductory pricing of $2/$10, rolling out first to trusted cyber defenders.',
  gemini_4_argon_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', gemini_4_argon_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Gemini 4 Argon: our next era of frontier intelligence', 'url', 'https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-4-argon/'),
    jsonb_build_object('title', 'Gemini 4 Argon model evaluation', 'url', 'https://storage.googleapis.com/deepmind-media/gemini/gemini_4_argon_model_evaluation.pdf'),
    jsonb_build_object('title', 'Google unveils Gemini 4 Argon', 'url', 'https://venturebeat.com/technology/google-unveils-gemini-4-argon-retaking-benchmark-lead-over-openai-and-anthropic-but-in-limited-release')
  ),
  jsonb_build_object(
    'category', 'Models & Architectures',
    'relatedTerms', jsonb_build_array('gemini', 'google-deepmind', 'claude-opus-5-5', 'claude-sonnet-5-5', 'gpt-6-astra'),
    'seoDescription', 'Gemini 4 Argon explained: Google DeepMind''s Sep 2026 frontier model, 1M-token output limit, benchmark tables vs GPT-6 Astra and Opus 5.5, $2/$10 introductory pricing, safeguards, and availability.',
    'seoKeywords', jsonb_build_array('Gemini 4 Argon', 'Gemini 4 Argon benchmarks', 'Gemini 4 Argon pricing', 'Gemini 4 Argon vs GPT-6 Astra', 'Gemini 4 Argon vs Claude Opus 5.5', 'Gemini 4 Argon context window', 'Gemini 4 Argon API', 'gemini-4-argon', 'Google Gemini 4 Argon', 'Gemini 4 Argon Fairwind Program', 'Gemini 4 Argon CWE-bench')
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-10-01',
  0
FROM gemini_4_argon_entry
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
    WHEN COALESCE(metadata->'relatedTerms', '[]'::jsonb) @> '["gemini-4-argon"]'::jsonb
      THEN metadata
    ELSE jsonb_set(
      COALESCE(metadata, '{}'::jsonb),
      '{relatedTerms}',
      COALESCE(metadata->'relatedTerms', '[]'::jsonb) || jsonb_build_array('gemini-4-argon'),
      true
    )
  END,
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug IN ('gemini', 'gemini-3-1-pro', 'gemini-3-8-flash', 'google-deepmind')
  AND parent_slug = '';