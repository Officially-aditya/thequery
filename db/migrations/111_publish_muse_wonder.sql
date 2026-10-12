-- Publish the user-supplied Muse Wonder family card with primary-source corrections.
-- Wonder is an API preview; Full is subscription-only and has no token price.
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order, updated_at
)
SELECT entry.id, 'glossary', entry.slug, '', 'glossary/' || entry.slug,
  entry.title, entry.summary, entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', entry.body)),
  entry.sources, entry.metadata, NULL, NULL, 'published', DATE '2026-10-13', 0,
  TIMESTAMPTZ '2026-10-13 00:00:00+00'
FROM jsonb_to_record($glossary$
{
  "id": "glossary:muse-wonder",
  "slug": "muse-wonder",
  "title": "Muse Wonder",
  "summary": "Meta's October 2026 multimodal frontier model pair: API-preview Muse Wonder at $4/$20 per million tokens, and the stronger Wonder Full on Muse Code subscriptions only.",
  "body": "Muse Wonder and Muse Wonder Full are Meta Superintelligence Labs' frontier-class [large language models](/glossary/large-language-model), released on October 12, 2026 by Meta Superintelligence Labs. Meta describes them as its most capable models to date, built for personal and professional agentic tasks such as working through an insurance claim, making sense of a lease, or building a working app, with added strength in law, finance, health, and coding. They follow the lightweight Muse Spark line, whose latest version, Muse Spark 1.3, they replace as Meta's top model, and they were trained at Meta's gigawatt-scale Prometheus data center. Both are proprietary, closed-weight models. Meta's developer documentation labels the API model a preview.\n\nThe two models are not equally available. Muse Wonder (`muse-wonder-1.0`) is on the Meta Model API and in Muse Code, Meta's software engineering agent, and Meta says it will soon power experiences in the Muse personal agent app, starting with subscribers. Muse Wonder Full (`muse-wonder-full-1.0`) is the higher-scoring variant and is available only to Muse Code subscribers on the Power Usage and Maximum Usage plans. It has no API access and no per-token price. Meta's documentation indicates that setting the API model to its maximum effort does not reach Full, so the two are separate models rather than one model at different settings.\n\n## Core profile\n\n| | Muse Wonder | Muse Wonder Full |\n| --- | --- | --- |\n| Model ID | `muse-wonder-1.0` | `muse-wonder-full-1.0` |\n| Access | Meta Model API preview, OpenRouter, Muse Code (pay-as-you-go and all four plans), Muse app soon | Muse Code, Power Usage ($50 per month) and Maximum Usage ($200 per month) plans only |\n| Price | $4 input, $0.20 cached input, $20 output per 1M tokens | No per-token price |\n| Context | 1,048,576 tokens | 1M tokens |\n| Weights | Closed | Closed |\n\nMeta's model card lists a one-million-token [context window](/glossary/context-window) at inference, input of text, images, video, and audio, and text-only output, with multilingual training data. The Meta Model API also accepts PDFs. The model card does not state a maximum output length, and the API pricing carries no long-context premium. On the API, reasoning effort accepts `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`. The `none` setting is not supported and returns an error, and if the parameter is omitted the model chooses a level itself.\n\nThe API follows OpenAI-style conventions. It is served at `https://api.meta.ai/v1` and accepts OpenAI Responses and Chat Completions requests as well as the Anthropic Messages format. It supports tool calling, structured output, search grounding, and computer use, and rate limits are 3,000 requests and 4 million tokens per minute per team. Contributor mode is unavailable for Wonder on the pay-as-you-go API. Muse Code subscriptions support both Standard and Contributor modes for Wonder and Full; Contributor mode is enabled by default and can be disabled in settings. Meta says Standard-mode content is not used to train its models.\n\nMeta ties the models to two trained behaviors rather than switches. Active Memory is the ability to carry important context about a user across conversations and retrieve the details relevant to the current task. Active Focus is sustained effort on a hard goal: trying alternative approaches, working through blockers, and asking for clarification or checking in before taking important actions. Meta also argues that the larger base models make post-training gains carry over better to unfamiliar tasks, and published a chart comparing Muse Wonder Full with the Muse Spark 1.3 trend line to support that claim.\n\nMeta's model card lists six intended uses: general-purpose assistance and reasoning, coding and software engineering, personal and professional agents, domain-specific knowledge work in areas such as law, finance, and health, multimodal understanding, and research and development support.\n\n## Benchmark profile\n\nMeta compares both models with GPT-6 Astra, Claude Fable 5.1, and Claude Opus 5.5, all at max reasoning effort. The table below is transcribed from page 3 of Meta's official model card. The Intelligence Index row uses the precise Artificial Analysis values rather than Meta's rounded display. Meta's table does not include Claude Sonnet 5.5, GPT-6.1 Sol, or Gemini 4 Argon, all released shortly before Wonder.\n\n| Evaluation (Meta-reported unless noted) | Muse Wonder Full | Muse Wonder | GPT-6 Astra | Claude Opus 5.5 | Claude Fable 5.1 |\n| --- | --- | --- | --- | --- | --- |\n| Personal Superintelligence Index (internal) | **59.1** | 54.3 | 54.5 | 55.2 | 49.9 |\n| Harvey LAB-AA v1.1 | **13.1%** | 10.0% | 8.6% | 4.2% | 6.4% |\n| Vals Finance Agent V2 | **59.1%** | 56.6% | 53.5% | 58.6% | 58.9% |\n| HealthBench Pro | 62.3% | 60.5% | 65.3% | **65.6%** | 62.1% |\n| AutomationBench-AA (strict pass@1) | **45.4%** | 39.1% | 41.6% | 41.2% | 32.1% |\n| GDP.pdf (Surge metric) | 33.2% | 30.8% | **34.2%** | 30.6% | 27.6% |\n| JobBench | 75.3% | 70.5% | 66.1% | **75.4%** | 73.1% |\n| OSWorld 2.1 (partial reward) | 78.4% | 72.6% | 75.8% | **81.8%** | 80.7% |\n| MemoryBench (internal) | 85.5% | **85.6%** | 81.4% | 81.5% | 79.0% |\n| DeepSWE v1.1 | **79.0%** | 78.8% | 74.1% | 74.2% | 67.4% |\n| SWE-Marathon | **57.5%** (xhigh) | 53.1% | 42.5% | 46.3% | 45.6% |\n| Terminal-Bench 4.0 | 57.1% | 52.0% | 59.1% | **59.6%** | 52.0% |\n| ProgramBench | 59.0% | 50.0% | 52.5% | **70.5%** | 55.5% |\n| AA Intelligence Index (direct AA values) | 55.8 | 52.4 | 52.7 | **57.6** | 53.4 |\n\nMeta reports strong results on memory, long-horizon coding and agentic business work. Opus 5.5 leads this comparison on ProgramBench, Terminal-Bench 4.0, OSWorld partial reward, HealthBench Pro and JobBench; Astra leads the Surge GDP.pdf metric. OSWorld's partial-reward figures are not strict completion rates: Meta reports 39.4% binary completion for Full and 34.3% for Wonder.\n\nThe methodology limits how far the table can be read. Meta says its own runs use max effort, with SWE-Marathon for Full at xhigh, and that SWE-Marathon rivals ran inside their own agents (Codex and Claude Code), so that row is not a controlled comparison. Where a third-party or provider-reported result exists Meta uses it, and otherwise it uses the highest of its own validated run, a self-reported score, or an official leaderboard score. It notes that competitor runs are best-effort and may not be optimally tuned. Refusals receive zero credit. Three of the headline metrics, the Personal Superintelligence Index, MemoryBench, and FocusBench, are Meta's own internal benchmarks rather than independently maintained leaderboards in the sources reviewed here. Most of the largest margins also belong to Full, which cannot be called through the API.\n\nArtificial Analysis independently evaluated both models. The following values come directly from its public model-page data reviewed on October 13, 2026, including the rival values from the same snapshot.\n\n| Artificial Analysis (max effort) | Muse Wonder Full | Muse Wonder | Claude Opus 5.5 | Claude Sonnet 5.5 | GPT-6 Astra | GPT-6.1 Sol |\n| --- | --- | --- | --- | --- | --- | --- |\n| Intelligence Index v4.3.2 | 55.8 | 52.4 | 57.6 | 56.0 | 52.7 | 51.8 |\n| GDPval-AA v2.1 (Elo) | 1857.09 | 1770.34 | 1866.27 | 1837.82 | 1573.90 | 1591.97 |\n| AA-Omniscience hallucination rate (lower is better) | 23.0% | 26.5% | 58.6% | 47.0% | 51.3% | 54.3% |\n| Cost per index task | No API | $4.52 | $5.98 | $5.46 | $3.26 | $0.72 |\n\nAmong the distinct models in this comparison, Full is third on the Intelligence Index, 0.2 points behind Sonnet 5.5. Wonder is 0.3 points below Astra and 0.6 above GPT-6.1 Sol, using rounded scores. It improves on Muse Spark 1.3's 48.1 by 4.3 points. Full's knowledge result remains lower than Opus 5.5's: its AA-Omniscience Index is 30.2 against 46.4, while its hallucination rate is 23.0% against 58.6%. Accuracy, index score and hallucination rate measure different aspects of the answers.\n\n| Additional AA evaluation (max effort) | Muse Wonder | Muse Wonder Full |\n| --- | ---: | ---: |\n| Terminal-Bench 4.0 | 52.0% | 57.1% |\n| SciCode | 60.3% | 62.6% |\n| CritPt | 28.3% | 32.9% |\n| Humanity's Last Exam | 50.1% | 53.6% |\n| AA-Briefcase v1.1 (Elo) | 1658.65 | 1729.32 |\n| GDP.pdf (AA all-pass) | 30.6% | 32.6% |\n| AutomationBench-AA (partial score) | 68.1% | 71.3% |\n| AA-Omniscience accuracy | 41.1% | 43.2% |\n| AA-Omniscience Index (points) | 25.5 | 30.2 |\n| AA-LCR v1.1 | 80.7% | 83.7% |\n\nAutomationBench's partial score awards credit for completed assertions; Meta's launch table above uses strict pass@1, requiring every assertion to pass. The two metrics must not be read as conflicting measurements of the same percentage. Likewise, Meta's GDP.pdf comparison uses Surge's metric, while Artificial Analysis publishes an all-pass score. The catalog uses the AA metric for each of these rows. No Terminal-Bench Science 0.1 result is present for either model in the reviewed AA snapshot.\n\nAt max effort, AA reports Wonder at 83.0 output tokens per second, 19.27 seconds to the first chunk and 43.36 seconds to the first answer token including reasoning. It has no API timing or per-task dollar price for Full. Coding-agent results also depend on the complete agent and its harness; they should be distinguished from the model-level index.\n\n## Pricing and efficiency\n\nMuse Wonder costs USD 4 per million input tokens, USD 0.20 per million cached input tokens, and USD 20 per million output tokens. That is the same rate card as Claude Opus 5.5, including the cache price, and 60 percent below Claude Fable 5.1 and GPT-6 Astra on input and output. Reasoning tokens bill as output, and Meta's built-in `web_search` tool adds USD 2.50 per 1,000 searches.\n\n| Price per 1M tokens | Muse Wonder | Claude Opus 5.5 | Claude Fable 5.1 | GPT-6 Astra | GPT-6.1 Sol |\n| --- | --- | --- | --- | --- | --- |\n| Input | $4 | $4 | $10 | $10 | $2 |\n| Cached input | $0.20 | $0.20 | $0.25 | $1 | $0.10 |\n| Output | $20 | $20 | $50 | $50 | $10 |\n| Cost per index task | $4.52 | $5.98 | $7.63 | $3.26 | $0.72 |\n\nMeta's AI chief Alexandr Wang had said the API would be priced meaningfully below comparable offerings from OpenAI and Anthropic. At list price that holds against Fable 5.1 and Astra but not against Opus 5.5, which matches Wonder exactly, and not against Sonnet 5.5 or GPT-6.1 Sol, which list at half Wonder's price or less.\n\nPer task, Artificial Analysis puts Wonder at USD 4.52, below Opus 5.5 and Fable 5.1, above Astra, nearly three times Muse Spark 1.3 (USD 1.60), and about six times GPT-6.1 Sol, which scores 0.6 points lower on the same index. The reviewed AA data reports about 134.2 million output tokens across Wonder's canonical index run, including about 101.8 million reasoning tokens. Its cost per index task is a benchmark-specific measure, not a fixed price for arbitrary production tasks.\n\nMuse Wonder Full has no per-token price. In Muse Code, Wonder is included in pay-as-you-go use and all four subscription plans, and Full is included in the Power Usage plan at USD 50 per month and the Maximum Usage plan at USD 200 per month, which Meta advertises as 20 times and 80 times the usage of the entry plan. Meta does not promise a fixed number of maximum-effort Full tasks, so a Full subscription cannot be read as unrestricted API access.\n\n## Safeguards\n\nMeta's model card, a 13-page document retrieved for this review, applies the Meta Superintelligence Scaling Framework. Its risk assessment uses Muse Wonder Full as the most capable variant. Without mitigations, Meta does not rule out the model reaching the framework's high-risk threshold in the chemical and biological domain, so it applies safety post-training and system-level classifiers. Unmitigated, it rates the model moderate or lower for cybersecurity and loss of control. With mitigations in place it assesses residual risk at moderate or lower across all domains, a conclusion its internal governance process reviewed before deployment. Meta says a full report with additional safety evaluations will be available later.\n\nOn cyber capability, Full scores 88.4% on CyberGym and 58.9% on ExploitBench. On ExploitGym it captured 92 flags, but a verification pass found only 21 used the intended vulnerability and 71 came through unintended ones, so 21 is the figure comparable with the published ranking. Attack success rates on adversarial cyber tests are low, at 0.8% on CyberBench, 1.1% to 2.3% on the PolyGuard cyber subset, and 8% to 9% on RedCode. On biology, Full scores 60.7% on average across SecureBio's capability tests, 89.0% on LAB-Bench ProtocolQA, and 95.6% on SeqQA with tools. Two ABC Bench tasks were not scored because of a high rate of refusals. Refusal coverage on high-risk prompts reaches 98.7% on BioTIER in English and 98.2% under adversarial transformations, and 100% on Meta's internal chemistry test in English.\n\nFor loss of control, Full scores 87.3% on MASK, which tests honesty under pressure, and 0% on Agentic Misalignment. Meta's launch chart reports a 3.4% violation rate on its internal test of mishandling personal information, a relevant limitation for an agent given access to messages and calendars. Meta's limitations list says the model can produce inaccurate or objectionable responses, may err in multi-step reasoning on novel tasks, remains weak at visual and spatial reasoning especially in 3D, was not evaluated in all languages, varies with serving configuration and agent harness, and is not intended for users under 18. It recommends human confirmation for irreversible actions in agentic settings.\n\n## API and behavior changes\n\nMuse Wonder is the only member of the series that can be called through the API. The `none` effort setting is rejected, so applications that disable reasoning need to use `minimal` or higher. On Chat Completions the model's reasoning does not carry from one turn to the next. To preserve it, developers use the Responses API with a previous response ID. Because the API accepts OpenAI Responses, OpenAI Chat Completions, and Anthropic Messages formats, supported request formats can be reused after a base URL and model name change, subject to Meta's documented parameter limits, and OpenRouter lists the model as `meta/muse-wonder`. Billing must be set up because there is no contributor tier. Availability depends on the user's country.\n\nReasoning is metered as output, and Artificial Analysis found the model verbose, so budget planning should use cost per task from the effort ladder rather than the list price alone. Active Focus changes agent behavior as well. A model trained to keep trying alternatives can pursue approaches the user did not intend, which is why Meta's own guidance calls for human confirmation before irreversible actions.\n\n## Applications and workflow fit\n\nMuse Wonder and Wonder Full are best suited to long-running coding and multi-hour agent work, legal and other document-heavy agent tasks, finance workflows, and personal agents that depend on memory across sessions. Meta's demonstrations include a playable 3D kitchen game generated from customizable image inputs, a running plan built in a browser across apps, custom swag designed and ordered from photos on a personal Instagram profile, and a wedding seating chart built from complex matching requirements. Meta also expects Wonder to power artifact generation in the Muse app and to help train specialized models for it.\n\nThey are not the obvious choice for shorter terminal tasks and building programs from scratch, where Opus 5.5 leads Meta's own table, for computer use, where Opus 5.5 scores higher on OSWorld V2.1, for health question answering, where Opus 5.5 leads HealthBench Pro, or for knowledge-heavy factual recall, where Artificial Analysis found Full weak. Cost-sensitive workloads should compare per-task cost, because GPT-6.1 Sol reaches nearly the same Artificial Analysis score for about one-sixth of the price per task. Teams that want Full's results through an API cannot get them, and an agent that handles private messages should weigh the personal-information violation rate Meta itself reports. As with other agent models, evaluating the full model-plus-[agent harness](/glossary/agent-harness) workflow matters more than any single [benchmark](/glossary/benchmark) row, with Meta's max-effort results and Full's xhigh SWE-Marathon exception kept explicit.\n\n## Assessment\n\nMuse Wonder is the first Meta model to sit in the frontier conversation, and Muse Wonder Full is the first to place near Anthropic's 5.5 models on Artificial Analysis's index. The API model scores close to GPT-6 Astra, matches Opus 5.5 on price per token, and leads Meta's table on memory, long-horizon coding, and several agentic business benchmarks. It also has limits: it trails Opus 5.5 and Sonnet 5.5 on the independent index, costs about six times GPT-6.1 Sol per task for a similar score, and keeps its strongest results behind a Muse Code subscription. Several headline benchmarks are Meta's own or rely on score-selection rules that favor the reporting vendor, and the full safety report is still to come.",
  "sources": [
    {
      "title": "Meta Research: Muse Wonder",
      "url": "https://research.meta.ai/blog/muse-wonder"
    },
    {
      "title": "Meta: Muse Wonder official model card",
      "url": "https://research.meta.ai/static/muse-wonder-model-card"
    },
    {
      "title": "Meta: Muse Wonder model and subscription profile",
      "url": "https://dev.meta.ai/models/muse-wonder/"
    },
    {
      "title": "Meta Model API: Models and availability",
      "url": "https://dev.meta.ai/docs/models"
    },
    {
      "title": "Meta Model API: Pricing and rate limits",
      "url": "https://dev.meta.ai/docs/pricing-rate-limits"
    },
    {
      "title": "Meta Model API: Reasoning behavior",
      "url": "https://dev.meta.ai/docs/reasoning"
    },
    {
      "title": "Meta: Muse Code subscriptions and data modes",
      "url": "https://dev.meta.ai/docs/muse-code/subscriptions"
    },
    {
      "title": "Artificial Analysis: Muse Wonder (max)",
      "url": "https://artificialanalysis.ai/models/muse-wonder"
    },
    {
      "title": "Artificial Analysis: Muse Wonder Full (max)",
      "url": "https://artificialanalysis.ai/models/muse-wonder-full"
    },
    {
      "title": "OpenRouter: Muse Wonder",
      "url": "https://openrouter.ai/meta/muse-wonder"
    }
  ],
  "metadata": {
    "category": "Models & Architectures",
    "relatedTerms": [
      "muse-spark",
      "claude-opus-5-5",
      "claude-fable-5",
      "large-language-model",
      "ai-agent",
      "agent-harness",
      "context-window",
      "benchmark",
      "api"
    ],
    "seoDescription": "Muse Wonder and Wonder Full model card: Meta API pricing, 1M context, subscription-only Full access, reasoning, verified benchmarks and safety limits.",
    "seoKeywords": [
      "Muse Wonder",
      "Muse Wonder Full",
      "Muse Wonder model card",
      "Muse Wonder benchmarks",
      "Muse Wonder pricing",
      "Muse Wonder Full API",
      "Muse Code subscriptions",
      "muse-wonder"
    ]
  }
}
$glossary$::jsonb) AS entry(
  id TEXT, slug TEXT, title TEXT, summary TEXT, body TEXT, sources JSONB, metadata JSONB
)
ON CONFLICT (kind, slug, parent_slug) DO UPDATE SET
  path = EXCLUDED.path, title = EXCLUDED.title, summary = EXCLUDED.summary,
  body = EXCLUDED.body, blocks = EXCLUDED.blocks, sources = EXCLUDED.sources,
  metadata = EXCLUDED.metadata, status = EXCLUDED.status,
  published_at = EXCLUDED.published_at, updated_at = EXCLUDED.updated_at;

INSERT INTO models (
  slug, name, developer, release_date, ga_date, access, family,
  comparison_data, sources, notes, metadata, verified_at
)
SELECT seed.slug, seed.name, seed.developer, seed.release_date::date,
  seed.ga_date::date, seed.access, seed.family,
  seed.comparison_data, seed.sources, seed.notes, seed.metadata, NOW()
FROM jsonb_to_recordset($models$
[
  {
    "slug": "muse-wonder",
    "name": "Muse Wonder",
    "developer": "Meta",
    "release_date": "2026-10-12",
    "ga_date": null,
    "access": "proprietary",
    "family": "Muse Wonder",
    "comparison_data": {
      "Developer": "Meta Superintelligence Labs",
      "Release date": "2026-10-12",
      "API model ID": "muse-wonder-1.0 (OpenRouter: meta/muse-wonder)",
      "Context window": "1,048,576 tokens",
      "Max output": "Not stated by Meta in the reviewed sources",
      "Knowledge cutoff": "Not published",
      "Reasoning / effort": "minimal, low, medium, high, xhigh, max; none returns an error. Omission uses a model-determined level; max does not select Wonder Full.",
      "Input / 1M tokens": "$4",
      "Cached input / 1M": "$0.20",
      "Cache write / 1M": "Not separately specified by Meta",
      "Output / 1M tokens": "$20; reasoning tokens billed at the same rate",
      "Batch / flex discount": "No Wonder Contributor tier on the API; separate batch/flex rate not documented",
      "Long-context surcharge": "None; same rates for long contexts",
      "Text input": "Yes",
      "Image / vision input": "Yes",
      "Audio input": "Yes",
      "Video input": "Yes",
      "File / document input": "PDF input supported",
      "Text output": "Yes",
      "Image output": "No",
      "Audio output": "No",
      "Video output": "No",
      "Tool / function calling": "Yes",
      "Computer use": "Yes; browser, desktop and mobile workflows through an agent harness",
      "API access": "Meta Model API preview; OpenRouter; OpenAI Responses/Chat Completions and Anthropic Messages request formats",
      "Product access": "Muse Code pay-as-you-go and all four plans; Muse personal-agent app rollout announced for subscribers",
      "Weights / license": "Proprietary; closed weights",
      "Primary focus": "Personal and professional agents, long-running coding, multimodal understanding and document-heavy legal, finance and health work",
      "Long-horizon work": "Active Memory and Active Focus trained behaviors; long-running coding and recovery from workflow blockers",
      "Agent orchestration": "Tool use, parallel calls and sub-agent delegation through the surrounding agent/harness",
      "User collaboration": "Meta describes clarification and checks before important actions; recommends confirmation for irreversible agent actions",
      "Efficiency / generation change": "AA Intelligence Index 52.39; index-task cost $4.52; 83.0 output tokens/s; 19.27s first chunk and 43.36s first answer including reasoning at max effort",
      "Safety / approvals": "Meta rates mitigated residual risk moderate or lower across assessed domains. Unmitigated chemical/biological high risk is not ruled out; safeguards and confirmation remain relevant.",
      "Terminal-Bench Science 0.1": "Not reported in the reviewed AA snapshot"
    },
    "sources": [
      {
        "title": "Meta Research: Muse Wonder",
        "url": "https://research.meta.ai/blog/muse-wonder"
      },
      {
        "title": "Meta: Muse Wonder official model card",
        "url": "https://research.meta.ai/static/muse-wonder-model-card"
      },
      {
        "title": "Meta: Muse Wonder model and subscription profile",
        "url": "https://dev.meta.ai/models/muse-wonder/"
      },
      {
        "title": "Meta Model API: Models and availability",
        "url": "https://dev.meta.ai/docs/models"
      },
      {
        "title": "Meta Model API: Pricing and rate limits",
        "url": "https://dev.meta.ai/docs/pricing-rate-limits"
      },
      {
        "title": "Meta Model API: Reasoning behavior",
        "url": "https://dev.meta.ai/docs/reasoning"
      },
      {
        "title": "Meta: Muse Code subscriptions and data modes",
        "url": "https://dev.meta.ai/docs/muse-code/subscriptions"
      },
      {
        "title": "Artificial Analysis: Muse Wonder (max)",
        "url": "https://artificialanalysis.ai/models/muse-wonder"
      },
      {
        "title": "Artificial Analysis: Muse Wonder Full (max)",
        "url": "https://artificialanalysis.ai/models/muse-wonder-full"
      },
      {
        "title": "OpenRouter: Muse Wonder",
        "url": "https://openrouter.ai/meta/muse-wonder"
      }
    ],
    "notes": "API preview. AA results are at max effort; default effort is model-determined. Standard API rates differ from Muse Code subscription data modes.",
    "metadata": {
      "verification_pass": "muse-wonder-2026-10-13",
      "catalog_status": "preview",
      "api_available": true,
      "weights_available": false,
      "glossary_slug": "muse-wonder",
      "model_id": "muse-wonder-1.0",
      "context_window_tokens": 1048576,
      "input_modalities": [
        "text",
        "image",
        "audio",
        "video"
      ],
      "output_modalities": [
        "text"
      ],
      "pricing": {
        "input_per_million_usd": 4,
        "cache_read_per_million_usd": 0.2,
        "output_per_million_usd": 20
      },
      "subscription_plans": null,
      "reasoning_efforts": [
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max"
      ],
      "default_effort": "model_determined",
      "independent_intelligence_index": 52.393894697293,
      "artificial_analysis_intelligence_index_version": "v4.3.2",
      "independent_index_cost_per_task_usd": 4.515677444063016,
      "artificial_analysis_observed_at": "2026-10-13",
      "meta_personal_superintelligence_index": 54.3,
      "meta_automationbench_strict_pass_percent": 39.1,
      "meta_gdp_pdf_surge_percent": 30.8,
      "artificial_analysis_output_tokens_per_second": 83.0194332210305,
      "artificial_analysis_time_to_first_chunk_seconds": 19.274186052,
      "artificial_analysis_time_to_first_answer_seconds": 43.36493110292333
    }
  },
  {
    "slug": "muse-wonder-full",
    "name": "Muse Wonder Full",
    "developer": "Meta",
    "release_date": "2026-10-12",
    "ga_date": null,
    "access": "proprietary",
    "family": "Muse Wonder",
    "comparison_data": {
      "Developer": "Meta Superintelligence Labs",
      "Release date": "2026-10-12",
      "API model ID": "No API; Muse Code ID: muse-wonder-full-1.0",
      "Context window": "1M tokens (1,000,000 reported by AA)",
      "Max output": "Not stated by Meta in the reviewed sources",
      "Knowledge cutoff": "Not published",
      "Reasoning / effort": "Muse Code supports max effort. Most published results use max; SWE-Marathon uses xhigh. No API effort parameter.",
      "Input / 1M tokens": "Not offered; subscription only",
      "Cached input / 1M": "Not offered; subscription only",
      "Cache write / 1M": "Not separately specified by Meta",
      "Output / 1M tokens": "Not offered; subscription only",
      "Batch / flex discount": "Not applicable; no API",
      "Long-context surcharge": "Not applicable; no API",
      "Text input": "Yes",
      "Image / vision input": "Yes",
      "Audio input": "Yes",
      "Video input": "Yes",
      "File / document input": "Through Muse Code file handling; no standalone API",
      "Text output": "Yes",
      "Image output": "No",
      "Audio output": "No",
      "Video output": "No",
      "Tool / function calling": "Yes",
      "Computer use": "Yes; browser, desktop and mobile workflows through an agent harness",
      "API access": "No; unavailable on Meta Model API and OpenRouter",
      "Product access": "Muse Code Power Usage ($50/month) and Maximum Usage ($200/month) only",
      "Weights / license": "Proprietary; closed weights",
      "Primary focus": "Personal and professional agents, long-running coding, multimodal understanding and document-heavy legal, finance and health work",
      "Long-horizon work": "Active Memory and Active Focus trained behaviors; long-running coding and recovery from workflow blockers",
      "Agent orchestration": "Tool use, parallel calls and sub-agent delegation through the surrounding agent/harness",
      "User collaboration": "Meta describes clarification and checks before important actions; recommends confirmation for irreversible agent actions",
      "Efficiency / generation change": "AA Intelligence Index 55.80; no API price or timing. Stronger results than Wonder are available only through Muse Code.",
      "Safety / approvals": "Meta rates mitigated residual risk moderate or lower across assessed domains. Unmitigated chemical/biological high risk is not ruled out; safeguards and confirmation remain relevant.",
      "Terminal-Bench Science 0.1": "Not reported in the reviewed AA snapshot"
    },
    "sources": [
      {
        "title": "Meta Research: Muse Wonder",
        "url": "https://research.meta.ai/blog/muse-wonder"
      },
      {
        "title": "Meta: Muse Wonder official model card",
        "url": "https://research.meta.ai/static/muse-wonder-model-card"
      },
      {
        "title": "Meta: Muse Wonder model and subscription profile",
        "url": "https://dev.meta.ai/models/muse-wonder/"
      },
      {
        "title": "Meta Model API: Models and availability",
        "url": "https://dev.meta.ai/docs/models"
      },
      {
        "title": "Meta Model API: Pricing and rate limits",
        "url": "https://dev.meta.ai/docs/pricing-rate-limits"
      },
      {
        "title": "Meta Model API: Reasoning behavior",
        "url": "https://dev.meta.ai/docs/reasoning"
      },
      {
        "title": "Meta: Muse Code subscriptions and data modes",
        "url": "https://dev.meta.ai/docs/muse-code/subscriptions"
      },
      {
        "title": "Artificial Analysis: Muse Wonder (max)",
        "url": "https://artificialanalysis.ai/models/muse-wonder"
      },
      {
        "title": "Artificial Analysis: Muse Wonder Full (max)",
        "url": "https://artificialanalysis.ai/models/muse-wonder-full"
      },
      {
        "title": "OpenRouter: Muse Wonder",
        "url": "https://openrouter.ai/meta/muse-wonder"
      }
    ],
    "notes": "Subscription-only proprietary model. No API and no per-token price. Keep separate from max-effort Wonder. AA metrics and Meta-reported results retain their distinct sources and metrics.",
    "metadata": {
      "verification_pass": "muse-wonder-2026-10-13",
      "catalog_status": "subscription_only",
      "api_available": false,
      "weights_available": false,
      "glossary_slug": "muse-wonder",
      "model_id": "muse-wonder-full-1.0",
      "context_window_tokens": 1000000,
      "input_modalities": [
        "text",
        "image",
        "audio",
        "video"
      ],
      "output_modalities": [
        "text"
      ],
      "pricing": null,
      "subscription_plans": [
        {
          "name": "Power Usage",
          "monthly_usd": 50
        },
        {
          "name": "Maximum Usage",
          "monthly_usd": 200
        }
      ],
      "reasoning_efforts": null,
      "default_effort": null,
      "independent_intelligence_index": 55.8021199855616,
      "artificial_analysis_intelligence_index_version": "v4.3.2",
      "independent_index_cost_per_task_usd": null,
      "artificial_analysis_observed_at": "2026-10-13",
      "meta_personal_superintelligence_index": 59.1,
      "meta_automationbench_strict_pass_percent": 45.4,
      "meta_gdp_pdf_surge_percent": 33.2
    }
  }
]
$models$::jsonb) AS seed(
  slug TEXT, name TEXT, developer TEXT, release_date TEXT, ga_date TEXT,
  access TEXT, family TEXT, comparison_data JSONB, sources JSONB, notes TEXT, metadata JSONB
)
ON CONFLICT (slug) DO UPDATE SET
  name = EXCLUDED.name, developer = EXCLUDED.developer,
  release_date = EXCLUDED.release_date, ga_date = EXCLUDED.ga_date,
  access = EXCLUDED.access, family = EXCLUDED.family,
  comparison_data = EXCLUDED.comparison_data, sources = EXCLUDED.sources,
  notes = EXCLUDED.notes, metadata = EXCLUDED.metadata,
  verified_at = EXCLUDED.verified_at, updated_at = NOW();

-- Keep AA partial metrics distinct from Meta's strict pass@1 and Surge metrics.
-- Composite indexes and subscription prices stay in model metadata.
INSERT INTO model_benchmarks (
  id, model_slug, category, benchmark_name, benchmark_version,
  score_numeric, score_display, score_unit, tools, reasoning_effort,
  harness, evaluator, evaluation_date, source, notes
)
SELECT evidence.id, evidence.model_slug, evidence.category,
  evidence.benchmark_name, evidence.benchmark_version,
  evidence.score_numeric, evidence.score_display, evidence.score_unit,
  evidence.tools, evidence.reasoning_effort, evidence.harness, evidence.evaluator,
  evidence.evaluation_date::date, evidence.source, evidence.notes
FROM jsonb_to_recordset($benchmarks$
[
  {
    "id": "tq-20261013-muse-wonder-terminal-bench-4-0",
    "model_slug": "muse-wonder",
    "category": "coding",
    "benchmark_name": "Terminal-Bench 4.0",
    "benchmark_version": null,
    "score_numeric": 52.020202020202,
    "score_display": "52.0% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row."
  },
  {
    "id": "tq-20261013-muse-wonder-scicode",
    "model_slug": "muse-wonder",
    "category": "coding",
    "benchmark_name": "SciCode",
    "benchmark_version": null,
    "score_numeric": 60.300925925925895,
    "score_display": "60.3% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row. AA marks the evaluation under review."
  },
  {
    "id": "tq-20261013-muse-wonder-critpt",
    "model_slug": "muse-wonder",
    "category": "math_reasoning",
    "benchmark_name": "CritPt",
    "benchmark_version": null,
    "score_numeric": 28.2857142857143,
    "score_display": "28.3% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row. AA marks the evaluation under review."
  },
  {
    "id": "tq-20261013-muse-wonder-humanity-s-last-exam",
    "model_slug": "muse-wonder",
    "category": "knowledge",
    "benchmark_name": "Humanity's Last Exam",
    "benchmark_version": null,
    "score_numeric": 50.0926784059314,
    "score_display": "50.1% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row."
  },
  {
    "id": "tq-20261013-muse-wonder-aa-omniscience-hallucination-rate",
    "model_slug": "muse-wonder",
    "category": "knowledge",
    "benchmark_name": "AA-Omniscience Hallucination Rate",
    "benchmark_version": null,
    "score_numeric": 26.513865308432372,
    "score_display": "26.5% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row. Lower is better."
  },
  {
    "id": "tq-20261013-muse-wonder-aa-omniscience-accuracy",
    "model_slug": "muse-wonder",
    "category": "knowledge",
    "benchmark_name": "AA-Omniscience Accuracy",
    "benchmark_version": null,
    "score_numeric": 41.099999999999994,
    "score_display": "41.1% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row."
  },
  {
    "id": "tq-20261013-muse-wonder-gdp-pdf",
    "model_slug": "muse-wonder",
    "category": "professional",
    "benchmark_name": "GDP.pdf",
    "benchmark_version": null,
    "score_numeric": 30.599999999999998,
    "score_display": "30.6% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row. AA all-pass metric; distinct from the Surge score in Meta's comparison."
  },
  {
    "id": "tq-20261013-muse-wonder-aa-lcr-v1-1",
    "model_slug": "muse-wonder",
    "category": "math_reasoning",
    "benchmark_name": "AA-LCR v1.1",
    "benchmark_version": null,
    "score_numeric": 80.6666666666667,
    "score_display": "80.7% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row."
  },
  {
    "id": "tq-20261013-muse-wonder-aa-omniscience-index",
    "model_slug": "muse-wonder",
    "category": "knowledge",
    "benchmark_name": "AA-Omniscience Index",
    "benchmark_version": null,
    "score_numeric": 25.483333333333334,
    "score_display": "25.5 points (AA)",
    "score_unit": "points",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row."
  },
  {
    "id": "tq-20261013-muse-wonder-gdpval-aa-v2-1",
    "model_slug": "muse-wonder",
    "category": "professional",
    "benchmark_name": "GDPval-AA v2.1",
    "benchmark_version": null,
    "score_numeric": 1770.34,
    "score_display": "1770.34 Elo (AA)",
    "score_unit": "elo",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row."
  },
  {
    "id": "tq-20261013-muse-wonder-aa-briefcase-v1-1",
    "model_slug": "muse-wonder",
    "category": "professional",
    "benchmark_name": "AA-Briefcase v1.1",
    "benchmark_version": null,
    "score_numeric": 1658.65,
    "score_display": "1658.65 Elo (AA)",
    "score_unit": "elo",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row."
  },
  {
    "id": "tq-20261013-muse-wonder-automationbench-aa",
    "model_slug": "muse-wonder",
    "category": "agentic_computer_use",
    "benchmark_name": "AutomationBench-AA",
    "benchmark_version": null,
    "score_numeric": 68.09455131207758,
    "score_display": "68.1% (AA; partial)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row. Partial assertion score, not the strict pass@1 percentage reported in Meta's launch comparison."
  },
  {
    "id": "tq-20261013-muse-wonder-deepswe-v1-1",
    "model_slug": "muse-wonder",
    "category": "coding",
    "benchmark_name": "DeepSWE v1.1",
    "benchmark_version": null,
    "score_numeric": 78.8,
    "score_display": "78.8% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Datacurve (via Meta)",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Datacurve third-party result reproduced by Meta; pass@1."
  },
  {
    "id": "tq-20261013-muse-wonder-swe-marathon",
    "model_slug": "muse-wonder",
    "category": "coding",
    "benchmark_name": "SWE-Marathon",
    "benchmark_version": null,
    "score_numeric": 53.1,
    "score_display": "53.1% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Abundant AI (via Meta)",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Abundant AI result reproduced by Meta; pass@1. Competitors can use different agent products."
  },
  {
    "id": "tq-20261013-muse-wonder-programbench",
    "model_slug": "muse-wonder",
    "category": "coding",
    "benchmark_name": "ProgramBench",
    "benchmark_version": null,
    "score_numeric": 50,
    "score_display": "50.0% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "mini-swe-agent",
    "evaluator": "Meta",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Percentage of rollouts passing at least 95% of tests; mini-swe-agent harness."
  },
  {
    "id": "tq-20261013-muse-wonder-memorybench",
    "model_slug": "muse-wonder",
    "category": "knowledge",
    "benchmark_name": "MemoryBench",
    "benchmark_version": null,
    "score_numeric": 85.6,
    "score_display": "85.6% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Meta",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Meta internal memory benchmark; not an independent leaderboard."
  },
  {
    "id": "tq-20261013-muse-wonder-harvey-lab-aa-v1-1",
    "model_slug": "muse-wonder",
    "category": "professional",
    "benchmark_name": "Harvey LAB-AA v1.1",
    "benchmark_version": null,
    "score_numeric": 10,
    "score_display": "10.0% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Artificial Analysis (via Meta)",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. AA result reproduced by Meta; strict all-criteria and hallucination-check pass rate. Direct Muse model-page snapshot does not expose this value."
  },
  {
    "id": "tq-20261013-muse-wonder-vals-finance-agent-v2",
    "model_slug": "muse-wonder",
    "category": "professional",
    "benchmark_name": "Vals Finance Agent V2",
    "benchmark_version": null,
    "score_numeric": 56.6,
    "score_display": "56.6% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Vals AI (via Meta)",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Vals private-test-set accuracy reproduced by Meta."
  },
  {
    "id": "tq-20261013-muse-wonder-healthbench-pro",
    "model_slug": "muse-wonder",
    "category": "professional",
    "benchmark_name": "HealthBench Pro",
    "benchmark_version": null,
    "score_numeric": 60.5,
    "score_display": "60.5% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Meta",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Meta common-framework length-normalized weighted rubric score, without tools."
  },
  {
    "id": "tq-20261013-muse-wonder-jobbench",
    "model_slug": "muse-wonder",
    "category": "professional",
    "benchmark_name": "JobBench",
    "benchmark_version": null,
    "score_numeric": 70.5,
    "score_display": "70.5% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Meta",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Meta common-framework mean reward across professional tasks."
  },
  {
    "id": "tq-20261013-muse-wonder-osworld-2-1",
    "model_slug": "muse-wonder",
    "category": "agentic_computer_use",
    "benchmark_name": "OSWorld 2.1",
    "benchmark_version": null,
    "score_numeric": 72.6,
    "score_display": "72.6% (Meta; partial)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Meta",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Mean per-task partial reward. Strict binary completion is 34.3%, a different metric."
  },
  {
    "id": "tq-20261013-muse-wonder-full-terminal-bench-4-0",
    "model_slug": "muse-wonder-full",
    "category": "coding",
    "benchmark_name": "Terminal-Bench 4.0",
    "benchmark_version": null,
    "score_numeric": 57.07070707070711,
    "score_display": "57.1% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder-full",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row."
  },
  {
    "id": "tq-20261013-muse-wonder-full-scicode",
    "model_slug": "muse-wonder-full",
    "category": "coding",
    "benchmark_name": "SciCode",
    "benchmark_version": null,
    "score_numeric": 62.6157407407407,
    "score_display": "62.6% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder-full",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row. AA marks the evaluation under review."
  },
  {
    "id": "tq-20261013-muse-wonder-full-critpt",
    "model_slug": "muse-wonder-full",
    "category": "math_reasoning",
    "benchmark_name": "CritPt",
    "benchmark_version": null,
    "score_numeric": 32.857142857142904,
    "score_display": "32.9% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder-full",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row. AA marks the evaluation under review."
  },
  {
    "id": "tq-20261013-muse-wonder-full-humanity-s-last-exam",
    "model_slug": "muse-wonder-full",
    "category": "knowledge",
    "benchmark_name": "Humanity's Last Exam",
    "benchmark_version": null,
    "score_numeric": 53.6144578313253,
    "score_display": "53.6% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder-full",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row."
  },
  {
    "id": "tq-20261013-muse-wonder-full-aa-omniscience-hallucination-rate",
    "model_slug": "muse-wonder-full",
    "category": "knowledge",
    "benchmark_name": "AA-Omniscience Hallucination Rate",
    "benchmark_version": null,
    "score_numeric": 23.011447020839448,
    "score_display": "23.0% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder-full",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row. Lower is better."
  },
  {
    "id": "tq-20261013-muse-wonder-full-aa-omniscience-accuracy",
    "model_slug": "muse-wonder-full",
    "category": "knowledge",
    "benchmark_name": "AA-Omniscience Accuracy",
    "benchmark_version": null,
    "score_numeric": 43.21666666666666,
    "score_display": "43.2% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder-full",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row."
  },
  {
    "id": "tq-20261013-muse-wonder-full-gdp-pdf",
    "model_slug": "muse-wonder-full",
    "category": "professional",
    "benchmark_name": "GDP.pdf",
    "benchmark_version": null,
    "score_numeric": 32.6,
    "score_display": "32.6% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder-full",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row. AA all-pass metric; distinct from the Surge score in Meta's comparison."
  },
  {
    "id": "tq-20261013-muse-wonder-full-aa-lcr-v1-1",
    "model_slug": "muse-wonder-full",
    "category": "math_reasoning",
    "benchmark_name": "AA-LCR v1.1",
    "benchmark_version": null,
    "score_numeric": 83.6666666666667,
    "score_display": "83.7% (AA)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder-full",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row."
  },
  {
    "id": "tq-20261013-muse-wonder-full-aa-omniscience-index",
    "model_slug": "muse-wonder-full",
    "category": "knowledge",
    "benchmark_name": "AA-Omniscience Index",
    "benchmark_version": null,
    "score_numeric": 30.15,
    "score_display": "30.2 points (AA)",
    "score_unit": "points",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder-full",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row."
  },
  {
    "id": "tq-20261013-muse-wonder-full-gdpval-aa-v2-1",
    "model_slug": "muse-wonder-full",
    "category": "professional",
    "benchmark_name": "GDPval-AA v2.1",
    "benchmark_version": null,
    "score_numeric": 1857.09,
    "score_display": "1857.09 Elo (AA)",
    "score_unit": "elo",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder-full",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row."
  },
  {
    "id": "tq-20261013-muse-wonder-full-aa-briefcase-v1-1",
    "model_slug": "muse-wonder-full",
    "category": "professional",
    "benchmark_name": "AA-Briefcase v1.1",
    "benchmark_version": null,
    "score_numeric": 1729.32,
    "score_display": "1729.32 Elo (AA)",
    "score_unit": "elo",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder-full",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row."
  },
  {
    "id": "tq-20261013-muse-wonder-full-automationbench-aa",
    "model_slug": "muse-wonder-full",
    "category": "agentic_computer_use",
    "benchmark_name": "AutomationBench-AA",
    "benchmark_version": null,
    "score_numeric": 71.33074973223975,
    "score_display": "71.3% (AA; partial)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Artificial Analysis",
    "evaluator": "Artificial Analysis",
    "evaluation_date": null,
    "source": "https://artificialanalysis.ai/models/muse-wonder-full",
    "notes": "Artificial Analysis public model-page data at max effort, observed October 13, 2026. Exact evaluation date and tool settings are not stated in the row. Partial assertion score, not the strict pass@1 percentage reported in Meta's launch comparison."
  },
  {
    "id": "tq-20261013-muse-wonder-full-deepswe-v1-1",
    "model_slug": "muse-wonder-full",
    "category": "coding",
    "benchmark_name": "DeepSWE v1.1",
    "benchmark_version": null,
    "score_numeric": 79,
    "score_display": "79.0% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Datacurve (via Meta)",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Datacurve third-party result reproduced by Meta; pass@1."
  },
  {
    "id": "tq-20261013-muse-wonder-full-swe-marathon",
    "model_slug": "muse-wonder-full",
    "category": "coding",
    "benchmark_name": "SWE-Marathon",
    "benchmark_version": null,
    "score_numeric": 57.5,
    "score_display": "57.5% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "xhigh",
    "harness": "Meta-reported evaluation",
    "evaluator": "Abundant AI (via Meta)",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Abundant AI result reproduced by Meta; pass@1. Competitors can use different agent products."
  },
  {
    "id": "tq-20261013-muse-wonder-full-programbench",
    "model_slug": "muse-wonder-full",
    "category": "coding",
    "benchmark_name": "ProgramBench",
    "benchmark_version": null,
    "score_numeric": 59,
    "score_display": "59.0% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "mini-swe-agent",
    "evaluator": "Meta",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Percentage of rollouts passing at least 95% of tests; mini-swe-agent harness."
  },
  {
    "id": "tq-20261013-muse-wonder-full-memorybench",
    "model_slug": "muse-wonder-full",
    "category": "knowledge",
    "benchmark_name": "MemoryBench",
    "benchmark_version": null,
    "score_numeric": 85.5,
    "score_display": "85.5% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Meta",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Meta internal memory benchmark; not an independent leaderboard."
  },
  {
    "id": "tq-20261013-muse-wonder-full-harvey-lab-aa-v1-1",
    "model_slug": "muse-wonder-full",
    "category": "professional",
    "benchmark_name": "Harvey LAB-AA v1.1",
    "benchmark_version": null,
    "score_numeric": 13.1,
    "score_display": "13.1% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Artificial Analysis (via Meta)",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. AA result reproduced by Meta; strict all-criteria and hallucination-check pass rate. Direct Muse model-page snapshot does not expose this value."
  },
  {
    "id": "tq-20261013-muse-wonder-full-vals-finance-agent-v2",
    "model_slug": "muse-wonder-full",
    "category": "professional",
    "benchmark_name": "Vals Finance Agent V2",
    "benchmark_version": null,
    "score_numeric": 59.1,
    "score_display": "59.1% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Vals AI (via Meta)",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Vals private-test-set accuracy reproduced by Meta."
  },
  {
    "id": "tq-20261013-muse-wonder-full-healthbench-pro",
    "model_slug": "muse-wonder-full",
    "category": "professional",
    "benchmark_name": "HealthBench Pro",
    "benchmark_version": null,
    "score_numeric": 62.3,
    "score_display": "62.3% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Meta",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Meta common-framework length-normalized weighted rubric score, without tools."
  },
  {
    "id": "tq-20261013-muse-wonder-full-jobbench",
    "model_slug": "muse-wonder-full",
    "category": "professional",
    "benchmark_name": "JobBench",
    "benchmark_version": null,
    "score_numeric": 75.3,
    "score_display": "75.3% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Meta",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Meta common-framework mean reward across professional tasks."
  },
  {
    "id": "tq-20261013-muse-wonder-full-osworld-2-1",
    "model_slug": "muse-wonder-full",
    "category": "agentic_computer_use",
    "benchmark_name": "OSWorld 2.1",
    "benchmark_version": null,
    "score_numeric": 78.4,
    "score_display": "78.4% (Meta; partial)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Meta",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta model-card result reviewed October 13, 2026. Mean per-task partial reward. Strict binary completion is 39.4%, a different metric."
  },
  {
    "id": "tq-20261013-muse-wonder-full-cybergym",
    "model_slug": "muse-wonder-full",
    "category": "coding",
    "benchmark_name": "CyberGym",
    "benchmark_version": null,
    "score_numeric": 88.4,
    "score_display": "88.4% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Meta",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta Wonder Full risk-assessment evaluation. Cyber capability score."
  },
  {
    "id": "tq-20261013-muse-wonder-full-exploitbench",
    "model_slug": "muse-wonder-full",
    "category": "coding",
    "benchmark_name": "ExploitBench",
    "benchmark_version": null,
    "score_numeric": 58.9,
    "score_display": "58.9% (Meta)",
    "score_unit": "percent",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Meta",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta Wonder Full risk-assessment evaluation. Cyber capability score."
  },
  {
    "id": "tq-20261013-muse-wonder-full-exploitgym",
    "model_slug": "muse-wonder-full",
    "category": "coding",
    "benchmark_name": "ExploitGym",
    "benchmark_version": null,
    "score_numeric": 21,
    "score_display": "21 intended-vulnerability flags (Meta)",
    "score_unit": "flags",
    "tools": null,
    "reasoning_effort": "max",
    "harness": "Meta-reported evaluation",
    "evaluator": "Meta",
    "evaluation_date": null,
    "source": "https://research.meta.ai/static/muse-wonder-model-card",
    "notes": "Meta Wonder Full risk-assessment evaluation. Only intended-vulnerability exploits count toward the published ranking; 92 total flags included 71 unintended-vulnerability captures."
  }
]
$benchmarks$::jsonb) AS evidence(
  id TEXT, model_slug TEXT, category TEXT, benchmark_name TEXT,
  benchmark_version TEXT, score_numeric NUMERIC, score_display TEXT, score_unit TEXT,
  tools BOOLEAN, reasoning_effort TEXT, harness TEXT, evaluator TEXT, evaluation_date TEXT,
  source TEXT, notes TEXT
)
ON CONFLICT (id) DO UPDATE SET
  model_slug = EXCLUDED.model_slug, category = EXCLUDED.category,
  benchmark_name = EXCLUDED.benchmark_name, benchmark_version = EXCLUDED.benchmark_version,
  score_numeric = EXCLUDED.score_numeric, score_display = EXCLUDED.score_display,
  score_unit = EXCLUDED.score_unit, tools = EXCLUDED.tools, reasoning_effort = EXCLUDED.reasoning_effort,
  harness = EXCLUDED.harness, evaluator = EXCLUDED.evaluator,
  evaluation_date = EXCLUDED.evaluation_date, source = EXCLUDED.source, notes = EXCLUDED.notes;
