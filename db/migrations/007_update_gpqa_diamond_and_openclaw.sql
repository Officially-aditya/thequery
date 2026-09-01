WITH glossary_updates(slug, title, summary, body, sources, metadata) AS (
  VALUES
  (
    'gpqa-diamond',
    'GPQA Diamond',
    'A 198-question, expert-validated subset of GPQA that tests graduate-level biology, chemistry, and physics through four-option multiple-choice questions.',
    $gpqa$
GPQA Diamond is a 198-question subset of **GPQA**, short for Graduate-Level Google-Proof Question Answering. It evaluates advanced scientific knowledge and reasoning through four-option multiple-choice questions written by people with, or pursuing, PhDs in biology, chemistry, and physics.

The full GPQA Main set contains 448 questions. Diamond is a filtered subset chosen for stronger agreement and difficulty: both domain-expert validators answered the item correctly, while no more than one of three skilled non-expert validators answered it correctly despite having unrestricted web access. It is called “Diamond” because it is the benchmark's high-confidence, hard subset, not because it covers every field or measures a separate capability.

## What “Google-proof” means

“Google-proof” is a description of the original human-validation experiment, not a claim that the answer can never be found online. Skilled non-experts spent more than 30 minutes per question on average and could use the web, yet performed poorly. The questions often require specialized concepts, multi-step reasoning, and the ability to distinguish plausible scientific distractors.

The term has limits. Search tools improve, explanations spread, and the benchmark questions have been public since 2023. A model with retrieval or web access is being tested under a different protocol from a closed-book model. The name should not be interpreted as permanent protection from data contamination or answer lookup.

## GPQA Main vs GPQA Diamond

| Property | GPQA Main | GPQA Diamond |
| --- | --- | --- |
| Number of questions | 448 | 198 |
| Domains | Biology, chemistry, and physics | Biology, chemistry, and physics |
| Format | Four-option multiple choice | Four-option multiple choice |
| Selection | Removes clearly weak or invalid submissions from the broader collection | Requires both expert validators correct and a majority of non-experts incorrect |
| Intended use | The original paper's recommended primary set | Higher-confidence subset widely used on frontier-model leaderboards |
| Random-guess baseline | 25% | 25% |

Diamond is not simply the 198 questions on which old AI models scored lowest. Its membership was determined through human validation. That makes it a quality filter, but it also means human figures measured on the selected subset require careful interpretation.

## Human baseline numbers

The original GPQA paper reported about **65% expert accuracy on GPQA overall**, rising to 74% after discounting clear mistakes identified in retrospective expert feedback. Skilled non-experts reached about 34% despite web access.

For Diamond, the paper's table reports **81.3% expert accuracy** and **22.1% non-expert accuracy** under its validation and selection process. The expert figure is selection-conditioned because Diamond requires both expert validators to get an item right for inclusion. It should not be presented as an independent universal “PhD score” or a hard ceiling on human performance.

Some later model reports use separately recruited human experts and produce different baselines. Those studies may use another population, protocol, time limit, or interpretation rule. A model score and a human score are comparable only when they come from a clearly matched setup.

## How GPQA Diamond is scored

The usual metric is **accuracy**: the percentage of the 198 questions for which the evaluator extracts the correct option. Because each question has four choices, random guessing averages 25%.

The model may be asked for an answer letter, the answer text, or a rationale followed by an answer. Evaluators typically randomize answer order and parse a final choice. The rationale is usually not graded by the canonical accuracy metric, so a model can receive credit for the right option even when its explanation is weak or partly incorrect.

One question is approximately **0.505 percentage points** of the Diamond score. At this sample size, a difference of a few points can reflect only a handful of items. Stochastic models should be run with multiple seeds or trials, and comparisons should include uncertainty rather than treating every decimal-place difference as a stable rank.

## What GPQA Diamond measures

GPQA Diamond provides evidence about whether a tested model or system can select correct answers to difficult, expert-authored science questions under a stated protocol. Strong performance can reflect domain knowledge, scientific reasoning, calculation, elimination of distractors, and test-taking skill.

It is especially useful because the questions were designed to create an **expert and non-expert gap**. The original research motivation included scalable oversight: studying how people might evaluate or supervise answers in domains where the AI may know more than the person supervising it.

## What it does not measure

GPQA Diamond does not directly test laboratory work, literature review, experimental design, open-ended scientific writing, uncertainty communication, citation quality, research originality, or whether a system will be reliable in deployment. It covers three broad sciences but not medicine, engineering, mathematics as a standalone field, social science, humanities, or software engineering.

A correct multiple-choice answer also does not prove that a model used a valid reasoning process. Conversely, an exacting answer parser can mark an otherwise correct response wrong if the final option is malformed. The score belongs to the complete evaluation configuration, not only to a model family name.

## Why GPQA Diamond scores differ across reports

| Evaluation choice | How it can change the score |
| --- | --- |
| Zero-shot vs few-shot prompting | Examples can clarify format or alter reasoning behavior |
| Direct answer vs chain-of-thought instruction | Reasoning instructions can help some models and hurt others |
| Reasoning effort or token budget | More test-time compute can improve difficult-item performance |
| One sample vs majority vote or best-of-n | Multiple attempts spend more compute and are not pass@1 |
| Closed-book vs tools or retrieval | External information changes the capability being measured |
| Answer-order randomization | Reduces position effects and answer-choice memorization |
| Parser and invalid-answer policy | Determines how malformed or ambiguous outputs are counted |
| Model checkpoint or API date | Provider updates can change behavior under the same product name |

Do not compare a vendor's best-of-many result with another model's single deterministic pass as though both were ordinary accuracy.

## Limitations in 2026

### Small test set

With 198 items, GPQA Diamond has limited resolution. Category-level results use even fewer examples, making rankings noisy. Confidence intervals and per-domain scores are more informative than a bare overall number.

### Saturation

Frontier systems now score near the top of the benchmark. As scores approach the ceiling, remaining errors increasingly reflect ambiguous items, evaluation details, or a narrow set of difficult questions. The benchmark can still reveal regressions and compare smaller systems, but it has less power to separate the strongest models.

### Contamination

The dataset is public and the repository includes a canary string asking dataset builders to filter it from training corpora. A canary helps responsible developers detect the source but cannot prove that questions, answers, explanations, screenshots, or derivatives never entered training or post-training data.

### Question quality

Expert-authored does not mean error-free. Scientific wording can be ambiguous, an answer key can be disputed, or more than one choice can be defensible under another assumption. Corrections and benchmark versions should be documented, and evaluators should state whether they used the original or a cleaned variant.

### Multiple-choice shortcuts

A model can sometimes exploit option patterns, partial knowledge, or elimination without producing a complete scientific solution. Four-choice accuracy also gives a 25% chance baseline. Open-ended answer matching or expert grading tests a different behavior and should not be merged silently with canonical multiple-choice scores.

## How to read a GPQA Diamond claim

Check the exact dataset version, whether all 198 items were used, model checkpoint, prompt, reasoning setting, tool access, temperature, number of samples, voting method, answer-order policy, parser, and invalid-response handling. Look for multiple runs, confidence intervals, per-domain results, and contamination disclosure.

A GPQA Diamond score is useful evidence when the protocol is reproducible. It is not proof that a model has a PhD, exceeds every scientist, or can conduct safe and reliable research.

## Bottom line

GPQA Diamond is a compact, difficult science benchmark selected for expert agreement and non-expert difficulty. It became popular because it challenged frontier models and highlighted the problem of supervising systems on questions most people cannot verify. In 2026, its small size, public exposure, and approaching saturation make protocol details and uncertainty just as important as the headline score.
$gpqa$::text,
    jsonb_build_array(
      jsonb_build_object('title', 'GPQA: A Graduate-Level Google-Proof Q&A Benchmark', 'url', 'https://openreview.net/forum?id=Ti67584b98'),
      jsonb_build_object('title', 'GPQA paper on arXiv', 'url', 'https://arxiv.org/abs/2311.12022'),
      jsonb_build_object('title', 'Official GPQA repository and evaluation code', 'url', 'https://github.com/idavidrein/gpqa'),
      jsonb_build_object('title', 'Official GPQA dataset on Hugging Face', 'url', 'https://huggingface.co/datasets/Idavidrein/gpqa')
    ),
    jsonb_build_object(
      'category', 'Foundations',
      'relatedTerms', jsonb_build_array('benchmark', 'large-language-model', 'artificial-analysis', 'lmarena'),
      'analogy', 'GPQA Diamond is like a compact graduate science qualifying exam whose questions were kept only when experts agreed and skilled web-using non-experts still struggled.',
      'seoDescription', 'GPQA Diamond is a 198-question graduate science benchmark. Learn its selection rules, human baselines, scoring protocol, limitations, and saturation risks.',
      'seoKeywords', jsonb_build_array('what is GPQA Diamond', 'GPQA Diamond explained', 'GPQA Diamond benchmark', 'GPQA Diamond 198 questions', 'GPQA Diamond human baseline', 'GPQA Diamond score', 'GPQA Diamond vs GPQA Main', 'Google-proof QA benchmark', 'graduate science AI benchmark', 'GPQA Diamond saturation', 'GPQA data contamination', 'GPQA evaluation protocol')
    )
  ),
  (
    'openclaw',
    'OpenClaw',
    'An open-source, self-hosted AI agent gateway that connects models, tools, persistent sessions, automations, devices, and messaging channels under user control.',
    $openclaw$
OpenClaw is an open-source, self-hosted platform for running personal or team AI agents. It connects language models to messaging channels, files, browsers, command-line tools, devices, skills, plugins, memory, and scheduled work through one long-running process called the **Gateway**.

OpenClaw is not an AI model. It is the agent runtime and integration layer around models from providers such as OpenAI, Anthropic, Google, hosted gateways, or local model services. The same installation can be a private assistant on one laptop, an always-on server reached through chat apps, or a shared gateway for a mutually trusted team.

The project began as **ClawdBot**, briefly used the name **MoltBot**, and became OpenClaw. It was created by Peter Steinberger and is now developed through the independent, nonprofit OpenClaw Foundation and its contributor community. The official project states that it is not affiliated with Anthropic.

## OpenClaw 2.0

**OpenClaw 2.0 is the product name for release v2026.8.1.** The official announcement is dated August 30, 2026 and describes it as the largest update in the project's history, built from more than 16,000 pull requests by 933 contributors, including 569 first-time contributors.

The release began as an effort to simplify setup and rebuild the browser app, then expanded across installation, messaging, memory, models, skills, plugins, automations, native apps, security, and the agent runtime. The “2.0” label is a product milestone, while the package and release tag remain **2026.8.1**.

## What changed in OpenClaw 2.0

| Area | OpenClaw 2.0 change | Practical effect |
| --- | --- | --- |
| Onboarding | Starts from existing ChatGPT or Claude subscriptions, API keys, and local models while deferring optional configuration | Gets a first conversation working sooner |
| Browser app | Rebuilt as a first-class Control UI that opens directly into chat | Setup, sessions, live work, and configuration share one browser experience |
| Conversation search | Searches visible transcript text by exact words or phrases | Finds and reopens earlier work without manually scanning sessions |
| Remote execution | Runs sessions on paired devices or cloud workers and can move the workspace with the session | Lets work continue on another machine and reuse warm workers or project seeds |
| Live progress | Keeps durable progress cards across reloads and shows subagent activity and accumulated edits | Makes long-running work easier to follow without losing state |
| Structured questions | Renders agent questions as interactive cards, messaging buttons, or plain text with free-text and Skip paths | Makes approval and clarification flows more consistent across clients |
| Interactive output | Adds chat widgets and pinnable session dashboards with scoped actions and allowed network origins | Turns results into controlled interfaces rather than static text only |
| Credentials | Requests secrets through masked prompts outside chat and model context, with an optional destination-limited proxy | Reduces accidental exposure of credentials to transcripts and models |
| Recurring approvals | Grants permission for one exact automation operation and requires new approval when the operation changes | Avoids repeated prompts without creating an unlimited standing grant |
| Media | Preserves richer audio and video across upload, generation, playback, and reload | Improves multimodal workflows in web and native clients |

These are release-level capabilities, not guarantees that every provider, plugin, channel, or mobile app supports every feature in exactly the same way.

## OpenClaw 2.0 breaking changes and upgrades

OpenClaw 2.0 includes migrations that existing installations must review:

- The bundled OpenProse plugin and `/prose` command were removed. Existing `.prose` source files can remain, but users should migrate to the upstream Agent Skill workflow.
- Shipped `codex/*` and `openai-codex/*` model references, provider settings, sessions, and automation routes move to `openai/*` while retaining Codex runtime intent.
- Many providers and channels are installed as separate official plugins instead of all shipping in the core package.
- Plugin SDK users must move away from deprecated broad subpaths to the focused public imports listed in the SDK migration guide.
- Signed and notarized macOS 2026.8.1 builds shipped with the release. The release notes say new iOS and Android distributions follow separately, so older mobile downloads should not be mislabeled as 2026.8.1 artifacts.

The supported first step after upgrading is **`openclaw doctor --fix`**. It cleans stale OpenProse configuration, migrates supported OpenAI routes, repairs missing configured official packages where possible, and flags conflicts that still need operator judgment. Back up the OpenClaw state directory before a major upgrade and read the full v2026.8.1 notes for plugin-specific compatibility.

## How OpenClaw works

The Gateway is OpenClaw's long-lived control plane and source of truth. It owns sessions, routing, channel connections, model access, automations, approvals, and connected nodes. Control clients such as the CLI, browser UI, macOS app, and automation processes connect to it, normally over WebSocket. Mobile or headless nodes pair with the Gateway and expose explicitly declared capabilities.

| Layer | Responsibility |
| --- | --- |
| Channels and clients | Receive messages and display replies through WebChat, Telegram, WhatsApp, Slack, Discord, Signal, iMessage, native apps, and other plugins |
| Gateway | Authenticates clients, routes messages, owns shared state, schedules work, and coordinates nodes |
| Agent runtime | Builds context, calls the selected model, runs the tool loop, manages compaction, and records the session |
| Workspace and memory | Stores agent instructions, project files, skills, preferences, and retrievable state |
| Tools and skills | Provide controlled actions such as browser use, files, shell commands, search, APIs, and specialized workflows |
| Plugins | Add channels, model providers, tools, services, and integrations inside the Gateway process |
| Nodes and workers | Supply device or remote compute capabilities under pairing and policy controls |

Each configured agent can have its own workspace, authentication profiles, model registry, and SQLite-backed session history. Multi-agent routing maps accounts or conversations to the intended agent instead of mixing every sender into one shared history.

## The agent loop

When a message arrives, the Gateway identifies the channel, sender, conversation, and target agent. The runtime assembles instructions and relevant context, calls the configured model, and offers the tools allowed by policy. The model may answer directly or request one or more tool calls. OpenClaw executes approved calls, returns their results to the model, and repeats until it reaches a final response or a limit.

The model supplies probabilistic reasoning and language generation. OpenClaw supplies persistence, tool wiring, routing, permissions, retries, delivery, and user interfaces. Changing the model can change reasoning quality, but it does not replace the Gateway or turn OpenClaw into that provider's product.

## Channels, models, and local operation

OpenClaw can connect one Gateway to many chat surfaces. Telegram and WebChat ship with the core install, while most other channels are official plugins installed during onboarding or with the plugin manager. Channel authentication, reply threading, group policy, media support, and rate limits remain platform-specific.

Users can bring subscription-backed access, API credentials, gateways, or local models. Local model services can start on demand when a request selects them. A local model may improve privacy and cost control, but quality, tool calling, context limits, memory use, and hardware requirements vary.

“Self-hosted” means the Gateway and its state run on hardware the operator controls. It does **not** mean every byte stays offline. Messages go through configured chat platforms, and prompts or tool results sent to a hosted model provider leave the machine under that provider's terms. A fully local setup requires local models and locally controlled integrations as well as a local Gateway.

## Memory, sessions, and shared work

OpenClaw stores active agent history and session metadata in per-agent SQLite databases. Sessions provide durable conversation context and can be opened from multiple clients. Memory features can retrieve preferences, project context, or past information, but they are application-managed context rather than changes to the model's trained weights.

OpenClaw 2.0 expands sessions beyond one Gateway by allowing paired devices or cloud workers to run work with the associated workspace. Shared cloud sessions can bring another team member into live work or hand off a session without discarding its context. This is collaboration within a trust boundary, not automatic hostile-user isolation.

## Skills, plugins, and automations

**Skills** provide reusable instructions, scripts, and resources that teach an agent how to perform a workflow. **Plugins** run code inside the Gateway and can register tools, channels, model providers, and services. Because plugins execute in process and skills can guide powerful tools, both should be installed only from trusted, reviewed sources.

**Automations** schedule one-shot or recurring work through the Gateway. They can run in an existing or isolated session and deliver results to a channel, webhook, or nowhere. The Gateway must be running for schedules to fire. OpenClaw 2.0's scoped recurring approval can authorize an exact repeated operation while requiring another approval if the job changes.

## OpenClaw security model

OpenClaw can read files, browse authenticated websites, run commands, call APIs, and send messages, so its useful capabilities are also its security risk. The official security model assumes **one trusted operator boundary per Gateway**. It supports one person or a mutually trusting team, but it is not a hostile multi-tenant boundary for unrelated users sharing one Gateway.

Important controls include:

- Keep the Gateway on loopback or authenticated private access unless remote exposure is intentional.
- Use channel pairing, sender allowlists, group mention rules, and narrow operator scopes.
- Give each agent only the tools and credentials it needs.
- Run risky tool execution in a sandbox and require approvals for commands or irreversible actions.
- Treat messages, websites, retrieved documents, skills, and plugin output as untrusted content that can contain prompt injection.
- Use separate Gateways, operating-system users, or hosts for people who do not share a trust boundary.
- Review plugins as trusted code and pin exact versions where practical.
- Run `openclaw security audit`, and use `--deep` before exposing new surfaces.
- Protect the state directory because it can contain transcripts, tokens, channel credentials, configuration, and private data.

Sandboxing reduces blast radius but is not a perfect security boundary. A model can still misuse an allowed tool, disclose information available inside the sandbox, or persuade a user to approve a harmful action. Human approval is useful only when the request clearly states the exact operation and consequence.

## OpenClaw vs a chatbot or model

| Product type | Main role | How OpenClaw differs |
| --- | --- | --- |
| Language model | Generates text, reasoning traces, tool requests, or multimodal output | OpenClaw can switch among models and surrounds them with state, tools, channels, and policy |
| Hosted chatbot | Provider-operated interface tied primarily to that provider's service | OpenClaw is self-hosted, multi-provider, extensible, and reachable through many channels |
| Coding agent | Works mainly inside repositories and development tools | OpenClaw can host coding workflows but also spans communication, personal tasks, devices, media, and automation |
| Workflow automation tool | Runs predefined triggers and actions | OpenClaw can use an LLM to interpret context and choose tools, while still supporting deterministic schedules and approvals |
| MCP server | Exposes resources, prompts, or tools through a protocol | OpenClaw is a complete agent gateway and may consume or expose integrations rather than being only one MCP server |

## Who OpenClaw is for

OpenClaw suits developers, power users, families, and mutually trusted teams that want an assistant they can operate and extend themselves. It is attractive when one agent should work across chat apps, retain durable sessions, run local tools, schedule work, or use different model providers without moving the whole product to one vendor.

It is a poor fit when nobody can maintain the host, review permissions, secure credentials, monitor costs, or recover state. A managed assistant may be safer for users who do not need system-level tools or self-hosting.

## Bottom line

OpenClaw is the open-source gateway around an AI agent, not the intelligence model itself. OpenClaw 2.0, released as v2026.8.1, makes installation and the browser experience simpler while adding portable sessions, live collaboration, interactive outputs, protected credentials, scoped recurring approvals, and a large set of runtime and plugin migrations. Its power comes from connecting models to real systems, which is exactly why deployment boundaries, permissions, upgrades, and security audits matter.
$openclaw$::text,
    jsonb_build_array(
      jsonb_build_object('title', 'OpenClaw 2.0, Accidentally — official announcement', 'url', 'https://openclaw.ai/blog/openclaw-2-accidentally'),
      jsonb_build_object('title', 'OpenClaw v2026.8.1 release notes', 'url', 'https://docs.openclaw.ai/releases/2026.8.1'),
      jsonb_build_object('title', 'OpenClaw official documentation', 'url', 'https://docs.openclaw.ai/'),
      jsonb_build_object('title', 'OpenClaw Gateway architecture', 'url', 'https://docs.openclaw.ai/architecture'),
      jsonb_build_object('title', 'OpenClaw agent runtime', 'url', 'https://docs.openclaw.ai/concepts/agent'),
      jsonb_build_object('title', 'OpenClaw security guide', 'url', 'https://docs.openclaw.ai/gateway/security'),
      jsonb_build_object('title', 'OpenClaw Foundation', 'url', 'https://www.openclaw.org/'),
      jsonb_build_object('title', 'OpenClaw source repository', 'url', 'https://github.com/openclaw/openclaw')
    ),
    jsonb_build_object(
      'category', 'Agents & Workflows',
      'relatedTerms', jsonb_build_array('ai-agent', 'agent-harness', 'tool-calling', 'mcp', 'prompt-injection', 'moltbot', 'moltbook', 'openshell'),
      'analogy', 'OpenClaw is like a self-hosted switchboard and workshop for AI: one Gateway connects the model to conversations, memory, tools, devices, and scheduled work.',
      'seoDescription', 'OpenClaw is a self-hosted open-source AI agent gateway. Learn its architecture, channels, tools, security model, and what changed in OpenClaw 2.0.',
      'seoKeywords', jsonb_build_array('what is OpenClaw', 'OpenClaw 2.0', 'OpenClaw 2026.8.1', 'OpenClaw 2.0 features', 'OpenClaw 2.0 release notes', 'OpenClaw installation', 'OpenClaw upgrade', 'OpenClaw doctor fix', 'OpenClaw AI agent', 'OpenClaw Gateway architecture', 'OpenClaw security', 'OpenClaw self hosted', 'OpenClaw vs ChatGPT', 'OpenClaw ClawdBot MoltBot', 'OpenClaw automations', 'OpenClaw browser app')
    )
  )
)
UPDATE content_items AS item
SET
  title = glossary_updates.title,
  summary = glossary_updates.summary,
  body = glossary_updates.body,
  blocks = jsonb_build_array(
    jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', glossary_updates.body)
  ),
  sources = glossary_updates.sources,
  metadata = COALESCE(item.metadata, '{}'::jsonb) || glossary_updates.metadata,
  published_at = DATE '2026-09-01',
  updated_at = NOW()
FROM glossary_updates
WHERE item.kind = 'glossary'
  AND item.slug = glossary_updates.slug
  AND item.parent_slug = '';
