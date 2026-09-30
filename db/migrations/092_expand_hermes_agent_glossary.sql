-- Expand the Hermes Agent glossary entry with the subheadings it was missing.
--
-- The term already shipped in migration 060 (publish gpt6 sol/luna glossary,
-- which also seeded the rest of the catalogue) and is live at
-- /glossary/hermes-agent. This rewrites the body to add three sections:
--
--   1. "Hermes Agent skills"   — the SKILL.md format, the three-level
--      progressive-disclosure load, skill_manage, the write-approval gate,
--      the advisory linter, /learn, bundled-skill update protection, and
--      blueprints.
--   2. "Integrations" plus "### Telegram integration" — the three outward
--      directions (model providers, MCP, terminal backends), the gateway,
--      and then Telegram specifically: setup, the three authorization gates,
--      DM pairing, the admin/user slash-command split, group privacy mode,
--      voice, topics, MEDIA delivery, webhook mode, and the local Bot API
--      size limit.
--   3. "Hermes Agent vs OpenClaw" — renames the old "Hermes and OpenClaw"
--      and adds the real architectural difference. OpenClaw's own vision
--      document, as reported in the site's OpenClaw article, rejects agent
--      hierarchies and heavy orchestration in favour of simple serialized
--      architecture. Hermes is the opposite bet: an active orchestration
--      layer with isolated sub-agents and per-profile isolation.
--
-- Source-correction carried in the same migration:
-- The entry's "Official Repository" source pointed at
-- github.com/hermes-agent-org/hermes, a 5-star lookalike created April 14,
-- 2026 and last pushed April 15, 2026. The canonical repository is
-- github.com/NousResearch/hermes-agent, MIT licensed, whose latest release
-- at the time of writing is v0.21.5 (tag v2026.9.24, September 24, 2026).
-- The bad URL is replaced rather than kept alongside.
--
-- Attribution kept in the body so the rendered page does not overclaim:
-- The 140,000-stars-in-under-three-months and "most used agent on
-- OpenRouter" figures are NVIDIA's, published May 13, 2026, and are stated
-- as a dated snapshot rather than a current count.
--
-- The body carries no "## References & Resources" or "## Related Terms"
-- heading because app/glossary/[term]/page.tsx renders both from the
-- `references` and `relatedTerms` fields. Adding them here would duplicate.
--
-- All relatedTerms resolve in the live glossary except none: ai-agent,
-- agentic-ai, agent-harness, agent-orchestration, ai-memory, openclaw,
-- open-source, mcp, prompt-engineering and grok-4-6 are all live rows.
-- prompt-engineering is new here and is used by the skills section.

WITH hermes_agent_entry AS (
  SELECT $body$
Hermes Agent is an open-source AI agent and runtime from Nous Research. It combines a language model with a terminal, tools, persistent memory, a skills system, scheduled jobs, messaging gateways, and optional delegation to parallel sub-agents. The project is not a single model. Users can select providers such as Nous Portal, OpenRouter, OpenAI, Anthropic, xAI, or a local endpoint while keeping the same agent environment.

## What Hermes actually is

A normal chat session ends with its context. Hermes is designed to carry useful state forward. It can maintain a persona in SOUL.md, a user profile, long-term and daily memories, workspace instructions, and a searchable history of previous sessions. It can also turn a successful procedure into a reusable skill and improve that skill after later runs.

The phrase self-improving describes this workflow loop, not weight training. Hermes does not silently retrain the underlying model after every conversation. It records knowledge, preferences, procedures, and corrections in files and structured state that the agent can retrieve in future sessions. That distinction matters: the behavior can become more personalized without changing the provider's model weights.

Adoption has been fast. In a May 2026 post, NVIDIA reported that the project had passed 140,000 GitHub stars in under three months and was the most-used agent on OpenRouter at that point. Those are dated third-party figures rather than a live count. The repository is MIT licensed, and the most recent release at the time of writing is v0.21.5, tagged v2026.9.24 on September 24, 2026.

## Architecture and model choice

Hermes has two main entry points. The interactive CLI provides a terminal interface for conversation, tool output, interrupts, context compression, and model switching. The gateway lets users talk to the same agent through messaging platforms such as Telegram, Discord, Slack, WhatsApp, Signal, and email. A scheduled task can therefore run on a server and deliver its result to the channel where the user already works.

The runtime can use several execution back ends, including a local process, Docker, SSH, and supported persistent or serverless environments. It also supports MCP connections, cron scheduling, voice workflows, and isolated sub-agents for parallel work. This makes Hermes an [agent harness](/glossary/agent-harness) and gateway as much as a chat interface: the model supplies reasoning, while Hermes controls tools, state, permissions, and the execution loop.

Hermes also supports xAI Grok through a browser-based OAuth device-code flow for SuperGrok or X Premium accounts. Its official guide lists Grok 4.6 as the default model for that provider. In that setup, Hermes is the runtime and Grok is the model provider. Switching models changes the reasoning engine without requiring a new set of memories, skills, or messaging integrations.

## Hermes Agent skills

Skills are the part of Hermes that is most often misdescribed. They are not plugins, not [prompt engineering](/glossary/prompt-engineering) templates, and not weight updates. A skill is a markdown file with YAML frontmatter that tells the agent how to handle one class of task. Skills live under `~/.hermes/skills/`, and the format is compatible with the agentskills.io open standard so skills can move between tools.

A skill is a folder containing a required `SKILL.md` plus optional `references/`, `scripts/`, `templates/`, and `assets/` files. That directory is the source of truth. Bundled skills are copied in on install and on every `hermes update`, and a manifest records the content hash each one arrived with, so a skill you have edited is treated as user-modified and is never silently replaced by an update.

The design goal is token efficiency, which is what separates skills from [AI memory](/glossary/ai-memory). Memory is declarative and injected into every session, and it should stay small. Skills are procedural and loaded on demand, and they can be long. The house rule is simple: if it belongs on a sticky note it is memory, and if it belongs in a reference document it is a skill.

Loading is progressive, so a large installed catalogue does not bill itself into every conversation:

| Level | Call | What it costs |
| --- | --- | --- |
| 0 | `skills_list()` | Names, descriptions, and categories, about 3k tokens |
| 1 | `skill_view(name)` | The full SKILL.md for one skill |
| 2 | `skill_view(name, path)` | A single reference file inside that skill |

The agent writes these files itself through a `skill_manage` tool exposing create, patch, edit, delete, write_file, and remove_file. It is prompted to do so after a complex task succeeds, after it works through errors or dead ends, and after the user corrects its approach. A patch carries only the changed text, which is why it is the preferred update action over a full rewrite.

Three details are worth knowing before letting that run unattended. By default those writes are unapproved, so `skills.write_approval: true` stages every change under `~/.hermes/pending/skills/` and routes it through the same approve-or-deny flow as dangerous commands. An advisory linter flags three failure shapes without blocking anything: a body shaped like an incident log full of ticket numbers, a skill carrying more than 60 reference files, and a SKILL.md past roughly 24,000 characters. And `/learn` is the shortcut for turning a pile of source material into a skill without hand-writing the file, where the agent does the authoring and folds new material into an existing skill rather than duplicating it.

Skills can also be installed rather than written. The Skills Hub serves community and official modules that pass a security scan on install. A skill carrying a `blueprint` block with a schedule is registered as a suggested cron job rather than scheduled outright, so installing something can never quietly create a recurring task on your server.

## Integrations

Hermes reaches outward in three directions. Model access runs through a shared resolver that maps a provider and model to an API mode, credentials, and base URL, covering hosted providers, OAuth logins such as xAI Grok and GitHub Copilot, and your own OpenAI-compatible endpoint. Capability extension runs through MCP, with `hermes mcp` handling install, configuration, and OAuth 2.1 authentication. Execution placement runs through terminal back ends, which include a local process, Docker, SSH, Singularity, Modal, Daytona, and Vercel Sandbox, so the same agent can run against a laptop or against a serverless environment that hibernates when idle.

The messaging gateway is a fourth surface and the one most people actually use. One long-lived process bridges the same agent to Telegram, Discord, Slack, WhatsApp, Signal, email, and a long tail of other platforms, handling session routing, user authorization, slash command dispatch, cron ticking, and outbound delivery. That is what lets a scheduled job run unattended on a server and post its result into the chat where you already work.

### Telegram integration

Telegram is the most complete of these adapters and the one the documentation treats as the reference example. It is built on python-telegram-bot and handles text, voice, images, and file attachments.

Setup is short. You create a bot with @BotFather, then either run `hermes gateway setup` and let the wizard write the configuration, or set two values yourself in `~/.hermes/.env`:

```
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_ALLOWED_USERS=123456789
```

`hermes gateway start` runs it, and `hermes gateway install` makes it survive a reboot as a user service, with a `--system` flag on Linux for a boot-time system service.

Access control matters here because this bot has terminal access. The gateway denies every user who is neither allowlisted nor paired, and authorization is split across three gates that are easy to confuse:

- `TELEGRAM_ALLOWED_USERS` covers direct messages, groups, and forums
- `TELEGRAM_GROUP_ALLOWED_USERS` authorizes specific senders in groups only, and grants no direct message access
- `TELEGRAM_GROUP_ALLOWED_CHATS` authorizes every member of a listed group, so group membership itself is the credential

An allowlist needs IDs collected in advance, which does not scale to a team. DM pairing is the alternative. An unknown sender who messages the bot receives a one-time pairing code, and you approve it with `hermes pairing approve telegram XKGH5N7P`. Codes expire after an hour, are rate limited to one request per user per ten minutes, lock the platform out for an hour after five failed approval attempts, and are stored with owner-only permissions. Approval takes effect immediately, so adding a teammate does not mean restarting the gateway.

A second tier split controls what an allowed user can do once through the door. By default every allowed user can run every slash command. Adding `allow_admin_from` alongside `allow_from` keeps full command access for admins while restricting everyone else to a `user_allowed_commands` list, with `/help` and `/whoami` always permitted so a restricted user can still inspect their own access. Plain conversation is unaffected, so a non-admin can keep working normally.

Group behaviour is a separate problem, because Telegram's privacy mode is on by default and the bot can then only see commands, replies to itself, and service messages. You either turn privacy mode off in BotFather or promote the bot to group admin, and if you change the privacy setting you must remove and re-add the bot, because Telegram caches that setting when the bot joins. On top of that, `require_mention: true` makes the bot ignore ordinary group chatter until it is addressed, and `observe_unmentioned_group_messages: true` lets it read that chatter as shared context without dispatching on it. `guest_mode: true` is the looser option for casual groups, allowing the bot on an explicit mention only, with no session stickiness so it never carries into a thread it was not pinged into.

The parts that make it feel like a real assistant rather than a chat window:

- Voice memos are transcribed on arrival by a configurable provider, either faster-whisper running locally with no API key, Groq, or OpenAI. Setting `stt.enabled: false` passes the raw audio path to the agent instead, which is the hook for diarization or long-term archiving.
- Generated audio is delivered as native Telegram voice bubbles, with OpenAI and ElevenLabs producing Opus directly and the free Edge TTS option requiring ffmpeg for the conversion.
- `/sethome` designates the chat that scheduled jobs deliver into, and `TELEGRAM_CRON_THREAD_ID` routes those deliveries to a specific forum topic.
- Bot API 9.4 added private chat topics, so one direct message can hold several isolated sessions with their own history and context, and a topic can be configured to auto-load a named skill each time its session starts.
- Files come back out through `MEDIA:` tags in the agent's reply, covering images, audio, video, documents, office files, archives, and packages.

Two deployment settings change the bill. The default is long polling, where the gateway makes outbound calls to fetch updates, which means the host must stay awake. Setting `TELEGRAM_WEBHOOK_URL` flips the direction so Telegram pushes to your HTTPS endpoint and the machine can sleep between messages, and that path requires `TELEGRAM_WEBHOOK_SECRET` or the gateway refuses to start. Separately, the public Bot API caps downloads at 20MB, so anything larger needs a self-hosted telegram-bot-api server. That raises the ceiling to 2GB but must be bound to loopback, because it authenticates by putting the bot token in the URL path with nothing else.

## Hermes Agent vs OpenClaw

Hermes and [OpenClaw](/glossary/openclaw) are related runtimes, not two names for the same product. Both support persistent personal agents, tools, memory, messaging, and multi-step work. The difference is in their implementation, defaults, and operational choices.

The clearest statement of the split is [OpenClaw's own design philosophy](/articles/openclaw-had-210000-github-stars), which explicitly rejects agent-hierarchy frameworks and heavy orchestration layers in favour of simple, serialized, debuggable architecture. Hermes takes the opposite position on that specific question. It is an active orchestration layer, it delegates to isolated sub-agents with their own conversations and terminals, and it supports profiles so several fully isolated instances can run concurrently. OpenClaw optimizes for a system a human can reason about. Hermes optimizes for a system that accumulates state.

They also are not competing for the same user, because Hermes ships a first-class path out of OpenClaw. The command `hermes claw migrate` imports an OpenClaw persona, memories, skills, approval patterns, messaging settings, workspace instructions, and selected API keys, with a dry-run preview and an interactive guided migration available. That makes the realistic transition a migration rather than a rewrite, so the practical question is usually whether Hermes's provider choice and skills loop are worth the move, and the answer depends mostly on whether you want the runtime unbound from a single model provider.

Migration should still be reviewed before it is run. Importing credentials and permissions is a security decision, not a file conversion, and the imported approval patterns and API keys are precisely what determines what the agent can reach afterward.

## Hermes and Grok Bot

[Grok Bot](/articles/grok-bot-openclaw-hermes-agent-popularity) is a managed product that combines Grok with a hosted cloud computer, a desktop or mobile interface, and xAI and Cursor account controls. Hermes is an open runtime that users can operate on their own machine, a VPS, a container, or a supported cloud environment. Hermes can use Grok, but it does not become Grok Bot when it does. One is a vendor-managed experience, the other a configurable control plane.

That difference changes the tradeoff. Grok Bot minimizes setup and infrastructure work. Hermes offers provider choice, portable state, and more control over where the runtime and credentials live. A hosted agent may be easier to start, while a self-managed agent requires more work to patch, isolate, monitor, and recover.

## Applications

Hermes is useful when a task repeats, takes more than one step, or benefits from remembering how a particular user or team works. Common applications include:

- Daily briefings, recurring reports, backups, audits, and other scheduled automations
- Research agents that search, read documents, summarize findings, and deliver a report through a messaging channel
- Coding and operations work that needs a terminal, files, tests, remote hosts, or GitHub workflows
- Team assistants that answer from persistent project context and route requests to specialized skills
- Personal workflows that improve over time because the agent keeps preferences and successful procedures

The strongest use cases are not fully hands-off decisions. They are bounded workflows with clear inputs, reversible actions, and an obvious delivery channel. A daily report or a pull-request review is easier to validate than an agent with unrestricted authority over production systems.

## Safety and tradeoffs

Persistent memory and terminal access make Hermes more useful, but they also make mistakes durable. A misconfigured skill can run repeatedly. A leaked API key can grant access to every scheduled job. A prompt injection in a document, webpage, or message can influence later steps if the agent treats untrusted text as instructions.

The skills loop adds its own risk surface. A skill is an instruction file that the agent can rewrite without you, so a procedure that was correct once can drift after a bad run, and a background review that misjudges what it learned can encode the mistake permanently. This is what `skills.write_approval` and `memory.write_approval` exist to gate, and they are the two settings worth turning on first in any shared or production deployment.

Use command approvals, narrow toolsets, isolated environments, paired messaging accounts, backups, and explicit checkpoints for consequential actions. Review what gets written into memory and skills, because a remembered instruction can affect future tasks long after the original conversation is gone. Self-hosting improves control over data placement, but it does not automatically provide isolation or safe defaults.

## Why Hermes matters

Hermes represents a shift from disposable prompts to software that develops a working relationship with its user. Its model-agnostic design separates the agent's accumulated memory and tools from any one provider, while its gateway and scheduler turn an assistant into something that can operate across a day or a week. The important capability is not that Hermes acts without humans. It is that humans can define a bounded workflow once, keep the useful state, and decide exactly where review remains necessary.
$body$::text AS body
)
INSERT INTO content_items (
  id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
  cover_image_url, cover_image_alt, status, published_at, sort_order
)
SELECT
  'glossary:hermes-agent',
  'glossary',
  'hermes-agent',
  '',
  'glossary/hermes-agent',
  'Hermes Agent',
  'An open-source, multi-provider AI agent from Nous Research that learns from user workflows through persistent memory, self-improving skills, scheduled automation, and messaging gateways.',
  hermes_agent_entry.body,
  jsonb_build_array(jsonb_build_object('id', 'markdown-1', 'type', 'markdown', 'content', hermes_agent_entry.body)),
  jsonb_build_array(
    jsonb_build_object('title', 'Hermes Agent - Official Repository', 'url', 'https://github.com/NousResearch/hermes-agent'),
    jsonb_build_object('title', 'Hermes Agent - Skills System', 'url', 'https://hermes-agent.nousresearch.com/docs/user-guide/features/skills'),
    jsonb_build_object('title', 'Hermes Agent - Creating Skills', 'url', 'https://hermes-agent.nousresearch.com/docs/developer-guide/creating-skills'),
    jsonb_build_object('title', 'Hermes Agent - Telegram Setup', 'url', 'https://hermes-agent.nousresearch.com/docs/user-guide/messaging/telegram'),
    jsonb_build_object('title', 'Hermes Agent - Messaging Gateway', 'url', 'https://hermes-agent.nousresearch.com/docs/user-guide/messaging/'),
    jsonb_build_object('title', 'Hermes Agent - Team Telegram Assistant', 'url', 'https://hermes-agent.nousresearch.com/docs/guides/team-telegram-assistant'),
    jsonb_build_object('title', 'Hermes Agent - Migrate from OpenClaw', 'url', 'https://hermes-agent.nousresearch.com/docs/guides/migrate-from-openclaw'),
    jsonb_build_object('title', 'Hermes Agent - xAI Grok OAuth', 'url', 'https://hermes-agent.nousresearch.com/docs/guides/xai-grok-oauth'),
    jsonb_build_object('title', 'Hermes Unlocks Self-Improving AI Agents (NVIDIA)', 'url', 'https://blogs.nvidia.com/blog/rtx-ai-garage-hermes-agent-dgx-spark/')
  ),
  jsonb_build_object(
    'category', 'Agents & Workflows',
    'relatedTerms', jsonb_build_array('ai-agent', 'agentic-ai', 'agent-harness', 'agent-orchestration', 'ai-memory', 'prompt-engineering', 'openclaw', 'open-source', 'mcp', 'grok-4-6'),
    'seoDescription', 'Hermes Agent explained: Nous Research''s self-improving open-source runtime, its skills system, Telegram gateway setup, and how it differs from OpenClaw.',
    'seoKeywords', jsonb_build_array('what is Hermes Agent', 'Hermes Agent Nous Research', 'Hermes Agent vs OpenClaw', 'Hermes Agent skills', 'Hermes Agent Telegram integration', 'hermes claw migrate', 'Hermes Agent gateway', 'open-source AI agent', 'AI agent with persistent memory', 'self-improving AI agent', 'AI agent skills system', 'Telegram AI agent', 'Hermes Agent Grok')
  ),
  NULL,
  NULL,
  'published',
  DATE '2026-08-20',
  0
FROM hermes_agent_entry
WHERE true
ON CONFLICT (kind, slug, parent_slug) DO UPDATE SET
  path = EXCLUDED.path,
  title = EXCLUDED.title,
  summary = EXCLUDED.summary,
  body = EXCLUDED.body,
  blocks = EXCLUDED.blocks,
  sources = EXCLUDED.sources,
  metadata = EXCLUDED.metadata,
  status = EXCLUDED.status,
  published_at = EXCLUDED.published_at,
  sort_order = EXCLUDED.sort_order,
  updated_at = NOW();

-- Backlink: OpenClaw's relatedTerms never pointed at Hermes Agent, so the
-- comparison was one-directional. Appended idempotently, same shape as the
-- GPT-6.1 Sol backlinks in migration 090.
UPDATE content_items
SET
  metadata = CASE
    WHEN COALESCE(metadata->'relatedTerms', '[]'::jsonb) @> '["hermes-agent"]'::jsonb
      THEN metadata
    ELSE jsonb_set(
      COALESCE(metadata, '{}'::jsonb),
      '{relatedTerms}',
      COALESCE(metadata->'relatedTerms', '[]'::jsonb) || jsonb_build_array('hermes-agent'),
      true
    )
  END,
  updated_at = NOW()
WHERE kind = 'glossary'
  AND slug = 'openclaw'
  AND parent_slug = '';
