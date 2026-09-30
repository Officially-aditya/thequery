import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

const migrationPath = path.join(root, "db/migrations/092_expand_hermes_agent_glossary.sql");

interface GlossaryEntry {
  name: string;
  slug: string;
  shortDef: string;
  fullDef: string;
  category: string;
  relatedTerms: string[];
  references: { title: string; url: string }[];
  seoDescription: string;
  seoKeywords: string[];
  lastUpdated: string;
  analogy?: string;
}

const migration = await readFile(migrationPath, "utf8");
const glossary = JSON.parse(
  await readFile(path.join(root, "data/glossary.json"), "utf8"),
) as GlossaryEntry[];
const entry = glossary.find((term) => term.slug === "hermes-agent");

function migrationBody(): string {
  const start = migration.indexOf("$body$") + "$body$".length;
  const end = migration.indexOf("$body$", start);
  assert.ok(start > "$body$".length - 1, "migration is missing its $body$ block");
  assert.ok(end > start, "migration $body$ block is not closed");
  return migration.slice(start, end);
}

test("Hermes Agent migration is registered and upserts the existing glossary row", async () => {
  const runner = await readFile(path.join(root, "scripts/migrate.mjs"), "utf8");

  assert.match(runner, /092_expand_hermes_agent_glossary/);
  assert.match(migration, /'glossary:hermes-agent'/);
  assert.match(migration, /'glossary\/hermes-agent'/);
  assert.match(migration, /ON CONFLICT \(kind, slug, parent_slug\) DO UPDATE SET/);
  assert.match(migration, /updated_at = NOW\(\)/);
  assert.match(migration, /'category', 'Agents & Workflows'/);
});

test("Hermes Agent migration preserves the original publish date", () => {
  // The term shipped in migration 060 and is already live. This is a content
  // revision, not a re-publish, so published_at must not move to today.
  assert.ok(entry);
  assert.match(migration, /DATE '2026-08-20'/);
  assert.ok(!migration.includes("DATE '2026-09-30'"));
  assert.equal(entry.lastUpdated, "2026-08-20");
});

test("Hermes Agent migration keeps the statement splitter intact", () => {
  const statements = migration
    .split(/;\s*(?:\r?\n|$)/)
    .map((statement) => statement.trim())
    .filter(Boolean);

  assert.equal(statements.length, 2);
  assert.match(statements[0], /INSERT INTO content_items/);
  assert.match(statements[1], /UPDATE content_items/);
  assert.ok(!statements[1].includes("SELECT $body$"));
});

test("the seed body matches the migration body once trimmed", () => {
  assert.ok(entry);
  // The migration wraps its body in a dollar-quoted block, so the literal
  // carries the newline after $body$ and the one before the closing tag. The
  // seed stores the trimmed text, matching tests/boosting-merge-glossary.test.mts.
  assert.equal(entry.fullDef, migrationBody().trim());
  assert.equal(entry.fullDef, entry.fullDef.trim());
});

test("Hermes Agent entry carries the three requested subheadings", () => {
  assert.ok(entry);

  assert.match(entry.fullDef, /^## Hermes Agent vs OpenClaw$/m);
  assert.match(entry.fullDef, /^## Hermes Agent skills$/m);
  assert.match(entry.fullDef, /^## Integrations$/m);
  assert.match(entry.fullDef, /^### Telegram integration$/m);

  // The old heading is renamed, not duplicated.
  assert.ok(!entry.fullDef.includes("## Hermes and OpenClaw"));
  assert.equal(entry.fullDef.match(/^## /gm)?.length, 9);
  assert.equal(entry.fullDef.match(/^### /gm)?.length, 1);
});

test("Hermes Agent skills section explains the format, loading, and gating", () => {
  assert.ok(entry);
  const section = entry.fullDef.slice(
    entry.fullDef.indexOf("## Hermes Agent skills"),
    entry.fullDef.indexOf("## Integrations"),
  );

  assert.match(section, /not weight updates/);
  assert.match(section, /agentskills\.io open standard/);
  assert.match(section, /`~\/\.hermes\/skills\/`/);
  assert.match(section, /required `SKILL\.md`/);
  assert.match(section, /skills_list\(\)/);
  assert.match(section, /skill_view\(name\)/);
  assert.match(section, /skill_view\(name, path\)/);
  assert.match(section, /skill_manage/);
  assert.match(section, /skills\.write_approval: true/);
  assert.match(section, /~\/\.hermes\/pending\/skills\//);
  assert.match(section, /more than 60 reference files/);
  assert.match(section, /`\/learn`/);
  assert.match(section, /blueprint/);
  // The skills/memory split is the conceptual point of the section.
  assert.match(section, /\[AI memory\]\(\/glossary\/ai-memory\)/);
  assert.match(section, /sticky note/);
});

test("Hermes Agent integrations section covers the three outward directions", () => {
  assert.ok(entry);
  const section = entry.fullDef.slice(
    entry.fullDef.indexOf("## Integrations"),
    entry.fullDef.indexOf("## Hermes Agent vs OpenClaw"),
  );

  assert.match(section, /`hermes mcp`/);
  assert.match(section, /OAuth 2\.1/);
  assert.match(section, /Docker, SSH, Singularity, Modal, Daytona, and Vercel Sandbox/);
  assert.match(section, /The messaging gateway is a fourth surface/);
  assert.match(section, /Telegram, Discord, Slack, WhatsApp, Signal, email/);
});

test("Telegram section documents setup, authorization, and the deployment limits", () => {
  assert.ok(entry);
  const section = entry.fullDef.slice(
    entry.fullDef.indexOf("### Telegram integration"),
    entry.fullDef.indexOf("## Hermes Agent vs OpenClaw"),
  );

  assert.match(section, /python-telegram-bot/);
  assert.match(section, /@BotFather/);
  assert.match(section, /`hermes gateway setup`/);
  assert.match(section, /TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ/);
  assert.match(section, /TELEGRAM_ALLOWED_USERS=123456789/);
  assert.match(section, /`hermes gateway install`/);

  // Three distinct authorization gates, not one.
  assert.match(section, /`TELEGRAM_ALLOWED_USERS` covers direct messages, groups, and forums/);
  assert.match(section, /`TELEGRAM_GROUP_ALLOWED_USERS` authorizes specific senders in groups only/);
  assert.match(section, /`TELEGRAM_GROUP_ALLOWED_CHATS` authorizes every member/);

  assert.match(section, /`hermes pairing approve telegram XKGH5N7P`/);
  assert.match(section, /expire after an hour/);
  assert.match(section, /`allow_admin_from`/);
  assert.match(section, /`user_allowed_commands`/);
  assert.match(section, /\/help` and `\/whoami` always permitted/);

  assert.match(section, /privacy mode is on by default/);
  assert.match(section, /remove and re-add the bot/);
  assert.match(section, /`require_mention: true`/);
  assert.match(section, /`guest_mode: true`/);

  assert.match(section, /faster-whisper/);
  assert.match(section, /`stt\.enabled: false`/);
  assert.match(section, /`\/sethome`/);
  assert.match(section, /TELEGRAM_CRON_THREAD_ID/);
  assert.match(section, /Bot API 9\.4 added private chat topics/);
  assert.match(section, /`MEDIA:` tags/);

  assert.match(section, /`TELEGRAM_WEBHOOK_URL`/);
  assert.match(section, /`TELEGRAM_WEBHOOK_SECRET` or the gateway refuses to start/);
  assert.match(section, /caps downloads at 20MB/);
  assert.match(section, /ceiling to 2GB but must be bound to loopback/);
});

test("Hermes Agent vs OpenClaw section states the real architectural split", () => {
  assert.ok(entry);
  const section = entry.fullDef.slice(
    entry.fullDef.indexOf("## Hermes Agent vs OpenClaw"),
    entry.fullDef.indexOf("## Hermes and Grok Bot"),
  );

  assert.match(section, /\[OpenClaw\]\(\/glossary\/openclaw\)/);
  assert.match(section, /rejects agent-hierarchy frameworks and heavy orchestration layers/);
  assert.match(section, /delegates to isolated sub-agents/);
  assert.match(section, /supports profiles/);
  assert.match(section, /OpenClaw optimizes for a system a human can reason about/);
  assert.match(section, /`hermes claw migrate`/);
  assert.match(section, /Importing credentials and permissions is a security decision/);
});

test("the entry keeps the site's existing sections and moves Applications after the comparisons", () => {
  assert.ok(entry);

  assert.match(entry.fullDef, /^## What Hermes actually is$/m);
  assert.match(entry.fullDef, /^## Architecture and model choice$/m);
  assert.match(entry.fullDef, /^## Hermes and Grok Bot$/m);
  assert.match(entry.fullDef, /^## Applications$/m);
  assert.match(entry.fullDef, /^## Safety and tradeoffs$/m);
  assert.match(entry.fullDef, /^## Why Hermes matters$/m);

  // Applications used to sit before the comparison sections. It reads better
  // after them, and the safety section now covers the skills write gate.
  assert.ok(
    entry.fullDef.indexOf("## Applications") > entry.fullDef.indexOf("## Hermes and Grok Bot"),
    "Applications should follow the comparison sections",
  );
  assert.match(entry.fullDef, /`skills\.write_approval` and `memory\.write_approval`/);
});

test("dated third-party adoption figures are attributed, not stated as current", () => {
  assert.ok(entry);

  assert.match(entry.fullDef, /In a May 2026 post, NVIDIA reported/);
  assert.match(entry.fullDef, /140,000 GitHub stars in under three months/);
  assert.match(entry.fullDef, /most-used agent on OpenRouter at that point/);
  assert.match(entry.fullDef, /dated third-party figures rather than a live count/);
  assert.match(entry.fullDef, /v0\.21\.5, tagged v2026\.9\.24 on September 24, 2026/);
  assert.match(entry.fullDef, /MIT licensed/);
});

test("the official repository source is corrected and the lookalike is gone", () => {
  assert.ok(entry);

  const repo = entry.references.find((ref) => ref.title === "Hermes Agent - Official Repository");
  assert.ok(repo, "official repository source is missing");
  assert.equal(repo.url, "https://github.com/NousResearch/hermes-agent");

  // github.com/hermes-agent-org/hermes is a 5-star lookalike created
  // 2026-04-14 and last pushed 2026-04-15. It must not resurface.
  assert.ok(!JSON.stringify(entry.references).includes("hermes-agent-org"));

  const titles = entry.references.map((ref) => ref.title);
  assert.deepEqual(titles, [
    "Hermes Agent - Official Repository",
    "Hermes Agent - Skills System",
    "Hermes Agent - Creating Skills",
    "Hermes Agent - Telegram Setup",
    "Hermes Agent - Messaging Gateway",
    "Hermes Agent - Team Telegram Assistant",
    "Hermes Agent - Migrate from OpenClaw",
    "Hermes Agent - xAI Grok OAuth",
    "Hermes Unlocks Self-Improving AI Agents (NVIDIA)",
  ]);
  assert.equal(entry.references.length, 9);
});

test("the OpenClaw backlink is appended idempotently", () => {
  assert.match(migration, /AND slug = 'openclaw'/);
  assert.match(
    migration,
    /COALESCE\(metadata->'relatedTerms', '\[\]'::jsonb\) @> '\["hermes-agent"\]'::jsonb/,
  );
});

test("Hermes Agent entry has no inline title or duplicated reference sections", () => {
  assert.ok(entry);

  assert.ok(!/^# /.test(entry.fullDef));
  assert.ok(!entry.fullDef.includes("## References"));
  assert.ok(!entry.fullDef.includes("## Related Terms"));
  assert.ok(!entry.fullDef.includes("https://www.thequery.in/glossary/"));
  assert.equal(entry.fullDef.match(/^\| ---/gm)?.length, 1);
  assert.equal(entry.fullDef.match(/^\| /gm)?.length, 5);
});

test("Hermes Agent entry keeps the summary, category, and SEO fields", () => {
  assert.ok(entry);

  assert.equal(entry.name, "Hermes Agent");
  assert.equal(entry.category, "Agents & Workflows");
  assert.match(
    entry.shortDef,
    /open-source, multi-provider AI agent from Nous Research/,
  );
  assert.ok(entry.seoDescription.length >= 140);
  assert.ok(entry.seoDescription.length <= 160);
  assert.ok(entry.seoKeywords.includes("Hermes Agent skills"));
  assert.ok(entry.seoKeywords.includes("Hermes Agent Telegram integration"));
  assert.ok(entry.seoKeywords.includes("Hermes Agent vs OpenClaw"));
  assert.equal(entry.seoKeywords.length, 13);
});

test("every Hermes Agent related term resolves in the seeded glossary", () => {
  assert.ok(entry);
  const slugs = new Set(glossary.map((term) => term.slug));

  for (const slug of entry.relatedTerms) {
    assert.ok(slugs.has(slug), `unresolved related term: ${slug}`);
  }
  // prompt-engineering is the one this revision adds.
  assert.ok(entry.relatedTerms.includes("prompt-engineering"));
});
