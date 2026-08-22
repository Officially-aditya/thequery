import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { neon } from "@neondatabase/serverless";
import nextEnv from "@next/env";

const projectRoot = fileURLToPath(new URL("..", import.meta.url));
const { loadEnvConfig } = nextEnv;
loadEnvConfig(projectRoot);

if (!process.argv.includes("--confirm")) {
  throw new Error("Refusing to copy data without the --confirm flag.");
}

const sourceUrl = process.env.DATABASE_URL;
const targetUrl = process.env.NEW_DATABASE_URL;
if (!sourceUrl || !targetUrl) {
  throw new Error("DATABASE_URL and NEW_DATABASE_URL are both required.");
}

const sourceHost = new URL(sourceUrl).hostname;
const targetHost = new URL(targetUrl).hostname;
if (sourceHost === targetHost) {
  throw new Error("Source and target must be different Neon projects.");
}

const source = neon(sourceUrl);
const target = neon(targetUrl);

function splitStatements(sql) {
  return sql
    .split(/;\s*(?:\r?\n|$)/)
    .map((statement) => statement.trim())
    .filter(Boolean);
}

function asCount(row) {
  return Number(row?.count ?? 0);
}

function asTimestamp(value) {
  return value instanceof Date ? value.toISOString() : value;
}

async function applyMigrations() {
  const migrations = [
    ["001_initial", "001_initial.sql"],
    ["002_add_cover_images", "002_add_cover_images.sql"],
  ];
  for (const [id, file] of migrations) {
    const sqlSource = await readFile(path.join(projectRoot, "db", "migrations", file), "utf8");
    for (const statement of splitStatements(sqlSource)) {
      await target.query(statement);
    }
    await target.query(
      "INSERT INTO schema_migrations (id) VALUES ($1) ON CONFLICT (id) DO NOTHING",
      [id],
    );
  }
}

async function copyAdminUsers() {
  const users = await source`
    SELECT id, email, password_hash, password_salt, created_at, updated_at
    FROM admin_users
    ORDER BY email
  `;

  for (const user of users) {
    await target.query(
      `INSERT INTO admin_users (id, email, password_hash, password_salt, created_at, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6)
       ON CONFLICT (id) DO UPDATE SET
         email = EXCLUDED.email,
         password_hash = EXCLUDED.password_hash,
         password_salt = EXCLUDED.password_salt,
         updated_at = EXCLUDED.updated_at`,
      [user.id, user.email, user.password_hash, user.password_salt, asTimestamp(user.created_at), asTimestamp(user.updated_at)],
    );
  }

  return users.length;
}

async function copyContent() {
  const items = await source`
    SELECT id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
      cover_image_url, cover_image_alt, status, published_at, sort_order, created_at, updated_at
    FROM content_items
    ORDER BY kind, parent_slug, sort_order, slug
  `;

  for (const item of items) {
    await target.query(
      `INSERT INTO content_items (
        id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
        cover_image_url, cover_image_alt, status, published_at, sort_order, created_at, updated_at
      ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb,
        $12, $13, $14, $15, $16, $17, $18
      )
      ON CONFLICT (id) DO UPDATE SET
        kind = EXCLUDED.kind,
        slug = EXCLUDED.slug,
        parent_slug = EXCLUDED.parent_slug,
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
        updated_at = EXCLUDED.updated_at`,
      [
        item.id,
        item.kind,
        item.slug,
        item.parent_slug,
        item.path,
        item.title,
        item.summary,
        item.body,
        JSON.stringify(item.blocks ?? []),
        JSON.stringify(item.sources ?? []),
        JSON.stringify(item.metadata ?? {}),
        item.cover_image_url,
        item.cover_image_alt,
        item.status,
        item.published_at,
        item.sort_order,
        asTimestamp(item.created_at),
        asTimestamp(item.updated_at),
      ],
    );
  }

  return items.length;
}

async function getContentCounts(sql) {
  const rows = await sql`
    SELECT kind, COUNT(*)::int AS count
    FROM content_items
    GROUP BY kind
    ORDER BY kind
  `;
  return Object.fromEntries(rows.map((row) => [row.kind, asCount(row)]));
}

async function run() {
  await applyMigrations();

  const existing = await target`
    SELECT
      (SELECT COUNT(*)::int FROM admin_users) AS admin_users,
      (SELECT COUNT(*)::int FROM content_items) AS content_items
  `;
  if (Number(existing[0]?.admin_users ?? 0) > 0 || Number(existing[0]?.content_items ?? 0) > 0) {
    throw new Error("Target already contains data; refusing to overwrite it.");
  }

  const [adminUserCount, contentCount] = await Promise.all([copyAdminUsers(), copyContent()]);
  const [sourceCounts, targetCounts] = await Promise.all([getContentCounts(source), getContentCounts(target)]);
  if (JSON.stringify(sourceCounts) !== JSON.stringify(targetCounts)) {
    throw new Error(`Content count mismatch. Source: ${JSON.stringify(sourceCounts)} Target: ${JSON.stringify(targetCounts)}`);
  }

  console.log(`Copied ${contentCount} content records and ${adminUserCount} admin users.`);
  console.log(`Verified content counts: ${JSON.stringify(targetCounts)}.`);
  console.log("Admin sessions were intentionally not copied; sign in again after switching production.");
}

run().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
