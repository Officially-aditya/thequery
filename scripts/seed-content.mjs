import { randomBytes, scryptSync } from "node:crypto";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { neon } from "@neondatabase/serverless";
import nextEnv from "@next/env";
import { extractSources, markdownBlock } from "../lib/content-utils.ts";

const projectRoot = fileURLToPath(new URL("..", import.meta.url));

const { loadEnvConfig } = nextEnv;
loadEnvConfig(projectRoot);

function getSql() {
  if (!process.env.DATABASE_URL) {
    throw new Error("DATABASE_URL is required. Add your Neon connection string to .env.");
  }
  return neon(process.env.DATABASE_URL);
}

function contentPath(kind, slug, parentSlug = "") {
  if (kind === "chapter") return `books/${parentSlug}/${slug}`;
  return `${kind === "glossary" ? "glossary" : `${kind}s`}/${slug}`;
}

function stripFrontmatter(content) {
  if (!content.startsWith("---")) return content;
  const closingIndex = content.indexOf("\n---", 3);
  return closingIndex < 0 ? content : content.slice(closingIndex + 4).replace(/^\n/, "");
}

async function upsertContent(sql, item) {
  await sql.query(
    `INSERT INTO content_items (
      id, kind, slug, parent_slug, path, title, summary, body, blocks, sources, metadata,
      cover_image_url, cover_image_alt, status, published_at, sort_order
    ) VALUES (
      $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb, $12, $13, $14, $15, $16
    ) ON CONFLICT (kind, slug, parent_slug) DO UPDATE SET
      title = EXCLUDED.title,
      summary = EXCLUDED.summary,
      body = EXCLUDED.body,
      blocks = EXCLUDED.blocks,
      sources = EXCLUDED.sources,
      metadata = EXCLUDED.metadata,
      cover_image_url = COALESCE(EXCLUDED.cover_image_url, content_items.cover_image_url),
      cover_image_alt = CASE
        WHEN EXCLUDED.cover_image_url IS NULL THEN content_items.cover_image_alt
        ELSE EXCLUDED.cover_image_alt
      END,
      status = EXCLUDED.status,
      published_at = EXCLUDED.published_at,
      sort_order = EXCLUDED.sort_order,
      path = EXCLUDED.path,
      updated_at = NOW()`,
    [
      `seed:${item.path}`,
      item.kind,
      item.slug,
      item.parentSlug ?? "",
      item.path,
      item.title,
      item.summary ?? "",
      item.body ?? "",
      JSON.stringify(item.blocks ?? []),
      JSON.stringify(item.sources ?? []),
      JSON.stringify(item.metadata ?? {}),
      item.coverImageUrl ?? null,
      item.coverImageAlt ?? null,
      "published",
      item.publishedAt ?? null,
      item.sortOrder ?? 0,
    ],
  );
}

async function syncConfiguredAdmin(sql) {
  const email = process.env.ADMIN_USER?.trim().toLowerCase();
  const password = process.env.ADMIN_PASSWORD;
  if (!email || !password) {
    console.warn("Skipped admin account: set ADMIN_USER and ADMIN_PASSWORD before running this command.");
    return false;
  }

  const salt = randomBytes(16).toString("hex");
  const passwordHash = scryptSync(password, salt, 64).toString("hex");
  await sql.query(
    `INSERT INTO admin_users (id, email, password_hash, password_salt)
     VALUES ($1, $2, $3, $4)
     ON CONFLICT (email) DO UPDATE SET
       password_hash = EXCLUDED.password_hash,
       password_salt = EXCLUDED.password_salt,
       updated_at = NOW()`,
    [`admin:${email}`, email, passwordHash, salt],
  );
  return true;
}

export async function seedContent() {
  const sql = getSql();
  const [articles, glossary, guides] = await Promise.all(
    ["articles", "glossary", "guides"].map(async (name) =>
      JSON.parse(await readFile(path.join(projectRoot, "data", `${name}.json`), "utf8")),
    ),
  );

  let imported = 0;
  for (const article of articles) {
    const { content, sources } = extractSources(article.content);
    await upsertContent(sql, {
      kind: "article",
      slug: article.slug,
      path: contentPath("article", article.slug),
      title: article.title,
      summary: article.summary,
      body: content,
      blocks: [markdownBlock(content)],
      sources,
      metadata: { manualGlossaryLinks: Boolean(article.manualGlossaryLinks) },
      publishedAt: article.date,
    });
    imported += 1;
  }

  for (const guide of guides) {
    const { content, sources } = extractSources(guide.content);
    await upsertContent(sql, {
      kind: "guide",
      slug: guide.slug,
      path: contentPath("guide", guide.slug),
      title: guide.title,
      summary: guide.summary,
      body: content,
      blocks: [markdownBlock(content)],
      sources,
      metadata: {},
      publishedAt: guide.date,
    });
    imported += 1;
  }

  for (const term of glossary) {
    await upsertContent(sql, {
      kind: "glossary",
      slug: term.slug,
      path: contentPath("glossary", term.slug),
      title: term.name,
      summary: term.shortDef,
      body: term.fullDef,
      blocks: [markdownBlock(term.fullDef)],
      sources: term.references ?? [],
      metadata: {
        category: term.category,
        relatedTerms: term.relatedTerms ?? [],
        analogy: term.analogy,
        seoDescription: term.seoDescription,
        seoKeywords: term.seoKeywords ?? [],
      },
      publishedAt: term.lastUpdated,
    });
    imported += 1;
  }

  const booksDirectory = path.join(projectRoot, "content", "books");
  const bookDirectories = await readdir(booksDirectory, { withFileTypes: true });
  for (const directory of bookDirectories.filter((entry) => entry.isDirectory())) {
    const bookDirectory = path.join(booksDirectory, directory.name);
    const book = JSON.parse(await readFile(path.join(bookDirectory, "meta.json"), "utf8"));
    await upsertContent(sql, {
      kind: "book",
      slug: book.slug,
      path: contentPath("book", book.slug),
      title: book.title,
      summary: book.description,
      metadata: { author: book.author, lastModified: book.lastModified },
      publishedAt: book.lastModified ?? null,
    });
    imported += 1;

    for (const [index, chapter] of book.chapters.entries()) {
      const rawContent = await readFile(path.join(bookDirectory, chapter.file), "utf8");
      const body = stripFrontmatter(rawContent);
      await upsertContent(sql, {
        kind: "chapter",
        slug: chapter.slug,
        parentSlug: book.slug,
        path: contentPath("chapter", chapter.slug, book.slug),
        title: chapter.title,
        body,
        blocks: [markdownBlock(body)],
        metadata: { lastModified: chapter.lastModified },
        publishedAt: chapter.lastModified ?? book.lastModified ?? null,
        sortOrder: index,
      });
      imported += 1;
    }
  }

  const hasAdmin = await syncConfiguredAdmin(sql);
  console.log(`Imported or updated ${imported} content records${hasAdmin ? " and the configured admin account" : ""}.`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  seedContent().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
