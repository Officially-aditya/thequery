import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { neon } from "@neondatabase/serverless";
import nextEnv from "@next/env";
import { buildGlossaryLinks } from "../lib/glossary-link-map.mjs";
import { normalizeBlocks } from "../lib/content-utils.ts";

export async function generateGlossaryLinks() {
  nextEnv.loadEnvConfig(fileURLToPath(new URL("..", import.meta.url)));
  if (!process.env.NEW_DATABASE_URL) throw new Error("NEW_DATABASE_URL is required.");
  const sql = neon(process.env.NEW_DATABASE_URL);
  const rows = await sql`
    SELECT kind, slug, parent_slug AS "parentSlug", title, body, blocks
    FROM content_items
    WHERE status = 'published' AND kind IN ('glossary', 'book', 'chapter', 'guide')
    ORDER BY title ASC
  `;
  return buildGlossaryLinks({
    terms: rows.filter((row) => row.kind === "glossary").map((row) => ({ name: row.title, slug: row.slug })),
    books: rows.filter((row) => row.kind === "book"),
    chapters: rows.filter((row) => row.kind === "chapter"),
    guides: rows.filter((row) => row.kind === "guide").map((row) => ({
      ...row,
      blocks: normalizeBlocks(typeof row.blocks === "string" ? JSON.parse(row.blocks) : row.blocks),
    })),
  });
}

async function main() {
  const output = await generateGlossaryLinks();
  const file = new URL("../data/glossary-links.json", import.meta.url);
  if (process.argv.includes("--check")) {
    const snapshot = JSON.parse(await readFile(file, "utf8"));
    if (JSON.stringify(snapshot) !== JSON.stringify(output)) {
      throw new Error("Glossary link snapshot is stale. Run npm run glossary:links.");
    }
  } else {
    await writeFile(file, `${JSON.stringify(output, null, 2)}\n`, "utf8");
  }
  console.log(JSON.stringify({ sources: output.sources, coverage: output.coverage }));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
