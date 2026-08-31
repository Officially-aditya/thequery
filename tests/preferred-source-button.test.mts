import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("preferred source badge is scoped to article pages above the global footer", async () => {
  const [layout, component, footer, articleIndex, articlePage] = await Promise.all([
    readFile(path.join(root, "app/layout.tsx"), "utf8"),
    readFile(path.join(root, "components/PreferredSourceButton.tsx"), "utf8"),
    readFile(path.join(root, "components/Footer.tsx"), "utf8"),
    readFile(path.join(root, "app/articles/page.tsx"), "utf8"),
    readFile(path.join(root, "app/articles/[slug]/page.tsx"), "utf8"),
  ]);

  assert.doesNotMatch(layout, /publisher\.js/);
  assert.match(component, /https:\/\/www\.google\.com\/preferences\/source\?q=thequery\.in/);
  assert.doesNotMatch(component, /Prefer TheQuery in Google Search\?/);
  assert.doesNotMatch(component, /make our AI coverage easier to find/);
  assert.match(component, /Add as a preferred/);
  assert.match(component, /source on Google/);
  assert.match(component, /min-h-16/);
  assert.match(component, /max-w-\[480px\]/);
  assert.match(component, /viewBox="0 0 48 48"/);
  assert.doesNotMatch(footer, /PreferredSourceButton/);
  assert.match(articleIndex, /<PreferredSourceButton className="mt-12" \/>/);
  assert.match(articlePage, /<PreferredSourceButton className="mt-12" \/>/);
});
