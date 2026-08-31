import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("preferred source CTA is scoped to article pages above the global footer", async () => {
  const [layout, component, footer, articleIndex, articlePage] = await Promise.all([
    readFile(path.join(root, "app/layout.tsx"), "utf8"),
    readFile(path.join(root, "components/PreferredSourceButton.tsx"), "utf8"),
    readFile(path.join(root, "components/Footer.tsx"), "utf8"),
    readFile(path.join(root, "app/articles/page.tsx"), "utf8"),
    readFile(path.join(root, "app/articles/[slug]/page.tsx"), "utf8"),
  ]);

  assert.doesNotMatch(layout, /publisher\.js/);
  assert.match(component, /https:\/\/www\.google\.com\/preferences\/source\?q=www\.thequery\.in/);
  assert.match(component, /Prefer TheQuery in Google Search\?/);
  assert.match(component, /make our AI coverage easier to find/);
  assert.match(component, /Add TheQuery as a Preferred Source/);
  assert.doesNotMatch(footer, /PreferredSourceButton/);
  assert.match(articleIndex, /<PreferredSourceButton className="mt-12" \/>/);
  assert.match(articlePage, /<PreferredSourceButton className="mx-auto mt-12 max-w-\[720px\]" \/>/);
});
