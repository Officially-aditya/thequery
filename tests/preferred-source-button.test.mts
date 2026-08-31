import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("preferred source integration is loaded once and used on article surfaces", async () => {
  const [layout, component, articleIndex, articlePage] = await Promise.all([
    readFile(path.join(root, "app/layout.tsx"), "utf8"),
    readFile(path.join(root, "components/PreferredSourceButton.tsx"), "utf8"),
    readFile(path.join(root, "app/articles/page.tsx"), "utf8"),
    readFile(path.join(root, "app/articles/[slug]/page.tsx"), "utf8"),
  ]);

  assert.match(layout, /https:\/\/news\.google\.com\/swg\/js\/v1\/publisher\.js/);
  assert.match(component, /google-add-preferred-source-btn/);
  assert.match(articleIndex, /<PreferredSourceButton/);
  assert.match(articlePage, /<PreferredSourceButton/);
});
