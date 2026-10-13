import assert from "node:assert/strict";
import test from "node:test";
import { GET } from "../app/sitemap.xml/route.ts";

test("sitemap.xml returns a valid sitemapindex containing all section sitemaps", async () => {
  const response = await GET();
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("Content-Type"), "application/xml; charset=utf-8");

  const xml = await response.text();
  assert.match(xml, /<sitemapindex xmlns="http:\/\/www\.sitemaps\.org\/schemas\/sitemap\/0\.9">/);
  assert.match(xml, /<loc>https:\/\/www\.thequery\.in\/sitemap\/pages\.xml<\/loc>/);
  assert.match(xml, /<loc>https:\/\/www\.thequery\.in\/sitemap\/books\.xml<\/loc>/);
  assert.match(xml, /<loc>https:\/\/www\.thequery\.in\/sitemap\/guides\.xml<\/loc>/);
  assert.match(xml, /<loc>https:\/\/www\.thequery\.in\/sitemap\/comparisons\.xml<\/loc>/);
  assert.match(xml, /<loc>https:\/\/www\.thequery\.in\/sitemap\/articles\.xml<\/loc>/);
  assert.match(xml, /<loc>https:\/\/www\.thequery\.in\/sitemap\/glossary\.xml<\/loc>/);
});
