import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const glossary = JSON.parse(
  await readFile(new URL("../data/glossary.json", import.meta.url), "utf8"),
);

test("technological singularity entry has the required reader and SEO content", () => {
  const matches = glossary.filter(
    ({ slug }) => slug === "technological-singularity",
  );

  assert.equal(matches.length, 1);
  const [entry] = matches;
  assert.match(entry.fullDef, /## Technological singularity vs AI singularity/);
  assert.match(entry.fullDef, /## Technological singularity timeline/);
  assert.ok(entry.seoDescription.length >= 140);
  assert.ok(entry.seoDescription.length <= 160);
  assert.ok(entry.relatedTerms.includes("agi"));
  assert.ok(entry.references.length >= 4);
});
