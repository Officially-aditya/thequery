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

test("API and MCP entries cover their core implementation details", () => {
  const expectations = [
    {
      slug: "api",
      headings: [
        "## The contract an API exposes",
        "## How a web API request works",
        "## Production API design",
      ],
      references: 3,
    },
    {
      slug: "mcp",
      headings: [
        "## The host-client-server model",
        "## Resources, prompts, and tools",
        "## Transports and versions",
        "## Security and trust boundaries",
      ],
      references: 5,
    },
  ];

  for (const expectation of expectations) {
    const matches = glossary.filter(({ slug }) => slug === expectation.slug);
    assert.equal(matches.length, 1);
    const [entry] = matches;

    assert.ok(entry.fullDef.length >= 4000);
    for (const heading of expectation.headings) {
      assert.match(entry.fullDef, new RegExp(heading));
    }
    assert.ok(entry.references.length >= expectation.references);
    assert.ok(entry.seoDescription.length >= 140);
    assert.ok(entry.seoDescription.length <= 160);
    assert.equal(entry.lastUpdated, "2026-08-28");
  }
});
