import assert from "node:assert/strict";
import test from "node:test";
import {
  canonicalComparisonModels,
  canonicalComparisonSlug,
  publicModelSlug,
  resolveComparisonSlug,
} from "../lib/model-comparison-route.ts";

test("Claude comparison URLs use the product family instead of the vendor prefix", () => {
  assert.equal(publicModelSlug("claude-fable-5-1"), "fable-5-1");
  assert.equal(
    canonicalComparisonSlug("claude-fable-5-1", "gpt-5-4-pro"),
    "fable-5-1-vs-gpt-5-4-pro",
  );
});

test("model pair order has one canonical URL", () => {
  const forward = canonicalComparisonSlug("claude-fable-5-1", "gpt-5-4-pro");
  const reverse = canonicalComparisonSlug("gpt-5-4-pro", "claude-fable-5-1");

  assert.equal(forward, reverse);
  assert.deepEqual(
    canonicalComparisonModels("gpt-5-4-pro", "claude-fable-5-1"),
    ["claude-fable-5-1", "gpt-5-4-pro"],
  );
});

test("canonical pair slugs resolve back to internal model slugs", () => {
  assert.deepEqual(
    resolveComparisonSlug(
      "fable-5-1-vs-gpt-5-4-pro",
      ["claude-fable-5-1", "gpt-5-4-pro", "gemini-3-8-flash"],
    ),
    ["claude-fable-5-1", "gpt-5-4-pro"],
  );
});
