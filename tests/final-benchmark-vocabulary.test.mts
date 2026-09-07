import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");
const source = (file: string) => readFile(path.join(root, file), "utf8");

test("final catalog benchmark labels are exposed in generated and authored comparison templates", async () => {
  const [generated, admin] = await Promise.all([
    source("lib/model-comparison.ts"),
    source("components/admin/admin-client.ts"),
  ]);

  for (const label of [
    "Multi-SWE-Bench",
    "NL2Repo",
    "VIBE-Pro",
    "ApexBench",
    "Arena Search",
    "τ²-bench Telecom",
  ]) {
    assert.ok(generated.includes(label), `${label} should be visible in generated comparisons`);
    assert.ok(admin.includes(label), `${label} should be available in the admin comparison template`);
  }
});
