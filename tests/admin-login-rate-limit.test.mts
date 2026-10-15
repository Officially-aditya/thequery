import assert from "node:assert/strict";
import test from "node:test";
import { consumeAdminLoginAttempt } from "../lib/admin-login-rate-limit.ts";

type Database = NonNullable<Parameters<typeof consumeAdminLoginAttempt>[1]>;

function database(attempts: number, keys: unknown[] = []) {
  return {
    async query(_query: string, params?: unknown[]) {
      if (!params) return [];
      keys.push(params[0]);
      return [{ attempts, retry_after: 897 }];
    },
  } as unknown as Database;
}

test("five attempts are allowed and subsequent attempts return the remaining wait", async () => {
  const headers = new Headers();
  assert.equal(await consumeAdminLoginAttempt(headers, database(1)), null);
  assert.equal(await consumeAdminLoginAttempt(headers, database(5)), null);
  assert.equal(await consumeAdminLoginAttempt(headers, database(6)), 897);
});

test("login attempts fail closed when the database cannot record the attempt", async () => {
  const unavailable = { query: async () => { throw new Error("Database unavailable"); } } as unknown as Database;
  await assert.rejects(consumeAdminLoginAttempt(new Headers(), unavailable), /Database unavailable/);

  const empty = { query: async () => [] } as unknown as Database;
  await assert.rejects(consumeAdminLoginAttempt(new Headers(), empty), /Unable to record/);
});

test("Vercel's trusted address determines the bucket and spoofed proxy headers do not", async (t) => {
  const previous = process.env.VERCEL;
  process.env.VERCEL = "1";
  t.after(() => {
    if (previous === undefined) delete process.env.VERCEL;
    else process.env.VERCEL = previous;
  });

  const keys: unknown[] = [];
  const sql = database(1, keys);
  for (const spoofed of ["198.51.100.1", "198.51.100.2"]) {
    await consumeAdminLoginAttempt(new Headers({
      "x-vercel-forwarded-for": "203.0.113.1",
      "x-forwarded-for": spoofed,
      "cf-connecting-ip": spoofed,
    }), sql);
  }
  await consumeAdminLoginAttempt(new Headers({ "x-vercel-forwarded-for": "203.0.113.2" }), sql);

  assert.equal(keys[0], keys[1]);
  assert.notEqual(keys[0], keys[2]);
  assert.match(String(keys[0]), /^[a-f0-9]{64}$/);
  assert.ok(!String(keys[0]).includes("203.0.113.1"));
});

test("missing trusted deployment metadata uses one shared bucket", async (t) => {
  const previous = process.env.VERCEL;
  delete process.env.VERCEL;
  t.after(() => {
    if (previous !== undefined) process.env.VERCEL = previous;
  });

  const keys: unknown[] = [];
  const sql = database(1, keys);
  await consumeAdminLoginAttempt(new Headers(), sql);
  await consumeAdminLoginAttempt(new Headers({ "x-vercel-forwarded-for": "203.0.113.1" }), sql);
  assert.equal(keys[0], keys[1]);
});
