import { createHash } from "node:crypto";
import { getSql } from "./db.ts";

export async function consumeAdminLoginAttempt(
  headers: Headers,
  sql = getSql(),
): Promise<number | null> {
  // Vercel sets this header at its edge. Caller-supplied Cloudflare and
  // X-Forwarded-For headers must not create fresh rate-limit buckets.
  const address = process.env.VERCEL === "1"
    ? headers.get("x-vercel-forwarded-for")?.trim()
    : null;
  const key = createHash("sha256").update(`admin-login:${address || "unknown"}`).digest("hex");

  await sql.query("DELETE FROM admin_login_rate_limits WHERE expires_at <= NOW()");
  const rows = await sql.query(`
    INSERT INTO admin_login_rate_limits (key_hash, attempts, expires_at)
    VALUES ($1, 1, NOW() + INTERVAL '15 minutes')
    ON CONFLICT (key_hash) DO UPDATE SET
      attempts = CASE
        WHEN admin_login_rate_limits.expires_at <= NOW() THEN 1
        ELSE LEAST(admin_login_rate_limits.attempts + 1, 6)
      END,
      expires_at = CASE
        WHEN admin_login_rate_limits.expires_at <= NOW() THEN NOW() + INTERVAL '15 minutes'
        ELSE admin_login_rate_limits.expires_at
      END
    RETURNING attempts,
      GREATEST(1, CEIL(EXTRACT(EPOCH FROM (expires_at - NOW()))))::integer AS retry_after
  `, [key]);
  const row = (rows as Array<{ attempts: number; retry_after: number }>)[0];
  if (!row) throw new Error("Unable to record admin login attempt.");

  return row.attempts > 5 ? row.retry_after : null;
}
