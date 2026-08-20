import "server-only";

import { createHash, randomBytes, scrypt as scryptCallback, timingSafeEqual } from "crypto";
import { promisify } from "util";
import { cookies } from "next/headers";
import { isConfiguredAdminEmail, normalizeAdminEmail } from "./admin-auth-config";
import { getSql } from "./db";

const SESSION_COOKIE = "thequery_admin_session";
const SESSION_DURATION_SECONDS = 60 * 60 * 24;
const scrypt = promisify(scryptCallback);

interface AdminUserRow {
  id: string;
  email: string;
  password_hash: string;
  password_salt: string;
}

export interface AdminSession {
  userId: string;
  email: string;
}

interface ConfiguredAdminCredentials {
  email: string;
  password: string;
}

function tokenHash(token: string): string {
  return createHash("sha256").update(token).digest("hex");
}

async function passwordHash(password: string, salt: string): Promise<string> {
  const derived = await scrypt(password, salt, 64) as Buffer;
  return Buffer.from(derived).toString("hex");
}

function safeHashMatch(actual: string, expected: string): boolean {
  const actualBuffer = Buffer.from(actual, "hex");
  const expectedBuffer = Buffer.from(expected, "hex");
  return actualBuffer.length === expectedBuffer.length && timingSafeEqual(actualBuffer, expectedBuffer);
}

/**
 * The environment variables provision the single database account. Once it
 * exists, sign-in verifies the submitted password against the stored salted
 * hash without rewriting the account on every request.
 */
function getConfiguredAdminCredentials(): ConfiguredAdminCredentials {
  const email = normalizeAdminEmail(process.env.ADMIN_USER);
  const password = process.env.ADMIN_PASSWORD;
  if (!email || !password) {
    throw new Error("ADMIN_USER and ADMIN_PASSWORD must be configured for admin access.");
  }

  return { email, password };
}

async function getOrProvisionConfiguredAdminUser(
  credentials: ConfiguredAdminCredentials,
): Promise<AdminUserRow | null> {
  const sql = getSql();
  const rows = await sql`
    SELECT id, email, password_hash, password_salt
    FROM admin_users
    WHERE email = ${credentials.email}
    LIMIT 1
  `;
  const existingUser = (rows as AdminUserRow[])[0];
  if (existingUser) return existingUser;

  const salt = randomBytes(16).toString("hex");
  const hash = await passwordHash(credentials.password, salt);
  const insertedRows = await sql`
    INSERT INTO admin_users (id, email, password_hash, password_salt)
    VALUES (${`admin:${credentials.email}`}, ${credentials.email}, ${hash}, ${salt})
    ON CONFLICT (email) DO NOTHING
    RETURNING id, email, password_hash, password_salt
  `;
  const insertedUser = (insertedRows as AdminUserRow[])[0];
  if (insertedUser) return insertedUser;

  const retryRows = await sql`
    SELECT id, email, password_hash, password_salt
    FROM admin_users
    WHERE email = ${credentials.email}
    LIMIT 1
  `;
  return (retryRows as AdminUserRow[])[0] ?? null;
}

export async function signIn(email: string, password: string): Promise<boolean> {
  const credentials = getConfiguredAdminCredentials();
  if (!password || !isConfiguredAdminEmail(email, credentials.email)) return false;

  const user = await getOrProvisionConfiguredAdminUser(credentials);
  if (!user) return false;

  const candidate = await passwordHash(password, user.password_salt);
  if (!safeHashMatch(candidate, user.password_hash)) return false;

  const sql = getSql();
  const token = randomBytes(32).toString("base64url");
  const expiresAt = new Date(Date.now() + SESSION_DURATION_SECONDS * 1000).toISOString();
  await sql`
    INSERT INTO admin_sessions (id, admin_user_id, token_hash, expires_at)
    VALUES (${randomBytes(16).toString("hex")}, ${user.id}, ${tokenHash(token)}, ${expiresAt})
  `;

  const cookieStore = await cookies();
  cookieStore.set(SESSION_COOKIE, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: SESSION_DURATION_SECONDS,
    path: "/",
  });
  return true;
}

export async function getAdminSession(): Promise<AdminSession | null> {
  const cookieStore = await cookies();
  const token = cookieStore.get(SESSION_COOKIE)?.value;
  if (!token) return null;

  const sql = getSql();
  const rows = await sql`
    SELECT admin_users.id AS user_id, admin_users.email
    FROM admin_sessions
    JOIN admin_users ON admin_users.id = admin_sessions.admin_user_id
    WHERE admin_sessions.token_hash = ${tokenHash(token)}
      AND admin_sessions.expires_at > NOW()
    LIMIT 1
  `;
  const row = (rows as Array<{ user_id: string; email: string }>)[0];
  return row && isConfiguredAdminEmail(row.email, process.env.ADMIN_USER)
    ? { userId: row.user_id, email: row.email }
    : null;
}

export async function isAuthenticated(): Promise<boolean> {
  return Boolean(await getAdminSession());
}

export async function destroySession(): Promise<void> {
  const cookieStore = await cookies();
  const token = cookieStore.get(SESSION_COOKIE)?.value;
  cookieStore.delete(SESSION_COOKIE);
  if (!token) return;

  const sql = getSql();
  await sql`DELETE FROM admin_sessions WHERE token_hash = ${tokenHash(token)}`;
}
