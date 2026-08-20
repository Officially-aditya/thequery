import "server-only";

import { createHash, randomBytes, scrypt as scryptCallback, timingSafeEqual } from "crypto";
import { promisify } from "util";
import { cookies } from "next/headers";
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

function normalizeEmail(value: string): string {
  return value.trim().toLowerCase();
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
 * The environment variables only provision the single database account. The
 * database stores a salted password hash; plaintext credentials never enter
 * application code or session cookies.
 */
async function syncConfiguredAdminUser(): Promise<string> {
  const email = normalizeEmail(process.env.ADMIN_USER ?? "");
  const password = process.env.ADMIN_PASSWORD;
  if (!email || !password) {
    throw new Error("ADMIN_USER and ADMIN_PASSWORD must be configured for admin access.");
  }

  const salt = randomBytes(16).toString("hex");
  const hash = await passwordHash(password, salt);
  const sql = getSql();
  await sql`
    INSERT INTO admin_users (id, email, password_hash, password_salt)
    VALUES (${`admin:${email}`}, ${email}, ${hash}, ${salt})
    ON CONFLICT (email) DO UPDATE SET
      password_hash = EXCLUDED.password_hash,
      password_salt = EXCLUDED.password_salt,
      updated_at = NOW()
  `;
  return email;
}

export async function signIn(email: string, password: string): Promise<boolean> {
  const configuredEmail = await syncConfiguredAdminUser();
  const normalizedEmail = normalizeEmail(email);
  if (!normalizedEmail || !password || normalizedEmail !== configuredEmail) return false;

  const sql = getSql();
  const rows = await sql`
    SELECT id, email, password_hash, password_salt
    FROM admin_users
    WHERE email = ${normalizedEmail}
    LIMIT 1
  `;
  const user = (rows as AdminUserRow[])[0];
  if (!user) return false;

  const candidate = await passwordHash(password, user.password_salt);
  if (!safeHashMatch(candidate, user.password_hash)) return false;

  const token = randomBytes(32).toString("base64url");
  const expiresAt = new Date(Date.now() + SESSION_DURATION_SECONDS * 1000).toISOString();
  await sql`DELETE FROM admin_sessions WHERE expires_at <= NOW()`;
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
  const configuredEmail = normalizeEmail(process.env.ADMIN_USER ?? "");
  return row && configuredEmail && row.email === configuredEmail
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
