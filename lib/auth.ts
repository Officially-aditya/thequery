import { cookies } from "next/headers";
import crypto from "crypto";

const SESSION_COOKIE = "admin_session";
const SESSION_DURATION = 60 * 60 * 24; // 24 hours in seconds

function getSessionToken(): string {
  const password = process.env.ADMIN_PASSWORD || "admin";
  return crypto.createHash("sha256").update(`session:${password}`).digest("hex");
}

export function verifyPassword(password: string): boolean {
  const expected = process.env.ADMIN_PASSWORD || "admin";
  return password === expected;
}

export async function createSession(): Promise<string> {
  const token = getSessionToken();
  const cookieStore = await cookies();
  cookieStore.set(SESSION_COOKIE, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    maxAge: SESSION_DURATION,
    path: "/",
  });
  return token;
}

export async function isAuthenticated(): Promise<boolean> {
  const cookieStore = await cookies();
  const session = cookieStore.get(SESSION_COOKIE);
  if (!session) return false;
  return session.value === getSessionToken();
}

export async function destroySession(): Promise<void> {
  const cookieStore = await cookies();
  cookieStore.delete(SESSION_COOKIE);
}
