import { neon } from "@neondatabase/serverless";

let client: ReturnType<typeof neon> | null = null;

export function getSql() {
  const databaseUrl = process.env.NEW_DATABASE_URL;
  if (!databaseUrl) {
    throw new Error("NEW_DATABASE_URL is required to access TheQuery content.");
  }

  if (!client) client = neon(databaseUrl);
  return client;
}
