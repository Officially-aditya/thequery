import { NextResponse } from "next/server";
import { getAdminSession } from "@/lib/auth";
import { getContentCounts } from "@/lib/content";

export async function GET() {
  const session = await getAdminSession();
  if (!session) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  return NextResponse.json({ email: session.email, counts: await getContentCounts() });
}
