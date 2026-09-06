import { NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getModels } from "@/lib/models";

export async function GET() {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    return NextResponse.json(await getModels());
  } catch {
    return NextResponse.json({ error: "Unable to load model catalog." }, { status: 500 });
  }
}
