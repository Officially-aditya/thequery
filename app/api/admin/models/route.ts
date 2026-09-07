import { NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getModelBySlug, getModelOptions } from "@/lib/models";

export async function GET(request: Request) {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const slug = new URL(request.url).searchParams.get("slug")?.trim();
    if (slug) {
      const model = await getModelBySlug(slug);
      if (!model) return NextResponse.json({ error: "Model not found." }, { status: 404 });
      return NextResponse.json(model, {
        headers: { "Cache-Control": "private, no-store" },
      });
    }

    return NextResponse.json(await getModelOptions(), {
      headers: { "Cache-Control": "private, max-age=300, stale-while-revalidate=300" },
    });
  } catch {
    return NextResponse.json({ error: "Unable to load model catalog." }, { status: 500 });
  }
}
