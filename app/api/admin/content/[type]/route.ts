import { NextRequest, NextResponse } from "next/server";
import { revalidatePath, revalidateTag } from "next/cache";
import { isAuthenticated } from "@/lib/auth";
import { deleteContentItem, getContentItems, upsertContent } from "@/lib/content";
import { isContentKind, parseContentInput } from "@/lib/content-validation";
import type { ContentKind } from "@/lib/content-types";

interface RouteContext {
  params: Promise<{ type: string }>;
}

async function getKind(context: RouteContext): Promise<ContentKind | null> {
  const { type } = await context.params;
  return isContentKind(type) ? type : null;
}

async function requireAdmin() {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  return null;
}

function revalidateContent(kind: ContentKind, slug: string, parentSlug?: string | null) {
  revalidateTag(`content:${kind}`, { expire: 0 });
  revalidatePath("/");
  revalidatePath("/sitemap.xml");
  if (kind === "article") {
    revalidatePath("/articles");
    revalidatePath(`/articles/${slug}`);
  } else if (kind === "guide") {
    revalidatePath("/guides");
    revalidatePath(`/guides/${slug}`);
  } else if (kind === "glossary") {
    revalidatePath("/glossary");
    revalidatePath(`/glossary/${slug}`);
    revalidatePath("/ai-word-of-the-day");
  } else if (kind === "book") {
    revalidatePath("/books");
    revalidatePath(`/books/${slug}`);
  } else if (parentSlug) {
    revalidatePath(`/books/${parentSlug}`);
    revalidatePath(`/books/${parentSlug}/${slug}`);
  }
}

export async function GET(req: NextRequest, context: RouteContext) {
  const authError = await requireAdmin();
  if (authError) return authError;
  const kind = await getKind(context);
  if (!kind) return NextResponse.json({ error: "Unknown content type" }, { status: 404 });

  const parentSlug = req.nextUrl.searchParams.get("parentSlug");
  if (kind === "chapter" && !parentSlug) {
    return NextResponse.json({ error: "parentSlug is required for chapters" }, { status: 400 });
  }
  return NextResponse.json(await getContentItems(kind, { parentSlug, includeDrafts: true }));
}

export async function POST(req: NextRequest, context: RouteContext) {
  const authError = await requireAdmin();
  if (authError) return authError;
  const kind = await getKind(context);
  if (!kind) return NextResponse.json({ error: "Unknown content type" }, { status: 404 });

  try {
    const parsed = parseContentInput(kind, await req.json());
    if (!parsed.data) return NextResponse.json({ errors: parsed.errors }, { status: 422 });
    const item = await upsertContent(parsed.data);
    revalidateContent(item.kind, item.slug, item.parentSlug);
    return NextResponse.json(item);
  } catch {
    return NextResponse.json({ error: "Unable to save content." }, { status: 500 });
  }
}

export async function DELETE(req: NextRequest, context: RouteContext) {
  const authError = await requireAdmin();
  if (authError) return authError;
  const kind = await getKind(context);
  if (!kind) return NextResponse.json({ error: "Unknown content type" }, { status: 404 });

  try {
    const body = await req.json();
    const slug = typeof body.slug === "string" ? body.slug : "";
    const parentSlug = typeof body.parentSlug === "string" ? body.parentSlug : null;
    if (!slug) return NextResponse.json({ error: "slug is required" }, { status: 422 });
    await deleteContentItem(kind, slug, parentSlug);
    revalidateContent(kind, slug, parentSlug);
    return NextResponse.json({ success: true });
  } catch {
    return NextResponse.json({ error: "Unable to delete content." }, { status: 500 });
  }
}
