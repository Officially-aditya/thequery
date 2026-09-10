import { NextRequest, NextResponse } from "next/server";
import { revalidatePath, revalidateTag } from "next/cache";
import { isAuthenticated } from "@/lib/auth";
import { deleteContentItem, getContentItem, getContentItems, getContentSummaries, upsertContent } from "@/lib/content";
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
  } else if (kind === "comparison") {
    revalidatePath("/comparisons");
    revalidatePath(`/comparisons/${slug}`);
    // Every comparison page renders the shared authored-pair picker.
    revalidatePath("/comparisons/[slug]", "page");
  } else if (kind === "glossary") {
    revalidatePath("/glossary");
    revalidatePath(`/glossary/${slug}`);
    revalidatePath("/ai-word-of-the-day");
    // Glossary auto-linking is embedded into these generated page families.
    revalidatePath("/articles/[slug]", "page");
    revalidatePath("/guides/[slug]", "page");
    revalidatePath("/books/[slug]/[chapter]", "page");
    revalidatePath("/comparisons/[slug]", "page");
  } else if (kind === "book") {
    revalidatePath("/books");
    revalidatePath(`/books/${slug}`);
    revalidatePath("/research");
    // Chapter pages include their parent book title/author/metadata.
    revalidatePath("/books/[slug]/[chapter]", "page");
  } else if (parentSlug) {
    // Chapter changes affect the book index count, parent table of contents,
    // the chapter itself, research book listings, and the sitemap.
    revalidatePath("/books");
    revalidatePath(`/books/${parentSlug}`);
    revalidatePath(`/books/${parentSlug}/${slug}`);
    revalidatePath("/research");
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
  const slug = req.nextUrl.searchParams.get("slug");
  if (slug) {
    const item = await getContentItem(kind, slug, parentSlug, true);
    return item ? NextResponse.json(item) : NextResponse.json({ error: "Content not found" }, { status: 404 });
  }
  if (req.nextUrl.searchParams.get("summary") === "1") {
    return NextResponse.json(await getContentSummaries(kind, { parentSlug, includeDrafts: true }));
  }
  if (req.nextUrl.searchParams.get("full") === "1") {
    return NextResponse.json(await getContentItems(kind, { parentSlug, includeDrafts: true }));
  }
  return NextResponse.json(await getContentSummaries(kind, { parentSlug, includeDrafts: true }));
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
