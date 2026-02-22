import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getAllTerms, saveAllTerms, type GlossaryTerm } from "@/lib/glossary";

async function checkAuth() {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  return null;
}

export async function GET() {
  const authErr = await checkAuth();
  if (authErr) return authErr;
  return NextResponse.json(getAllTerms());
}

export async function POST(req: NextRequest) {
  const authErr = await checkAuth();
  if (authErr) return authErr;

  const term: GlossaryTerm = await req.json();
  const terms = getAllTerms();

  const existing = terms.findIndex((t) => t.slug === term.slug);
  if (existing >= 0) {
    terms[existing] = term;
  } else {
    terms.push(term);
  }

  saveAllTerms(terms);
  return NextResponse.json({ success: true });
}

export async function DELETE(req: NextRequest) {
  const authErr = await checkAuth();
  if (authErr) return authErr;

  const { slug } = await req.json();
  const terms = getAllTerms().filter((t) => t.slug !== slug);
  saveAllTerms(terms);
  return NextResponse.json({ success: true });
}
