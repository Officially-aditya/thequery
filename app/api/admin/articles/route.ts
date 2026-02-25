import { NextRequest, NextResponse } from "next/server";
import { isAuthenticated } from "@/lib/auth";
import { getAllIssues, saveAllIssues, type Article } from "@/lib/articles";

async function checkAuth() {
  if (!(await isAuthenticated())) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  return null;
}

export async function GET() {
  const authErr = await checkAuth();
  if (authErr) return authErr;
  return NextResponse.json(getAllIssues());
}

export async function POST(req: NextRequest) {
  const authErr = await checkAuth();
  if (authErr) return authErr;

  const issue: Article = await req.json();
  const issues = getAllIssues();

  const existing = issues.findIndex((i) => i.slug === issue.slug);
  if (existing >= 0) {
    issues[existing] = issue;
  } else {
    issues.push(issue);
  }

  saveAllIssues(issues);
  return NextResponse.json({ success: true });
}

export async function DELETE(req: NextRequest) {
  const authErr = await checkAuth();
  if (authErr) return authErr;

  const { slug } = await req.json();
  const issues = getAllIssues().filter((i) => i.slug !== slug);
  saveAllIssues(issues);
  return NextResponse.json({ success: true });
}
