import { getAllTermSummaries } from "@/lib/glossary";
import GlossarySearch from "@/components/GlossarySearch";
import { createOpenGraphMetadata, SITE_URL } from "@/lib/site";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AI Glossary",
  description: "Clear, concise definitions of key AI and machine learning terms. Search and browse by category.",
  openGraph: createOpenGraphMetadata({
    title: "AI Glossary - TheQuery",
    description: "AI and ML terms explained clearly - from backpropagation to knowledge graphs.",
    url: `${SITE_URL}/glossary`,
  }),
};

export const revalidate = 300;

export default async function GlossaryPage() {
  const terms = await getAllTermSummaries();
  const clientTerms = terms.map(({ name, slug, shortDef, category }) => ({
    name,
    slug,
    shortDef,
    category,
  }));

  return (
    <div className="max-w-[960px] mx-auto px-4 py-12">
      <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">AI Glossary</h1>
      <p className="text-text-secondary mb-8">
        Key terms and concepts in artificial intelligence and machine learning.
      </p>
      <GlossarySearch terms={clientTerms} />
    </div>
  );
}
