import { getAllTerms } from "@/lib/glossary";
import GlossarySearch from "@/components/GlossarySearch";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AI Glossary",
  description: "Clear, concise definitions of key AI and machine learning terms. Search and browse by category.",
  openGraph: {
    title: "AI Glossary - TheQuery",
    description: "100+ AI and ML terms explained clearly - from backpropagation to knowledge graphs.",
  },
};

export default function GlossaryPage() {
  const terms = getAllTerms().sort((a, b) => a.name.localeCompare(b.name));
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
