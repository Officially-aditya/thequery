import Link from "next/link";
import { getTodaysWord } from "@/lib/ai-word-of-the-day";

export default function WordOfTheDay() {
  const wotd = getTodaysWord();
  if (!wotd) return null;

  return (
    <Link
      href="/ai-word-of-the-day"
      className="block border border-border rounded-xl px-6 py-5 hover:border-accent transition-colors group"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium uppercase tracking-wider text-text-muted">
          AI Word of the Day
        </span>
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="text-text-muted group-hover:text-accent transition-colors"
        >
          <line x1="5" y1="12" x2="19" y2="12" />
          <polyline points="12 5 19 12 12 19" />
        </svg>
      </div>
      <h3 className="font-serif text-lg font-semibold text-text-primary mb-1">
        {wotd.term.name}
      </h3>
      <p className="text-sm text-text-secondary leading-relaxed line-clamp-2">
        {wotd.term.shortDef}
      </p>
    </Link>
  );
}
