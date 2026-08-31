const PREFERRED_SOURCE_URL =
  "https://www.google.com/preferences/source?q=thequery.in";

type PreferredSourceButtonProps = {
  className?: string;
};

export default function PreferredSourceButton({
  className = "",
}: PreferredSourceButtonProps) {
  return (
    <section
      aria-labelledby="preferred-source-title"
      className={`flex flex-col gap-4 border-y border-border py-5 sm:flex-row sm:items-center sm:justify-between ${className}`}
    >
      <div className="max-w-xl">
        <h2
          id="preferred-source-title"
          className="font-serif text-lg font-semibold text-text-primary"
        >
          Prefer TheQuery in Google Search?
        </h2>
        <p className="mt-1 text-sm leading-relaxed text-text-secondary">
          Add TheQuery as a preferred source to make our AI coverage easier to find.
        </p>
      </div>
      <a
        href={PREFERRED_SOURCE_URL}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-2 rounded-md bg-accent px-4 py-2 text-sm font-medium text-bg-primary transition-colors hover:bg-accent-hover focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
      >
        Add TheQuery as a Preferred Source
        <svg
          aria-hidden="true"
          xmlns="http://www.w3.org/2000/svg"
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M14 3h7v7" />
          <path d="M10 14 21 3" />
          <path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5" />
        </svg>
      </a>
    </section>
  );
}
