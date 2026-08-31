const PREFERRED_SOURCE_URL =
  "https://www.google.com/preferences/source?q=thequery.in";

type PreferredSourceButtonProps = {
  className?: string;
};

export default function PreferredSourceButton({
  className = "",
}: PreferredSourceButtonProps) {
  return (
    <a
      href={PREFERRED_SOURCE_URL}
      target="_blank"
      rel="noopener noreferrer"
      aria-label="Add TheQuery as a preferred source on Google"
      className={`mx-auto flex min-h-16 w-full max-w-[480px] items-center justify-center gap-4 rounded-[1.5rem] border-2 border-[#777] bg-white px-7 py-2 text-left font-sans text-lg font-medium leading-tight text-black shadow-sm transition-colors hover:bg-[#f5f5f5] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent sm:min-h-[68px] sm:text-xl ${className}`}
    >
      <svg
        aria-hidden="true"
        className="shrink-0"
        xmlns="http://www.w3.org/2000/svg"
        width="40"
        height="40"
        viewBox="0 0 48 48"
      >
        <path
          fill="#EA4335"
          d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.61 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.05 17.74 9.5 24 9.5z"
        />
        <path
          fill="#4285F4"
          d="M46.98 24.55c0-1.64-.15-3.22-.42-4.73H24v9.02h12.94c-.56 2.98-2.25 5.5-4.79 7.2v6h7.77c4.55-4.2 7.06-10.39 7.06-17.49z"
        />
        <path
          fill="#FBBC05"
          d="M10.54 28.59A14.4 14.4 0 0 1 9.5 24c0-1.59.36-3.13 1.04-4.59v-6.19H2.56A24 24 0 0 0 0 24c0 3.87.93 7.54 2.56 10.78l7.98-6.19z"
        />
        <path
          fill="#34A853"
          d="M24 48c6.47 0 11.9-2.13 15.87-5.96l-7.77-6c-2.15 1.44-4.9 2.3-8.1 2.3-6.26 0-11.57-3.55-13.46-8.59l-7.98 6.19C6.51 42.62 14.61 48 24 48z"
        />
      </svg>
      <span>
        Add as a preferred
        <br />
        source on Google
      </span>
    </a>
  );
}
