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
      <div google-add-preferred-source-btn data-theme="light" />
    </section>
  );
}
