import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description: "Privacy policy for TheQuery - how we handle your data.",
  openGraph: {
    title: "Privacy Policy - TheQuery",
    description: "Privacy policy for TheQuery - how we handle your data.",
  },
};

export default function PrivacyPage() {
  return (
    <div className="max-w-[720px] mx-auto px-4 py-12">
      <h1 className="font-serif text-3xl font-bold text-text-primary mb-2">
        Privacy Policy
      </h1>
      <p className="text-sm text-text-muted mb-8">
        Last updated: March 10, 2026
      </p>

      <div className="space-y-6 text-text-secondary leading-relaxed">
        <h2 className="font-serif text-xl font-semibold text-text-primary">
          Overview
        </h2>
        <p>
          TheQuery (thequery.in) is an educational website about artificial
          intelligence. We are committed to protecting your privacy and being
          transparent about what data we collect.
        </p>

        <h2 className="font-serif text-xl font-semibold text-text-primary">
          Information We Collect
        </h2>
        <p>
          We use Google Analytics and Vercel Analytics to collect anonymous usage
          data, including pages visited, time on site, browser type, and
          approximate geographic location. This data helps us understand how our
          content is used and improve the site.
        </p>
        <p>
          We do not collect personal information such as names, email addresses,
          or payment details through the website. If you contact us via email,
          your message and email address are used solely to respond to your
          inquiry.
        </p>

        <h2 className="font-serif text-xl font-semibold text-text-primary">
          Cookies
        </h2>
        <p>
          We use cookies set by Google Analytics to distinguish unique users and
          track sessions. We also store your theme preference (light or dark
          mode) in your browser&apos;s local storage. No advertising or tracking
          cookies from third parties are used.
        </p>

        <h2 className="font-serif text-xl font-semibold text-text-primary">
          Third-Party Services
        </h2>
        <p>
          We use the following third-party services: Google Analytics for
          website analytics, Vercel for hosting and edge analytics, and jsDelivr
          CDN for serving open-source font files. Each service has its own
          privacy policy governing data it collects.
        </p>

        <h2 className="font-serif text-xl font-semibold text-text-primary">
          Data Sharing
        </h2>
        <p>
          We do not sell, rent, or share your data with third parties for
          marketing purposes. Analytics data is processed by Google and Vercel
          under their respective privacy policies.
        </p>

        <h2 className="font-serif text-xl font-semibold text-text-primary">
          Contact
        </h2>
        <p>
          If you have questions about this privacy policy, contact us at{" "}
          <a
            href="mailto:addy@thequery.in"
            className="text-accent hover:text-accent-hover transition-colors"
          >
            addy@thequery.in
          </a>
          .
        </p>
      </div>
    </div>
  );
}
