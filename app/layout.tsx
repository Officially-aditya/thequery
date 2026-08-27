import type { Metadata } from "next";
import { Lora, Source_Serif_4, JetBrains_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import {
  AUTHOR,
  ORGANIZATION_ID,
  SITE_NAME,
  SITE_URL,
  createOpenGraphMetadata,
  createTwitterMetadata,
  organizationJsonLd,
} from "@/lib/site";
import "./globals.css";

const lora = Lora({
  variable: "--font-lora",
  subsets: ["latin"],
  display: "swap",
});

const sourceSerif = Source_Serif_4({
  variable: "--font-source-serif",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL("https://www.thequery.in"),
  alternates: {
    canonical: "./",
  },
  title: {
    default: "TheQuery - AI Knowledge from First Principles",
    template: "%s | TheQuery",
  },
  authors: [{ name: AUTHOR.name, url: AUTHOR.url }],
  creator: AUTHOR.name,
  publisher: SITE_NAME,
  description: "TheQuery is where developers go to understand AI, not just use it. Glossary, books, and articles covering AI from first principles.",
  openGraph: createOpenGraphMetadata({
    title: "TheQuery - AI Knowledge from First Principles",
    description: "TheQuery is where developers go to understand AI, not just use it. Glossary, books, and articles covering AI from first principles.",
    url: SITE_URL,
  }),
  twitter: createTwitterMetadata({
    title: "TheQuery - AI Knowledge from First Principles",
    description: "TheQuery is where developers go to understand AI, not just use it.",
  }),
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const siteJsonLd = {
    "@context": "https://schema.org",
    ...organizationJsonLd,
    "@id": ORGANIZATION_ID,
  };

  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(siteJsonLd) }}
        />
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-SXRT67W8V7" />
        <script
          dangerouslySetInnerHTML={{
            __html: `
              window.dataLayer = window.dataLayer || [];
              function gtag(){dataLayer.push(arguments);}
              gtag('js', new Date());
              gtag('config', 'G-SXRT67W8V7');
            `,
          }}
        />
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                var theme = localStorage.getItem('theme');
                if (theme === 'dark' || (!theme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                  document.documentElement.classList.add('dark');
                }
              })();
            `,
          }}
        />
      </head>
      <body className={`${lora.variable} ${sourceSerif.variable} ${jetbrainsMono.variable} antialiased min-h-screen flex flex-col`}>
        <Header />
        <main className="flex-1">{children}</main>
        <Footer />
        <Analytics />
      </body>
    </html>
  );
}
