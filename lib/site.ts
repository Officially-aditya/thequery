export const SITE_URL = "https://www.thequery.in";
export const SITE_NAME = "TheQuery";

type SocialOpenGraphType = "website" | "article" | "book";

export function createOpenGraphMetadata({
  title,
  description,
  url,
  type = "website",
  image,
}: {
  title: string;
  description: string;
  url: string;
  type?: SocialOpenGraphType;
  image?: string | null;
}) {
  return {
    title,
    description,
    url,
    siteName: SITE_NAME,
    type,
    images: image ? [image] : ["/opengraph-image"],
  };
}

export function createTwitterMetadata({
  title,
  description,
  image,
}: {
  title: string;
  description: string;
  image?: string | null;
}) {
  return {
    card: "summary_large_image" as const,
    title,
    description,
    images: image ? [image] : ["/twitter-image"],
  };
}

export const AUTHOR = {
  name: "Addy",
  url: `${SITE_URL}/about`,
  email: "addy@thequery.in",
  jobTitle: "Independent AI educator and technical writer",
};

export const ORGANIZATION_ID = `${SITE_URL}/#organization`;
export const ORGANIZATION_LOGO = `${SITE_URL}/logo.png`;

export const organizationJsonLd = {
  "@type": "Organization",
  "@id": ORGANIZATION_ID,
  name: SITE_NAME,
  url: SITE_URL,
  logo: {
    "@type": "ImageObject",
    url: ORGANIZATION_LOGO,
  },
  email: `mailto:${AUTHOR.email}`,
  founder: {
    "@type": "Person",
    name: AUTHOR.name,
    url: AUTHOR.url,
  },
};

export const authorJsonLd = {
  "@type": "Person",
  name: AUTHOR.name,
  url: AUTHOR.url,
  email: `mailto:${AUTHOR.email}`,
  jobTitle: AUTHOR.jobTitle,
  worksFor: {
    "@id": ORGANIZATION_ID,
  },
};
