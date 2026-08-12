export const SITE_URL = "https://www.thequery.in";
export const SITE_NAME = "TheQuery";

export const AUTHOR = {
  name: "Addy",
  url: `${SITE_URL}/about`,
  email: "addy@thequery.in",
  jobTitle: "Independent AI educator and technical writer",
};

export const ORGANIZATION_ID = `${SITE_URL}/#organization`;
export const ORGANIZATION_LOGO = `${SITE_URL}/icon.svg`;

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
