import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Admin",
  robots: "noindex, nofollow",
};

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto max-w-[1440px] px-4 py-8 sm:px-6 lg:py-10">{children}</div>;
}
