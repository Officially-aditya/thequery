import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Admin",
  robots: "noindex, nofollow",
};

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  return <div className="max-w-[960px] mx-auto px-4 py-12">{children}</div>;
}
