import AdminShell from "@/components/admin/AdminShell";
import EditorialCollection from "@/components/admin/EditorialCollection";

export default function AdminArticlesPage() {
  return (
    <AdminShell title="Articles">
      <EditorialCollection kind="article" noun="Article" description="Publish timely analysis with ordered text, comparison-table, and chart blocks—then keep all source links in one reliable bottom section." />
    </AdminShell>
  );
}
