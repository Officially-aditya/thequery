import AdminShell from "@/components/admin/AdminShell";
import EditorialCollection from "@/components/admin/EditorialCollection";

export default function AdminComparisonsPage() {
  return (
    <AdminShell title="Comparisons">
      <EditorialCollection kind="comparison" noun="Comparison" description="Publish side-by-side comparisons with tables, charts, and citations - then keep all source links in one reliable bottom section." />
    </AdminShell>
  );
}
