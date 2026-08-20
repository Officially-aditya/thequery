import AdminShell from "@/components/admin/AdminShell";
import EditorialCollection from "@/components/admin/EditorialCollection";

export default function AdminGuidesPage() {
  return (
    <AdminShell title="Guides">
      <EditorialCollection kind="guide" noun="Guide" description="Write evergreen learning material with an ordered, structured content canvas and a live page preview." />
    </AdminShell>
  );
}
