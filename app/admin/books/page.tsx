import AdminShell from "@/components/admin/AdminShell";
import BooksManager from "@/components/admin/BooksManager";

export default function AdminBooksPage() {
  return <AdminShell title="Books"><BooksManager /></AdminShell>;
}
