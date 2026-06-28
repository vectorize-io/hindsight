import { VectorAdminView } from "@/components/vector-admin-view";

export const dynamic = "force-dynamic";

export default function VectorAdminPage() {
  return (
    <div className="min-h-screen bg-background">
      <VectorAdminView />
    </div>
  );
}
