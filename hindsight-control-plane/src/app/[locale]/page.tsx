import { redirect } from "next/navigation";

// Force dynamic — redirects should not be prerendered.
export const dynamic = "force-dynamic";

export default function Home() {
  redirect("/dashboard");
}
