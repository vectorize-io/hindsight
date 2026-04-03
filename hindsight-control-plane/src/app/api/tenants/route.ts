import { NextResponse } from "next/server";
import { getTenantNames, isMultiTenant } from "@/lib/hindsight-client";

export async function GET() {
  return NextResponse.json({
    tenants: getTenantNames(),
    multi_tenant: isMultiTenant(),
  });
}
