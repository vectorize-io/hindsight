import { NextRequest, NextResponse } from "next/server";
import { sdk, lowLevelClient } from "@/lib/hindsight-client";

/**
 * Verify that the requested bank exists and is accessible.
 * This prevents Insecure Direct Object Reference (IDOR) attacks.
 */
async function verifyBankAccess(bankId: string): Promise<boolean> {
  try {
    // Get list of accessible banks
    const response = await sdk.listBanks({ client: lowLevelClient });
    if (!response.data?.banks) {
      return false;
    }
    
    // Verify the requested bank exists in the accessible banks list
    const bankIds = response.data.banks.map((bank: any) => bank.bank_id);
    return bankIds.includes(bankId);
  } catch (error) {
    console.error("Error verifying bank access:", error);
    return false;
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  try {
    const { bankId } = await params;
    
    // Authorization check: Verify bank access to prevent IDOR
    const hasAccess = await verifyBankAccess(bankId);
    if (!hasAccess) {
      return NextResponse.json(
        { error: "Forbidden: Access denied to this bank" },
        { status: 403 }
      );
    }
    
    const response = await sdk.getBankProfile({
      client: lowLevelClient,
      path: { bank_id: bankId },
    });
    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error fetching bank profile:", error);
    return NextResponse.json({ error: "Failed to fetch bank profile" }, { status: 500 });
  }
}

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ bankId: string }> }
) {
  try {
    const { bankId } = await params;
    
    // Authorization check: Verify bank access to prevent IDOR
    const hasAccess = await verifyBankAccess(bankId);
    if (!hasAccess) {
      return NextResponse.json(
        { error: "Forbidden: Access denied to this bank" },
        { status: 403 }
      );
    }
    
    const body = await request.json();

    const response = await sdk.createOrUpdateBank({
      client: lowLevelClient,
      path: { bank_id: bankId },
      body: body,
    });
    return NextResponse.json(response.data, { status: 200 });
  } catch (error) {
    console.error("Error updating bank profile:", error);
    return NextResponse.json({ error: "Failed to update bank profile" }, { status: 500 });
  }
}
