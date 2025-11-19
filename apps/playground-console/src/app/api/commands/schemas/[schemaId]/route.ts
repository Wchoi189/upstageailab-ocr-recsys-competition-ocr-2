import { NextRequest } from "next/server";
import { proxyToFastAPI } from "@/lib/api-proxy";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ schemaId: string }> }
) {
  const { schemaId } = await params;
  return proxyToFastAPI(`/commands/schemas/${schemaId}`);
}
