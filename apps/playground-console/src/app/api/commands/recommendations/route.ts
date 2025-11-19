import { NextRequest } from "next/server";
import { proxyToFastAPI } from "@/lib/api-proxy";

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const architecture = searchParams.get("architecture");

  const query = architecture ? `?architecture=${architecture}` : "";
  return proxyToFastAPI(`/commands/recommendations${query}`);
}
