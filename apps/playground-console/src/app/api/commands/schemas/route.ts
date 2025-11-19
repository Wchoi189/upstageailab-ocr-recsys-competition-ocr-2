import { NextRequest } from "next/server";
import { proxyToFastAPI } from "@/lib/api-proxy";

export async function GET(request: NextRequest) {
  return proxyToFastAPI("/commands/schemas");
}
