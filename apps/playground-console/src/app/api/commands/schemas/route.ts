import { proxyToFastAPI } from "@/lib/api-proxy";

export async function GET() {
  return proxyToFastAPI("/commands/schemas");
}
