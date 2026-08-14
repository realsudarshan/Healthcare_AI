import { getAvailableModels, getCapabilities, isDemo } from "@/lib/ai/models";

export function GET() {
  const headers = {
    "Cache-Control": "no-store",
  };

  const curatedCapabilities = getCapabilities();

  if (isDemo) {
    const models = getAvailableModels();
    const capabilities = Object.fromEntries(
      models.map((m) => [m.id, curatedCapabilities[m.id] ?? m.capabilities])
    );

    return Response.json({ capabilities, models }, { headers });
  }

  return Response.json(curatedCapabilities, { headers });
}
