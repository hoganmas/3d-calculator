import type { VercelRequest, VercelResponse } from "@vercel/node";
import { validateSharePayload } from "./_lib/sharePayload.js";
import { signOgUpload } from "./_lib/ogSignature.js";

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "GET") {
    res.status(405).send("Method not allowed");
    return;
  }

  const raw = String(req.query.e ?? "");
  let payload: string;
  try {
    payload = validateSharePayload(raw);
  } catch {
    res.status(400).send("Invalid share payload");
    return;
  }

  res.setHeader("Cache-Control", "no-store");
  const signed = signOgUpload(payload);
  res.status(200).json(signed ?? { token: null });
}
