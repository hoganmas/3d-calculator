import { head, put } from "@vercel/blob";
import type { VercelRequest, VercelResponse } from "@vercel/node";
import { decodeSharePayload, sharePanelsFromRows, validateSharePayload } from "./_lib/sharePayload.js";
import { ogBlobKey } from "./_lib/ogBlob.js";
import { verifyOgUploadToken } from "./_lib/ogSignature.js";
import { isDynamicOgEnabled } from "./_lib/ogFeatureFlag.js";

const MAX_BYTES = 3 * 1024 * 1024;
const PNG_MAGIC = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

async function readRawBody(req: VercelRequest): Promise<Buffer> {
  if (Buffer.isBuffer(req.body)) return req.body;
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
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

  const token = String(req.query.token ?? "");
  if (!verifyOgUploadToken(payload, token)) {
    res.status(401).send("Invalid or expired signature");
    return;
  }

  // Must actually decode to a real, visual share — rejects garbage payloads
  // that were never legitimately created, and avoids storing junk under a
  // syntactically-valid-looking but meaningless key.
  try {
    const rows = await decodeSharePayload(payload);
    if (!sharePanelsFromRows(rows).length) {
      res.status(400).send("No visual expressions in payload");
      return;
    }
  } catch {
    res.status(400).send("Could not decode share payload");
    return;
  }

  const body = await readRawBody(req);
  if (!body.length || body.length > MAX_BYTES) {
    res.status(413).send("Image too large or empty");
    return;
  }
  if (!body.subarray(0, 8).equals(PNG_MAGIC)) {
    res.status(400).send("Not a PNG");
    return;
  }

  // Dynamic OG storage is off until there's an auth system to gate this by
  // identity — everything above still runs (keeps the client-facing
  // protocol fully exercised), but no request ever touches Blob storage
  // while this is off, so there's nothing to squat/inject.
  if (!isDynamicOgEnabled()) {
    res.status(200).send("Dynamic OG storage disabled");
    return;
  }

  const key = ogBlobKey(payload);

  // Write-once: the first successful upload for a payload wins. Blocks a
  // bot/griefer from overwriting an already-shared image with something
  // else. Combined with the signature check above, a fresh (never shared)
  // payload can no longer be griefed by an unauthenticated POST either —
  // see api/_lib/ogSignature.ts for what the token does and doesn't protect
  // against.
  try {
    const existing = await head(key);
    if (existing) {
      res.status(200).send("Already stored");
      return;
    }
  } catch {
    // head() throws when the blob doesn't exist yet — fall through to store.
  }

  try {
    await put(key, body, {
      access: "public",
      contentType: "image/png",
      addRandomSuffix: false,
    });
    res.status(201).send("Stored");
  } catch (err) {
    console.error("[api/upload-og]", err);
    res.status(500).send("Upload failed");
  }
}
