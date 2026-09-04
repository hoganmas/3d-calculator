import { createHmac, timingSafeEqual } from "node:crypto";

/**
 * Server-issued, short-lived tokens authorizing a client-captured OG PNG
 * upload for a specific share payload. Not a real authorization boundary —
 * the payload is the public share URL, so anyone with it can also hit
 * /api/og-sign — but it stops casual/scripted injection that doesn't speak
 * the app's actual sign->upload protocol. See the OG signing plan for the
 * full threat-model discussion.
 *
 * Fails open: if OG_SIGNING_SECRET isn't configured, signing is disabled
 * everywhere (signOgUpload returns null, verifyOgUploadToken always true) so
 * an unconfigured environment behaves exactly like the pre-signing code did.
 */

const TTL_MS = 5 * 60 * 1000;

function getSecret(): string | null {
  return process.env.OG_SIGNING_SECRET || null;
}

function hmac(secret: string, payload: string, expiresAt: number): string {
  return createHmac("sha256", secret).update(`${payload}:${expiresAt}`).digest("hex");
}

export function signOgUpload(payload: string): { token: string; expiresAt: number } | null {
  const secret = getSecret();
  if (!secret) return null;
  const expiresAt = Date.now() + TTL_MS;
  return { token: `${expiresAt}.${hmac(secret, payload, expiresAt)}`, expiresAt };
}

export function verifyOgUploadToken(payload: string, token: string): boolean {
  const secret = getSecret();
  if (!secret) return true;

  const [expiresAtStr, sig] = token.split(".");
  if (!expiresAtStr || !sig) return false;

  const expiresAt = Number(expiresAtStr);
  if (!Number.isFinite(expiresAt) || Date.now() > expiresAt) return false;

  let expected: Buffer;
  let got: Buffer;
  try {
    expected = Buffer.from(hmac(secret, payload, expiresAt), "hex");
    got = Buffer.from(sig, "hex");
  } catch {
    return false;
  }
  return expected.length === got.length && timingSafeEqual(expected, got);
}
