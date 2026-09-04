import { signOgUpload, verifyOgUploadToken } from "../../../api/_lib/ogSignature.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function withSecret<T>(secret: string | undefined, fn: () => T): T {
  const prev = process.env.OG_SIGNING_SECRET;
  if (secret === undefined) delete process.env.OG_SIGNING_SECRET;
  else process.env.OG_SIGNING_SECRET = secret;
  try {
    return fn();
  } finally {
    if (prev === undefined) delete process.env.OG_SIGNING_SECRET;
    else process.env.OG_SIGNING_SECRET = prev;
  }
}

export async function run() {
  return runSuite("persistence / ogSignature (api)", [
    {
      name: "valid token verifies",
      fn: () =>
        withSecret("test-secret", () => {
          const signed = signOgUpload("payloadA");
          assert(!!signed, "token minted");
          assert(verifyOgUploadToken("payloadA", signed!.token), "valid token accepted");
        }),
    },
    {
      name: "expired token rejected",
      fn: () =>
        withSecret("test-secret", () => {
          const expiresAt = Date.now() - 1;
          // Reconstruct a token as if minted in the past, using the same
          // format signOgUpload produces (expiresAt.sig).
          const stale = `${expiresAt}.deadbeef`;
          assert(!verifyOgUploadToken("payloadA", stale), "expired token rejected");
        }),
    },
    {
      name: "tampered signature rejected",
      fn: () =>
        withSecret("test-secret", () => {
          const signed = signOgUpload("payloadA")!;
          const [expiresAt, sig] = signed.token.split(".");
          const flipped = sig!.startsWith("0") ? `1${sig!.slice(1)}` : `0${sig!.slice(1)}`;
          assert(!verifyOgUploadToken("payloadA", `${expiresAt}.${flipped}`), "tampered signature rejected");
        }),
    },
    {
      name: "token minted for one payload fails for another",
      fn: () =>
        withSecret("test-secret", () => {
          const signed = signOgUpload("payloadA")!;
          assert(!verifyOgUploadToken("payloadB", signed.token), "cross-payload token rejected");
        }),
    },
    {
      name: "malformed token does not throw",
      fn: () =>
        withSecret("test-secret", () => {
          assert(!verifyOgUploadToken("payloadA", ""), "empty token rejected");
          assert(!verifyOgUploadToken("payloadA", "garbage"), "no-dot token rejected");
          assert(!verifyOgUploadToken("payloadA", "notanumber.abcd"), "non-numeric expiry rejected");
        }),
    },
    {
      name: "secret unset disables signing (fail open)",
      fn: () =>
        withSecret(undefined, () => {
          assert(signOgUpload("payloadA") === null, "signing disabled returns null");
          assert(verifyOgUploadToken("payloadA", "anything"), "verification passes when unconfigured");
          assert(verifyOgUploadToken("payloadA", ""), "verification passes even with empty token");
        }),
    },
  ]);
}
