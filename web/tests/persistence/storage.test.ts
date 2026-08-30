import {
  checksumOf,
  createMemoryStorage,
  readDocument,
  setStorageBackend,
  writeDocument,
  type StorageHead,
} from "../../src/app/persistence/storage.js";
import { DOCUMENT_VERSION } from "../../src/app/persistence/documentSchema.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function sampleJson(revision: number, latex = "\\sin(x)") {
  const doc = {
    format: "laplacian",
    version: DOCUMENT_VERSION,
    revision,
    savedAt: "2026-01-01T00:00:00.000Z",
    expressions: [{ latex }],
    params: {},
    render: {
      deg: 20,
      scale: 2.5,
      steps: 16,
      boxSize: 5,
      marchDownscale: 2,
      showGridAxes: true,
      preset: "sincos",
    },
    flow: {
      flowDt: 0.05,
      flowSpeed: 0.1,
      flowVMax: 0,
      flowOpacity: 0.5,
      flowAgeMax: 30,
      flowParticleCount: 1000,
      flowTrailSteps: 32,
      flowTrailWidth: 10,
    },
  };
  return JSON.stringify(doc);
}

export async function run() {
  return runSuite("persistence / storage", [
    {
      name: "write A then B returns latest revision",
      fn: async () => {
        const mem = createMemoryStorage();
        setStorageBackend(mem);
        const jsonA = sampleJson(1, "\\sin(x)");
        await writeDocument(jsonA, 1);
        const jsonB = sampleJson(2, "\\cos(y)");
        await writeDocument(jsonB, 2);
        const doc = await readDocument();
        assert(doc != null, "doc read");
        assert(doc!.revision === 2, "revision B");
        assert(doc!.expressions[0]?.latex === "\\cos(y)", "payload B");
        setStorageBackend(null);
      },
    },
    {
      name: "checksum detects corrupted slot and falls back",
      fn: async () => {
        const mem = createMemoryStorage();
        setStorageBackend(mem);
        await writeDocument(sampleJson(1, "first"), 1);
        await writeDocument(sampleJson(2, "second"), 2);
        const headRaw = await mem.get("head");
        assert(headRaw != null, "head exists");
        const head = JSON.parse(headRaw!) as StorageHead;
        const slotKey = head.slot === "A" ? "slotA" : "slotB";
        await mem.put(slotKey, sampleJson(99, "CORRUPT"));
        const doc = await readDocument();
        assert(doc != null, "fallback read");
        assert(doc!.expressions[0]?.latex === "first", "inactive slot");
        setStorageBackend(null);
      },
    },
    {
      name: "interrupted commit recovers from previous slot",
      fn: async () => {
        const mem = createMemoryStorage();
        setStorageBackend(mem);
        const good = sampleJson(1, "good");
        await writeDocument(good, 1);
        const headRaw = await mem.get("head");
        const head = JSON.parse(headRaw!) as StorageHead;
        const nextSlot = head.slot === "A" ? "B" : "A";
        const badHead: StorageHead = {
          revision: 2,
          savedAt: "2026-01-02T00:00:00.000Z",
          checksum: checksumOf(sampleJson(2, "missing")),
          slot: nextSlot,
        };
        await mem.put("head", JSON.stringify(badHead));
        // active slot empty / missing — should fall back to slot 1
        const doc = await readDocument();
        assert(doc != null, "recovered doc");
        assert(doc!.expressions[0]?.latex === "good", "previous slot");
        setStorageBackend(null);
      },
    },
    {
      name: "checksumOf is stable",
      fn: () => {
        const a = checksumOf("hello");
        const b = checksumOf("hello");
        assert(a === b, "stable hash");
        assert(a !== checksumOf("hello!"), "different payload");
      },
    },
  ]);
}
