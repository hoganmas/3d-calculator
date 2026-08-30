import {
  DOCUMENT_VERSION,
  validateDocument,
  type LaplaciDocument,
} from "../../src/app/persistence/documentSchema.js";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function sampleDoc(revision = 1): LaplaciDocument {
  return {
    format: "laplaci",
    version: DOCUMENT_VERSION,
    revision,
    savedAt: "2026-01-01T00:00:00.000Z",
    meta: { preset: "sincos" },
    expressions: [{ latex: "\\sin(x)+\\cos(y)", enabled: true }],
    params: { a: { value: 1, min: 0, max: 2, animating: false } },
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
      flowAlpha: 0.1,
      flowNoiseScale: 0.3,
      flowGridPoints: false,
      flowDt: 0.05,
      flowSpeed: 0.1,
      flowVMax: 0,
      flowOpacity: 0.5,
      flowAgeMax: 30,
      flowVizMode: "particles",
      flowParticleCount: 1000,
      flowTrailSteps: 32,
      flowTrailWidth: 10,
    },
    camera: {
      position: [5.2, 4.0, 6.8],
      target: [0, 0, 0],
    },
  };
}

export async function run() {
  return runSuite("persistence / document", [
    {
      name: "round-trip JSON preserves core fields",
      fn: () => {
        const doc = sampleDoc(3);
        const parsed = validateDocument(JSON.parse(JSON.stringify(doc)));
        assert(parsed.revision === 3, "revision");
        assert(parsed.expressions[0]?.latex === doc.expressions[0]?.latex, "latex");
        assert(parsed.params.a?.value === 1, "param value");
        assert(parsed.render.deg === 20, "render.deg");
        assert(parsed.flow.flowVizMode === "particles", "flow mode");
        assert(parsed.camera?.position[0] === 5.2, "camera x");
      },
    },
    {
      name: "rejects wrong version",
      fn: () => {
        const raw = { ...sampleDoc(), version: 99 };
        let threw = false;
        try {
          validateDocument(raw);
        } catch (e) {
          threw = true;
          assert(String(e).includes("version"), "version error");
        }
        assert(threw, "should throw");
      },
    },
    {
      name: "rejects malformed JSON shape",
      fn: () => {
        let threw = false;
        try {
          validateDocument({ format: "laplaci", version: 1 });
        } catch {
          threw = true;
        }
        assert(threw, "partial doc rejected");
      },
    },
    {
      name: "rejects torn / invalid expression rows",
      fn: () => {
        const raw = sampleDoc();
        raw.expressions = [{ latex: 42 } as unknown as LaplaciDocument["expressions"][0]];
        let threw = false;
        try {
          validateDocument(raw);
        } catch (e) {
          threw = true;
          assert(String(e).includes("latex"), "latex type error");
        }
        assert(threw, "bad expr rejected");
      },
    },
    {
      name: "render.showGridAxes defaults to true when omitted",
      fn: () => {
        const raw = sampleDoc();
        const render = { ...raw.render } as Record<string, unknown>;
        delete render.showGridAxes;
        const parsed = validateDocument({ ...raw, render });
        assert(parsed.render.showGridAxes === true, "default showGridAxes");
      },
    },
    {
      name: "revision is preserved from payload",
      fn: () => {
        const a = validateDocument(sampleDoc(1));
        const b = validateDocument(sampleDoc(5));
        assert(b.revision > a.revision, "monotonic revisions in fixtures");
      },
    },
  ]);
}
