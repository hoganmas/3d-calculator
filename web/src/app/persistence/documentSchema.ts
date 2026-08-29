/**
 * Laplacian document v1 types and strict validation (no app/DOM deps).
 */
import type { ExprItem, ParamState } from "../../types/models.js";

export const DOCUMENT_VERSION = 1 as const;

export interface LaplacianDocumentMeta {
  title?: string;
  preset?: string;
}

export interface LaplacianRenderSnapshot {
  deg: number;
  scale: number;
  steps: number;
  boxSize: number;
  marchDownscale: number;
  showGridAxes: boolean;
  preset: string;
}

export interface LaplacianFlowSnapshot {
  flowAlpha: number;
  flowNoiseScale: number;
  flowGridPoints: boolean;
  flowDt: number;
  flowSpeed: number;
  flowVMax: number;
  flowOpacity: number;
  flowAgeMax: number;
  flowVizMode: "particles" | "ibfv";
  flowParticleCount: number;
  flowTrailSteps: number;
  flowTrailWidth: number;
}

export interface LaplacianCameraSnapshot {
  position: [number, number, number];
  target: [number, number, number];
}

export interface LaplacianDocument {
  format: "laplacian";
  version: typeof DOCUMENT_VERSION;
  revision: number;
  savedAt: string;
  meta?: LaplacianDocumentMeta;
  expressions: Partial<ExprItem>[];
  params: Record<string, Partial<ParamState>>;
  render: LaplacianRenderSnapshot;
  flow: LaplacianFlowSnapshot;
  camera?: LaplacianCameraSnapshot;
}

let revision = 0;

export function getDocumentRevision() {
  return revision;
}

export function setDocumentRevision(next: number) {
  if (Number.isFinite(next) && next >= 0) revision = Math.floor(next);
}

export function bumpDocumentRevision() {
  revision += 1;
  return revision;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return !!v && typeof v === "object" && !Array.isArray(v);
}

function isFiniteNum(v: unknown): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

function validateExprItem(v: unknown, path: string) {
  if (!isRecord(v)) throw new Error(`${path}: expected object`);
  if (typeof v.latex !== "string") throw new Error(`${path}.latex: expected string`);
  if (v.enabled != null && typeof v.enabled !== "boolean") {
    throw new Error(`${path}.enabled: expected boolean`);
  }
  if (v.role != null && typeof v.role !== "string") {
    throw new Error(`${path}.role: expected string`);
  }
  if (v.id != null && typeof v.id !== "string") throw new Error(`${path}.id: expected string`);
}

function validateParams(v: unknown) {
  if (!isRecord(v)) throw new Error("params: expected object");
  for (const [name, p] of Object.entries(v)) {
    if (!name) throw new Error("params: empty key");
    if (!isRecord(p)) throw new Error(`params.${name}: expected object`);
    if (p.value != null && !isFiniteNum(p.value)) throw new Error(`params.${name}.value: invalid`);
    if (p.min != null && !isFiniteNum(p.min)) throw new Error(`params.${name}.min: invalid`);
    if (p.max != null && !isFiniteNum(p.max)) throw new Error(`params.${name}.max: invalid`);
    if (p.speed != null && !isFiniteNum(p.speed)) throw new Error(`params.${name}.speed: invalid`);
    if (p.phase != null && !isFiniteNum(p.phase)) throw new Error(`params.${name}.phase: invalid`);
    if (p.animating != null && typeof p.animating !== "boolean") {
      throw new Error(`params.${name}.animating: expected boolean`);
    }
    if (p.animMode != null && p.animMode !== "loop" && p.animMode !== "pingpong") {
      throw new Error(`params.${name}.animMode: invalid`);
    }
  }
}

function validateRender(v: unknown): LaplacianRenderSnapshot {
  if (!isRecord(v)) throw new Error("render: expected object");
  const deg = v.deg;
  const scale = v.scale;
  const steps = v.steps;
  const boxSize = v.boxSize;
  const marchDownscale = v.marchDownscale;
  const preset = v.preset;
  if (!isFiniteNum(deg)) throw new Error("render.deg: out of range");
  if (deg < 1 || deg > 128) throw new Error("render.deg: out of range");
  if (!isFiniteNum(scale)) throw new Error("render.scale: invalid");
  if (scale <= 0) throw new Error("render.scale: invalid");
  if (!isFiniteNum(steps)) throw new Error("render.steps: out of range");
  if (steps < 8 || steps > 96) throw new Error("render.steps: out of range");
  if (!isFiniteNum(boxSize)) throw new Error("render.boxSize: invalid");
  if (boxSize <= 0) throw new Error("render.boxSize: invalid");
  if (!isFiniteNum(marchDownscale)) throw new Error("render.marchDownscale: out of range");
  if (marchDownscale < 1 || marchDownscale > 16) {
    throw new Error("render.marchDownscale: out of range");
  }
  const showGridAxes = v.showGridAxes;
  if (showGridAxes != null && typeof showGridAxes !== "boolean") {
    throw new Error("render.showGridAxes: expected boolean");
  }
  if (typeof preset !== "string") throw new Error("render.preset: expected string");
  return {
    deg: deg as number,
    scale: scale as number,
    steps: steps as number,
    boxSize: boxSize as number,
    marchDownscale: marchDownscale as number,
    showGridAxes: showGridAxes !== false,
    preset,
  };
}

function validateFlow(v: unknown): LaplacianFlowSnapshot {
  if (!isRecord(v)) throw new Error("flow: expected object");
  const flowVizMode = v.flowVizMode;
  if (flowVizMode !== "particles" && flowVizMode !== "ibfv") {
    throw new Error("flow.flowVizMode: invalid");
  }
  const nums = [
    "flowAlpha",
    "flowNoiseScale",
    "flowDt",
    "flowSpeed",
    "flowVMax",
    "flowOpacity",
    "flowAgeMax",
    "flowParticleCount",
    "flowTrailSteps",
    "flowTrailWidth",
  ] as const;
  for (const k of nums) {
    if (!isFiniteNum(v[k])) throw new Error(`flow.${k}: invalid`);
  }
  if (typeof v.flowGridPoints !== "boolean") throw new Error("flow.flowGridPoints: expected boolean");
  return {
    flowAlpha: v.flowAlpha as number,
    flowNoiseScale: v.flowNoiseScale as number,
    flowGridPoints: v.flowGridPoints,
    flowDt: v.flowDt as number,
    flowSpeed: v.flowSpeed as number,
    flowVMax: v.flowVMax as number,
    flowOpacity: v.flowOpacity as number,
    flowAgeMax: v.flowAgeMax as number,
    flowVizMode,
    flowParticleCount: v.flowParticleCount as number,
    flowTrailSteps: v.flowTrailSteps as number,
    flowTrailWidth: v.flowTrailWidth as number,
  };
}

function validateCamera(v: unknown): LaplacianCameraSnapshot | undefined {
  if (v == null) return undefined;
  if (!isRecord(v)) throw new Error("camera: expected object");
  const pos = v.position;
  const tgt = v.target;
  if (!Array.isArray(pos) || pos.length !== 3 || !pos.every(isFiniteNum)) {
    throw new Error("camera.position: expected [x,y,z]");
  }
  if (!Array.isArray(tgt) || tgt.length !== 3 || !tgt.every(isFiniteNum)) {
    throw new Error("camera.target: expected [x,y,z]");
  }
  return {
    position: [pos[0]!, pos[1]!, pos[2]!],
    target: [tgt[0]!, tgt[1]!, tgt[2]!],
  };
}

export function validateDocument(raw: unknown): LaplacianDocument {
  if (!isRecord(raw)) throw new Error("Document must be a JSON object");
  if (raw.format !== "laplacian") throw new Error('format must be "laplacian"');
  if (raw.version !== DOCUMENT_VERSION) throw new Error(`unsupported version: ${raw.version}`);
  if (!isFiniteNum(raw.revision)) throw new Error("revision: invalid");
  if (raw.revision < 0) throw new Error("revision: invalid");
  if (typeof raw.savedAt !== "string" || !raw.savedAt) throw new Error("savedAt: required");
  if (!Array.isArray(raw.expressions)) throw new Error("expressions: expected array");
  raw.expressions.forEach((e, i) => validateExprItem(e, `expressions[${i}]`));
  validateParams(raw.params);
  const render = validateRender(raw.render);
  const flow = validateFlow(raw.flow);
  const cameraSnap = validateCamera(raw.camera);
  let meta: LaplacianDocumentMeta | undefined;
  if (raw.meta != null) {
    if (!isRecord(raw.meta)) throw new Error("meta: expected object");
    meta = {};
    if (raw.meta.title != null) {
      if (typeof raw.meta.title !== "string") throw new Error("meta.title: expected string");
      meta.title = raw.meta.title;
    }
    if (raw.meta.preset != null) {
      if (typeof raw.meta.preset !== "string") throw new Error("meta.preset: expected string");
      meta.preset = raw.meta.preset;
    }
  }
  return {
    format: "laplacian",
    version: DOCUMENT_VERSION,
    revision: raw.revision as number,
    savedAt: raw.savedAt,
    meta,
    expressions: raw.expressions as Partial<ExprItem>[],
    params: raw.params as Record<string, Partial<ParamState>>,
    render,
    flow,
    camera: cameraSnap,
  };
}
