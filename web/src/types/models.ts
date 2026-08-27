/** Shared domain types for Laplacian (expressions → bake → GPU). */

export type ExprRole = "auto" | "density" | "constraint";
export type AnimMode = "pingpong" | "loop";
export type ExprKind = "parameter" | "constraint" | "definition" | "bare";
export type LayerRole = "parameter" | "density" | "constraint";

export interface ExprItem {
  id: string;
  latex: string;
  color: string;
  color2: string;
  colors: string[];
  role: ExprRole;
  enabled: boolean;
  sliderMin: number;
  sliderMax: number;
  sliderSpeed: number;
  sliderAnimating: boolean;
  sliderPhase: number;
  sliderAnimMode: AnimMode;
  autoParam: boolean;
}

export interface ParamState {
  value: number;
  min: number;
  max: number;
  step: number;
  animating: boolean;
  speed: number;
  phase: number;
  animMode: AnimMode;
  latex: string;
  exprId: string | null;
  driven: boolean;
  freeParams: string[];
  error: string | null;
}

export interface KeyframeBlend {
  i0: number;
  i1: number;
  t: number;
}

export interface KeyframeFrame {
  dens: Float32Array;
  cheb?: Float32Array;
  fitRel?: number;
}

export interface DensLayer {
  id?: string;
  dens: Float32Array;
  color: number[];
  color2: number[];
  colors?: number[][];
  cheb?: Float32Array;
  fitRel?: number;
  keyframes?: KeyframeFrame[];
}

export interface ConstraintLayer {
  id?: string;
  dens?: Float32Array;
  gx?: Float32Array;
  gy?: Float32Array;
  gz?: Float32Array;
  color: number[];
  color2: number[];
  colors?: number[][];
  isoLevel: number;
  cheb?: Float32Array;
  fitRel?: number;
  keyframes?: KeyframeFrame[];
  blend?: KeyframeBlend;
}

export interface SceneBake {
  densLayers: DensLayer[];
  constraints: ConstraintLayer[];
  M: number;
  dens: Float32Array | null;
  deg?: number;
  half?: number;
  fittedCount?: number;
  keyframedCount?: number;
  keyframeBaked?: boolean;
  densKeyframedCpu?: boolean;
}

export interface ExprMeta {
  kind: ExprKind;
  shade: string;
  isoLevel: number;
  label: string;
}

export interface FitTiming {
  sampleMs: number;
  chebMs: number;
  monoMs: number;
  l2Ms: number;
  totalMs: number;
  uploadMs?: number;
}

export interface CompiledLayer {
  item: ExprItem;
  role: LayerRole;
  fn: (x: number, y: number, z: number) => number;
  compiled: {
    freeParams: string[];
    isoLevel?: number;
    kind?: ExprKind;
  };
}

export interface ExprListApi {
  render: (focus?: { id: string; pos?: number } | null) => void;
  syncAllParamSliders?: () => void;
  syncParamChrome?: () => boolean;
}
