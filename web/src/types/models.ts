/** Shared domain types for Laplacian (expressions → bake → GPU). */

export type AnimMode = "pingpong" | "loop";
export type ExprKind = "parameter" | "alias" | "funcdef" | "constraint" | "definition" | "bare";
export type DeclSymbolKind = "parameter" | "alias" | "funcdef";
export type LayerRole = "parameter" | "cloud" | "isosurface" | "flow";
export type VectorFieldKind = "tuple" | "gradient" | "curl" | "cross" | "reference";
export type ScalarFieldOperator =
  | "none"
  | "laplacian"
  | "divergence"
  | "partial"
  | "definite_integral"
  | "dot_product"
  | "grad_dot";

export interface IntegralAxisSpec {
  axis: 0 | 1 | 2;
  aLatex: string;
  bLatex: string;
}

export interface ExprItem {
  id: string;
  latex: string;
  color: string;
  color2: string;
  colors: string[];
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
  dens?: Float32Array;
  gx?: Float32Array;
  gy?: Float32Array;
  gz?: Float32Array;
  fx?: Float32Array;
  fy?: Float32Array;
  fz?: Float32Array;
  cheb?: Float32Array;
  fitRel?: number;
}

export interface CloudLayer {
  id?: string;
  dens: Float32Array;
  color: number[];
  color2: number[];
  colors?: number[][];
  cheb?: Float32Array;
  fitRel?: number;
  keyframes?: KeyframeFrame[];
  blend?: KeyframeBlend;
  latex?: string;
  freeParams?: string[];
}

export interface IsosurfaceLayer {
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
  latex?: string;
  freeParams?: string[];
}

export interface FlowLayer {
  id?: string;
  fx: Float32Array;
  fy: Float32Array;
  fz: Float32Array;
  color: number[];
  color2: number[];
  colors?: number[][];
  cheb?: Float32Array;
  fitRel?: number;
  keyframes?: KeyframeFrame[];
  blend?: KeyframeBlend;
  latex?: string;
  freeParams?: string[];
}

export interface SceneBake {
  cloudLayers: CloudLayer[];
  isosurfaceLayers: IsosurfaceLayer[];
  flowLayers: FlowLayer[];
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
  kind: ExprKind | "mixed";
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
  fittedCount?: number;
  keyframedCount?: number;
  kfBakeMs?: number;
  kfLerpMs?: number;
  kfK?: number;
  kfSampleMs?: number;
  kfChebMs?: number;
  kfIdctMs?: number;
  kfGradMs?: number;
}

export interface CompiledLayer {
  item: ExprItem;
  role: LayerRole;
  compiled?: CompiledExpr;
  vectorCompiled?: CompiledVectorExpr;
  fn?: (x: number, y: number, z: number) => number;
  vectorFn?: (x: number, y: number, z: number) => [number, number, number];
}

export interface ClassifiedVectorExpr {
  kind: VectorFieldKind;
  label: string;
  /** Tuple/cross: component LaTeX. Gradient: scalar LaTeX. Cross: six (a,b,c,d,e,f). */
  compileParts: string[];
  /** Scalar multiplier on tuple components (e.g. 0.5(x,y,z)). */
  scale?: number;
}

export interface CompiledVectorExpr {
  freeParams: string[];
  usesSpace: boolean;
  kind: VectorFieldKind;
  classifyLabel: string;
  /** Scalar fit path for gradient-sourced fields. */
  scalarCompileLatex?: string;
  bind: (
    params?: Record<string, number>,
  ) => (x: number, y: number, z: number) => [number, number, number];
  bindScalar?: (
    params?: Record<string, number>,
  ) => (x: number, y: number, z: number) => number;
  bindTuple?: (
    params?: Record<string, number>,
  ) => (x: number, y: number, z: number) => [number, number, number];
}

export interface VectorFitResult {
  fx: Float32Array;
  fy: Float32Array;
  fz: Float32Array;
  cheb?: Float32Array;
  fitRel?: number;
  M: number;
  source: VectorFieldKind;
}

export interface ExprListApi {
  render: (focus?: { id: string; pos?: number } | null) => void;
  syncAllParamSliders?: () => void;
  syncParamChrome?: () => boolean;
  clearAll?: () => void;
}

export type ClassifiedShade = "iso" | "volume" | "none";

export interface ClassifiedExpr {
  kind: ExprKind;
  shade: ClassifiedShade;
  isoLevel: number;
  compileLatex: string;
  label: string;
  paramName?: string;
  aliasName?: string;
  funcName?: string;
  funcArgs?: string[];
}

/** Field expressions (not parameter rows). */
export type FieldKind = "constraint" | "definition" | "bare";

export interface CompiledExpr {
  freeParams: string[];
  usesSpace: boolean;
  kind: FieldKind;
  shade: ClassifiedShade;
  isoLevel: number;
  classifyLabel: string;
  operator?: ScalarFieldOperator;
  /** Inner scalar for \\laplacian f. */
  scalarCompileLatex?: string;
  /** Tuple components for \\div(Fx,Fy,Fz). */
  divergenceParts?: [string, string, string];
  /** Axis for \\partial_x etc. */
  partialAxis?: 0 | 1 | 2;
  /** Chained definite integrals (innermost first). */
  integralAxes?: IntegralAxisSpec[];
  bind: (params?: Record<string, number>) => (x: number, y: number, z: number) => number;
  bindScalar?: (
    params?: Record<string, number>,
  ) => (x: number, y: number, z: number) => number;
  bindTuple?: (
    params?: Record<string, number>,
  ) => (x: number, y: number, z: number) => [number, number, number];
}

export interface CompiledParam {
  name: string;
  rhsLatex: string;
  freeParams: string[];
  isConstant: boolean;
  constantValue: number | null;
  eval: (scope?: Record<string, number>) => number;
}

export interface PresetParamSeed {
  value?: number;
  min?: number;
  max?: number;
  speed?: number;
  animate?: boolean;
  animating?: boolean;
  phase?: number;
  animMode?: AnimMode;
}

export interface PresetDef {
  label: string;
  latex?: string;
  expressions?: Partial<ExprItem>[];
  params?: Record<string, PresetParamSeed>;
}

export interface ChebFitTiming {
  sampleMs: number;
  chebMs: number;
  monoMs: number;
  l2Ms: number;
  totalMs: number;
}

export interface ChebFitResult {
  cheb: Float32Array;
  mono: Float32Array | null;
  deg: number;
  half: number;
  fitRelL2: number;
  fMin: number;
  fMax: number;
  timing: ChebFitTiming;
}

export interface Idct3DResult {
  dens: Float32Array;
  M: number;
  deg: number;
  n: number;
}

export interface IdctGrad3DResult {
  gx: Float32Array;
  gy: Float32Array;
  gz: Float32Array;
  M: number;
  deg: number;
  n: number;
}

export interface IdctCurl3DResult {
  fx: Float32Array;
  fy: Float32Array;
  fz: Float32Array;
  M: number;
  deg: number;
  n: number;
}

export interface ScalarFitResult {
  dens: Float32Array;
  cheb: Float32Array;
  fitRelL2: number;
  M: number;
  deg: number;
}

export interface CompileLayerResult {
  item: ExprItem;
  compiled?: CompiledExpr;
  vectorCompiled?: CompiledVectorExpr;
  role: LayerRole;
  fn?: (x: number, y: number, z: number) => number;
  vectorFn?: (x: number, y: number, z: number) => [number, number, number];
}

export interface CompileAllResult {
  freeParams: string[];
  layers: CompileLayerResult[];
  warnings: string[];
}
