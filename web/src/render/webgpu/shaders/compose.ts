import gradientWgsl from "./common/gradient.wgsl?raw";
import isoHermiteWgsl from "./isoHermite.wgsl?raw";
import isoEdgeWgsl from "./isoEdge.wgsl?raw";
import isoUpsampleWgsl from "./isoUpsample.wgsl?raw";
import isoRefineAppendWgsl from "./isoRefineAppend.wgsl?raw";
import beerWgsl from "./beer.wgsl?raw";
import beerRefineAppendWgsl from "./beerRefineAppend.wgsl?raw";
import gridWgsl from "./grid.wgsl?raw";
import axisLabelWgsl from "./axisLabel.wgsl?raw";
import fxaaWgsl from "./fxaa.wgsl?raw";
import blitWgsl from "./blit.wgsl?raw";
import flowParticlesWgsl from "./flowParticles.wgsl?raw";
import { MAX_GRAD_STOPS } from "../../../model/expressions.js";
import { MAX_FLOW_TRAIL_STEPS } from "../../../math/fitVector.js";

function inject(src: string, vars: Record<string, string | number>): string {
  return Object.entries(vars).reduce(
    (s, [k, v]) => s.replaceAll(`{{${k}}}`, String(v)),
    src,
  );
}

function gradientBlock(maxGradStops: number): string {
  return inject(gradientWgsl, { MAX_GRAD_STOPS: maxGradStops });
}

export function getIsoShader(maxGradStops: number): string {
  return inject(isoHermiteWgsl, {
    GRADIENT_WGSL: gradientBlock(maxGradStops),
    MAX_GRAD_STOPS: maxGradStops,
  });
}

export function getIsoRefineShader(maxGradStops: number): string {
  return getIsoShader(maxGradStops) + "\n" + inject(isoRefineAppendWgsl, {
    ISO_EDGE_WGSL: isoEdgeWgsl,
  });
}

export function getIsoUpsampleShader(): string {
  return inject(isoUpsampleWgsl, { ISO_EDGE_WGSL: isoEdgeWgsl });
}

export function getBeerShader(maxGradStops: number, maxDensLayers: number): string {
  return inject(beerWgsl, {
    GRADIENT_WGSL: gradientBlock(maxGradStops),
    MAX_GRAD_STOPS: maxGradStops,
    MAX_DENS_LAYERS: maxDensLayers,
  });
}

export function getBeerRefineShader(maxGradStops: number, maxDensLayers: number): string {
  return getBeerShader(maxGradStops, maxDensLayers) + "\n" + inject(beerRefineAppendWgsl, {
    ISO_EDGE_WGSL: isoEdgeWgsl,
  });
}

export function getGridShader(): string {
  return gridWgsl;
}

export function getAxisLabelShader(): string {
  return axisLabelWgsl;
}

export function getFxaaShader(): string {
  return fxaaWgsl;
}

export function getBlitShader(): string {
  return inject(blitWgsl, { ISO_EDGE_WGSL: isoEdgeWgsl });
}

export function getFlowParticlesShader(): string {
  return inject(flowParticlesWgsl, {
    MAX_GRAD_STOPS,
    MAX_FLOW_TRAIL_STEPS,
  });
}
