import gradientWgsl from "./common/gradient.wgsl?raw";
import isoHermiteWgsl from "./isoHermite.wgsl?raw";
import isoTrilinearWgsl from "./isoTrilinear.wgsl?raw";
import beerWgsl from "./beer.wgsl?raw";
import gridWgsl from "./grid.wgsl?raw";
import axisLabelWgsl from "./axisLabel.wgsl?raw";
import fxaaWgsl from "./fxaa.wgsl?raw";
import ssaoWgsl from "./ssao.wgsl?raw";

function inject(src: string, vars: Record<string, string | number>): string {
  return Object.entries(vars).reduce(
    (s, [k, v]) => s.replaceAll(`{{${k}}}`, String(v)),
    src,
  );
}

function gradientBlock(maxGradStops: number): string {
  return inject(gradientWgsl, { MAX_GRAD_STOPS: maxGradStops });
}

export function getIsoShader(useHermite: boolean, maxGradStops: number): string {
  const base = useHermite ? isoHermiteWgsl : isoTrilinearWgsl;
  return inject(base, {
    GRADIENT_WGSL: gradientBlock(maxGradStops),
    MAX_GRAD_STOPS: maxGradStops,
  });
}

export function getBeerShader(maxGradStops: number, maxDensLayers: number): string {
  return inject(beerWgsl, {
    GRADIENT_WGSL: gradientBlock(maxGradStops),
    MAX_GRAD_STOPS: maxGradStops,
    MAX_DENS_LAYERS: maxDensLayers,
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

export function getSsaoShader(): string {
  return ssaoWgsl;
}
