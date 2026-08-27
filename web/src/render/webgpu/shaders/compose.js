import gradientWgsl from "./common/gradient.wgsl?raw";
import isoHermiteWgsl from "./isoHermite.wgsl?raw";
import isoTrilinearWgsl from "./isoTrilinear.wgsl?raw";
import beerWgsl from "./beer.wgsl?raw";
import gridWgsl from "./grid.wgsl?raw";
import axisLabelWgsl from "./axisLabel.wgsl?raw";
import fxaaWgsl from "./fxaa.wgsl?raw";
import ssaoWgsl from "./ssao.wgsl?raw";

/** @param {string} src @param {Record<string, string | number>} vars */
function inject(src, vars) {
  return Object.entries(vars).reduce(
    (s, [k, v]) => s.replaceAll(`{{${k}}}`, String(v)),
    src,
  );
}

function gradientBlock(maxGradStops) {
  return inject(gradientWgsl, { MAX_GRAD_STOPS: maxGradStops });
}

/** @param {boolean} useHermite @param {number} maxGradStops */
export function getIsoShader(useHermite, maxGradStops) {
  const base = useHermite ? isoHermiteWgsl : isoTrilinearWgsl;
  return inject(base, {
    GRADIENT_WGSL: gradientBlock(maxGradStops),
    MAX_GRAD_STOPS: maxGradStops,
  });
}

/** @param {number} maxGradStops @param {number} maxDensLayers */
export function getBeerShader(maxGradStops, maxDensLayers) {
  return inject(beerWgsl, {
    GRADIENT_WGSL: gradientBlock(maxGradStops),
    MAX_GRAD_STOPS: maxGradStops,
    MAX_DENS_LAYERS: maxDensLayers,
  });
}

export function getGridShader() {
  return gridWgsl;
}

export function getAxisLabelShader() {
  return axisLabelWgsl;
}

export function getFxaaShader() {
  return fxaaWgsl;
}

export function getSsaoShader() {
  return ssaoWgsl;
}
