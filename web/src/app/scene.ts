import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { onThemeChange, readThemeColors } from "../ui/theme.js";
import { createLavaBackground } from "../render/background.js";
import { syncClipGpuWorldGrid, applyClipGpuTheme } from "../render/webgpu/march.js";
import { els } from "./dom.js";
import { state } from "./state.js";

export const renderer = new THREE.WebGLRenderer({
  antialias: false,
  alpha: true,
  powerPreference: "low-power",
});
renderer.setPixelRatio(1);
renderer.setClearColor(0x000000, 0);
els.viewport.appendChild(renderer.domElement);

/** Axis labels for the WebGL fallback (always-on-top). WebGPU draws them
 *  in the march pass with iso depth-test instead. */
export const labelRenderer = new THREE.WebGLRenderer({
  antialias: false,
  alpha: true,
  powerPreference: "low-power",
});
labelRenderer.setPixelRatio(1);
labelRenderer.setClearColor(0x000000, 0);
labelRenderer.domElement.className = "axis-labels";
labelRenderer.domElement.style.cssText =
  "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:2;";
els.viewport.appendChild(labelRenderer.domElement);
export const labelScene = new THREE.Scene();

export let themeColors = readThemeColors();
export const lavaBg = createLavaBackground(themeColors);
applyClipGpuTheme(themeColors);

export const scene = new THREE.Scene();
scene.add(lavaBg.mesh);
export const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
/** World axes: +x right, +y forward, +z up. */
camera.up.set(0, 0, 1);
export const DEFAULT_CAMERA_POSITION = new THREE.Vector3(5.2, 6.8, 4.0);
camera.position.copy(DEFAULT_CAMERA_POSITION);

export const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = true;
controls.screenSpacePanning = true;
controls.target.set(0, 0, 0);
renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());

export const boxMat = new THREE.LineBasicMaterial({ color: themeColors.boxEdge });
export const boxHelper = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(4, 4, 4)),
  boxMat,
);
scene.add(boxHelper);

/** World reference frame: unit grids + RGB axes (calculator-style). */
export const worldGrid = new THREE.Group();
worldGrid.renderOrder = -1;
scene.add(worldGrid);
/** Axis letter sprites — own overlay canvas so volumes never cover them. */
export const worldLabels = new THREE.Group();
labelScene.add(worldLabels);

let boundClipUniforms: THREE.ShaderMaterial["uniforms"] | null = null;

export function bindClipUniforms(clipUniforms: THREE.ShaderMaterial["uniforms"]) {
  boundClipUniforms = clipUniforms;
}

function makeAxisLabel(
  text: string,
  color: string,
  position: THREE.Vector3,
  labelStroke: string,
) {
  // High-res canvas so sprites stay sharp under orbit / retina.
  const css = 128;
  const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
  const size = Math.round(css * dpr);
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return new THREE.Sprite(new THREE.SpriteMaterial());
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.clearRect(0, 0, size, size);
  ctx.scale(dpr, dpr);
  ctx.font = "600 72px 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  // Soft stroke reduces shimmer / jagged edges when minified by the GPU.
  ctx.lineWidth = 3;
  ctx.strokeStyle = labelStroke || "rgba(26, 18, 40, 0.58)";
  ctx.strokeText(text, css / 2, css / 2 + 1);
  ctx.fillStyle = color;
  ctx.fillText(text, css / 2, css / 2 + 1);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  const mat = new THREE.SpriteMaterial({
    map: tex,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    sizeAttenuation: true,
  });
  const spr = new THREE.Sprite(mat);
  spr.scale.set(0.42, 0.42, 0.42);
  spr.position.copy(position);
  spr.renderOrder = 10;
  return spr;
}

function styleGrid(grid: THREE.GridHelper, opacity: number) {
  const mats = Array.isArray(grid.material) ? grid.material : [grid.material];
  for (const m of mats) {
    m.transparent = true;
    m.opacity = opacity;
    m.depthWrite = false;
  }
}

export function rebuildWorldGrid(half: number) {
  while (worldGrid.children.length) {
    const child = worldGrid.children.pop() as THREE.Mesh | undefined;
    child?.geometry?.dispose?.();
    if (child?.material) {
      const mats = Array.isArray(child.material) ? child.material : [child.material];
      for (const m of mats) {
        const tex = (m as THREE.MeshStandardMaterial).map;
        tex?.dispose?.();
        m.dispose?.();
      }
    }
  }
  while (worldLabels.children.length) {
    const child = worldLabels.children.pop() as THREE.Sprite | undefined;
    const mat = child?.material as THREE.SpriteMaterial | undefined;
    mat?.map?.dispose?.();
    mat?.dispose?.();
  }

  const h = Math.max(0.5, half);
  // Grid a bit past the fit box; aim for ~1 world-unit cells.
  const extent = Math.ceil(h + 0.5);
  const size = extent * 2;
  const divisions = Math.max(2, size);
  const tc = readThemeColors();
  const major = tc.gridMajor;
  const minor = tc.gridMinor;

  const gridFloor = new THREE.GridHelper(size, divisions, major, minor);
  gridFloor.rotation.x = -Math.PI / 2;
  styleGrid(gridFloor, 0.55);
  worldGrid.add(gridFloor);

  const gridXZ = new THREE.GridHelper(size, divisions, major, minor);
  styleGrid(gridXZ, 0.35);
  worldGrid.add(gridXZ);

  const gridYZ = new THREE.GridHelper(size, divisions, major, minor);
  gridYZ.rotation.z = Math.PI / 2;
  styleGrid(gridYZ, 0.35);
  worldGrid.add(gridYZ);

  const axisLen = extent + 0.25;
  const axisPositions = new Float32Array([
    0, 0, 0, axisLen, 0, 0, // +X
    0, 0, 0, 0, axisLen, 0, // +Y
    0, 0, 0, 0, 0, axisLen, // +Z
  ]);
  const axisColors = new Float32Array([
    ...tc.axisXRgb, ...tc.axisXRgb,
    ...tc.axisYRgb, ...tc.axisYRgb,
    ...tc.axisZRgb, ...tc.axisZRgb,
  ]);
  const axisGeo = new THREE.BufferGeometry();
  axisGeo.setAttribute("position", new THREE.BufferAttribute(axisPositions, 3));
  axisGeo.setAttribute("color", new THREE.BufferAttribute(axisColors, 3));
  const axes = new THREE.LineSegments(
    axisGeo,
    new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      depthWrite: false,
    }),
  );
  worldGrid.add(axes);

  const tip = extent + 0.45;
  worldLabels.add(makeAxisLabel("x", tc.axisX, new THREE.Vector3(tip, 0, 0), tc.labelStroke));
  worldLabels.add(makeAxisLabel("y", tc.axisY, new THREE.Vector3(0, tip, 0), tc.labelStroke));
  worldLabels.add(makeAxisLabel("z", tc.axisZ, new THREE.Vector3(0, 0, tip), tc.labelStroke));

  // WebGPU path draws the same grid against iso depth (no texture copy).
  syncClipGpuWorldGrid(h);
}

export function resetCameraView() {
  camera.up.set(0, 0, 1);
  camera.position.copy(DEFAULT_CAMERA_POSITION);
  controls.target.set(0, 0, 0);
  controls.update();
  state.clipDirty = true;
}

/** Fit / march use half-extent h; UI “box size” is full edge length S = 2h. */
export function setBoxSize(size: number) {
  const s = Math.max(1e-6, size);
  const h = 0.5 * s;
  boxHelper.geometry.dispose();
  boxHelper.geometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(s, s, s));
  if (boundClipUniforms) boundClipUniforms.uHalf.value = h;
  rebuildWorldGrid(h);
}

export function applyThemeToScene() {
  themeColors = readThemeColors();
  lavaBg.setColors(themeColors);
  boxMat.color.setHex(themeColors.boxEdge);
  if (boundClipUniforms) {
    boundClipUniforms.uAbsorbColor.value.setRGB(...themeColors.beerAbsorb);
    boundClipUniforms.uEmitColor.value.setRGB(...themeColors.beerEmit);
  }
  applyClipGpuTheme(themeColors);
  rebuildWorldGrid(boundClipUniforms?.uHalf?.value ?? 2);
  state.clipDirty = true;
}

export function initScene() {
  rebuildWorldGrid(2);

  onThemeChange((_split, pref) => {
    if (els.themePref && els.themePref.value !== pref) els.themePref.value = pref;
  });
  onThemeChange(() => applyThemeToScene());
  // initTheme() runs before listeners are registered; sync scene colors now.
  applyThemeToScene();
}
