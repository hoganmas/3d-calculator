/**
 * Headset-agnostic XR locomotion: squeeze-grab to move/rotate the volume root,
 * trigger (select) to recenter the fit box in front of the viewer.
 */
import * as THREE from "three";
import type { WebGLRenderer } from "three";
import { scene, xrWorld } from "../scene.js";

/** Default seated placement: volume ~1.2 m up, 3 m forward in local space. */
export const XR_DEFAULT_POSITION = new THREE.Vector3(0, 1.2, -3);

const _offset = new THREE.Matrix4();
const _controllerInv = new THREE.Matrix4();
const _world = new THREE.Matrix4();

let grabController: THREE.Object3D | null = null;
let navReady = false;

/** Place the calculator volume in front of a seated / standing viewer. */
export function resetXrView() {
  xrWorld.position.copy(XR_DEFAULT_POSITION);
  xrWorld.rotation.set(0, 0, 0);
  xrWorld.scale.set(1, 1, 1);
  xrWorld.updateMatrixWorld(true);
}

/** Identity transform for desktop OrbitControls (volume at world origin). */
export function resetXrWorldDesktop() {
  xrWorld.position.set(0, 0, 0);
  xrWorld.rotation.set(0, 0, 0);
  xrWorld.scale.set(1, 1, 1);
  xrWorld.updateMatrixWorld(true);
  grabController = null;
}

function onSqueezeStart(this: THREE.Object3D) {
  grabController = this;
  _controllerInv.copy(this.matrixWorld).invert();
  _offset.multiplyMatrices(_controllerInv, xrWorld.matrixWorld);
}

function onSqueezeEnd(this: THREE.Object3D) {
  if (grabController === this) grabController = null;
}

function onSelectStart() {
  resetXrView();
}

/** Per-frame: keep grabbed world glued to the controller. */
export function tickXrNav() {
  if (!grabController) return;
  _world.multiplyMatrices(grabController.matrixWorld, _offset);
  _world.decompose(xrWorld.position, xrWorld.quaternion, xrWorld.scale);
}

export function initXrNav(renderer: WebGLRenderer) {
  if (navReady) return;
  navReady = true;

  for (let i = 0; i < 2; i++) {
    const controller = renderer.xr.getController(i);
    controller.addEventListener("squeezestart", onSqueezeStart);
    controller.addEventListener("squeezeend", onSqueezeEnd);
    controller.addEventListener("selectstart", onSelectStart);
    scene.add(controller);
  }
}
