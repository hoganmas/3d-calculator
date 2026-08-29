/**
 * Progressive Lobatto degree ladder for interactive refits.
 * Yields between steps so the UI stays responsive; cancels on new edits.
 */

import {
  lobattoLadderDegrees,
  type LobattoFitState,
} from "../math/chebLobatto.js";
import { els } from "./dom.js";
import { state, FIT_DEBOUNCE_MS } from "./state.js";
import { syncExprCompileState } from "./hud.js";
import { tearLog } from "./tearDebug.js";

let generation = 0;
let stepTimer = 0;
const lobattoByLayer = new Map<string, LobattoFitState>();

/** Enable progressive Lobatto ladder for non-animation refits. */
export const USE_LOBATTO_PROGRESSIVE = true;

export function isProgressiveLobattoEnabled(): boolean {
  return USE_LOBATTO_PROGRESSIVE;
}

export function getLobattoLayerCache(layerId: string): LobattoFitState | null {
  return lobattoByLayer.get(layerId) ?? null;
}

export function setLobattoLayerCache(layerId: string, fit: LobattoFitState): void {
  lobattoByLayer.set(layerId, fit);
}

export function clearLobattoFitCache(): void {
  lobattoByLayer.clear();
}

/** Invalidate any in-flight progressive ladder. */
export function cancelProgressiveFit(): void {
  generation++;
  if (stepTimer) {
    clearTimeout(stepTimer);
    stepTimer = 0;
  }
}

export function currentProgressiveGeneration(): number {
  return generation;
}

type UploadFn = (opts: {
  fromAnim?: boolean;
  fitDeg?: number;
  progressive?: boolean;
  progressiveFinal?: boolean;
}) => void;

export function scheduleProgressiveUploadFit(
  uploadFit: UploadFn,
  delay = FIT_DEBOUNCE_MS,
  opts: { fromAnim?: boolean } = {},
): void {
  state.pendingFitOpts = opts;
  if (state.fitTimer) clearTimeout(state.fitTimer);
  state.fitTimer = window.setTimeout(() => {
    state.fitTimer = 0;
    const fitOpts = state.pendingFitOpts;
    state.pendingFitOpts = {};
    if (!syncExprCompileState()) return;

    const fromAnim = !!fitOpts.fromAnim;
    const targetDeg = Number(els.deg.value);

    if (
      !fromAnim &&
      isProgressiveLobattoEnabled() &&
      targetDeg > 4 &&
      lobattoLadderDegrees(targetDeg).length > 1
    ) {
      startProgressiveLadder(uploadFit, targetDeg, fitOpts);
      return;
    }

    cancelProgressiveFit();
    clearLobattoFitCache();
    uploadFit(fitOpts);
  }, delay);
}

function startProgressiveLadder(
  uploadFit: UploadFn,
  targetDeg: number,
  baseOpts: { fromAnim?: boolean },
): void {
  cancelProgressiveFit();
  const runGen = generation;
  clearLobattoFitCache();

  const steps = lobattoLadderDegrees(targetDeg);
  let stepIdx = 0;

  const pump = () => {
    if (runGen !== generation) return;
    const fitDeg = steps[stepIdx]!;
    const isFinal = stepIdx >= steps.length - 1;
    tearLog("progressive-step", {
      fitDeg,
      targetDeg,
      stepIdx,
      stepCount: steps.length,
      isFinal,
      sceneM: fitDeg + 1,
    });
    uploadFit({
      ...baseOpts,
      fitDeg,
      progressive: true,
      progressiveFinal: isFinal,
    });
    stepIdx++;
    if (stepIdx < steps.length && runGen === generation) {
      stepTimer = window.setTimeout(pump, 0);
    }
  };

  pump();
}
