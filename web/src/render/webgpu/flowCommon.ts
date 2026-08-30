import { FLOW_PARTICLE_SPAWN_GRID_SPACING } from "../../math/fitVector.js";
import { state } from "../../app/state.js";

export { FLOW_PARTICLE_SPAWN_GRID_SPACING };

export function effectiveFlowDt(): number {
  return Math.max(state.flowDt, 1e-6) * Math.max(state.flowSpeed, 0.01);
}

export function effectiveFlowVMax(): number {
  if (state.flowVMax > 1e-8) return state.flowVMax;
  return FLOW_PARTICLE_SPAWN_GRID_SPACING / effectiveFlowDt();
}
