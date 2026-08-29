import { getParam } from "../../model/params.js";

export type ParamPlayChrome = {
  visible: boolean;
  animating: boolean;
  driven: boolean;
  disabled: boolean;
  icon: "▶" | "⏸";
  title: string;
};

const HIDDEN: ParamPlayChrome = {
  visible: false,
  animating: false,
  driven: false,
  disabled: true,
  icon: "▶",
  title: "",
};

/**
 * Play-button chrome for a parameter row. Pass `tick` from reactive UI state so
 * callers re-read the params map after external mutations.
 */
export function readParamPlayChrome(paramName: string, tick = 0): ParamPlayChrome {
  void tick;
  const p = getParam(paramName);
  if (!p) return HIDDEN;

  const driven = !!p.driven;
  const animating = !!p.animating && !driven;

  return {
    visible: true,
    animating,
    driven,
    disabled: driven,
    icon: animating ? "⏸" : "▶",
    title: driven
      ? "Driven by equation"
      : animating
        ? "Pause animation"
        : "Animate between min and max",
  };
}
