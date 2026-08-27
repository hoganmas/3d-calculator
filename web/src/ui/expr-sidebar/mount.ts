import { mount } from "svelte";
import ExprSidebar from "./ExprSidebar.svelte";
import type { ExprListApi } from "../../types/models.js";

export interface MountExprListOpts {
  root: HTMLElement;
  onExprChange: () => void;
  onStructuralChange: () => void;
  onColorChange?: () => void;
  onParamChange?: () => void;
}

/** Mount the expression sidebar (Svelte). */
export function mountExprList(opts: MountExprListOpts): ExprListApi {
  const { root, ...props } = opts;
  root.replaceChildren();

  type SidebarInst = InstanceType<typeof ExprSidebar> & ExprListApi;
  const inst = mount(ExprSidebar, { target: root, props }) as SidebarInst;

  return {
    render: (focus) => inst.render(focus),
    syncAllParamSliders: () => inst.syncAllParamSliders(),
    syncParamChrome: () => inst.syncParamChrome() ?? false,
  };
}
