import { mount } from "svelte";
import ExprSidebar from "./ExprSidebar.svelte";
import MobileExprFooter from "./MobileExprFooter.svelte";
import type { ExprListApi } from "../../types/models.js";

export interface MountExprListOpts {
  root: HTMLElement;
  footerRoot?: HTMLElement | null;
  onExprChange: () => void;
  onStructuralChange: () => void;
  onColorChange?: () => void;
  onParamChange?: () => void;
  onCollapsePanel?: () => void;
  onReturnToViewport?: () => void;
}

/** Mount the expression sidebar (Svelte) and optional mobile scene footer. */
export function mountExprList(opts: MountExprListOpts): ExprListApi {
  const { root, footerRoot, onCollapsePanel, onReturnToViewport, ...props } = opts;
  root.replaceChildren();

  type SidebarInst = InstanceType<typeof ExprSidebar> & {
    render: (focus?: { id: string; pos?: number } | null) => void;
    syncAllParamSliders: () => void;
    syncParamChrome: () => boolean;
    clearAll: () => void;
  };
  const inst = mount(ExprSidebar, {
    target: root,
    props,
  }) as SidebarInst;

  type FooterInst = InstanceType<typeof MobileExprFooter> & {
    syncFromList: () => void;
  };
  let footerInst: FooterInst | null = null;
  if (footerRoot) {
    footerRoot.replaceChildren();
    footerInst = mount(MobileExprFooter, {
      target: footerRoot,
      props: {
        onExprChange: props.onExprChange,
        onStructuralChange: props.onStructuralChange,
        onColorChange: props.onColorChange,
        onParamChange: props.onParamChange ?? props.onExprChange,
        onSelectionSync: () => inst.syncParamChrome(),
        onReturnToViewport: onReturnToViewport ?? onCollapsePanel,
      },
    }) as FooterInst;
  }

  return {
    render: (focus) => {
      inst.render(focus);
      footerInst?.syncFromList();
    },
    syncAllParamSliders: () => inst.syncAllParamSliders(),
    syncParamChrome: () => inst.syncParamChrome(),
    clearAll: () => inst.clearAll(),
  };
}
