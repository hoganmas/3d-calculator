<script lang="ts">
  import type { ExprItem } from "../../types/models.js";
  import { getParam, toggleParamAnimate } from "../../model/params.js";
  import { updateExprSilent } from "../../model/expressions.js";
  import { ANIM_OPTS_ICON, PAUSE_ICON, PLAY_ICON } from "./helpers.ts";
  import {
    openAnimOptions,
    isAnimPopoverOpen,
    closeAnimPopover,
  } from "./popovers.ts";
  import { readParamPlayChrome } from "./paramChrome.ts";

  interface Props {
    item: ExprItem;
    paramName: string;
    paramTick?: number;
    onParamChange: () => void;
    onAnimSync?: () => void;
  }

  let {
    item,
    paramName,
    paramTick = 0,
    onParamChange,
    onAnimSync,
  }: Props = $props();

  let optsBtn: HTMLButtonElement | undefined = $state();

  function syncOptsChrome() {
    const p = getParam(paramName);
    if (!optsBtn || !p) return;
    optsBtn.disabled = !!p.driven;
    const mode = p.animMode === "loop" ? "loop" : "back & forth";
    optsBtn.title = p.driven
      ? "Driven by equation"
      : `Animation options (${mode}, ${p.speed}×)`;
  }

  function syncAll() {
    syncOptsChrome();
    onAnimSync?.();
  }

  $effect(() => {
    void paramTick;
    syncOptsChrome();
  });

  function onPlayClick(ev: MouseEvent) {
    ev.stopPropagation();
    const next = toggleParamAnimate(paramName);
    if (!next) return;
    updateExprSilent(item.id, { sliderAnimating: next.animating, sliderPhase: next.phase });
    syncAll();
    onParamChange();
  }

  function onOptsClick(ev: MouseEvent) {
    ev.stopPropagation();
    if (isAnimPopoverOpen(paramName)) {
      closeAnimPopover();
      return;
    }
    if (optsBtn) {
      openAnimOptions(optsBtn, item, paramName, onParamChange, syncAll);
    }
  }

  const p = $derived.by(() => {
    void paramTick;
    return getParam(paramName);
  });

  const playChrome = $derived.by(() => readParamPlayChrome(paramName, paramTick));
</script>

{#if p}
  <div class="expr-param-side" data-param-side={paramName}>
    <button
      type="button"
      class="expr-param-play"
      class:on={playChrome.animating}
      disabled={playChrome.disabled}
      title={playChrome.title}
      aria-label={playChrome.title}
      onclick={onPlayClick}
    >
      {@html playChrome.icon === "pause" ? PAUSE_ICON : PLAY_ICON}
    </button>
    <button
      bind:this={optsBtn}
      type="button"
      class="expr-param-anim-opts"
      disabled={p.driven}
      aria-label="Animation options"
      title="Animation options"
      onclick={onOptsClick}
    >
      {@html ANIM_OPTS_ICON}
    </button>
  </div>
{/if}
