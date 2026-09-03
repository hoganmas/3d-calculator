<script lang="ts">
  import type { ExprItem } from "../../types/models.js";
  import { getParam, toggleParamAnimate } from "../../model/params.js";
  import { updateExprSilent } from "../../model/expressions.js";
  // Aliased: this component uses the $state rune, and Svelte 5's compiler
  // gets confused if a plain import is also named `state`.
  import { state as appState } from "../../app/state.js";
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
    // playChrome is derived from the paramTick *prop*, which only advances
    // when the parent re-renders it — normally driven by loop.ts bumping it
    // after a value change on the per-frame animation tick. Pausing flips
    // `animating` without changing the value, so that tick never fires
    // again for this param and the button silently keeps showing "playing"
    // forever unless something pokes the shared tick directly here.
    appState.exprListApi?.syncAllParamSliders?.();
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
    // Only forces an upload when starting to play (kicks off/ensures the
    // keyframe cache). Pausing doesn't touch the value, so there's nothing to
    // refit — calling onParamChange() here schedules uploadFit({fromAnim:
    // false}), and fromAnim=false disables the keyframe-blend branch in
    // pipeline.ts entirely, dropping straight to a from-scratch non-keyframe
    // fit that starts at the bottom of the ladder. That discarded whatever
    // high-quality keyframe data was already on screen the instant you
    // paused, and the background pump (loop.ts) already keeps that keyframe
    // data improving on its own without needing this call.
    if (next.animating) onParamChange();
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
