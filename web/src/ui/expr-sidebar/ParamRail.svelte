<script lang="ts">
  import { onMount } from "svelte";
  import type { ExprItem } from "../../types/models.js";
  import {
    getParam,
    setParamValue,
    updateParam,
  } from "../../model/params.js";
  import { updateExprSilent } from "../../model/expressions.js";
  import { mountLiquidThumb, syncLiquidThumb } from "../liquidSlider.js";
  import { fmtNum, isMathFieldFocused } from "./helpers.ts";

  interface Props {
    item: ExprItem;
    paramName: string;
    paramTick?: number;
    onParamChange: () => void;
    getMathField: () => MathfieldElement | null;
  }

  let { item, paramName, paramTick = 0, onParamChange, getMathField }: Props = $props();

  let minEl: HTMLInputElement | undefined = $state();
  let maxEl: HTMLInputElement | undefined = $state();
  let sliderEl: HTMLInputElement | undefined = $state();
  let trackEl: HTMLDivElement | undefined = $state();
  let zeroEl: HTMLSpanElement | undefined = $state();

  export function syncFromParam() {
    const p = getParam(paramName);
    if (!p) return;
    const min = p.min;
    const max = p.max;
    if (sliderEl && document.activeElement !== sliderEl) {
      sliderEl.min = String(min);
      sliderEl.max = String(max);
      sliderEl.step = p.animating && !p.driven ? "any" : String(p.step);
      sliderEl.disabled = !!p.driven;
      sliderEl.value = String(p.value);
      sliderEl.title = p.driven
        ? `value ${fmtNum(p.value)} (driven)`
        : `${paramName} = ${fmtNum(p.value)}`;
      syncLiquidThumb(sliderEl);
    }
    if (minEl && document.activeElement !== minEl) minEl.value = fmtNum(min);
    if (maxEl && document.activeElement !== maxEl) maxEl.value = fmtNum(max);
    if (zeroEl) {
      const span = max - min;
      const zeroPct = span > 1e-12 ? ((0 - min) / span) * 100 : 50;
      const show = min < 0 && max > 0;
      zeroEl.hidden = !show;
      if (show) zeroEl.style.left = `${Math.min(100, Math.max(0, zeroPct))}%`;
    }
  }

  $effect(() => {
    void paramTick;
    syncFromParam();
  });

  onMount(() => {
    if (trackEl && sliderEl) mountLiquidThumb(trackEl, sliderEl);
    syncFromParam();
  });

  function onMinChange() {
    const n = Number(minEl?.value);
    if (!Number.isFinite(n)) {
      syncFromParam();
      return;
    }
    updateParam(paramName, { min: n });
    updateExprSilent(item.id, { sliderMin: n });
    syncFromParam();
    onParamChange();
  }

  function onMaxChange() {
    const n = Number(maxEl?.value);
    if (!Number.isFinite(n)) {
      syncFromParam();
      return;
    }
    updateParam(paramName, { max: n });
    updateExprSilent(item.id, { sliderMax: n });
    syncFromParam();
    onParamChange();
  }

  function onSliderInput() {
    if (!sliderEl) return;
    const next = setParamValue(paramName, Number(sliderEl.value), {
      stopAnim: true,
      rewriteLatex: true,
    });
    if (!next) return;
    const mf = getMathField();
    updateExprSilent(item.id, { latex: next.latex, sliderAnimating: false });
    if (mf && !isMathFieldFocused(mf)) {
      if (typeof mf.setValue === "function") {
        mf.setValue(next.latex, { silenceNotifications: true });
      } else {
        mf.value = next.latex;
      }
    }
    syncFromParam();
    onParamChange();
  }

  const p = $derived.by(() => {
    void paramTick;
    return getParam(paramName);
  });
</script>

{#if p}
  <div class="expr-param-block" data-param-block={paramName}>
    <div class="expr-param-rail">
      <input
        bind:this={minEl}
        type="text"
        inputmode="decimal"
        class="expr-param-min"
        title="Minimum"
        onchange={onMinChange}
      />
      <div class="expr-param-track" bind:this={trackEl}>
        <span class="expr-param-zero" bind:this={zeroEl} aria-hidden="true"></span>
        <input
          bind:this={sliderEl}
          type="range"
          class="expr-param-slider"
          disabled={p.driven}
          oninput={onSliderInput}
        />
      </div>
      <input
        bind:this={maxEl}
        type="text"
        inputmode="decimal"
        class="expr-param-max"
        title="Maximum"
        onchange={onMaxChange}
      />
    </div>
  </div>
{/if}
