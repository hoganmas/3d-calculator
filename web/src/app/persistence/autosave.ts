/**
 * Debounced IndexedDB autosave + silent restore on load.
 */
import { controls } from "../scene.js";
import { els } from "../dom.js";
import {
  applyDocument,
  getDocumentRevision,
  isPersistSuspended,
  serializeDocument,
  type LaplacianDocument,
} from "./document.js";
import { readDocument, writeDocument } from "./storage.js";

const DEBOUNCE_MS = 1500;
const LAST_SAVE_KEY = "laplacian-last-save-at";
const LAST_REV_KEY = "laplacian-last-revision";

export type AutosaveStatus = "saved" | "saving" | "dirty" | "error";

let timer = 0;
let writeGen = 0;
let status: AutosaveStatus = "saved";
let unloadFlushBound = false;

function statusTargets(): HTMLElement[] {
  return [els.autosaveStatus, els.autosaveStatusBar].filter(
    (el): el is HTMLElement => el != null,
  );
}

function setStatus(next: AutosaveStatus, detail?: string) {
  if (next !== "error") status = next;

  let text: string;
  let state: string;
  switch (next) {
    case "saving":
      text = "Saving…";
      state = "saving";
      break;
    case "saved":
      text = "Saved";
      state = "saved";
      if (detail) console.info("[autosave]", detail);
      else console.info("[autosave] saved");
      break;
    case "error":
      text = detail ?? "Save failed";
      state = "error";
      console.warn("[autosave]", detail ?? "save failed");
      break;
    default:
      text = "Unsaved changes";
      state = "dirty";
      break;
  }

  for (const el of statusTargets()) {
    el.textContent = text;
    el.dataset.state = state;
  }
}

/** Surface import/validation errors in all status targets. */
export function setAutosaveError(message: string) {
  setStatus("error", message);
}

function writeMeta(savedAt: string, revision: number) {
  try {
    localStorage.setItem(LAST_SAVE_KEY, savedAt);
    localStorage.setItem(LAST_REV_KEY, String(revision));
  } catch {
    /* ignore */
  }
}

export function getAutosaveStatus() {
  return status;
}

export function getCurrentRevision() {
  return getDocumentRevision();
}

function hasPendingSave() {
  return status === "dirty" || timer !== 0;
}

export function scheduleAutosave() {
  if (isPersistSuspended()) return;
  setStatus("dirty");
  if (timer) clearTimeout(timer);
  timer = window.setTimeout(() => {
    timer = 0;
    void flushAutosave();
  }, DEBOUNCE_MS);
}

async function flushAutosave() {
  const gen = ++writeGen;
  setStatus("saving");
  try {
    const doc = serializeDocument();
    const json = JSON.stringify(doc);
    await writeDocument(json, doc.revision, doc.savedAt);
    if (gen === writeGen) {
      writeMeta(doc.savedAt, doc.revision);
      setStatus("saved", `saved rev ${doc.revision} · ${doc.savedAt}`);
    }
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.warn("[autosave] write failed", e);
    if (gen === writeGen) setStatus("error", msg);
  }
}

/** Flush debounced edits immediately (tab hide, import, manual). */
export async function flushPendingAutosave() {
  if (isPersistSuspended() || !hasPendingSave()) return;
  if (timer) {
    clearTimeout(timer);
    timer = 0;
  }
  await flushAutosave();
}

/** Immediate persist (e.g. after file import). */
export async function persistNow() {
  if (timer) {
    clearTimeout(timer);
    timer = 0;
  }
  const gen = ++writeGen;
  setStatus("saving");
  try {
    const doc = serializeDocument();
    const json = JSON.stringify(doc);
    await writeDocument(json, doc.revision, doc.savedAt);
    if (gen === writeGen) {
      writeMeta(doc.savedAt, doc.revision);
      setStatus("saved", `saved rev ${doc.revision} · ${doc.savedAt}`);
    }
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.warn("[autosave] write failed", e);
    if (gen === writeGen) setStatus("error", msg);
  }
}

/** Silent restore on load; returns true when a document was applied. */
export async function restoreAutosave(): Promise<boolean> {
  const doc = await readDocument();
  if (!doc) return false;
  await applyDocument(doc);
  setStatus("saved", `restored rev ${doc.revision} · ${doc.savedAt}`);
  return true;
}

export async function applyImportedDocument(doc: LaplacianDocument) {
  await applyDocument(doc);
  await persistNow();
}

function bindInput(el: HTMLElement | null | undefined) {
  el?.addEventListener("input", scheduleAutosave);
  el?.addEventListener("change", scheduleAutosave);
}

function bindUnloadFlush() {
  if (unloadFlushBound) return;
  unloadFlushBound = true;

  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState !== "hidden") return;
    void flushPendingAutosave();
  });

  window.addEventListener("pagehide", () => {
    if (isPersistSuspended() || !hasPendingSave()) return;
    if (timer) {
      clearTimeout(timer);
      timer = 0;
    }
    void flushAutosave();
  });
}

export function initAutosave() {
  bindUnloadFlush();
  controls.addEventListener("end", () => scheduleAutosave());
  bindInput(els.deg);
  bindInput(els.scale);
  bindInput(els.steps);
  bindInput(els.boxSize);
  bindInput(els.marchDownscale);
  bindInput(els.preset);
  bindInput(els.flowAlpha);
  bindInput(els.flowGridMode);
  bindInput(els.flowNoiseScale);
  bindInput(els.flowDt);
  bindInput(els.flowSpeed);
  bindInput(els.flowVMax);
  bindInput(els.flowOpacity);
  bindInput(els.flowAgeMax);
  bindInput(els.flowVizMode);
  bindInput(els.flowParticleCount);
  bindInput(els.flowTrailSteps);
  bindInput(els.flowTrailWidth);
  setStatus("saved");
}
