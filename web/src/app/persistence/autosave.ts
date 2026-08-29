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

export type AutosaveStatus = "saved" | "saving" | "dirty";

let timer = 0;
let writeGen = 0;
let status: AutosaveStatus = "saved";

function setStatus(next: AutosaveStatus) {
  status = next;
  if (!els.autosaveStatus) return;
  if (next === "saving") {
    els.autosaveStatus.textContent = "Saving…";
    els.autosaveStatus.dataset.state = "saving";
  } else if (next === "saved") {
    els.autosaveStatus.textContent = "Saved";
    els.autosaveStatus.dataset.state = "saved";
  } else {
    els.autosaveStatus.textContent = "Unsaved changes";
    els.autosaveStatus.dataset.state = "dirty";
  }
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
      setStatus("saved");
    }
  } catch (e) {
    console.warn("[autosave] write failed", e);
    if (gen === writeGen) setStatus("dirty");
  }
}

/** Immediate persist (e.g. after file import). */
export async function persistNow() {
  if (timer) {
    clearTimeout(timer);
    timer = 0;
  }
  const gen = ++writeGen;
  setStatus("saving");
  const doc = serializeDocument();
  const json = JSON.stringify(doc);
  await writeDocument(json, doc.revision, doc.savedAt);
  if (gen === writeGen) {
    writeMeta(doc.savedAt, doc.revision);
    setStatus("saved");
  }
}

/** Silent restore on load; returns true when a document was applied. */
export async function restoreAutosave(): Promise<boolean> {
  const doc = await readDocument();
  if (!doc) return false;
  await applyDocument(doc);
  setStatus("saved");
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

export function initAutosave() {
  controls.addEventListener("end", () => scheduleAutosave());
  bindInput(els.deg);
  bindInput(els.scale);
  bindInput(els.steps);
  bindInput(els.boxSize);
  bindInput(els.marchDownscale);
  bindInput(els.isoInterp);
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
