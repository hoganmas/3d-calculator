/**
 * Download, import, and share Laplacian scene documents.
 */
import {
  applyDocument,
  serializeDocument,
  validateDocument,
  type LaplacianDocument,
} from "./document.js";
import { persistNow } from "./autosave.js";

export const LAP_FILE_EXT = ".lap.json";

export function canShareFiles(): boolean {
  if (typeof navigator.share !== "function") return false;
  if (typeof navigator.canShare !== "function") return false;
  try {
    const probe = new File([""], `probe${LAP_FILE_EXT}`, { type: "application/json" });
    return navigator.canShare({ files: [probe] });
  } catch {
    return false;
  }
}

export function downloadDocument(doc?: LaplacianDocument, filename = "scene.lap.json") {
  const payload = doc ?? serializeDocument();
  const json = JSON.stringify(payload, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function readDocumentFromFile(file: File): Promise<LaplacianDocument> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(await file.text());
  } catch {
    throw new Error("Could not read file — invalid JSON");
  }
  return validateDocument(parsed);
}

export async function importDocumentFromFile(file: File) {
  const doc = await readDocumentFromFile(file);
  await applyDocument(doc);
  await persistNow();
}

export async function shareDocument(doc?: LaplacianDocument): Promise<boolean> {
  const payload = doc ?? serializeDocument();
  const json = JSON.stringify(payload, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const file = new File([blob], `scene${LAP_FILE_EXT}`, { type: "application/json" });
  if (typeof navigator.share === "function" && navigator.canShare?.({ files: [file] })) {
    await navigator.share({
      files: [file],
      title: payload.meta?.title ?? "Laplacian scene",
    });
    return true;
  }
  return false;
}

export function exportCurrentDocument(): LaplacianDocument {
  return serializeDocument();
}
