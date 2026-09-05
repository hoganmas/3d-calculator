/**
 * Share dialog: opens immediately on clicking Share and shows the rendered
 * OG image preview plus copy/share options, instead of the button silently
 * rendering in the background and then handing off to (or standing in for)
 * the native share sheet with no visible feedback in between.
 *
 * shareCapture.ts is only ever reached via dynamic import here, matching
 * exprShare.ts's own dynamic import of it — it touches the live canvas/
 * scene, which isn't Node-test-safe.
 */
import { buildExpressionShareUrl } from "./persistence/exprShare.js";
import { setAutosaveError } from "./persistence/autosave.js";

const DIALOG_ID = "shareDialog";
const CLOSE_ID = "closeShareDialog";
const PREVIEW_IMG_ID = "sharePreviewImg";
const PREVIEW_LOADING_ID = "sharePreviewLoading";
const URL_INPUT_ID = "shareUrlInput";
const COPY_BTN_ID = "shareCopyBtn";
const NATIVE_BTN_ID = "shareNativeBtn";

let wired = false;
let previewObjectUrl: string | null = null;
// Bumped on every open; an in-flight render checks it before touching the
// DOM so a stale result from a closed/reopened dialog can't write over
// whatever the current one is showing.
let openToken = 0;

function dialogEl() {
  return document.getElementById(DIALOG_ID) as HTMLDialogElement | null;
}
function previewImgEl() {
  return document.getElementById(PREVIEW_IMG_ID) as HTMLImageElement | null;
}
function previewLoadingEl() {
  return document.getElementById(PREVIEW_LOADING_ID) as HTMLElement | null;
}
function urlInputEl() {
  return document.getElementById(URL_INPUT_ID) as HTMLInputElement | null;
}
function copyBtnEl() {
  return document.getElementById(COPY_BTN_ID) as HTMLButtonElement | null;
}
function nativeBtnEl() {
  return document.getElementById(NATIVE_BTN_ID) as HTMLButtonElement | null;
}

function revokePreview() {
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
}

function resetDialogState() {
  revokePreview();
  const img = previewImgEl();
  if (img) {
    img.hidden = true;
    img.removeAttribute("src");
  }
  const loading = previewLoadingEl();
  if (loading) loading.hidden = false;
  const urlInput = urlInputEl();
  if (urlInput) urlInput.value = "";
  const copyBtn = copyBtnEl();
  if (copyBtn) {
    copyBtn.disabled = true;
    copyBtn.textContent = "Copy link";
  }
}

export function closeShareDialog() {
  const dialog = dialogEl();
  if (dialog?.open) dialog.close();
  revokePreview();
}

export function openSharePopup() {
  const dialog = dialogEl();
  if (!dialog || typeof dialog.showModal !== "function") return;

  const token = ++openToken;
  resetDialogState();
  if (!dialog.open) dialog.showModal();

  void runShareCapture(token);
}

async function runShareCapture(token: number) {
  const url = await buildExpressionShareUrl();
  if (token !== openToken) return; // dialog closed/reopened since

  const urlInput = urlInputEl();
  if (urlInput) urlInput.value = url;
  const copyBtn = copyBtnEl();
  if (copyBtn) copyBtn.disabled = false;

  try {
    const { renderShareOgImage, uploadShareOgImage } = await import("./shareCapture.js");
    const blob = await renderShareOgImage(url);
    if (token !== openToken) return;

    const loading = previewLoadingEl();
    if (blob) {
      revokePreview();
      previewObjectUrl = URL.createObjectURL(blob);
      const img = previewImgEl();
      if (img) {
        img.src = previewObjectUrl;
        img.hidden = false;
      }
      if (loading) loading.hidden = true;
      // Fire-and-forget: api/og.ts falls back to a server-side render if
      // this never lands, so it doesn't need to block anything here.
      void uploadShareOgImage(url, blob).catch(() => {});
    } else if (loading) {
      loading.hidden = true;
    }
  } catch {
    if (token !== openToken) return;
    const loading = previewLoadingEl();
    if (loading) loading.hidden = true;
  }
}

async function onCopyClick(btn: HTMLButtonElement) {
  const url = urlInputEl()?.value;
  if (!url) return;
  try {
    await navigator.clipboard.writeText(url);
    const prev = btn.textContent;
    btn.textContent = "Copied!";
    window.setTimeout(() => {
      btn.textContent = prev;
    }, 1600);
  } catch {
    setAutosaveError("Could not copy link");
  }
}

async function onNativeShareClick() {
  const url = urlInputEl()?.value;
  if (!url || typeof navigator.share !== "function") return;
  try {
    await navigator.share({ title: "laplaci", url });
    closeShareDialog();
  } catch (e) {
    // The user dismissing the native share sheet is a deliberate cancel,
    // not a failure — leave the dialog open, don't surface an error.
    if (e instanceof DOMException && e.name === "AbortError") return;
    setAutosaveError("Could not share link");
  }
}

export function initSharePopup() {
  if (wired) return;
  wired = true;

  const dialog = dialogEl();
  const closeBtn = document.getElementById(CLOSE_ID) as HTMLButtonElement | null;
  const copyBtn = copyBtnEl();
  const nativeBtn = nativeBtnEl();

  closeBtn?.addEventListener("click", () => closeShareDialog());
  dialog?.addEventListener("click", (ev) => {
    if (ev.target === dialog) closeShareDialog();
  });
  dialog?.addEventListener("close", () => revokePreview());
  copyBtn?.addEventListener("click", () => {
    if (copyBtn) void onCopyClick(copyBtn);
  });

  if (typeof navigator.share === "function") {
    nativeBtn?.addEventListener("click", () => void onNativeShareClick());
  } else if (nativeBtn) {
    nativeBtn.hidden = true;
  }
}
