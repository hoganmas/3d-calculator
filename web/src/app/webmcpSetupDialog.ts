import { buildMcpConfigJson, dismissSetupPrompt } from "./webmcpConfig.js";
import { enableWebMCP } from "./webmcp.js";

const DIALOG_ID = "webmcpSetupDialog";
const SNIPPET_ID = "webmcpConfigSnippet";
const COPY_ID = "copyWebmcpConfig";
const CLOSE_ID = "closeWebmcpSetup";
const OPEN_BTN_ID = "openWebmcpSetup";

let wired = false;

function refreshSnippet() {
  const ta = document.getElementById(SNIPPET_ID) as HTMLTextAreaElement | null;
  if (ta) ta.value = buildMcpConfigJson();
}

export function openWebmcpSetupDialog() {
  const dialog = document.getElementById(DIALOG_ID) as HTMLDialogElement | null;
  if (!dialog || typeof dialog.showModal !== "function") return;
  refreshSnippet();
  void enableWebMCP();
  if (!dialog.open) dialog.showModal();
}

export function closeWebmcpSetupDialog(dismiss = false) {
  const dialog = document.getElementById(DIALOG_ID) as HTMLDialogElement | null;
  if (dismiss) dismissSetupPrompt();
  if (dialog?.open) dialog.close();
}

export function initWebmcpSetupDialog() {
  if (wired) return;
  wired = true;

  const dialog = document.getElementById(DIALOG_ID) as HTMLDialogElement | null;
  const copyBtn = document.getElementById(COPY_ID) as HTMLButtonElement | null;
  const closeBtn = document.getElementById(CLOSE_ID) as HTMLButtonElement | null;
  const openBtn = document.getElementById(OPEN_BTN_ID) as HTMLButtonElement | null;

  openBtn?.addEventListener("click", () => openWebmcpSetupDialog());
  closeBtn?.addEventListener("click", () => closeWebmcpSetupDialog(true));
  copyBtn?.addEventListener("click", () => void copyMcpConfigSnippet(copyBtn));
  dialog?.addEventListener("click", (ev) => {
    if (ev.target === dialog) closeWebmcpSetupDialog(true);
  });
  dialog?.addEventListener("close", () => refreshSnippet());
}

async function copyMcpConfigSnippet(btn: HTMLButtonElement) {
  const text = buildMcpConfigJson();
  try {
    await navigator.clipboard.writeText(text);
    const prev = btn.textContent;
    btn.textContent = "Copied";
    window.setTimeout(() => {
      btn.textContent = prev;
    }, 1600);
  } catch {
    const ta = document.getElementById(SNIPPET_ID) as HTMLTextAreaElement | null;
    ta?.focus();
    ta?.select();
  }
}
