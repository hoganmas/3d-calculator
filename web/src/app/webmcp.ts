/**
 * WebMCP bootstrap: polyfill, local-relay embed, Laplacian tool registration.
 */
import { initializeWebMCPPolyfill } from "@mcp-b/webmcp-polyfill";
import { registerLaplacianTools } from "./webmcpTools.js";
import { maybeShowWebmcpSetupPrompt } from "./webmcpSetupDialog.js";

const STORAGE_KEY = "laplacian-webmcp";
const RELAY_SCRIPT_ID = "laplacian-webmcp-relay";
const RELAY_CDN =
  "https://cdn.jsdelivr.net/npm/@mcp-b/webmcp-local-relay@latest/dist/browser/embed.js";

/** @type {AbortController | null} */
let registration: AbortController | null = null;

export function isWebMCPEnabled(): boolean {
  const q = new URLSearchParams(location.search);
  if (q.get("webmcp") === "0") return false;
  if (q.get("webmcp") === "1") return true;
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "1") return true;
    if (stored === "0") return false;
  } catch {
    /* ignore */
  }
  if (import.meta.env.DEV) return true;
  // Production builds default on (any host); opt out via ?webmcp=0 or localStorage.
  if (import.meta.env.PROD) {
    try {
      if (location.protocol !== "file:") return true;
    } catch {
      /* ignore */
    }
  }
  return false;
}

/** Persist opt-in/out (survives reload; overrides DEV default when set). */
export function setWebMCPEnabled(on: boolean) {
  try {
    localStorage.setItem(STORAGE_KEY, on ? "1" : "0");
  } catch {
    /* ignore */
  }
}

function injectRelayEmbed() {
  if (document.getElementById(RELAY_SCRIPT_ID)) return;
  const script = document.createElement("script");
  script.id = RELAY_SCRIPT_ID;
  script.src = RELAY_CDN;
  script.dataset.requestTimeout = "120000";
  // Pin IPv4 + default port — avoids [::1] scan storms when relays restart.
  script.dataset.relayHost = "127.0.0.1";
  script.dataset.relayPort = "9333";
  if (import.meta.env.DEV) script.dataset.debug = "";
  document.head.appendChild(script);
}

/** Initialize polyfill, relay embed, and tool registrations. Safe to call once. */
export async function initWebMCP() {
  if (!isWebMCPEnabled()) return;

  initializeWebMCPPolyfill();
  injectRelayEmbed();

  if (!document.modelContext?.registerTool) {
    console.warn("[WebMCP] modelContext unavailable after polyfill init");
    return;
  }

  registration?.abort();
  registration = new AbortController();
  await registerLaplacianTools(registration.signal);
  maybeShowWebmcpSetupPrompt();
}
