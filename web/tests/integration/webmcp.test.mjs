/**
 * WebMCP smoke test (Playwright). Requires dev server: npm run dev
 * Run: npm run test:integration
 */
import { chromium } from "playwright";
import { BASE } from "./base-url.mjs";

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

const logs = [];
page.on("console", (msg) => logs.push(`[${msg.type()}] ${msg.text()}`));

await page.goto(BASE, { waitUntil: "networkidle", timeout: 60000 });
await page.waitForTimeout(4000);

const result = await page.evaluate(async () => {
  const unwrap = (result) => {
    let r = result;
    if (typeof r === "string") {
      try {
        r = JSON.parse(r);
      } catch {
        return null;
      }
    }
    if (r == null) return null;
    if (r.data != null && typeof r.data === "object") return r.data;
    if (r.content != null && typeof r.content === "object") return r.content;
    return r;
  };
  const ctx = document.modelContext;
  if (!ctx) return { error: "no modelContext" };
  const tools = await ctx.getTools();
  const names = tools.map((t) => t.name).sort();
  const getState = tools.find((t) => t.name === "laplacian_get_state");
  let stateResult = null;
  if (getState && ctx.executeTool) {
    stateResult = unwrap(await ctx.executeTool(getState, "{}"));
  }
  const reset = tools.find((t) => t.name === "laplacian_reset_camera");
  let resetResult = null;
  if (reset && ctx.executeTool) {
    resetResult = unwrap(await ctx.executeTool(reset, "{}"));
  }
  return {
    toolCount: names.length,
    laplacianTools: names.filter((n) => n.startsWith("laplacian_")),
    hasRelayScript: !!document.getElementById("laplacian-webmcp-relay"),
    stateResult,
    resetResult,
  };
});

console.log(
  JSON.stringify(
    {
      webmcpLogs: logs.filter((l) => l.includes("WebMCP") || l.includes("webmcp")),
      result,
    },
    null,
    2,
  ),
);

await browser.close();
