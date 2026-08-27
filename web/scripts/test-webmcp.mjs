import { chromium } from "playwright";

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

const logs = [];
page.on("console", (msg) => logs.push(`[${msg.type()}] ${msg.text()}`));

await page.goto("http://127.0.0.1:5173/", { waitUntil: "networkidle", timeout: 60000 });
await page.waitForTimeout(4000);

const result = await page.evaluate(async () => {
  const ctx = document.modelContext;
  if (!ctx) return { error: "no modelContext" };
  const tools = await ctx.getTools();
  const names = tools.map((t) => t.name).sort();
  const getState = tools.find((t) => t.name === "laplacian_get_state");
  let stateResult = null;
  if (getState && ctx.executeTool) {
    stateResult = await ctx.executeTool(getState, "{}");
  }
  const reset = tools.find((t) => t.name === "laplacian_reset_camera");
  let resetResult = null;
  if (reset && ctx.executeTool) {
    resetResult = await ctx.executeTool(reset, "{}");
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
