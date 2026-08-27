/**
 * Call a Laplacian WebMCP tool via the local relay WITHOUT opening new browser tabs.
 *
 * Usage:
 *   node scripts/webmcp-relay-call.mjs laplacian_setup_lava_lamp
 *   node scripts/webmcp-relay-call.mjs laplacian_get_state '{}'
 *   node scripts/webmcp-relay-call.mjs laplacian_set_render_settings '{"boxSize":6}'
 *
 * Prerequisite: ONE tab already open at http://127.0.0.1:5173/3d-calculator/?webmcp=1
 * Your MCP client’s laplacian-webmcp server must be running (or this script starts its own relay).
 */
import { spawn } from "node:child_process";

const ORIGIN = process.env.WEBMCP_ORIGIN ?? "http://127.0.0.1:5173,http://localhost:5173";
const WAIT_MS = Number(process.env.WEBMCP_WAIT_MS ?? 15000);
const toolName = process.argv[2];
const argsJson = process.argv[3] ?? "{}";

if (!toolName) {
  console.error("Usage: node scripts/webmcp-relay-call.mjs <tool_name> [json_args]");
  process.exit(1);
}

let toolArgs;
try {
  toolArgs = JSON.parse(argsJson);
} catch {
  console.error("Invalid JSON args:", argsJson);
  process.exit(1);
}

const proc = spawn(
  "npx",
  ["-y", "@mcp-b/webmcp-local-relay@latest", "--widget-origin", ORIGIN],
  { stdio: ["pipe", "pipe", "pipe"] },
);
proc.stderr.on("data", (d) => process.stderr.write(d));

const pending = new Map();
let buf = "";
proc.stdout.on("data", (d) => {
  buf += d;
  let idx;
  while ((idx = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, idx);
    buf = buf.slice(idx + 1);
    if (!line.trim()) continue;
    try {
      const msg = JSON.parse(line);
      if (msg.id != null && pending.has(msg.id)) {
        pending.get(msg.id).resolve(msg);
        pending.delete(msg.id);
      }
    } catch {
      /* ignore */
    }
  }
});

let nextId = 1;
function request(method, params) {
  const id = nextId++;
  return new Promise((resolve) => {
    pending.set(id, { resolve });
    proc.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
  });
}

function parseResult(msg) {
  if (msg.error) throw new Error(msg.error.message || JSON.stringify(msg.error));
  const text = msg.result?.content?.[0]?.text;
  if (!text) return msg.result;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

await request("initialize", {
  protocolVersion: "2024-11-05",
  capabilities: {},
  clientInfo: { name: "webmcp-relay-call", version: "0" },
});
proc.stdin.write(JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }) + "\n");

const deadline = Date.now() + WAIT_MS;
let sources = { count: 0 };
while (Date.now() < deadline) {
  sources = parseResult(await request("tools/call", { name: "webmcp_list_sources", arguments: {} }));
  if (sources.count > 0) break;
  await new Promise((r) => setTimeout(r, 400));
}

if (sources.count === 0) {
  console.error(
    `No browser tab connected after ${WAIT_MS}ms.\n` +
      `Keep ONE tab open at ${ORIGIN}/3d-calculator/?webmcp=1 — do not use webmcp_open_page.`,
  );
  proc.stdin.end();
  proc.kill();
  process.exit(1);
}

console.error(`Connected: ${sources.sources[0].title} (${sources.sources[0].toolCount} tools)`);

const result = parseResult(
  await request("tools/call", { name: toolName, arguments: toolArgs }),
);
console.log(JSON.stringify(result, null, 2));

proc.stdin.end();
proc.kill();
