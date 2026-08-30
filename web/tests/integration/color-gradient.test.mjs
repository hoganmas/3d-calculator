/**
 * Playwright smoke: MCP laplaci_set_expression color/color2 updates layer gradient.
 * Requires dev server: npm run dev
 * Run: npm run test:integration
 */
import { chromium } from "playwright";
import { BASE } from "./base-url.mjs";

function normHex(h) {
  return String(h || "").toLowerCase();
}

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

try {
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
    if (!ctx?.executeTool) return { error: "no modelContext.executeTool" };
    const tools = await ctx.getTools();
    const byName = (n) => tools.find((t) => t.name === n);
    const listTool = byName("laplaci_list_expressions");
    const setTool = byName("laplaci_set_expression");
    if (!listTool || !setTool) return { error: "missing MCP tools" };

    const listed = unwrap(await ctx.executeTool(listTool, "{}"));
    const rows = listed?.expressions ?? [];
    let target = rows.find((e) => {
      const latex = String(e.latex || "").trim();
      return latex && !/^[^=]+=/.test(latex);
    });
    if (!target?.id) target = rows.find((e) => String(e.latex || "").trim());
    if (!target?.id) {
      const addTool = byName("laplaci_add_expression");
      if (!addTool) return { error: "no expression row and no add tool", rows: rows.length };
      const added = unwrap(await ctx.executeTool(addTool, JSON.stringify({ latex: "x^2+y^2" })));
      target = added?.expression;
    }
    if (!target?.id) return { error: "no expression row", rows: rows.length };

    const before = { color: target.color, color2: target.color2 };
    const patch = {
      id: target.id,
      color: "#010203",
      color2: "#040506",
    };
    const setRes = unwrap(await ctx.executeTool(setTool, JSON.stringify(patch)));
    const row = setRes?.expression;
    const listed2 = unwrap(await ctx.executeTool(listTool, "{}"));
    const rows2 = listed2?.expressions ?? [];
    const after = rows2.find((e) => e.id === target.id);

    return {
      id: target.id,
      before,
      setRes: { compileOk: setRes?.compileOk, color: row?.color, color2: row?.color2 },
      after: after ? { color: after.color, color2: after.color2 } : null,
    };
  });

  if (result.error) {
    console.error(JSON.stringify(result, null, 2));
    process.exit(1);
  }

  const c1 = normHex(result.setRes?.color ?? result.after?.color);
  const c2 = normHex(result.setRes?.color2 ?? result.after?.color2);
  if (c1 !== "#010203" || c2 !== "#040506") {
    console.error("color patch did not stick:", JSON.stringify(result, null, 2));
    process.exit(1);
  }

  console.log(
    JSON.stringify(
      {
        ok: true,
        id: result.id,
        before: result.before,
        after: { color: c1, color2: c2 },
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
}
