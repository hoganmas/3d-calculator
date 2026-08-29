/**
 * Playwright smoke: intentional param creation via Tab and prune on undo.
 * Requires dev server: npm run dev
 * Run: npm run test:integration
 */
import { chromium } from "playwright";
import { BASE } from "./base-url.mjs";

function unwrap(result) {
  let r = result;
  if (typeof r === "string") {
    try {
      r = JSON.parse(r);
    } catch {
      return null;
    }
  }
  if (r?.data != null && typeof r.data === "object") return r.data;
  if (r?.content != null && typeof r.content === "object") return r.content;
  return r;
}

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

try {
  await page.goto(BASE, { waitUntil: "networkidle", timeout: 60000 });
  await page.waitForTimeout(3000);

  const typed = await page.evaluate(async () => {
    const rows = [...document.querySelectorAll(".expr-row")];
    const fieldRow =
      rows.find((r) => !r.classList.contains("is-param-def") && r.querySelector("math-field")) ??
      rows.find((r) => r.querySelector("math-field"));
    if (!fieldRow) return { error: "no field row" };
    const mf = fieldRow.querySelector("math-field");
    if (!mf) return { error: "no math-field" };
    const rowId = fieldRow.dataset.id ?? null;
    try {
      if (typeof mf.setValue === "function") {
        mf.setValue("b x", { format: "latex" });
      } else {
        mf.value = "b x";
      }
      mf.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true }));
    } catch (e) {
      return { error: `setValue failed: ${e}` };
    }
    mf.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true }));
    mf.blur?.();
    document.body.click();
    return { ok: true, rowId };
  });

  if (typed.error) {
    console.error(JSON.stringify(typed, null, 2));
    process.exit(1);
  }

  await page.waitForTimeout(800);

  const afterType = await page.evaluate(async (rowId) => {
    const unwrap = (result) => {
      let r = result;
      if (typeof r === "string") {
        try {
          r = JSON.parse(r);
        } catch {
          return null;
        }
      }
      if (r?.data != null && typeof r.data === "object") return r.data;
      if (r?.content != null && typeof r.content === "object") return r.content;
      return r;
    };
    const ctx = document.modelContext;
    if (!ctx?.executeTool) return { error: "no modelContext.executeTool" };
    const tools = await ctx.getTools();
    const listTool = tools.find((t) => t.name === "laplacian_list_expressions");
    if (!listTool) return { error: "missing list tool" };
    const listed = unwrap(await ctx.executeTool(listTool, "{}"));
    const rows = listed?.expressions ?? [];
    const b = rows.find((e) => String(e.latex || "").startsWith("b="));
    const field = rows.find((e) => e.id === rowId) ?? rows.find((e) => String(e.latex || "").includes("b"));
    return {
      rowCount: rows.length,
      fieldLatex: field?.latex ?? null,
      bLatex: b?.latex ?? null,
      bAuto: b?.autoParam ?? null,
    };
  }, typed.rowId);

  if (afterType.error) {
    console.error(JSON.stringify(afterType, null, 2));
    process.exit(1);
  }

  if (!afterType.bLatex) {
    console.error("auto-param b= row missing after typing b x:", JSON.stringify(afterType, null, 2));
    process.exit(1);
  }

  const afterUndo = await page.evaluate(async (rowId) => {
    const unwrap = (result) => {
      let r = result;
      if (typeof r === "string") {
        try {
          r = JSON.parse(r);
        } catch {
          return null;
        }
      }
      if (r?.data != null && typeof r.data === "object") return r.data;
      if (r?.content != null && typeof r.content === "object") return r.content;
      return r;
    };
    const ctx = document.modelContext;
    const tools = await ctx.getTools();
    const setTool = tools.find((t) => t.name === "laplacian_set_expression");
    const listTool = tools.find((t) => t.name === "laplacian_list_expressions");
    if (!setTool || !listTool) return { error: "missing tools" };

    let targetId = rowId;
    if (!targetId) {
      const listed0 = unwrap(await ctx.executeTool(listTool, "{}"));
      const rows0 = listed0?.expressions ?? [];
      const hit = rows0.find((e) => String(e.latex || "").includes("b"));
      targetId = hit?.id;
    }
    if (!targetId) return { error: "no target field row" };

    await ctx.executeTool(setTool, JSON.stringify({ id: targetId, latex: "x^2" }));
    await new Promise((r) => setTimeout(r, 400));
    const listed = unwrap(await ctx.executeTool(listTool, "{}"));
    const rows = listed?.expressions ?? [];
    return {
      fieldLatex: rows.find((e) => e.id === targetId)?.latex ?? null,
      bGone: !rows.some((e) => String(e.latex || "").startsWith("b=")),
    };
  }, typed.rowId);

  if (afterUndo.error) {
    console.error(JSON.stringify(afterUndo, null, 2));
    process.exit(1);
  }

  if (!afterUndo.bGone) {
    console.error("orphan b= row remained after undo:", JSON.stringify(afterUndo, null, 2));
    process.exit(1);
  }

  console.log(
    JSON.stringify(
      {
        ok: true,
        afterType: {
          fieldLatex: afterType.fieldLatex,
          bLatex: afterType.bLatex,
          bAuto: afterType.bAuto,
        },
        afterUndo: {
          fieldLatex: afterUndo.fieldLatex,
          bGone: afterUndo.bGone,
        },
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
}
