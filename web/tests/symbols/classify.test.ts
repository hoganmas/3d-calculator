import { classifyExpr, compileExpr, expandDefinitions } from "../../src/math/fit.ts";
import { SymbolRegistry } from "../../src/model/symbols.ts";
import { assert, assertNear } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function makeRegistry(entries: { kind: "alias" | "funcdef"; name: string; rhs: string; args?: string[] }[]) {
  const registry = new SymbolRegistry();
  for (const e of entries) {
    if (e.kind === "alias") {
      registry.tryAdd({
        kind: "alias",
        name: e.name,
        rhsLatex: e.rhs,
        latex: `${e.name}=${e.rhs}`,
        exprId: "t1",
      });
    } else {
      const args = e.args ?? ["x"];
      registry.tryAdd({
        kind: "funcdef",
        name: e.name,
        rhsLatex: e.rhs,
        latex: `${e.name}(${args.join(",")})=${e.rhs}`,
        exprId: "t2",
        funcArgs: args,
      });
    }
  }
  return registry;
}

export async function run() {
  return runSuite("symbols / classify", [
    {
      name: "a=1 classifies as parameter",
      fn: () => {
        const c = classifyExpr("a=1");
        assert(c.kind === "parameter", `expected parameter, got ${c.kind}`);
        assert(c.paramName === "a", "param name");
      },
    },
    {
      name: "T=x^2+y^2 classifies as alias",
      fn: () => {
        const c = classifyExpr(String.raw`T=x^2+y^2`);
        assert(c.kind === "alias", `expected alias, got ${c.kind}`);
        assert(c.aliasName === "T", "alias name");
      },
    },
    {
      name: "f(x)=sin(x)+y classifies as funcdef",
      fn: () => {
        const c = classifyExpr(String.raw`f(x)=\sin(x)+y`);
        assert(c.kind === "funcdef", `expected funcdef, got ${c.kind}`);
        assert(c.funcName === "f", "func name");
        assert(c.funcArgs?.length === 1 && c.funcArgs[0] === "x", "func args");
      },
    },
    {
      name: "param and funcdef may share a name",
      fn: () => {
        const registry = new SymbolRegistry();
        assert(registry.tryAdd({
          kind: "parameter",
          name: "a",
          rhsLatex: "1",
          latex: "a=1",
          exprId: "e1",
        }) === null);
        assert(registry.tryAdd({
          kind: "funcdef",
          name: "a",
          rhsLatex: "x",
          latex: "a(x)=x",
          exprId: "e2",
          funcArgs: ["x"],
        }) === null);
      },
    },
    {
      name: "duplicate parameter names error within namespace",
      fn: () => {
        const registry = new SymbolRegistry();
        registry.tryAdd({
          kind: "parameter",
          name: "a",
          rhsLatex: "1",
          latex: "a=1",
          exprId: "e1",
        });
        const err = registry.tryAdd({
          kind: "parameter",
          name: "a",
          rhsLatex: "2",
          latex: "a=2",
          exprId: "e2",
        });
        assert(!!err, "expected duplicate error");
      },
    },
    {
      name: "alias expands and graphs bare T",
      fn: () => {
        const registry = makeRegistry([{ kind: "alias", name: "T", rhs: String.raw`x^2+y^2` }]);
        const expanded = expandDefinitions("T", registry);
        assert(expanded.includes("x"), "expanded uses x");
        const fn = compileExpr("T", registry).bind({});
        assertNear(fn(1, 2, 0), 5, 1e-9, "T at (1,2,0)");
      },
    },
    {
      name: "funcdef expands at call site",
      fn: () => {
        const registry = makeRegistry([
          { kind: "funcdef", name: "f", rhs: String.raw`\sin(x)+y`, args: ["x"] },
        ]);
        const fn = compileExpr(String.raw`f(x)`, registry).bind({});
        assertNear(fn(0, 1, 0), 1, 1e-9, "f(0)+y");
      },
    },
  ]);
}
