import { classifyExpr, compileExpr, expandDefinitions } from "../../src/math/fit.ts";
import { compileVectorExpr, isVectorFieldLatex } from "../../src/math/fitVector.ts";
import { resolveExprRole } from "../../src/model/expressions.ts";
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
    {
      name: "f(x)=sin(x) bare f(x) compiles",
      fn: () => {
        const registry = makeRegistry([
          { kind: "funcdef", name: "f", rhs: String.raw`\sin(x)`, args: ["x"] },
        ]);
        const expanded = expandDefinitions(String.raw`f(x)`, registry);
        assert(expanded.includes("sin") || expanded.includes("\\sin"), "expands to sin(x)");
        const fn = compileExpr(String.raw`f(x)`, registry).bind({});
        assertNear(fn(0, 0, 0), 0, 1e-9, "sin(0)");
        assertNear(fn(Math.PI / 2, 0, 0), 1, 1e-9, "sin(pi/2)");
      },
    },
    {
      name: "f(x)=sin(x) f(1) compiles to constant sin(1)",
      fn: () => {
        const registry = makeRegistry([
          { kind: "funcdef", name: "f", rhs: String.raw`\sin(x)`, args: ["x"] },
        ]);
        const expanded = expandDefinitions("f(1)", registry);
        assert(!expanded.includes("f("), `expected no call, got ${expanded}`);
        const compiled = compileExpr("f(1)", registry);
        assert(!compiled.usesSpace, "constant field");
        const fn = compiled.bind({});
        assertNear(fn(0, 0, 0), Math.sin(1), 1e-9, "sin(1)");
      },
    },
    {
      name: "g(x,y)=x+y g(x,1) substitutes y",
      fn: () => {
        const registry = makeRegistry([
          { kind: "funcdef", name: "g", rhs: "x+y", args: ["x", "y"] },
        ]);
        const expanded = expandDefinitions("g(x,1)", registry);
        assert(expanded.includes("x") && expanded.includes("1"), `expanded: ${expanded}`);
        const fn = compileExpr("g(x,1)", registry).bind({});
        assertNear(fn(2, 0, 0), 3, 1e-9, "x+1 at x=2");
      },
    },
    {
      name: "formal param h(a) h(x) and h(1)",
      fn: () => {
        const registry = makeRegistry([
          { kind: "funcdef", name: "h", rhs: String.raw`\sin(a)`, args: ["a"] },
        ]);
        const fnX = compileExpr("h(x)", registry).bind({});
        assertNear(fnX(1, 0, 0), Math.sin(1), 1e-9, "h(x)");
        const fn1 = compileExpr("h(1)", registry).bind({});
        assertNear(fn1(0, 0, 0), Math.sin(1), 1e-9, "h(1)");
      },
    },
    {
      name: "f(a) with param reference stays symbolic",
      fn: () => {
        const registry = makeRegistry([
          { kind: "funcdef", name: "f", rhs: "a+1", args: ["a"] },
        ]);
        const compiled = compileExpr("f(a)", registry);
        assert(compiled.freeParams.includes("a"), "a remains free param");
        const fn = compiled.bind({ a: 3 });
        assertNear(fn(0, 0, 0), 4, 1e-9, "a+1 with a=3");
      },
    },
    {
      name: "arity mismatch produces warning",
      fn: () => {
        const registry = makeRegistry([
          { kind: "funcdef", name: "g", rhs: "x+y", args: ["x", "y"] },
        ]);
        const warns: string[] = [];
        expandDefinitions("g(x)", registry, warns);
        assert(warns.length > 0, "expected arity warning");
        assert(warns[0]!.includes("2"), "mentions expected arity");
      },
    },
    {
      name: "nested f(g(x)) expands",
      fn: () => {
        const registry = makeRegistry([
          { kind: "funcdef", name: "g", rhs: "x+1", args: ["x"] },
          { kind: "funcdef", name: "f", rhs: String.raw`\sin(x)`, args: ["x"] },
        ]);
        const fn = compileExpr(String.raw`f(g(x))`, registry).bind({});
        assertNear(fn(0, 0, 0), Math.sin(1), 1e-9, "sin(x+1) at 0");
      },
    },
    {
      name: "alias declaration row produces a layer",
      fn: () => {
        const classified = classifyExpr(String.raw`T=x^2+y^2`);
        assert(classified.kind === "alias", "alias kind");
        const role = resolveExprRole("auto", classified.kind, classified.compileLatex);
        assert(role === "cloud", "alias body role");
        const compiled = compileExpr(classified.compileLatex);
        assert(compiled.usesSpace, "uses space");
        assert(compiled.shade !== "none", "graphable shade");
      },
    },
    {
      name: "funcdef declaration row produces a layer",
      fn: () => {
        const classified = classifyExpr(String.raw`f(x)=\sin(x)+y`);
        assert(classified.kind === "funcdef", "funcdef kind");
        const role = resolveExprRole("auto", classified.kind, classified.compileLatex);
        assert(role === "cloud", "funcdef body role");
        const compiled = compileExpr(classified.compileLatex);
        assert(compiled.usesSpace, "uses space");
        assertNear(compiled.bind({})(0, 1, 0), 1, 1e-9, "funcdef body at (0,1,0)");
      },
    },
    {
      name: "vector alias V=(x,y,0) classifies and compiles as vector",
      fn: () => {
        const c = classifyExpr(String.raw`V=(x,y,0)`);
        assert(c.kind === "alias", `expected alias, got ${c.kind}`);
        assert(c.aliasName === "V", "alias name");
        assert(isVectorFieldLatex(c.compileLatex), "RHS is vector syntax");
        const compiled = compileVectorExpr(c.compileLatex);
        assert(compiled.kind === "tuple", "tuple vector field");
        const v = compiled.bind({})(1, 2, 0);
        assertNear(v[0]!, 1, 1e-9, "Vx");
        assertNear(v[1]!, 2, 1e-9, "Vy");
        assertNear(v[2]!, 0, 1e-9, "Vz");
      },
    },
    {
      name: "vector funcdef expands at call site for flow",
      fn: () => {
        const registry = makeRegistry([
          { kind: "funcdef", name: "f", rhs: String.raw`(-y,x,0)`, args: ["x"] },
        ]);
        const compiled = compileVectorExpr(String.raw`f(x)`, registry);
        assert(compiled.kind === "tuple", "expanded tuple");
        const v = compiled.bind({})(0, 1, 0);
        assertNear(v[0]!, -1, 1e-9, "rotated Vx");
        assertNear(v[1]!, 0, 1e-9, "rotated Vy");
        assertNear(v[2]!, 0, 1e-9, "rotated Vz");
      },
    },
    {
      name: "alias and reference both graph",
      fn: () => {
        const classified = classifyExpr(String.raw`T=x^2+y^2`);
        const registry = makeRegistry([
          { kind: "alias", name: "T", rhs: classified.compileLatex },
        ]);
        const decl = compileExpr(classified.compileLatex);
        const ref = compileExpr("T", registry);
        assert(decl.usesSpace && ref.usesSpace, "both graph");
        assertNear(decl.bind({})(1, 2, 0), ref.bind({})(1, 2, 0), 1e-9, "same field");
      },
    },
    {
      name: "vector alias declaration graphs as flow",
      fn: () => {
        const classified = classifyExpr(String.raw`V=(x,y,0)`);
        const role = resolveExprRole("auto", classified.kind, classified.compileLatex);
        assert(role === "flow", "vector alias auto role");
        const compiled = compileVectorExpr(classified.compileLatex);
        assert(compiled.usesSpace, "uses space");
      },
    },
    {
      name: "hidden funcdef stays in scope for curl",
      fn: () => {
        const registry = makeRegistry([
          { kind: "funcdef", name: "f", rhs: String.raw`(-y,x,0)`, args: ["x", "y", "z"] },
        ]);
        const classified = classifyExpr(String.raw`\del \times f`);
        const role = resolveExprRole("auto", classified.kind, classified.compileLatex, registry);
        assert(role === "flow", "curl row is flow");
        const compiled = compileVectorExpr(classified.compileLatex, registry);
        assert(compiled.kind === "curl", "curl compiles");
        const [, , fz] = compiled.bind({})(0.2, 0.3, 0.1);
        assertNear(fz, 2, 0.05, "curl fz");
      },
    },
    {
      name: "del cross funcdef infers flow role",
      fn: () => {
        const registry = makeRegistry([
          { kind: "funcdef", name: "f", rhs: String.raw`(-y,x,0)`, args: ["x", "y", "z"] },
        ]);
        const classified = classifyExpr(String.raw`\del \times f`);
        const role = resolveExprRole("auto", classified.kind, classified.compileLatex, registry);
        assert(role === "flow", "del cross f auto role");
        const compiled = compileVectorExpr(classified.compileLatex, registry);
        assert(compiled.kind === "curl", "curl field");
        const [, , fz] = compiled.bind({})(0.2, 0.3, 0.1);
        assertNear(fz, 2, 0.05, "curl fz");
      },
    },
    {
      name: "divergence of vector alias",
      fn: () => {
        const registry = makeRegistry([
          { kind: "alias", name: "V", rhs: String.raw`(-y,x,0)` },
        ]);
        const compiled = compileExpr(String.raw`\div(V)`, registry);
        assert(compiled.operator === "divergence", "divergence operator");
        assertNear(compiled.bind({})(0.2, 0.3, 0.1), 0, 0.05, "div of rotation field");
      },
    },
    {
      name: "nabla dot vector alias",
      fn: () => {
        const registry = makeRegistry([
          { kind: "alias", name: "V", rhs: String.raw`(-y,x,0)` },
        ]);
        const compiled = compileExpr(String.raw`\nabla\cdot V`, registry);
        assert(compiled.operator === "divergence", "divergence operator");
        assertNear(compiled.bind({})(0.2, 0.3, 0.1), 0, 0.05, "nabla dot alias");
      },
    },
    {
      name: "curl of vector alias",
      fn: () => {
        const registry = makeRegistry([
          { kind: "alias", name: "V", rhs: String.raw`(-y,x,0)` },
        ]);
        const compiled = compileVectorExpr(String.raw`\curl(V)`, registry);
        assert(compiled.kind === "curl", "curl operator");
        const [fx, fy, fz] = compiled.bind({})(0.2, 0.3, 0.1);
        assertNear(fx, 0, 0.05, "curl fx");
        assertNear(fy, 0, 0.05, "curl fy");
        assertNear(fz, 2, 0.05, "curl fz");
      },
    },
    {
      name: "curl of vector funcdef with bare name",
      fn: () => {
        const registry = makeRegistry([
          { kind: "funcdef", name: "F", rhs: String.raw`(-y,x,0)`, args: ["x", "y", "z"] },
        ]);
        const compiled = compileVectorExpr(String.raw`\curl(F)`, registry);
        assert(compiled.kind === "curl", "curl operator");
        const [fx, fy, fz] = compiled.bind({})(0.2, 0.3, 0.1);
        assertNear(fz, 2, 0.05, "curl F implicit call");
        void fx;
        void fy;
      },
    },
    {
      name: "grad of scalar alias",
      fn: () => {
        const registry = makeRegistry([
          { kind: "alias", name: "f", rhs: String.raw`x^2+y^2` },
        ]);
        const compiled = compileVectorExpr(String.raw`\grad f`, registry);
        assert(compiled.kind === "gradient", "gradient operator");
        const [gx, gy, gz] = compiled.bind({})(1, 2, 0);
        assertNear(gx, 2, 0.05, "grad fx");
        assertNear(gy, 4, 0.05, "grad fy");
        assertNear(gz, 0, 0.05, "grad fz");
      },
    },
    {
      name: "nabla dot mathrm vector alias",
      fn: () => {
        const registry = makeRegistry([
          { kind: "alias", name: "V", rhs: String.raw`(-y,x,0)` },
        ]);
        const compiled = compileExpr(String.raw`\nabla\cdot\mathrm{V}`, registry);
        assert(compiled.operator === "divergence", "divergence operator");
        assertNear(compiled.bind({})(0.2, 0.3, 0.1), 0, 0.05, "div mathrm alias");
      },
    },
    {
      name: "curl of mathbf vector alias",
      fn: () => {
        const registry = makeRegistry([
          { kind: "alias", name: "V", rhs: String.raw`(-y,x,0)` },
        ]);
        const compiled = compileVectorExpr(String.raw`\nabla\times\mathbf{V}`, registry);
        assert(compiled.kind === "curl", "curl operator");
        const [, , fz] = compiled.bind({})(0.2, 0.3, 0.1);
        assertNear(fz, 2, 0.05, "curl mathbf alias");
      },
    },
    {
      name: "deg normalizes to radians for CE",
      fn: () => {
        const fn = compileExpr(String.raw`\sin(30\deg)`).bind({});
        assertNear(fn(0, 0, 0), 0.5, 1e-9, "sin(30 deg)");
      },
    },
    {
      name: "del dot vector alias",
      fn: () => {
        const registry = makeRegistry([
          { kind: "alias", name: "V", rhs: String.raw`(-y,x,0)` },
        ]);
        const compiled = compileExpr(String.raw`\del\cdot V`, registry);
        assert(compiled.operator === "divergence", "divergence operator");
        assertNear(compiled.bind({})(0.2, 0.3, 0.1), 0, 0.05, "del dot alias");
      },
    },
    {
      name: "vector calc operators are not free params",
      fn: () => {
        for (const latex of [
          String.raw`\operatorname{div}`,
          String.raw`\operatorname{laplacian}`,
        ]) {
          const compiled = compileExpr(latex);
          assert(
            compiled.freeParams.length === 0,
            `${latex} must not promote operator name (${compiled.freeParams})`,
          );
        }
        let curlRejected = false;
        try {
          compileExpr(String.raw`\operatorname{curl}`);
        } catch (e) {
          curlRejected = e instanceof Error && /Vector field|unsupported|Could not compile/i.test(e.message);
        }
        assert(curlRejected, "bare operatorname{curl} rejected as scalar field");
        let gradRejected = false;
        try {
          compileExpr(String.raw`\operatorname{grad}`);
        } catch (e) {
          gradRejected = e instanceof Error && /Vector field|unsupported|Could not compile/i.test(e.message);
        }
        assert(gradRejected, "bare operatorname{grad} rejected as scalar field");
      },
    },
    {
      name: "grad mathrm scalar alias",
      fn: () => {
        const registry = makeRegistry([
          { kind: "alias", name: "f", rhs: String.raw`x^2+y^2` },
        ]);
        const compiled = compileVectorExpr(String.raw`\grad\mathrm{f}`, registry);
        assert(compiled.kind === "gradient", "gradient operator");
        const [gx] = compiled.bind({})(1, 2, 0);
        assertNear(gx, 2, 0.05, "grad mathrm fx");
      },
    },
  ]);
}
