/**
 * Named symbols: parameters, spatial aliases, and user function definitions.
 * Each kind has its own namespace — `a=1` and `f(x)=…` may share the name `a`.
 */

import type { DeclSymbolKind } from "../types/models.js";

export interface SymbolEntry {
  kind: DeclSymbolKind;
  name: string;
  rhsLatex: string;
  latex: string;
  exprId: string;
  funcArgs?: string[];
}

export class SymbolRegistry {
  private params = new Map<string, SymbolEntry>();
  private aliases = new Map<string, SymbolEntry>();
  private funcdefs = new Map<string, SymbolEntry>();

  private mapFor(kind: DeclSymbolKind) {
    if (kind === "parameter") return this.params;
    if (kind === "alias") return this.aliases;
    return this.funcdefs;
  }

  tryAdd(entry: SymbolEntry): string | null {
    const map = this.mapFor(entry.kind);
    if (map.has(entry.name)) {
      const kindLabel =
        entry.kind === "parameter" ? "parameter" : entry.kind === "alias" ? "alias" : "function";
      return `${kindLabel.charAt(0).toUpperCase()}${kindLabel.slice(1)} “${entry.name}” is already declared`;
    }
    map.set(entry.name, entry);
    return null;
  }

  getAlias(name: string) {
    return this.aliases.get(name);
  }

  getFuncdef(name: string) {
    return this.funcdefs.get(name);
  }

  getParam(name: string) {
    return this.params.get(name);
  }
}

export function isDeclSymbolKind(kind: string | undefined): kind is DeclSymbolKind {
  return kind === "parameter" || kind === "alias" || kind === "funcdef";
}
