/** MCP relay config derived from the current page URL (no hardcoded deploy origin). */

export function getPageOrigin(): string {
  try {
    return location.origin;
  } catch {
    return "http://127.0.0.1:5173";
  }
}

/** Comma-separated `--widget-origin` value for @mcp-b/webmcp-local-relay. */
export function getWidgetOriginsForRelay(origin = getPageOrigin()): string {
  try {
    const { protocol, hostname, port } = location;
    const withHost = (host: string) =>
      `${protocol}//${host}${port ? `:${port}` : ""}`;
    if (hostname === "localhost" || hostname === "127.0.0.1") {
      const a = withHost("localhost");
      const b = withHost("127.0.0.1");
      return a === b ? a : [a, b].join(",");
    }
    return origin;
  } catch {
    return origin;
  }
}

export function buildMcpConfig(origin = getPageOrigin()): {
  mcpServers: {
    "laplacian-webmcp": {
      command: string;
      args: string[];
    };
  };
} {
  return {
    mcpServers: {
      "laplacian-webmcp": {
        command: "npx",
        args: [
          "-y",
          "@mcp-b/webmcp-local-relay@latest",
          "--widget-origin",
          getWidgetOriginsForRelay(origin),
        ],
      },
    },
  };
}

export function buildMcpConfigJson(origin = getPageOrigin()): string {
  return JSON.stringify(buildMcpConfig(origin), null, 2);
}

export function setupDismissStorageKey(origin = getPageOrigin()): string {
  return `laplacian-webmcp-setup-dismissed:${origin}`;
}

export function dismissSetupPrompt(origin = getPageOrigin()) {
  try {
    localStorage.setItem(setupDismissStorageKey(origin), "1");
  } catch {
    /* ignore */
  }
}
