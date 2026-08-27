/** Minimal WebMCP ModelContext types (native Chrome + @mcp-b/webmcp-polyfill). */

interface WebMCPToolDescriptor {
  name: string;
  description?: string;
  title?: string;
  inputSchema?: Record<string, unknown>;
  annotations?: {
    readOnlyHint?: boolean;
    untrustedContentHint?: boolean;
  };
}

interface WebMCPToolRegistration extends WebMCPToolDescriptor {
  execute: (input: Record<string, unknown>, ctx?: { signal?: AbortSignal }) => Promise<unknown>;
}

interface WebMCPRegisterOptions {
  signal?: AbortSignal;
  exposedTo?: string[];
}

interface WebMCPModelContext {
  registerTool(tool: WebMCPToolRegistration, options?: WebMCPRegisterOptions): Promise<void>;
  getTools(options?: { fromOrigins?: string[] }): Promise<WebMCPToolDescriptor[]>;
  executeTool?(tool: WebMCPToolDescriptor, inputJson: string, options?: { signal?: AbortSignal }): Promise<unknown>;
  addEventListener(type: "toolchange", listener: () => void): void;
  removeEventListener(type: "toolchange", listener: () => void): void;
}

interface Document {
  modelContext: WebMCPModelContext;
}
