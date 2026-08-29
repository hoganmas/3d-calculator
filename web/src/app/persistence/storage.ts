/**
 * IndexedDB two-slot write-ahead autosave storage.
 */
import { validateDocument, type LaplacianDocument } from "./documentSchema.js";

export const DB_NAME = "laplacian-autosave";
const STORE_NAME = "records";
const DB_VERSION = 1;

export type StorageSlot = "A" | "B";

export interface StorageHead {
  revision: number;
  savedAt: string;
  checksum: string;
  slot: StorageSlot;
}

type RecordKey = "head" | "slotA" | "slotB";

export interface StorageBackend {
  get(key: RecordKey): Promise<string | null>;
  put(key: RecordKey, value: string): Promise<void>;
  transaction<T>(fn: (tx: StorageBackend) => Promise<T>): Promise<T>;
}

/** FNV-1a 32-bit checksum (hex) for corruption detection. */
export function checksumOf(text: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

let backend: StorageBackend | null = null;

function openIndexedDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(req.error ?? new Error("IndexedDB open failed"));
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    req.onsuccess = () => resolve(req.result);
  });
}

function idbBackend(db: IDBDatabase): StorageBackend {
  return {
    async get(key) {
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readonly");
        const store = tx.objectStore(STORE_NAME);
        const req = store.get(key);
        req.onerror = () => reject(req.error ?? new Error("IndexedDB get failed"));
        req.onsuccess = () => resolve((req.result as string | undefined) ?? null);
      });
    },
    async put(key, value) {
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        const store = tx.objectStore(STORE_NAME);
        const req = store.put(value, key);
        req.onerror = () => reject(req.error ?? new Error("IndexedDB put failed"));
        req.onsuccess = () => resolve();
      });
    },
    async transaction(fn) {
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        const store = tx.objectStore(STORE_NAME);
        const adapter: StorageBackend = {
          get: (key) =>
            new Promise((res, rej) => {
              const r = store.get(key);
              r.onerror = () => rej(r.error ?? new Error("get failed"));
              r.onsuccess = () => res((r.result as string | undefined) ?? null);
            }),
          put: (key, value) =>
            new Promise((res, rej) => {
              const r = store.put(value, key);
              r.onerror = () => rej(r.error ?? new Error("put failed"));
              r.onsuccess = () => res();
            }),
          transaction: (inner) => inner(adapter),
        };
        fn(adapter)
          .then(resolve)
          .catch(reject);
        tx.onerror = () => reject(tx.error ?? new Error("transaction failed"));
      });
    },
  };
}

async function getBackend(): Promise<StorageBackend> {
  if (backend) return backend;
  const db = await openIndexedDb();
  backend = idbBackend(db);
  return backend;
}

/** Replace storage backend (in-memory shim for tests). */
export function setStorageBackend(next: StorageBackend | null) {
  backend = next;
}

/** In-memory two-slot backend for unit tests. */
export function createMemoryStorage(): StorageBackend {
  const data = new Map<RecordKey, string>();
  const backendImpl: StorageBackend = {
    async get(key) {
      return data.get(key) ?? null;
    },
    async put(key, value) {
      data.set(key, value);
    },
    async transaction(fn) {
      const snapshot = new Map(data);
      try {
        const result = await fn(backendImpl);
        return result;
      } catch (e) {
        data.clear();
        for (const [k, v] of snapshot) data.set(k, v);
        throw e;
      }
    },
  };
  return backendImpl;
}

function inactiveSlot(head: StorageHead): "slotA" | "slotB" {
  return head.slot === "A" ? "slotB" : "slotA";
}

function slotKey(slot: StorageSlot): "slotA" | "slotB" {
  return slot === "A" ? "slotA" : "slotB";
}

function tryParseSlot(json: string | null, expectedChecksum?: string): LaplacianDocument | null {
  if (!json) return null;
  if (expectedChecksum && checksumOf(json) !== expectedChecksum) return null;
  try {
    return validateDocument(JSON.parse(json));
  } catch {
    return null;
  }
}

export async function writeDocument(json: string, revision: number, savedAt?: string): Promise<void> {
  const parsed = validateDocument(JSON.parse(json));
  if (parsed.revision !== revision) {
    throw new Error("revision mismatch on write");
  }
  const sum = checksumOf(json);
  const store = await getBackend();
  const headRaw = await store.get("head");
  const prevHead: StorageHead | null = headRaw ? JSON.parse(headRaw) : null;
  const nextSlot: StorageSlot = prevHead?.slot === "A" ? "B" : "A";
  const head: StorageHead = {
    revision,
    savedAt: savedAt ?? parsed.savedAt,
    checksum: sum,
    slot: nextSlot,
  };
  const payloadKey = slotKey(nextSlot);
  await store.transaction(async (tx) => {
    await tx.put(payloadKey, json);
    await tx.put("head", JSON.stringify(head));
  });
}

export async function readDocument(): Promise<LaplacianDocument | null> {
  const store = await getBackend();
  const headRaw = await store.get("head");
  if (!headRaw) return null;
  let head: StorageHead;
  try {
    head = JSON.parse(headRaw);
  } catch {
    return null;
  }
  if (!head || typeof head !== "object") return null;

  const activeKey = slotKey(head.slot);
  const inactiveKey = inactiveSlot(head);
  const activeJson = await store.get(activeKey);
  const doc = tryParseSlot(activeJson, head.checksum);
  if (doc) return doc;

  const inactiveJson = await store.get(inactiveKey);
  return tryParseSlot(inactiveJson);
}
