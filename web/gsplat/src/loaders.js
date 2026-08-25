/** SH degree-0 constant: RGB = 0.5 + C0 * f_dc */
const SH_C0 = 0.28209479177387814;

function clamp01(x) {
  return Math.min(1, Math.max(0, x));
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sceneFromArrays(means, scales, quats, colors, opacities, meta = {}) {
  const n = means.length;
  return {
    format: meta.format || "web-gsplat-v1",
    count: n,
    means,
    scales,
    quats,
    colors,
    opacities,
    notes: meta.notes || "",
    source: meta.source || "",
  };
}

/** Axis-aligned bounds + camera fit for orbit controls. */
export function sceneBounds(scene) {
  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;
  for (const m of scene.means) {
    minX = Math.min(minX, m[0]);
    minY = Math.min(minY, m[1]);
    minZ = Math.min(minZ, m[2]);
    maxX = Math.max(maxX, m[0]);
    maxY = Math.max(maxY, m[1]);
    maxZ = Math.max(maxZ, m[2]);
  }
  const target = [(minX + maxX) * 0.5, (minY + maxY) * 0.5, (minZ + maxZ) * 0.5];
  const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-3);
  return { target, extent, radius: extent * 1.6 };
}

function subsampleIndices(n, maxCount, seed = 1) {
  if (!maxCount || n <= maxCount) {
    return Array.from({ length: n }, (_, i) => i);
  }
  // Deterministic stride + jitter for even coverage
  const out = new Array(maxCount);
  const step = n / maxCount;
  let s = seed >>> 0;
  for (let i = 0; i < maxCount; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const jitter = (s / 0xffffffff) * step * 0.5;
    out[i] = Math.min(n - 1, Math.floor(i * step + jitter));
  }
  return out;
}

function pick(arr, indices) {
  return indices.map((i) => arr[i]);
}

/**
 * Decode INRIA / nerfstudio-style 3DGS PLY (binary_little_endian or ascii).
 * Uses DC SH for color; exp(scale); sigmoid(opacity).
 */
export function parseGaussianPly(buffer, opts = {}) {
  const maxCount = opts.maxCount ?? 0;
  const bytes = new Uint8Array(buffer);
  const headerMax = Math.min(bytes.length, 65536);
  let headerText = "";
  let headerEnd = -1;
  for (let i = 0; i < headerMax - 10; i++) {
    if (
      bytes[i] === 101 &&
      bytes[i + 1] === 110 &&
      bytes[i + 2] === 100 &&
      bytes[i + 3] === 95 &&
      bytes[i + 4] === 104
    ) {
      // "end_header\n" or "\r\n"
      let j = i;
      while (j < headerMax && bytes[j] !== 10) j++;
      headerEnd = j + 1;
      headerText = new TextDecoder().decode(bytes.subarray(0, headerEnd));
      break;
    }
  }
  if (headerEnd < 0) throw new Error("PLY: missing end_header");

  const lines = headerText.split(/\r?\n/);
  let format = null;
  let vertexCount = 0;
  const props = [];
  let inVertex = false;

  for (const line of lines) {
    const t = line.trim();
    if (t.startsWith("format ")) {
      format = t.slice(7).split(/\s+/)[0];
    } else if (t.startsWith("element ")) {
      const parts = t.split(/\s+/);
      inVertex = parts[1] === "vertex";
      if (inVertex) vertexCount = Number(parts[2]);
    } else if (inVertex && t.startsWith("property ")) {
      const parts = t.split(/\s+/);
      // property <type> <name>  OR  property list ...
      if (parts[1] === "list") continue;
      props.push({ type: parts[1], name: parts[2] });
    }
  }

  if (!format) throw new Error("PLY: no format line");
  if (!vertexCount) throw new Error("PLY: no vertex element");

  const need = ["x", "y", "z"];
  for (const n of need) {
    if (!props.some((p) => p.name === n)) {
      throw new Error(`PLY: missing property '${n}' (not a 3DGS PLY?)`);
    }
  }

  const typeSize = {
    char: 1,
    uchar: 1,
    short: 2,
    ushort: 2,
    int: 4,
    uint: 4,
    float: 4,
    double: 8,
    int8: 1,
    uint8: 1,
    int16: 2,
    uint16: 2,
    int32: 4,
    uint32: 4,
    float32: 4,
    float64: 8,
  };

  const propIndex = Object.fromEntries(props.map((p, i) => [p.name, i]));
  const stride = props.reduce((s, p) => s + (typeSize[p.type] || 4), 0);

  const hasScale = "scale_0" in propIndex;
  const hasOpacity = "opacity" in propIndex;
  const hasDc = "f_dc_0" in propIndex;
  const hasRot = "rot_0" in propIndex;
  const hasRgb = "red" in propIndex || "r" in propIndex;

  const indices = subsampleIndices(vertexCount, maxCount);
  const means = new Array(indices.length);
  const scales = new Array(indices.length);
  const quats = new Array(indices.length);
  const colors = new Array(indices.length);
  const opacities = new Array(indices.length);

  if (format === "binary_little_endian" || format === "binary_big_endian") {
    const le = format === "binary_little_endian";
    const view = new DataView(buffer, headerEnd);
    const offsets = [];
    let off = 0;
    for (const p of props) {
      offsets.push(off);
      off += typeSize[p.type] || 4;
    }

    function readProp(base, name) {
      const pi = propIndex[name];
      if (pi === undefined) return 0;
      const o = base + offsets[pi];
      const t = props[pi].type;
      if (t === "float" || t === "float32") return view.getFloat32(o, le);
      if (t === "double" || t === "float64") return view.getFloat64(o, le);
      if (t === "uchar" || t === "uint8") return view.getUint8(o);
      if (t === "char" || t === "int8") return view.getInt8(o);
      if (t === "ushort" || t === "uint16") return view.getUint16(o, le);
      if (t === "short" || t === "int16") return view.getInt16(o, le);
      if (t === "uint" || t === "uint32") return view.getUint32(o, le);
      if (t === "int" || t === "int32") return view.getInt32(o, le);
      return view.getFloat32(o, le);
    }

    for (let k = 0; k < indices.length; k++) {
      const i = indices[k];
      const base = i * stride;
      means[k] = [readProp(base, "x"), readProp(base, "y"), readProp(base, "z")];

      if (hasScale) {
        scales[k] = [
          Math.exp(readProp(base, "scale_0")),
          Math.exp(readProp(base, "scale_1")),
          Math.exp(readProp(base, "scale_2")),
        ];
      } else {
        scales[k] = [0.01, 0.01, 0.01];
      }

      if (hasRot) {
        let w = readProp(base, "rot_0");
        let x = readProp(base, "rot_1");
        let y = readProp(base, "rot_2");
        let z = readProp(base, "rot_3");
        const nrm = Math.hypot(w, x, y, z) || 1;
        quats[k] = [w / nrm, x / nrm, y / nrm, z / nrm];
      } else {
        quats[k] = [1, 0, 0, 0];
      }

      if (hasOpacity) {
        opacities[k] = clamp01(sigmoid(readProp(base, "opacity")));
      } else {
        opacities[k] = 0.8;
      }

      if (hasDc) {
        colors[k] = [
          clamp01(0.5 + SH_C0 * readProp(base, "f_dc_0")),
          clamp01(0.5 + SH_C0 * readProp(base, "f_dc_1")),
          clamp01(0.5 + SH_C0 * readProp(base, "f_dc_2")),
        ];
      } else if (hasRgb) {
        const rName = "red" in propIndex ? "red" : "r";
        const gName = "green" in propIndex ? "green" : "g";
        const bName = "blue" in propIndex ? "blue" : "b";
        const scale = props[propIndex[rName]].type.includes("uchar") ? 1 / 255 : 1;
        colors[k] = [
          clamp01(readProp(base, rName) * scale),
          clamp01(readProp(base, gName) * scale),
          clamp01(readProp(base, bName) * scale),
        ];
      } else {
        colors[k] = [0.8, 0.8, 0.8];
      }
    }
  } else if (format === "ascii") {
    const text = new TextDecoder().decode(bytes.subarray(headerEnd));
    const rows = text.trim().split(/\r?\n/);
    if (rows.length < vertexCount) {
      throw new Error(`PLY ascii: expected ${vertexCount} vertices, got ${rows.length}`);
    }
    for (let k = 0; k < indices.length; k++) {
      const parts = rows[indices[k]].trim().split(/\s+/).map(Number);
      const get = (name) => {
        const pi = propIndex[name];
        return pi === undefined ? 0 : parts[pi];
      };
      means[k] = [get("x"), get("y"), get("z")];
      scales[k] = hasScale
        ? [Math.exp(get("scale_0")), Math.exp(get("scale_1")), Math.exp(get("scale_2"))]
        : [0.01, 0.01, 0.01];
      if (hasRot) {
        let w = get("rot_0"),
          x = get("rot_1"),
          y = get("rot_2"),
          z = get("rot_3");
        const nrm = Math.hypot(w, x, y, z) || 1;
        quats[k] = [w / nrm, x / nrm, y / nrm, z / nrm];
      } else quats[k] = [1, 0, 0, 0];
      opacities[k] = hasOpacity ? clamp01(sigmoid(get("opacity"))) : 0.8;
      colors[k] = hasDc
        ? [
            clamp01(0.5 + SH_C0 * get("f_dc_0")),
            clamp01(0.5 + SH_C0 * get("f_dc_1")),
            clamp01(0.5 + SH_C0 * get("f_dc_2")),
          ]
        : [0.8, 0.8, 0.8];
    }
  } else {
    throw new Error(`PLY: unsupported format '${format}'`);
  }

  return sceneFromArrays(means, scales, quats, colors, opacities, {
    format: "3dgs-ply",
    notes: `Loaded ${indices.length}/${vertexCount} Gaussians from PLY`,
    source: "ply",
    totalInFile: vertexCount,
    loaded: indices.length,
  });
}

/**
 * antimatter15 / Kevin Kwok .splat — 32 bytes/splat:
 * float32 xyz ×3, float32 scale ×3, uint8 rgba ×4, uint8 quat ×4
 */
export function parseSplatFile(buffer, opts = {}) {
  const maxCount = opts.maxCount ?? 0;
  const bytes = buffer.byteLength;
  if (bytes % 32 !== 0) {
    throw new Error(`.splat size ${bytes} is not a multiple of 32`);
  }
  const total = bytes / 32;
  const indices = subsampleIndices(total, maxCount);
  const view = new DataView(buffer);
  const means = new Array(indices.length);
  const scales = new Array(indices.length);
  const quats = new Array(indices.length);
  const colors = new Array(indices.length);
  const opacities = new Array(indices.length);

  for (let k = 0; k < indices.length; k++) {
    const base = indices[k] * 32;
    means[k] = [
      view.getFloat32(base + 0, true),
      view.getFloat32(base + 4, true),
      view.getFloat32(base + 8, true),
    ];
    scales[k] = [
      view.getFloat32(base + 12, true),
      view.getFloat32(base + 16, true),
      view.getFloat32(base + 20, true),
    ];
    colors[k] = [
      view.getUint8(base + 24) / 255,
      view.getUint8(base + 25) / 255,
      view.getUint8(base + 26) / 255,
    ];
    opacities[k] = view.getUint8(base + 27) / 255;
    // quat stored as uint8 in [-1,1] mapped; order often (x,y,z,w) in this format
    const qx = (view.getUint8(base + 28) - 128) / 128;
    const qy = (view.getUint8(base + 29) - 128) / 128;
    const qz = (view.getUint8(base + 30) - 128) / 128;
    const qw = (view.getUint8(base + 31) - 128) / 128;
    const nrm = Math.hypot(qw, qx, qy, qz) || 1;
    quats[k] = [qw / nrm, qx / nrm, qy / nrm, qz / nrm];
  }

  return sceneFromArrays(means, scales, quats, colors, opacities, {
    format: "antimatter-splat",
    notes: `Loaded ${indices.length}/${total} Gaussians from .splat`,
    source: "splat",
    totalInFile: total,
    loaded: indices.length,
  });
}

export function parseSceneJson(text, opts = {}) {
  const data = typeof text === "string" ? JSON.parse(text) : text;
  if (!data.means || !data.count) {
    throw new Error("JSON scene missing means/count (expected web-gsplat-v1)");
  }
  const maxCount = opts.maxCount ?? 0;
  if (maxCount && data.count > maxCount) {
    const idx = subsampleIndices(data.count, maxCount);
    return sceneFromArrays(
      pick(data.means, idx),
      pick(data.scales || Array(data.count).fill([0.05, 0.05, 0.05]), idx),
      pick(data.quats || Array(data.count).fill([1, 0, 0, 0]), idx),
      pick(data.colors || Array(data.count).fill([0.8, 0.8, 0.8]), idx),
      pick(data.opacities || Array(data.count).fill(0.5), idx),
      {
        format: data.format || "web-gsplat-v1",
        notes: `Subsampled ${idx.length}/${data.count} from JSON`,
        source: "json",
        totalInFile: data.count,
        loaded: idx.length,
      },
    );
  }
  return {
    ...data,
    source: "json",
    totalInFile: data.count,
    loaded: data.count,
  };
}

export async function loadSceneFromFile(file, opts = {}) {
  const name = file.name.toLowerCase();
  const buf = await file.arrayBuffer();
  if (name.endsWith(".ply")) return parseGaussianPly(buf, opts);
  if (name.endsWith(".splat")) return parseSplatFile(buf, opts);
  if (name.endsWith(".json")) {
    return parseSceneJson(new TextDecoder().decode(buf), opts);
  }
  // sniff
  const head = new TextDecoder().decode(buf.slice(0, 4));
  if (head === "ply\n" || head.startsWith("ply")) return parseGaussianPly(buf, opts);
  if (head.trim().startsWith("{")) {
    return parseSceneJson(new TextDecoder().decode(buf), opts);
  }
  if (buf.byteLength % 32 === 0) return parseSplatFile(buf, opts);
  throw new Error(`Unrecognized 3DGS file: ${file.name} (use .ply / .splat / .json)`);
}
