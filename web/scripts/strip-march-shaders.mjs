import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const marchPath = path.join(__dirname, "../src/render/webgpu/march.js");
let src = fs.readFileSync(marchPath, "utf8");

const start = src.indexOf("/** Spatial + normal gradient factor");
const end = src.indexOf("/** @type {GPUDevice | null} */");
if (start === -1 || end === -1) throw new Error("shader block bounds not found");

const header = src.slice(0, start);
const tail = src.slice(end);

const importLine =
  'import { getIsoShader, getBeerShader, getGridShader, getAxisLabelShader, getFxaaShader, getSsaoShader } from "./shaders/compose.js";\n';

const newSrc =
  header.trimEnd() +
  "\n" +
  importLine +
  "\n" +
  tail;

fs.writeFileSync(marchPath, newSrc);
console.log("Removed inline WGSL from march.js; new length", newSrc.split("\n").length);
