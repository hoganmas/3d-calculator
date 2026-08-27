import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const marchPath = path.join(__dirname, "../src/render/webgpu/march.js");
const dir = path.join(__dirname, "../src/render/webgpu/shaders");
fs.mkdirSync(path.join(dir, "common"), { recursive: true });

const src = fs.readFileSync(marchPath, "utf8");

const gradientMatch = src.match(/const GRADIENT_WGSL = `\n([\s\S]*?)`;/);
if (!gradientMatch) throw new Error("GRADIENT_WGSL not found");
const gradientBody = gradientMatch[1].replace(/\$\{MAX_GRAD_STOPS\}/g, "{{MAX_GRAD_STOPS}}");
fs.writeFileSync(path.join(dir, "common/gradient.wgsl"), `${gradientBody.trim()}\n`);

function extractMakeFn(name) {
  const re = new RegExp(`function ${name}\\(\\) \\{\\s*return /\\* wgsl \\*/ \`([\\s\\S]*?)\`;\\s*\\}`);
  const m = src.match(re);
  if (!m) throw new Error(`${name} not found`);
  return m[1]
    .replace(/\$\{GRADIENT_WGSL\}/g, "{{GRADIENT_WGSL}}")
    .replace(/\$\{MAX_GRAD_STOPS\}/g, "{{MAX_GRAD_STOPS}}")
    .replace(/\$\{MAX_DENS_LAYERS\}/g, "{{MAX_DENS_LAYERS}}");
}

for (const [fn, file] of [
  ["makeBeerMultiWgsl", "beer.wgsl"],
  ["makeGridWgsl", "grid.wgsl"],
  ["makeAxisLabelWgsl", "axisLabel.wgsl"],
  ["makeFxaaWgsl", "fxaa.wgsl"],
  ["makeSsaoWgsl", "ssao.wgsl"],
]) {
  fs.writeFileSync(path.join(dir, file), `${extractMakeFn(fn).trim()}\n`);
}

const isoStart = src.indexOf("function makeIsoWgsl()");
const isoEnd = src.indexOf("\nfunction makeBeerMultiWgsl()");
const isoFn = src.slice(isoStart, isoEnd);

function extractSnippet(varName) {
  const re = new RegExp(
    `const ${varName} = isoInterpHermite\\s*\\? /\\* wgsl \\*/ \`([\\s\\S]*?)\`\\s*: /\\* wgsl \\*/ \`([\\s\\S]*?)\`;`,
  );
  const m = isoFn.match(re);
  if (!m) throw new Error(`${varName} not found`);
  return { hermite: m[1], trilinear: m[2] };
}

const vol = extractSnippet("volumeSample");
const grad = extractSnippet("gradSample");

const isoTemplateMatch = isoFn.match(/return \/\* wgsl \*\/ `([\s\S]*?)`;\n\}/);
if (!isoTemplateMatch) throw new Error("iso template not found");
const isoTemplate = isoTemplateMatch[1];

function buildIso(volumeSample, gradSample) {
  return isoTemplate
    .replace("${volumeSample}", volumeSample.trim())
    .replace("${gradSample}", gradSample.trim())
    .replace(/\$\{GRADIENT_WGSL\}/g, "{{GRADIENT_WGSL}}")
    .replace(/\$\{MAX_GRAD_STOPS\}/g, "{{MAX_GRAD_STOPS}}");
}

fs.writeFileSync(path.join(dir, "isoHermite.wgsl"), `${buildIso(vol.hermite, grad.hermite).trim()}\n`);
fs.writeFileSync(path.join(dir, "isoTrilinear.wgsl"), `${buildIso(vol.trilinear, grad.trilinear).trim()}\n`);

console.log("Extracted shaders to", dir);
