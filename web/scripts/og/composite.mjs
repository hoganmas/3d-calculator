export function buildCompositeHtml(scenes, logoSvg) {
  const logoData = `data:image/svg+xml;base64,${Buffer.from(logoSvg).toString("base64")}`;
  const cols = Math.min(3, Math.max(1, scenes.length));
  const cards = scenes
    .map(
      (scene) => `
    <figure class="card">
      <img class="shot" src="data:image/png;base64,${scene.png}" alt="" />
      <figcaption>${escapeHtml(scene.label)}</figcaption>
    </figure>`,
    )
    .join("");

  return `<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    width: 1200px; height: 630px; overflow: hidden;
    background: #1a1228;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    color: #f5eef8;
  }
  .wrap {
    position: relative;
    width: 100%; height: 100%;
  }
  .shots {
    width: 100%; height: 100%;
    display: grid;
    grid-template-columns: repeat(${cols}, 1fr);
    gap: 3px;
  }
  .card {
    position: relative;
    min-width: 0;
    min-height: 0;
    overflow: hidden;
  }
  /* Each panel is captured at its exact on-composite pixel size (see
     renderShareOg.mjs), so the shot fills its cell with no scaling/cropping. */
  .shot {
    display: block;
    width: 100%;
    height: 100%;
  }
  figcaption {
    position: absolute;
    left: 0; right: 0; bottom: 0;
    padding: 8px 12px 7px;
    background: linear-gradient(to top, rgba(15,10,24,0.72), rgba(15,10,24,0));
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 12px;
    color: #f0e8f7;
    text-align: center;
    line-height: 1.3;
  }
  .brand {
    position: absolute;
    top: 18px; left: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    z-index: 1;
    filter: drop-shadow(0 1px 6px rgba(0,0,0,0.55));
  }
  .brand img { width: 34px; height: 34px; flex: 0 0 auto; }
  .brand span {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #fff;
  }
</style>
</head>
<body>
  <div class="wrap">
    <div class="shots">${cards}</div>
    <div class="brand">
      <img src="${logoData}" alt="" />
      <span>laplaci</span>
    </div>
  </div>
</body></html>`;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
