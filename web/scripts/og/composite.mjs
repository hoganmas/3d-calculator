export function buildCompositeHtml(scenes, logoSvg) {
  const logoData = `data:image/svg+xml;base64,${Buffer.from(logoSvg).toString("base64")}`;
  const cols = Math.min(3, Math.max(1, scenes.length));
  const cards = scenes
    .map(
      (scene) => `
    <figure class="card">
      <div class="shot"><img src="data:image/png;base64,${scene.png}" alt="" /></div>
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
    width: 100%; height: 100%;
    display: flex;
    flex-direction: column;
    gap: 28px;
    padding: 36px 44px 32px;
  }
  .brand {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    text-align: center;
  }
  .logo-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 22px;
  }
  .logo-row img { width: 88px; height: 88px; flex: 0 0 auto; }
  .logo-row h1 {
    font-size: 72px;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1;
  }
  .tagline {
    font-size: 30px;
    color: #c9b8d9;
    line-height: 1.2;
    letter-spacing: -0.01em;
  }
  .shots {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: repeat(${cols}, 1fr);
    gap: 18px;
    align-items: stretch;
  }
  .card {
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 0;
    min-height: 0;
  }
  .shot {
    flex: 1;
    min-height: 0;
    border-radius: 18px;
    overflow: hidden;
    border: 2px solid #3d2d52;
    background: #0f0a18;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .shot img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    object-position: center;
    display: block;
  }
  figcaption {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 12px;
    color: #9a86b0;
    text-align: center;
    line-height: 1.3;
    flex: 0 0 auto;
  }
</style>
</head>
<body>
  <div class="wrap">
    <header class="brand">
      <div class="logo-row">
        <img src="${logoData}" alt="" />
        <h1>laplaci</h1>
      </div>
      <p class="tagline">3D Graphing Calculator</p>
    </header>
    <div class="shots">${cards}</div>
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
