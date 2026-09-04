/** Default scenes for static public/og-image.png generation. */
export const DEFAULT_OG_SCENES = [
  {
    latex: String.raw`z=-\cos\left(x\right)\sin\left(2y\right)`,
    label: "z = −cos(x) sin(2y)",
    palette: 0,
    camera: { position: [6.8, 6.2, 4.8], target: [0, 0, 0] },
    settleMs: 2000,
  },
  {
    latex: String.raw`\left(0,z,-y\right)`,
    label: "(0, z, −y)",
    palette: 2,
    camera: { position: [7.2, 1.2, 5.8], target: [0, 0, 0] },
    settleMs: 3500,
  },
  {
    latex: String.raw`e^{-2.5r}\abs\left(2z^{2}-x^{2}-y^{2}\right)`,
    label: "e^{−2.5r}|2z² − x² − y²|",
    palette: 1,
    camera: { position: [6.4, 5.8, 5.2], target: [0, 0, 0] },
    settleMs: 3000,
  },
];
