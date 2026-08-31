/** Mobile expression editing uses the footer carousel instead of the sidebar list. */
export function isMobileExprUi(): boolean {
  return document.documentElement.dataset.panelLayout === "horizontal";
}
