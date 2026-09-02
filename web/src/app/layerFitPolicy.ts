/** Whether a layer must refit this uploadFit pass. */
export function layerNeedsRefit(
  _fromAnim: boolean,
  contentDirty: boolean,
  paramDepends: boolean,
): boolean {
  return contentDirty || paramDepends;
}
