/** Whether a layer must refit this uploadFit pass. */
export function layerNeedsRefit(
  fromAnim: boolean,
  contentDirty: boolean,
  paramDepends: boolean,
): boolean {
  if (!fromAnim) return contentDirty;
  return contentDirty || paramDepends;
}
