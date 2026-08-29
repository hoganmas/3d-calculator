export function assert(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

export function assertNear(
  actual: number,
  expected: number,
  tol: number,
  label = "",
): void {
  if (!Number.isFinite(actual)) {
    throw new Error(`${label} not finite: ${actual}`);
  }
  if (Math.abs(actual - expected) > tol) {
    throw new Error(`${label} expected ${expected}, got ${actual} (tol ${tol})`);
  }
}
