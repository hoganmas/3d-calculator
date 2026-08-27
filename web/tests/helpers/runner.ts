export type TestCase = {
  name: string;
  fn: () => void | Promise<void>;
};

export async function runSuite(suite: string, cases: TestCase[]): Promise<number> {
  console.log(`\n${suite}`);
  let failed = 0;
  for (const c of cases) {
    try {
      await c.fn();
      console.log(`  ok ${c.name}`);
    } catch (e) {
      failed++;
      const msg = e instanceof Error ? e.message : String(e);
      console.error(`  FAIL ${c.name}: ${msg}`);
    }
  }
  return failed;
}
