import { isDynamicOgEnabled } from "../../../api/_lib/ogFeatureFlag.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function withFlag<T>(value: string | undefined, fn: () => T): T {
  const prev = process.env.OG_DYNAMIC_ENABLED;
  if (value === undefined) delete process.env.OG_DYNAMIC_ENABLED;
  else process.env.OG_DYNAMIC_ENABLED = value;
  try {
    return fn();
  } finally {
    if (prev === undefined) delete process.env.OG_DYNAMIC_ENABLED;
    else process.env.OG_DYNAMIC_ENABLED = prev;
  }
}

export async function run() {
  return runSuite("persistence / ogFeatureFlag (api)", [
    {
      name: "unset defaults to disabled",
      fn: () => withFlag(undefined, () => assert(!isDynamicOgEnabled(), "disabled by default")),
    },
    {
      name: "'1' enables",
      fn: () => withFlag("1", () => assert(isDynamicOgEnabled(), "enabled by '1'")),
    },
    {
      name: "'true' enables (case-insensitive, trims whitespace)",
      fn: () => withFlag(" True \n", () => assert(isDynamicOgEnabled(), "enabled by ' True '")),
    },
    {
      name: "'0'/'false'/garbage stay disabled",
      fn: () => {
        withFlag("0", () => assert(!isDynamicOgEnabled(), "'0' disabled"));
        withFlag("false", () => assert(!isDynamicOgEnabled(), "'false' disabled"));
        withFlag("yes", () => assert(!isDynamicOgEnabled(), "'yes' disabled"));
      },
    },
  ]);
}
