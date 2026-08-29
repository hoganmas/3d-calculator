import "../helpers/setup-dom.ts";
import {
  shouldCommitAutoParamsAfterBlur,
  scheduleDoubleRaf,
  scheduleCommitIfLeftExpr,
  createSuppressAutoCommitCounter,
  type DeferredScheduler,
} from "../../src/ui/expr-sidebar/autoCommit.ts";
import { assert } from "../helpers/assert.ts";
import { runSuite } from "../helpers/runner.ts";

function makeScheduler(): { scheduler: DeferredScheduler; flush: () => void } {
  const microtasks: (() => void)[] = [];
  const rafs: (() => void)[] = [];
  const scheduler: DeferredScheduler = {
    microtask(fn) {
      microtasks.push(fn);
    },
    raf(fn) {
      rafs.push(fn);
    },
  };
  const flush = () => {
    while (microtasks.length) {
      const batch = microtasks.splice(0);
      for (const fn of batch) fn();
    }
    while (rafs.length) {
      const batch = rafs.splice(0);
      for (const fn of batch) fn();
    }
  };
  return { scheduler, flush };
}

export async function run() {
  return runSuite("ui / auto-commit", [
    {
      name: "shouldCommitAutoParamsAfterBlur: false while suppressed",
      fn: () => {
        assert(
          !shouldCommitAutoParamsAfterBlur({
            fromExprId: "e1",
            suppressAutoCommit: true,
            focusedExprId: null,
          }),
          "suppressed",
        );
      },
    },
    {
      name: "shouldCommitAutoParamsAfterBlur: false when focus stayed on row",
      fn: () => {
        assert(
          !shouldCommitAutoParamsAfterBlur({
            fromExprId: "e1",
            suppressAutoCommit: false,
            focusedExprId: "e1",
          }),
          "same row focused",
        );
      },
    },
    {
      name: "shouldCommitAutoParamsAfterBlur: true after leaving row",
      fn: () => {
        assert(
          shouldCommitAutoParamsAfterBlur({
            fromExprId: "e1",
            suppressAutoCommit: false,
            focusedExprId: "e2",
          }),
          "left row",
        );
        assert(
          shouldCommitAutoParamsAfterBlur({
            fromExprId: "e1",
            suppressAutoCommit: false,
            focusedExprId: null,
          }),
          "blurred entirely",
        );
      },
    },
    {
      name: "scheduleDoubleRaf runs microtask then two rAFs",
      fn: () => {
        const { scheduler, flush } = makeScheduler();
        const order: string[] = [];
        scheduleDoubleRaf(() => order.push("done"), scheduler);
        order.push("sync");
        flush();
        assert(order.join(",") === "sync,done", `order ${order.join(",")}`);
      },
    },
    {
      name: "scheduleCommitIfLeftExpr commits after blur timing",
      fn: () => {
        const { scheduler, flush } = makeScheduler();
        let committed = false;
        let changed = false;
        scheduleCommitIfLeftExpr(
          "e1",
          {
            isSuppressingAutoCommit: () => false,
            getFocusedExprId: () => "e2",
            commitAutoParams: () => {
              committed = true;
            },
            onExprChange: () => {
              changed = true;
            },
          },
          scheduler,
        );
        flush();
        assert(committed, "committed");
        assert(changed, "expr change");
      },
    },
    {
      name: "scheduleCommitIfLeftExpr skips when suppress active at run time",
      fn: () => {
        const { scheduler, flush } = makeScheduler();
        let committed = false;
        scheduleCommitIfLeftExpr(
          "e1",
          {
            isSuppressingAutoCommit: () => true,
            getFocusedExprId: () => null,
            commitAutoParams: () => {
              committed = true;
            },
            onExprChange: () => {},
          },
          scheduler,
        );
        flush();
        assert(!committed, "not committed while suppressed");
      },
    },
    {
      name: "createSuppressAutoCommitCounter defers end until double rAF",
      fn: () => {
        const { scheduler, flush } = makeScheduler();
        const ctrl = createSuppressAutoCommitCounter(scheduler);
        ctrl.begin();
        assert(ctrl.isActive(), "active after begin");
        ctrl.end();
        assert(ctrl.isActive(), "still active before flush");
        flush();
        assert(!ctrl.isActive(), "inactive after flush");
      },
    },
  ]);
}
