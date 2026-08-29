/** Scheduling helpers for ephemeral auto-param commit on blur / render. */

export function shouldCommitAutoParamsAfterBlur(args: {
  fromExprId: string;
  suppressAutoCommit: boolean;
  focusedExprId: string | null;
}): boolean {
  if (args.suppressAutoCommit) return false;
  if (args.focusedExprId === args.fromExprId) return false;
  return true;
}

export type DeferredScheduler = {
  microtask: (fn: () => void) => void;
  raf: (fn: () => void) => number;
};

export const defaultScheduler: DeferredScheduler = {
  microtask: (fn) => queueMicrotask(fn),
  raf: (fn) => requestAnimationFrame(fn),
};

/** microtask → rAF → rAF (matches ExprSidebar blur commit timing). */
export function scheduleDoubleRaf(
  fn: () => void,
  scheduler: DeferredScheduler = defaultScheduler,
): void {
  scheduler.microtask(() => {
    scheduler.raf(() => {
      scheduler.raf(fn);
    });
  });
}

export function scheduleCommitIfLeftExpr(
  fromExprId: string,
  ctx: {
    isSuppressingAutoCommit: () => boolean;
    getFocusedExprId: () => string | null;
    commitAutoParams: () => void;
    onExprChange: () => void;
  },
  scheduler: DeferredScheduler = defaultScheduler,
): void {
  scheduleDoubleRaf(() => {
    if (
      !shouldCommitAutoParamsAfterBlur({
        fromExprId,
        suppressAutoCommit: ctx.isSuppressingAutoCommit(),
        focusedExprId: ctx.getFocusedExprId(),
      })
    ) {
      return;
    }
    ctx.commitAutoParams();
    ctx.onExprChange();
  }, scheduler);
}

/** Counter suppressed during list render / drag so blur does not commit mid-update. */
export function createSuppressAutoCommitCounter(scheduler: DeferredScheduler = defaultScheduler) {
  let count = 0;
  return {
    begin() {
      count++;
    },
    end() {
      scheduleDoubleRaf(() => {
        count = Math.max(0, count - 1);
      }, scheduler);
    },
    isActive() {
      return count > 0;
    },
    /** Test hook */
    getCount() {
      return count;
    },
  };
}
