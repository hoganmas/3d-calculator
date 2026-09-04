/** Vercel Blob key for a share payload's OG image — shared by api/og.ts and api/upload-og.ts. */
export function ogBlobKey(payload: string): string {
  return `og/${payload}.png`;
}
