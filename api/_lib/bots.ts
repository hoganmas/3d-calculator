const CRAWLER_RE =
  /bot|facebookexternalhit|twitterbot|slackbot|discordbot|telegrambot|linkedinbot|embedly|whatsapp|pinterest|googlebot|bingpreview|applebot|preview/i;

export function isCrawlerUserAgent(ua: string | undefined): boolean {
  if (!ua) return false;
  return CRAWLER_RE.test(ua);
}
