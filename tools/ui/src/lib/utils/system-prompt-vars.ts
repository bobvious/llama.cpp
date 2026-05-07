/**
 * Template variable interpolation for the system prompt.
 *
 * Supported tokens (case-insensitive, double-curly):
 *   {{CURRENT_DATETIME}} - "2026-05-07 14:23:08" in browser's local timezone
 *   {{CURRENT_DATE}}     - "2026-05-07"
 *   {{CURRENT_TIME}}     - "14:23:08"
 *   {{CURRENT_WEEKDAY}}  - "Wednesday"
 *   {{ISO_DATETIME}}     - "2026-05-07T21:23:08.000Z" (UTC)
 *   {{TZ}}               - resolved timezone, e.g. "America/Phoenix"
 *   {{MODEL}}            - model name, if known at send time
 *
 * Interpolation runs at SEND time (in chat.service.ts), not at save time,
 * so the persisted system message stays as the template — every request
 * gets fresh values.
 */

const TWO_DIGIT = (n: number) => n.toString().padStart(2, '0');

function buildVars(model?: string): Record<string, string> {
	const now = new Date();
	const date = `${now.getFullYear()}-${TWO_DIGIT(now.getMonth() + 1)}-${TWO_DIGIT(now.getDate())}`;
	const time = `${TWO_DIGIT(now.getHours())}:${TWO_DIGIT(now.getMinutes())}:${TWO_DIGIT(
		now.getSeconds()
	)}`;
	const weekday = now.toLocaleDateString(undefined, { weekday: 'long' });
	let tz = '';
	try {
		tz = Intl.DateTimeFormat().resolvedOptions().timeZone || '';
	} catch {
		tz = '';
	}

	return {
		CURRENT_DATETIME: `${date} ${time}`,
		CURRENT_DATE: date,
		CURRENT_TIME: time,
		CURRENT_WEEKDAY: weekday,
		ISO_DATETIME: now.toISOString(),
		TZ: tz,
		MODEL: model ?? ''
	};
}

const TOKEN_RE = /\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}/g;

export function interpolateSystemPrompt(text: string, opts?: { model?: string }): string {
	if (!text || !text.includes('{{')) return text;
	const vars = buildVars(opts?.model);
	return text.replace(TOKEN_RE, (match, name: string) => {
		const upper = name.toUpperCase();
		return Object.prototype.hasOwnProperty.call(vars, upper) ? vars[upper] : match;
	});
}
