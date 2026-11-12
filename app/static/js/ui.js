/* Small UI helpers shared by all views: icons, theme, toast, speech, clipboard. */

export const $ = (id, root = document) => root.getElementById ? root.getElementById(id) : root.querySelector('#' + id);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/* Stroke icons, 24x24. (the hand glyph follows Lucide, ISC licence) */
const ICONS = {
    camera: '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3.2"/>',
    upload: '<path d="M12 16V4m0 0L8 8m4-4 4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    hand: '<path d="M18 11V6a2 2 0 0 0-2-2 2 2 0 0 0-2 2"/><path d="M14 10V4a2 2 0 0 0-2-2 2 2 0 0 0-2 2v2"/><path d="M10 10.5V6a2 2 0 0 0-2-2 2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/>',
    book: '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5z"/><path d="M4 19.5A2.5 2.5 0 0 0 6.5 22H20v-5"/>',
    chart: '<path d="M4 20V10m6 10V4m6 16v-7m4 7H2"/>',
    play: '<path d="M7 4.5v15l12-7.5z"/>',
    stop: '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
    mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
    copy: '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/>',
    volume: '<path d="M4 9v6h4l5 4V5L8 9z"/><path d="M16.5 8.5a5 5 0 0 1 0 7"/>',
    trash: '<path d="M4 7h16M10 11v6m4-6v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.4 1.4m11.2 11.2L19 19M5 19l1.4-1.4M17.6 6.4 19 5"/>',
    moon: '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>',
    menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
    backspace: '<path d="M21 5H9l-6 7 6 7h12z"/><path d="m13 9.5 5 5m0-5-5 5"/>',
    record: '<circle cx="12" cy="12" r="6"/>',
    download: '<path d="M12 4v12m0 0-4-4m4 4 4-4M4 20h16"/>',
    space: '<path d="M4 10v5h16v-5"/>',
    type: '<path d="M4 7V5h16v2M9 19h6M12 5v14"/>',
    alpha: '<path d="m3 19 6-14 6 14M5.5 14h7M17 8h4M19 6v4"/>',
};
export function icon(name, size = 18) {
    return `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ''}</svg>`;
}
export function hydrateIcons(root = document) {
    root.querySelectorAll('i[data-icon]').forEach(el => { el.outerHTML = icon(el.dataset.icon, +el.dataset.size || 18); });
}

/* ---- theme (light default, remembered) ---- */
const THEME_KEY = 'theme';
export function currentTheme() { return document.documentElement.getAttribute('data-theme') || 'light'; }
export function setTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    try { localStorage.setItem(THEME_KEY, t); } catch { /* private mode */ }
    document.querySelectorAll('iframe.docs-frame').forEach(f => syncDocsTheme(f));
    window.dispatchEvent(new CustomEvent('themechange', { detail: t }));
}
export function syncDocsTheme(frame) {
    try {
        const d = frame.contentDocument; if (!d) return;
        d.documentElement.setAttribute('data-theme', currentTheme());
        d.documentElement.setAttribute('data-embedded', '1');
        frame.contentWindow.localStorage.setItem(THEME_KEY, currentTheme());
    } catch { /* not same-origin yet */ }
}

/* ---- toast ---- */
let toastTimer;
export function toast(msg) {
    let t = document.querySelector('.toast');
    if (!t) { t = document.createElement('div'); t.className = 'toast'; t.setAttribute('role', 'status'); document.body.appendChild(t); }
    t.textContent = msg; t.classList.add('show');
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 1800);
}

/* ---- speech + clipboard ---- */
export function speak(text, rate = 0.95) {
    if (!('speechSynthesis' in window) || !text.trim()) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text.trim()); u.rate = rate; u.lang = 'en-IN';
    speechSynthesis.speak(u);
}
export async function copyText(text) {
    try { await navigator.clipboard.writeText(text); toast('Copied'); } catch { toast('Copy not available'); }
}
