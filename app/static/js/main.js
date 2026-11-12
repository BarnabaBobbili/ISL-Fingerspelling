/* App shell: hash router, theme toggle, mobile menu, server status. Views are loaded lazily. */
import { $, $$, hydrateIcons, icon, setTheme, currentTheme } from '/static/js/ui.js';

const TITLES = { home: 'ISL fingerspelling — recognition and generation', recognise: 'Recognise · ISL fingerspelling', sign: 'Sign · ISL fingerspelling', alphabet: 'Alphabet · ISL fingerspelling', results: 'Results · ISL fingerspelling', docs: 'Documentation · ISL fingerspelling' };
const modules = {};
let active = null;

function parse() {
    const h = location.hash.replace(/^#\/?/, '');
    const [name, ...rest] = h.split('/');
    return { name: TITLES[name] ? name : 'home', params: rest };
}

async function show() {
    const { name, params } = parse();
    if (active && active.name !== name) { try { active.mod.leave && active.mod.leave(); } catch (e) { console.error(e); } }
    $$('.view').forEach(v => v.classList.toggle('active', v.dataset.view === name));
    $$('#nav a').forEach(a => a.getAttribute('data-route') === name ? a.setAttribute('aria-current', 'page') : a.removeAttribute('aria-current'));
    document.title = TITLES[name];
    document.getElementById('nav').classList.remove('open');
    document.getElementById('menu-btn').setAttribute('aria-expanded', 'false');
    document.getElementById('site-footer').hidden = name === 'docs';
    if (!modules[name]) modules[name] = import(`/static/js/views/${name}.js`).then(m => m.default);
    const mod = await modules[name];
    const first = !active || active.name !== name;
    active = { name, mod };
    await mod.enter(params, first);
    if (first && name !== 'docs') window.scrollTo({ top: 0 });
}

function initTheme() {
    const btn = $('theme-btn');
    const paint = () => { const t = currentTheme(); btn.innerHTML = icon(t === 'dark' ? 'sun' : 'moon'); btn.setAttribute('aria-label', t === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'); };
    btn.onclick = () => setTheme(currentTheme() === 'dark' ? 'light' : 'dark');
    window.addEventListener('themechange', paint); paint();
}

async function pollStatus() {
    const s = $('status'), label = s.querySelector('span');
    try {
        const d = await (await fetch('/api/info')).json();
        s.className = 'status ' + (d.ready ? 'ok' : ''); label.textContent = d.ready ? 'models ready' : 'loading models…';
        $('foot-models').textContent = `static: ${d.static_model} · continuous: ${d.continuous_model || (d.continuous_available ? 'available' : 'not installed')}`;
        if (!d.ready) setTimeout(pollStatus, 1500);
    } catch { s.className = 'status bad'; label.textContent = 'server offline'; setTimeout(pollStatus, 3000); }
}

hydrateIcons();
initTheme();
$('menu-btn').onclick = () => { const n = $('nav'); const o = n.classList.toggle('open'); $('menu-btn').setAttribute('aria-expanded', String(o)); };
window.addEventListener('hashchange', show);
pollStatus();
show();
