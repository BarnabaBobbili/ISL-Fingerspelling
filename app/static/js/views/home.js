/* Home: hero signer cycling through words, headline figures taken from the measured results. */
import { $ } from '/static/js/ui.js';
import { Avatar } from '/static/js/avatar.js';

const WORDS = ['NAMASTE', 'INDIA', 'SIGN', 'WELCOME', 'THANK YOU'];
let av = null, running = false, i = 0, factsDone = false;

function wordHtml(w, idx = -1) {
    return [...w].map((c, k) => `<span style="${k === idx ? 'color:var(--accent)' : ''}">${c === ' ' ? '&nbsp;' : c}</span>`).join('');
}

function next() {
    if (!running) return;
    const w = WORDS[i++ % WORDS.length];
    $('hero-word').innerHTML = wordHtml(w);
    av.play(w, {
        speed: 0.85,
        onItem: it => { if (it.ch && it.ch !== ' ') $('hero-word').innerHTML = wordHtml(w, it.idx); },
        onDone: () => { $('hero-word').innerHTML = wordHtml(w); setTimeout(next, 700); },
    });
}

async function facts() {
    factsDone = true;
    try {
        const d = await (await fetch('/static/data/results.json')).json(), h = d.headline;
        const set = (k, n, unit) => { const el = document.querySelector(`#home-facts .fact:nth-child(${k}) .n`); if (el && n != null) el.innerHTML = `${n}${unit ? `<small>${unit}</small>` : ''}`; };
        set(1, h.real_acc, '%'); set(2, h.cer, '%');
    } catch { /* keep the static figures */ }
}

export default {
    async enter() {
        if (!factsDone) facts();
        if (!av) { av = new Avatar($('hero-canvas'), { noControls: true }); await av.load(); av.setView('front'); }
        running = true; av.start(); next();
    },
    leave() { running = false; if (av) { av.stop(false); av.pause(); } },
};
