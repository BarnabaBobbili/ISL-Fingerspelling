/* Alphabet: all 26 letters with filters and a detail view. */
import { $, $$ } from '/static/js/ui.js';
import { Avatar, makeThumbnails } from '/static/js/avatar.js';

const LETTERS = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'];
let ready = false, poses = null, thumbs = null, detail = null, filter = 'all', selected = null;

function render() {
    const box = $('alpha'); box.innerHTML = '';
    let n = 0;
    LETTERS.forEach(L => {
        const p = poses.letters[L];
        if (filter !== 'all' && String(p.hands) !== filter) return;
        n++;
        const c = document.createElement('button'); c.type = 'button'; c.className = 'acard'; c.setAttribute('aria-pressed', L === selected);
        c.innerHTML = `<img src="${thumbs[L]}" alt="Letter ${L}, ${p.hands === 2 ? 'two hands' : 'one hand'}"><div><b>${L}</b><small>${p.hands === 2 ? '2 hands' : '1 hand'}<br>${p.contacts.length ? p.contacts.length + (p.contacts.length > 1 ? ' contacts' : ' contact') : '&nbsp;'}</small></div>`;
        c.onclick = () => choose(L);
        box.appendChild(c);
    });
    $('alpha-count').textContent = `${n} letter${n === 1 ? '' : 's'}`;
}

async function choose(L) {
    selected = L; render();
    const p = poses.letters[L];
    $('alpha-detail').hidden = false;
    $('ad-letter').textContent = L;
    $('ad-hands').textContent = p.hands === 2 ? 'two hands' : 'one hand (the other rests)';
    $('ad-contacts').textContent = p.contacts.length ? `${p.contacts.length} joint pair${p.contacts.length > 1 ? 's' : ''} touch` : 'none needed';
    $('ad-samples').textContent = p.n_candidates;
    if (!detail) { detail = new Avatar($('alpha-canvas')); await detail.load(); }
    detail.start(); detail.stop(false); detail.setView('close'); detail.showLetter(L);
    $('alpha-detail').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

export default {
    async enter() {
        if (!ready) {
            ready = true;
            const box = $('alpha'); box.innerHTML = '<p class="small">Rendering 26 poses…</p>';
            [thumbs, poses] = await Promise.all([makeThumbnails(LETTERS), fetch('/static/data/poses.json').then(r => r.json())]);
            $$('#alpha-filter button').forEach(b => b.onclick = () => { filter = b.dataset.filter; $$('#alpha-filter button').forEach(x => x.setAttribute('aria-pressed', x === b)); render(); });
            $('ad-close').onclick = () => { $('alpha-detail').hidden = true; selected = null; if (detail) detail.pause(); render(); };
            $('ad-play').onclick = () => { location.hash = `#/sign/${selected}`; };
            render();
        }
        if (detail && selected) detail.start();
    },
    leave() { if (detail) detail.pause(); },
};
