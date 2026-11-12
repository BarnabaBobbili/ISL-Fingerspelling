/* Sign: the 3D signer spells typed or dictated text. */
import { $, $$, speak, toast, icon } from '/static/js/ui.js';
import { Avatar } from '/static/js/avatar.js';

let av = null, wired = false, current = '', recorder = null;

const clean = t => t.toUpperCase().replace(/[^A-Z ]/g, '');

function renderChips() {
    const t = clean($('sg-text').value);
    $('sg-chips').innerHTML = [...t].map((c, i) => c === ' ' ? '<span class="chip sp" aria-hidden="true">·</span>' : `<button type="button" class="chip" data-i="${i}" data-c="${c}">${c}</button>`).join('');
    $$('#sg-chips .chip[data-c]').forEach(ch => ch.onclick = () => { av.showLetter(ch.dataset.c); setLetter(ch.dataset.c, t, +ch.dataset.i); });
}

function setLetter(ch, text, idx = -1) {
    current = ch; $('av-letter').textContent = ch || '';
    $('av-word').innerHTML = [...text].map((c, i) => `<span class="${i === idx ? 'on' : i < idx ? 'done' : ''}">${c === ' ' ? '&nbsp;' : c}</span>`).join('');
    $$('#sg-chips .chip').forEach(c => c.classList.toggle('on', +c.dataset.i === idx));
    const info = ch && av.info(ch);
    $('av-hands').textContent = info ? (info.hands === 2 ? 'two-handed' : 'one-handed') : '';
    $('av-contact').textContent = info && info.contacts.length && av.useTouched ? `${info.contacts.length} contact${info.contacts.length > 1 ? 's' : ''} aligned` : '';
}

function play(text) {
    const t = clean(text); if (!t.trim()) { toast('Type some letters first'); return; }
    speechSynthesis.cancel(); av.start();
    av.play(t, {
        speed: +$('sg-speed').value,
        onItem: it => { if (it.ch && it.ch !== ' ') { setLetter(it.ch, t, it.idx); if ($('sg-speak').checked) speak(it.ch, 1.1); } else if (it.end) setLetter('', t, t.length); },
        onDone: () => { if (recorder && recorder.state === 'recording') toggleRecord(); },
    });
}

function dictate() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { toast('Dictation needs Chrome or Edge'); return; }
    const r = new SR(); r.lang = 'en-IN'; r.interimResults = false;
    const b = $('sg-mic'); b.disabled = true; b.lastChild.textContent = 'Listening…';
    r.onresult = e => { const t = e.results[0][0].transcript; $('sg-text').value = t.toUpperCase(); renderChips(); play(t); };
    r.onend = r.onerror = () => { b.disabled = false; b.lastChild.textContent = 'Dictate'; };
    r.start();
}

function toggleRecord() {
    const b = $('sg-rec');
    if (recorder && recorder.state === 'recording') { recorder.stop(); return; }
    if (!window.MediaRecorder) { toast('Recording is not supported in this browser'); return; }
    const chunks = [];
    recorder = new MediaRecorder($('av-canvas').captureStream(30), { mimeType: MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm' });
    recorder.ondataavailable = e => e.data.size && chunks.push(e.data);
    recorder.onstop = () => {
        const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob(chunks, { type: 'video/webm' })); a.download = 'isl-fingerspelling-sign.webm'; a.click();
        b.classList.remove('rec'); b.innerHTML = icon('record') + 'Record a clip';
    };
    recorder.start(); b.classList.add('rec'); b.innerHTML = icon('stop') + 'Stop and save';
    play($('sg-text').value);
}

async function init() {
    av = new Avatar($('av-canvas'));
    await av.load();
    const apply = () => { av.useTouched = $('sg-touch').checked; av.showMarkers = $('sg-markers').checked; av.showSkeleton = $('sg-skel').checked; };
    ['sg-touch', 'sg-markers', 'sg-skel'].forEach(id => $(id).onchange = () => { apply(); if (current && !av.seq) av.showLetter(current); });
    $('sg-skin').onchange = e => av.setSkin(e.target.value);
    $('sg-text').oninput = renderChips;
    $('sg-play').onclick = () => play($('sg-text').value);
    $('sg-stop').onclick = () => { av.stop(); speechSynthesis.cancel(); setLetter('', ''); };
    $('sg-mic').onclick = dictate; $('sg-rec').onclick = toggleRecord;
    $$('[data-view-preset]').forEach(b => b.onclick = () => { av.setView(b.dataset.viewPreset); $$('[data-view-preset]').forEach(x => x.setAttribute('aria-pressed', x === b)); });
    apply(); renderChips();
}

export default {
    async enter(params) {
        if (!wired) { wired = true; await init(); }
        const q = new URLSearchParams(location.search).get('text') || (params[0] ? decodeURIComponent(params[0]) : '');
        if (q) { $('sg-text').value = q; renderChips(); play(q); history.replaceState(null, '', '#/sign'); }
        av.start();
    },
    leave() { if (av) { av.pause(); speechSynthesis.cancel(); } },
};
