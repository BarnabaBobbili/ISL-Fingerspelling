import { Live } from '/static/js/live.js';
import { Avatar, makeThumbnails } from '/static/js/avatar.js';
import { drawHud, drawSkeleton, containRect } from '/static/js/hud.js';

const $ = id => document.getElementById(id);
const LETTERS = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'];

/* ------------------------------------------------------------------ helpers */
function speak(text, rate = 0.9) {
    if (!('speechSynthesis' in window) || !text.trim()) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text.trim()); u.rate = rate; u.lang = 'en-IN';
    speechSynthesis.speak(u);
}
const copy = t => navigator.clipboard && navigator.clipboard.writeText(t);

/* ------------------------------------------------------------------- status */
async function pollStatus() {
    const s = $('status');
    try {
        const d = await (await fetch('/api/info')).json();
        s.className = 'status ' + (d.ready ? 'ok' : ''); s.querySelector('span').textContent = d.ready ? 'models ready' : 'loading models…';
        $('foot-cont').textContent = 'continuous model: ' + (d.continuous_model || (d.continuous_available ? 'available' : 'not installed'));
        if (!d.ready) setTimeout(pollStatus, 1500);
    } catch { s.className = 'status bad'; s.querySelector('span').textContent = 'server offline'; setTimeout(pollStatus, 3000); }
}

/* --------------------------------------------------------------------- tabs */
const inited = {};
function showTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
    document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
    if (name === 'sign') initSign().then(() => avatar.start()); else avatar && avatar.pause();
    if (name === 'alpha') initAlpha();
    if (name !== 'live' && live.running) stopLive();
}
document.querySelectorAll('.tab').forEach(t => t.onclick = () => showTab(t.dataset.tab));

/* --------------------------------------------------------------------- live */
const live = new Live({
    canvas: $('live-stage'), msg: $('live-msg'),
    onText: t => { $('live-text').textContent = t || ' '; },
    onMetrics: m => { $('m-fps').textContent = m.fps.toFixed(0); $('m-ms').textContent = m.ms ?? '–'; $('m-hands').textContent = m.hands; },
});
async function startLive() {
    $('live-start').disabled = true;
    try { await live.start($('live-mode').value); $('live-stop').disabled = false; }
    catch (e) { $('live-start').disabled = false; }
}
function stopLive() { live.stop(); $('live-start').disabled = false; $('live-stop').disabled = true; }
$('live-start').onclick = startLive; $('live-stop').onclick = stopLive;
$('live-mode').onchange = () => { if (live.running) { stopLive(); startLive(); } };
$('live-mirror').onchange = e => live.mirror = e.target.checked;
$('live-skel').onchange = e => live.skeleton = e.target.checked;
$('live-speak').onclick = () => speak($('live-text').textContent);
$('live-copy').onclick = () => copy($('live-text').textContent.trim());
$('live-space').onclick = () => live.cmd('space'); $('live-back').onclick = () => live.cmd('backspace');
$('live-clear').onclick = () => { live.cmd('reset'); $('live-text').textContent = ' '; };

/* ------------------------------------------------------------------- upload */
let up = null;
const drop = $('drop'), file = $('file');
drop.onclick = () => file.click();
['dragenter', 'dragover'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', ev => ev.dataTransfer.files[0] && handleFile(ev.dataTransfer.files[0]));
file.onchange = () => file.files[0] && handleFile(file.files[0]);

async function handleFile(f) {
    const prog = $('up-progress'); prog.classList.remove('hidden'); setProg(0, 'uploading…');
    $('up-text').textContent = ' '; $('up-meta').textContent = '';
    $('up-keyframes-card').classList.add('hidden'); $('up-dl').classList.add('hidden'); $('up-json').classList.add('hidden');
    const v = $('up-video'); v.src = URL.createObjectURL(f); $('up-stagewrap').classList.remove('hidden'); up = null;
    const fd = new FormData(); fd.append('file', f); fd.append('mode', $('up-mode').value);
    let job;
    try { const r = await fetch('/api/video', { method: 'POST', body: fd }); if (!r.ok) throw new Error((await r.json()).detail); job = (await r.json()).job; }
    catch (e) { setProg(0, 'error: ' + e.message); return; }
    const poll = async () => {
        const d = await (await fetch('/api/job/' + job)).json();
        if (d.status === 'error') return setProg(0, 'error: ' + d.error);
        if (d.status === 'done') { setProg(1, 'done'); return showResult(d.result, job); }
        setProg(d.progress || 0, d.status === 'queued' ? 'queued…' : `analysing ${Math.round((d.progress || 0) * 100)}%`);
        setTimeout(poll, 600);
    };
    poll();
}
function setProg(p, t) { $('up-bar').style.width = (p * 100) + '%'; $('up-ptext').textContent = t; }

function showResult(res, job) {
    up = res;
    $('up-text').textContent = res.text || '(no letters recognised)';
    $('up-meta').innerHTML = `<span>mode <b>${res.mode}</b></span><span>letters <b>${res.letters.length}</b></span><span>${res.duration}s · ${res.n_frames} frames</span><span>processed in <b>${res.elapsed}s</b></span>`;
    const strip = $('up-strip'); strip.innerHTML = '';
    res.letters.forEach(l => {
        const d = document.createElement('div'); d.className = 'kf';
        d.innerHTML = `<img src="${res.base}${l.thumb}" alt="${l.letter}"><div><b>${l.letter}</b><small>${(l.conf * 100).toFixed(0)}% · ${l.t.toFixed(1)}s</small></div>`;
        d.onclick = () => { const v = $('up-video'); v.currentTime = l.t; v.pause(); };
        strip.appendChild(d);
    });
    $('up-keyframes-card').classList.toggle('hidden', !res.letters.length);
    if (res.annotated) { const a = $('up-dl'); a.href = res.base + res.annotated; a.download = 'annotated_' + res.annotated; a.classList.remove('hidden'); }
    const j = $('up-json'); j.href = URL.createObjectURL(new Blob([JSON.stringify(res)], { type: 'application/json' })); j.classList.remove('hidden');
    overlayLoop();
}
let overlayOn = false;
function overlayLoop() {
    if (overlayOn) return; overlayOn = true;
    const v = $('up-video'), c = $('up-canvas'), ctx = c.getContext('2d');
    const tick = () => {
        if (!up) { overlayOn = false; return; }
        const dpr = window.devicePixelRatio || 1, W = c.clientWidth * dpr, H = c.clientHeight * dpr;
        if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
        ctx.clearRect(0, 0, W, H);
        const i = Math.min(up.frames.length - 1, Math.max(0, Math.floor(v.currentTime * up.fps)));
        const st = up.frames[i];
        if (st) {
            const r = containRect(v.videoWidth || up.width, v.videoHeight || up.height, W, H);
            drawSkeleton(ctx, st.hands, r, false);
            ctx.save(); ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip(); ctx.translate(r.x, r.y); drawHud(ctx, r.w, r.h, st); ctx.restore();
        }
        requestAnimationFrame(tick);
    };
    tick();
}
$('up-speak').onclick = () => speak($('up-text').textContent);
$('up-copy').onclick = () => copy($('up-text').textContent.trim());

/* --------------------------------------------------------------- text -> sign */
let avatar = null, recorder = null;
async function initSign() {
    if (inited.sign) return; inited.sign = true;
    avatar = new Avatar($('av-canvas'));
    await avatar.load();
    const upd = () => { avatar.useTouched = $('sg-touch').checked; avatar.showMarkers = $('sg-markers').checked; avatar.showSkeleton = $('sg-skel').checked; };
    ['sg-touch', 'sg-markers', 'sg-skel'].forEach(id => $(id).onchange = () => { upd(); const L = currentLetter; if (L && !avatar.seq) avatar.showLetter(L); });
    $('sg-skin').onchange = e => avatar.setSkin(e.target.value);
    $('sg-text').oninput = renderChips; renderChips();
    $('sg-play').onclick = () => playText($('sg-text').value);
    $('sg-stop').onclick = () => { avatar.stop(); speechSynthesis.cancel(); setLetter('', ''); };
    $('sg-mic').onclick = dictate; $('sg-rec').onclick = toggleRecord;
    upd();
    const sp = new URLSearchParams(location.hash.slice(1)).get('text'); if (sp) { $('sg-text').value = sp; renderChips(); }
}
let currentLetter = '';
function renderChips() {
    const t = $('sg-text').value.toUpperCase().replace(/[^A-Z ]/g, '');
    $('sg-chips').innerHTML = [...t].map((c, i) => c === ' ' ? '<span class="chip sp">·</span>' : `<span class="chip" data-i="${i}" data-c="${c}">${c}</span>`).join('');
    $('sg-chips').querySelectorAll('.chip[data-c]').forEach(ch => ch.onclick = () => { avatar.showLetter(ch.dataset.c); setLetter(ch.dataset.c, t, +ch.dataset.i); });
}
function setLetter(ch, text, idx = -1) {
    currentLetter = ch; $('av-letter').textContent = ch || ' ';
    $('av-word').innerHTML = [...text].map((c, i) => `<span class="${i === idx ? 'on' : i < idx ? 'done' : ''}">${c === ' ' ? '&nbsp;' : c}</span>`).join('');
    document.querySelectorAll('#sg-chips .chip').forEach(c => c.classList.toggle('on', +c.dataset.i === idx));
    const info = ch && avatar.info(ch);
    $('av-hands').textContent = info ? (info.hands === 2 ? 'two-handed sign' : 'one-handed sign') : '';
    $('av-contact').textContent = info && info.contacts.length && avatar.useTouched ? `${info.contacts.length} contact point${info.contacts.length > 1 ? 's' : ''} aligned` : '';
}
function playText(text) {
    const clean = text.toUpperCase().replace(/[^A-Z ]/g, ''); if (!clean.trim()) return;
    speechSynthesis.cancel();
    avatar.start();
    avatar.play(clean, {
        speed: +$('sg-speed').value,
        onItem: it => { if (it.ch && it.ch !== ' ') { setLetter(it.ch, clean, it.idx); if ($('sg-speak').checked) speak(it.ch, 1.1); } else if (it.end) setLetter('', clean, clean.length); },
        onDone: () => { if (recorder && recorder.state === 'recording') toggleRecord(); },
    });
}
function dictate() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { alert('Speech recognition needs Chrome or Edge.'); return; }
    const r = new SR(); r.lang = 'en-IN'; r.interimResults = false; $('sg-mic').textContent = '… listening';
    r.onresult = e => { const t = e.results[0][0].transcript; $('sg-text').value = t.toUpperCase(); renderChips(); playText(t); };
    r.onend = () => { $('sg-mic').innerHTML = '&#127908; Dictate'; };
    r.start();
}
function toggleRecord() {
    const b = $('sg-rec');
    if (recorder && recorder.state === 'recording') { recorder.stop(); return; }
    const stream = $('av-canvas').captureStream(30), chunks = [];
    recorder = new MediaRecorder(stream, { mimeType: MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm' });
    recorder.ondataavailable = e => e.data.size && chunks.push(e.data);
    recorder.onstop = () => {
        const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob(chunks, { type: 'video/webm' }));
        a.download = 'isl-sign.webm'; a.click(); b.classList.remove('rec'); b.innerHTML = '&#9210; Record clip';
    };
    recorder.start(); b.classList.add('rec'); b.innerHTML = '&#9632; Stop &amp; save';
    playText($('sg-text').value);
}

/* ----------------------------------------------------------------- alphabet */
async function initAlpha() {
    if (inited.alpha) return; inited.alpha = true;
    const box = $('alpha'); box.innerHTML = '<p class="hint">rendering 26 poses…</p>';
    const thumbs = await makeThumbnails(LETTERS);
    const poses = await (await fetch('/static/data/poses.json')).json();
    box.innerHTML = '';
    LETTERS.forEach(L => {
        const p = poses.letters[L], c = document.createElement('div'); c.className = 'acard';
        c.innerHTML = `<img src="${thumbs[L]}" alt="${L}"><div><b>${L}</b><small>${p.hands === 2 ? '2 hands' : '1 hand'}${p.contacts.length ? ' · ' + p.contacts.length + ' contacts' : ''}</small></div>`;
        c.onclick = async () => { showTab('sign'); await initSign(); $('sg-text').value = L; renderChips(); playText(L); };
        box.appendChild(c);
    });
}

pollStatus();
