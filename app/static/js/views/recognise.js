/* Recognise: live camera (browser tracking) and video-file analysis. */
import { $, speak, copyText, toast } from '/static/js/ui.js';
import { Live } from '/static/js/live.js';
import { drawHud, drawSkeleton, containRect } from '/static/js/hud.js';

let live = null, wired = false, up = null, overlayOn = false;

const readout = (el, text, tent = '', empty = 'Nothing yet') => {
    el.classList.toggle('empty', !text && !tent);
    el.innerHTML = (text || tent) ? `${text}<span class="tent">${tent}</span>` : empty;
};

function tab(which) {
    const liveTab = which !== 'upload';
    $('tab-live').setAttribute('aria-selected', liveTab); $('tab-upload').setAttribute('aria-selected', !liveTab);
    $('pane-live').hidden = !liveTab; $('pane-upload').hidden = liveTab;
    if (!liveTab && live && live.running) stopLive();
    history.replaceState(null, '', liveTab ? '#/recognise' : '#/recognise/upload');
}

/* ------------------------------------------------------------------ live */
function hintFor(mode) {
    $('live-hint').textContent = mode === 'continuous'
        ? 'Spell at a natural pace and lower the hands out of view briefly between words. Letters in grey are still tentative.'
        : 'Hold each letter still for about a third of a second. To repeat a letter, lower the hands slightly in between.';
}
async function startLive() {
    $('live-start').disabled = true;
    try { await live.start($('live-mode').value); $('live-stop').disabled = false; }
    catch { $('live-start').disabled = false; }
}
function stopLive() { live.stop(); $('live-start').disabled = false; $('live-stop').disabled = true; }

/* ---------------------------------------------------------------- upload */
const setProg = (p, t) => { $('up-bar').style.width = (p * 100) + '%'; $('up-ptext').textContent = t; };

async function handleFile(f) {
    $('up-progress').hidden = false; setProg(0, 'uploading…');
    readout($('up-text'), '', '', 'Analysing…'); $('up-meta').textContent = '';
    $('up-keyframes-card').hidden = true; $('up-dl').hidden = true; $('up-json').hidden = true;
    const v = $('up-video'); v.src = URL.createObjectURL(f); $('up-stagewrap').hidden = false; up = null;
    const fd = new FormData(); fd.append('file', f); fd.append('mode', $('up-mode').value);
    let job;
    try { const r = await fetch('/api/video', { method: 'POST', body: fd }); if (!r.ok) throw new Error((await r.json()).detail); job = (await r.json()).job; }
    catch (e) { setProg(0, 'Could not upload: ' + e.message); return; }
    const poll = async () => {
        const d = await (await fetch('/api/job/' + job)).json();
        if (d.status === 'error') return setProg(0, 'Analysis failed: ' + d.error);
        if (d.status === 'done') { setProg(1, 'done'); return showResult(d.result); }
        setProg(d.progress || 0, d.status === 'queued' ? 'waiting in queue…' : `analysing ${Math.round((d.progress || 0) * 100)} %`);
        setTimeout(poll, 600);
    };
    poll();
}

function showResult(res) {
    up = res;
    readout($('up-text'), res.text, '', 'No letters found');
    $('up-meta').innerHTML = `<span>${res.mode === 'continuous' ? 'flowing' : 'held letters'}</span><span>letters <b>${res.letters.length}</b></span><span>${res.duration}s</span><span>took <b>${res.elapsed}s</b></span>`;
    const strip = $('up-strip'); strip.innerHTML = '';
    res.letters.forEach(l => {
        const b = document.createElement('button'); b.className = 'kf'; b.type = 'button';
        b.innerHTML = `<img src="${res.base}${l.thumb}" alt="Key frame for ${l.letter}"><div><b>${l.letter}</b><small>${(l.conf * 100).toFixed(0)}% · ${l.t.toFixed(1)}s</small></div>`;
        b.onclick = () => { const v = $('up-video'); v.currentTime = l.t; v.pause(); };
        strip.appendChild(b);
    });
    $('up-keyframes-card').hidden = !res.letters.length;
    if (res.annotated) { const a = $('up-dl'); a.href = res.base + res.annotated; a.download = 'annotated_' + res.annotated; a.hidden = false; }
    const j = $('up-json'); j.href = URL.createObjectURL(new Blob([JSON.stringify(res)], { type: 'application/json' })); j.hidden = false;
    overlayLoop();
}

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

/* ------------------------------------------------------------------ wiring */
function wire() {
    wired = true;
    live = new Live({
        canvas: $('live-stage'), msg: $('live-msg'),
        onText: (t, tent) => readout($('live-text'), t, tent),
        onMetrics: m => {
            $('m-fps').textContent = m.fps.toFixed(0); $('m-ms').textContent = m.ms != null ? m.ms.toFixed(0) : '–';
            $('m-hands').textContent = m.hands; $('m-path').textContent = m.path === 'browser' ? 'tracking in browser' : 'tracking on server';
        },
    });
    $('tab-live').onclick = () => tab('live'); $('tab-upload').onclick = () => tab('upload');
    $('live-start').onclick = startLive; $('live-stop').onclick = stopLive;
    $('live-mode').onchange = () => { hintFor($('live-mode').value); if (live.running) { stopLive(); startLive(); } };
    $('live-mirror').onchange = e => live.mirror = e.target.checked;
    $('live-skel').onchange = e => live.skeleton = e.target.checked;
    $('live-speak').onclick = () => speak(($('live-text').textContent || '').trim());
    $('live-copy').onclick = () => copyText(($('live-text').textContent || '').trim());
    $('live-space').onclick = () => live.cmd('space'); $('live-back').onclick = () => live.cmd('backspace');
    $('live-clear').onclick = () => { live.cmd('reset'); readout($('live-text'), ''); };
    const drop = $('drop'), file = $('file');
    drop.onclick = () => file.click();
    drop.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); file.click(); } };
    ['dragenter', 'dragover'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('over'); }));
    drop.addEventListener('drop', ev => ev.dataTransfer.files[0] && handleFile(ev.dataTransfer.files[0]));
    file.onchange = () => file.files[0] && handleFile(file.files[0]);
    $('up-speak').onclick = () => speak(($('up-text').textContent || '').trim());
    $('up-copy').onclick = () => copyText(($('up-text').textContent || '').trim());
    hintFor($('live-mode').value);
}

export default {
    enter(params) {
        if (!wired) wire();
        tab(params[0] === 'upload' ? 'upload' : 'live');
    },
    leave() { if (live && live.running) stopLive(); },
};
