/* Results: renders app/static/data/results.json (written by src/evaluation/make_site_data.py). */
import { $ } from '/static/js/ui.js';

let done = false;
const pct = (x, d = 1) => `${x.toFixed(d)}%`;
const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

function bars(rows, { max = 100, fmt = pct } = {}) {
    return `<div class="bars">${rows.map(r => `<div class="bar-row"><span>${esc(r.label)}</span><div class="track"><div class="fill ${r.cls || ''}" style="width:${Math.max(1, Math.min(100, r.value / max * 100))}%"></div></div><span class="val">${fmt(r.value)}</span></div>`).join('')}</div>`;
}
function table(head, rows, { hl = -1 } = {}) {
    return `<div class="table-wrap"><table class="data"><thead><tr>${head.map((h, i) => `<th class="${i ? 'num' : ''}">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map((r, k) => `<tr class="${k === hl ? 'hl' : ''}">${r.map((c, i) => `<td class="${i ? 'num' : ''}">${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function render(d) {
    const s = d.static, best = s.rows[s.best];
    const cross = d.cross[d.cross.length - 1];
    let h = '';

    h += `<section class="result-block"><h2>Reading single letters</h2>
      <p class="lede">The same model, scored three ways. A random split of the training images says 98%. Held-out news video — small hands, a full-body shot, a different studio — says ${pct(best.real)}. A camera the model has never seen says ${pct(cross.acc)}. The honest figure for <em>your</em> webcam lies somewhere between the last two.</p>
      <div class="cols-2"><div>
        <h3 style="margin-bottom:var(--s-2)">${esc(best.name)}</h3>
        ${bars([{ label: 'Random split (near-duplicate frames)', value: best.in_domain, cls: 'muted' }, { label: 'Held-out news video', value: best.real }, { label: 'Unseen camera and signer', value: cross.acc, cls: 'warn' }])}
        <p class="hint">Confidence is calibrated: expected calibration error on news video is ${best.ece_cal.toFixed(3)} (${s.rows[0].ece.toFixed(2)} for the first version).</p>
      </div><div>
        ${table(['Model', 'Random split', 'News video', 'ECE'], s.rows.map(r => [esc(r.name), pct(r.in_domain, 2), pct(r.real, 2), (r.ece_cal ?? r.ece).toFixed(3)]), { hl: s.best })}
      </div></div></section>`;

    h += `<section class="result-block"><h2>Why the unseen camera is harder</h2>
      <p class="lede">Training without the ISL_DATA collection and testing on it gives a realistic picture of a new setting. Most of the loss comes from MediaPipe finding only one of two overlapping hands, which is why second-hand recovery was added.</p>
      ${table(['Model (trained without ISL_DATA)', 'Accuracy', 'Both hands found', 'One hand found', 'ECE'], d.cross.map(r => [esc(r.name), `<b>${pct(r.acc)}</b>`, pct(r.both), pct(r.one), r.ece.toFixed(3)]), { hl: d.cross.length - 1 })}</section>`;

    const c = d.continuous, cb = c.rows[c.best];
    h += `<section class="result-block"><h2>Spelling whole names</h2>
      <p class="lede">Flowing fingerspelling from the news-interpreter dataset, scored as character error rate on videos the model never saw. Lower is better. The sequence model only beats the older recurrent baseline when it starts from the pretrained hand-shape features.</p>
      <div class="cols-2"><div>${bars(c.rows.map((r, k) => ({ label: r.name, value: r.test, cls: k === c.best ? '' : 'muted' })), { max: 25, fmt: v => pct(v, 2) })}</div>
      <div>${table(['Model', 'Parameters', 'Validation', 'Test'], c.rows.map(r => [esc(r.name), r.params, pct(r.val, 2), pct(r.test, 2)]), { hl: c.best })}</div></div></section>`;

    if (d.streaming && d.streaming.rows.length) {
        const st = d.streaming;
        h += `<section class="result-block"><h2>Live, while frames arrive</h2>
          <p class="lede">The live recogniser re-reads a sliding window and commits a letter once it is safely in the past. These runs replay held-out videos through exactly that code, at faster signing speeds and on slower cameras.</p>
          ${table(st.rows.some(r => r.before != null) ? ['Condition', 'Previous model (live)', 'Whole clip', 'Live decoder', 'Letter delay (median)'] : ['Condition', 'Whole clip', 'Live decoder', 'Letter delay (median)', '90th percentile'], st.rows.map(r => st.rows.some(x => x.before != null) ? [esc(r.condition), pct(r.before, 2), pct(r.whole, 2), `<b>${pct(r.stream, 2)}</b>`, `${r.lat_median.toFixed(2)} s`] : [esc(r.condition), pct(r.whole, 2), `<b>${pct(r.stream, 2)}</b>`, `${r.lat_median.toFixed(2)} s`, `${r.lat_p90.toFixed(2)} s`]))}
          <p class="hint">${esc(st.note || '')}</p></section>`;
    }

    h += `<section class="result-block"><h2>Making joints touch</h2>
      <p class="lede">Two-handed letters are defined by contact. Raw landmarks leave gaps or poke through; a rigid alignment plus finger inverse kinematics closes them. Distances are in hand sizes (one is about 9.5 cm).</p>
      <div class="cols-2"><div>${bars([{ label: 'Landmarks as detected', value: d.avatar.raw, cls: 'muted' }, { label: 'After alignment', value: d.avatar.touched }], { max: d.avatar.raw * 1.1, fmt: v => v.toFixed(3) })}
      <p class="hint">Mean gap over the ${d.avatar.letters} letters that have contacts.</p></div></div></section>`;

    h += `<section class="result-block"><h2>Where it falls short</h2>
      <ul style="max-width:68ch;color:var(--ink-2);padding-left:1.2em"><li>The same few interpreters appear across the news videos, so those scores measure new clips, not new people.</li>
      <li>On a camera it has not seen, expect noticeably lower accuracy and over-confident scores; fine-tuning on your own recordings is the best remedy.</li>
      <li>Q and R still lose a hand in most frames; J, Q and S have no reliable contact point for the signer.</li>
      <li>The continuous model has no space symbol, so words are separated by a short pause.</li></ul></section>`;
    $('results-body').innerHTML = h;
}

export default {
    async enter() {
        if (done) return; done = true;
        try { render(await (await fetch('/static/data/results.json')).json()); }
        catch (e) { $('results-body').innerHTML = '<p class="small">Results are not available yet. Run <code>python src/evaluation/make_site_data.py</code>.</p>'; console.error(e); }
    },
    leave() {},
};
