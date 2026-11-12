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

    h += `<section class="result-block"><h2>Isolated letter recognition</h2>
      <p class="lede">The proposed model is evaluated under three protocols. A random split of the training images gives ${pct(best.in_domain, 0)}. Held-out frames from real continuous-fingerspelling videos give ${pct(best.real)}. A dataset recorded with a different camera and never used in training gives ${pct(cross.acc)}. Accuracy on a new camera is expected to lie between the last two figures.</p>
      <div class="cols-2"><div>
        <h3 style="margin-bottom:var(--s-2)">${esc(best.name)}</h3>
        ${bars([{ label: 'Random split (near-duplicate images)', value: best.in_domain, cls: 'muted' }, { label: 'Held-out real-video frames', value: best.real }, { label: 'Unseen dataset (different camera)', value: cross.acc, cls: 'warn' }])}
        <p class="hint">Confidence is calibrated: the expected calibration error on the real-video frames is ${best.ece_cal.toFixed(3)}, against ${s.rows[0].ece.toFixed(2)} for the baseline.</p>
      </div><div>
        ${table(['Model', 'Random split', 'Real-video frames', 'ECE'], s.rows.map(r => [esc(r.name), pct(r.in_domain, 2), pct(r.real, 2), (r.ece_cal ?? r.ece).toFixed(3)]), { hl: s.best })}
      </div></div></section>`;

    h += `<section class="result-block"><h2>Generalisation to an unseen dataset</h2>
      <p class="lede">The model is trained without Static Dataset B and tested on it, which approximates use with a new camera. Most of the loss is caused by MediaPipe detecting only one of two overlapping hands; second-hand recovery was introduced to address this.</p>
      ${table(['Model (trained without Static Dataset B)', 'Accuracy', 'Both hands found', 'One hand found', 'ECE'], d.cross.map(r => [esc(r.name), `<b>${pct(r.acc)}</b>`, pct(r.both), pct(r.one), r.ece.toFixed(3)]), { hl: d.cross.length - 1 })}</section>`;

    const c = d.continuous, cb = c.rows[c.best];
    h += `<section class="result-block"><h2>Continuous name recognition</h2>
      <p class="lede">Character error rate (lower is better) on held-out test videos of the Continuous Fingerspelling Dataset. The Conformer outperforms the BiLSTM baseline only when it uses the pretrained hand-shape features.</p>
      <div class="cols-2"><div>${bars(c.rows.map((r, k) => ({ label: r.name, value: r.test, cls: k === c.best ? '' : 'muted' })), { max: 25, fmt: v => pct(v, 2) })}</div>
      <div>${table(['Model', 'Parameters', 'Validation', 'Test'], c.rows.map(r => [esc(r.name), r.params, pct(r.val, 2), pct(r.test, 2)]), { hl: c.best })}</div></div></section>`;

    if (d.streaming && d.streaming.rows.length) {
        const st = d.streaming;
        h += `<section class="result-block"><h2>Live decoding</h2>
          <p class="lede">The live recogniser decodes a sliding window and finalises a letter once it lies sufficiently far in the past. Held-out videos were replayed through the same code at faster signing speeds and on slower cameras.</p>
          ${table(st.rows.some(r => r.before != null) ? ['Condition', 'Without streaming training (live)', 'Whole clip', 'Live decoder', 'Letter delay (median)'] : ['Condition', 'Whole clip', 'Live decoder', 'Letter delay (median)', '90th percentile'], st.rows.map(r => st.rows.some(x => x.before != null) ? [esc(r.condition), pct(r.before, 2), pct(r.whole, 2), `<b>${pct(r.stream, 2)}</b>`, `${r.lat_median.toFixed(2)} s`] : [esc(r.condition), pct(r.whole, 2), `<b>${pct(r.stream, 2)}</b>`, `${r.lat_median.toFixed(2)} s`, `${r.lat_p90.toFixed(2)} s`]))}
          <p class="hint">${esc(st.note || '')}</p></section>`;
    }

    h += `<section class="result-block"><h2>Contact alignment</h2>
      <p class="lede">Two-handed letters are defined by contact. Raw landmarks leave gaps or interpenetrate; a rigid alignment followed by finger inverse kinematics closes them. Distances are in hand sizes (one hand size is about 9.5 cm).</p>
      <div class="cols-2"><div>${bars([{ label: 'Landmarks as detected', value: d.avatar.raw, cls: 'muted' }, { label: 'After alignment', value: d.avatar.touched }], { max: d.avatar.raw * 1.1, fmt: v => v.toFixed(3) })}
      <p class="hint">Mean gap over the ${d.avatar.letters} letters that have contact points.</p></div></div></section>`;

    h += `<section class="result-block"><h2>Limitations</h2>
      <ul style="max-width:68ch;color:var(--ink-2);padding-left:1.2em"><li>The documentation of the continuous dataset does not state the number of signers, so scores on held-out videos measure unseen clips and not necessarily unseen signers.</li>
      <li>On an unseen camera, accuracy is noticeably lower and the confidence is over-estimated; fine-tuning on recordings from the target camera is expected to be the most effective remedy.</li>
      <li>Letters Q and R are frequently detected with one hand only; J, Q and S have no reliable contact points for the 3D hands.</li>
      <li>The continuous model has no space symbol; words are separated by a short pause.</li></ul></section>`;
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
