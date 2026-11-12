/* Recognition HUD drawn ON TOP of the picture (live canvas and uploaded-video overlay).
 * Letter tile + confidence meter, top-3 alternatives, and the text so far (tentative tail dimmed). */

const INK = '#f3ecdc';
const PANEL = 'rgba(14, 20, 23, 0.80)';
const PEACOCK = '#62beb5';
const OK = '#67c186', WARN = '#e0a23c', BAD = '#ee7b6e', MUTE = 'rgba(243, 236, 220, 0.58)';
const STATUS = { no_hand: ['no hand', MUTE], moving: ['moving', WARN], holding: ['holding', PEACOCK], committed: ['typed', OK], reading: ['reading', PEACOCK] };
const confColor = c => (c < 0.5 ? BAD : c < 0.8 ? WARN : OK);
const DISPLAY = '"Fraunces", Georgia, serif', MONO = '"IBM Plex Mono", Consolas, monospace', UI = '"Instrument Sans", system-ui, sans-serif';
const CONN = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

export function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

/** rect = {x,y,w,h}: where the video picture sits inside the canvas. mirror flips x. */
export function drawSkeleton(ctx, hands, rect, mirror) {
    if (!hands) return;
    const pt = p => [rect.x + (mirror ? 1 - p[0] : p[0]) * rect.w, rect.y + p[1] * rect.h];
    const u = Math.max(1, rect.w / 640);
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    for (const h of hands) {
        ctx.strokeStyle = 'rgba(8,12,14,.8)'; ctx.lineWidth = 4.5 * u; ctx.beginPath();
        for (const [a, b] of CONN) { const A = pt(h[a]), B = pt(h[b]); ctx.moveTo(...A); ctx.lineTo(...B); }
        ctx.stroke();
        ctx.strokeStyle = PEACOCK; ctx.lineWidth = 2.2 * u; ctx.beginPath();
        for (const [a, b] of CONN) { const A = pt(h[a]), B = pt(h[b]); ctx.moveTo(...A); ctx.lineTo(...B); }
        ctx.stroke();
        h.forEach((p, i) => {
            const [x, y] = pt(p), tip = i === 4 || i === 8 || i === 12 || i === 16 || i === 20;
            ctx.fillStyle = 'rgba(8,12,14,.9)'; ctx.beginPath(); ctx.arc(x, y, (tip ? 5 : 3.6) * u, 0, 7); ctx.fill();
            ctx.fillStyle = tip ? '#efb03a' : INK; ctx.beginPath(); ctx.arc(x, y, (tip ? 3.5 : 2.3) * u, 0, 7); ctx.fill();
        });
    }
}

export function drawHud(ctx, W, H, st) {
    const s = Math.max(0.6, Math.min(W / 960, H / 540));
    const status = st.status || 'no_hand', conf = st.conf || 0, letter = st.letter;
    const [label, lcol] = STATUS[status] || STATUS.no_hand;
    ctx.save();
    ctx.textBaseline = 'alphabetic';

    // --- letter tile ---
    const x0 = 18 * s, y0 = 18 * s, tw = 158 * s, th = 178 * s;
    ctx.fillStyle = PANEL; roundRect(ctx, x0, y0, tw, th, 12 * s); ctx.fill();
    ctx.fillStyle = lcol; ctx.font = `500 ${11.5 * s}px ${MONO}`;
    ctx.fillText(label.toUpperCase(), x0 + 16 * s, y0 + 26 * s);
    const has = letter && status !== 'no_hand';
    if (has) {
        ctx.fillStyle = INK; ctx.font = `450 ${104 * s}px ${DISPLAY}`; ctx.textAlign = 'center';
        ctx.fillText(letter, x0 + tw / 2, y0 + 126 * s); ctx.textAlign = 'left';
        const bx = x0 + 16 * s, by = y0 + th - 30 * s, bw = tw - 32 * s, bh = 5 * s;
        ctx.fillStyle = 'rgba(243,236,220,.16)'; roundRect(ctx, bx, by, bw, bh, bh / 2); ctx.fill();
        ctx.fillStyle = confColor(conf); roundRect(ctx, bx, by, Math.max(bh, bw * conf), bh, bh / 2); ctx.fill();
        ctx.fillStyle = INK; ctx.font = `500 ${13 * s}px ${MONO}`; ctx.fillText(`${(conf * 100).toFixed(0)}%`, bx, by + 20 * s);
    } else {
        ctx.fillStyle = MUTE; ctx.font = `400 ${15 * s}px ${UI}`; ctx.fillText('show a sign', x0 + 16 * s, y0 + 100 * s);
    }

    // --- alternatives ---
    const alts = (st.top3 || []).slice(0, 3);
    alts.forEach(([l, p], i) => {
        const x = x0 + tw + 10 * s, y = y0 + i * 34 * s;
        ctx.fillStyle = PANEL; roundRect(ctx, x, y, 112 * s, 28 * s, 8 * s); ctx.fill();
        ctx.fillStyle = INK; ctx.font = `500 ${15 * s}px ${MONO}`; ctx.fillText(l, x + 11 * s, y + 19.5 * s);
        ctx.fillStyle = confColor(p); ctx.fillRect(x + 36 * s, y + 12 * s, 30 * s * Math.max(0.03, p), 4 * s);
        ctx.fillStyle = MUTE; ctx.font = `400 ${11.5 * s}px ${MONO}`; ctx.fillText(`${(p * 100).toFixed(0)}%`, x + 72 * s, y + 18.5 * s);
    });

    // --- text strip ---
    const bh = 66 * s;
    ctx.fillStyle = PANEL; ctx.fillRect(0, H - bh, W, bh);
    ctx.fillStyle = MUTE; ctx.font = `500 ${10.5 * s}px ${MONO}`; ctx.fillText('TEXT', 20 * s, H - bh + 18 * s);
    const text = st.text || '', tent = st.tentative || '';
    ctx.font = `500 ${34 * s}px ${MONO}`;
    const cw = ctx.measureText('M').width, maxCh = Math.max(4, Math.floor((W - 40 * s) / cw));
    const cut = Math.max(0, text.length + tent.length - maxCh);
    const a = text.slice(cut), b = tent.slice(Math.max(0, cut - text.length));
    ctx.fillStyle = INK; ctx.fillText(a || (b ? '' : '…'), 20 * s, H - 16 * s);
    const wa = ctx.measureText(a).width;
    ctx.fillStyle = MUTE; ctx.fillText(b, 20 * s + wa, H - 16 * s);
    if (Math.floor(performance.now() / 520) % 2 === 0) { ctx.fillStyle = PEACOCK; ctx.fillRect(20 * s + wa + ctx.measureText(b).width + 3 * s, H - 16 * s - 28 * s, 3 * s, 32 * s); }
    ctx.restore();
}

/** where a (vw x vh) picture lands inside a (W x H) canvas with 'contain' fitting */
export function containRect(vw, vh, W, H) {
    const r = Math.min(W / vw, H / vh); const w = vw * r, h = vh * r;
    return { x: (W - w) / 2, y: (H - h) / 2, w, h };
}
