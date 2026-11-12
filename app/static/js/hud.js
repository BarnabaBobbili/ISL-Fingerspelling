/* Recognition HUD drawn ON TOP of video frames (live canvas and uploaded video overlay). */

const COL = { no_hand: '#94a3b8', moving: '#fbbf24', holding: '#5eead4', committed: '#4ade80' };
const confColor = c => (c < 0.5 ? '#f87171' : c < 0.8 ? '#fbbf24' : '#4ade80');
const CONN = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

export function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

/** rect = {x,y,w,h} where the video picture sits inside the canvas; mirror flips x. */
export function drawSkeleton(ctx, hands, rect, mirror) {
    if (!hands) return;
    const pt = p => [rect.x + (mirror ? 1 - p[0] : p[0]) * rect.w, rect.y + p[1] * rect.h];
    ctx.lineCap = 'round';
    for (const h of hands) {
        for (const [a, b] of CONN) { const A = pt(h[a]), B = pt(h[b]);
            ctx.strokeStyle = 'rgba(8,10,20,.85)'; ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(...A); ctx.lineTo(...B); ctx.stroke();
            ctx.strokeStyle = '#5eead4'; ctx.lineWidth = 2.4; ctx.beginPath(); ctx.moveTo(...A); ctx.lineTo(...B); ctx.stroke(); }
        h.forEach((p, i) => { const [x, y] = pt(p); const tip = [4, 8, 12, 16, 20].includes(i);
            ctx.fillStyle = 'rgba(8,10,20,.9)'; ctx.beginPath(); ctx.arc(x, y, tip ? 5.5 : 4, 0, 7); ctx.fill();
            ctx.fillStyle = tip ? '#fbbf24' : '#fff'; ctx.beginPath(); ctx.arc(x, y, tip ? 3.6 : 2.6, 0, 7); ctx.fill(); });
    }
}

export function drawHud(ctx, W, H, st) {
    const s = Math.max(0.55, Math.min(W / 960, H / 540));
    const status = st.status || 'no_hand', conf = st.conf || 0, letter = st.letter;
    // letter card
    const cx = 16 * s, cy = 16 * s, cw = 218 * s, ch = 200 * s;
    ctx.save();
    ctx.fillStyle = 'rgba(10,14,30,.72)'; roundRect(ctx, cx, cy, cw, ch, 16 * s); ctx.fill();
    ctx.fillStyle = COL[status] || '#94a3b8'; roundRect(ctx, cx, cy, 7 * s, ch, 4 * s); ctx.fill();
    if (letter && status !== 'no_hand') {
        ctx.fillStyle = '#fff'; ctx.font = `800 ${118 * s}px "Segoe UI",system-ui,sans-serif`; ctx.textBaseline = 'alphabetic';
        ctx.fillText(letter, cx + 34 * s, cy + 112 * s);
        const bx = cx + 30 * s, by = cy + 132 * s, bw = 170 * s, bh = 11 * s;
        ctx.fillStyle = 'rgba(255,255,255,.12)'; roundRect(ctx, bx, by, bw, bh, bh / 2); ctx.fill();
        ctx.fillStyle = confColor(conf); roundRect(ctx, bx, by, Math.max(bh, bw * conf), bh, bh / 2); ctx.fill();
        ctx.fillStyle = '#e8ecff'; ctx.font = `600 ${15 * s}px "Segoe UI",system-ui`; ctx.fillText(`${(conf * 100).toFixed(0)}% confidence`, bx, by + 30 * s);
    } else {
        ctx.fillStyle = '#93a0cf'; ctx.font = `600 ${20 * s}px "Segoe UI",system-ui`; ctx.fillText('show a sign', cx + 30 * s, cy + 96 * s);
    }
    ctx.fillStyle = COL[status] || '#94a3b8'; ctx.font = `700 ${12 * s}px "Segoe UI",system-ui`;
    ctx.fillText((status === 'committed' ? 'LOCKED' : status).toUpperCase().replace('_', ' '), cx + 30 * s, cy + ch - 12 * s);
    // top-3 chips
    (st.top3 || []).slice(0, 3).forEach(([l, p], i) => {
        const x = cx + cw + 10 * s, y = cy + i * 40 * s;
        ctx.fillStyle = 'rgba(10,14,30,.72)'; roundRect(ctx, x, y, 138 * s, 33 * s, 10 * s); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.font = `700 ${19 * s}px "Cascadia Mono",Consolas,monospace`; ctx.fillText(l, x + 12 * s, y + 24 * s);
        ctx.fillStyle = confColor(p); ctx.font = `600 ${15 * s}px "Segoe UI",system-ui`; ctx.fillText(`${(p * 100).toFixed(0)}%`, x + 56 * s, y + 23 * s);
        ctx.fillStyle = 'rgba(255,255,255,.12)'; ctx.fillRect(x + 100 * s, y + 13 * s, 30 * s, 6 * s);
        ctx.fillStyle = confColor(p); ctx.fillRect(x + 100 * s, y + 13 * s, 30 * s * p, 6 * s);
    });
    // text banner
    const bh = 78 * s;
    ctx.fillStyle = 'rgba(10,14,30,.8)'; ctx.fillRect(0, H - bh, W, bh);
    ctx.fillStyle = '#93a0cf'; ctx.font = `700 ${12 * s}px "Segoe UI",system-ui`; ctx.fillText('TEXT', 20 * s, H - bh + 20 * s);
    const text = st.text || '';
    ctx.fillStyle = '#5eead4'; ctx.font = `700 ${44 * s}px "Cascadia Mono",Consolas,monospace`;
    let shown = text.slice(-Math.floor(W / (27 * s)));
    ctx.fillText(shown || '…', 20 * s, H - 20 * s);
    if (Math.floor(performance.now() / 500) % 2 === 0) { const w = ctx.measureText(shown).width; ctx.fillRect(20 * s + w + 4 * s, H - 56 * s, 4 * s, 40 * s); }
    ctx.restore();
}

/** compute where a (vw x vh) picture lands inside a (W x H) canvas with 'contain' fitting */
export function containRect(vw, vh, W, H) {
    const r = Math.min(W / vw, H / vh); const w = vw * r, h = vh * r;
    return { x: (W - w) / 2, y: (H - h) / 2, w, h };
}
