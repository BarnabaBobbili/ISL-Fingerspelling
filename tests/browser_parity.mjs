// Browser-tracker parity harness.
// Frame-steps WebM videos through the in-browser hand tracker (app/static/js/tracker.js) in a headless
// browser and writes the landmarks it produced, for src/evaluation/browser_parity.py to score.
//
//   npm i puppeteer-core            (once, anywhere on NODE_PATH or next to this file)
//   python app/server.py            (the page is served by the app)
//   node tests/browser_parity.mjs <dir containing uids.txt and <uid>.webm> video|image out.json
//
// BROWSER_PATH overrides the browser executable (default: Microsoft Edge on Windows, else google-chrome).
// MODE video = tracking between frames (fast, used live); image = full detection every frame (what training used).
import puppeteer from 'puppeteer-core';
import fs from 'fs';

const [DIR, MODE = 'video', OUT = 'browser_landmarks.json'] = process.argv.slice(2);
const exe = process.env.BROWSER_PATH || (process.platform === 'win32'
  ? 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe' : 'google-chrome');
const uids = fs.readFileSync(`${DIR}/uids.txt`, 'utf8').split(/\r?\n/).filter(Boolean).slice(0, +(process.env.N || 6));

const browser = await puppeteer.launch({ executablePath: exe, headless: 'new', args: ['--no-sandbox', '--autoplay-policy=no-user-gesture-required'] });
const page = await browser.newPage();
await page.goto(process.env.APP_URL || 'http://127.0.0.1:8000/', { waitUntil: 'load' });
const res = {};
for (const uid of uids) {
  const b64 = fs.readFileSync(`${DIR}/${uid}.webm`).toString('base64');
  res[uid] = await page.evaluate(async (b64, mode) => {
    const { BrowserTracker } = await import('/static/js/tracker.js');
    const tr = await new BrowserTracker({ mode, recover: false, delegate: 'CPU' }).init();
    const v = document.createElement('video'); v.muted = true;
    v.src = URL.createObjectURL(new Blob([Uint8Array.from(atob(b64), c => c.charCodeAt(0))], { type: 'video/webm' }));
    await new Promise((ok, no) => { v.onloadeddata = ok; v.onerror = () => no(new Error('video load failed')); });
    const fps = 29.97, n = Math.floor(v.duration * fps), frames = [], times = [];
    for (let i = 0; i < n; i++) {
      v.currentTime = (i + 0.5) / fps;
      await new Promise(r => { v.onseeked = r; });
      const o = tr.detect(v, i * (1000 / fps)); frames.push(o.hands); times.push(o.ms);
    }
    return { w: v.videoWidth, h: v.videoHeight, n, frames, times };
  }, b64, MODE);
  const t = res[uid].times; console.log(uid, `${res[uid].n} frames, mean ${(t.reduce((a, b) => a + b, 0) / t.length).toFixed(1)} ms/frame`);
}
fs.writeFileSync(OUT, JSON.stringify(res));
await browser.close();
