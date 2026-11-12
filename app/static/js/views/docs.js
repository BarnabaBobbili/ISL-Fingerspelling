/* Docs: the Docusaurus manual embedded under the site header; location and theme stay in sync. */
import { $, syncDocsTheme } from '/static/js/ui.js';

let timer = null;

export default {
    enter(params) {
        const f = $('docs-frame');
        const want = '/docs/' + params.join('/');
        const have = f.contentWindow && f.contentWindow.location ? f.contentWindow.location.pathname : '';
        if (!f.getAttribute('src')) { f.src = want; f.onload = () => { syncDocsTheme(f); }; }
        else if (params.length && have && !have.startsWith(want.replace(/\/$/, ''))) f.contentWindow.location.replace(want);
        clearInterval(timer);
        timer = setInterval(() => {                         // mirror in-manual navigation into the address bar
            try {
                const p = f.contentWindow.location.pathname.replace(/^\/docs\/?/, '');
                const h = '#/docs' + (p ? '/' + p.replace(/\/$/, '') : '');
                if (location.hash !== h && p !== undefined) history.replaceState(null, '', h);
            } catch { /* cross-origin while loading */ }
        }, 600);
    },
    leave() { clearInterval(timer); },
};
