// Cloudflare Pages "advanced mode" worker.
// Serves the static site, and forwards the API calls the page makes to the backend (BACKEND_URL), so the browser only ever
// talks to this domain (no CORS) and the backend address stays out of the page. Operational endpoints are never forwarded.
//
// Settings (Pages project > Settings > Variables, or `wrangler pages secret put BACKEND_URL`):
//   BACKEND_URL   e.g. https://cosmetic-api.onrender.com   (no trailing path)

const FORWARD = [/^\/api\/v1\//, /^\/health$/, /^\/model-info$/, /^\/skincare\/products$/];
const NEVER = [/^\/api\/v1\/ops(\/|$)/];            // stats and drift are for operators only
const METHODS = new Set(['GET', 'POST']);
const MAX_BODY = 6 * 1024 * 1024;                  // the backend accepts photos up to 5 MB
const TIMEOUT_MS = 60000;                          // a sleeping free-tier backend needs up to about a minute to wake
const DROP_REQUEST = ['host', 'cookie', 'authorization', 'x-forwarded-for', 'x-real-ip', 'cf-connecting-ip'];

const json = (body, status) => new Response(JSON.stringify(body), {status, headers: {'content-type': 'application/json', 'cache-control': 'no-store'}});

export default {
  async fetch(request, env) {
    const url = new URL(request.url), path = url.pathname;
    if (!FORWARD.some(r => r.test(path))) return env.ASSETS.fetch(request);         // the static site
    if (NEVER.some(r => r.test(path))) return json({detail: 'Not found'}, 404);
    if (!METHODS.has(request.method)) return json({detail: 'Method not allowed'}, 405);
    if (!env.BACKEND_URL) return json({detail: 'The backend is not configured. Set BACKEND_URL for this Pages project.'}, 503);
    if (Number(request.headers.get('content-length') || 0) > MAX_BODY) return json({detail: 'Request too large'}, 413);

    const headers = new Headers(request.headers);
    DROP_REQUEST.forEach(h => headers.delete(h));
    const init = {method: request.method, headers, redirect: 'manual'};
    if (request.method === 'POST') {
      const body = await request.arrayBuffer();
      if (body.byteLength > MAX_BODY) return json({detail: 'Request too large'}, 413);
      init.body = body;
    }
    const ctrl = new AbortController(), timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    try {
      const upstream = await fetch(new URL(path + url.search, env.BACKEND_URL), {...init, signal: ctrl.signal});
      const out = new Headers(upstream.headers);
      out.delete('set-cookie');
      out.set('cache-control', 'no-store');
      return new Response(upstream.body, {status: upstream.status, statusText: upstream.statusText, headers: out});
    } catch (err) {
      return err && err.name === 'AbortError'
        ? json({detail: 'The server took too long to respond. It may be waking up, so try again in a minute.'}, 504)
        : json({detail: 'The server could not be reached.'}, 502);
    } finally { clearTimeout(timer); }
  },
};
