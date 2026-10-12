import assert from 'node:assert/strict';
import worker from '../../frontend/_worker.js';

const calls = [];
globalThis.fetch = async (url, init) => { calls.push({url: String(url), init}); return new Response('{"ok":1}', {status: 200, headers: {'set-cookie': 'a=b'}}); };
const env = {BACKEND_URL: 'https://api.example.com', ASSETS: {fetch: async () => new Response('static')}};
const run = (path, init = {}, e = env) => worker.fetch(new Request('https://site.example' + path, init), e);

assert.equal(await (await run('/')).text(), 'static');
assert.equal(await (await run('/assets/a.jpg')).text(), 'static');
let r = await run('/api/v1/consult?x=1', {method: 'POST', body: '{}', headers: {cookie: 'secret', 'content-type': 'application/json'}});
assert.equal(r.status, 200); assert.equal(calls.at(-1).url, 'https://api.example.com/api/v1/consult?x=1');
assert.equal(calls.at(-1).init.headers.get('cookie'), null); assert.equal(r.headers.get('set-cookie'), null);
for (const p of ['/api/v1/ops/stats', '/api/v1/ops']) assert.equal((await run(p)).status, 404);
assert.equal((await run('/health', {method: 'DELETE'})).status, 405);
assert.equal((await run('/health', {}, {...env, BACKEND_URL: ''})).status, 503);
const n = calls.length;
r = await run('/api/v1/face/analyze', {method: 'POST', body: new Uint8Array(7 * 1024 * 1024)});
assert.equal(r.status, 413); assert.equal(calls.length, n);
globalThis.fetch = async () => { throw new Error('down'); };
assert.equal((await run('/health')).status, 502);
console.log('worker ok');
