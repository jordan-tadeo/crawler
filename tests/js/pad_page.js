// The pad page (crawler/telemetry/pad.html) and the map page's tab to it, in
// a phone-sized headless Chromium, against a stand-in car this script
// answers for. Run by tests/test_pad_page_js.py: node pad_page.js PAD_HTML MAP_HTML
// Prints one JSON line: {"name": ok-or-why, ...}.
const fs = require("fs");
const { chromium } = require("playwright");
const padHtml = fs.readFileSync(process.argv[2], "utf8");
const mapHtml = fs.readFileSync(process.argv[3], "utf8");
const BASE = "http://car.test";

function car() {
  // The car's side: PadRepair.status() and the refusals its start() gives.
  const c = { posts: [], mode: "manual", st: { state: "idle", why: "", mac: null, left_s: null,
    scan_s: 90, running: false } };
  c.press = () => {
    if (c.mode === "auto") return { ok: false, why: "Stop auto first" };
    if (c.st.running) return { ok: false, why: "Already re-pairing" };
    c.st = Object.assign({}, c.st, { state: "removing", why: "", running: true });
    return { ok: true, why: "" };
  };
  return c;
}

async function open(b, c, path) {
  const ctx = await b.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 3,
    isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.route("**/*", async (route) => {
    const req = route.request(), p = new URL(req.url()).pathname;
    if (p === "/pad") return route.fulfill({ status: 200, contentType: "text/html", body: padHtml });
    if (p === "/map") return route.fulfill({ status: 200, contentType: "text/html", body: mapHtml });
    if (req.method() === "POST" && p === "/api/cmd") {
      const body = JSON.parse(req.postData()); c.posts.push(body);
      const r = body.cmd === "repair_pad" ? c.press() : { ok: false, why: "?" };
      return route.fulfill({ status: r.ok ? 202 : 400, contentType: "application/json",
        body: JSON.stringify(r) });
    }
    if (p === "/api/pad") return route.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify(c.st) });
    if (p === "/api/state") return route.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify({ ready: false, age_s: null, turn_radius_m: 0.7, tracks: [] }) });
    return route.fulfill({ status: 404, body: "" });
  });
  await page.goto(BASE + path);
  await page.waitForTimeout(400);
  return { ctx, page };
}

const txt = (page, id) => page.evaluate((i) => document.getElementById(i).textContent, id);

(async () => {
  const b = await chromium.launch();
  const out = {};
  const check = async (name, fn) => {
    try { await fn(); out[name] = "ok"; } catch (e) { out[name] = String(e && e.message || e); }
  };
  const must = (cond, why) => { if (!cond) throw new Error(why); };

  await check("the_button_sends_repair_pad_and_shows_each_state", async () => {
    const c = car(), { ctx, page } = await open(b, c, "/pad");
    must(/Hold the pair button on top of the pad until the light flashes fast/.test(
      await page.textContent("main")), "no pairing instruction");
    must(await txt(page, "state") === "Not running", "idle shows " + await txt(page, "state"));
    await page.tap("#go");
    await page.waitForTimeout(300);
    must(JSON.stringify(c.posts) === JSON.stringify([{ cmd: "repair_pad" }]), "sent " + JSON.stringify(c.posts));
    must(await page.evaluate(() => document.getElementById("go").disabled), "enabled while running");
    const seen = {};
    for (const [st, extra] of [["removing", {}], ["waiting", { left_s: 73 }], ["pairing", { why: "9C:AA:1B:B6:6B:34" }],
      ["connected", { why: "9C:AA:1B:B6:6B:34", running: false }],
      ["failed", { why: "pair: Failed to pair: org.bluez.Error.AuthenticationFailed", running: false }],
      ["timed_out", { why: "no pad in pairing mode in 90 s", running: false }]]) {
      c.st = Object.assign({ state: st, why: "", mac: null, left_s: null, scan_s: 90, running: true }, extra);
      await page.waitForTimeout(1200);
      seen[st] = [await txt(page, "state"), await txt(page, "why"),
        await page.evaluate(() => document.getElementById("go").disabled)];
    }
    must(seen.removing[0] === "Forgetting the old pairing" && seen.removing[2], "removing " + seen.removing);
    must(seen.waiting[0] === "Waiting for the pad in pairing mode · 73 s left", "waiting " + seen.waiting);
    must(seen.pairing[0] === "Pairing with the pad", "pairing " + seen.pairing);
    must(seen.connected[0] === "Connected. Ready to drive" && !seen.connected[2], "connected " + seen.connected);
    must(seen.failed[0] === "Failed" && /AuthenticationFailed/.test(seen.failed[1]) && !seen.failed[2],
      "failed " + seen.failed);
    must(seen.timed_out[0] === "Timed out" && /90 s/.test(seen.timed_out[1]), "timed out " + seen.timed_out);
    await ctx.close();
  });

  await check("a_refusal_says_why_and_leaves_the_button_usable", async () => {
    const c = car(); c.mode = "auto";
    const { ctx, page } = await open(b, c, "/pad");
    await page.tap("#go");
    await page.waitForTimeout(1300);
    must(await txt(page, "why") === "Stop auto first", "said " + await txt(page, "why"));
    must(await txt(page, "state") === "Not running", "state " + await txt(page, "state"));
    must(!(await page.evaluate(() => document.getElementById("go").disabled)), "button stuck disabled");
    await ctx.close();
  });

  await check("the_map_page_has_a_pad_tab", async () => {
    const c = car(), { ctx, page } = await open(b, c, "/map");
    const tabs = await page.evaluate(() => Array.from(document.querySelectorAll("nav.tabs a")).map(a => {
      const r = a.getBoundingClientRect(); return [a.getAttribute("href"), a.textContent, r.width, r.top]; }));
    must(JSON.stringify(tabs.map(t => t[0])) === '["/","/map","/pad"]', "tabs " + JSON.stringify(tabs));
    must(tabs.every(t => t[2] > 80 && t[3] === tabs[0][3]), "tabs not one row: " + JSON.stringify(tabs));
    await page.tap("text=Pad");
    await page.waitForTimeout(500);
    must(new URL(page.url()).pathname === "/pad", "went to " + page.url());
    must(c.posts.length === 0, "the tab sent a command");
    await ctx.close();
  });

  console.log(JSON.stringify(out));
  await b.close();
})();
