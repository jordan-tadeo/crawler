// The map page (crawler/telemetry/map.html) in a phone-sized headless
// Chromium with touch, against a stand-in car this script answers for.
// Run by tests/test_map_page_js.py: node map_page.js MAP_HTML PNG
// Prints one JSON line: {"name": ok-or-why, ...}.
const fs = require("fs");
const { chromium } = require("playwright");
const html = fs.readFileSync(process.argv[2], "utf8");
const png = fs.readFileSync(process.argv[3]);
const BASE = "http://car.test";

function car() {
  // The car's side: the patrol's points and acks, and what it was sent.
  const c = { points: [], acks: [], said: [], posts: [], gets: [], ready: true, pad: true,
    apply: true, refuse: false, down: false, running: false };
  c.state = () => c.ready ? {
    ready: true, age_s: 0.2, turn_radius_m: 0.7, tracks: [], pad: c.pad, mode: "manual",
    patrol: { frame: 0, present: true, pose: [0, 0, 0], use: "patrol", state: "editing", mode: "loop",
      running: c.running, index: 0, points: c.points.slice(), name: null, dirty: true, stuck_s: 0,
      skip_stuck: false, strokes: 0, pending: null, fit: {}, said: c.said.slice(), acks: c.acks.slice() },
  } : { ready: false, age_s: null, turn_radius_m: 0.7, tracks: [] };
  return c;
}

async function open(b, c) {
  const ctx = await b.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 3,
    isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.route("**/*", async (route) => {
    const req = route.request(), u = new URL(req.url()), p = u.pathname;
    if (p === "/map") return route.fulfill({ status: 200, contentType: "text/html", body: html });
    if (c.down) return route.abort("connectionrefused");
    if (req.method() === "POST" && p === "/api/cmd") {
      const body = JSON.parse(req.postData()); c.posts.push(body);
      if (c.apply) {
        if (body.cmd === "stroke" && !c.refuse) body.points.forEach(q => c.points.push(q));
        if (c.refuse) c.said.push(["warning", "Stroke refused: no SLAM map yet"]);
        if (body.cmd === "start") c.running = true;
        c.acks.push([body.id, !c.refuse]);
      }
      return route.fulfill({ status: 202, contentType: "application/json", body: '{"ok":true,"why":""}' });
    }
    c.gets.push(p);
    if (p === "/api/state") return route.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify(c.state()) });
    if (p === "/api/map.png") return route.fulfill({ status: 200, contentType: "image/png", body: png,
      headers: { "X-Map": "-2.000 -2.000 0.1000 40 40", "ETag": '"m1"' } });
    return route.fulfill({ status: 404, body: "" });
  });
  await page.goto(BASE + "/map");
  await page.waitForTimeout(1200);
  const cdp = await ctx.newCDPSession(page);
  return { ctx, page, cdp };
}

async function drag(page, cdp, n, during) {
  const box = await page.locator("#cv").boundingBox();
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  const pts = [];
  for (let k = 0; k <= n; k++) pts.push({ x: cx - 120 + 240 * k / n, y: cy + 40 * Math.sin(k / 6) });
  await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [pts[0]] });
  for (let k = 1; k <= n; k++) {
    await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: [pts[k]] });
    await new Promise(r => setTimeout(r, 16));
  }
  if (during) await during();
  await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
}

const shown = (page) => page.evaluate(() => window.__mapStats.shown().filter(q => q).length);
const edits = (page) => page.evaluate(() => window.__mapStats.edits());
const text = (page, id) => page.evaluate((i) => {
  const e = document.getElementById(i); return e.classList.contains("show") ? e.textContent : ""; }, id);

(async () => {
  const b = await chromium.launch();
  const out = {};
  const check = async (name, fn) => {
    try { await fn(); out[name] = "ok"; } catch (e) { out[name] = String(e && e.message || e); }
  };
  const must = (cond, why) => { if (!cond) throw new Error(why); };

  await check("stroke_is_one_command_on_lift", async () => {
    const c = car(), { ctx, page, cdp } = await open(b, c);
    await page.tap("#b_draw");
    await page.waitForTimeout(300);
    const posts0 = c.posts.length;
    let mid = null;
    await drag(page, cdp, 60, async () => {
      mid = { posts: c.posts.length, gets: c.gets.length,
        draws: await page.evaluate(() => window.__mapStats.draws),
        overlay: await page.evaluate(() => window.__mapStats.overlay) };
      await page.waitForTimeout(1200);           // two state polls' worth, finger still down
      mid.gets2 = c.gets.length;
      mid.draws2 = await page.evaluate(() => window.__mapStats.draws);
    });
    await page.waitForTimeout(300);
    must(mid.posts === posts0, "a command went out during the drag");
    must(mid.gets2 === mid.gets, `polled ${mid.gets2 - mid.gets} times with a finger down`);
    must(mid.draws2 === mid.draws, "a full redraw with a finger down");
    must(mid.overlay > 5, `the live stroke drew ${mid.overlay} frames`);
    const strokes = c.posts.slice(posts0).filter(p => p.cmd === "stroke");
    must(c.posts.length - posts0 === 1 && strokes.length === 1, `sent ${c.posts.length - posts0} commands`);
    must(strokes[0].points.length >= 4 && typeof strokes[0].id === "string", "the stroke's points and id");
    const ta = await page.evaluate(() => getComputedStyle(document.getElementById("cv")).touchAction);
    must(ta === "none", "touch-action " + ta);
    await ctx.close();
  });

  await check("poll_never_clobbers_an_unanswered_stroke", async () => {
    const c = car(); c.apply = false;            // the car queues it and says nothing yet
    const { ctx, page, cdp } = await open(b, c);
    await page.tap("#b_draw");
    await drag(page, cdp, 40);
    await page.waitForTimeout(400);
    const n = c.posts[0].points.length;
    must(await shown(page) === n, "not shown at the lift");
    await page.waitForTimeout(2500);             // five polls of a state without it
    must(await shown(page) === n, "a poll took the stroke off the screen");
    must((await edits(page))[0].state === "sent", "not waiting");
    // The car takes it: then it is the car's, once.
    c.points.push(...c.posts[0].points); c.acks.push([c.posts[0].id, true]);
    await page.waitForTimeout(1200);
    must(await shown(page) === n, `${await shown(page)} shown, not ${n}, once taken`);
    must((await edits(page)).length === 0, "still pending after the ack");
    must(await text(page, "err") === "", "an error for a stroke that was taken");
    await ctx.close();
  });

  await check("no_answer_is_an_error_with_retry", async () => {
    const c = car(); c.apply = false;
    const { ctx, page, cdp } = await open(b, c);
    await page.tap("#b_draw");
    await drag(page, cdp, 40);
    await page.waitForTimeout(6500);
    const n = c.posts[0].points.length;
    const msg = await text(page, "err");
    must(/Car didn't take it/.test(msg) && /no answer/.test(msg), "error: " + msg);
    must(await shown(page) === n, "the points went with the error");
    c.apply = true;
    await page.tap("#e_retry");
    await page.waitForTimeout(1500);
    must(c.posts.length === 2 && c.posts[1].cmd === "stroke"
      && JSON.stringify(c.posts[1].points) === JSON.stringify(c.posts[0].points), "retry sent something else");
    must(await text(page, "err") === "", "the error stayed after it was taken");
    must(await shown(page) === n && (await edits(page)).length === 0, "after the retry");
    await ctx.close();
  });

  await check("a_refusal_says_why_and_keeps_the_points", async () => {
    const c = car(); c.refuse = true;
    const { ctx, page, cdp } = await open(b, c);
    await page.tap("#b_draw");
    await drag(page, cdp, 40);
    await page.waitForTimeout(1500);
    const msg = await text(page, "err");
    must(/Car didn't take it: Stroke refused: no SLAM map yet/.test(msg), "error: " + msg);
    must(await shown(page) === c.posts[0].points.length, "points dropped on a refusal");
    await page.tap("#e_drop");
    await page.waitForTimeout(200);
    must(await shown(page) === 0 && await text(page, "err") === "", "discard");
    await ctx.close();
  });

  await check("the_not_ready_banner_says_why", async () => {
    const c = car(); c.ready = false;
    const { ctx, page } = await open(b, c);
    let msg = await text(page, "nr");
    must(/Car not ready/.test(msg) && /control loop/.test(msg), "not ready: " + msg);
    c.ready = true; c.pad = false;
    await page.waitForTimeout(800);
    msg = await text(page, "nr");
    must(/No controller/.test(msg) && /Connect the controller to drive/.test(msg), "no pad: " + msg);
    await page.tap("#b_run");
    await page.waitForTimeout(800);
    must(c.posts.some(p => p.cmd === "start"), "start not sent");
    msg = await text(page, "nr");
    must(/Patrol armed\. Connect the controller to drive/.test(msg), "armed: " + msg);
    c.pad = true;
    await page.waitForTimeout(800);
    must(await text(page, "nr") === "", "a banner with everything fine: " + await text(page, "nr"));
    c.down = true;
    await page.waitForTimeout(1300);
    msg = await text(page, "nr");
    must(/No connection to the car/.test(msg), "down: " + msg);
    await ctx.close();
  });

  await check("unknown_is_drawn_apart_from_free", async () => {
    const c = car(), { ctx, page } = await open(b, c);
    // A row of pixels off the map (unknown: hatched, so more than one
    // colour) and one on the map's free floor (one flat colour, unlike
    // anything in the unknown row).
    const px = await page.evaluate(() => {
      const cv = document.getElementById("cv"), g = cv.getContext("2d"), d = window.devicePixelRatio;
      const row = (x, y) => { const a = g.getImageData(Math.round(x * d), Math.round(y * d), Math.round(30 * d), 1).data, o = [];
        for (let i = 0; i < a.length; i += 4) o.push(a[i] + "," + a[i + 1] + "," + a[i + 2]); return o; };
      const w = cv.width / d, h = cv.height / d;
      return { far: row(5, h - 5), free: row(w / 2 + 10, h / 2 + 30) };
    });
    const far = new Set(px.far), free = new Set(px.free);
    must(far.size >= 2, "unknown is flat: " + [...far]);
    must(free.size === 1 && !far.has([...free][0]), "free " + [...free] + " against unknown " + [...far]);
    await ctx.close();
  });

  console.log(JSON.stringify(out));
  await b.close();
})();
