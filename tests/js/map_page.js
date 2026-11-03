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
    apply: true, refuse: false, down: false, running: false, strokes: [], refuseUndo: false,
    exploring: false, tags: null, mode: "manual", decisions: null };
  c.state = () => c.ready ? {
    ready: true, age_s: 0.2, turn_radius_m: 0.7, tracks: [], pad: c.pad, mode: c.mode,
    decisions: c.decisions,
    patrol: { frame: 0, present: true, pose: [0, 0, 0], use: "patrol", state: "editing", mode: "loop",
      running: c.running, index: 0, points: c.points.slice(), name: null, dirty: true, stuck_s: 0,
      skip_stuck: false, strokes: c.strokes.length, pending: null, fit: {}, said: c.said.slice(), acks: c.acks.slice(),
      explore: { on: c.exploring, target: c.exploring ? [1.0, 0.5] : null, kind: c.exploring ? "frontier" : "",
        distance_m: c.exploring ? 1.1 : -1, targets: 0, arrived: 0, skipped: 0, frontiers: 1 },
      tags: c.tags ? Object.assign({}, c.tags) : null },
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
        // As the patrol answers: [id, taken, why], why the refusal's own words.
        let took = !c.refuse, why = c.refuse ? "Stroke refused: no SLAM map yet" : "";
        if (body.cmd === "stroke" && took) {
          body.points.forEach(q => c.points.push(q)); c.strokes.push(body.points.length);
        }
        if (body.cmd === "unstroke" && took) {
          if (c.strokes.length && !c.refuseUndo) c.points.splice(c.points.length - c.strokes.pop());
          else { took = false; why = "Nothing to undo: every stroke is already taken back"; }
        }
        if (!took) c.said.push(["warning", why]);
        if (body.cmd === "start") { c.running = true; c.exploring = false; }
        if (body.cmd === "explore") { c.exploring = body.on; if (body.on) c.running = false; }
        if (body.cmd === "tags") ["clean"].forEach(k => { if (k in body) c.tags[k] = body[k]; });
        c.acks.push([body.id, took, took ? "" : why]);
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

async function drag(page, cdp, n, during, at) {
  const box = await page.locator("#cv").boundingBox();
  const cx = at ? at.x : box.x + box.width / 2, cy = at ? at.y : box.y + box.height / 2;
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
    must(/Car didn't take the stroke/.test(msg) && /no answer/.test(msg), "error: " + msg);
    must(await shown(page) === n, "the points went with the error");
    c.apply = true;
    await page.tap("#e_retry");
    await page.waitForTimeout(1500);
    must(c.posts.length === 2 && c.posts[1].cmd === "stroke"
      && JSON.stringify(c.posts[1].points) === JSON.stringify(c.posts[0].points), "retry sent something else");
    must(c.posts[1].id !== c.posts[0].id, "a retry under the same id as the try the car already answered");
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
    must(/Car didn't take the stroke of \d+ waypoints: Stroke refused: no SLAM map yet\. Retry/.test(msg), "error: " + msg);
    must(await shown(page) === c.posts[0].points.length, "points dropped on a refusal");
    // The map comes up; Retry is a new try the car takes, not the old
    // refusal read again off its acks.
    c.refuse = false;
    await page.tap("#e_retry");
    await page.waitForTimeout(1500);
    must(await text(page, "err") === "", "still an error after the retry was taken: " + await text(page, "err"));
    must(await shown(page) === c.posts[0].points.length && c.points.length === c.posts[0].points.length,
      "shown " + await shown(page) + ", the car has " + c.points.length);
    // Refused again and discarded: gone.
    c.refuse = true;
    await drag(page, cdp, 40);
    await page.waitForTimeout(1500);
    must(/Car didn't take the stroke/.test(await text(page, "err")), "second refusal not said");
    await page.tap("#e_drop");
    await page.waitForTimeout(200);
    must(await shown(page) === c.points.length && await text(page, "err") === "", "discard");
    await ctx.close();
  });

  await check("a_waypoint_lands_under_the_finger_after_a_banner", async () => {
    // The owner's "20 to 100 px above my finger": the page sized its drawing
    // at load, then a banner (no controller, here) took height off the map
    // box and the drawing was squashed into it, so what went under the
    // finger showed up higher, more so further down the screen.
    const c = car(), { ctx, page, cdp } = await open(b, c);
    await page.tap("#b_draw");
    c.pad = false;                               // the no-controller banner comes up
    await page.waitForTimeout(900);
    must(/No controller/.test(await text(page, "nr")), "no banner to move the map");
    const box = await page.locator("#cv").boundingBox(), worst = [];
    for (const f of [0.15, 0.5, 0.9]) {
      const at = { x: box.x + box.width * 0.3, y: box.y + box.height * f };
      const n0 = c.posts.length;
      await page.touchscreen.tap(at.x, at.y);
      await page.waitForTimeout(250);
      const q = c.posts[n0].points[0], s = await page.evaluate((q) => window.__mapStats.screen(q[0], q[1]), q);
      worst.push(Math.hypot(s[0] - at.x, s[1] - at.y));
      // The first point of a drawn stroke, too.
      const m0 = c.posts.length;
      await drag(page, cdp, 30, null, { x: box.x + box.width * 0.7, y: box.y + box.height * f });
      await page.waitForTimeout(250);
      const p = c.posts[m0].points[0], t = await page.evaluate((q) => window.__mapStats.screen(q[0], q[1]), p);
      worst.push(Math.hypot(t[0] - (box.x + box.width * 0.7 - 120), t[1] - (box.y + box.height * f)));
    }
    must(Math.max(...worst) < 2, "px off the finger: " + worst.map(d => d.toFixed(1)).join(" "));
    // And the drawing is the box's size: nothing stretched.
    const fit = await page.evaluate(() => { const cv = document.getElementById("cv"), r = cv.getBoundingClientRect();
      return [cv.width, Math.round(r.width * devicePixelRatio), cv.height, Math.round(r.height * devicePixelRatio)]; });
    must(fit[0] === fit[1] && fit[2] === fit[3], "backing store " + fit);
    await ctx.close();
  });

  await check("a_refused_undo_never_comes_back_for_the_next_loop", async () => {
    // What the owner hit: an Undo with nothing left to undo was refused and
    // kept with a Retry; the next loop drawn showed "Car didn't take it", and
    // Retry sent that old Undo again, which took the new loop back.
    const c = car(), { ctx, page, cdp } = await open(b, c);
    await page.tap("#b_draw");
    await drag(page, cdp, 40);
    await page.waitForTimeout(1200);
    await page.tap("#b_unstroke");
    await page.waitForTimeout(1200);
    must(c.points.length === 0, "the first undo was not taken");
    const n0 = c.posts.length;
    await page.tap("#b_unstroke");               // nothing left to undo
    await page.waitForTimeout(600);
    must(c.posts.length === n0, "an Undo sent with nothing to undo");
    must(/Nothing to undo/.test(await page.evaluate(() => document.getElementById("toast").textContent)), "not said");
    // The car refuses one anyway (its own count): said, and gone.
    c.strokes.push(0); c.refuseUndo = true;
    await page.waitForTimeout(700);
    await page.tap("#b_unstroke");
    await page.waitForTimeout(1200);
    c.strokes.pop(); c.refuseUndo = false;
    must(c.posts.length === n0 + 1 && c.posts[n0].cmd === "unstroke", "the refused undo not sent");
    must(/Nothing to undo/.test(await page.evaluate(() => document.getElementById("toast").textContent)), "refusal not said");
    must((await edits(page)).length === 0 && await text(page, "err") === "", "a refused undo was kept");
    // The loop the owner is happy with: taken, no error, nothing resent.
    await drag(page, cdp, 60);
    await page.waitForTimeout(1500);
    must(await text(page, "err") === "", "an error next to a loop the car took: " + await text(page, "err"));
    must(c.points.length > 4 && await shown(page) === c.points.length, "the loop");
    must(c.posts.slice(n0 + 1).every(p => p.cmd === "stroke"), "something besides the stroke went");
    await ctx.close();
  });

  await check("a_finger_down_never_times_out_a_stroke_the_car_took", async () => {
    // Polls stop while a finger is down, so the car's ack cannot arrive.
    // The page gave up after 5 s of clock regardless, then kept the stroke
    // as not taken even once the ack came: shown twice and "didn't take it".
    const c = car(); c.apply = false;
    const { ctx, page, cdp } = await open(b, c);
    await page.tap("#b_draw");
    await drag(page, cdp, 30);
    await page.waitForTimeout(100);
    const first = c.posts[0];
    c.points.push(...first.points); c.strokes.push(first.points.length); c.acks.push([first.id, true, ""]);
    const box = await page.locator("#cv").boundingBox();
    await drag(page, cdp, 30, async () => { await page.waitForTimeout(7000); },
      { x: box.x + box.width / 2, y: box.y + box.height * 0.8 });
    await page.waitForTimeout(1500);
    const e = await edits(page);
    must(!e.some(x => x.state === "failed") && !/no answer/.test(await text(page, "err")),
      "gave up on the first stroke: " + JSON.stringify(e));
    must(e.length === 1 && e[0].state === "sent", "the second is waiting: " + JSON.stringify(e));
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

  await check("auto_decisions_that_gave_up_say_so", async () => {
    const c = car(); c.mode = "auto";
    c.decisions = { state: "late", why: "Auto decisions are late. Holding neutral until one lands.", age_s: 0.4 };
    const { ctx, page } = await open(b, c);
    let msg = await text(page, "nr");
    must(/decisions are late/.test(msg), "late: " + msg);
    c.decisions = { state: "gave up", why: "Auto decisions keep crashing (ValueError: x). Holding neutral in auto; START out of auto and back in to try again.", age_s: null };
    await page.waitForTimeout(800);
    msg = await text(page, "nr");
    must(/keep crashing/.test(msg), "gave up: " + msg);
    must(/bad/.test(await page.evaluate(() => document.getElementById("nr").className)), "not loud");
    c.decisions = { state: "ok", why: "", age_s: 0.05 };
    await page.waitForTimeout(800);
    must(await text(page, "nr") === "", "ok still says: " + await text(page, "nr"));
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

  await check("explore_is_a_toggle_next_to_start_and_draws_its_target", async () => {
    const c = car(), { ctx, page } = await open(b, c);
    const sub = async () => page.evaluate(() => document.querySelector("#b_explore .sub").textContent);
    must(await sub() === "off", "off: " + await sub());
    await page.tap("#b_explore");
    await page.waitForTimeout(900);
    const on = c.posts.filter(p => p.cmd === "explore");
    must(on.length === 1 && on[0].on === true, "sent: " + JSON.stringify(on));
    must(/Armed\. START on the pad to drive/.test(await page.evaluate(() =>
      document.getElementById("toast").textContent)), "not armed");
    must(await sub() === "on", "not shown on: " + await sub());
    // The target's ring, in its own colour, where the car says it is.
    const hit = await page.evaluate(() => {
      const cv = document.getElementById("cv"), g = cv.getContext("2d"), d = window.devicePixelRatio;
      const w = cv.width / d, h = cv.height / d;
      let n = 0;
      for (let dx = -14; dx <= 14; dx++) for (let dy = -14; dy <= 14; dy++) {
        const a = g.getImageData(Math.round((w / 2 + 40 + dx) * d), Math.round((h / 2 - 20 + dy) * d), 1, 1).data;
        if (a[0] > 200 && a[1] > 130 && a[1] < 190 && a[2] < 90) n++; }
      return n;
    });
    must(hit > 5, "no target ring: " + hit);
    await page.tap("#b_explore");
    await page.waitForTimeout(900);
    const off = c.posts.filter(p => p.cmd === "explore");
    must(off.length === 2 && off[1].on === false, "off sent: " + JSON.stringify(off));
    must(await sub() === "off", "not off again: " + await sub());
    await ctx.close();
  });

  await check("the_clean_button_sends_the_tag_and_shows_the_cars", async () => {
    const off = car(), o = await open(b, off);
    must(await o.page.evaluate(() => document.getElementById("t_clean").disabled),
      "enabled on a car that keeps no tags");
    await o.ctx.close();
    const c = car(); c.tags = { clean: false };
    const { ctx, page } = await open(b, c);
    const look = async (id) => page.evaluate((i) => { const e = document.getElementById(i);
      return (e.classList.contains("on") ? "on " : "off ") + e.querySelector(".sub").textContent; }, id);
    must(await page.evaluate(() => !document.getElementById("t_outdoor")), "an Outdoor button");
    must(await look("t_clean") === "off off", "not off at first");
    await page.tap("#t_clean");
    await page.waitForTimeout(900);
    const sent = c.posts.filter(p => p.cmd === "tags").map(p => { const q = Object.assign({}, p); delete q.id; return q; });
    must(JSON.stringify(sent) === JSON.stringify([{ cmd: "tags", clean: true }]),
      "sent: " + JSON.stringify(sent));
    must(await look("t_clean") === "on on", "not shown on");
    // The car's word wins: a tag it says is off shows off.
    c.tags.clean = false;
    await page.waitForTimeout(900);
    must(await look("t_clean") === "off off", "shows " + await look("t_clean"));
    await page.tap("#t_clean");
    await page.waitForTimeout(900);
    must(c.posts.filter(p => p.cmd === "tags").pop().clean === true, "a tap sends the other side of the car's");
    await ctx.close();
  });

  console.log(JSON.stringify(out));
  await b.close();
})();
