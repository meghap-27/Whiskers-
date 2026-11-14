/*
 * FolderFloat, ported from the React Bits component (React + matter-js) to plain JavaScript
 * so it runs inside the Streamlit page without a React build.
 *
 * Same behaviour as the original: pills rise out of the folder in rows, then a zero-gravity
 * matter-js world lets them drift, collide and be dragged. Additions for this app:
 *   - cat-ear folder tab and tabby stripes on the flap (CSS)
 *   - opens on keyboard focus as well as hover; on touch screens it toggles on tap
 *   - each pill can carry a hint (shown as a tooltip) and an icon glyph
 *   - optional `onPeek` callback while hovering a pill
 */
(function () {
  const PAD = 28, CHAR = 6.8, GAP = 12, ROW = 52, DRAG_MIN = 4, ZONE_PAD = 8;

  const jitter = i => { const x = Math.sin(i * 12.9898 + 4.1414) * 43758.5453; return x - Math.floor(x); };

  function layout(list, spread, lift, tilt, sizes) {
    const rows = []; let row = [], width = 0;
    list.forEach((item, i) => {
      const pw = (sizes[i] && sizes[i].w) || PAD + item.label.length * CHAR;
      if (row.length && width + GAP + pw > spread * 2) { rows.push({ items: row, width }); row = []; width = 0; }
      row.push({ i, pw }); width += (row.length > 1 ? GAP : 0) + pw;
    });
    if (row.length) rows.push({ items: row, width });
    const pos = [];
    rows.forEach((r, ri) => {
      let x = -r.width / 2;
      const shift = (ri % 2 ? 1 : -1) * Math.min(16, spread * 0.1);
      r.items.forEach(({ i, pw }) => {
        const j = jitter(i);
        pos[i] = { x: x + pw / 2 + shift + (j - 0.5) * 6, y: -lift - ri * ROW - j * 6, r: tilt * (j * 2 - 1) };
        x += pw + GAP;
      });
    });
    return pos;
  }

  function FolderFloat(mount, opts) {
    const o = Object.assign({
      items: [], label: "Folder", sublabel: "", trigger: "hover", defaultOpen: false, closeOnSelect: true,
      physics: true, drift: 0.5, onSelect: null, onOpenChange: null, onPeek: null,
      folderColor: "#3f3f46", frontColor: "#52525b", paperColor: "#f5f5f5", itemColor: "#f5f5f5",
      itemTextColor: "#18181b", labelColor: "#f5f5f5", width: 200, height: 148, radius: 14, spread: 180,
      lift: 26, tilt: 8, flapAngle: 34, restAngle: 16, openDuration: 520, stagger: 45, bounce: 0.3, className: "",
    }, opts || {});
    if (o.trigger === "hover" && window.matchMedia("(hover: none)").matches) o.trigger = "click";

    const list = o.items.map(it => (typeof it === "string" ? { label: it, value: it } : Object.assign({ value: it.label }, it)));
    const n = list.length;
    const sub = o.sublabel || `${n} ${n === 1 ? "note" : "notes"}`;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
    const Matter = window.Matter;

    // ---------- DOM
    const root = document.createElement("div");
    root.className = "folder-float" + (o.className ? " " + o.className : "");
    root.dataset.trigger = o.trigger;
    if (o.physics && Matter) root.dataset.physics = "";
    const vars = {
      "--ff-w": `${o.width}px`, "--ff-h": `${o.height}px`, "--ff-r": `${o.radius}px`,
      "--ff-back": o.folderColor, "--ff-front": o.frontColor, "--ff-paper": o.paperColor,
      "--ff-item": o.itemColor, "--ff-item-ink": o.itemTextColor, "--ff-label": o.labelColor,
      "--ff-spread": `${o.spread}px`, "--ff-lift": `${o.lift}px`, "--ff-angle": `${o.flapAngle}deg`,
      "--ff-rest": `${o.restAngle}deg`, "--ff-open": `${o.openDuration}ms`,
      "--ff-close": `${Math.round(o.openDuration * 0.6)}ms`, "--ff-stagger": `${o.stagger}ms`, "--ff-n": n,
      "--ff-spring": `cubic-bezier(0.34, ${(1 + o.bounce * 1.9).toFixed(2)}, 0.64, 1)`,
    };
    Object.entries(vars).forEach(([k, v]) => root.style.setProperty(k, v));

    const anchor = document.createElement("div");
    anchor.className = "folder-float__items";
    const pills = list.map((item, i) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "folder-float__item"; b.tabIndex = -1;
      b.setAttribute("aria-hidden", "true");
      b.style.setProperty("--i", i);
      if (item.hint) b.title = item.hint;
      const s = document.createElement("span");
      s.className = "folder-float__drift";
      s.innerHTML = (item.icon ? `<i class="folder-float__icon" aria-hidden="true">${item.icon}</i>` : "");
      s.appendChild(document.createTextNode(item.label));
      b.appendChild(s); anchor.appendChild(b);
      return b;
    });

    const folder = document.createElement("div");
    folder.className = "folder-float__folder";
    folder.innerHTML = `
      <span class="folder-float__back" aria-hidden="true"></span>
      <span class="folder-float__paper" aria-hidden="true"></span>
      <span class="folder-float__front" aria-hidden="true">
        <span class="folder-float__label"></span><span class="folder-float__sub"></span>
      </span>`;
    folder.querySelector(".folder-float__label").textContent = o.label;
    folder.querySelector(".folder-float__sub").textContent = sub;
    const trig = document.createElement("button");
    trig.type = "button"; trig.className = "folder-float__trigger";
    trig.setAttribute("aria-expanded", "false"); trig.setAttribute("aria-label", `${o.label}, ${sub}`);
    folder.appendChild(trig);
    root.append(anchor, folder);
    mount.appendChild(root);

    // ---------- state
    let open = false, sizes = [], pos = [];
    const w = { engine: null, bodies: [], sizes: [], raf: 0, last: 0, t0: 0, drag: null, zone: null, live: false };
    let liveTimer, popTimer;

    function place() {
      pos = layout(list, o.spread, o.lift, o.tilt, sizes);
      pills.forEach((el, i) => {
        el.style.setProperty("--x", `${pos[i].x.toFixed(1)}px`);
        el.style.setProperty("--y", `${pos[i].y.toFixed(1)}px`);
        el.style.setProperty("--r", `${pos[i].r.toFixed(2)}deg`);
      });
    }
    function measure() { sizes = pills.map(el => ({ w: el.offsetWidth, h: el.offsetHeight })); place(); }
    place(); measure();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(measure);

    function stopPhysics() {
      clearTimeout(liveTimer); cancelAnimationFrame(w.raf); w.raf = 0;
      if (w.engine) {
        w.bodies.forEach((b, i) => {
          pills[i].style.setProperty("--x", `${b.position.x.toFixed(1)}px`);
          pills[i].style.setProperty("--y", `${(b.position.y - w.sizes[i].h / 2).toFixed(1)}px`);
        });
        Matter.Composite.clear(w.engine.world, false, true); Matter.Engine.clear(w.engine); w.engine = null;
      }
      w.bodies = []; w.drag = null; w.live = false; delete root.dataset.live;
    }

    function startPhysics() {
      if (w.engine || !Matter) return;
      const { Bodies, Body, Composite, Engine } = Matter;
      const engine = Engine.create({ gravity: { x: 0, y: 0 } });
      engine.enableSleeping = false; w.engine = engine;
      w.sizes = pills.map(el => ({ w: el.offsetWidth, h: el.offsetHeight }));
      const ys = pos.map(p => p.y);
      const zone = { left: -o.spread - ZONE_PAD, right: o.spread + ZONE_PAD, top: Math.min(...ys) - ZONE_PAD,
                     bottom: -o.lift + Math.max(...w.sizes.map(s => s.h)) };
      w.zone = zone;
      w.bodies = pills.map((el, i) => {
        const { w: bw, h: bh } = w.sizes[i];
        const b = Bodies.rectangle(pos[i].x, pos[i].y + bh / 2, bw, bh, {
          chamfer: { radius: Math.min(bh / 2 - 1, 16) }, restitution: 0.55, friction: 0, frictionAir: 0.08, inertia: Infinity });
        b.plugin = { phase: jitter(i) * Math.PI * 2 };
        return b;
      });
      const T = 80, cx = (zone.left + zone.right) / 2, cy = (zone.top + zone.bottom) / 2;
      const zw = zone.right - zone.left + 2 * T, zh = zone.bottom - zone.top + 2 * T;
      Composite.add(engine.world, [...w.bodies,
        Bodies.rectangle(cx, zone.top - T / 2, zw, T, { isStatic: true }),
        Bodies.rectangle(cx, zone.bottom + T / 2, zw, T, { isStatic: true }),
        Bodies.rectangle(zone.left - T / 2, cy, T, zh, { isStatic: true }),
        Bodies.rectangle(zone.right + T / 2, cy, T, zh, { isStatic: true })]);
      w.live = true; w.last = 0; w.t0 = performance.now(); root.dataset.live = "";
      const tick = now => {
        if (!w.engine) return;
        const dt = w.last ? Math.min(32, now - w.last) : 16; w.last = now;
        const t = (now - w.t0) / 1000, k = o.drift * 0.00005 * Math.min(1, t / 2);
        w.bodies.forEach((b, i) => {
          if (w.drag && w.drag.i === i) return;
          const ph = b.plugin.phase;
          Body.applyForce(b, b.position, { x: Math.sin(t * 0.9 + ph) * k * b.mass, y: Math.cos(t * 1.3 + ph * 1.7) * k * b.mass });
        });
        Engine.update(w.engine, dt);
        w.bodies.forEach((b, i) => {
          pills[i].style.setProperty("--x", `${b.position.x.toFixed(1)}px`);
          pills[i].style.setProperty("--y", `${(b.position.y - w.sizes[i].h / 2).toFixed(1)}px`);
        });
        w.raf = requestAnimationFrame(tick);
      };
      w.raf = requestAnimationFrame(tick);
    }

    function set(next) {
      if (!next) stopPhysics();
      if (open === next) return;
      open = next;
      if (open) { place(); root.dataset.open = ""; } else delete root.dataset.open;
      trig.setAttribute("aria-expanded", String(open));
      pills.forEach(el => { el.tabIndex = open ? 0 : -1; el.setAttribute("aria-hidden", String(!open)); });
      o.onOpenChange && o.onOpenChange(open);
      clearTimeout(liveTimer);
      if (open && o.physics && Matter && !reduce.matches) {
        liveTimer = setTimeout(startPhysics, o.openDuration + (n - 1) * o.stagger + 80);
      }
    }

    function pick(item, i) {
      o.onSelect && o.onSelect(item.value, i, item);
      clearTimeout(popTimer);
      pills[i].dataset.pop = "";
      popTimer = setTimeout(() => delete pills[i].dataset.pop, 320);
      if (o.closeOnSelect) setTimeout(() => set(false), 180);
    }

    const pointerAt = e => { const r = anchor.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    pills.forEach((el, i) => {
      const item = list[i];
      el.addEventListener("pointerdown", e => {
        if (!w.live || e.button !== 0) return;
        const b = w.bodies[i]; if (!b) return;
        const p = pointerAt(e);
        w.drag = { i, id: e.pointerId, dx: b.position.x - p.x, dy: b.position.y - p.y, sx: e.clientX, sy: e.clientY, moved: false };
        try { el.setPointerCapture(e.pointerId); } catch (_) {}
      });
      el.addEventListener("pointermove", e => {
        const d = w.drag;
        if (!d || d.i !== i || d.id !== e.pointerId) return;
        if (!d.moved && Math.hypot(e.clientX - d.sx, e.clientY - d.sy) >= DRAG_MIN) { d.moved = true; el.dataset.drag = ""; }
        if (!d.moved) return;
        const b = w.bodies[i], { w: bw, h: bh } = w.sizes[i], z = w.zone, p = pointerAt(e);
        const x = Math.min(z.right - bw / 2, Math.max(z.left + bw / 2, p.x + d.dx));
        const y = Math.min(z.bottom - bh / 2, Math.max(z.top + bh / 2, p.y + d.dy));
        Matter.Body.setVelocity(b, { x: (x - b.position.x) * 0.6, y: (y - b.position.y) * 0.6 });
        Matter.Body.setPosition(b, { x, y });
      });
      const up = e => {
        const d = w.drag;
        if (!d || d.i !== i || d.id !== e.pointerId) return;
        w.drag = null; delete el.dataset.drag;
        try { el.releasePointerCapture(e.pointerId); } catch (_) {}
        if (!d.moved && e.type === "pointerup") pick(item, i);
      };
      el.addEventListener("pointerup", up);
      el.addEventListener("pointercancel", up);
      el.addEventListener("click", e => { if (!w.live || e.detail === 0) pick(item, i); });
      el.addEventListener("pointerenter", () => o.onPeek && o.onPeek(item, i));
      el.addEventListener("focus", () => o.onPeek && o.onPeek(item, i));
    });

    if (o.trigger === "hover") {
      root.addEventListener("pointerenter", () => set(true));
      root.addEventListener("pointerleave", () => { if (!w.drag) set(false); });
      root.addEventListener("focusin", () => set(true));
      root.addEventListener("focusout", e => { if (!root.contains(e.relatedTarget)) set(false); });
    }
    trig.addEventListener("click", () => set(!open));
    root.addEventListener("keydown", e => { if (e.key === "Escape" && open) { e.stopPropagation(); set(false); trig.focus(); } });

    if (o.defaultOpen) set(true);
    return { root, open: () => set(true), close: () => set(false), destroy: () => { stopPhysics(); root.remove(); } };
  }

  window.FolderFloat = FolderFloat;
})();
