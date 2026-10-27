// Deck de la charla PyConES 2026 — sin dependencias externas para que funcione
// sin red en la sala. Dos modos:
//   /pycon.html            slides (se escalan a la ventana)
//   /pycon.html?presenter  vista de ponente: notas, reloj, siguiente slide
// Ambas ventanas se sincronizan por BroadcastChannel (misma pestaña/navegador).

(() => {
  const DESIGN_W = 1280;
  const DESIGN_H = 720;
  const CHROME_H = 64; // toolbar + footer

  const isPresenter = new URLSearchParams(location.search).has("presenter");
  const slides = [...document.querySelectorAll(".slide")];
  const stage = document.getElementById("stage");
  const channel = "BroadcastChannel" in window ? new BroadcastChannel("osteolab-pycon") : null;

  let index = 0;
  let step = 0;

  const fragsOf = (i) => [...slides[i].querySelectorAll(".frag")];
  const titleOf = (i) => slides[i]?.dataset.title || `Slide ${i + 1}`;
  const notesOf = (i) => slides[i]?.querySelector(".notes")?.innerHTML.trim() || "<em>Sin notas.</em>";

  // "12:30" -> 750 s
  const parseAt = (s) => {
    if (!s) return null;
    const [m, sec] = s.split(":").map(Number);
    return m * 60 + (sec || 0);
  };
  const fmt = (secs) => {
    const sign = secs < 0 ? "-" : "";
    secs = Math.abs(Math.round(secs));
    return `${sign}${String(Math.floor(secs / 60)).padStart(2, "0")}:${String(secs % 60).padStart(2, "0")}`;
  };

  // ------------------------------------------------------------ resaltado --
  const PY_KEYWORDS = new Set((
    "False None True and as assert async await break class continue def del elif else " +
    "except finally for from global if import in is lambda nonlocal not or pass raise " +
    "return try while with yield"
  ).split(" "));
  const TOKEN_RE = /("""[\s\S]*?"""|[rfb]?"(?:\\.|[^"\\\n])*"|[rfb]?'(?:\\.|[^'\\\n])*')|(#[^\n]*)|(@[\w.]+)|(\b\d[\d_]*(?:\.\d+)?\b)|([A-Za-z_]\w*)/g;
  const escapeHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function highlight(codeEl) {
    const src = codeEl.textContent;
    const isPy = codeEl.classList.contains("lang-py");
    let out = "";
    let last = 0;
    for (const m of src.matchAll(TOKEN_RE)) {
      out += escapeHtml(src.slice(last, m.index));
      last = m.index + m[0].length;
      const text = escapeHtml(m[0]);
      if (m[1]) out += `<span class="tok-str">${text}</span>`;
      else if (m[2]) out += `<span class="tok-com">${text}</span>`;
      else if (m[3] && isPy) out += `<span class="tok-dec">${text}</span>`;
      else if (m[4]) out += `<span class="tok-num">${text}</span>`;
      else if (m[5] && isPy && PY_KEYWORDS.has(m[0])) out += `<span class="tok-kw">${text}</span>`;
      else if (m[5] && isPy && /^\s*\(/.test(src.slice(last))) out += `<span class="tok-fn">${text}</span>`;
      else out += text;
    }
    out += escapeHtml(src.slice(last));
    codeEl.innerHTML = out;
  }
  document.querySelectorAll(".code code").forEach(highlight);

  // ------------------------------------------------------------ cronómetro --
  // Estado compartido: en localStorage (para sobrevivir a recargas) y por el
  // canal (para que la vista de ponente y las slides marquen lo mismo).
  const TIMER_KEY = "osteolab-pycon-timer";
  let timer = { startedAt: null, accumulated: 0 };
  try {
    const saved = JSON.parse(localStorage.getItem(TIMER_KEY) || "null");
    if (saved && typeof saved.accumulated === "number") timer = saved;
  } catch { /* storage bloqueado: cronómetro solo en memoria */ }

  const elapsed = () => timer.accumulated + (timer.startedAt ? (Date.now() - timer.startedAt) / 1000 : 0);

  function saveTimer(broadcast = true) {
    try { localStorage.setItem(TIMER_KEY, JSON.stringify(timer)); } catch { /* ignorar */ }
    if (broadcast) channel?.postMessage({ type: "timer", timer });
    renderTimer();
  }
  function toggleTimer() {
    if (timer.startedAt) {
      timer.accumulated = elapsed();
      timer.startedAt = null;
    } else {
      timer.startedAt = Date.now();
    }
    saveTimer();
  }
  function resetTimer() {
    timer = { startedAt: null, accumulated: 0 };
    saveTimer();
  }

  // Compara el tiempo transcurrido con el minuto previsto (data-at) de la
  // slide actual y de la siguiente.
  function paceText() {
    const t = elapsed();
    const planned = parseAt(slides[index].dataset.at);
    const nextPlanned = parseAt(slides[index + 1]?.dataset.at) ?? 35 * 60;
    if (planned == null || t === 0) return { text: planned == null ? "" : `plan ${fmt(planned)}`, cls: "" };
    if (t > nextPlanned) return { text: `plan ${fmt(planned)} · ${fmt(t - nextPlanned)} de retraso`, cls: "behind" };
    if (t < planned - 15) return { text: `plan ${fmt(planned)} · ${fmt(planned - t)} de margen`, cls: "ahead" };
    return { text: `plan ${fmt(planned)} · en tiempo`, cls: "" };
  }

  function renderTimer() {
    const label = `⏱ ${fmt(elapsed())}`;
    const pace = paceText();
    const timerBtn = document.getElementById("timer");
    if (timerBtn) {
      timerBtn.textContent = label;
      timerBtn.classList.toggle("running", !!timer.startedAt);
    }
    const paceEl = document.getElementById("paceInfo");
    if (paceEl) {
      paceEl.textContent = pace.text;
      paceEl.className = `pace ${pace.cls}`;
    }
    const pvClock = document.getElementById("pvClock");
    if (pvClock) pvClock.textContent = fmt(elapsed());
    const pvPace = document.getElementById("pvPace");
    if (pvPace) {
      pvPace.textContent = pace.text;
      pvPace.className = `pace ${pace.cls}`;
    }
    const pvToggle = document.getElementById("pvToggle");
    if (pvToggle) pvToggle.textContent = timer.startedAt ? "Pausar ⏸" : "Arrancar ⏱";
  }
  setInterval(renderTimer, 1000);

  // ------------------------------------------------------------ navegación --
  function render(broadcast = true) {
    slides.forEach((s, i) => s.classList.toggle("active", i === index));
    fragsOf(index).forEach((f, i) => f.classList.toggle("shown", i < step));

    const counter = document.getElementById("slideCounter");
    if (counter) counter.textContent = `${index + 1} / ${slides.length}`;
    const fill = document.getElementById("progressFill");
    if (fill) fill.style.width = `${((index + 1) / slides.length) * 100}%`;

    const notesTitle = document.getElementById("notesTitle");
    if (notesTitle) {
      notesTitle.textContent = `${index + 1}. ${titleOf(index)}`;
      document.getElementById("notesNext").textContent = slides[index + 1] ? `Siguiente: ${titleOf(index + 1)}` : "Última slide";
      document.getElementById("notesBody").innerHTML = notesOf(index);
    }
    renderPresenter();
    renderTimer();

    if (!isPresenter) history.replaceState(null, "", `#${index + 1}${step ? "." + step : ""}`);
    if (broadcast) channel?.postMessage({ type: "goto", index, step });
  }

  function goTo(i, s = 0) {
    index = Math.max(0, Math.min(slides.length - 1, i));
    step = Math.max(0, Math.min(fragsOf(index).length, s));
    render();
  }
  function next() {
    if (step < fragsOf(index).length) goTo(index, step + 1);
    else if (index < slides.length - 1) goTo(index + 1, 0);
  }
  function prev() {
    if (step > 0) goTo(index, step - 1);
    else if (index > 0) goTo(index - 1, fragsOf(index - 1).length);
  }

  channel?.addEventListener("message", (e) => {
    const msg = e.data;
    if (msg?.type === "goto") {
      index = msg.index;
      step = msg.step;
      render(false);
    } else if (msg?.type === "timer") {
      timer = msg.timer;
      saveTimer(false);
    } else if (msg?.type === "hello" && !isPresenter) {
      // la vista de ponente acaba de abrirse: que se ponga en la misma slide
      channel.postMessage({ type: "goto", index, step });
    }
  });

  // ------------------------------------------------------------ escalado --
  function fit() {
    if (isPresenter || window.innerWidth <= 760) {
      stage.style.transform = "";
      return;
    }
    const full = document.body.classList.contains("fullscreen");
    const availH = window.innerHeight - (full ? 0 : CHROME_H);
    const scale = Math.min(window.innerWidth / DESIGN_W, availH / DESIGN_H) * (full ? 1 : 0.97);
    stage.style.transform = `scale(${scale})`;
  }
  window.addEventListener("resize", fit);
  document.addEventListener("fullscreenchange", () => {
    document.body.classList.toggle("fullscreen", !!document.fullscreenElement);
    fit();
  });

  // ------------------------------------------------------------ extras --
  const notesPanel = document.getElementById("notesPanel");
  const overview = document.getElementById("overview");

  function toggleOverview(force) {
    const show = force ?? overview.hidden;
    overview.hidden = !show;
    if (!show) return;
    const grid = document.getElementById("overviewGrid");
    grid.innerHTML = "";
    slides.forEach((s, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "overview-item" + (i === index ? " current" : "");
      b.innerHTML = `<small>${i + 1} · ${s.dataset.at || ""}</small><span></span>`;
      b.querySelector("span").textContent = titleOf(i);
      b.addEventListener("click", () => { toggleOverview(false); goTo(i, 0); });
      grid.appendChild(b);
    });
    grid.querySelector(".current")?.focus();
  }

  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen?.();
  }

  function toggleTheme() {
    const root = document.documentElement;
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
  }

  function openPresenter() {
    window.open(`/pycon.html?presenter#${index + 1}`, "osteolab-pycon-presenter", "width=1000,height=720");
  }

  document.getElementById("timer")?.addEventListener("click", toggleTimer);

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (!overview.hidden) {
      if (e.key === "Escape" || e.key.toLowerCase() === "o") toggleOverview(false);
      return;
    }
    switch (e.key) {
      case "ArrowRight": case "ArrowDown": case "PageDown": case " ": case "Enter":
        e.preventDefault(); next(); break;
      case "ArrowLeft": case "ArrowUp": case "PageUp": case "Backspace":
        e.preventDefault(); prev(); break;
      case "Home": goTo(0); break;
      case "End": goTo(slides.length - 1, fragsOf(slides.length - 1).length); break;
      default:
        switch (e.key.toLowerCase()) {
          case "n": notesPanel.hidden = !notesPanel.hidden; break;
          case "o": toggleOverview(true); break;
          case "f": toggleFullscreen(); break;
          case "t": toggleTimer(); break;
          case "r": resetTimer(); break;
          case "d": toggleTheme(); break;
          case "p": if (!isPresenter) openPresenter(); break;
        }
    }
  });

  // Swipe en tablet (en móvil las slides van apiladas con scroll).
  let touchX = null;
  document.addEventListener("touchstart", (e) => { touchX = e.touches[0].clientX; }, { passive: true });
  document.addEventListener("touchend", (e) => {
    if (touchX == null || window.innerWidth <= 760) return;
    const dx = e.changedTouches[0].clientX - touchX;
    if (Math.abs(dx) > 50) (dx < 0 ? next : prev)();
    touchX = null;
  });

  // ------------------------------------------------------------ ponente --
  function buildPresenter() {
    document.body.classList.add("presenter");
    document.title = "Ponente · ¿De quién es este hueso?";
    const view = document.createElement("div");
    view.className = "presenter-view";
    view.innerHTML = `
      <div class="pv-top">
        <div><div class="pv-title" id="pvTitle"></div><div class="pv-next" id="pvNext"></div></div>
        <div style="text-align:right"><div class="pv-clock" id="pvClock">00:00</div><div id="pvPace" class="pace"></div></div>
      </div>
      <div class="pv-next" id="pvStep"></div>
      <div class="pv-notes" id="pvNotes"></div>
      <div class="pv-controls">
        <button type="button" id="pvPrev">← Anterior</button>
        <button type="button" class="primary" id="pvNextBtn">Siguiente →</button>
        <button type="button" id="pvToggle">Arrancar ⏱</button>
        <button type="button" id="pvReset">Reiniciar</button>
      </div>`;
    document.body.prepend(view);
    document.getElementById("pvPrev").addEventListener("click", prev);
    document.getElementById("pvNextBtn").addEventListener("click", next);
    document.getElementById("pvToggle").addEventListener("click", toggleTimer);
    document.getElementById("pvReset").addEventListener("click", resetTimer);
  }

  function renderPresenter() {
    if (!isPresenter) return;
    const frags = fragsOf(index).length;
    document.getElementById("pvTitle").textContent = `${index + 1} / ${slides.length} · ${titleOf(index)}`;
    document.getElementById("pvNext").textContent = slides[index + 1] ? `Siguiente: ${titleOf(index + 1)}` : "Última slide";
    document.getElementById("pvStep").textContent = frags ? `Elementos mostrados: ${step} / ${frags}` : "";
    document.getElementById("pvNotes").innerHTML = notesOf(index);
  }

  // ------------------------------------------------------------ arranque --
  if (isPresenter) buildPresenter();

  const m = location.hash.match(/^#(\d+)(?:\.(\d+))?/);
  if (m) {
    index = Math.max(0, Math.min(slides.length - 1, Number(m[1]) - 1));
    step = Math.min(fragsOf(index).length, Number(m[2] || 0));
  }
  fit();
  render(false);
  if (isPresenter) channel?.postMessage({ type: "hello" });
})();
