/* ATLAS console frontend — vanilla JS. Talks only to the FastAPI backend (/api/*).
   Layout: the map is the canvas; the console (overview) floats glass panels over
   it; the other views open as one large glass workspace over the same map. */
"use strict";

const $ = (s, el) => (el || document).querySelector(s);
const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const latFmt = (v) => `${fmt(Math.abs(v), 2)}°${v < 0 ? "S" : "N"}`;
const lonFmt = (v) => `${fmt(Math.abs(v), 2)}°${v < 0 ? "W" : "E"}`;

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

/* ------------------------------------------------------------------ state */
const state = {
  bases: [], sites: [], winners: [], health: null,
  base: "Shackleton", preset: "balanced", weights: null,
  ranking: null, map: null, siteLayer: null, focusLayer: null, tileDark: null, tileGlobe: null,
  log: [], logTab: "all",
  compareLoaded: false, marsLoaded: false, methodologyLoaded: false,
};
const PRESET_WEIGHTS = {
  balanced: { slope: 30, flat: 20, temperature: 20, soil: 15, relief: 15 },
  rover:    { slope: 45, flat: 35, temperature: 5,  soil: 5,  relief: 10 },
  habitat:  { slope: 10, flat: 35, temperature: 20, soil: 10, relief: 25 },
  isru:     { slope: 15, flat: 10, temperature: 25, soil: 35, relief: 15 },
};
const PRESET_LABELS = { balanced: "BALANCED", rover: "ROVER MOBILITY", habitat: "HABITAT & STRUCTURES", isru: "SCIENCE & ISRU" };
const PILLAR_LABELS = { slope: "Slope", flat: "Flat area", temperature: "Temperature", soil: "Soil proxy", relief: "Relief" };
const PHASES = { overview: "CONSOLE", bases: "BASE SURVEY", finder: "RANKING", compare: "COMPARISON", mars: "MARS SURVEY", mllab: "ML LAB", methodology: "DOCUMENTATION" };
const BASE_NOTES = {
  "Peary": "North-pole base — processed from the 80N/75N LOLA grids and the Diviner poln temperature tables.",
  "Shackleton": "IAU centre 89.67°S 129.78°E — the pole itself sits on the rim; a third of the study area never sees the Sun.",
  "de Gerlache": "Rim site with the warmest seasonal mean of the four bases — still −184.4 °C.",
  "Amundsen": "Largest study area (60 km radius), measured from the wider 80S mosaic — it lies outside the 87–90°S high-resolution cap.",
};

/* ------------------------------------------------------------- session clock */
const T0 = Date.now();
const elapsed = () => {
  const s = Math.floor((Date.now() - T0) / 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(Math.floor(s / 3600))}:${p(Math.floor((s % 3600) / 60))}:${p(s % 60)}`;
};
setInterval(() => { const el = $("#topTimer"); if (el) el.textContent = elapsed(); }, 1000);

/* --------------------------------------------------------------- mission log */
function addLog(type, text) {
  state.log.unshift({ type, text, ts: elapsed() });
  if (state.log.length > 40) state.log.pop();
  renderLog();
}
function renderLog() {
  const box = $("#missionLog");
  if (!box) return;
  const alerts = state.log.filter((e) => e.type === "alert").length;
  $("#alertCount").textContent = alerts;
  const rows = state.log.filter((e) =>
    state.logTab === "all" ? true : e.type === state.logTab);
  box.innerHTML = rows.length ? rows.map((e) => `
    <div class="lrow ${e.type === "alert" ? "alert" : ""}">
      <span class="lt ${e.type}">${e.type.toUpperCase()}</span>
      <span class="lts mono">${e.ts}</span>
      <span class="ltx">${esc(e.text)}</span>
    </div>`).join("") : `<p class="loading">No entries in this tab yet.</p>`;
}
$$("#logTabs .ltab").forEach((b) => b.addEventListener("click", () => {
  $$("#logTabs .ltab").forEach((x) => x.classList.toggle("on", x === b));
  state.logTab = b.dataset.logtab;
  renderLog();
}));

/* ---------------------------------------------------------------- navigation */
function showPage(name) {
  $("#stage").dataset.view = name;
  $$(".view").forEach((p) => p.classList.toggle("active", p.id === `page-${name}`));
  $$(".ritem[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === name));
  $("#topPhase").textContent = PHASES[name] || "CONSOLE";
  if (name === "compare") initCompare();
  if (name === "mars") initMars();
  if (name === "mllab" && window.initMlLab) initMlLab();
  if (name === "methodology") initMethodology();
  if (state.map) setTimeout(() => state.map.invalidateSize(), 80);
}
function route() {
  const name = (location.hash || "#overview").slice(1);
  const valid = ["overview", "bases", "finder", "compare", "mars", "mllab", "methodology"];
  showPage(valid.includes(name) ? name : "overview");
}
window.addEventListener("hashchange", route);
$("#railData").addEventListener("click", () => {
  setTimeout(() => { const d = $("#downloads"); if (d) d.scrollIntoView({ behavior: "smooth", block: "center" }); }, 350);
});

/* ------------------------------------------------------- operator guide */
const infoWrap = $("#infoWrap"), infoBtn = $("#infoBtn");
infoBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  const open = infoWrap.classList.toggle("open");
  infoBtn.setAttribute("aria-expanded", open);
  infoBtn.setAttribute("aria-label", open ? "Close user guide" : "Open user guide");
});
document.addEventListener("click", (e) => {
  if (!infoWrap.contains(e.target)) {
    infoWrap.classList.remove("open");
    infoBtn.setAttribute("aria-expanded", "false");
  }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    infoWrap.classList.remove("open");
    infoBtn.setAttribute("aria-expanded", "false");
  }
});

/* --------------------------------------------------------------------- map */
function initMap() {
  const box = $("#map");
  if (typeof L === "undefined") { $("#mapFallback").hidden = false; return; }
  try {
    state.map = L.map("map", {
      zoomControl: false, worldCopyJump: true,
      dragging: true,            /* pan with the left mouse button / touch */
      scrollWheelZoom: true,     /* zoom in/out at the cursor position */
      doubleClickZoom: true, boxZoom: true, keyboard: true,
      preferCanvas: true,        /* rings/paths on canvas — faster than SVG */
    }).setView([24, 6], 2);
    /* Google map tiles via the free embedded tile service — no API key needed.
       MAP = Google roadmap, rendered dark by a CSS filter on the tile pane.
       GLOBE = Google hybrid satellite (imagery + labels). */
    state.tileDark = L.tileLayer("https://mt{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
      { attribution: "Map data © Google", maxZoom: 18, subdomains: ["0", "1", "2", "3"], updateWhenZooming: false, keepBuffer: 2 });
    state.tileGlobe = L.tileLayer("https://mt{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
      { attribution: "Imagery © Google", maxZoom: 18, subdomains: ["0", "1", "2", "3"], updateWhenZooming: false, keepBuffer: 2 });
    state.tileDark.addTo(state.map);
    state.map.getContainer().classList.add("dark-tiles");
    state.siteLayer = L.layerGroup().addTo(state.map);
    state.focusLayer = L.layerGroup().addTo(state.map);
    let tileErrs = 0;
    state.tileDark.on("tileerror", () => { if (++tileErrs > 6) $("#mapFallback").hidden = false; });
    state.tileDark.on("tileload", () => { tileErrs = 0; $("#mapFallback").hidden = true; });
    state.tileGlobe.on("tileerror", () => { if (++tileErrs > 6) $("#mapFallback").hidden = false; });
    state.tileGlobe.on("tileload", () => { tileErrs = 0; $("#mapFallback").hidden = true; });
  } catch (e) { $("#mapFallback").hidden = false; }
}
function setTileMode(globe) {
  if (!state.map) return;
  $("#tglMap").classList.toggle("on", !globe);
  $("#tglGlobe").classList.toggle("on", globe);
  state.map.getContainer().classList.toggle("dark-tiles", !globe);
  if (globe) { state.map.removeLayer(state.tileDark); state.tileGlobe.addTo(state.map); }
  else { state.map.removeLayer(state.tileGlobe); state.tileDark.addTo(state.map); }
}
$("#tglMap").addEventListener("click", () => setTileMode(false));
$("#tglGlobe").addEventListener("click", () => setTileMode(true));
$("#zoomIn").addEventListener("click", () => state.map && state.map.zoomIn());
$("#zoomOut").addEventListener("click", () => state.map && state.map.zoomOut());

function renderMapFocus(data) {
  if (!state.map || !data) return;
  state.siteLayer.clearLayers();
  state.focusLayer.clearLayers();
  if (!state.mapFitted && data.results.length) {
    state.mapFitted = true;
    const bounds = L.latLngBounds(data.results.map((r) => [r.lat, r.lon]));
    if (window.innerWidth > 1020) {
      state.map.fitBounds(bounds, { paddingTopLeft: [345, 75], paddingBottomRight: [415, 105], maxZoom: 3 });
    } else {
      state.map.fitBounds(bounds, { padding: [18, 18], maxZoom: 3 });
    }
  }
  const res = data.results;
  res.forEach((r) => {
    const is1 = r.display_rank === 1, is2 = r.display_rank === 2;
    const color = is1 ? "#ff7a1a" : is2 ? "#4fe3c1" : "#9aa7b1";
    L.circleMarker([r.lat, r.lon], {
      radius: is1 ? 8 : is2 ? 7 : 5, color, weight: 2,
      fillColor: color, fillOpacity: is1 || is2 ? 0.9 : 0.55,
    }).bindPopup(`<b>${esc(r.earth_site)}</b> (${esc(r.country)})<br>${esc(r.terrain_type)}<br>` +
      `Analog Score: <b>${fmt(r.display_score)}</b> · Rank #${r.display_rank} for ${esc(data.base)}<br>` +
      `Slope: ${fmt(r.slope_mean_deg, 2)}° · Temp: ${fmt(r.temp_mean_c)} °C`)
      .addTo(state.siteLayer);
  });
  const top = res.slice(0, 3);
  if (top.length) {
    const [a, b] = top;
    L.circle([a.lat, a.lon], { radius: 430000, color: "#ff7a1a", weight: 2, fillOpacity: 0.05 }).addTo(state.focusLayer);
    L.circle([a.lat, a.lon], { radius: 680000, color: "#ff7a1a", weight: 1, dashArray: "5 7", fillOpacity: 0 }).addTo(state.focusLayer);
    L.marker([a.lat + 7.5, a.lon], { interactive: false,
      icon: L.divIcon({ className: "", html: `<span class="map-label">BEST MATCH · WP1</span>` }) }).addTo(state.focusLayer);
    if (b) {
      L.circle([b.lat, b.lon], { radius: 330000, color: "#4fe3c1", weight: 2, fillOpacity: 0.05 }).addTo(state.focusLayer);
      L.marker([b.lat + 6, b.lon], { interactive: false,
        icon: L.divIcon({ className: "", html: `<span class="map-label teal">RUNNER-UP</span>` }) }).addTo(state.focusLayer);
    }
    if (top.length > 2) {
      L.polyline(top.map((r) => [r.lat, r.lon]),
        { color: "#8b96a1", weight: 1.5, dashArray: "4 6", opacity: 0.8 }).addTo(state.focusLayer);
    }
  }
}

/* ------------------------------------------------------------ base selection */
function baseObj(name) { return state.bases.find((b) => b.base_name === name); }

function renderSelectedBase() {
  const b = baseObj(state.base);
  if (!b) return;
  $("#selBaseChip").textContent = `${b.pole === "N" ? "N POLE" : "S POLE"} · RANKED`;
  $("#selBaseName").textContent = `${b.base_name} Crater`;
  $("#selBaseCoords").textContent = `${latFmt(b.lat)} · ${lonFmt(b.lon)} · LRO LOLA / DIVINER`;
  $("#selBaseStats").innerHTML = [
    ["PSR area", `${fmt(b.psr_pct)}%`],
    ["Sun vis", `${fmt(b.sun_visibility_pct)}%`],
    ["Mean temp", `${fmt(b.temp_seasonal_mean_c, 0)} °C`],
  ].map(([k, v]) => `<div class="sel-stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  $("#selBaseNote").textContent = BASE_NOTES[b.base_name] || "";
  $$("#baseCards .bcard").forEach((c) => c.classList.toggle("on", c.dataset.base === b.base_name));
  $$("#baseSeg button").forEach((x) => x.classList.toggle("on", x.dataset.base === b.base_name));
  const sel = $("#topBase"); if (sel && sel.value !== b.base_name) sel.value = b.base_name;
}

function renderBaseCards() {
  $("#baseCards").innerHTML = state.bases.map((b) => `
    <button type="button" class="glass bcard ${b.base_name === state.base ? "on" : ""}" data-base="${esc(b.base_name)}">
      <div class="n">${esc(b.base_name.toUpperCase())} · ${b.pole}</div>
      <div class="t mono">${fmt(b.temp_seasonal_mean_c, 0)} °C</div>
      <div class="bbar"><i style="width:${Math.min(100, Math.max(4, b.sun_visibility_pct * 4))}%"></i></div>
      <div class="s mono">SUN ${fmt(b.sun_visibility_pct)}%</div>
    </button>`).join("");
  $$("#baseCards .bcard").forEach((c) =>
    c.addEventListener("click", () => setBase(c.dataset.base, true)));
}

function setBase(name, announce) {
  if (!baseObj(name)) return;
  state.base = name;
  renderSelectedBase();
  if (announce) addLog("mission", `Target base set to ${name} — ranking recomputed`);
  loadRanking();
}

/* ------------------------------------------------------------------ ranking */
async function loadRanking() {
  const params = new URLSearchParams({ base: state.base, preset: state.preset });
  if (state.weights) for (const k of Object.keys(PILLAR_LABELS)) params.set(k, state.weights[k]);
  try {
    const data = await api(`/api/rankings?${params}`);
    state.ranking = data;
    $("#topPreset").textContent = state.weights ? "CUSTOM" : PRESET_LABELS[state.preset];
    renderConsoleRanking(data);
    renderFinderList(data);
    renderFeed(data);
    renderMapFocus(data);
  } catch (err) {
    $("#consoleRanking").innerHTML = `<p class="loading">Ranking unavailable (${esc(err.message)})</p>`;
    addLog("alert", `Ranking request failed for ${state.base} — backend unreachable`);
  }
}

function statusOf(r) {
  if (r.sand_pct === null || r.sand_pct === undefined) return { cls: "o", txt: "Soil gap" };
  if (r.display_rank === 1) return { cls: "", txt: "Best match" };
  return { cls: "t", txt: "Ranked" };
}

function renderConsoleRanking(data) {
  $("#rankPanelTitle").textContent = `Analog Ranking — ${data.base}`;
  $("#consoleRanking").innerHTML = data.results.slice(0, 4).map((r) => {
    const st = statusOf(r);
    return `<div class="frow">
      <span class="fid">${String(r.display_rank).padStart(2, "0")} · ${esc(r.earth_site)}</span>
      <span class="ft mono">${fmt(r.slope_mean_deg, 1)}° · ${fmt(r.pct_slope_under_10deg, 0)}%</span>
      <span class="dotst"><span class="dot ${st.cls}"></span>${st.txt}</span>
    </div>`;
  }).join("");
}

function renderFinderList(data) {
  const banner = $("#customBanner");
  if (data.official) { banner.hidden = true; }
  else {
    banner.hidden = false;
    banner.textContent = "Custom ranking — weights: " + Object.keys(PILLAR_LABELS)
      .map((k) => `${PILLAR_LABELS[k]} ${data.weights[k]}`).join(" · ") +
      ". The official published ranking uses the Balanced profile.";
  }
  const notes = Object.fromEntries(state.sites.map((s) => [s.site_name, s]));
  $("#rankList").innerHTML = data.results.map((r) => {
    const s = notes[r.earth_site];
    const gap = r.sand_pct === null || r.sand_pct === undefined;
    return `<button type="button" class="rrow ${r.display_rank === 1 ? "top1" : ""}" data-site="${esc(r.earth_site)}">
      <span class="rrank mono">${String(r.display_rank).padStart(2, "0")}</span>
      <span>
        <span class="rname">${esc(r.earth_site)}</span><span class="rtag ${gap ? "gap" : ""}">${gap ? "SOIL GAP" : esc(r.terrain_type).toUpperCase()}</span>
        <span class="rmeta mono">${esc(r.country).toUpperCase()} · SLOPE ${fmt(r.slope_mean_deg, 2)}° · FLAT&lt;10° ${fmt(r.pct_slope_under_10deg)}% · TEMP ${fmt(r.temp_mean_c)} °C · SEASON ±${fmt(r.temp_seasonal_range_c)} °C · SOIL ${gap ? "NO DATA" : `${fmt(r.sand_pct, 0)}/${fmt(r.clay_pct, 0)}/${fmt(r.coarse_fragments_pct, 0)}`}</span>
        ${s && s.analog_note ? `<span class="rnote" hidden>${esc(s.analog_note)}</span>` : ""}
      </span>
      <span class="rscore">
        <span class="rsv mono">${fmt(r.display_score)}</span>
        <span class="rbar"><i style="width:${Math.min(100, Math.max(0, r.display_score || 0))}%"></i></span>
        <span class="rregime">${esc(r.regime || "")}</span>
      </span>
    </button>`;
  }).join("");
  $$("#rankList .rrow").forEach((row) => row.addEventListener("click", () => {
    const note = $(".rnote", row);
    if (note) note.hidden = !note.hidden;
  }));
}

/* --------------------------------------------------------------------- feed */
function tape(el, center, step, dec, label) {
  if (!el || center === null || center === undefined) { if (el) el.innerHTML = ""; return; }
  const vals = [center + 2 * step, center + step, center, center - step, center - 2 * step];
  el.innerHTML = vals.map((v, i) =>
    i === 2 ? `<span class="hl">${fmt(v, dec)}°</span>` : `<span>${fmt(v, dec)}°</span>`).join("") +
    `<span class="tk">${label}</span>`;
}
function renderFeed(data) {
  const top = data.results[0];
  const b = baseObj(data.base);
  if (!top || !b) return;
  const site = state.sites.find((s) => s.site_name === top.earth_site) || {};
  $("#feedTitle").textContent = `Site Feed — ${top.earth_site}`;
  const img = $("#feedImg");
  img.src = b.hillshade_url;
  img.alt = `Shaded relief of ${b.base_name} crater (LOLA)`;
  tape($("#tapeSlope"), top.slope_mean_deg, 3, 1, "SLOPE");
  tape($("#tapeTemp"), top.temp_mean_c, 6, 0, "TEMP");
  $("#feedScore").textContent = fmt(top.display_score);
  $("#feedSite").textContent = `#1 for ${data.base}${data.official ? "" : " · custom weights"}`;
  $("#feedCap").textContent = `BASE TERRAIN — ${data.base.toUpperCase()} · LOLA HILLSHADE`;
  $("#feedFoot").innerHTML =
    `<span>${latFmt(top.lat)} · ${lonFmt(top.lon)}</span>` +
    `<span>${esc(top.country).toUpperCase()}</span>` +
    `<span>ELEV ${site.elev_mean_m != null ? fmt(site.elev_mean_m, 0) + " M" : "—"}</span>`;
}

/* --------------------------------------------------------------- model health */
function renderHealth() {
  const n = state.sites.length || 14;
  const cov = (key) => state.sites.filter((s) => s[key] !== null && s[key] !== undefined).length;
  const cells = [
    ["Slope", cov("slope_mean_deg")], ["Flat", cov("pct_slope_under_10deg")],
    ["Temp", cov("temp_mean_c")], ["Soil", cov("sand_pct")], ["Relief", cov("elev_std_m")],
  ];
  $("#modelHealth").innerHTML = cells.map(([k, c]) => {
    const score = Math.round((10 * c) / n);
    return `<div class="hcell"><span class="dot ${score >= 10 ? "" : "o"}"></span>
      <div class="k">${k}</div><div class="v">${score}/10</div></div>`;
  }).join("");
}

/* -------------------------------------------------------------------- bases */
function renderBaseProfiles() {
  const rows = [
    ["Mean elevation", "elev_mean_m", " m", 1], ["Relief — elevation std", "elev_std_m", " m", 1],
    ["Mean slope", "slope_mean_deg", "°", 2], ["Area flatter than 10°", "pct_slope_under_10deg", "%", 1],
    ["Surface roughness (mean)", "roughness_mean_m", " m", 2],
    ["Permanently shadowed area", "psr_pct", "%", 1],
    ["Sun visibility", "sun_visibility_pct", "%", 1], ["Earth visibility", "earth_visibility_pct", "%", 1],
    ["Mean temperature (seasonal)", "temp_seasonal_mean_c", " °C", 1],
    ["Winter cold level (1st pct)", "temp_winter_cold_p1_k", " K", 1],
  ];
  $("#baseProfiles").innerHTML = state.bases.map((b) => `
    <article class="base-profile">
      <div>
        <h3>${esc(b.base_name)} <small>${latFmt(b.lat)}, ${lonFmt(b.lon)} · ${fmt(b.crater_diameter_km, 0)} km crater · ${fmt(b.roi_radius_km, 0)} km study area</small></h3>
        ${rows.map(([label, key, unit, d]) =>
          `<div class="metric-line"><span>${label}</span><b>${fmt(b[key], d)}${unit}</b></div>`).join("")}
        <p class="sources">Sources: ${esc(b.sources)}</p>
      </div>
      <figure>
        <img src="${esc(b.hillshade_url)}" alt="Shaded relief map of ${esc(b.base_name)} crater" loading="lazy">
        <figcaption>Shaded relief (LOLA DEM) — ${esc(b.base_name)}</figcaption>
      </figure>
    </article>`).join("") +
    `<div class="note-box"><strong>Validation, not decoration:</strong> before discovering anything, the
     engine ran on the classic catalogued analogs and — untuned — ranked McMurdo Dry Valleys #1 for three
     bases and Atacama #1 for Peary, the literature's own answers. Those sites were then removed and the
     14 new sites ranked. Full story in the Methodology.</div>`;
}

/* ------------------------------------------------------------------- finder */
function buildSliders(weights) {
  $("#weightSliders").innerHTML = Object.keys(PILLAR_LABELS).map((k) => `
    <div class="slider-row">
      <label for="w-${k}">${PILLAR_LABELS[k]} <output id="wo-${k}">${weights[k]}</output></label>
      <input type="range" id="w-${k}" min="0" max="60" step="1" value="${weights[k]}" data-pillar="${k}">
    </div>`).join("");
  $$("#weightSliders input[type=range]").forEach((inp) => {
    inp.addEventListener("input", () => {
      $(`#wo-${inp.dataset.pillar}`).textContent = inp.value;
      state.weights = currentSliderWeights();
      debounceRanking();
    });
  });
}
function currentSliderWeights() {
  const w = {};
  $$("#weightSliders input[type=range]").forEach((i) => (w[i.dataset.pillar] = Number(i.value)));
  return w;
}
let rankTimer = null;
function debounceRanking() { clearTimeout(rankTimer); rankTimer = setTimeout(loadRanking, 180); }

function initFinderControls() {
  const seg = $("#baseSeg");
  seg.innerHTML = state.bases.map((b) =>
    `<button type="button" data-base="${esc(b.base_name)}" class="${b.base_name === state.base ? "on" : ""}">${esc(b.base_name)}</button>`).join("");
  $$("#baseSeg button").forEach((x) => x.addEventListener("click", () => setBase(x.dataset.base, true)));
  $$("#presetChips .chipb").forEach((chip) => chip.addEventListener("click", () => {
    $$("#presetChips .chipb").forEach((c) => c.classList.toggle("active", c === chip));
    state.preset = chip.dataset.preset;
    state.weights = null;
    buildSliders(PRESET_WEIGHTS[state.preset]);
    addLog("mission", `${PRESET_LABELS[state.preset]} preset applied — 14 sites re-ranked for ${state.base}`);
    loadRanking();
  }));
  buildSliders(PRESET_WEIGHTS.balanced);
}

/* ------------------------------------------------------------------ compare */
async function initCompare() {
  if (state.compareLoaded || !state.bases.length) return;
  state.compareLoaded = true;
  $("#cmpBase").innerHTML = state.bases.map((b) => `<option>${esc(b.base_name)}</option>`).join("");
  $("#cmpSite").innerHTML = state.sites.map((s) => `<option>${esc(s.site_name)}</option>`).join("");
  const top = state.winners.find((w) => w.moon_base === $("#cmpBase").value);
  if (top) $("#cmpSite").value = top.earth_site;
  $("#cmpBase").addEventListener("change", loadCompare);
  $("#cmpSite").addEventListener("change", loadCompare);
  loadCompare();
}
async function loadCompare() {
  const base = $("#cmpBase").value, site = $("#cmpSite").value;
  try {
    const d = await api(`/api/compare?base=${encodeURIComponent(base)}&site=${encodeURIComponent(site)}`);
    $("#compareHead").innerHTML =
      `<span class="big">${fmt(d.analog_score)}</span>
       <span><b>${esc(site)}</b> for <b>${esc(base)}</b><br>
       <span style="color:var(--mut);font-size:.8rem">Rank #${d.rank_for_base} of ${d.of_sites} novel sites · official weights</span></span>`;
    $("#cmpBaseName").textContent = base;
    $("#cmpSiteName").textContent = site;
    const rows = [
      ["Mean slope (°)", d.moon.slope_mean_deg, d.earth.slope_mean_deg, 2],
      ["Area flatter than 10° (%)", d.moon.pct_slope_under_10deg, d.earth.pct_slope_under_10deg, 1],
      ["Mean temperature (°C)", d.moon.temp_seasonal_mean_c, d.earth.temp_mean_c, 1],
      ["Relief — elevation std (m)", d.moon.elev_std_m, d.earth.elev_std_m, 1],
      ["Mean elevation (m)", d.moon.elev_mean_m, d.earth.elev_mean_m, 1],
    ];
    $("#compareTable tbody").innerHTML = rows.map(([l, m, e, dec]) =>
      `<tr><td>${l}</td><td><b>${fmt(m, dec)}</b></td><td><b>${fmt(e, dec)}</b></td></tr>`).join("") +
      (d.earth.analog_note ? `<tr><td colspan="3" style="white-space:normal;color:var(--mut)">
        <b style="color:var(--txt)">Why this site is new:</b> ${esc(d.earth.analog_note)}</td></tr>` : "");
    drawRadar(d);
  } catch (err) {
    $("#compareHead").innerHTML = `<p class="loading">Failed to load (${esc(err.message)})</p>`;
  }
}
function drawRadar(d) {
  const keys = ["slope", "flat", "temperature", "relief", "soil"];
  const labels = ["Slope", "Flat area", "Temperature", "Relief", "Soil proxy"];
  const cx = 170, cy = 160, R = 108;
  const pt = (i, v) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / keys.length;
    return [cx + R * v * Math.cos(a), cy + R * v * Math.sin(a)];
  };
  let svg = `<svg viewBox="0 0 340 320" role="img" aria-label="Radar chart of per-factor similarity">`;
  [0.25, 0.5, 0.75, 1].forEach((ring) => {
    const pts = keys.map((_, i) => pt(i, ring).join(",")).join(" ");
    svg += `<polygon points="${pts}" fill="none" stroke="rgba(255,255,255,.13)" stroke-width="1"/>`;
  });
  keys.forEach((_, i) => {
    const [x, y] = pt(i, 1);
    svg += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="rgba(255,255,255,.13)"/>`;
  });
  const vals = keys.map((k) => (d.radar_similarity[k] === null ? 0.5 : d.radar_similarity[k]));
  const poly = vals.map((v, i) => pt(i, v).join(",")).join(" ");
  svg += `<polygon points="${poly}" fill="rgba(255,122,26,.22)" stroke="#ff7a1a" stroke-width="2"/>`;
  vals.forEach((v, i) => {
    const [x, y] = pt(i, v);
    const neutral = d.radar_similarity[keys[i]] === null;
    svg += `<circle cx="${x}" cy="${y}" r="4" fill="${neutral ? "#0d1218" : "#4fe3c1"}" stroke="#4fe3c1" stroke-width="2"/>`;
  });
  labels.forEach((lab, i) => {
    const [x, y] = pt(i, 1.18);
    svg += `<text x="${x}" y="${y}" fill="#98a3ad" font-size="12" font-weight="600" text-anchor="middle">${lab}</text>`;
  });
  svg += `</svg>`;
  const missing = keys.filter((k) => d.radar_similarity[k] === null);
  $("#radarBox").innerHTML = `<h3>Per-factor similarity — ${esc(d.base)} vs ${esc(d.site)}</h3>` +
    svg + `<p class="radar-note">1 = identical on that factor.${missing.length ?
      ` Hollow marker: ${missing.map((k) => PILLAR_LABELS[k]).join(", ")} not scored for this site (no data coverage), drawn neutral.` : ""}</p>`;
}

/* ---------------------------------------------------------------------- mars */
async function initMars() {
  if (state.marsLoaded) return;
  state.marsLoaded = true;
  try {
    const d = await api("/api/mars");
    const r = d.region;
    $("#marsBox").innerHTML = `
      <div class="mcard">
        <h3>${esc(r.mars_region)} <small>${fmt(r.lat, 2)}° N, ${fmt(r.lon, 2)}° E</small></h3>
        <div class="metric-line"><span>Mean elevation</span><b>${fmt(r.elev_mean_m)} m</b></div>
        <div class="metric-line"><span>Relief — elevation std</span><b>${fmt(r.elev_std_m)} m</b></div>
        <div class="metric-line"><span>Mean slope</span><b>${fmt(r.slope_mean_deg, 2)}°</b></div>
        <div class="metric-line"><span>Area flatter than 5°</span><b>${fmt(r.pct_slope_under_5deg)}%</b></div>
        <div class="metric-line"><span>Area flatter than 10°</span><b>${fmt(r.pct_slope_under_10deg)}%</b></div>
        <p class="sources">Source: ${esc(r.source)}</p>
      </div>
      <h3 style="margin:16px 0 4px;font-size:15px">Which Earth sites move most like Jezero? (closest mean slope)</h3>
      <div class="mars-grid">${d.closest_earth_sites.map((s) => `
        <div class="mcard"><h3>${esc(s.site_name)}</h3>
          <div class="sources" style="margin:0 0 4px">${esc(s.country)}</div>
          <div class="metric-line"><span>Mean slope</span><b>${fmt(s.slope_mean_deg, 2)}°</b></div>
          <div class="metric-line"><span>Area flatter than 10°</span><b>${fmt(s.pct_slope_under_10deg)}%</b></div>
        </div>`).join("")}</div>`;
  } catch (err) {
    $("#marsBox").innerHTML = `<p class="loading">Failed to load (${esc(err.message)})</p>`;
  }
}

/* --------------------------------------------------------------- methodology */
async function initMethodology() {
  if (state.methodologyLoaded) return;
  state.methodologyLoaded = true;
  try {
    const r = await fetch("/api/methodology");
    $("#methodologyBody").innerHTML = await r.text();
  } catch (err) {
    $("#methodologyBody").innerHTML = `<p>Failed to load methodology (${esc(err.message)})</p>`;
  }
}

/* ---------------------------------------------------------------------- boot */
async function boot() {
  initMap();
  try {
    const [health, bases, sites, winners] = await Promise.all([
      api("/api/health"), api("/api/moon-bases"), api("/api/earth-sites"), api("/api/winners"),
    ]);
    state.health = health; state.bases = bases; state.sites = sites; state.winners = winners;
    /* warm the browser cache with the four LOLA hillshades during idle time,
       so switching bases (Site Feed + Compare) paints instantly */
    const warm = () => bases.forEach((b) => { const i = new Image(); i.src = b.hillshade_url; });
    if (window.requestIdleCallback) requestIdleCallback(warm, { timeout: 2500 }); else setTimeout(warm, 1200);
    $("#topScored").textContent = `${health.score_rows}/${bases.length * sites.length}`;
    $("#topBase").innerHTML = bases.map((b) => `<option>${esc(b.base_name)}</option>`).join("");
    $("#topBase").value = state.base;
    $("#topBase").addEventListener("change", (e) => setBase(e.target.value, true));
    renderBaseCards();
    renderSelectedBase();
    renderHealth();
    renderBaseProfiles();
    initFinderControls();
    addLog("info", `Pipeline outputs synced — ${health.score_rows}/${bases.length * sites.length} pair scores live`);
    addLog("info", "Diviner PCP polar temperature tables loaded (pols + poln)");
    sites.filter((s) => s.sand_pct === null || s.sand_pct === undefined)
      .forEach((s) => addLog("alert", `SoilGrids gap — ${s.site_name}: soil pillar not scored for this site`));
    addLog("mission", `Console initialised — target base ${state.base}`);
    loadRanking();
  } catch (err) {
    $("#consoleRanking").innerHTML = `<p class="loading">Could not load data (${esc(err.message)}). Is the backend running?</p>`;
    addLog("alert", "Backend unreachable — check /api/health");
  }
  route();
}
boot();
