/* ATLAS static API shim.
   The console originally talked to a FastAPI backend (/api/*). This static
   build bundles a snapshot of that backend's own responses (static-data.js)
   and answers the same requests in the browser, so the site deploys as pure
   static files. The rankings re-weighting and compare logic below mirror
   webapp/backend/main.py exactly: with the Balanced preset the stored
   official scores are returned untouched; custom weights rescore with
   D = sqrt( sum(w_k * dist_k^2) / sum(w over pillars with data) ),
   score = 100 * exp(-D), and pillars without data are never scored. */
"use strict";

(function () {
  const D = window.ATLAS_STATIC;
  if (!D) return;
  const PRESETS = {
    balanced: { slope: 30, flat: 20, temperature: 20, soil: 15, relief: 15 },
    rover:    { slope: 45, flat: 35, temperature: 5,  soil: 5,  relief: 10 },
    habitat:  { slope: 10, flat: 35, temperature: 20, soil: 10, relief: 25 },
    isru:     { slope: 15, flat: 10, temperature: 25, soil: 35, relief: 15 },
  };
  const PILLARS = ["slope", "flat", "temperature", "soil", "relief"];
  const round1 = (x) => Math.round(x * 10) / 10;

  function fakeResponse(data, isText) {
    const body = isText ? String(data) : JSON.stringify(data);
    return {
      ok: true, status: 200,
      json: async () => (isText ? {} : data),
      text: async () => body,
    };
  }

  function rescore(row, weights) {
    let num = 0, den = 0;
    for (const k of PILLARS) {
      const d = row[`dist_${k}`];
      if (d !== null && d !== undefined && !Number.isNaN(d) && weights[k] > 0) {
        num += weights[k] * d * d;
        den += weights[k];
      }
    }
    if (den === 0) return null;
    return round1(100.0 * Math.exp(-Math.sqrt(num / den)));
  }

  function rankings(params) {
    const base = params.get("base");
    const preset = params.get("preset") || "balanced";
    if (!D.rankings[base]) return fakeResponse({ detail: `Unknown base '${base}'` });
    const weights = { ...PRESETS[preset] };
    let custom = false;
    for (const k of PILLARS) {
      const v = params.get(k);
      if (v !== null) { weights[k] = Number(v); custom = true; }
    }
    const isDefault = !custom && preset === "balanced";
    const sub = D.rankings[base].results.map((r) => ({ ...r }));
    if (isDefault) {
      for (const r of sub) { r.display_score = r.analog_score; /* display_rank already the official rank */ }
    } else {
      for (const r of sub) r.display_score = rescore(r, weights);
      const scored = sub.filter((r) => r.display_score !== null).map((r) => r.display_score);
      for (const r of sub) {
        r.display_rank = r.display_score === null ? null
          : 1 + scored.filter((x) => x > r.display_score).length;
      }
      sub.sort((a, b) => (a.display_rank === null) - (b.display_rank === null) ||
        (a.display_rank || 0) - (b.display_rank || 0));
    }
    return fakeResponse({ base, preset, weights, official: isDefault, results: sub });
  }

  function compare(params) {
    const base = params.get("base"), site = params.get("site");
    const b = D.bases.find((r) => r.base_name === base);
    const e = D.sites.find((r) => r.site_name === site);
    const s = (D.rankings[base] ? D.rankings[base].results : [])
      .find((r) => r.earth_site === site);
    if (!b || !e || !s) return fakeResponse({ detail: "Unknown base/site pair" });
    const radar = {};
    for (const k of PILLARS) {
      const d = s[`dist_${k}`];
      radar[k] = (d === null || d === undefined) ? null : Math.round((1 - d) * 1000) / 1000;
    }
    return fakeResponse({
      base, site,
      analog_score: s.analog_score, rank_for_base: s.display_rank,
      of_sites: D.sites.length,
      radar_similarity: radar,
      moon: {
        slope_mean_deg: b.slope_mean_deg, pct_slope_under_10deg: b.pct_slope_under_10deg,
        temp_seasonal_mean_c: b.temp_seasonal_mean_c, elev_std_m: b.elev_std_m,
        elev_mean_m: b.elev_mean_m, psr_pct: b.psr_pct,
        sun_visibility_pct: b.sun_visibility_pct, earth_visibility_pct: b.earth_visibility_pct,
      },
      earth: {
        slope_mean_deg: e.slope_mean_deg, pct_slope_under_10deg: e.pct_slope_under_10deg,
        temp_mean_c: e.temp_mean_c, elev_std_m: e.elev_std_m, elev_mean_m: e.elev_mean_m,
        temp_seasonal_range_c: e.temp_seasonal_range_c, analog_note: e.analog_note ?? null,
      },
    });
  }

  const nativeFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    const url = typeof input === "string" ? input : (input && input.url) || "";
    const u = new URL(url, window.location.origin);
    if (u.pathname.startsWith("/api/")) {
      switch (u.pathname) {
        case "/api/health": return Promise.resolve(fakeResponse(D.health));
        case "/api/moon-bases": return Promise.resolve(fakeResponse(D.bases));
        case "/api/earth-sites": return Promise.resolve(fakeResponse(D.sites));
        case "/api/winners": return Promise.resolve(fakeResponse(D.winners));
        case "/api/mars": return Promise.resolve(fakeResponse(D.mars));
        case "/api/rankings": return Promise.resolve(rankings(u.searchParams));
        case "/api/compare": return Promise.resolve(compare(u.searchParams));
        case "/api/methodology": return Promise.resolve(fakeResponse(D.methodologyHtml, true));
        default: break;
      }
    }
    return nativeFetch(input, init);
  };
})();
