/* Nirdhar web UI */
const $ = id => document.getElementById(id);
let META = null, MAP = null, LAST = null;

async function api(path, opts) {
  const r = await fetch(path, opts);
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail ? (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)) : r.statusText);
  return j;
}

/* ------------------------------------------------ tabs */
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach(x => x.classList.toggle("on", x === b));
  for (const t of ["value", "model", "api"]) $("tab-" + t).classList.toggle("hidden", b.dataset.tab !== t);
  if (b.dataset.tab === "model") renderModel();
  if (b.dataset.tab === "value" && MAP) setTimeout(() => MAP.invalidateSize(), 50);
});

/* ------------------------------------------------ form */
function fillLocalities() {
  const city = $("city").value;
  $("micro_market").innerHTML = META.localities[city].map(l => `<option>${esc(l.name)}</option>`).join("");
  fillProjects();
}
function fillProjects() {
  const city = $("city").value, mm = $("micro_market").value;
  const ps = META.projects.filter(p => p.city === city && p.micro_market === mm);
  $("project_id").innerHTML = `<option value="">— Not in RERA list / standalone —</option>` +
    ps.map(p => `<option value="${p.project_id}">${esc(p.name)} · ${esc(p.builder)}</option>`).join("");
  projHint();
}
function projHint() {
  const p = META.projects.find(x => x.project_id === $("project_id").value);
  $("proj-hint").textContent = p ? `${p.rera_id ? "RERA " + p.rera_id : "Pre-RERA"} · ${p.floors} floors · ${p.completed ? "possession " + p.possession : "under construction, possession " + p.possession}` : "Location will use the micro-market centroid";
  if (p) $("total_floors").placeholder = `${p.floors} (RERA)`;
}
$("city").onchange = fillLocalities;
$("micro_market").onchange = fillProjects;
$("project_id").onchange = projHint;
document.querySelectorAll("#parking button").forEach(b => b.onclick = () => {
  document.querySelectorAll("#parking button").forEach(x => x.classList.toggle("on", x === b));
});

function setForm(req) {
  $("city").value = req.city; fillLocalities();
  $("micro_market").value = req.micro_market; fillProjects();
  $("project_id").value = req.project_id || ""; projHint();
  for (const k of ["property_type", "commercial_kind", "bhk", "carpet_area", "area_basis", "floor", "total_floors", "age_years", "facing"])
    $(k).value = req[k] ?? "";
  for (const k of ["agreement_value", "loan_amount", "asking_price"]) $(k).value = req[k] ? (req[k] / 1e5).toFixed(1) : "";
  const pv = req.parking === undefined || req.parking === null ? "" : String(req.parking);
  document.querySelectorAll("#parking button").forEach(x => x.classList.toggle("on", x.dataset.v === pv));
}
function readForm() {
  const n = id => $(id).value === "" ? null : Number($(id).value);
  const lakh = id => $(id).value === "" ? null : Math.round(Number($(id).value) * 1e5);
  const pk = document.querySelector("#parking button.on").dataset.v;
  return {
    city: $("city").value, micro_market: $("micro_market").value, project_id: $("project_id").value || null,
    property_type: $("property_type").value, commercial_kind: $("property_type").value === "commercial" ? $("commercial_kind").value : null, bhk: n("bhk"), carpet_area: n("carpet_area"), area_basis: $("area_basis").value,
    floor: n("floor"), total_floors: n("total_floors"), age_years: n("age_years"), facing: $("facing").value || null,
    parking: pk === "" ? null : pk === "true",
    agreement_value: lakh("agreement_value"), loan_amount: lakh("loan_amount"), asking_price: lakh("asking_price"),
  };
}

$("form").onsubmit = async e => {
  e.preventDefault();
  const btn = $("go");
  btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>Valuing…`;
  $("form-err").classList.add("hidden");
  try {
    const t0 = performance.now();
    const r = await api("/api/valuations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(readForm()) });
    r._ms = Math.round(performance.now() - t0);
    LAST = r; render(r);
  } catch (err) {
    $("form-err").textContent = err.message; $("form-err").classList.remove("hidden");
  } finally { btn.disabled = false; btn.textContent = "Value property"; }
};

/* ------------------------------------------------ results */
function provenanceHtml(p) {
  if (!p || !Object.keys(p).length) return "";
  return `<div class="eyebrow" style="margin:16px 0 8px">Data provenance</div><table class="kv">` +
    Object.entries(p).map(([k, v]) => `<tr><td>${esc(k)}</td><td style="font-weight:500;color:${/synthetic|approximate/.test(v) ? "#b26a00" : "#1f7a4a"}">${esc(v)}</td></tr>`).join("") + `</table>`;
}

function renderCommercial(r) {
  const out = $("out"), v = r.valuation, p = r.property, rt = r.route, c = r.commercial;
  if (!v) {
    out.innerHTML = `<div class="card"><h3>Commercial · income approach</h3>${flagsHtml(r.flags)}<div class="route field"><b>${esc(rt.route)}</b>${esc(rt.reason)}</div></div>`;
    return;
  }
  out.innerHTML = `
  <div class="card"><div class="hero"><div>
    <div class="eyebrow">Estimated value · income approach</div>
    <div class="bigval num">${inr(v.value_inr)}</div>
    <div class="range-txt">Range <b class="num">${inr(v.p05_inr)} – ${inr(v.p95_inr)}</b> (±${v.halfwidth_pct}%) · ₹${num(v.ppsf)}/sq ft carpet</div>
    <div class="prop-line">${esc(p.kind)} · ${num(p.carpet_area)} sq ft carpet · floor ${p.floor} · ${esc(p.micro_market)}, ${esc(p.city)}</div>
    <div class="grade"><div class="g g${v.grade}">${v.grade}</div><div><b>${v.grade_label}</b><div class="sub">${esc(v.confidence_status)}</div></div></div>
    ${v.grade_notes.map(n => `<div class="sub" style="margin-top:6px">⚠ ${esc(n)}</div>`).join("")}
    <div style="margin-top:12px;display:flex;gap:6px;flex-wrap:wrap"><span class="pill navy">${esc(rt.route)}</span><span class="pill exp">Assumption-driven · illustrative</span></div>
  </div><div>
    <div class="eyebrow">Evidence near the property</div>
    <div class="dq" style="margin-top:8px">
      <div><b class="num">${c.rent_listings_1500m}</b><span>real rent listings within 1.5 km (Makaan)</span></div>
      <div><b class="num">${c.footfall_poi_1km}</b><span>shops and eateries within 1 km (OSM)</span></div>
      <div><b class="num">${num(c.dist_metro_m)} m</b><span>to ${esc(c.nearest_metro)} metro (operational)</span></div>
    </div></div></div></div>
  <div class="grid2">
    <div class="card"><h3>How the value is built</h3><table class="kv">${c.steps.map(s => `<tr><td>${esc(s.step)}<span class="s">${esc(s.source)}</span></td><td class="num">${esc(s.value)}</td></tr>`).join("")}</table>
      <div class="method">Commercial property is valued by the income approach: what it would rent for, after vacancy and costs, divided by the yield investors accept. Multipliers and yields are stated assumptions to be calibrated on lease and sale records in the pilot.</div></div>
    <div class="card"><h3>Lending decision (LAP)</h3><div class="route ${rt.code}"><b>${esc(rt.route)}</b>${esc(rt.reason)}</div>
      <table class="kv"><tr><td>LTV base<span class="s">${esc(rt.ltv_base_label)}</span></td><td class="num">${inr(rt.ltv_base_inr)}</td></tr>
      <tr><td>Maximum eligible loan<span class="s">${esc(rt.ltv_rule)}</span></td><td class="num">${inr(rt.max_eligible_loan_inr)}</td></tr>
      ${rt.requested_loan_inr ? `<tr><td>Requested loan</td><td class="num" style="color:${rt.requested_within_limit ? "#1f7a4a" : "#c8102e"}">${inr(rt.requested_loan_inr)} ${rt.requested_within_limit ? "✓ within limit" : "✗ exceeds limit"}</td></tr>` : ""}</table>
      <h3 style="margin-top:16px">Flags</h3>${flagsHtml(r.flags)}${provenanceHtml(r.data_provenance)}</div>
  </div>
  <div class="card foot"><div class="sub">Valuation <span class="mono">${r.valuation_id}</span> · model <span class="mono">${esc(r.model.model_version)}</span></div>
    <a class="btn ghost" href="/static/report.html?id=${r.valuation_id}" target="_blank">Open valuation report ↗</a></div>`;
  if (MAP) { MAP.remove(); MAP = null; }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function render(r) {
  $("empty").classList.add("hidden");
  const out = $("out"); out.classList.remove("hidden");
  if (r.mode === "commercial_income") { renderCommercial(r); return; }
  if (!r.supported) {
    out.innerHTML = `<div class="card"><h3>Out of scope</h3><div class="route field"><b>${esc(r.route.route)}</b>${esc(r.route.reason)}</div>
      <p class="sub">Valuation id <span class="mono">${r.valuation_id}</span> logged to the audit store.</p></div>`;
    return;
  }
  const v = r.valuation, p = r.property, rt = r.route;
  const title = `${p.bhk} BHK · ${num(p.carpet_area)} sq ft carpet · ${p.project_name ? esc(p.project_name) + ", " : ""}${esc(p.micro_market)}, ${esc(p.city)}`;
  out.innerHTML = `
  <div class="card">
    <div class="hero">
      <div>
        <div class="eyebrow">Estimated market value</div>
        <div class="bigval num">${inr(v.value_inr)}</div>
        <div class="range-txt">90% range <b class="num">${inr(v.p05_inr)} – ${inr(v.p95_inr)}</b> (±${v.halfwidth_pct}%) · <span class="num">₹${num(v.ppsf)}</span>/sq ft carpet</div>
        <div class="prop-line">${title}${p.area_basis !== "carpet" ? ` · entered ${num(p.area_input)} sq ft ${p.area_basis}` : ""}</div>
        <div class="grade"><div class="g g${v.grade}">${v.grade}</div><div><b>${v.grade_label}</b><div class="sub">${esc(v.confidence_status)}</div></div></div>
        ${v.grade_notes.map(n => `<div class="sub" style="margin-top:6px">⚠ ${esc(n)}</div>`).join("")}
        <div style="margin-top:12px;display:flex;gap:6px;flex-wrap:wrap"><span class="pill navy">${esc(rt.route)}</span>
          <span class="pill soft">${r.flags.filter(f => f.severity !== "info").length} risk flag(s)</span>
          <span class="pill exp">Synthetic data · illustrative</span><span class="pill soft">${r._ms ?? "—"} ms</span></div>
      </div>
      <div><div class="eyebrow">Three-price reconciliation</div>${numberLine(r)}</div>
    </div>
  </div>

  <div class="grid2">
    <div class="card"><h3>Price reconciliation</h3>
      <table class="kv">${r.reconciliation.items.map(i => `<tr><td>${esc(i.label)}<span class="s">${esc(i.source)}</span></td>
        <td class="num">${inr(i.inr)}${i.status ? (i.status === "ok" ? " ✓" : ` <span style="color:#c8102e">● ${i.status}</span>`) : ""}${i.range ? `<span class="s">${inr(i.range[0])} – ${inr(i.range[1])}</span>` : ""}</td></tr>`).join("")}</table>
      <ul class="notes">${r.reconciliation.notes.map(n => `<li>${esc(n)}</li>`).join("")}</ul>
    </div>
    <div class="card"><h3>Lending decision</h3>
      <div class="route ${rt.code}"><b>${esc(rt.route)}</b>${esc(rt.reason)}</div>
      <table class="kv">
        <tr><td>LTV base<span class="s">${esc(rt.ltv_base_label)}</span></td><td class="num">${inr(rt.ltv_base_inr)}</td></tr>
        <tr><td>Maximum eligible loan<span class="s">${esc(rt.ltv_rule)}</span></td><td class="num">${inr(rt.max_eligible_loan_inr)}</td></tr>
        ${rt.requested_loan_inr ? `<tr><td>Requested loan</td><td class="num" style="color:${rt.requested_within_limit ? "#1f7a4a" : "#c8102e"}">${inr(rt.requested_loan_inr)} ${rt.requested_within_limit ? "✓ within limit" : "✗ exceeds limit"}</td></tr>` : ""}
      </table>
      <div class="method">Routing precedence: ${esc(rt.precedence)}.</div>
    </div>
  </div>

  <div class="grid2">
    <div class="card"><h3>Why this value</h3>${driversHtml(r.drivers)}<div class="method">${esc(r.drivers.method)}</div></div>
    <div class="card"><h3>Risk flags</h3>${flagsHtml(r.flags)}</div>
  </div>

  <div class="card"><h3>Comparable sales used by the model</h3>
    <p class="sub" style="margin-top:-6px">${r.comparables.considered} candidate sales within 5 km → ${r.comparables.selected} selected by learned attention.
      Each comp is time-adjusted to ${esc(r.model.as_of)} and adjusted to the subject for size, age, floor and parking.
      Comparables-only estimate: ₹${num(r.comparables.comps_value_ppsf)}/sq ft.</p>
    <div id="map"></div>
    ${compsTable(r.comparables.items, 10)}
  </div>

  <div class="grid2">
    <div class="card"><h3>Project intelligence</h3>${projectHtml(r.project_intel)}</div>
    <div class="card"><h3>Confidence &amp; uncertainty</h3>
      <div class="eyebrow" style="margin-bottom:6px">Model disagreement</div>${modelsHtml(r.disagreement, v)}
      <div class="eyebrow" style="margin:16px 0 8px">Data quality</div>
      <div class="dq">
        <div><b class="num">${r.data_quality.comps_within_1km}</b><span>sales within 1 km</span></div>
        <div><b class="num">${num(r.data_quality.comps_effective_n, 1)}</b><span>effective comps (attention)</span></div>
        <div><b class="num">${Math.round(r.data_quality.comps_deed_share_top10 * 100)}%</b><span>top-10 comps from registered deeds</span></div>
        <div><b class="num">${num(r.data_quality.calibration_effective_n, 0)}</b><span>calibration deeds nearby (effective)</span></div>
        <div><b class="num">${r.data_quality.ood_score.toFixed(2)}</b><span>out-of-distribution score (max |z| ${r.data_quality.ood_max_z})</span></div>
        <div><b>${esc(r.data_quality.density_bucket)}</b><span>local data density</span></div>
      </div>
      <div class="method">Range: spatially weighted split-conformal prediction on normalised residuals, calibrated on held-out registered deeds (conformal q = ${v.conformal_q}, σ = ${v.sigma}).
        Grade from the range width: A ≤ ±10% (the PE10 standard), B ≤ ±15%, C ≤ ±20%, else D; capped for out-of-distribution properties. Location: ${esc(r.data_quality.location_source)}.</div>
      ${provenanceHtml(r.data_provenance)}
    </div>
  </div>

  <div class="card foot">
    <div class="sub">Valuation <span class="mono">${r.valuation_id}</span> · model <span class="mono">${esc(r.model.model_version)}</span> · data snapshot <span class="mono">${esc(r.model.data_snapshot_hash)}</span> · as of ${esc(r.model.as_of)}</div>
    <div style="display:flex;gap:8px"><button class="btn ghost" id="replay">Replay (audit)</button>
      <a class="btn ghost" href="/static/report.html?id=${r.valuation_id}" target="_blank">Open valuation report ↗</a></div>
  </div>`;
  $("replay").onclick = async () => {
    const j = await api(`/api/valuations/${r.valuation_id}/replay`, { method: "POST" });
    $("replay").textContent = j.exact_match ? "✓ Replayed: exact match" : "✗ Replay differs";
  };
  drawMap(r);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function drawMap(r) {
  const el = $("map");
  if (typeof L === "undefined") { el.className = "map-off"; el.textContent = "Map unavailable offline"; return; }
  if (MAP) { MAP.remove(); MAP = null; }
  MAP = L.map(el, { scrollWheelZoom: false });
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 18, attribution: "© OpenStreetMap contributors" }).addTo(MAP);
  const p = r.property, pts = [[p.lat, p.lon]];
  const maxw = Math.max(...r.comparables.items.map(c => c.weight));
  for (const c of r.comparables.context || []) {
    if (c.selected) continue;
    L.circleMarker([c.lat, c.lon], { radius: 3 + Math.min(6, Math.sqrt(c.sales)), color: "#8a97aa", weight: 1, fillOpacity: .25 })
      .bindTooltip(`${esc(c.project)}<br>${c.sales} sales · median ₹${num(c.median_adj_ppsf)}/sq ft adj<br><i>not selected</i>`).addTo(MAP);
  }
  // keep at least ~1.5 km of context around the subject
  pts.push([p.lat + 0.007, p.lon + 0.007], [p.lat - 0.007, p.lon - 0.007]);
  r.comparables.items.forEach((c, i) => {
    const jitter = c.distance_m < 5 ? 0.0004 * Math.cos(i * 2.4) : 0;
    const ll = [c.lat + jitter, c.lon + (c.distance_m < 5 ? 0.0004 * Math.sin(i * 2.4) : 0)];
    pts.push(ll);
    L.circleMarker(ll, { radius: 4 + 10 * c.weight / maxw, color: c.source.startsWith("Reg") ? "#1f7a4a" : "#3a5a85", weight: 1, fillOpacity: .55 })
      .bindTooltip(`${esc(c.project)}<br>₹${num(c.ppsf_adj_to_subject)}/sq ft adj · w ${c.weight}`).addTo(MAP);
  });
  L.circleMarker([p.lat, p.lon], { radius: 9, color: "#fff", weight: 3, fillColor: "#c8102e", fillOpacity: 1 }).bindTooltip("Subject property").addTo(MAP);
  MAP.fitBounds(L.latLngBounds(pts).pad(0.25));
}

/* ------------------------------------------------ model & validation */
let MODEL_DONE = false;
async function renderModel() {
  if (MODEL_DONE) return;
  const m = await api("/api/model");
  MODEL_DONE = true;
  const H = m.results.Hyderabad, P = m.results.Pune, lad = m.ladder;
  const maxE = Math.max(...Object.values(lad).map(x => x.MdAPE));
  const rows = Object.entries(lad).map(([k, x], i) => {
    const cls = i < 2 ? "base" : (i === Object.keys(lad).length - 1 ? "final" : "");
    return `<div class="lrow ${cls}"><div>${esc(k)}</div><div><div class="lb" style="width:${(x.MdAPE / maxE * 100).toFixed(1)}%"></div></div>
      <div class="num">${x.MdAPE.toFixed(2)}%</div><div class="num">${x.PE10}%</div><div class="num">${x.COD}</div><div class="num">${x.PRD}</div></div>`;
  }).join("");
  const gradeTbl = (res) => `<table class="tbl"><thead><tr><th>Grade</th><th class="r">Share</th><th class="r">MdAPE</th><th class="r">90% coverage</th><th class="r">n</th></tr></thead><tbody>` +
    Object.entries(res.by_grade).map(([g, x]) => `<tr><td><b>${g}</b></td><td class="r">${x.share}%</td><td class="r">${x.MdAPE}%</td><td class="r">${x.coverage_90}%</td><td class="r">${x.n}</td></tr>`).join("") + "</tbody></table>";
  const pr = m.pipeline_report;
  const cityCard = (name, x) => `<div class="card"><h3>${name} · registered-deed test set</h3>
    <div class="dq">
      <div><b>${x.MdAPE}%</b><span>MdAPE</span></div><div><b>${x.PE10}%</b><span>within ±10% (PE10)</span></div>
      <div><b>${x.PE20}%</b><span>within ±20% (PE20)</span></div><div><b>${x.coverage_90}%</b><span>90% range coverage</span></div>
      <div><b>${x.COD}</b><span>COD (uniformity)</span></div><div><b>${x.PRD}</b><span>PRD (price-band bias)</span></div>
      <div><b>${x.hit_rate_AB}%</b><span>hit rate (grade A or B)</span></div><div><b>${x.n}</b><span>deeds, ${esc(m.splits.test)}</span></div>
    </div><div style="margin-top:12px">${gradeTbl(x)}</div></div>`;
  $("model-out").innerHTML = `
  <div class="card"><h3>Ablation ladder · ${esc(m.ladder_city || "Pune")} registered deeds (${(m.results[m.ladder_city || "Pune"] || P).n}), ${esc(m.splits.test)}</h3>
    <p class="sub" style="margin-top:-6px">Each fix from the solution deck is added in turn and scored on held-out registered sale values (time split). Lower MdAPE is better; COD measures uniformity; PRD near 1.00 means no price-band bias.</p>
    <div class="ladder"><div class="lrow head"><div>Model</div><div>MdAPE</div><div>MdAPE</div><div>PE10</div><div>COD</div><div>PRD</div></div>${rows}</div>
    <ul class="notes">
      <li>Deed labels alone barely help while area is still mixed (carpet / built-up / super built-up): the two biases partly cancel. Normalising to carpet area is the largest single gain.</li>
      <li>Real location layers (OpenStreetMap metro, amenities, river proximity, real rents) and project identity each improve MdAPE, PE10 and COD. Numbers are reported as measured, not tuned.</li>
      <li>Results are on synthetic data and show the method and protocol; real-data numbers come from the pilot.</li>
    </ul></div>
  <div class="grid2">${cityCard("Pune (primary)", P)}${cityCard("Hyderabad (expansion)", H)}</div>
  <div class="card"><h3>Data provenance by city</h3>${Object.entries(m.data_provenance || {}).map(([city, p]) => `<div class="eyebrow" style="margin:12px 0 0">${esc(city)}</div>` + provenanceHtml(p)).join("")}</div>
  <div class="grid2">
    <div class="card"><h3>Data pipeline report</h3><table class="kv">
      <tr><td>Raw listings · deeds · RERA projects · location cells</td><td>${num(pr.input.listings)} · ${num(pr.input.deeds)} · ${pr.input.projects} · ${num(pr.input.cells)}</td></tr>
      <tr><td>Listings resolved to a RERA project<span class="s">fuzzy name + geo + building height</span></td><td>${(pr.project_resolution.matched_share * 100).toFixed(1)}% · precision ${(pr.project_resolution.precision_vs_hidden_truth * 100).toFixed(1)}%</td></tr>
      <tr><td>Cross-portal duplicates removed</td><td>${num(pr.dedup.removed)} (recall ${(pr.dedup.recall * 100).toFixed(0)}%)</td></tr>
      <tr><td>Outliers removed (robust z &gt; 3.5)</td><td>${pr.outliers.removed} (${(pr.outliers.share * 100).toFixed(1)}%)</td></tr>
      <tr><td>Listing → deed gap learned on Pune Index-II<span class="s">${num(pr.deed_calibration.pune_pairs)} matched pairs</span></td><td>${(pr.deed_calibration.mean_listing_to_deed_gap * 100).toFixed(1)}%</td></tr>
      <tr><td>Calibration error: Pune holdout → Hyderabad transfer</td><td>${pr.deed_calibration.pune_holdout_mae_pct}% → ${pr.deed_calibration.hyderabad_transfer_mae_pct}%</td></tr>
      <tr><td>Clean sales rows (deeds + calibrated listings)</td><td>${num(pr.output.sales_rows)}</td></tr>
      <tr><td>Price index, last 12 months</td><td>Hyd ${(pr.time_index_yoy.Hyderabad * 100).toFixed(1)}% · Pune ${(pr.time_index_yoy.Pune * 100).toFixed(1)}%</td></tr>
    </table></div>
    <div class="card"><h3>Model card</h3><table class="kv">
      <tr><td>Version</td><td class="mono">${esc(m.model_version)}</td></tr>
      <tr><td>Trained</td><td>${esc(m.trained_at)}</td></tr>
      <tr><td>Data snapshot hash</td><td class="mono">${esc(m.data_snapshot_hash)}</td></tr>
      ${Object.entries(m.splits).map(([k, v]) => `<tr><td>Split · ${k}</td><td>${esc(v)} (${num(m.rows[k])} rows)</td></tr>`).join("")}
      ${Object.entries(m.stacker_weights).map(([k, w]) => `<tr><td>Stacker weights · ${k} density</td><td>${Object.entries(w).map(([a, b]) => `${a.split(" ")[0]} ${b}`).join(" · ")}</td></tr>`).join("")}
      <tr><td>Grade cuts (90% half-width)</td><td>A ≤ ${m.grade_cuts[0] * 100}% · B ≤ ${m.grade_cuts[1] * 100}% · C ≤ ${m.grade_cuts[2] * 100}%</td></tr>
      <tr><td>Training data</td><td>${m.data_is_synthetic ? "Synthetic (prototype)" : "Licensed"}</td></tr>
    </table></div>
  </div>`;
}

/* ------------------------------------------------ boot */
(async function boot() {
  META = await api("/api/meta");
  $("city").innerHTML = Object.keys(META.localities).map(c => `<option>${c}</option>`).join("");
  fillLocalities();
  $("micro_market").value = "Kothrud"; fillProjects();
  const kindToggle = () => $("kind-field").classList.toggle("hidden", $("property_type").value !== "commercial");
  $("property_type").addEventListener("change", kindToggle); kindToggle();
  $("presets").innerHTML = META.presets.map(p => `<button type="button" class="preset" data-k="${p.key}"><b>${esc(p.title)}</b><span>${esc(p.subtitle)}</span></button>`).join("");
  document.querySelectorAll(".preset").forEach(b => b.onclick = () => {
    const p = META.presets.find(x => x.key === b.dataset.k);
    setForm(p.req); $("kind-field").classList.toggle("hidden", $("property_type").value !== "commercial"); $("form").requestSubmit();
  });
})();
