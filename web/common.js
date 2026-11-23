/* Shared formatting + render helpers (used by the app and the printable report). */
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function inr(v, d) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  const a = Math.abs(v), s = v < 0 ? "−" : "";
  if (a >= 1e7) return `${s}₹${(a / 1e7).toFixed(d ?? 2)} Cr`;
  if (a >= 1e5) return `${s}₹${(a / 1e5).toFixed(d ?? 1)} L`;
  return `${s}₹${Math.round(a).toLocaleString("en-IN")}`;
}
const signedInr = v => (v > 0 ? "+" : v < 0 ? "−" : "") + inr(Math.abs(v)).replace("−", "");
const num = (v, d = 0) => (v === null || v === undefined || isNaN(v)) ? "—" : Number(v).toLocaleString("en-IN", { maximumFractionDigits: d, minimumFractionDigits: d });
const pct = (v, d = 1) => (v === null || v === undefined) ? "—" : `${Number(v).toFixed(d)}%`;

function numberLine(r) {
  const v = r.valuation, rec = r.reconciliation.items;
  const find = l => rec.find(i => i.label === l);
  const pts = [
    { k: "Guidance", v: find("Government guidance value")?.inr, c: "#8a97aa", up: true },
    { k: "Asking", v: find("Asking price")?.inr, c: "#3a5a85", up: false },
    { k: "Agreement", v: find("Agreement value")?.inr, c: find("Agreement value")?.status === "ok" ? "#1f7a4a" : "#c8102e", up: true },
  ].filter(p => p.v);
  const all = [v.p05_inr, v.p95_inr, v.value_inr, ...pts.map(p => p.v)];
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const x = val => ((val - lo) / (hi - lo) * 100).toFixed(2) + "%";
  let h = `<div class="nline"><div class="axis"></div>
    <div class="band" style="left:${x(v.p05_inr)};width:calc(${x(v.p95_inr)} - ${x(v.p05_inr)})"></div>
    <div class="mk down" style="left:${x(v.value_inr)}"><div class="dot" style="background:#172f4f;width:16px;height:16px"></div><b>${inr(v.value_inr)}</b><br>Nirdhar</div>
    <div class="mk down" style="left:${x(v.p05_inr)};top:66px;color:#5d6b7e">P5 ${inr(v.p05_inr)}</div>
    <div class="mk down" style="left:${x(v.p95_inr)};top:66px;color:#5d6b7e">P95 ${inr(v.p95_inr)}</div>`;
  for (const p of pts) {
    h += `<div class="mk up" style="left:${x(p.v)};color:${p.c}"><b>${p.k}</b> ${inr(p.v)}<div class="dot" style="background:${p.c}"></div><div class="tick" style="background:${p.c}"></div></div>`;
  }
  return h + "</div>";
}

function driversHtml(d) {
  const max = Math.max(...d.items.map(i => Math.abs(i.inr)), 1);
  let h = `<div class="drv total"><div>${esc(d.baseline_label)}</div><div></div><div class="amt num">${inr(d.baseline_inr)}</div></div>`;
  for (const it of d.items) {
    const w = Math.abs(it.inr) / max * 50;
    const left = it.inr >= 0 ? 50 : 50 - w;
    h += `<div class="drv"><div>${esc(it.factor)}</div>
      <div class="bar-wrap"><div class="mid"></div><div class="bar ${it.inr >= 0 ? "pos" : "neg"}" style="left:${left}%;width:${Math.max(w, 0.4)}%"></div></div>
      <div class="amt num" style="color:${it.inr >= 0 ? "#172f4f" : "#c8102e"}">${Math.abs(it.inr) < 500 ? "≈ 0" : signedInr(it.inr)}</div></div>`;
  }
  h += `<div class="drv total"><div>Nirdhar estimate</div><div></div><div class="amt num">${inr(d.value_inr)}</div></div>`;
  return h;
}

function flagsHtml(flags) {
  if (!flags.length) return `<div class="noflags">✓ No risk flags: agreement value in range, approvals on record, no flood, delay or data issues.</div>`;
  return flags.map(f => `<div class="flag ${f.severity}"><div class="sw"></div><div><b>${esc(f.title)}</b><span>${esc(f.detail)}</span></div></div>`).join("");
}

function compsTable(items, n = 10) {
  const maxw = Math.max(...items.map(i => i.weight));
  return `<div class="tw"><table class="tbl"><thead><tr><th>#</th><th>Project</th><th class="r">Distance</th><th>Sold</th><th>Unit</th><th>Source</th>
    <th class="r">₹/sq ft</th><th class="r">Adj. to subject</th><th style="width:90px">Weight</th></tr></thead><tbody>` +
    items.slice(0, n).map((c, i) => `<tr><td>${i + 1}</td><td><b>${esc(c.project)}</b><div class="why">${esc(c.why)}</div></td>
      <td class="r num">${num(c.distance_m)} m</td><td>${esc(c.month)}</td>
      <td>${c.bhk} BHK · ${num(c.carpet_area)} sq ft · fl ${c.floor}</td>
      <td class="${c.source.startsWith("Registered") ? "src-deed" : "src-list"}">${esc(c.source)}</td>
      <td class="r num">${num(c.ppsf_time_adj)}</td><td class="r num"><b>${num(c.ppsf_adj_to_subject)}</b></td>
      <td><div class="wbar" style="width:${(c.weight / maxw * 100).toFixed(0)}%"></div><span class="why num">${c.weight.toFixed(3)}</span></td></tr>`).join("") +
    `</tbody></table></div>`;
}

function projectHtml(p) {
  if (!p.linked) return `<p class="sub">${esc(p.message)}</p>`;
  const b = p.builder;
  return `<h2>${esc(p.name)}</h2><div class="sub">${esc(p.micro_market)} · ${p.rera_id ? "RERA " + esc(p.rera_id) : "Not RERA-registered"} · ${esc(p.status)}</div>
    <table class="kv" style="margin-top:10px">
      <tr><td>Launch → promised → revised possession</td><td>${esc(p.launch)} → ${esc(p.promised_possession)} → ${esc(p.revised_possession)}</td></tr>
      <tr><td>Delay</td><td style="color:${p.delay_months >= 12 ? "#c8102e" : "inherit"}">${p.delay_months} months</td></tr>
      <tr><td>Units · approved floors · loading</td><td>${num(p.total_units)} · ${p.approved_floors} · ${p.loading_factor.toFixed(2)}×</td></tr>
      <tr><td>Occupancy certificate</td><td>${p.oc_received ? "Received" : "Not on record"}</td></tr>
      <tr><td>RERA complaints</td><td>${p.complaints ?? "n/a (pre-RERA)"}</td></tr>
      <tr><td>Transactions in Nirdhar data</td><td>${p.transactions_in_pool}</td></tr>
    </table>
    <div style="margin:12px 0 6px" class="eyebrow">Approvals</div>
    <div class="chips">${p.approvals_on_record.map(a => `<span class="chip">✓ ${esc(a)}</span>`).join("")}
      ${p.approvals_missing.map(a => `<span class="chip miss">✗ ${esc(a)}</span>`).join("")}</div>
    <div style="margin:14px 0 6px" class="eyebrow">Builder · ${esc(b.name)}</div>
    <table class="kv">
      <tr><td>Reliability score<span class="s">0.7 × on-time delivery + 0.3 × OC rate, from RERA filings</span></td><td>${b.reliability_score.toFixed(2)}</td></tr>
      <tr><td>Projects · average delay</td><td>${b.projects_count} · ${b.avg_delay_months} mo</td></tr>
      <tr><td>OC rate on completed projects</td><td>${b.oc_rate === null ? "—" : Math.round(b.oc_rate * 100) + "%"}</td></tr>
    </table>`;
}

function modelsHtml(dis, v) {
  const vals = dis.models.filter(m => m.value_inr).map(m => m.value_inr).concat([v.p05_inr, v.p95_inr]);
  const lo = Math.min(...vals) * 0.98, hi = Math.max(...vals) * 1.02;
  const x = q => ((q - lo) / (hi - lo) * 100).toFixed(1);
  return dis.models.map(m => `<div class="mbar"><div>${esc(m.name)}<div class="why">weight ${dis.weights[m.name] ?? 0}</div></div>
     <div class="t"><div class="bandm" style="left:${x(v.p05_inr)}%;width:${x(v.p95_inr) - x(v.p05_inr)}%"></div>${m.value_inr ? `<i style="left:${x(m.value_inr)}%"></i>` : ""}</div>
     <div class="num" style="text-align:right">${inr(m.value_inr)}</div></div>`).join("") +
    `<div class="sub" style="margin-top:6px">Spread across models: <b>${dis.spread_pct}%</b> (${dis.level} disagreement). Shaded band = 90% range.</div>`;
}
