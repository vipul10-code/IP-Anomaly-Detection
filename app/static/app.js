// QualityGuard dashboard client
const $ = s => document.querySelector(s);
const MODELS = {
  patchcore: { label: "PatchCore", color: "#8C2F39" },
  efficientad: { label: "EfficientAD", color: "#B8892A" },
  cae: { label: "CAE", color: "#8A857C" },
  draem: { label: "DRAEM", color: "#2E7A62" },
};
const METRICS = [["image_auroc", "Image AUROC"], ["image_ap", "Image AP"], ["pixel_auroc", "Pixel AUROC"], ["pixel_aupro", "AUPRO"], ["val_f1", "Val F1"], ["oracle_f1", "Oracle F1"]];
const SAMPLES = {
  leather: [["cut", "Cut"], ["good", "Normal"]],
  transistor: [["bent_lead", "Bent lead"], ["good", "Normal"]],
};
const mkey = n => Object.keys(MODELS).find(k => String(n).toLowerCase().replace(/[^a-z]/g, "").includes(k)) || String(n).toLowerCase();
const mlabel = k => (MODELS[k] || { label: k }).label;
const mcolor = k => (MODELS[k] || { color: "#8A857C" }).color;
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const f3 = v => (v == null || isNaN(v)) ? "—" : Number(v).toFixed(3);
const api = u => fetch(u).then(r => { if (!r.ok) throw new Error(u); return r.json(); });

const S = { bench: [], info: [], metric: "image_auroc", cat: "leather", model: "patchcore", view: "overlay", file: null, result: null, defCat: "all", gModel: "patchcore", gCat: "leather", gDefect: "all" };

function seg(el, items, current, onPick) {
  el.innerHTML = items.map(([v, t, dis]) => `<button data-v="${v}" class="${v === current ? "on" : ""}" ${dis ? "disabled" : ""}>${esc(t)}</button>`).join("");
  el.onclick = e => { const b = e.target.closest("button"); if (b && !b.disabled) { onPick(b.dataset.v); } };
}

// Tabs
document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("on", x === t));
  document.querySelectorAll(".view").forEach(v => v.classList.toggle("on", v.id === t.dataset.tab));
  ({ defects: renderDefects, gallery: renderGallery, history: renderHistory })[t.dataset.tab]?.();
});

// ---------- Benchmark ----------
async function loadBench() {
  try {
    const [bench, info] = await Promise.all([api("/api/benchmark"), api("/api/models")]);
    S.bench = bench.map(r => ({ ...r, key: mkey(r.model) }));
    S.info = info.models;
    const ready = S.info.flatMap(m => Object.values(m.available)), n = ready.filter(Boolean).length;
    $("#status").textContent = `${n} of ${ready.length} checkpoints ready`;
    renderBench(); renderInspectControls();
  } catch (e) { $("#headline").textContent = "Benchmark results aren’t available yet"; $("#subline").textContent = "Run the benchmark from the CLI (python main.py benchmark), then reload this page."; }
}

function renderBench() {
  const cats = [...new Set(S.bench.map(r => r.category))];
  const best = c => S.bench.filter(r => r.category === c).sort((a, b) => b.image_auroc - a.image_auroc)[0];
  const winners = cats.map(best).filter(Boolean);
  const same = winners.length && winners.every(w => w.key === winners[0].key);
  $("#headline").textContent = same ? `${mlabel(winners[0].key)} ranks first on every part.` : "The best model depends on the part.";
  $("#subline").textContent = winners.map(w => `${w.category}: ${mlabel(w.key)} at ${f3(w.image_auroc)} image AUROC`).join(". ") + ".";
  seg($("#metricSeg"), METRICS.map(([k, t]) => [k, t]), S.metric, v => { S.metric = v; renderBench(); });

  $("#boards").innerHTML = cats.map(c => {
    const rows = S.bench.filter(r => r.category === c && r[S.metric] != null).sort((a, b) => b[S.metric] - a[S.metric]);
    return `<div class="board"><h3>${esc(c)}</h3>${rows.map((r, i) => `
      <div class="lb ${i === 0 ? "first" : ""}"><span>${mlabel(r.key)}</span>
      <div class="bar2"><i style="width:${Math.max(0, Math.min(1, r[S.metric])) * 100}%;background:${mcolor(r.key)}"></i></div><b>${f3(r[S.metric])}</b></div>`).join("")}</div>`;
  }).join("");

  const cols = [...METRICS, ["inference_ms", "Latency (ms)"], ["model_size_mb", "Size (MB)"]];
  const isMin = k => k === "inference_ms" || k === "model_size_mb";
  let html = `<thead><tr><th>Model</th>${cols.map(c => `<th>${c[1]}</th>`).join("")}</tr></thead><tbody>`;
  cats.forEach(c => {
    const rows = S.bench.filter(r => r.category === c);
    const bestOf = k => { const v = rows.map(r => r[k]).filter(x => x != null && !isNaN(x)); return v.length ? (isMin(k) ? Math.min(...v) : Math.max(...v)) : null; };
    html += `<tr class="cat"><td colspan="${cols.length + 1}">${esc(c)}</td></tr>` + rows.map(r =>
      `<tr><td>${mlabel(r.key)}</td>${cols.map(([k]) => `<td class="${r[k] != null && r[k] === bestOf(k) ? "best" : ""}">${isMin(k) ? (r[k] == null ? "—" : Number(r[k]).toFixed(1)) : f3(r[k])}</td>`).join("")}</tr>`).join("");
  });
  $("#matrix").innerHTML = html + "</tbody>";
}

// ---------- Inspect ----------
function renderInspectControls() {
  seg($("#catSeg"), [["leather", "Leather"], ["transistor", "Transistor"]], S.cat, v => { S.cat = v; renderInspectControls(); });
  const avail = S.info.filter(m => m.categories.includes(S.cat));
  if (!avail.find(m => m.id === S.model && m.available[S.cat])) S.model = (avail.find(m => m.available[S.cat]) || avail[0] || {}).id;
  $("#modelList").innerHTML = S.info.map(m => {
    const ok = m.categories.includes(S.cat) && m.available[S.cat];
    return `<button class="mrow ${m.id === S.model ? "on" : ""}" data-id="${m.id}" ${ok ? "" : "disabled"}><i style="background:${mcolor(m.id)}"></i>${esc(m.name)}<small>${ok ? "" : m.categories.includes(S.cat) ? "no checkpoint" : "not trained"}</small></button>`;
  }).join("");
  $("#samples").innerHTML = `<small style="width:100%;color:var(--mute)">Or try a test image</small>` + SAMPLES[S.cat].map(([k, t]) => `<button data-s="${k}">${t}</button>`).join("");
}
$("#modelList").onclick = e => { const b = e.target.closest(".mrow"); if (b && !b.disabled) { S.model = b.dataset.id; renderInspectControls(); } };
$("#samples").onclick = async e => {
  const k = e.target.dataset.s; if (!k) return;
  try {
    const blob = await (await fetch(`/data/mvtec/${S.cat}/test/${k}/000.png`)).blob();
    if (!blob.type.startsWith("image")) throw 0;
    pick(new File([blob], `${S.cat}_${k}_000.png`, { type: "image/png" }));
  } catch { $("#fileName").textContent = "Sample not found in data/mvtec"; }
};
const drop = $("#drop");
drop.onclick = () => $("#file").click();
drop.onkeydown = e => { if (e.key === "Enter" || e.key === " ") $("#file").click(); };
$("#file").onchange = e => e.target.files[0] && pick(e.target.files[0]);
drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("over"); e.dataTransfer.files[0] && pick(e.dataTransfer.files[0]); };

function pick(file) {
  S.file = file; S.result = null;
  $("#fileName").textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  $("#go").disabled = false; $("#stamp").hidden = true; $("#ruler").hidden = true;
  const r = new FileReader();
  r.onload = e => show(e.target.result);
  r.readAsDataURL(file);
  seg($("#viewSeg"), [["original", "Original"]], "original", () => {});
}
function show(src) { $("#shot").src = src; $("#shot").hidden = false; $("#empty").hidden = true; }

$("#go").onclick = async () => {
  const b = $("#go"); b.disabled = true; b.textContent = "Inspecting…";
  const fd = new FormData(); fd.append("image", S.file); fd.append("model", S.model); fd.append("category", S.cat);
  try {
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || "Inspection failed");
    S.result = d; S.view = "overlay"; renderResult(d);
  } catch (e) { $("#fileName").textContent = e.message; }
  b.disabled = false; b.textContent = "Inspect part";
};

function renderResult(d) {
  const views = [["original", "Original", d.original_b64], ["overlay", "Heatmap", d.overlay_b64], ["mask", "Mask", d.mask_b64], ["recon", "Reconstruction", d.reconstruction_b64]].filter(v => v[2]);
  const draw = () => { seg($("#viewSeg"), views.map(v => [v[0], v[1]]), S.view, v => { S.view = v; draw(); }); show(views.find(v => v[0] === S.view)[2]); };
  draw();
  const bad = d.verdict === "DEFECT", st = $("#stamp");
  st.hidden = false; st.textContent = bad ? "DEFECT" : "PASS"; st.className = `stamp ${bad ? "d" : "n"}`;
  void st.offsetWidth; st.classList.add("hit");
  const dom = Math.max(d.score, d.threshold || 0, 1e-6) * 1.5, pct = v => Math.min(100, v / dom * 100);
  $("#ruler").hidden = false;
  $("#scoreTxt").textContent = d.score.toFixed(4);
  $("#mark").style.left = "0%"; requestAnimationFrame(() => requestAnimationFrame(() => $("#mark").style.left = pct(d.score) + "%"));
  const hasT = d.threshold != null;
  $("#tick").style.display = $("#over").style.display = hasT ? "" : "none";
  if (hasT) { $("#tick").style.left = pct(d.threshold) + "%"; $("#over").style.left = pct(d.threshold) + "%"; }
  $("#thrTxt").textContent = hasT ? `Threshold ${d.threshold.toFixed(4)} · ${d.latency_ms} ms` : `No threshold saved · ${d.latency_ms} ms`;
  $("#statTxt").textContent = `Map min ${d.stats.min.toFixed(2)} · max ${d.stats.max.toFixed(2)} · mean ${d.stats.mean.toFixed(2)}`;
}

// ---------- Defect types ----------
async function renderDefects() {
  seg($("#defCat"), [["all", "All"], ["leather", "Leather"], ["transistor", "Transistor"]], S.defCat, v => { S.defCat = v; renderDefects(); });
  const combos = S.info.flatMap(m => m.categories.map(c => [m.id, c])).filter(([, c]) => S.defCat === "all" || c === S.defCat);
  const data = await Promise.all(combos.map(([m, c]) => api(`/api/defects/${m}/${c}`).then(rows => ({ m, c, rows })).catch(() => null)));
  const got = data.filter(Boolean), models = [...new Set(got.map(g => g.m))], cats = [...new Set(got.map(g => g.c))];
  if (!got.length) { $("#heat").innerHTML = `<tr><td class="note">No defect breakdown files found in outputs/results.</td></tr>`; return; }
  let html = `<thead><tr><th>Defect</th><th>Images</th>${models.map(m => `<th><span style="color:${mcolor(m)}">●</span> ${mlabel(m)}</th>`).join("")}</tr></thead><tbody>`;
  cats.forEach(c => {
    const types = [...new Set(got.filter(g => g.c === c).flatMap(g => g.rows.map(r => r.defect_type)))].sort((a, b) => (a === "good") - (b === "good"));
    html += `<tr class="cat"><td colspan="${models.length + 2}">${esc(c)}</td></tr>`;
    types.forEach(t => {
      const good = t === "good", n = got.flatMap(g => g.c === c ? g.rows : []).find(r => r.defect_type === t)?.n_images ?? "—";
      html += `<tr><td>${good ? "Normal parts (false alarms)" : esc(t.replace(/_/g, " "))}</td><td>${n}</td>` + models.map(m => {
        const row = got.find(g => g.m === m && g.c === c)?.rows.find(r => r.defect_type === t);
        const v = row ? (row.detection_rate_val_th ?? row.detection_rate) : null;
        if (v == null) return `<td class="c" style="color:var(--mute)">—</td>`;
        const a = good ? v : v, col = good ? "140,47,57" : "46,122,98";
        return `<td class="c" style="background:rgba(${col},${(0.08 + a * 0.62).toFixed(2)});color:${a > 0.6 ? "#fff" : "inherit"}">${(v * 100).toFixed(1)}%</td>`;
      }).join("") + "</tr>";
    });
  });
  $("#heat").innerHTML = html + "</tbody>";
}

// ---------- Gallery ----------
async function renderGallery() {
  seg($("#gModel"), Object.keys(MODELS).map(k => [k, mlabel(k)]), S.gModel, v => { S.gModel = v; S.gDefect = "all"; renderGallery(); });
  seg($("#gCat"), [["leather", "Leather"], ["transistor", "Transistor"]], S.gCat, v => { S.gCat = v; S.gDefect = "all"; renderGallery(); });
  try {
    const { heatmaps } = await api(`/api/heatmaps/${S.gModel}/${S.gCat}`);
    const types = [...new Set(heatmaps.map(h => h.defect_type))];
    $("#chips").innerHTML = ["all", ...types].map(t => `<button data-t="${t}" class="${t === S.gDefect ? "on" : ""}">${t === "all" ? "All defects" : esc(t.replace(/_/g, " "))}</button>`).join("");
    $("#chips").onclick = e => { if (e.target.dataset.t) { S.gDefect = e.target.dataset.t; renderGallery(); } };
    const list = heatmaps.filter(h => S.gDefect === "all" || h.defect_type === S.gDefect);
    $("#grid").innerHTML = list.length ? list.map(h => `<a href="${h.url}" target="_blank" rel="noopener"><img src="${h.url}" alt="${esc(h.filename)}" loading="lazy"><span>${esc(h.defect_type.replace(/_/g, " "))} · ${esc(h.filename.replace(".png", ""))}</span></a>`).join("") : `<p class="note">No saved heatmaps for ${mlabel(S.gModel)} on ${S.gCat}.</p>`;
  } catch { $("#grid").innerHTML = `<p class="note">Couldn’t load heatmaps. Check that the server is running.</p>`; }
}

// ---------- Log ----------
async function renderHistory() {
  const h = await api("/api/history").catch(() => []);
  const bad = h.filter(x => x.verdict === "DEFECT").length;
  $("#histSum").textContent = h.length ? `${h.length} parts inspected, ${bad} flagged as defective.` : "No parts inspected yet. Run one from the Inspect tab.";
  $("#histTbl").innerHTML = h.length ? `<thead><tr><th>Time</th><th>File</th><th>Part</th><th>Model</th><th>Score</th><th>Threshold</th><th>Latency</th><th>Verdict</th></tr></thead><tbody>` +
    h.map(x => `<tr><td>${x.timestamp}</td><td>${esc(x.filename)}</td><td>${esc(x.category)}</td><td>${esc(x.model)}</td><td>${x.score.toFixed(4)}</td><td>${x.threshold ? x.threshold.toFixed(4) : "—"}</td><td>${x.latency_ms} ms</td><td><span class="pill ${x.verdict === "DEFECT" ? "d" : "n"}">${x.verdict === "DEFECT" ? "Defect" : "Pass"}</span></td></tr>`).join("") + "</tbody>" : "";
}
$("#exportBtn").onclick = async () => {
  const h = await api("/api/history").catch(() => []);
  if (!h.length) return;
  const q = v => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const csv = [Object.keys(h[0]).map(q).join(","), ...h.map(r => Object.values(r).map(q).join(","))].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = `inspection_log_${new Date().toISOString().slice(0, 10)}.csv`; a.click();
};

loadBench();
