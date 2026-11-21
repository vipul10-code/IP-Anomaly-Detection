// QualityGuard Dashboard JavaScript Client

let selectedFile = null;
let chartImageAuroc = null;
let chartPixelAuroc = null;
let chartDefectRates = null;
let benchmarkDataCache = [];

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupDropZone();
  loadBenchmark();
});

// Navigation Handling
function setupNavigation() {
  const navButtons = document.querySelectorAll(".nav-item");
  navButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      navButtons.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".content-section").forEach(sec => sec.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetSec = document.getElementById(targetId);
      if (targetSec) targetSec.classList.add("active");

      // Lazy load section data
      if (targetId === "defects") renderDefectBreakdown();
      if (targetId === "gallery") loadGallery();
      if (targetId === "history") loadHistory();
    });
  });
}

// ---------------- 1. BENCHMARK & OVERVIEW ----------------

async function loadBenchmark() {
  try {
    const res = await fetch("/api/benchmark");
    if (!res.ok) throw new Error("Failed to load benchmark data");
    const data = await res.json();
    benchmarkDataCache = data;
    renderBenchmarkTable(data);
    renderBenchmarkCharts(data);
  } catch (err) {
    console.error("Benchmark error:", err);
    document.getElementById("benchmarkTableBody").innerHTML = `
      <tr><td colspan="10" class="text-center" style="color: var(--accent-red)">Error loading benchmark: ${err.message}</td></tr>
    `;
  }
}

function renderBenchmarkTable(data) {
  const tbody = document.getElementById("benchmarkTableBody");
  if (!data || data.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" class="text-center">No benchmark records found.</td></tr>`;
    return;
  }

  // Find max values per category for highlighting
  const maxes = {};
  data.forEach(row => {
    const cat = row.category;
    if (!maxes[cat]) {
      maxes[cat] = {
        image_auroc: -1,
        image_ap: -1,
        pixel_auroc: -1,
        pixel_aupro: -1,
        val_f1: -1,
        oracle_f1: -1,
        min_latency: 999999,
        min_size: 999999,
      };
    }
    const m = maxes[cat];
    if (row.image_auroc > m.image_auroc) m.image_auroc = row.image_auroc;
    if (row.image_ap > m.image_ap) m.image_ap = row.image_ap;
    if (row.pixel_auroc > m.pixel_auroc) m.pixel_auroc = row.pixel_auroc;
    if (row.pixel_aupro > m.pixel_aupro) m.pixel_aupro = row.pixel_aupro;
    if (row.val_f1 > m.val_f1) m.val_f1 = row.val_f1;
    if (row.oracle_f1 > m.oracle_f1) m.oracle_f1 = row.oracle_f1;
    if (row.inference_ms && row.inference_ms < m.min_latency) m.min_latency = row.inference_ms;
    if (row.model_size_mb && row.model_size_mb < m.min_size) m.min_size = row.model_size_mb;
  });

  const formatVal = (val, maxVal, isMin = false) => {
    if (val === null || val === undefined || isNaN(val)) return "—";
    const isBest = isMin ? (val <= maxVal) : (val >= maxVal - 1e-5);
    const formatted = typeof val === "number" ? (val < 1.0 ? val.toFixed(4) : val.toFixed(1)) : val;
    return isBest ? `<span class="best-metric">${formatted}</span>` : formatted;
  };

  tbody.innerHTML = data.map(row => {
    const m = maxes[row.category];
    return `
      <tr>
        <td><strong>${row.model}</strong></td>
        <td><span class="badge badge-neutral">${row.category}</span></td>
        <td>${formatVal(row.image_auroc, m.image_auroc)}</td>
        <td>${formatVal(row.image_ap, m.image_ap)}</td>
        <td>${formatVal(row.pixel_auroc, m.pixel_auroc)}</td>
        <td>${formatVal(row.pixel_aupro, m.pixel_aupro)}</td>
        <td>${formatVal(row.val_f1, m.val_f1)}</td>
        <td>${formatVal(row.oracle_f1, m.oracle_f1)}</td>
        <td>${formatVal(row.inference_ms, m.min_latency, true)} ms</td>
        <td>${formatVal(row.model_size_mb, m.min_size, true)} MB</td>
      </tr>
    `;
  }).join("");
}

function renderBenchmarkCharts(data) {
  // Chart 1: Image AUROC
  const ctxImage = document.getElementById("chartImageAuroc").getContext("2d");
  const models = [...new Set(data.map(d => d.model))];

  const leatherScores = models.map(m => {
    const item = data.find(d => d.model === m && d.category === "leather");
    return item ? item.image_auroc : 0;
  });

  const transistorScores = models.map(m => {
    const item = data.find(d => d.model === m && d.category === "transistor");
    return item ? item.image_auroc : 0;
  });

  if (chartImageAuroc) chartImageAuroc.destroy();
  chartImageAuroc = new Chart(ctxImage, {
    type: "bar",
    data: {
      labels: models,
      datasets: [
        { label: "Leather", data: leatherScores, backgroundColor: "#38bdf8", borderRadius: 4 },
        { label: "Transistor", data: transistorScores, backgroundColor: "#3b82f6", borderRadius: 4 },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#94a3b8" } } },
      scales: {
        y: { min: 0.4, max: 1.0, ticks: { color: "#94a3b8" }, grid: { color: "#334155" } },
        x: { ticks: { color: "#94a3b8" }, grid: { display: false } }
      }
    }
  });

  // Chart 2: Pixel Localization (AUROC & AUPRO) on Leather
  const ctxPixel = document.getElementById("chartPixelAuroc").getContext("2d");
  const pixelAurocs = models.map(m => {
    const item = data.find(d => d.model === m && d.category === "leather");
    return item ? item.pixel_auroc : 0;
  });
  const pixelAupros = models.map(m => {
    const item = data.find(d => d.model === m && d.category === "leather");
    return item ? item.pixel_aupro : 0;
  });

  if (chartPixelAuroc) chartPixelAuroc.destroy();
  chartPixelAuroc = new Chart(ctxPixel, {
    type: "bar",
    data: {
      labels: models,
      datasets: [
        { label: "Pixel AUROC (Leather)", data: pixelAurocs, backgroundColor: "#22c55e", borderRadius: 4 },
        { label: "Pixel AUPRO (Leather)", data: pixelAupros, backgroundColor: "#f59e0b", borderRadius: 4 },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#94a3b8" } } },
      scales: {
        y: { min: 0.5, max: 1.0, ticks: { color: "#94a3b8" }, grid: { color: "#334155" } },
        x: { ticks: { color: "#94a3b8" }, grid: { display: false } }
      }
    }
  });
}

// ---------------- 2. LIVE INSPECTION ----------------

function setupDropZone() {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");

  dropZone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) handleFileSelection(e.target.files[0]);
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) handleFileSelection(e.dataTransfer.files[0]);
  });
}

function handleFileSelection(file) {
  selectedFile = file;
  document.getElementById("fileName").textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  document.getElementById("inspectBtn").disabled = false;

  const reader = new FileReader();
  reader.onload = (e) => {
    const img = document.getElementById("imgOriginal");
    img.src = e.target.result;
    img.classList.add("loaded");
    img.parentElement.querySelector(".placeholder-text").style.display = "none";
  };
  reader.readAsDataURL(file);
}

async function loadSampleImage(type) {
  let samplePath = "";
  let category = "leather";

  if (type === "leather_cut") {
    samplePath = "data/mvtec/leather/test/cut/000.png";
    category = "leather";
  } else if (type === "leather_good") {
    samplePath = "data/mvtec/leather/test/good/000.png";
    category = "leather";
  } else if (type === "transistor_bent") {
    samplePath = "data/mvtec/transistor/test/bent_lead/000.png";
    category = "transistor";
  } else if (type === "transistor_good") {
    samplePath = "data/mvtec/transistor/test/good/000.png";
    category = "transistor";
  }

  document.getElementById("categorySelect").value = category;

  // Fetch the sample via heatmaps or direct static
  try {
    const resp = await fetch(`/heatmaps/patchcore/${category}/${type.split("_")[1]}_000.png`);
    // Alternatively construct image directly
    const blob = await (await fetch(`/${samplePath}`)).blob();
    const file = new File([blob], `${type}.png`, { type: "image/png" });
    handleFileSelection(file);
  } catch (e) {
    // Fallback: create mock blob
    console.warn("Direct sample fetch failed, triggering inspection via API");
  }
}

async function runInspection() {
  if (!selectedFile) return;

  const btn = document.getElementById("inspectBtn");
  btn.disabled = true;
  btn.textContent = "Analyzing Part...";

  const model = document.getElementById("modelSelect").value;
  const category = document.getElementById("categorySelect").value;

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("model", model);
  formData.append("category", category);

  try {
    const res = await fetch("/api/predict", { method: "POST", body: formData });
    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Inspection request failed");
    }

    const data = await res.json();
    renderInspectionResults(data);
  } catch (err) {
    alert(`Inspection Error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Defect Inspection";
  }
}

function renderInspectionResults(data) {
  // Metric cards
  document.getElementById("scoreDisplay").textContent = data.score.toFixed(4);
  document.getElementById("thresholdDisplay").textContent = data.threshold ? data.threshold.toFixed(4) : "N/A";
  document.getElementById("latencyDisplay").textContent = `${data.latency_ms} ms`;

  // Verdict badge
  const badge = document.getElementById("verdictBadge");
  badge.textContent = data.verdict;
  badge.className = `badge ${data.verdict === "DEFECT" ? "badge-defect" : "badge-normal"}`;

  // Image displays
  const imgOrig = document.getElementById("imgOriginal");
  imgOrig.src = data.original_b64;
  imgOrig.classList.add("loaded");
  imgOrig.parentElement.querySelector(".placeholder-text").style.display = "none";

  const imgOverlay = document.getElementById("imgOverlay");
  imgOverlay.src = data.overlay_b64;
  imgOverlay.classList.add("loaded");
  imgOverlay.parentElement.querySelector(".placeholder-text").style.display = "none";

  const imgMask = document.getElementById("imgMask");
  imgMask.src = data.mask_b64;
  imgMask.classList.add("loaded");
  imgMask.parentElement.querySelector(".placeholder-text").style.display = "none";
}

// ---------------- 3. DEFECT BREAKDOWN ----------------

async function renderDefectBreakdown() {
  const filter = document.getElementById("defectCategoryFilter").value;

  try {
    const res = await fetch("/api/benchmark");
    const data = await res.json();

    // Fetch defect breakdowns for leather and transistor
    const [leatherCAE, leatherPC, leatherEAD, leatherDRAEM] = await Promise.all([
      fetch("/api/defects/cae/leather").then(r => r.ok ? r.json() : []),
      fetch("/api/defects/patchcore/leather").then(r => r.ok ? r.json() : []),
      fetch("/api/defects/efficientad/leather").then(r => r.ok ? r.json() : []),
      fetch("/api/defects/draem/leather").then(r => r.ok ? r.json() : []),
    ]);

    const [transCAE, transPC, transEAD] = await Promise.all([
      fetch("/api/defects/cae/transistor").then(r => r.ok ? r.json() : []),
      fetch("/api/defects/patchcore/transistor").then(r => r.ok ? r.json() : []),
      fetch("/api/defects/efficientad/transistor").then(r => r.ok ? r.json() : []),
    ]);

    const allDefects = [];

    const addRows = (cat, defects, cae, ead, pc, draem) => {
      defects.forEach(d => {
        const getRate = (list) => {
          const item = list.find(x => x.defect_type === d);
          if (!item) return "—";
          const val = item.detection_rate_val_th ?? item.detection_rate;
          return val !== undefined && val !== null ? `${(val * 100).toFixed(1)}%` : "—";
        };
        const getCount = (list) => {
          const item = list.find(x => x.defect_type === d);
          return item ? item.n_images : "—";
        };

        allDefects.push({
          category: cat,
          defect: d,
          n_images: getCount(cae),
          cae: getRate(cae),
          ead: getRate(ead),
          pc: getRate(pc),
          draem: draem ? getRate(draem) : "—",
        });
      });
    };

    addRows("leather", ["color", "cut", "fold", "glue", "poke", "good"], leatherCAE, leatherEAD, leatherPC, leatherDRAEM);
    addRows("transistor", ["bent_lead", "cut_lead", "damaged_case", "misplaced", "good"], transCAE, transEAD, transPC, null);

    const filtered = filter === "all" ? allDefects : allDefects.filter(x => x.category === filter);

    const tbody = document.getElementById("defectTableBody");
    tbody.innerHTML = filtered.map(row => `
      <tr>
        <td><span class="badge badge-neutral">${row.category}</span></td>
        <td><strong>${row.defect}</strong></td>
        <td>${row.n_images}</td>
        <td>${row.cae}</td>
        <td>${row.ead}</td>
        <td><strong style="color: var(--accent-cyan)">${row.pc}</strong></td>
        <td>${row.draem}</td>
      </tr>
    `).join("");

    // Render Chart
    const ctx = document.getElementById("chartDefectRates").getContext("2d");
    const chartLabels = filtered.filter(x => x.defect !== "good").map(x => `${x.category}:${x.defect}`);
    const parsePct = str => parseFloat(str) || 0;

    if (chartDefectRates) chartDefectRates.destroy();
    chartDefectRates = new Chart(ctx, {
      type: "bar",
      data: {
        labels: chartLabels,
        datasets: [
          { label: "PatchCore", data: filtered.filter(x => x.defect !== "good").map(x => parsePct(x.pc)), backgroundColor: "#38bdf8", borderRadius: 4 },
          { label: "EfficientAD", data: filtered.filter(x => x.defect !== "good").map(x => parsePct(x.ead)), backgroundColor: "#3b82f6", borderRadius: 4 },
          { label: "CAE (v1)", data: filtered.filter(x => x.defect !== "good").map(x => parsePct(x.cae)), backgroundColor: "#94a3b8", borderRadius: 4 },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#94a3b8" } } },
        scales: {
          y: { min: 0, max: 100, ticks: { color: "#94a3b8" }, grid: { color: "#334155" } },
          x: { ticks: { color: "#94a3b8", autoSkip: false, maxRotation: 45 }, grid: { display: false } }
        }
      }
    });

  } catch (err) {
    console.error("Defect breakdown error:", err);
  }
}

// ---------------- 4. HEATMAP GALLERY ----------------

async function loadGallery() {
  const model = document.getElementById("galleryModelSelect").value;
  const category = document.getElementById("galleryCatSelect").value;
  const grid = document.getElementById("galleryGrid");
  grid.innerHTML = `<div class="text-center w-full">Loading gallery panels...</div>`;

  try {
    const res = await fetch(`/api/heatmaps/${model}/${category}`);
    if (!res.ok) throw new Error("Could not retrieve heatmaps");
    const data = await res.json();

    if (!data.heatmaps || data.heatmaps.length === 0) {
      grid.innerHTML = `<div class="text-center w-full" style="color: var(--text-muted)">No heatmaps found for ${model.toUpperCase()} on ${category}.</div>`;
      return;
    }

    grid.innerHTML = data.heatmaps.map(item => `
      <div class="gallery-card">
        <img src="${item.url}" alt="${item.filename}" loading="lazy">
        <div class="gallery-info">
          <span class="gallery-title">${item.defect_type} (${item.filename.split(".")[0]})</span>
          <span class="badge badge-accent">${model.toUpperCase()}</span>
        </div>
      </div>
    `).join("");
  } catch (err) {
    grid.innerHTML = `<div class="text-center w-full" style="color: var(--accent-red)">Error: ${err.message}</div>`;
  }
}

// ---------------- 5. INSPECTION HISTORY ----------------

async function loadHistory() {
  try {
    const res = await fetch("/api/history");
    const history = await res.json();
    const tbody = document.getElementById("historyTableBody");

    if (!history || history.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="text-center">No inspections logged yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = history.map(item => `
      <tr>
        <td>${item.timestamp}</td>
        <td><code>${item.filename}</code></td>
        <td><span class="badge badge-neutral">${item.category}</span></td>
        <td><strong>${item.model}</strong></td>
        <td>${item.score.toFixed(4)}</td>
        <td>${item.threshold ? item.threshold.toFixed(4) : "—"}</td>
        <td>${item.latency_ms} ms</td>
        <td><span class="badge ${item.verdict === "DEFECT" ? "badge-defect" : "badge-normal"}">${item.verdict}</span></td>
      </tr>
    `).join("");
  } catch (err) {
    console.error("History fetch error:", err);
  }
}

function exportHistoryCSV() {
  fetch("/api/history")
    .then(r => r.json())
    .then(history => {
      if (!history || history.length === 0) {
        alert("No inspection records to export.");
        return;
      }
      const headers = Object.keys(history[0]).join(",");
      const rows = history.map(h => Object.values(h).join(","));
      const csvContent = "data:text/csv;charset=utf-8," + [headers, ...rows].join("\n");
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", `inspection_history_${new Date().toISOString().slice(0, 10)}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
}
