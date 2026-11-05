const $ = (id) => document.getElementById(id);

const fileInput = $("fileInput");
const dropZone = $("dropZone");
const uploadForm = $("uploadForm");
const analyzeBtn = $("analyzeBtn");
const fileLabel = $("fileLabel");
const progressPanel = $("progressPanel");
const resultPanel = $("resultPanel");
const healthStatus = $("healthStatus");

let selectedFile = null;

async function api(path, options) {
  const res = await fetch(path, options);
  let data = null;
  try {
    data = await res.json();
  } catch (_) {
    data = null;
  }
  if (!res.ok) {
    const detail = (data && (data.detail || data.message)) || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function setHealth(ok, text) {
  healthStatus.textContent = text;
  healthStatus.className = "status " + (ok ? "ok" : "bad");
}

async function refreshHealth() {
  try {
    const h = await api("/api/health");
    if (!h.checkpoint_exists) {
      setHealth(false, "Checkpoint missing — train the model first");
      return;
    }
    setHealth(true, h.model_loaded ? "Ready" : "Starting…");
  } catch (err) {
    setHealth(false, "Backend offline");
  }
}

function renderResult(data) {
  resultPanel.hidden = false;
  const badge = $("verdictBadge");
  badge.textContent = data.verdict;
  badge.className = "verdict " + data.verdict;

  $("meanScore").textContent = Number(data.mean_score).toFixed(4);
  $("verdictDetail").textContent = data.verdict_detail || "";

  $("statGrid").innerHTML = [
    ["Frames", data.frames_seen],
    ["Faces", data.faces_kept],
    ["Sequences", data.n_sequences],
    ["Majority", String(data.video_majority || "").toUpperCase()],
    ["Threshold", data.threshold],
    ["Runtime", `${Number(data.elapsed_seconds).toFixed(1)}s`],
  ]
    .map(
      ([k, v]) =>
        `<div class="stat"><span>${k}</span><strong>${v}</strong></div>`
    )
    .join("");

  const scores = data.sequence_scores || [];
  const thr = Number(data.threshold) || 0.5;
  $("scoreBars").innerHTML = scores
    .map((score, i) => {
      const fake = score >= thr;
      const width = Math.max(2, Math.round(score * 100));
      return `<div class="bar-row">
        <span>Seq ${i}</span>
        <span>${score.toFixed(3)}</span>
        <div class="bar-track"><div class="bar-fill ${fake ? "fake" : "real"}" style="width:${width}%"></div></div>
      </div>`;
    })
    .join("");

  const faces = (data.urls && data.urls.faces) || [];
  $("faceGallery").innerHTML = faces
    .map((url, i) => {
      const score = scores[i];
      const label =
        score == null ? "" : score >= thr ? `FAKE ${score.toFixed(3)}` : `REAL ${score.toFixed(3)}`;
      return `<figure><img src="${url}" alt="face ${i}"/><figcaption>${label}</figcaption></figure>`;
    })
    .join("") || `<p class="empty">No face crops</p>`;

  const cam = data.urls && data.urls.gradcam;
  const camImg = $("gradcamImg");
  const camHeading = $("camHeading");
  if (cam) {
    camImg.src = cam + "?t=" + Date.now();
    camImg.hidden = false;
    camHeading.hidden = false;
  } else {
    camImg.hidden = true;
    camHeading.hidden = true;
  }

  $("htmlReportLink").href = data.urls.report_html;
  $("jsonLink").href = data.urls.analysis_json;
  // Keep the on-screen note short; full research disclaimer stays in the HTML report.
  const verdict = String(data.verdict || "").toUpperCase();
  const p = Number(data.mean_score);
  $("resultNote").textContent = Number.isFinite(p)
    ? `${verdict}: estimated chance this face is manipulated is ${(p * 100).toFixed(1)}%.`
    : "";
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runUploadAnalyze() {
  if (!selectedFile) return;
  progressPanel.hidden = false;
  resultPanel.hidden = true;
  analyzeBtn.disabled = true;

  const body = new FormData();
  body.append("file", selectedFile);
  body.append("folder_hint", "upload");

  try {
    const data = await api("/api/analyze", { method: "POST", body });
    renderResult(data);
    await refreshReports();
  } catch (err) {
    alert(err.message || String(err));
  } finally {
    progressPanel.hidden = true;
    analyzeBtn.disabled = !selectedFile;
  }
}

async function refreshReports() {
  const list = $("reportList");
  try {
    const data = await api("/api/reports");
    if (!data.items.length) {
      list.innerHTML = `<p class="empty">No reports yet.</p>`;
      return;
    }
    list.innerHTML = data.items
      .map((item) => {
        const href = item.urls.report_html;
        return `<div class="row">
          <div>
            <strong>${item.verdict}</strong> · ${item.media_name || item.id}
            <p class="meta">P(fake)=${Number(item.mean_score).toFixed(4)} · ${item.created_at || ""}</p>
          </div>
          <a class="btn" href="${href}" target="_blank" rel="noopener">Open</a>
        </div>`;
      })
      .join("");
  } catch (err) {
    list.innerHTML = `<p class="empty">${err.message}</p>`;
  }
}

function chooseFile(file) {
  selectedFile = file || null;
  fileLabel.textContent = selectedFile ? selectedFile.name : "No file selected";
  analyzeBtn.disabled = !selectedFile;
}

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => chooseFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("drag");
  });
});
["dragleave", "drop"].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag");
  });
});
dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) chooseFile(file);
});

uploadForm.addEventListener("submit", (e) => {
  e.preventDefault();
  runUploadAnalyze();
});

$("refreshReports").addEventListener("click", refreshReports);

refreshHealth();
refreshReports();
setInterval(refreshHealth, 15000);
