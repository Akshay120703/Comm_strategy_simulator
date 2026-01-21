// Adversarial Communication Strategy Simulator
// Discrete-time RL-style adaptation for a transmitter and jammer.

const BINS = 48; // frequency bins over the channel
const TX_FREQ_SLOTS = 12; // discrete choices transmitter can hop across
const MODS = ["BPSK", "QPSK", "16QAM"];
const MOD_EFF = { BPSK: 1, QPSK: 2, "16QAM": 4 };
const TX_POWER_LEVELS = [0.45, 0.72, 1.0];
const JAM_POWER_LEVELS = [0.35, 0.65, 1.0];
const JAM_SPANS = [2, 4, 8, 12];
const HISTORY_POINTS = 180;

const dom = {};
const state = {
  t: 0,
  paused: false,
  freezeLearning: false,
  jamAdaptive: true,
  tx: { q: [], eps: 0.08, lr: 0.18, lastAction: null },
  jam: { q: [], eps: 0.12, lr: 0.14, lastAction: null },
  history: [],
  chart: null,
  spectrumCtx: null,
};

function byId(id) {
  return document.getElementById(id);
}

function setupDomRefs() {
  [
    "snrInput",
    "bwInput",
    "noiseInput",
    "jamPowerInput",
    "jamSpanInput",
    "txPowerInput",
    "lrInput",
    "epsInput",
    "freezeLearningToggle",
    "jamAdaptiveToggle",
    "mutateBtn",
    "restartBtn",
    "pauseBtn",
    "statusDot",
    "statusText",
    "timeline",
    "stateGrid",
    "spectrumCanvas",
  ].forEach((id) => (dom[id] = byId(id)));
}

function buildActionSpace() {
  const actions = [];
  for (let f = 0; f < TX_FREQ_SLOTS; f++) {
    for (const mod of MODS) {
      for (const p of TX_POWER_LEVELS) {
        actions.push({ fSlot: f, mod, p });
      }
    }
  }
  return actions;
}

function buildJamActions() {
  const actions = [];
  for (let f = 0; f < TX_FREQ_SLOTS; f++) {
    for (const span of JAM_SPANS) {
      for (const p of JAM_POWER_LEVELS) {
        actions.push({ fSlot: f, span, p });
      }
    }
  }
  return actions;
}

const TX_ACTIONS = buildActionSpace();
const JAM_ACTIONS = buildJamActions();

function initQ(table, size) {
  table.length = size;
  for (let i = 0; i < size; i += 1) table[i] = 0;
}

function argMax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

function pickAction(q, eps) {
  if (Math.random() < eps) return Math.floor(Math.random() * q.length);
  return argMax(q);
}

function linToDb(x) {
  return 10 * Math.log10(x);
}
function dbToLin(db) {
  return Math.pow(10, db / 10);
}

function resetSimulation() {
  state.t = 0;
  initQ(state.tx.q, TX_ACTIONS.length);
  initQ(state.jam.q, JAM_ACTIONS.length);
  state.history = [];
  state.paused = false;
  dom.statusText.textContent = "Running";
  dom.statusDot.classList.add("live");
  dom.statusDot.classList.remove("paused");
}

function readSettings() {
  return {
    snrDb: Number(dom.snrInput.value),
    bwMHz: Number(dom.bwInput.value),
    noiseDensity: Number(dom.noiseInput.value), // mW/Hz
    jamMaxPower: Number(dom.jamPowerInput.value),
    jamMaxSpan: Number(dom.jamSpanInput.value),
    txMaxPower: Number(dom.txPowerInput.value),
    lr: Number(dom.lrInput.value),
    eps: Number(dom.epsInput.value),
    freeze: dom.freezeLearningToggle.checked,
    jamAdaptive: dom.jamAdaptiveToggle.checked,
  };
}

function mapFreqSlotToBin(slot) {
  const binsPerSlot = Math.floor(BINS / TX_FREQ_SLOTS);
  const base = slot * binsPerSlot;
  return Math.min(BINS - 1, base + Math.floor(binsPerSlot / 2));
}

function computeOverlap(txBin, jamCenter, spanBins) {
  const half = Math.floor(spanBins / 2);
  const start = Math.max(0, jamCenter - half);
  const end = Math.min(BINS - 1, jamCenter + half);
  if (txBin < start || txBin > end) return 0;
  const dist = Math.abs(txBin - jamCenter) + 1;
  return Math.max(0, 1 - dist / (half + 1)); // triangular weighting
}

function stepSimulation() {
  const cfg = readSettings();
  state.tx.lr = cfg.lr;
  state.tx.eps = cfg.eps;
  state.freezeLearning = cfg.freeze;
  state.jamAdaptive = cfg.jamAdaptive;

  const txIdx = pickAction(state.tx.q, state.freezeLearning ? 0 : state.tx.eps);
  const txAct = TX_ACTIONS[txIdx];
  const jamIdx = state.jamAdaptive
    ? pickAction(state.jam.q, state.freezeLearning ? 0 : state.jam.eps)
    : Math.floor(Math.random() * JAM_ACTIONS.length);
  const jamAct = JAM_ACTIONS[jamIdx];

  // Map to physical values
  const txBin = mapFreqSlotToBin(txAct.fSlot);
  const jamCenter = mapFreqSlotToBin(jamAct.fSlot);
  const jamSpanBins = Math.min(jamAct.span, cfg.jamMaxSpan);

  const snrLin = dbToLin(cfg.snrDb);
  const bwHz = cfg.bwMHz * 1e6;
  const noisePower = cfg.noiseDensity * bwHz;
  const txPower = cfg.txMaxPower * txAct.p;
  const jamPower = cfg.jamMaxPower * jamAct.p;
  const overlapFactor = computeOverlap(txBin, jamCenter, jamSpanBins);
  const interference = jamPower * overlapFactor;
  const signal = txPower * snrLin * MOD_EFF[txAct.mod];
  const sinr = signal / (noisePower + interference + 1);
  const spectralEff = Math.log2(1 + sinr);
  const throughputMbps = (bwHz / 1e6) * spectralEff;
  const jammingIntensity = interference / (interference + noisePower + 1);

  const txReward = throughputMbps - 0.02 * txPower - 6 * overlapFactor;
  const jamReward = jammingIntensity * jamPower - 0.08 * throughputMbps;

  if (!state.freezeLearning) {
    state.tx.q[txIdx] += state.tx.lr * (txReward - state.tx.q[txIdx]);
    if (state.jamAdaptive) {
      state.jam.q[jamIdx] += state.jam.lr * (jamReward - state.jam.q[jamIdx]);
    }
  }

  state.tx.lastAction = { ...txAct, txBin };
  state.jam.lastAction = { ...jamAct, jamCenter, spanBins: jamSpanBins };

  state.history.push({
    t: state.t,
    throughputMbps,
    jammingIntensity,
    sinrDb: linToDb(sinr),
    overlapFactor,
    tx: state.tx.lastAction,
    jam: state.jam.lastAction,
  });
  if (state.history.length > HISTORY_POINTS) state.history.shift();

  state.t += 1;

  updateCharts();
  drawSpectrum();
  renderTimeline();
  renderState();
}

function initChart() {
  const ctx = document.getElementById("throughputChart");
  state.chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Throughput (Mbps)",
          data: [],
          tension: 0.15,
          borderColor: "#22c55e",
          backgroundColor: "rgba(34,197,94,0.15)",
          fill: true,
        },
        {
          label: "Jamming Intensity",
          data: [],
          tension: 0.15,
          borderColor: "#f472b6",
          backgroundColor: "rgba(244,114,182,0.12)",
          yAxisID: "y1",
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      animation: false,
      interaction: { intersect: false, mode: "index" },
      plugins: { legend: { labels: { color: "#e2e8f0" } } },
      scales: {
        x: { ticks: { color: "#cbd5e1" }, grid: { color: "#1f2937" } },
        y: { ticks: { color: "#cbd5e1" }, grid: { color: "#1f2937" } },
        y1: {
          position: "right",
          min: 0,
          max: 1,
          ticks: { color: "#f472b6" },
          grid: { display: false },
        },
      },
    },
  });
}

function updateCharts() {
  if (!state.chart) return;
  const labels = state.history.map((h) => h.t);
  state.chart.data.labels = labels;
  state.chart.data.datasets[0].data = state.history.map((h) => h.throughputMbps);
  state.chart.data.datasets[1].data = state.history.map((h) => h.jammingIntensity);
  state.chart.update("none");
}

function drawSpectrum() {
  const ctx = state.spectrumCtx;
  if (!ctx) return;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  const binW = w / BINS;

  // Base grid
  for (let i = 0; i < BINS; i++) {
    ctx.fillStyle = i % 2 === 0 ? "#0f172a" : "#0b1220";
    ctx.fillRect(i * binW, 0, binW + 1, h);
  }

  if (!state.tx.lastAction || !state.jam.lastAction) return;
  const txBin = state.tx.lastAction.txBin;
  const jamCenter = state.jam.lastAction.jamCenter;
  const span = state.jam.lastAction.spanBins;
  const half = Math.floor(span / 2);
  const start = jamCenter - half;
  const end = jamCenter + half;

  // Jammer heat
  for (let b = start; b <= end; b++) {
    if (b < 0 || b >= BINS) continue;
    const overlap = computeOverlap(b, jamCenter, span);
    const alpha = 0.1 + overlap * 0.55;
    ctx.fillStyle = `rgba(244,114,182,${alpha})`;
    ctx.fillRect(b * binW, 0, binW + 1, h);
  }

  // Transmitter marker
  ctx.fillStyle = "rgba(34,197,94,0.7)";
  ctx.fillRect(txBin * binW, h * 0.15, binW, h * 0.7);

  // Overlap highlight
  if (txBin >= start && txBin <= end) {
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.fillRect(txBin * binW, h * 0.3, binW, h * 0.4);
  }

  // Axes annotation
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "12px Inter, sans-serif";
  ctx.fillText("Frequency bins →", 10, h - 12);
}

function renderTimeline() {
  const container = dom.timeline;
  container.innerHTML = "";
  const recent = state.history.slice(-20).reverse();
  recent.forEach((h) => {
    const row = document.createElement("div");
    row.className = "timeline-item";
    row.innerHTML = `
      <span class="tag tx">Tx</span>
      <span>f${h.tx.fSlot} | ${h.tx.mod} | ${Math.round(h.tx.p * 100)}% P</span>
      <span class="tag jam">Jam</span>
      <span>f${h.jam.fSlot} | span ${h.jam.spanBins} | ${Math.round(
      h.jam.p * 100
    )}% P</span>
      <span style="color:#22c55e;">${h.throughputMbps.toFixed(2)} Mbps</span>
      <span style="color:#f472b6;">J ${h.jammingIntensity.toFixed(2)}</span>
    `;
    container.appendChild(row);
  });
}

function tile(label, value, sub = "") {
  return `
    <div class="state-tile">
      <h4>${label}</h4>
      <div class="value">${value}</div>
      ${sub ? `<small>${sub}</small>` : ""}
    </div>
  `;
}

function renderState() {
  const latest = state.history[state.history.length - 1];
  if (!latest) return;
  dom.stateGrid.innerHTML = `
    ${tile(
      "Throughput",
      `${latest.throughputMbps.toFixed(2)} Mbps`,
      `SINR ${latest.sinrDb.toFixed(2)} dB`
    )}
    ${tile(
      "Jamming Intensity",
      latest.jammingIntensity.toFixed(3),
      `Overlap ${latest.overlapFactor.toFixed(2)}`
    )}
    ${tile(
      "Tx Strategy",
      `${latest.tx.mod} @ f${latest.tx.fSlot}`,
      `Power ${Math.round(latest.tx.p * 100)}%`
    )}
    ${tile(
      "Jam Strategy",
      `f${latest.jam.fSlot} span ${latest.jam.spanBins}`,
      `Power ${Math.round(latest.jam.p * 100)}%`
    )}
    ${tile(
      "Learning",
      state.freezeLearning ? "Frozen" : "Updating",
      `ε ${state.tx.eps.toFixed(2)} | α ${state.tx.lr.toFixed(2)}`
    )}
  `;
}

function attachHandlers() {
  dom.restartBtn.addEventListener("click", resetSimulation);
  dom.pauseBtn.addEventListener("click", () => {
    state.paused = !state.paused;
    dom.statusText.textContent = state.paused ? "Paused" : "Running";
    dom.statusDot.classList.toggle("live", !state.paused);
    dom.statusDot.classList.toggle("paused", state.paused);
    dom.pauseBtn.textContent = state.paused ? "Resume" : "Pause";
  });
  dom.freezeLearningToggle.addEventListener("change", () => {
    state.freezeLearning = dom.freezeLearningToggle.checked;
  });
  dom.jamAdaptiveToggle.addEventListener("change", () => {
    state.jamAdaptive = dom.jamAdaptiveToggle.checked;
  });
  dom.mutateBtn.addEventListener("click", () => {
    // inject a new jammer prior by biasing Q-table randomly
    for (let i = 0; i < state.jam.q.length; i++) {
      state.jam.q[i] = (Math.random() - 0.5) * 4;
    }
  });
}

function main() {
  setupDomRefs();
  state.spectrumCtx = dom.spectrumCanvas.getContext("2d");
  resetSimulation();
  initChart();
  attachHandlers();
  setInterval(() => {
    if (!state.paused) stepSimulation();
  }, 350);
}

document.addEventListener("DOMContentLoaded", main);
