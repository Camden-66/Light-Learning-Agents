const state = {
  session: null,
  play: null,
  mle: null,
  rlPlay: null,
  rlTrain: null,
  emergent: null,
  emergentPool: null,
};

const room = document.getElementById("room");
const logEl = document.getElementById("log");
const statusEl = document.getElementById("status");

function api(path, body) {
  return fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  }).then((r) => r.json());
}

function buildRoom() {
  room.innerHTML = "";
  for (let t = 0; t < 32; t += 1) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "window";
    b.dataset.slot = String(t);
    b.innerHTML = `<span>${t}</span>`;
    b.addEventListener("click", () => onSlot(t));
    room.appendChild(b);
  }
}

function paintWindows() {
  const play = state.play;
  const mle = state.mle;
  [...room.children].forEach((el, t) => {
    el.disabled = !play || play.done;
    el.classList.remove("on", "off", "true", "you", "mle", "rl", "emergent");
    if (!play) return;
    const visits = play.visit_counts[t];
    const ons = play.on_counts[t];
    if (visits > 0) el.classList.add(ons > 0 ? "on" : "off");
    if (play.theta === t) el.classList.add("true");
    if (play.theta_hat === t) el.classList.add("you");
    if (mle && mle.theta_hat === t) el.classList.add("mle");
    if (state.rlPlay && state.rlPlay.theta_hat === t) el.classList.add("rl");
    if (state.emergent && state.emergent.theta_hat === t) el.classList.add("emergent");
  });
}

function paintComparisonControls() {
  const available = Boolean(state.play && state.play.done);
  document.getElementById("run-mle").disabled = !available;
  document.getElementById("run-rl").disabled = !available;
  document.getElementById("run-emergent").disabled = !available;
}

function setStatus(text) {
  statusEl.textContent = text;
}

function addLog(text) {
  const li = document.createElement("li");
  li.textContent = text;
  logEl.prepend(li);
}

function renderMeters() {
  const p = state.play;
  document.getElementById("remaining").textContent = p ? String(p.remaining) : "—";
  document.getElementById("phase").textContent = !p
    ? "—"
    : p.done
      ? "done"
      : p.commit_phase
        ? "commit"
        : "look";
  document.getElementById("you-err").textContent =
    p && p.done && p.absolute_error !== undefined ? String(p.absolute_error) : "not committed";
  document.getElementById("mle-err").textContent =
    state.mle ? String(state.mle.absolute_error) : "not run";
  document.getElementById("rl-err").textContent =
    state.rlPlay ? String(state.rlPlay.absolute_error) : "not run";
  document.getElementById("emergent-err").textContent =
    state.emergent ? String(state.emergent.absolute_error) : "not run";
}

function drawCurve(canvas, values, marks) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#3a3732";
  ctx.beginPath();
  ctx.moveTo(40, h - 28);
  ctx.lineTo(w - 12, h - 28);
  ctx.stroke();
  if (!values) {
    ctx.fillStyle = "#6d6456";
    ctx.font = "16px Palatino, serif";
    ctx.fillText("Peak still hidden. Commit your estimate to reveal the true curve.", 48, h / 2);
    return;
  }
  const x = (i) => 40 + (i / 31) * (w - 60);
  const y = (v) => h - 28 - v * (h - 50);
  ctx.strokeStyle = "#e2b23d";
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((v, i) => {
    const X = x(i);
    const Y = y(v);
    if (i === 0) ctx.moveTo(X, Y);
    else ctx.lineTo(X, Y);
  });
  ctx.stroke();
  if (marks && marks.inferred) {
    ctx.strokeStyle = "#c084fc";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    marks.inferred.forEach((v, i) => {
      const X = x(i);
      const Y = y(v);
      if (i === 0) ctx.moveTo(X, Y);
      else ctx.lineTo(X, Y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (marks) {
    if (marks.theta != null) {
      ctx.strokeStyle = "#c45c26";
      ctx.beginPath();
      ctx.moveTo(x(marks.theta), 12);
      ctx.lineTo(x(marks.theta), h - 28);
      ctx.stroke();
    }
    if (marks.you != null) {
      ctx.strokeStyle = "#ebe4d4";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x(marks.you), 12);
      ctx.lineTo(x(marks.you), h - 28);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (marks.mle != null) {
      ctx.strokeStyle = "#6a8f4e";
      ctx.beginPath();
      ctx.moveTo(x(marks.mle), 12);
      ctx.lineTo(x(marks.mle), h - 28);
      ctx.stroke();
    }
    if (marks.rl != null) {
      ctx.strokeStyle = "#6ea8d4";
      ctx.beginPath();
      ctx.moveTo(x(marks.rl), 12);
      ctx.lineTo(x(marks.rl), h - 28);
      ctx.stroke();
    }
    if (marks.emergent != null) {
      ctx.strokeStyle = "#c084fc";
      ctx.beginPath();
      ctx.moveTo(x(marks.emergent), 12);
      ctx.lineTo(x(marks.emergent), h - 28);
      ctx.stroke();
    }
  }
}

function drawLL(canvas, rows) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!rows) {
    ctx.fillStyle = "#6d6456";
    ctx.font = "16px Palatino, serif";
    ctx.fillText("MLE has not run. It scores every candidate peak by log-likelihood.", 48, h / 2);
    return;
  }
  const lls = rows.map((r) => r.ll);
  const min = Math.min(...lls);
  const max = Math.max(...lls);
  const span = max - min || 1;
  const barW = (w - 50) / rows.length;
  rows.forEach((r, i) => {
    const bh = ((r.ll - min) / span) * (h - 40);
    ctx.fillStyle = r.best ? "#6a8f4e" : "#5a4e32";
    ctx.fillRect(40 + i * barW, h - 24 - bh, Math.max(barW - 1, 1), bh);
  });
}

function paintObs(obs) {
  const wrap = document.getElementById("obs-strip");
  if (!obs) {
    wrap.hidden = true;
    return;
  }
  wrap.hidden = false;
  const rows = wrap.querySelector(".obs-rows");
  rows.innerHTML = "";
  const blocks = [obs.slice(0, 32), obs.slice(32, 64)];
  blocks.forEach((block) => {
    const row = document.createElement("div");
    row.className = "obs-row";
    const m = Math.max(...block, 0.001);
    block.forEach((v) => {
      const c = document.createElement("div");
      c.className = "obs-cell";
      const g = Math.round(40 + (v / m) * 180);
      c.style.background = `rgb(${g},${g - 20},40)`;
      row.appendChild(c);
    });
    rows.appendChild(row);
  });
}

function refreshCharts() {
  const p = state.play || {};
  const m = state.mle || {};
  drawCurve(document.getElementById("curve"), p.curve || m.curve || (state.emergent && state.emergent.curve), {
    theta: p.theta ?? m.theta ?? (state.emergent && state.emergent.theta),
    you: p.theta_hat,
    mle: m.theta_hat,
    rl: state.rlPlay ? state.rlPlay.theta_hat : null,
    emergent: state.emergent ? state.emergent.theta_hat : null,
    inferred: state.emergent ? state.emergent.inferred_curve : null,
  });
  drawLL(document.getElementById("ll"), m.likelihood);
}

async function newEpisode() {
  const budget = Number(document.getElementById("budget").value);
  state.mle = null;
  state.rlPlay = null;
  state.emergent = null;
  logEl.innerHTML = "";
  state.play = await api("/api/new", { budget });
  state.session = state.play.session;
  setStatus(`Episode ${state.play.episode_id.slice(0, 8)}… Look, then commit. Compare MLE / RL / emergent on this peak after that.`);
  addLog(`Start. Budget ${budget}. Same frozen RL policy will be reused; emergent runs on this episode.`);
  paintWindows();
  paintComparisonControls();
  renderMeters();
  paintObs(state.play.obs);
  refreshCharts();
  ensureRl(budget);
}

async function onSlot(t) {
  if (!state.session) {
    setStatus("Start an episode first.");
    return;
  }
  const p = state.play;
  if (p.done) return;
  if (!p.commit_phase) {
    const res = await api("/api/look", { session: state.session, slot: t });
    if (res.error) {
      setStatus(res.error);
      return;
    }
    state.play = res;
    addLog(`Look t=${t} → ${res.light_on ? "ON" : "OFF"}. ${res.remaining} left.`);
    if (res.commit_phase) setStatus("Budget spent. Click one slot — that is your peak estimate.");
  } else {
    const res = await api("/api/commit", { session: state.session, slot: t });
    if (res.error) {
      setStatus(res.error);
      return;
    }
    state.play = res;
    addLog(`You guessed θ̂=${t}. True θ=${res.theta}, error ${res.absolute_error}.`);
    setStatus(`True peak ${res.theta}. Now compare oracle MLE, the frozen RL policy, and the emergent model on this episode.`);
  }
  paintWindows();
  paintComparisonControls();
  renderMeters();
  paintObs(state.play.obs);
  refreshCharts();
}

async function runMle() {
  if (!state.session) {
    setStatus("Start an episode first.");
    return;
  }
  if (!state.play || !state.play.done) {
    setStatus("Commit your estimate before revealing comparison results.");
    return;
  }
  const res = await api("/api/mle", { session: state.session });
  if (res.error) {
    setStatus(res.error);
    return;
  }
  state.mle = res;
  const n = state.mle.trace.filter((e) => e.phase === "observe").length;
  addLog(`MLE sampled ${n} uniform slots, guessed ${state.mle.theta_hat}, error ${state.mle.absolute_error}.`);
  setStatus("MLE scored every candidate with the true formula. Green bar / outline is its guess.");
  paintWindows();
  renderMeters();
  refreshCharts();
}

function drawRl(canvas, history) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#3a3732";
  ctx.beginPath();
  ctx.moveTo(40, h - 28);
  ctx.lineTo(w - 12, h - 28);
  ctx.stroke();
  if (!history || !history.length) {
    ctx.fillStyle = "#6d6456";
    ctx.font = "16px Palatino, serif";
    ctx.fillText("Train the REINFORCE demo to see its rolling MAE.", 48, h / 2);
    return;
  }
  const xs = history.map((p) => p.episode);
  const ys = history.map((p) => p.mae_window);
  const maxX = Math.max(...xs, 1);
  const maxY = Math.max(...ys, 1);
  const x = (ep) => 40 + (ep / maxX) * (w - 60);
  const y = (v) => h - 28 - (v / maxY) * (h - 50);
  ctx.strokeStyle = "#6ea8d4";
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((p, i) => {
    const X = x(p.episode);
    const Y = y(p.mae_window);
    if (i === 0) ctx.moveTo(X, Y);
    else ctx.lineTo(X, Y);
  });
  ctx.stroke();
}

function renderRlTrain(data) {
  state.rlTrain = data;
  const status = document.getElementById("rl-status");
  if (!data || !data.trained) {
    document.getElementById("rl-n").textContent = "—";
    document.getElementById("rl-mae").textContent = "—";
    document.getElementById("rl-hit").textContent = "—";
    drawRl(document.getElementById("rl-chart"), null);
    return;
  }
  document.getElementById("rl-n").textContent = String(data.n_episodes);
  document.getElementById("rl-mae").textContent = data.demo_mae.toFixed(2);
  document.getElementById("rl-hit").textContent = `${(data.demo_hit_within_one * 100).toFixed(0)}%`;
  const seedNote = data.train_seed == null ? "" : ` train seed ${data.train_seed}.`;
  status.textContent = `Frozen REINFORCE for B=${data.budget}.${seedNote} Smoke MAE ${data.demo_mae.toFixed(2)} on other rooms. Compare applies this policy to the current episode.`;
  drawRl(document.getElementById("rl-chart"), data.history);
}

async function ensureRl(budget, { retrain = false, seed = 0 } = {}) {
  if (!retrain) {
    const existing = await api(`/api/rl?budget=${budget}`);
    if (existing.trained) {
      renderRlTrain(existing);
      return existing;
    }
  }
  document.getElementById("rl-status").textContent = retrain
    ? `Retraining REINFORCE with seed ${seed} (other rooms, not this episode)…`
    : `Preparing a REINFORCE policy for B=${budget} on other rooms…`;
  const data = await api("/api/rl/train", { budget, episodes: 250, seed });
  if (data.error) {
    document.getElementById("rl-status").textContent = data.error;
    return data;
  }
  renderRlTrain(data);
  addLog(`REINFORCE policy for B=${budget} ready (seed ${data.train_seed}). Reused until you retrain.`);
  return data;
}

async function trainRl() {
  const btn = document.getElementById("train-rl");
  const budget = Number(document.getElementById("budget").value);
  btn.disabled = true;
  try {
    await ensureRl(budget, { retrain: true, seed: Math.floor(Math.random() * 1e9) });
  } finally {
    btn.disabled = false;
  }
}

async function runRl() {
  if (!state.session) {
    setStatus("Start an episode first.");
    return;
  }
  if (!state.play || !state.play.done) {
    setStatus("Commit your estimate before revealing comparison results.");
    return;
  }
  const budget = Number(document.getElementById("budget").value);
  await ensureRl(budget);
  const res = await api("/api/rl/play", { session: state.session });
  if (res.error) {
    setStatus(res.error);
    document.getElementById("rl-status").textContent = res.error;
    return;
  }
  state.rlPlay = res;
  addLog(`Frozen REINFORCE guessed ${res.theta_hat}, error ${res.absolute_error} on this episode.`);
  setStatus("Blue = frozen RL policy on this episode. Violet = emergent. Green = oracle MLE.");
  paintWindows();
  renderMeters();
  refreshCharts();
}

function renderEmergentMeters() {
  const e = state.emergent;
  const pool = state.emergentPool;
  document.getElementById("em-n").textContent = pool && pool.pooled ? String(pool.n_episodes) : "—";
  if (e) {
    document.getElementById("em-sigma").textContent = Number(e.sigma).toFixed(1);
    document.getElementById("em-mass").textContent = `${(Number(e.sigma_mass_at_3) * 100).toFixed(0)}%`;
    document.getElementById("em-ab").textContent = `${Number(e.background).toFixed(2)}, ${Number(e.amplitude).toFixed(2)}`;
  } else {
    document.getElementById("em-sigma").textContent = "—";
    document.getElementById("em-mass").textContent =
      pool && pool.pooled ? `${(Number(pool.sigma_mass_at_3) * 100).toFixed(0)}%` : "—";
    document.getElementById("em-ab").textContent = "—";
  }
}

function renderEmergentPool(data) {
  state.emergentPool = data;
  const status = document.getElementById("emergent-status");
  if (!data || !data.pooled) {
    status.textContent = "No pooled shape yet. Compare still works in-episode only.";
    renderEmergentMeters();
    return;
  }
  status.textContent = `Pooled ${data.n_episodes} rooms at budget ${data.budget}. Mass on σ=3 is ${(data.sigma_mass_at_3 * 100).toFixed(0)}%. ${data.note}`;
  renderEmergentMeters();
}

async function poolEmergent() {
  const btn = document.getElementById("pool-emergent");
  const budget = Number(document.getElementById("budget").value);
  btn.disabled = true;
  document.getElementById("emergent-status").textContent = "Pooling lamp shape from training rooms… a few seconds.";
  try {
    const data = await api("/api/emergent/pool", { budget, episodes: 8, seed: 0 });
    if (data.error) {
      document.getElementById("emergent-status").textContent = data.error;
      return;
    }
    renderEmergentPool(data);
    addLog(`Pooled emergent shape on ${data.n_episodes} rooms. Mass on σ=3: ${(data.sigma_mass_at_3 * 100).toFixed(0)}%.`);
  } finally {
    btn.disabled = false;
  }
}

async function runEmergent() {
  if (!state.session) {
    setStatus("Start an episode first.");
    return;
  }
  if (!state.play || !state.play.done) {
    setStatus("Commit your estimate before revealing comparison results.");
    return;
  }
  const res = await api("/api/emergent/play", { session: state.session });
  if (res.error) {
    setStatus(res.error);
    return;
  }
  state.emergent = res;
  addLog(
    `Emergent (${res.condition}) guessed ${res.theta_hat}, error ${res.absolute_error}, σ=${Number(res.sigma).toFixed(1)}, mass@3=${(Number(res.sigma_mass_at_3) * 100).toFixed(0)}%.`
  );
  setStatus("Violet = emergent peak and dashed inferred P(on|t). It did not receive the canonical formula.");
  paintWindows();
  renderMeters();
  renderEmergentMeters();
  refreshCharts();
}

async function loadMetrics() {
  const metrics = await api("/api/metrics");
  const host = document.getElementById("metric-cards");
  document.getElementById("metrics-warning").textContent = metrics.warning;
  host.innerHTML = "";
  Object.keys(metrics.budgets || {})
    .sort((a, b) => Number(a) - Number(b))
    .forEach((b) => {
      const s = metrics.budgets[b];
      const el = document.createElement("div");
      el.className = "card";
      el.innerHTML = `<span>Budget B=${b}</span><b>${s.mae.toFixed(2)}</b><span>Demo MAE · hit within 1 slot ${(s.hit_within_one * 100).toFixed(0)}%</span>`;
      host.appendChild(el);
    });
}

document.getElementById("new").addEventListener("click", newEpisode);
document.getElementById("budget").addEventListener("change", newEpisode);
document.getElementById("run-mle").addEventListener("click", runMle);
document.getElementById("run-rl").addEventListener("click", runRl);
document.getElementById("run-emergent").addEventListener("click", runEmergent);
document.getElementById("pool-emergent").addEventListener("click", poolEmergent);
document.getElementById("train-rl").addEventListener("click", trainRl);
buildRoom();
paintComparisonControls();
drawCurve(document.getElementById("curve"), null);
drawLL(document.getElementById("ll"), null);
drawRl(document.getElementById("rl-chart"), null);
loadMetrics();
api("/api/emergent").then(renderEmergentPool);
newEpisode();
