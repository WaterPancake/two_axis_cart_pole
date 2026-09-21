"use strict";

const controllerSelect = document.querySelector("#controllerSelect");
const scenarioSelect = document.querySelector("#scenarioSelect");
const controllerNote = document.querySelector("#controllerNote");
const resetButton = document.querySelector("#resetButton");
const pauseButton = document.querySelector("#pauseButton");
const pauseIcon = document.querySelector("#pauseIcon");
const authorityCard = document.querySelector("#authorityCard");
const authorityText = document.querySelector("#authorityText");
const modeText = document.querySelector("#modeText");
const runtimeDot = document.querySelector("#runtimeDot");
const runtimeText = document.querySelector("#runtimeText");
const simClock = document.querySelector("#simClock");
const canvasAlert = document.querySelector("#canvasAlert");
const worldCanvas = document.querySelector("#worldCanvas");
const traceCanvas = document.querySelector("#traceCanvas");

const heldKeys = new Set();
const history = [];
let config = null;
let latestState = null;
let paused = false;
let connected = false;
let inputRevision = 0;
let mutationQueue = Promise.resolve();
const clientId = window.crypto && window.crypto.randomUUID
  ? window.crypto.randomUUID()
  : `browser-${Date.now()}-${Math.random().toString(16).slice(2)}`;

const keyAliases = {
  w: "w", a: "a", s: "s", d: "d",
  arrowup: "arrowup", arrowleft: "arrowleft",
  arrowdown: "arrowdown", arrowright: "arrowright",
};

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function mutate(path, body) {
  const operation = mutationQueue.catch(() => undefined).then(() => api(path, body));
  mutationQueue = operation;
  return operation;
}

function setConnected(value) {
  connected = value;
  runtimeDot.classList.toggle("online", value);
  runtimeText.textContent = value ? "simulation online" : "reconnecting";
}

function populateSelect(select, items, selected) {
  select.replaceChildren();
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.label + (item.enabled === false ? " (unavailable)" : "");
    option.disabled = item.enabled === false;
    option.dataset.description = item.description || "";
    select.append(option);
  }
  select.value = selected;
}

async function initialize() {
  try {
    config = await api("/api/config");
    populateSelect(controllerSelect, config.controllers, config.selected_controller);
    populateSelect(scenarioSelect, config.scenarios, config.selected_scenario);
    updateControllerNote();
    const loadedPolicies = config.policy_checkpoints
      ? Object.entries(config.policy_checkpoints)
        .filter(([, path]) => path)
        .map(([name, path]) => `${name}: ${path}`)
        .join(" | ")
      : "";
    document.querySelector("#checkpointText").textContent = config.policy_error
      ? `checkpoint error: ${config.policy_error}`
      : loadedPolicies
      ? loadedPolicies
      : config.checkpoint
      ? `checkpoint: ${config.checkpoint}`
      : "checkpoint: none (hybrid and manual available)";
    setConnected(true);
    pollState();
  } catch (error) {
    console.error(error);
    setConnected(false);
    window.setTimeout(initialize, 1000);
  }
}

function updateControllerNote() {
  const option = controllerSelect.selectedOptions[0];
  controllerNote.textContent = option ? option.dataset.description : "";
}

controllerSelect.addEventListener("change", async () => {
  const requestedController = controllerSelect.value;
  updateControllerNote();
  try {
    await mutate("/api/settings", {controller: requestedController});
  } catch (error) {
    window.alert(error.message);
    if (latestState) controllerSelect.value = latestState.controller;
  }
});

scenarioSelect.addEventListener("change", () => resetScenario());
resetButton.addEventListener("click", () => resetScenario());

async function resetScenario() {
  history.length = 0;
  const requestedScenario = scenarioSelect.value;
  try {
    await mutate("/api/reset", {scenario: requestedScenario});
  } catch (error) {
    window.alert(error.message);
    if (latestState) scenarioSelect.value = latestState.scenario;
  }
}

pauseButton.addEventListener("click", async () => {
  const requestedPause = !paused;
  try {
    await mutate("/api/settings", {paused: requestedPause});
    paused = requestedPause;
  } catch (error) {
    console.error(error);
  }
  pauseIcon.textContent = paused ? ">" : "II";
  pauseButton.setAttribute("aria-label", paused ? "Resume simulation" : "Pause simulation");
});

function normalizedKey(key) { return keyAliases[key.toLowerCase()] || null; }

function setVisualKeys() {
  document.querySelectorAll(".key").forEach((element) => {
    const key = element.dataset.key;
    const aliases = key === "w" ? ["w", "arrowup"]
      : key === "a" ? ["a", "arrowleft"]
      : key === "s" ? ["s", "arrowdown"] : ["d", "arrowright"];
    element.classList.toggle("active", aliases.some((name) => heldKeys.has(name)));
  });
}

function sendKeys() {
  setVisualKeys();
  inputRevision += 1;
  return api("/api/input", {
    keys: Array.from(heldKeys),
    client_id: clientId,
    revision: inputRevision,
  }).catch(() => setConnected(false));
}

window.addEventListener("keydown", (event) => {
  const key = normalizedKey(event.key);
  if (!key || event.target instanceof HTMLSelectElement) return;
  event.preventDefault();
  if (!heldKeys.has(key)) {
    heldKeys.add(key);
    sendKeys();
  }
});

window.addEventListener("keyup", (event) => {
  const key = normalizedKey(event.key);
  if (!key) return;
  event.preventDefault();
  heldKeys.delete(key);
  sendKeys();
});

window.addEventListener("blur", () => {
  heldKeys.clear();
  sendKeys();
});

document.querySelectorAll(".key").forEach((button) => {
  const key = button.dataset.key;
  button.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    button.setPointerCapture(event.pointerId);
    heldKeys.add(key);
    sendKeys();
  });
  const release = () => {
    heldKeys.delete(key);
    sendKeys();
  };
  button.addEventListener("pointerup", release);
  button.addEventListener("pointercancel", release);
});

window.setInterval(() => {
  if (heldKeys.size) sendKeys();
}, 100);

async function pollState() {
  try {
    latestState = await api("/api/state");
    setConnected(true);
    render(latestState);
  } catch (error) {
    console.error(error);
    setConnected(false);
  }
  window.setTimeout(pollState, connected ? 40 : 500);
}

function fixed(value, digits = 3) {
  const normalized = Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value;
  return normalized.toFixed(digits).padStart(digits + 3, " ");
}

function formatArray(values, breakAt = 4) {
  const first = values.slice(0, breakAt).map((value) => fixed(value)).join(", ");
  if (values.length <= breakAt) return `array([${first}])`;
  const second = values.slice(breakAt).map((value) => fixed(value)).join(", ");
  return `array([${first},\n       ${second}])`;
}

function render(state) {
  if (controllerSelect.value !== state.controller) controllerSelect.value = state.controller;
  if (scenarioSelect.value !== state.scenario) scenarioSelect.value = state.scenario;
  const manual = state.authority === "manual";
  const sources = state.control_sources || [];
  const sourceSummary = sources
    .map((source) => `${source.label} ${Math.round(source.weight * 100)}%`)
    .join(" + ");
  authorityCard.classList.toggle("manual", manual);
  authorityText.textContent = manual
    ? "MANUAL OVERRIDE"
    : sources.map((source) => source.label).join(" + ").toUpperCase();
  modeText.textContent = manual
    ? `${state.keys.join(" + ").toUpperCase()} held; policy disengaged`
    : `${sourceSummary} active`;
  simClock.textContent = `t = ${state.simulation_time.toFixed(2)} s`;
  canvasAlert.hidden = !state.terminated;
  paused = state.paused;
  pauseIcon.textContent = paused ? ">" : "II";

  document.querySelector("#angleMetric").textContent = state.metrics.upright_angle.toFixed(3);
  document.querySelector("#rateMetric").textContent = state.metrics.tangent_rate.toFixed(3);
  document.querySelector("#speedMetric").textContent = state.metrics.cart_velocity.toFixed(3);
  document.querySelector("#cartMetric").textContent = state.metrics.cart_position.toFixed(3);
  document.querySelector("#stepMetric").textContent = state.step_count.toLocaleString();
  document.querySelector("#angleMeter").style.width = `${Math.min(state.metrics.upright_angle / Math.PI, 1) * 100}%`;
  document.querySelector("#stateArray").textContent = formatArray(state.state);
  document.querySelector("#directionArray").textContent = formatArray(state.direction, 3);
  document.querySelector("#positionReadout").textContent = `cart = [${fixed(state.state[0])}, ${fixed(state.state[1])} ]`;
  document.querySelector("#actionReadout").textContent = `ctrl = [${fixed(state.applied_action[0])}, ${fixed(state.applied_action[1])} ]`;

  history.push({angle: state.metrics.upright_angle / Math.PI, position: state.metrics.cart_position / 5});
  if (history.length > 250) history.shift();
  drawWorld(state);
  drawTrace();
}

function canvasContext(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(Math.round(rect.width * ratio), 1);
  const height = Math.max(Math.round(rect.height * ratio), 1);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {context, width: rect.width, height: rect.height};
}

function drawWorld(state) {
  const {context: ctx, width, height} = canvasContext(worldCanvas);
  ctx.clearRect(0, 0, width, height);
  const scale = Math.min(width / 13.5, height / 8.5);
  const origin = {x: width / 2, y: height * 0.57};
  const groundZ = -0.14;
  const project = (x, y, z = 0) => ({
    x: origin.x + (x - y) * scale,
    y: origin.y + (x + y) * scale * 0.48 - z * scale * 1.15,
  });
  const polygon = (points, fill, stroke = null, lineWidth = 1) => {
    ctx.beginPath();
    points.forEach((point, index) => {
      if (index) ctx.lineTo(point.x, point.y);
      else ctx.moveTo(point.x, point.y);
    });
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }
  };

  // One-meter ground tiles provide an unambiguous x/y position reference.
  for (let sum = -10; sum <= 8; sum += 1) {
    for (let x = -5; x < 5; x += 1) {
      const y = sum - x;
      if (y < -5 || y >= 5) continue;
      const tile = [
        project(x, y, groundZ),
        project(x + 1, y, groundZ),
        project(x + 1, y + 1, groundZ),
        project(x, y + 1, groundZ),
      ];
      const even = (x + y) % 2 === 0;
      polygon(tile, even ? "#eef1f6" : "#dce2ec", "rgba(77, 119, 207, .13)", .7);
    }
  }

  const centerTile = [
    project(-.5, -.5, groundZ - .002), project(.5, -.5, groundZ - .002),
    project(.5, .5, groundZ - .002), project(-.5, .5, groundZ - .002),
  ];
  polygon(centerTile, "rgba(240, 199, 66, .13)", "rgba(177, 139, 16, .72)", 1.5);

  // Rail axes remain visible over the checkerboard.
  const xStart = project(-5, 0, groundZ - .005), xEnd = project(5, 0, groundZ - .005);
  const yStart = project(0, -5, groundZ - .005), yEnd = project(0, 5, groundZ - .005);
  ctx.strokeStyle = "rgba(53, 90, 168, .48)";
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(xStart.x, xStart.y); ctx.lineTo(xEnd.x, xEnd.y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(yStart.x, yStart.y); ctx.lineTo(yEnd.x, yEnd.y); ctx.stroke();

  const [cartX, cartY] = state.state;
  const direction = state.direction;
  const shadowOffset = {x: .09, y: .09};
  const cartHalf = .10;
  const cartShadow = [
    project(cartX - cartHalf + shadowOffset.x, cartY - cartHalf + shadowOffset.y, groundZ - .01),
    project(cartX + cartHalf + shadowOffset.x, cartY - cartHalf + shadowOffset.y, groundZ - .01),
    project(cartX + cartHalf + shadowOffset.x, cartY + cartHalf + shadowOffset.y, groundZ - .01),
    project(cartX - cartHalf + shadowOffset.x, cartY + cartHalf + shadowOffset.y, groundZ - .01),
  ];
  polygon(cartShadow, "rgba(31, 39, 53, .22)");

  const baseShadow = project(cartX + shadowOffset.x, cartY + shadowOffset.y, groundZ - .012);
  const tipShadow = project(
    cartX + direction[0] * .6 + shadowOffset.x,
    cartY + direction[1] * .6 + shadowOffset.y,
    groundZ - .012,
  );
  ctx.strokeStyle = "rgba(31, 39, 53, .2)";
  ctx.lineWidth = Math.max(2, .04 * scale);
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(baseShadow.x, baseShadow.y);
  ctx.lineTo(tipShadow.x, tipShadow.y);
  ctx.stroke();
  ctx.fillStyle = "rgba(31, 39, 53, .24)";
  ctx.beginPath();
  ctx.ellipse(tipShadow.x, tipShadow.y, Math.max(3, .05 * scale), Math.max(1.5, .025 * scale), 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.lineCap = "butt";

  // The MuJoCo cart is a 0.2 m cube. Draw three separately shaded faces.
  const b00 = project(cartX - cartHalf, cartY - cartHalf, -.10);
  const b10 = project(cartX + cartHalf, cartY - cartHalf, -.10);
  const b11 = project(cartX + cartHalf, cartY + cartHalf, -.10);
  const b01 = project(cartX - cartHalf, cartY + cartHalf, -.10);
  const t00 = project(cartX - cartHalf, cartY - cartHalf, .10);
  const t10 = project(cartX + cartHalf, cartY - cartHalf, .10);
  const t11 = project(cartX + cartHalf, cartY + cartHalf, .10);
  const t01 = project(cartX - cartHalf, cartY + cartHalf, .10);
  polygon([b10, b11, t11, t10], "#3159a5", "#294b8b", 1);
  polygon([b11, b01, t01, t11], "#406bc0", "#294b8b", 1);
  polygon([t00, t10, t11, t01], "#5f86d7", "#294b8b", 1);

  // Pole dimensions match the XML: 0.6 m long, 0.04 m diameter, 0.05 m tip radius.
  const poleBase = project(cartX, cartY, 0);
  const tip = project(
    cartX + direction[0] * .6,
    cartY + direction[1] * .6,
    direction[2] * .6,
  );
  const poleWidth = Math.max(3, .04 * scale);
  ctx.lineCap = "round";
  ctx.strokeStyle = "#8d7112";
  ctx.lineWidth = poleWidth + 2;
  ctx.beginPath(); ctx.moveTo(poleBase.x, poleBase.y); ctx.lineTo(tip.x, tip.y); ctx.stroke();
  ctx.strokeStyle = "#f0c742";
  ctx.lineWidth = poleWidth;
  ctx.beginPath(); ctx.moveTo(poleBase.x - 1, poleBase.y - 1); ctx.lineTo(tip.x - 1, tip.y - 1); ctx.stroke();
  ctx.strokeStyle = "rgba(255, 249, 205, .9)";
  ctx.lineWidth = Math.max(1, poleWidth * .24);
  ctx.beginPath(); ctx.moveTo(poleBase.x - 1.5, poleBase.y - 1.5); ctx.lineTo(tip.x - 1.5, tip.y - 1.5); ctx.stroke();
  ctx.lineCap = "butt";

  const tipRadius = Math.max(4, .05 * scale);
  ctx.fillStyle = "#202936";
  ctx.beginPath(); ctx.arc(tip.x, tip.y, tipRadius + 1.5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#3f4d61";
  ctx.beginPath(); ctx.arc(tip.x, tip.y, tipRadius, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "rgba(255, 255, 255, .72)";
  ctx.beginPath(); ctx.arc(tip.x - tipRadius * .34, tip.y - tipRadius * .34, tipRadius * .28, 0, Math.PI * 2); ctx.fill();

  const pivotRadius = Math.max(2.5, .035 * scale);
  ctx.fillStyle = "#1f2c40";
  ctx.beginPath(); ctx.arc(poleBase.x, poleBase.y, pivotRadius, 0, Math.PI * 2); ctx.fill();

  const actionScale = scale * .65;
  const actionOrigin = project(cartX, cartY, .12);
  const end = {
    x: actionOrigin.x + (state.applied_action[0] - state.applied_action[1]) * actionScale,
    y: actionOrigin.y + (state.applied_action[0] + state.applied_action[1]) * actionScale * .48,
  };
  ctx.strokeStyle = state.authority === "manual" ? "#c29913" : "#355aa8";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(actionOrigin.x, actionOrigin.y); ctx.lineTo(end.x, end.y); ctx.stroke();
  ctx.fillStyle = ctx.strokeStyle;
  ctx.beginPath(); ctx.arc(end.x, end.y, 3, 0, Math.PI * 2); ctx.fill();
}

function drawTrace() {
  const {context: ctx, width, height} = canvasContext(traceCanvas);
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#e4e7ed";
  ctx.lineWidth = 1;
  for (let row = 1; row < 4; row += 1) {
    const y = row * height / 4;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
  const plot = (key, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    history.forEach((sample, index) => {
      const x = history.length <= 1 ? 0 : index / (history.length - 1) * width;
      const y = height - Math.min(Math.max(sample[key], 0), 1) * height * .86 - height * .07;
      index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  };
  plot("angle", "#4d77cf");
  plot("position", "#d3a90f");
}

window.addEventListener("resize", () => {
  if (latestState) render(latestState);
});

initialize();
