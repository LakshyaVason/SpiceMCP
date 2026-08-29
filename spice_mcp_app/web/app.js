/* UI logic. All real work happens in Python via window.pywebview.api; this file only
   renders and collects input.

   The diff modal is the approval gate's front end: a proposed patch is never written by
   the model, only by an explicit Apply here. */

const $ = (id) => document.getElementById(id);
const chat = $("chat");

let selected = null;      // path of the chosen circuit
let pendingPatch = null;  // the proposal currently shown in the modal
let busy = false;

/* --- small helpers ----------------------------------------------------------- */

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

/* Just enough markdown for what the model actually writes: bold, inline code and
   bullet lists. A full parser is not worth the dependency here. */
function renderMarkdown(text) {
  const lines = escapeHtml(text).split("\n");
  let html = "";
  let inList = false;

  for (const line of lines) {
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    if (bullet) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${inline(bullet[1])}</li>`;
      continue;
    }
    if (inList) { html += "</ul>"; inList = false; }
    if (line.trim() === "") { html += ""; continue; }
    html += `<p>${inline(line)}</p>`;
  }
  if (inList) html += "</ul>";
  return html;

  function inline(s) {
    return s
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  }
}

function bubble(cls, html) {
  const node = document.createElement("div");
  node.className = `msg ${cls}`;
  node.innerHTML = html;
  chat.appendChild(node);
  chat.scrollTop = chat.scrollHeight;
  return node;
}

function banner(message) {
  const el = $("banner");
  if (!message) { el.classList.add("hidden"); return; }
  el.textContent = message;
  el.classList.remove("hidden");
}

function setBusy(state, label) {
  busy = state;
  $("send").disabled = state;
  $("resim").disabled = state || !selected;
  $("send").textContent = state ? (label || "Working…") : "Send";
}

/* --- startup ----------------------------------------------------------------- */

window.addEventListener("pywebviewready", async () => {
  const started = await window.pywebview.api.start();
  if (!started.ok) {
    $("model").textContent = "not connected";
    banner(started.error);
    return;
  }

  $("model").textContent = `${started.model} · ${started.tools.length} tools`;
  $("session-path").textContent = started.session_path;

  const initial = await window.pywebview.api.get_initial_folder();
  if (!initial.ok) return;

  if (initial.folder) {
    loadFolder(await window.pywebview.api.list_folder(initial.folder));
    // Launched on one circuit: select it now so the static checks are already on screen.
    if (initial.circuit) await selectCircuit(initial.circuit, findCircuitRow(initial.circuit));
  }

  // Last, because loadFolder and selectCircuit both clear the banner on success. A problem
  // from before the window existed has nowhere else to go - launched from Explorer via
  // pythonw there is no console it could have been printed to.
  if (initial.note) banner(initial.note);
});

/* --- circuit list ------------------------------------------------------------ */

$("pick").addEventListener("click", async () => {
  loadFolder(await window.pywebview.api.pick_folder());
});

function loadFolder(result) {
  if (!result || !result.ok) { if (result) banner(result.error); return; }
  if (result.cancelled) return;
  banner(null);

  $("folder").textContent = result.folder;
  const list = $("circuits");
  list.innerHTML = "";

  if (!result.circuits.length) {
    $("folder").textContent = `${result.folder} — no .asc/.net/.cir files here.`;
    return;
  }

  for (const circuit of result.circuits) {
    const li = document.createElement("li");
    // The key findCircuitRow matches on. It cannot use li.title, which holds the
    // ExpressPCB warning instead of the path for a shadowed entry.
    li.dataset.path = circuit.path;
    if (circuit.shadowed) li.classList.add("shadowed");
    li.innerHTML =
      `<span>${escapeHtml(circuit.name)}</span>` +
      `<span class="tag">${circuit.shadowed ? "likely ExpressPCB" : circuit.suffix}</span>`;
    li.title = circuit.shadowed
      ? "A .net beside a .asc is usually an ExpressPCB export, not SPICE. Prefer the .asc."
      : circuit.path;
    li.addEventListener("click", () => selectCircuit(circuit.path, li));
    list.appendChild(li);
  }
}

/* Find the sidebar row for a path. Windows paths differ in case harmlessly, and walking
   the list avoids escaping backslashes into a CSS selector. */
function findCircuitRow(path) {
  const wanted = String(path).toLowerCase();
  for (const li of document.querySelectorAll("#circuits li")) {
    if ((li.dataset.path || "").toLowerCase() === wanted) return li;
  }
  return null;
}

async function selectCircuit(path, li) {
  document.querySelectorAll("#circuits li").forEach((n) => n.classList.remove("active"));
  // Optional: a circuit opened from the command line may have no row, e.g. a .asy or a
  // file outside the listed folder. Selecting it must still work, just unhighlighted.
  if (li) li.classList.add("active");

  const result = await window.pywebview.api.select_circuit(path);
  if (!result.ok) { banner(result.error); return; }
  banner(null);

  selected = path;
  $("resim").disabled = busy;
  renderChecks(result.checks);
  bubble("system", `<p class="muted small">Selected <code>${escapeHtml(result.name)}</code> — ${escapeHtml(result.checks.summary || "")}</p>`);
  // Once per session: we read the .asc from disk, so unsaved GUI edits are invisible here.
  if (result.warning) bubble("system", `<p class="sev-warning small">${escapeHtml(result.warning)}</p>`);
}

function renderChecks(checks) {
  const box = $("checks");
  if (!checks) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");

  const findings = checks.findings || [];
  if (!findings.length) {
    box.innerHTML = `<span class="sev-ok">Static checks clean.</span>
      <div class="muted tiny">A clean static pass is not proof the circuit meets its spec.</div>`;
    return;
  }

  box.innerHTML =
    `<strong>${findings.length} static finding${findings.length === 1 ? "" : "s"}</strong><ul>` +
    findings.map((f) =>
      `<li class="sev-${escapeHtml(f.severity)}">${escapeHtml(f.message)}</li>`).join("") +
    "</ul>";
}

/* --- chat ------------------------------------------------------------------- */

$("send").addEventListener("click", send);
$("input").addEventListener("keydown", (event) => {
  // Enter sends; Shift+Enter is a newline.
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); send(); }
});

async function send() {
  const input = $("input");
  const text = input.value.trim();
  if (!text || busy) return;

  input.value = "";
  bubble("user", escapeHtml(text).replace(/\n/g, "<br>"));

  const spinner = bubble("assistant", `<div class="thinking">reading the circuit<span class="dot">…</span></div>`);
  setBusy(true);

  let result;
  try {
    result = await window.pywebview.api.send_message(text);
  } catch (err) {
    spinner.remove();
    bubble("error", escapeHtml(String(err)));
    setBusy(false);
    return;
  }

  spinner.remove();
  setBusy(false);

  if (!result.ok) { bubble("error", escapeHtml(result.error)); return; }

  let html = "";
  if (result.tool_calls && result.tool_calls.length) html += renderToolCalls(result.tool_calls);
  html += renderMarkdown(result.text);
  html += `<div class="muted tiny">${result.usage.input_tokens} in · ${result.usage.output_tokens} out · ${result.rounds} round${result.rounds === 1 ? "" : "s"}</div>`;
  bubble("assistant", html);

  updateTotals(result.totals);
  if (result.pending_patch) showDiff(result.pending_patch);
}

function renderToolCalls(calls) {
  const names = calls.map((c) => c.name).join(", ");
  return `<details class="tools"><summary>${calls.length} tool call${calls.length === 1 ? "" : "s"}: ${escapeHtml(names)}</summary>` +
    calls.map((c) =>
      `<div class="call">
         <span class="name ${c.is_error ? "is-error" : ""}">${escapeHtml(c.name)}(${escapeHtml(JSON.stringify(c.arguments))})</span>
         <pre>${escapeHtml(c.result)}</pre>
       </div>`).join("") +
    "</details>";
}

function updateTotals(totals) {
  if (!totals) return;
  $("tok-in").textContent = totals.input_tokens;
  $("tok-out").textContent = totals.output_tokens;
  $("turns").textContent = totals.turns;
}

/* --- diff approval ---------------------------------------------------------- */

function showDiff(patch) {
  pendingPatch = patch;
  $("diff-summary").textContent =
    `${patch.ref}: ${patch.old_value ?? "(blank)"} → ${patch.new_value}  ·  line ${patch.line_no} of ${patch.asc_path}`;
  $("diff-note").textContent =
    `Only that line is rewritten. Encoding (${patch.encoding}) and line endings are preserved, so the file stays openable in the LTspice GUI.`;
  $("diff-body").innerHTML = colourDiff(patch.diff);
  $("diff-modal").classList.remove("hidden");
}

function colourDiff(text) {
  return escapeHtml(text).split("\n").map((line) => {
    if (line.startsWith("+++") || line.startsWith("---")) return `<span class="meta">${line}</span>`;
    if (line.startsWith("@@")) return `<span class="hunk">${line}</span>`;
    if (line.startsWith("+")) return `<span class="add">${line}</span>`;
    if (line.startsWith("-")) return `<span class="del">${line}</span>`;
    return line;
  }).join("\n");
}

$("diff-reject").addEventListener("click", () => {
  $("diff-modal").classList.add("hidden");
  bubble("system", `<p class="muted small">Change rejected. Nothing was written.</p>`);
  pendingPatch = null;
});

$("diff-apply").addEventListener("click", async () => {
  if (!pendingPatch) return;
  const patch = pendingPatch;
  $("diff-modal").classList.add("hidden");
  pendingPatch = null;
  setBusy(true, "Writing…");

  const result = await window.pywebview.api.apply_patch(patch.asc_path, patch.ref, patch.new_value);
  setBusy(false);

  if (!result.ok) { bubble("error", escapeHtml(result.error)); return; }
  bubble("system", `<p class="sev-ok small">${escapeHtml(result.summary)}</p>`);
  // The file is written either way; this says so plainly rather than letting the fix
  // vanish the next time the user saves from the LTspice GUI.
  if (result.warning) bubble("system", `<p class="sev-warning small">${escapeHtml(result.warning)}</p>`);

  // Re-check and re-simulate straight away: an applied fix that was never verified is
  // not a finished fix.
  await resimulate();
});

/* --- re-simulate ------------------------------------------------------------ */

$("resim").addEventListener("click", resimulate);

async function resimulate() {
  if (!selected) return;
  setBusy(true, "Simulating…");
  const result = await window.pywebview.api.resimulate();
  setBusy(false);

  if (!result.ok) { bubble("error", escapeHtml(result.error)); return; }
  renderChecks(result.checks);
  bubble("system",
    `<p class="${result.succeeded ? "sev-ok" : "sev-error"} small">
       Re-simulated: ${escapeHtml(result.summary || "")}
     </p>`);
}

/* --- session log ----------------------------------------------------------- */

$("export").addEventListener("click", async () => {
  const result = await window.pywebview.api.export_session();
  if (!result.ok) { banner(result.error); return; }
  if (result.cancelled) return;
  bubble("system", `<p class="muted small">Session log exported to <code>${escapeHtml(result.path)}</code></p>`);
});

$("resolved").addEventListener("change", async (event) => {
  const result = await window.pywebview.api.mark_resolved(event.target.checked);
  if (!result.ok) banner(result.error);
});
