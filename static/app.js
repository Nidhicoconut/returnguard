// Shared helpers for the ReturnGuard pages.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const rupees = (n) => "₹" + Number(n).toLocaleString("en-IN");

async function api(path, opts = {}) {
  const r = await fetch(path, {credentials: "same-origin", ...opts});
  if (r.status === 401) { location.href = "/login"; throw new Error("Sign in first."); }
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = body.detail;
    throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join(" ") : "Something went wrong. Try again.");
  }
  return body;
}

const postJSON = (path, data) => api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)});

async function logout() {
  await fetch("/api/logout", {method: "POST", credentials: "same-origin"});
  location.href = "/login";
}

// Small inline icon set (stroke icons, 24px grid).
const ICON_PATHS = {
  back: '<path d="M15 18l-6-6 6-6"/>',
  tag: '<path d="M20.6 13.4l-7.2 7.2a2 2 0 0 1-2.8 0L3 13V3h10l7.6 7.6a2 2 0 0 1 0 2.8z"/><circle cx="7.5" cy="7.5" r="1.5"/>',
  check: '<path d="M20 6L9 17l-5-5"/>',
  alert: '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
  x: '<path d="M18 6L6 18M6 6l12 12"/>',
  camera: '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
  logout: '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>',
  arrow: '<path d="M5 12h14M12 5l7 7-7 7"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  clock: '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
  shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/>',
  user: '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
  search: '<circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>',
  flag: '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><path d="M4 22v-7"/>',
  zoom: '<circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35M11 8v6M8 11h6"/>',
  box: '<rect x="3" y="3" width="18" height="18" rx="2"/>',
  send: '<path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>',
  refresh: '<path d="M23 4v6h-6M1 20v-6h6"/><path d="M3.5 9a9 9 0 0 1 14.9-3.4L23 10M1 14l4.6 4.4A9 9 0 0 0 20.5 15"/>',
  eye: '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>',
  eyeOff: '<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19M14.12 14.12a3 3 0 1 1-4.24-4.24"/><path d="M1 1l22 22"/>',
};
const icon = (name, cls = "") => `<svg class="i ${cls}" viewBox="0 0 24 24" aria-hidden="true">${ICON_PATHS[name] || ""}</svg>`;
const brand = () => `<span class="brand"><span class="tag">${icon("tag", "")}</span>ReturnGuard</span>`;
const initials = (name) => (name || "?").split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase();

// Boxes are normalised 0-1 (x, y, w, h), so they sit on the photo at any size.
function boxesHTML(regions) {
  return (regions || []).map((r) => {
    const flip = r.y < 0.08 ? " flip" : "", weak = r.weak ? " weak" : "";
    return `<div class="box${flip}${weak}" style="left:${r.x * 100}%;top:${r.y * 100}%;width:${r.w * 100}%;height:${r.h * 100}%"
      title="${r.weak ? "Weak hint: below the confident-detection score" : "Detected damage"}"><span>${r.weak ? "weak " : ""}${esc(r.type)} ${Number(r.score).toFixed(2)}</span></div>`;
  }).join("");
}

const DECISION_PILL = {
  AUTO_APPROVE: ["ok", "Auto-approve"],
  REQUEST_MORE_EVIDENCE: ["warn", "Request evidence"],
  ESCALATE_TO_HUMAN: ["bad", "Escalate"],
  AUTO_REJECT: ["mute", "Auto-decline"],
};
const STATUS_PILL = {"Checking": "info", "Approved": "ok", "Photos needed": "warn", "With reviewer": "bad", "Rejected": "mute"};
