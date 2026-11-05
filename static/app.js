// Shared helpers for the ReturnGuard pages.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const rupees = (n) => "₹" + Number(n).toLocaleString("en-IN");
const caseNo = (id) => "RG-" + String(id).padStart(6, "0");
const fmtDate = (iso) => iso ? new Date(iso.length === 10 ? iso + "T00:00:00" : iso).toLocaleDateString("en-IN", {day: "numeric", month: "short", year: "numeric"}) : "";
const fmtDateTime = (iso) => iso ? new Date(iso).toLocaleString("en-IN", {day: "numeric", month: "short", hour: "2-digit", minute: "2-digit"}) : "";
const addDays = (iso, n) => { const d = new Date(iso + "T00:00:00"); d.setDate(d.getDate() + n); return d.toISOString().slice(0, 10); };

async function api(path, opts = {}) {
  const r = await fetch(path, {credentials: "same-origin", ...opts});
  if (r.status === 401) { location.href = "/login"; throw new Error("Please sign in."); }
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = body.detail;
    throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join(" ") : "Something went wrong. Please try again.");
  }
  return body;
}
const postJSON = (path, data) => api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)});

async function logout() {
  await fetch("/api/logout", {method: "POST", credentials: "same-origin"});
  location.href = "/login";
}

// Stroke icons on a 24px grid.
const ICON_PATHS = {
  truck: '<path d="M1 3h15v13H1z"/><path d="M16 8h4l3 3v5h-7V8z"/><circle cx="5.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/>',
  pin: '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>',
  key: '<circle cx="7.5" cy="15.5" r="5.5"/><path d="M11.4 11.6L21 2M16 7l3 3M19 4l2 2"/>',
  user: '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
  shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/>',
  back: '<path d="M15 18l-6-6 6-6"/>', chevron: '<path d="M9 18l6-6-6-6"/>', down: '<path d="M6 9l6 6 6-6"/>',
  check: '<path d="M20 6L9 17l-5-5"/>', x: '<path d="M18 6L6 18M6 6l12 12"/>',
  alert: '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
  clock: '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
  camera: '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
  upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/>',
  logout: '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>',
  package: '<path d="M16.5 9.4l-9-5.2M21 16V8a2 2 0 0 0-1-1.7l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.7l7 4a2 2 0 0 0 2 0l7-4a2 2 0 0 0 1-1.7z"/><path d="M3.3 7L12 12l8.7-5M12 22V12"/>',
  return: '<path d="M9 14L4 9l5-5"/><path d="M20 20v-7a4 4 0 0 0-4-4H4"/>',
  help: '<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3M12 17h.01"/>',
  inbox: '<path d="M22 12h-6l-2 3h-4l-2-3H2"/><path d="M5.5 5.1L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.5-6.9A2 2 0 0 0 16.8 4H7.2a2 2 0 0 0-1.7 1.1z"/>',
  list: '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
  archive: '<path d="M21 8v13H3V8M1 3h22v5H1zM10 12h4"/>',
  search: '<circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>',
  flag: '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1zM4 22v-7"/>',
  zoom: '<circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35M11 8v6M8 11h6"/>',
  file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8"/>',
  refresh: '<path d="M23 4v6h-6M1 20v-6h6"/><path d="M3.5 9a9 9 0 0 1 14.9-3.4L23 10M1 14l4.6 4.4A9 9 0 0 0 20.5 15"/>',
  eye: '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>',
  eyeOff: '<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19M14.12 14.12a3 3 0 1 1-4.24-4.24"/><path d="M1 1l22 22"/>',
};
const icon = (name, cls = "") => `<svg class="i ${cls}" viewBox="0 0 24 24" aria-hidden="true">${ICON_PATHS[name] || ""}</svg>`;
const logoHTML = (light = false, href = "/") => `<a class="logo${light ? " light" : ""}" href="${href}"><span class="mark"><svg viewBox="0 0 24 24" aria-hidden="true">${ICON_PATHS.shield}</svg></span>ReturnGuard</a>`;
const initials = (name) => (name || "?").split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase();
const footerHTML = () => `<footer class="site-footer"><div class="inner"><span>© ${new Date().getFullYear()} ReturnGuard</span>
  <nav aria-label="Footer"><a href="/policy">Return policy</a><a href="/account#help">Help</a></nav></div></footer>`;

// Boxes are normalised 0-1 (x, y, w, h), so they sit on the photo at any size.
function boxesHTML(regions) {
  return (regions || []).map((r) => {
    const flip = r.y < 0.08 ? " flip" : "", weak = r.weak ? " weak" : "";
    return `<div class="box${flip}${weak}" style="left:${r.x * 100}%;top:${r.y * 100}%;width:${r.w * 100}%;height:${r.h * 100}%"
      title="${r.weak ? "Weak signal: below the confident-detection score" : "Detected damage"}"><span>${esc(r.type)} ${Number(r.score).toFixed(2)}</span></div>`;
  }).join("");
}

// Customer-facing status names and badge colours.
const STATUS = {
  "Checking": ["info", "Under review"], "Approved": ["ok", "Approved"], "Photos needed": ["warn", "Action needed"],
  "With reviewer": ["info", "Under review"], "Rejected": ["bad", "Not approved"],
  "Pickup scheduled": ["info", "Pickup scheduled"], "Refunded": ["ok", "Refunded"], "Credit issued": ["ok", "Credit issued"],
};
// Where each kind of account lands after signing in.
const ROLE_HOME = {customer: "/account", reviewer: "/admin", courier: "/pickup"};
const statusBadge = (s) => { const [cls, label] = STATUS[s] || ["mute", s]; return `<span class="badge ${cls}">${esc(label)}</span>`; };
const DECISION_BADGE = {AUTO_APPROVE: ["ok", "Approve"], REQUEST_MORE_EVIDENCE: ["warn", "Request photos"],
                        ESCALATE_TO_HUMAN: ["bad", "Manual review"], AUTO_REJECT: ["mute", "Decline"]};
