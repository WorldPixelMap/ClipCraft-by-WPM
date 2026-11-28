const $ = (id) => document.getElementById(id);

const PROFILES_STORAGE_KEY = "clipcraft_studio_profiles_v1";

const PAYPAL_CONFIG = {
  businessEmail: "hmjsarkar@gmail.com",
  hostedButtonId: "",
  paypalMeHandle: "",
  itemName: "Support Local ClipCraft by WPM",
  currencyCode: "USD",
};

function buildPayPalDonateUrl(amount) {
  const numericAmt = Number(amount);
  const hasAmt = Number.isFinite(numericAmt) && numericAmt > 0;
  if (PAYPAL_CONFIG.paypalMeHandle) {
    const slug = PAYPAL_CONFIG.paypalMeHandle.replace(/^\/+|\/+$/g, "");
    return hasAmt ? `https://paypal.me/${slug}/${numericAmt}${PAYPAL_CONFIG.currencyCode}` : `https://paypal.me/${slug}`;
  }
  const params = new URLSearchParams();
  if (PAYPAL_CONFIG.hostedButtonId) {
    params.set("cmd", "_s-xclick");
    params.set("hosted_button_id", PAYPAL_CONFIG.hostedButtonId);
    return `https://www.paypal.com/cgi-bin/webscr?${params.toString()}`;
  }
  params.set("cmd", "_xclick");
  params.set("business", PAYPAL_CONFIG.businessEmail);
  params.set("item_name", PAYPAL_CONFIG.itemName);
  params.set("currency_code", PAYPAL_CONFIG.currencyCode);
  params.set("no_shipping", "1");
  params.set("amount", hasAmt ? numericAmt.toFixed(2) : "5.00");
  return `https://www.paypal.com/cgi-bin/webscr?${params.toString()}`;
}

const state = {
  numClips: 3,
  highlightColor: "yellow",
  aspectRatio: "9:16",
  captionLanguage: "auto",
  camSource: "off",
  camTarget: "top-right",
  splitCamPosition: "top",
  currentJobId: null,
  localFile: null,
  reelFile: null,
  reelLanguage: "auto",
  reelColor: "yellow",
  reelAspect: "original",
  bgMusicId: "",
  localAiInfo: { available: false, ollama: { available: false, models: [] }, lmstudio: { available: false, models: [] }, models: [] },
  donateAmount: "5",
  eventSource: null,
  ytAuthTimer: null,
  queueTimer: null,
};

function fmtTime(sec) {
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  const h = Math.floor(m / 60);
  if (h > 0) return `${h}:${String(m % 60).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
}

function parseTimeInput(raw) {
  const txt = String(raw || "").trim();
  if (!txt) return null;
  if (/^\d+(?:\.\d+)?$/.test(txt)) {
    const val = Number(txt);
    return Number.isFinite(val) && val >= 0 ? val : null;
  }
  const parts = txt.split(":").map((p) => Number(p.trim()));
  if (parts.some((n) => !Number.isFinite(n) || n < 0)) return null;
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  return null;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showError(msg) {
  const el = $("errorBanner");
  el.innerHTML = "";
  const span = document.createElement("span");
  span.textContent = msg;
  el.appendChild(span);

  if (/sign in|sign-in|cookie|not a bot/i.test(msg)) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "err-inline-btn";
    btn.textContent = "Sign in to YouTube";
    btn.addEventListener("click", startYouTubeLogin);
    el.appendChild(btn);
  }
  el.classList.remove("hidden");
}

function clearError() {
  $("errorBanner").classList.add("hidden");
}

function updateCookiesBadge(loaded) {
  const badge = $("cookiesBadge");
  const clearBtn = $("clearCookiesBtn");
  const loginOptBtn = $("ytLoginOptBtn");
  const pill = $("ytLoginPill");
  const pillText = $("ytLoginPillText");
  if (badge) {
    badge.textContent = loaded ? "YouTube Signed In" : "Not Signed In";
    badge.classList.toggle("loaded", Boolean(loaded));
  }
  if (loginOptBtn) {
    loginOptBtn.textContent = loaded ? "Switch Account" : "Sign in to YouTube";
  }
  if (clearBtn) clearBtn.classList.toggle("hidden", !loaded);
  if (pill && pillText) {
    pill.classList.toggle("pill-good", Boolean(loaded));
    pillText.textContent = loaded ? "YouTube Signed In" : "Sign in to YouTube";
  }
}

async function startYouTubeLogin() {
  clearError();
  try {
    const res = await fetch("/api/youtube-login/start", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Could not launch sign-in browser window.");
    $("ytAuthBarMsg").textContent = data.message;
    $("ytAuthBar").classList.remove("hidden");
    pollYouTubeLogin();
  } catch (err) {
    showError(err.message);
  }
}

function pollYouTubeLogin() {
  if (state.ytAuthTimer) clearInterval(state.ytAuthTimer);
  state.ytAuthTimer = setInterval(async () => {
    try {
      const res = await fetch("/api/youtube-login/status");
      const data = await res.json();
      updateCookiesBadge(data.cookies_file_loaded);
      if (data.status === "signed_in") {
        clearInterval(state.ytAuthTimer);
        state.ytAuthTimer = null;
        $("ytAuthBar").classList.add("hidden");
        clearError();
      } else if (data.status === "cancelled" || data.status === "idle") {
        clearInterval(state.ytAuthTimer);
        state.ytAuthTimer = null;
        $("ytAuthBar").classList.add("hidden");
        if (data.message && !data.cookies_file_loaded) {
          showError(data.message);
        }
      }
    } catch {
      clearInterval(state.ytAuthTimer);
      state.ytAuthTimer = null;
    }
  }, 1100);
}

function setClipCount(n) {
  const clamped = Math.max(1, Math.min(10, Number(n) || 3));
  state.numClips = clamped;
  $("numClipsInput").value = String(clamped);
  $("clipCountDisplay").textContent = String(clamped);
  document.querySelectorAll(".q-btn").forEach((b) => {
    b.classList.toggle("active", Number(b.dataset.count) === clamped);
  });
}

function setActiveSeg(groupSelector, attrName, value) {
  document.querySelectorAll(`${groupSelector} .seg-btn`).forEach((b) => {
    b.classList.toggle("active", b.dataset[attrName] === value);
  });
}

function updatePipDiagram() {
  state.camSource = $("camSourceSelect").value || "off";
  const enabled = state.camSource !== "off";
  const isSplit = state.aspectRatio === "split";

  $("camTargetWrap").classList.toggle("hidden", isSplit);
  $("splitPosWrap").classList.toggle("hidden", !isSplit);
  $("camTargetGroup").classList.toggle("disabled", !enabled || isSplit);
  document.querySelectorAll("#camTargetGroup .corner-btn").forEach((btn) => {
    btn.disabled = !enabled || isSplit;
  });

  const box = $("pipBox");
  const splitLine = $("pipSplitLine");
  const splitCam = $("pipSplitCam");
  if (splitLine) splitLine.classList.toggle("hidden", !isSplit);
  if (splitCam) {
    splitCam.classList.toggle("hidden", !isSplit);
    splitCam.classList.toggle("split-bottom", state.splitCamPosition === "bottom");
  }

  if (isSplit) {
    box.classList.add("hidden");
  } else {
    box.classList.toggle("hidden", !enabled);
    box.classList.remove("top-left", "top-right", "bottom-left", "bottom-right");
    box.classList.add(state.camTarget);
  }

  const capBar = $("pipCapBar");
  if (capBar) {
    const pos = $("captionPosSelect")?.value || "bottom";
    capBar.classList.remove("cap-top", "cap-center", "cap-bottom");
    capBar.classList.add(`cap-${pos}`);
    capBar.classList.toggle("hidden", !$("captionsToggle").checked);
  }

  const statusEl = $("pipStatusLine");
  if (statusEl) {
    statusEl.classList.toggle("active", enabled || isSplit);
    if (isSplit) {
      const partLabel = state.splitCamPosition === "bottom" ? "Down Part (Bottom 50%)" : "Up Part (Top 50%)";
      statusEl.textContent = enabled
        ? `Split 50/50: Face (${state.camSource}) in ${partLabel}`
        : `Split 50/50: Auto Face in ${partLabel}`;
    } else if (!enabled) {
      statusEl.textContent = "Cam Frame Off";
    } else if (state.camSource === "auto") {
      statusEl.textContent = `Auto Real-Face Fit (Anywhere) → Place ${state.camTarget}`;
    } else {
      statusEl.textContent = `Fit Real Face (${state.camSource}) → Place ${state.camTarget}`;
    }
  }
}

async function copyText(btn, text, origLabel) {
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = "Copied!";
    btn.classList.add("copied");
    setTimeout(() => {
      btn.textContent = origLabel;
      btn.classList.remove("copied");
    }, 1500);
  } catch {
    btn.textContent = "Select & Ctrl+C";
  }
}

/* Creator Presets (localStorage) */
function readSavedProfiles() {
  try {
    const raw = localStorage.getItem(PROFILES_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function writeSavedProfiles(map) {
  try {
    localStorage.setItem(PROFILES_STORAGE_KEY, JSON.stringify(map));
  } catch {
    // Ignore storage quota errors
  }
}

function captureCurrentPreset() {
  return {
    numClips: state.numClips,
    aspectRatio: state.aspectRatio,
    captionsOn: $("captionsToggle").checked,
    highlightColor: state.highlightColor,
    captionLanguage: state.captionLanguage,
    captionStyle: $("captionStyleSelect").value,
    captionPosition: $("captionPosSelect").value,
    captionScale: $("captionScaleSelect").value,
    camSource: $("camSourceSelect").value,
    camTarget: state.camTarget,
    splitCamPosition: state.splitCamPosition,
    captionMode: $("captionModeSelect").value,
    clipDuration: $("clipDurSelect").value,
    quality: $("qualitySelect").value,
    watermark: $("watermarkInput").value.trim(),
    hookBanner: $("hookBannerInput").value.trim(),
    encoderMode: $("encoderModeSelect").value,
    musicVolume: $("musicVolumeInput").value,
    audioBoost: $("audioBoostToggle").checked,
    trimSilence: $("trimSilenceToggle").checked,
    keepSource: $("keepSourceToggle").checked,
    useSourceCache: $("useSourceCacheToggle").checked,
    aiProvider: $("aiProviderSelect").value,
    ollamaEnabled: $("ollamaEnabledToggle").checked,
    ollamaModel: $("ollamaModelSelect").value,
    ollamaTopic: $("ollamaTopicInput").value.trim(),
  };
}

const DEFAULT_PRESET = {
  numClips: 3,
  aspectRatio: "9:16",
  captionsOn: true,
  highlightColor: "yellow",
  captionLanguage: "auto",
  captionStyle: "classic",
  captionPosition: "bottom",
  captionScale: "1.0",
  camSource: "off",
  camTarget: "top-right",
  splitCamPosition: "top",
  captionMode: "auto",
  clipDuration: "35",
  quality: "720p",
  watermark: "",
  hookBanner: "",
  encoderMode: "auto",
  musicVolume: "14",
  audioBoost: true,
  trimSilence: false,
  keepSource: false,
  useSourceCache: true,
  aiProvider: "auto",
  ollamaEnabled: false,
  ollamaModel: "",
  ollamaTopic: "",
};

function applyPreset(p) {
  const cfg = { ...DEFAULT_PRESET, ...(p || {}) };
  setClipCount(cfg.numClips);
  state.aspectRatio = cfg.aspectRatio;
  setActiveSeg("#aspectGroup", "aspect", cfg.aspectRatio);

  $("captionsToggle").checked = Boolean(cfg.captionsOn);
  $("captionsStatusText").textContent = cfg.captionsOn ? "Word Highlight ON" : "Captions OFF";
  $("colorDots").style.opacity = cfg.captionsOn ? "1" : "0.35";
  $("colorDots").style.pointerEvents = cfg.captionsOn ? "auto" : "none";

  state.highlightColor = cfg.highlightColor;
  document.querySelectorAll("#colorDots .c-dot").forEach((d) => {
    d.classList.toggle("active", d.dataset.color === cfg.highlightColor);
  });

  state.captionLanguage = cfg.captionLanguage;
  setActiveSeg("#langGroup", "lang", cfg.captionLanguage);

  $("captionStyleSelect").value = cfg.captionStyle;
  $("captionPosSelect").value = cfg.captionPosition;
  $("captionScaleSelect").value = cfg.captionScale;
  $("camSourceSelect").value = cfg.camSource;
  state.camTarget = cfg.camTarget;
  document.querySelectorAll("#camTargetGroup .corner-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.corner === cfg.camTarget);
  });
  state.splitCamPosition = cfg.splitCamPosition || "top";
  setActiveSeg("#splitPosGroup", "splitpos", state.splitCamPosition);
  $("captionModeSelect").value = cfg.captionMode;
  $("clipDurSelect").value = String(cfg.clipDuration);
  $("qualitySelect").value = cfg.quality;
  $("watermarkInput").value = cfg.watermark || "";
  $("hookBannerInput").value = cfg.hookBanner || "";
  $("encoderModeSelect").value = cfg.encoderMode || "auto";
  $("musicVolumeInput").value = String(cfg.musicVolume || "14");
  $("musicVolVal").textContent = `${$("musicVolumeInput").value}%`;
  $("audioBoostToggle").checked = Boolean(cfg.audioBoost);
  $("trimSilenceToggle").checked = Boolean(cfg.trimSilence);
  $("keepSourceToggle").checked = Boolean(cfg.keepSource);
  $("useSourceCacheToggle").checked = Boolean(cfg.useSourceCache);
  $("aiProviderSelect").value = cfg.aiProvider || "auto";
  renderLocalAiModels(state.localAiInfo);
  $("ollamaEnabledToggle").checked = Boolean(cfg.ollamaEnabled);
  if (cfg.ollamaModel && [...$("ollamaModelSelect").options].some((o) => o.value === cfg.ollamaModel)) {
    $("ollamaModelSelect").value = cfg.ollamaModel;
  }
  $("ollamaTopicInput").value = cfg.ollamaTopic || "";
  updatePipDiagram();
}

function refreshProfileSelect(selectedName = "__default__") {
  const profiles = readSavedProfiles();
  const sel = $("profileSelect");
  sel.innerHTML = `<option value="__default__">Default Studio</option>`;
  for (const name of Object.keys(profiles).sort()) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    sel.appendChild(opt);
  }
  sel.value = profiles[selectedName] ? selectedName : "__default__";
  $("deleteProfileBtn").classList.toggle("hidden", sel.value === "__default__");
}

function renderLocalAiModels(localAi) {
  const info = localAi || { available: false, ollama: { available: false, models: [] }, lmstudio: { available: false, models: [] }, models: [] };
  state.localAiInfo = info;

  const badge = $("ollamaBadge");
  const sel = $("ollamaModelSelect");
  const provider = $("aiProviderSelect")?.value || "auto";
  if (!badge || !sel) return;

  const olCount = (info.ollama?.models || []).length;
  const lmCount = (info.lmstudio?.models || []).length;
  const olOn = Boolean(info.ollama?.available);
  const lmOn = Boolean(info.lmstudio?.available);

  const filtered = (info.models || []).filter((item) => provider === "auto" || item.provider === provider);
  const prevVal = sel.value;

  badge.classList.toggle("loaded", filtered.length > 0 || (olOn && provider !== "lmstudio") || (lmOn && provider !== "ollama"));
  if (olOn && lmOn) {
    badge.textContent = `Ollama (${olCount}) + LM Studio (${lmCount})`;
  } else if (lmOn) {
    badge.textContent = lmCount ? `LM Studio Ready (${lmCount})` : "LM Studio Running (Load Model)";
  } else if (olOn) {
    badge.textContent = olCount ? `Ollama Ready (${olCount})` : "Ollama Running (No Models)";
  } else {
    badge.textContent = "Local AI Offline";
  }

  sel.innerHTML = "";
  if (filtered.length) {
    for (const item of filtered) {
      const o = document.createElement("option");
      o.value = item.value;
      o.textContent = item.label;
      sel.appendChild(o);
    }
    if (prevVal && [...sel.options].some((o) => o.value === prevVal)) {
      sel.value = prevVal;
    }
  } else if (provider === "lmstudio") {
    sel.innerHTML = `<option value="">${lmOn ? "Load a model in LM Studio Local Server" : "LM Studio server not detected on :1234"}</option>`;
  } else if (provider === "ollama") {
    sel.innerHTML = `<option value="">${olOn ? "Pull a model in Ollama first" : "Ollama server not detected on :11434"}</option>`;
  } else {
    sel.innerHTML = `<option value="">Start Ollama (:11434) or LM Studio (:1234)</option>`;
  }
}

async function refreshLocalAiStatus() {
  try {
    const res = await fetch("/api/local-ai/status");
    if (!res.ok) return;
    const data = await res.json();
    renderLocalAiModels(data);
  } catch {
    // Ignore optional local AI scan error
  }
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const pill = $("offlineModelPill");
    if (data.offline_model_cached) {
      pill.classList.add("pill-good");
      pill.innerHTML = `<span class="dot"></span>Offline Whisper-Tiny Ready`;
    } else {
      pill.innerHTML = `<span class="dot"></span>Offline Model Auto-Caches on Fallback`;
    }
    updateCookiesBadge(data.cookies_file_loaded);

    // Annotate GPU encoders in select
    const gpus = new Set(data.available_gpu_encoders || []);
    const encSel = $("encoderModeSelect");
    if (encSel) {
      const labels = {
        nvenc: gpus.has("nvenc") ? "NVIDIA NVENC (Ready)" : "NVIDIA NVENC (Not Detected)",
        qsv: gpus.has("qsv") ? "Intel Quick Sync (Ready)" : "Intel Quick Sync (Not Detected)",
        amf: gpus.has("amf") ? "AMD AMF (Ready)" : "AMD AMF (Not Detected)",
      };
      for (const opt of encSel.options) {
        if (labels[opt.value]) {
          opt.textContent = labels[opt.value];
          opt.disabled = !gpus.has(opt.value);
        }
      }
    }

    // Local AI (Ollama + LM Studio)
    const fallbackLocalAi = data.local_ai || {
      available: Boolean(data.ollama?.available),
      ollama: data.ollama || { available: false, models: [] },
      lmstudio: data.lmstudio || { available: false, models: [] },
      models: (data.ollama?.models || []).map((m) => ({
        provider: "ollama",
        id: m,
        value: `ollama::${m}`,
        label: `Ollama • ${m}`,
      })),
    };
    renderLocalAiModels(fallbackLocalAi);

    const cacheBtn = $("clearSourceCacheBtn");
    if (cacheBtn && typeof data.source_cache_count === "number") {
      cacheBtn.textContent = `(Clear ${data.source_cache_count} cached)`;
    }
  } catch {
    $("offlineModelPill").textContent = "Server Connecting...";
  }
}

async function refreshQueue() {
  try {
    const res = await fetch("/api/queue");
    if (!res.ok) return;
    const active = await res.json();
    const sec = $("queueSection");
    const list = $("queueList");
    if (!active.length) {
      sec.classList.add("hidden");
      list.innerHTML = "";
      if (state.queueTimer) {
        clearInterval(state.queueTimer);
        state.queueTimer = null;
      }
      return;
    }

    list.innerHTML = "";
    for (const j of active) {
      const row = document.createElement("div");
      row.className = "queue-item";
      row.innerHTML = `
        <div class="queue-item-info" title="Click to watch this job">
          <div class="queue-item-title">${escapeHtml(j.video_title || j.platform || j.job_id)}</div>
          <div class="queue-item-sub">${escapeHtml(j.stage)} • ${j.progress}% — ${escapeHtml(j.message)}</div>
        </div>
        <button type="button" class="btn-ghost-danger btn-xs js-q-cancel">Cancel</button>
      `;
      row.querySelector(".queue-item-info").addEventListener("click", () => {
        state.currentJobId = j.job_id;
        updateProgressUI(j);
        startListening(j.job_id);
      });
      row.querySelector(".js-q-cancel").addEventListener("click", async () => {
        await fetch(`/api/jobs/${j.job_id}/cancel`, { method: "POST" });
        refreshQueue();
      });
      list.appendChild(row);
    }
    sec.classList.remove("hidden");
    if (!state.queueTimer) {
      state.queueTimer = setInterval(refreshQueue, 1600);
    }
  } catch {
    // Ignore optional queue poll errors
  }
}

async function loadHistory() {
  try {
    const res = await fetch("/api/jobs");
    if (!res.ok) return;
    const jobs = await res.json();
    const sec = $("historySection");
    const list = $("historyList");
    list.innerHTML = "";
    if (!jobs.length) {
      sec.classList.add("hidden");
      return;
    }
    for (const j of jobs) {
      const wrap = document.createElement("div");
      wrap.className = "history-item";
      wrap.innerHTML = `
        <button type="button" class="history-chip" title="Open exported clips">
          <div class="hc-title">${escapeHtml(j.video_title || j.job_id)}</div>
          <div class="hc-meta">${escapeHtml(j.platform || "Video")} • ${j.clips.length} clips</div>
        </button>
        <button type="button" class="hc-del-btn" title="Delete export folder">✕</button>
      `;
      wrap.querySelector(".history-chip").addEventListener("click", () => {
        state.currentJobId = j.job_id;
        anchorJobOutputAfter(
          j.platform === "Pre-Edited Reel"
            ? $("reelCaptionSection")
            : $("clipForm").closest(".hero-card")
        );
        renderClips(j);
        $("resultsSection").scrollIntoView({ behavior: "smooth" });
      });
      wrap.querySelector(".hc-del-btn").addEventListener("click", async (e) => {
        e.stopPropagation();
        await fetch(`/api/jobs/${j.job_id}`, { method: "DELETE" });
        if (state.currentJobId === j.job_id) {
          $("resultsSection").classList.add("hidden");
        }
        loadHistory();
      });
      list.appendChild(wrap);
    }
    sec.classList.remove("hidden");
  } catch {
    // Ignore optional history failures
  }
}

async function inspectUrl() {
  clearError();
  const url = $("urlInput").value.trim();
  if (!url) {
    showError("Paste a YouTube, Twitch, Kick, or streaming link first.");
    return;
  }
  const btn = $("inspectBtn");
  btn.disabled = true;
  btn.textContent = "Checking...";

  try {
    const res = await fetch("/api/inspect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, cookies_browser: $("cookiesBrowserSelect")?.value || "" }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Unable to inspect video.");

    $("previewThumb").src = data.thumbnail || "";
    $("previewTitle").textContent = data.title;
    $("previewSub").textContent = `${data.platform} • ${data.channel} • Duration ${fmtTime(data.duration)}`;

    const badges = [`<span class="mini-badge">${escapeHtml(data.platform)}</span>`];
    if (data.has_heatmap) {
      badges.push(`<span class="mini-badge">Heatmap Detected (${data.heatmap_points} pts)</span>`);
    }
    if (data.has_captions) {
      badges.push(`<span class="mini-badge">Zero-AI Captions (${data.caption_languages.slice(0, 2).join(", ")})</span>`);
    } else {
      badges.push(`<span class="mini-badge">Offline Whisper-Tiny Ready for Captions</span>`);
    }
    $("previewBadges").innerHTML = badges.join("");
    $("videoPreview").classList.remove("hidden");
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Preview";
  }
}

function captionSourceLabel(src) {
  switch (src) {
    case "youtube_word_sync":
      return "Word Sync (Zero AI)";
    case "youtube_audio_paced":
      return "Audio-Paced Captions (Zero AI)";
    case "custom_script_paced":
      return "Custom Script Sync (Zero AI)";
    case "offline_whisper_tiny":
      return "Inbuilt Offline Whisper-Tiny";
    case "edited_manual":
      return "Edited Captions";
    default:
      return "No Captions";
  }
}

function renderClips(job) {
  $("aiNote").textContent = job.ai_note || "";
  $("downloadAllBtn").href = `/api/jobs/${job.job_id}/download-all`;
  const srcBtn = $("downloadSourceBtn");
  if (srcBtn) {
    if (job.source_video_url) {
      srcBtn.href = job.source_video_url;
      srcBtn.download = job.source_video_name || "source.mp4";
      srcBtn.classList.remove("hidden");
    } else {
      srcBtn.classList.add("hidden");
    }
  }

  const grid = $("clipsGrid");
  grid.innerHTML = "";

  for (const clip of job.clips) {
    const card = document.createElement("article");
    card.className = "clip-card";
    const reasonsHtml = (clip.reasons || []).map((r) => `<span class="reason-tag">${escapeHtml(r)}</span>`).join("");
    const srtBtn = clip.srt_url
      ? `<a class="dl-srt js-dl-srt" href="${clip.srt_url}" download="${clip.srt_filename || `clip_${clip.rank}.srt`}">Download .SRT</a>`
      : "";

    const social = clip.social || {
      title: `Highlight Clip #${clip.rank}`,
      youtube_caption: "",
      facebook_caption: "",
      hashtags: [],
      copy_all_youtube: "",
      copy_all_facebook: "",
    };
    const tagsStr = (social.hashtags || []).join(" ");
    const editableWordsText = (clip.words && clip.words.length)
      ? clip.words.map((w) => w.word).join(" ")
      : (clip.transcript_preview || "");

    card.innerHTML = `
      <div class="clip-video-wrap">
        <button type="button" class="safe-zone-toggle js-sz-btn" title="Preview TikTok / Reels / Shorts safe zones">Safe Zones</button>
        <div class="safe-zone-overlay hidden js-sz-overlay">
          <div class="sz-top"></div>
          <div class="sz-right"></div>
          <div class="sz-bottom">UI / Caption Safe Zone</div>
        </div>
        <video controls playsinline preload="metadata" poster="${clip.thumb_url || ""}">
          <source src="${clip.video_url}" type="video/mp4" />
        </video>
      </div>
      <div class="clip-body">
        <div class="clip-meta-top">
          <span class="clip-rank">Clip #${clip.rank}</span>
          <span class="score-pill">Score ${clip.score}/100</span>
        </div>
        <div class="clip-time js-clip-time">${fmtTime(clip.start)} – ${fmtTime(clip.end)} (${Math.round(clip.duration)}s) • ${captionSourceLabel(clip.caption_source)}</div>
        <div class="reason-list">${reasonsHtml}</div>

        <button type="button" class="btn-ghost btn-xs js-toggle-edit" style="justify-content:flex-start;">✂ Trim &amp; Edit Captions</button>

        <div class="clip-edit-panel hidden js-edit-panel">
          <div class="clip-trim-row">
            <div class="trim-field">
              <label>Start (s)</label>
              <div class="trim-ctrl">
                <button type="button" class="trim-nudge js-s-minus">-1s</button>
                <input type="number" step="0.5" min="0" class="js-trim-start" value="${clip.start}" />
                <button type="button" class="trim-nudge js-s-plus">+1s</button>
              </div>
            </div>
            <div class="trim-field">
              <label>End (s)</label>
              <div class="trim-ctrl">
                <button type="button" class="trim-nudge js-e-minus">-1s</button>
                <input type="number" step="0.5" min="0.5" class="js-trim-end" value="${clip.end}" />
                <button type="button" class="trim-nudge js-e-plus">+1s</button>
              </div>
            </div>
          </div>
          <div class="trim-field">
            <label>Edit Caption Words</label>
            <textarea rows="2" class="js-edit-text" placeholder="Edit words to fix spelling or wording...">${escapeHtml(editableWordsText)}</textarea>
          </div>
          <div class="clip-edit-actions">
            <span class="mini-status js-edit-msg">Re-burns this clip from cached source</span>
            <button type="button" class="btn-secondary btn-xs js-save-edit">Re-burn Clip</button>
          </div>
        </div>

        <div class="social-box">
          <div class="social-head">
            <span class="social-title-label">Upload Copy Kit</span>
            <div class="social-tabs">
              <button type="button" class="s-tab active" data-plat="yt">YouTube</button>
              <button type="button" class="s-tab" data-plat="fb">Facebook</button>
            </div>
          </div>
          <div class="copy-row">
            <div class="copy-val"><strong>Title:</strong> <span class="js-title">${escapeHtml(social.title)}</span></div>
            <button type="button" class="copy-btn js-copy-title">Copy</button>
          </div>
          <div class="copy-row">
            <div class="copy-val copy-caption js-caption">${escapeHtml(social.youtube_caption)}</div>
            <button type="button" class="copy-btn js-copy-cap">Copy</button>
          </div>
          <div class="copy-row">
            <div class="copy-val"><strong>Tags:</strong> <span class="js-tags">${escapeHtml(tagsStr)}</span></div>
            <button type="button" class="copy-btn js-copy-tags">Copy</button>
          </div>
          <button type="button" class="copy-full-btn js-copy-all">Copy Full YouTube Post (Title + Caption + Hashtags)</button>
        </div>

        <div class="clip-actions">
          <a class="dl-mp4 js-dl-mp4" href="${clip.video_url}" download="${clip.filename}">Download MP4</a>
          ${srtBtn}
        </div>
      </div>
    `;

    // Safe-zone preview toggle
    const szBtn = card.querySelector(".js-sz-btn");
    const szOverlay = card.querySelector(".js-sz-overlay");
    szBtn.addEventListener("click", () => {
      const on = szOverlay.classList.toggle("hidden");
      szBtn.classList.toggle("active", !on);
    });

    // Inline Trim & Caption Editor
    const editPanel = card.querySelector(".js-edit-panel");
    const toggleEditBtn = card.querySelector(".js-toggle-edit");
    const startInput = card.querySelector(".js-trim-start");
    const endInput = card.querySelector(".js-trim-end");
    const textArea = card.querySelector(".js-edit-text");
    const editMsg = card.querySelector(".js-edit-msg");
    const saveEditBtn = card.querySelector(".js-save-edit");

    toggleEditBtn.addEventListener("click", () => editPanel.classList.toggle("hidden"));
    card.querySelector(".js-s-minus").addEventListener("click", () => {
      startInput.value = Math.max(0, Number(startInput.value || 0) - 1).toFixed(1);
    });
    card.querySelector(".js-s-plus").addEventListener("click", () => {
      startInput.value = (Number(startInput.value || 0) + 1).toFixed(1);
    });
    card.querySelector(".js-e-minus").addEventListener("click", () => {
      endInput.value = Math.max(1, Number(endInput.value || 1) - 1).toFixed(1);
    });
    card.querySelector(".js-e-plus").addEventListener("click", () => {
      endInput.value = (Number(endInput.value || 1) + 1).toFixed(1);
    });

    saveEditBtn.addEventListener("click", async () => {
      saveEditBtn.disabled = true;
      saveEditBtn.textContent = "Re-burning...";
      editMsg.textContent = "Rendering updated clip...";
      try {
        const res = await fetch(`/api/jobs/${job.job_id}/clips/${clip.rank}/update`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            start: Number(startInput.value),
            end: Number(endInput.value),
            caption_text: textArea.value,
          }),
        });
        const updated = await res.json();
        if (!res.ok) throw new Error(updated.detail || "Could not update clip.");

        const videoEl = card.querySelector("video");
        videoEl.poster = updated.thumb_url || "";
        videoEl.querySelector("source").src = updated.video_url;
        videoEl.load();

        card.querySelector(".js-clip-time").textContent =
          `${fmtTime(updated.start)} – ${fmtTime(updated.end)} (${Math.round(updated.duration)}s) • ${captionSourceLabel(updated.caption_source)}`;
        card.querySelector(".js-dl-mp4").href = updated.video_url;
        const srtEl = card.querySelector(".js-dl-srt");
        if (srtEl && updated.srt_url) srtEl.href = updated.srt_url;
        editMsg.textContent = "Updated!";
        editMsg.classList.add("loaded");
      } catch (err) {
        editMsg.textContent = err.message;
        editMsg.classList.remove("loaded");
      } finally {
        saveEditBtn.disabled = false;
        saveEditBtn.textContent = "Re-burn Clip";
      }
    });

    // Social copy tabs
    let activePlatform = "yt";
    const tabBtns = card.querySelectorAll(".s-tab");
    const capEl = card.querySelector(".js-caption");
    const copyAllBtn = card.querySelector(".js-copy-all");

    tabBtns.forEach((tb) => {
      tb.addEventListener("click", () => {
        activePlatform = tb.dataset.plat;
        tabBtns.forEach((b) => b.classList.toggle("active", b === tb));
        capEl.textContent = activePlatform === "yt" ? social.youtube_caption : social.facebook_caption;
        copyAllBtn.textContent =
          activePlatform === "yt"
            ? "Copy Full YouTube Post (Title + Caption + Hashtags)"
            : "Copy Full Facebook Post (Title + Caption + Hashtags)";
      });
    });

    card.querySelector(".js-copy-title").addEventListener("click", (e) => copyText(e.currentTarget, social.title, "Copy"));
    card.querySelector(".js-copy-cap").addEventListener("click", (e) =>
      copyText(e.currentTarget, activePlatform === "yt" ? social.youtube_caption : social.facebook_caption, "Copy")
    );
    card.querySelector(".js-copy-tags").addEventListener("click", (e) => copyText(e.currentTarget, tagsStr, "Copy"));
    copyAllBtn.addEventListener("click", (e) =>
      copyText(
        e.currentTarget,
        activePlatform === "yt" ? social.copy_all_youtube : social.copy_all_facebook,
        e.currentTarget.textContent
      )
    );

    grid.appendChild(card);
  }

  $("resultsSection").classList.remove("hidden");
}

function anchorJobOutputAfter(sectionEl) {
  const prog = $("progressCard");
  const res = $("resultsSection");
  if (!sectionEl || !prog || !res) return;
  sectionEl.insertAdjacentElement("afterend", prog);
  prog.insertAdjacentElement("afterend", res);
}

function updateProgressUI(job) {
  $("progressCard").classList.remove("hidden");
  $("stageBadge").textContent = job.stage;
  $("progressTitle").textContent = job.video_title || `Processing ${job.platform || "video"}...`;
  $("progressPct").textContent = `${job.progress}%`;
  $("progressFill").style.width = `${job.progress}%`;
  $("progressMsg").textContent = job.message;
  const cancelBtn = $("cancelCurrentJobBtn");
  if (cancelBtn) {
    cancelBtn.classList.toggle("hidden", !["queued", "running"].includes(job.status));
  }
}

function startListening(jobId) {
  if (state.eventSource) {
    state.eventSource.close();
  }
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  state.eventSource = es;

  es.onmessage = (ev) => {
    const job = JSON.parse(ev.data);
    updateProgressUI(job);
    if (job.status === "completed") {
      es.close();
      renderClips(job);
      loadHistory();
      refreshQueue();
      checkHealth();
    } else if (job.status === "failed") {
      es.close();
      showError(job.error || "Failed to generate clips.");
      refreshQueue();
    } else if (job.status === "cancelled") {
      es.close();
      refreshQueue();
    }
  };

  es.onerror = async () => {
    es.close();
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (!res.ok) return;
      const job = await res.json();
      updateProgressUI(job);
      if (job.status === "completed") {
        renderClips(job);
        loadHistory();
        refreshQueue();
        checkHealth();
      } else if (job.status === "failed") {
        showError(job.error || "Failed to generate clips.");
        refreshQueue();
      } else if (job.status === "cancelled") {
        refreshQueue();
      } else {
        setTimeout(() => startListening(jobId), 1200);
      }
    } catch {
      refreshQueue();
    }
  };
}

function buildClipRequestPayload(url) {
  return {
    url,
    num_clips: state.numClips,
    highlighted_captions: $("captionsToggle").checked,
    caption_language: state.captionLanguage,
    aspect_ratio: state.aspectRatio,
    highlight_color: state.highlightColor,
    caption_style: $("captionStyleSelect").value,
    caption_position: $("captionPosSelect").value,
    caption_scale: Number($("captionScaleSelect").value || 1.0),
    caption_mode: $("captionModeSelect").value,
    clip_duration: Number($("clipDurSelect").value),
    quality: $("qualitySelect").value,
    manual_start: parseTimeInput($("manualStartInput").value),
    manual_end: parseTimeInput($("manualEndInput").value),
    trim_silence: $("trimSilenceToggle").checked,
    hook_banner: $("hookBannerInput").value.trim(),
    streamer_cam_source: $("camSourceSelect").value,
    streamer_cam_target: state.camTarget,
    split_cam_position: state.splitCamPosition,
    watermark_text: $("watermarkInput").value.trim(),
    audio_boost: $("audioBoostToggle").checked,
    keep_source: $("keepSourceToggle").checked,
    use_source_cache: $("useSourceCacheToggle").checked,
    encoder_mode: $("encoderModeSelect").value,
    music_volume: Number($("musicVolumeInput").value || 14) / 100.0,
    bg_music_id: state.bgMusicId || "",
    ai_provider: $("aiProviderSelect").value || "auto",
    ollama_enabled: $("ollamaEnabledToggle").checked,
    ollama_model: $("ollamaModelSelect").value || "",
    ollama_topic: $("ollamaTopicInput").value.trim(),
    cookies_browser: $("cookiesBrowserSelect")?.value || "",
  };
}

document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  loadHistory();
  refreshQueue();
  refreshProfileSelect();

  const outputCorners = new Set(["top-left", "top-right", "bottom-left", "bottom-right"]);
  if ($("camSourceSelect").value && outputCorners.has($("camSourceSelect").value)) {
    state.camTarget = $("camSourceSelect").value;
    document.querySelectorAll("#camTargetGroup .corner-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.corner === state.camTarget);
    });
  }
  updatePipDiagram();

  $("refreshLocalAiBtn").addEventListener("click", refreshLocalAiStatus);
  $("aiProviderSelect").addEventListener("change", () => renderLocalAiModels(state.localAiInfo));

  // Presets handlers
  $("profileSelect").addEventListener("change", (e) => {
    const val = e.target.value;
    if (val === "__default__") {
      applyPreset(DEFAULT_PRESET);
      $("deleteProfileBtn").classList.add("hidden");
    } else {
      const profiles = readSavedProfiles();
      if (profiles[val]) applyPreset(profiles[val]);
      $("deleteProfileBtn").classList.remove("hidden");
    }
  });
  $("saveProfileBtn").addEventListener("click", () => {
    $("profileSaveBox").classList.toggle("hidden");
    $("profileNameInput").focus();
  });
  $("cancelSaveProfileBtn").addEventListener("click", () => {
    $("profileSaveBox").classList.add("hidden");
  });
  $("confirmSaveProfileBtn").addEventListener("click", () => {
    const name = $("profileNameInput").value.trim();
    if (!name) return;
    const profiles = readSavedProfiles();
    profiles[name] = captureCurrentPreset();
    writeSavedProfiles(profiles);
    refreshProfileSelect(name);
    $("profileNameInput").value = "";
    $("profileSaveBox").classList.add("hidden");
  });
  $("deleteProfileBtn").addEventListener("click", () => {
    const val = $("profileSelect").value;
    if (val === "__default__") return;
    const profiles = readSavedProfiles();
    delete profiles[val];
    writeSavedProfiles(profiles);
    refreshProfileSelect("__default__");
    applyPreset(DEFAULT_PRESET);
  });

  // Batch queue drawer
  $("batchQueueToggleBtn").addEventListener("click", () => {
    $("batchBox").classList.toggle("hidden");
  });
  $("cancelBatchBtn").addEventListener("click", () => {
    $("batchBox").classList.add("hidden");
  });
  $("submitBatchBtn").addEventListener("click", async () => {
    clearError();
    const lines = $("batchUrlsInput").value
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (!lines.length) {
      showError("Paste one or more video links (one per line) to add to the batch queue.");
      return;
    }
    anchorJobOutputAfter($("clipForm").closest(".hero-card"));
    let firstJob = null;
    for (const link of lines) {
      try {
        const res = await fetch("/api/clips", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(buildClipRequestPayload(link)),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `Invalid link: ${link}`);
        if (!firstJob) firstJob = data;
      } catch (err) {
        showError(err.message);
      }
    }
    if (firstJob) {
      $("batchUrlsInput").value = "";
      $("batchBox").classList.add("hidden");
      state.currentJobId = firstJob.job_id;
      updateProgressUI(firstJob);
      startListening(firstJob.job_id);
      refreshQueue();
    }
  });

  $("pasteBtn").addEventListener("click", async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) $("urlInput").value = text.trim();
    } catch {
      $("urlInput").focus();
    }
  });

  $("inspectBtn").addEventListener("click", inspectUrl);

  $("decClips").addEventListener("click", () => setClipCount(state.numClips - 1));
  $("incClips").addEventListener("click", () => setClipCount(state.numClips + 1));
  document.querySelectorAll(".q-btn").forEach((btn) => {
    btn.addEventListener("click", () => setClipCount(Number(btn.dataset.count)));
  });

  $("captionsToggle").addEventListener("change", (e) => {
    const on = e.target.checked;
    $("captionsStatusText").textContent = on ? "Word Highlight ON" : "Captions OFF";
    $("colorDots").style.opacity = on ? "1" : "0.35";
    $("colorDots").style.pointerEvents = on ? "auto" : "none";
    updatePipDiagram();
  });

  $("captionPosSelect").addEventListener("change", updatePipDiagram);

  document.querySelectorAll("#colorDots .c-dot").forEach((dot) => {
    dot.addEventListener("click", () => {
      state.highlightColor = dot.dataset.color;
      document.querySelectorAll("#colorDots .c-dot").forEach((d) => d.classList.toggle("active", d === dot));
    });
  });

  document.querySelectorAll("#aspectGroup .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.aspectRatio = btn.dataset.aspect;
      setActiveSeg("#aspectGroup", "aspect", state.aspectRatio);
      updatePipDiagram();
    });
  });

  document.querySelectorAll("#langGroup .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.captionLanguage = btn.dataset.lang;
      setActiveSeg("#langGroup", "lang", state.captionLanguage);
    });
  });

  const onCamSourceChange = (e) => {
    const val = e.target.value || "off";
    state.camSource = val;
    if (outputCorners.has(val)) {
      state.camTarget = val;
      document.querySelectorAll("#camTargetGroup .corner-btn").forEach((b) => {
        b.classList.toggle("active", b.dataset.corner === val);
      });
    }
    updatePipDiagram();
  };
  $("camSourceSelect").addEventListener("change", onCamSourceChange);
  $("camSourceSelect").addEventListener("input", onCamSourceChange);

  document.querySelectorAll("#camTargetGroup .corner-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.camTarget = btn.dataset.corner;
      document.querySelectorAll("#camTargetGroup .corner-btn").forEach((b) => b.classList.toggle("active", b === btn));
      updatePipDiagram();
    });
  });

  document.querySelectorAll("#splitPosGroup .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.splitCamPosition = btn.dataset.splitpos;
      setActiveSeg("#splitPosGroup", "splitpos", state.splitCamPosition);
      updatePipDiagram();
    });
  });

  // Background Music Upload
  $("musicVolumeInput").addEventListener("input", (e) => {
    $("musicVolVal").textContent = `${e.target.value}%`;
  });
  $("uploadMusicBtn").addEventListener("click", () => $("bgMusicInput").click());
  $("bgMusicInput").addEventListener("change", async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    clearError();
    const fd = new FormData();
    fd.append("file", f);
    try {
      const res = await fetch("/api/music/upload", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to upload audio track.");
      state.bgMusicId = data.bg_music_id;
      $("musicSelectedName").textContent = data.filename;
      $("musicSelectedName").classList.add("loaded");
      $("clearMusicBtn").classList.remove("hidden");
    } catch (err) {
      showError(err.message);
    } finally {
      $("bgMusicInput").value = "";
    }
  });
  $("clearMusicBtn").addEventListener("click", () => {
    state.bgMusicId = "";
    $("musicSelectedName").textContent = "None selected";
    $("musicSelectedName").classList.remove("loaded");
    $("clearMusicBtn").classList.add("hidden");
  });

  $("clearSourceCacheBtn").addEventListener("click", async (e) => {
    e.preventDefault();
    e.stopPropagation();
    await fetch("/api/cache/sources", { method: "DELETE" });
    checkHealth();
  });

  $("cancelCurrentJobBtn").addEventListener("click", async () => {
    if (!state.currentJobId) return;
    await fetch(`/api/jobs/${state.currentJobId}/cancel`, { method: "POST" });
    refreshQueue();
  });

  $("openFolderBtn").addEventListener("click", async () => {
    if (!state.currentJobId) return;
    await fetch(`/api/jobs/${state.currentJobId}/open-folder`, { method: "POST" });
  });

  $("pickLocalBtn").addEventListener("click", () => $("localFileInput").click());
  $("localFileInput").addEventListener("change", (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    state.localFile = f;
    const mb = (f.size / (1024 * 1024)).toFixed(1);
    $("localFileName").textContent = `${f.name} (${mb} MB)`;
    $("localFileBar").classList.remove("hidden");
    clearError();
  });
  $("clearLocalBtn").addEventListener("click", () => {
    state.localFile = null;
    $("localFileInput").value = "";
    $("localFileBar").classList.add("hidden");
  });

  $("ytLoginPill").addEventListener("click", () => {
    if ($("ytLoginPill").classList.contains("pill-good")) {
      $("advancedDrawer").open = true;
      $("advancedDrawer").scrollIntoView({ behavior: "smooth" });
    } else {
      startYouTubeLogin();
    }
  });
  $("ytLoginOptBtn").addEventListener("click", startYouTubeLogin);
  $("ytAuthSaveBtn").addEventListener("click", async () => {
    clearError();
    try {
      const res = await fetch("/api/youtube-login/save", { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Not signed in yet.");
      updateCookiesBadge(data.cookies_file_loaded);
      $("ytAuthBar").classList.add("hidden");
      if (state.ytAuthTimer) clearInterval(state.ytAuthTimer);
    } catch (err) {
      showError(err.message);
    }
  });
  $("ytAuthCancelBtn").addEventListener("click", async () => {
    await fetch("/api/youtube-login/cancel", { method: "POST" });
    if (state.ytAuthTimer) clearInterval(state.ytAuthTimer);
    $("ytAuthBar").classList.add("hidden");
  });

  $("uploadCookiesBtn").addEventListener("click", () => $("cookiesFileInput").click());
  $("cookiesFileInput").addEventListener("change", async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    clearError();
    const fd = new FormData();
    fd.append("file", f);
    try {
      const res = await fetch("/api/cookies", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to load cookies file.");
      updateCookiesBadge(data.cookies_file_loaded);
    } catch (err) {
      showError(err.message);
    } finally {
      $("cookiesFileInput").value = "";
    }
  });

  $("clearCookiesBtn").addEventListener("click", async () => {
    const res = await fetch("/api/cookies", { method: "DELETE" });
    const data = await res.json();
    updateCookiesBadge(data.cookies_file_loaded);
  });

  $("pasteCookiesToggleBtn").addEventListener("click", () => {
    $("cookiesPasteBox").classList.toggle("hidden");
  });
  $("cancelPasteCookiesBtn").addEventListener("click", () => {
    $("cookiesPasteBox").classList.add("hidden");
  });

  $("saveCookiesTextBtn").addEventListener("click", async () => {
    clearError();
    const raw = $("cookiesTextInput").value.trim();
    if (!raw) {
      showError("Paste your Netscape cookies.txt content first.");
      return;
    }
    const fd = new FormData();
    fd.append("cookies_text", raw);
    try {
      const res = await fetch("/api/cookies", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Invalid cookies text.");
      updateCookiesBadge(data.cookies_file_loaded);
      $("cookiesTextInput").value = "";
      $("cookiesPasteBox").classList.add("hidden");
    } catch (err) {
      showError(err.message);
    }
  });

  $("clipForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();
    const url = $("urlInput").value.trim();
    if (!state.localFile && !url) {
      showError("Please enter a video/stream link or pick a Local Video file.");
      return;
    }

    const mStart = parseTimeInput($("manualStartInput").value);
    const mEnd = parseTimeInput($("manualEndInput").value);
    if (mStart !== null && mEnd !== null && mEnd <= mStart + 0.5) {
      showError("Manual End Time must be greater than Manual Start Time.");
      return;
    }

    anchorJobOutputAfter($("clipForm").closest(".hero-card"));
    $("resultsSection").classList.add("hidden");
    $("clipsGrid").innerHTML = "";

    try {
      let res;
      if (state.localFile) {
        const fd = new FormData();
        fd.append("file", state.localFile);
        fd.append("num_clips", String(state.numClips));
        fd.append("highlighted_captions", String($("captionsToggle").checked));
        fd.append("caption_language", state.captionLanguage);
        fd.append("aspect_ratio", state.aspectRatio);
        fd.append("highlight_color", state.highlightColor);
        fd.append("caption_style", $("captionStyleSelect").value);
        fd.append("caption_position", $("captionPosSelect").value);
        fd.append("caption_scale", $("captionScaleSelect").value);
        fd.append("caption_mode", $("captionModeSelect").value);
        fd.append("clip_duration", $("clipDurSelect").value);
        fd.append("quality", $("qualitySelect").value);
        if (mStart !== null) fd.append("manual_start", String(mStart));
        if (mEnd !== null) fd.append("manual_end", String(mEnd));
        fd.append("trim_silence", String($("trimSilenceToggle").checked));
        fd.append("hook_banner", $("hookBannerInput").value.trim());
        fd.append("streamer_cam_source", $("camSourceSelect").value);
        fd.append("streamer_cam_target", state.camTarget);
        fd.append("split_cam_position", state.splitCamPosition);
        fd.append("watermark_text", $("watermarkInput").value.trim());
        fd.append("audio_boost", String($("audioBoostToggle").checked));
        fd.append("keep_source", String($("keepSourceToggle").checked));
        fd.append("use_source_cache", String($("useSourceCacheToggle").checked));
        fd.append("encoder_mode", $("encoderModeSelect").value);
        fd.append("music_volume", String(Number($("musicVolumeInput").value || 14) / 100.0));
        fd.append("bg_music_id", state.bgMusicId || "");
        fd.append("ai_provider", $("aiProviderSelect").value || "auto");
        fd.append("ollama_enabled", String($("ollamaEnabledToggle").checked));
        fd.append("ollama_model", $("ollamaModelSelect").value || "");
        fd.append("ollama_topic", $("ollamaTopicInput").value.trim());
        res = await fetch("/api/clips/upload", { method: "POST", body: fd });
      } else {
        const payload = buildClipRequestPayload(url);
        res = await fetch("/api/clips", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      }
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Could not start job.");
      state.currentJobId = data.job_id;
      updateProgressUI(data);
      startListening(data.job_id);
      refreshQueue();
    } catch (err) {
      showError(err.message);
    }
  });

  // Pre-Edited Reel Captioner controls
  function selectReelFile(f) {
    if (!f) return;
    state.reelFile = f;
    const mb = (f.size / (1024 * 1024)).toFixed(1);
    $("reelFileLabel").textContent = `${f.name} (${mb} MB)`;
    $("reelDropPrompt").classList.add("hidden");
    $("reelFileSelected").classList.remove("hidden");
    $("reelErrorBanner").classList.add("hidden");
  }

  const dropZone = $("reelDropZone");
  dropZone.addEventListener("click", (e) => {
    if (e.target.id === "clearReelBtn") return;
    $("reelFileInput").click();
  });
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      $("reelFileInput").click();
    }
  });
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const f = e.dataTransfer?.files?.[0];
    if (f) selectReelFile(f);
  });

  $("reelFileInput").addEventListener("change", (e) => {
    const f = e.target.files?.[0];
    if (f) selectReelFile(f);
  });

  $("clearReelBtn").addEventListener("click", (e) => {
    e.stopPropagation();
    state.reelFile = null;
    $("reelFileInput").value = "";
    $("reelFileSelected").classList.add("hidden");
    $("reelDropPrompt").classList.remove("hidden");
  });

  document.querySelectorAll("#reelLangGroup .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.reelLanguage = btn.dataset.lang;
      setActiveSeg("#reelLangGroup", "lang", state.reelLanguage);
    });
  });

  document.querySelectorAll("#reelColorDots .c-dot").forEach((dot) => {
    dot.addEventListener("click", () => {
      state.reelColor = dot.dataset.color;
      document.querySelectorAll("#reelColorDots .c-dot").forEach((d) => d.classList.toggle("active", d === dot));
    });
  });

  document.querySelectorAll("#reelAspectGroup .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.reelAspect = btn.dataset.aspect;
      setActiveSeg("#reelAspectGroup", "aspect", state.reelAspect);
    });
  });

  function syncDonateCheckoutLink() {
    const rawAmt =
      state.donateAmount === "custom" ? $("donateCustomInput").value.trim() : state.donateAmount;
    const valid = Number(rawAmt) > 0 ? Number(rawAmt) : 5;
    const url = buildPayPalDonateUrl(valid);
    const goBtn = $("paypalDonateGoBtn");
    if (goBtn) {
      goBtn.href = url;
      goBtn.textContent = `Continue to PayPal ($${valid}) →`;
    }
  }

  if ($("footerDonateLink")) {
    $("footerDonateLink").href = buildPayPalDonateUrl(5);
  }
  if ($("paypalEmailText")) {
    $("paypalEmailText").textContent = PAYPAL_CONFIG.businessEmail;
  }

  if ($("donatePill") && $("donateModal")) {
    $("donatePill").addEventListener("click", () => {
      syncDonateCheckoutLink();
      $("donateModal").classList.remove("hidden");
    });
    $("closeDonateModalBtn").addEventListener("click", () => {
      $("donateModal").classList.add("hidden");
    });
    $("donateModal").addEventListener("click", (e) => {
      if (e.target === $("donateModal")) $("donateModal").classList.add("hidden");
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !$("donateModal").classList.contains("hidden")) {
        $("donateModal").classList.add("hidden");
      }
    });
    document.querySelectorAll("#donateAmountGroup .seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.donateAmount = btn.dataset.amt;
        setActiveSeg("#donateAmountGroup", "amt", state.donateAmount);
        const isCustom = state.donateAmount === "custom";
        $("donateCustomWrap").classList.toggle("hidden", !isCustom);
        if (isCustom) $("donateCustomInput").focus();
        syncDonateCheckoutLink();
      });
    });
    $("donateCustomInput").addEventListener("input", syncDonateCheckoutLink);
    $("copyPaypalEmailBtn").addEventListener("click", (e) =>
      copyText(e.currentTarget, PAYPAL_CONFIG.businessEmail, "Copy")
    );
  }

  $("reelForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("reelErrorBanner").classList.add("hidden");
    if (!state.reelFile) {
      $("reelErrorBanner").textContent = "Please drop or select your pre-edited reel video first.";
      $("reelErrorBanner").classList.remove("hidden");
      return;
    }

    anchorJobOutputAfter($("reelCaptionSection"));
    $("resultsSection").classList.add("hidden");
    $("clipsGrid").innerHTML = "";

    const fd = new FormData();
    fd.append("file", state.reelFile);
    fd.append("caption_language", state.reelLanguage);
    fd.append("highlight_color", state.reelColor);
    fd.append("aspect_ratio", state.reelAspect);
    fd.append("quality", $("reelQualitySelect").value);
    fd.append("custom_script", $("reelScriptInput").value.trim());
    fd.append("watermark_text", $("reelWatermarkInput").value.trim());
    fd.append("audio_boost", String($("reelAudioBoostToggle").checked));
    fd.append("caption_style", $("captionStyleSelect").value);
    fd.append("caption_position", $("captionPosSelect").value);
    fd.append("caption_scale", $("captionScaleSelect").value);
    fd.append("hook_banner", $("reelHookBannerInput").value.trim());
    fd.append("trim_silence", String($("reelTrimSilenceToggle").checked));
    fd.append("keep_source", String($("keepSourceToggle").checked));
    fd.append("encoder_mode", $("encoderModeSelect").value);
    fd.append("bg_music_id", state.bgMusicId || "");
    fd.append("music_volume", String(Number($("musicVolumeInput").value || 14) / 100.0));

    try {
      const res = await fetch("/api/reels/caption", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Could not process reel.");
      state.currentJobId = data.job_id;
      updateProgressUI(data);
      $("progressCard").scrollIntoView({ behavior: "smooth" });
      startListening(data.job_id);
      refreshQueue();
    } catch (err) {
      $("reelErrorBanner").textContent = err.message;
      $("reelErrorBanner").classList.remove("hidden");
    }
  });
});
