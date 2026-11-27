const $ = (id) => document.getElementById(id);

const PAYPAL_CONFIG = {
  businessEmail: "hmjsarkar@gmail.com",
  hostedButtonId: "", // Optional: set PayPal hosted_button_id to override email link
  paypalMeHandle: "", // Optional: set PayPal.Me handle (e.g. "hmjsarkar") to use paypal.me
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
  // Use standard _xclick checkout (avoids "Donations aren't supported in this organisation's country" in IN/restricted regions)
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
  currentJobId: null,
  localFile: null,
  reelFile: null,
  reelLanguage: "auto",
  reelColor: "yellow",
  reelAspect: "original",
  donateAmount: "5",
  eventSource: null,
  ytAuthTimer: null,
};

function fmtTime(sec) {
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  const h = Math.floor(m / 60);
  if (h > 0) return `${h}:${String(m % 60).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
  return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
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

function updateCookiesBadge(loaded) {
  const badge = $("cookiesBadge");
  const clearBtn = $("clearCookiesBtn");
  const pill = $("ytLoginPill");
  const pillText = $("ytLoginPillText");
  if (badge) {
    badge.textContent = loaded ? "(YouTube Signed In)" : "(Not Signed In)";
    badge.classList.toggle("loaded", Boolean(loaded));
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

function clearError() {
  $("errorBanner").classList.add("hidden");
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

function updatePipDiagram() {
  state.camSource = $("camSourceSelect").value || "off";
  const enabled = state.camSource !== "off";
  $("camTargetGroup").classList.toggle("disabled", !enabled);
  document.querySelectorAll("#camTargetGroup .corner-btn").forEach((btn) => {
    btn.disabled = !enabled;
  });
  const box = $("pipBox");
  box.classList.toggle("hidden", !enabled);
  box.classList.remove("top-left", "top-right", "bottom-left", "bottom-right");
  box.classList.add(state.camTarget);
  const statusEl = $("pipStatusLine");
  if (statusEl) {
    statusEl.classList.toggle("active", enabled);
    if (!enabled) {
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
  } catch {
    $("offlineModelPill").textContent = "Server Connecting...";
  }
}

async function loadHistory() {
  try {
    const res = await fetch("/api/jobs");
    if (!res.ok) return;
    const jobs = await res.json();
    if (!jobs.length) return;
    const list = $("historyList");
    list.innerHTML = "";
    for (const j of jobs) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "history-chip";
      btn.innerHTML = `
        <div class="hc-title">${escapeHtml(j.video_title || j.job_id)}</div>
        <div class="hc-meta">${escapeHtml(j.platform || "Video")} • ${j.clips.length} clips</div>
      `;
      btn.addEventListener("click", () => {
        state.currentJobId = j.job_id;
        anchorJobOutputAfter(
          j.platform === "Pre-Edited Reel"
            ? $("reelCaptionSection")
            : $("clipForm").closest(".hero-card")
        );
        renderClips(j);
        $("resultsSection").scrollIntoView({ behavior: "smooth" });
      });
      list.appendChild(btn);
    }
    $("historySection").classList.remove("hidden");
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
    default:
      return "No Captions";
  }
}

function renderClips(job) {
  $("aiNote").textContent = job.ai_note || "";
  $("downloadAllBtn").href = `/api/jobs/${job.job_id}/download-all`;
  const grid = $("clipsGrid");
  grid.innerHTML = "";

  for (const clip of job.clips) {
    const card = document.createElement("article");
    card.className = "clip-card";
    const reasonsHtml = (clip.reasons || []).map((r) => `<span class="reason-tag">${escapeHtml(r)}</span>`).join("");
    const srtBtn = clip.srt_url
      ? `<a class="dl-srt" href="${clip.srt_url}" download="${clip.srt_filename || `clip_${clip.rank}.srt`}">Download .SRT</a>`
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

    card.innerHTML = `
      <div class="clip-video-wrap">
        <video controls playsinline preload="metadata" poster="${clip.thumb_url || ""}">
          <source src="${clip.video_url}" type="video/mp4" />
        </video>
      </div>
      <div class="clip-body">
        <div class="clip-meta-top">
          <span class="clip-rank">Clip #${clip.rank}</span>
          <span class="score-pill">Score ${clip.score}/100</span>
        </div>
        <div class="clip-time">${fmtTime(clip.start)} – ${fmtTime(clip.end)} (${Math.round(clip.duration)}s) • ${captionSourceLabel(clip.caption_source)}</div>
        <div class="reason-list">${reasonsHtml}</div>

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
          <a class="dl-mp4" href="${clip.video_url}" download="${clip.filename}">Download MP4</a>
          ${srtBtn}
        </div>
      </div>
    `;

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
      $("generateBtn").disabled = false;
      if ($("reelSubmitBtn")) $("reelSubmitBtn").disabled = false;
      renderClips(job);
      loadHistory();
    } else if (job.status === "failed") {
      es.close();
      $("generateBtn").disabled = false;
      if ($("reelSubmitBtn")) $("reelSubmitBtn").disabled = false;
      showError(job.error || "Failed to generate clips.");
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
        $("generateBtn").disabled = false;
        if ($("reelSubmitBtn")) $("reelSubmitBtn").disabled = false;
        renderClips(job);
        loadHistory();
      } else if (job.status === "failed") {
        $("generateBtn").disabled = false;
        if ($("reelSubmitBtn")) $("reelSubmitBtn").disabled = false;
        showError(job.error || "Failed to generate clips.");
      } else {
        setTimeout(() => startListening(jobId), 1200);
      }
    } catch {
      $("generateBtn").disabled = false;
      if ($("reelSubmitBtn")) $("reelSubmitBtn").disabled = false;
    }
  };
}

document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  loadHistory();
  const outputCorners = new Set(["top-left", "top-right", "bottom-left", "bottom-right"]);
  if ($("camSourceSelect").value && outputCorners.has($("camSourceSelect").value)) {
    state.camTarget = $("camSourceSelect").value;
    document.querySelectorAll("#camTargetGroup .corner-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.corner === state.camTarget);
    });
  }
  updatePipDiagram();

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
  });

  document.querySelectorAll(".c-dot").forEach((dot) => {
    dot.addEventListener("click", () => {
      state.highlightColor = dot.dataset.color;
      document.querySelectorAll(".c-dot").forEach((d) => d.classList.toggle("active", d === dot));
    });
  });

  document.querySelectorAll("#aspectGroup .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.aspectRatio = btn.dataset.aspect;
      document.querySelectorAll("#aspectGroup .seg-btn").forEach((b) => b.classList.toggle("active", b === btn));
    });
  });

  document.querySelectorAll("#langGroup .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.captionLanguage = btn.dataset.lang;
      document.querySelectorAll("#langGroup .seg-btn").forEach((b) => b.classList.toggle("active", b === btn));
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

    anchorJobOutputAfter($("clipForm").closest(".hero-card"));
    $("generateBtn").disabled = true;
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
        fd.append("caption_mode", $("captionModeSelect").value);
        fd.append("clip_duration", $("clipDurSelect").value);
        fd.append("quality", $("qualitySelect").value);
        fd.append("streamer_cam_source", $("camSourceSelect").value);
        fd.append("streamer_cam_target", state.camTarget);
        fd.append("watermark_text", $("watermarkInput").value.trim());
        fd.append("audio_boost", String($("audioBoostToggle").checked));
        res = await fetch("/api/clips/upload", { method: "POST", body: fd });
      } else {
        const payload = {
          url,
          num_clips: state.numClips,
          highlighted_captions: $("captionsToggle").checked,
          caption_language: state.captionLanguage,
          aspect_ratio: state.aspectRatio,
          highlight_color: state.highlightColor,
          caption_mode: $("captionModeSelect").value,
          clip_duration: Number($("clipDurSelect").value),
          quality: $("qualitySelect").value,
          streamer_cam_source: $("camSourceSelect").value,
          streamer_cam_target: state.camTarget,
          watermark_text: $("watermarkInput").value.trim(),
          audio_boost: $("audioBoostToggle").checked,
          cookies_browser: $("cookiesBrowserSelect")?.value || "",
        };
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
    } catch (err) {
      $("generateBtn").disabled = false;
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
      document.querySelectorAll("#reelLangGroup .seg-btn").forEach((b) => b.classList.toggle("active", b === btn));
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
      document.querySelectorAll("#reelAspectGroup .seg-btn").forEach((b) => b.classList.toggle("active", b === btn));
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
        document.querySelectorAll("#donateAmountGroup .seg-btn").forEach((b) => b.classList.toggle("active", b === btn));
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
    $("reelSubmitBtn").disabled = true;
    $("generateBtn").disabled = true;
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

    try {
      const res = await fetch("/api/reels/caption", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Could not process reel.");
      state.currentJobId = data.job_id;
      updateProgressUI(data);
      $("progressCard").scrollIntoView({ behavior: "smooth" });
      startListening(data.job_id);
    } catch (err) {
      $("reelSubmitBtn").disabled = false;
      $("generateBtn").disabled = false;
      $("reelErrorBanner").textContent = err.message;
      $("reelErrorBanner").classList.remove("hidden");
    }
  });
});
