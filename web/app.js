(() => {
  const LABEL_TONE = {
    存在夸大功效: "efficacy",
    存在虚假收益承诺: "yield",
    存在诱导消费: "induce",
    存在站外导流风险线索: "offsite",
    存在其他线索: "other",
    正常: "ok",
  };
  const ENGINE_NAME = {
    rules: "规则",
    model: "模型",
    model_audio: "口播",
    model_visual: "画面",
    model_claims: "宣称",
  };
  const SOURCE_NAME = { audio: "口播", visual: "画面", unknown: "原文" };
  const STEP_MAP = {
    start: "ingest",
    load_asr: "extract",
    load_ocr: "extract",
    cache: "extract",
    extract: "extract",
    llm: "review",
    detect: "review",
    classify: "review",
    fuse: "review",
    done: "done",
  };

  const els = {
    file: document.getElementById("file"),
    pick: document.getElementById("pick-file"),
    replace: document.getElementById("replace-file"),
    next: document.getElementById("next-file"),
    drop: document.getElementById("drop"),
    frames: document.getElementById("frames"),
    frameGallery: document.getElementById("frame-gallery"),
    device: document.getElementById("device"),
    player: document.getElementById("player"),
    scan: document.getElementById("scan"),
    name: document.getElementById("video-name"),
    dur: document.getElementById("video-dur"),
    openLib: document.getElementById("open-library"),
    drawer: document.getElementById("drawer"),
    closeLib: document.getElementById("close-library"),
    libSearch: document.getElementById("lib-search"),
    libGrid: document.getElementById("lib-grid"),
    modes: document.getElementById("modes"),
    force: document.getElementById("force"),
    run: document.getElementById("run"),
    runHint: document.getElementById("run-hint"),
    steps: document.getElementById("steps"),
    live: document.getElementById("live"),
    result: document.getElementById("result"),
    verdict: document.getElementById("verdict"),
    stamp: document.getElementById("stamp"),
    verdictKicker: document.getElementById("verdict-kicker"),
    verdictTitle: document.getElementById("verdict-title"),
    explain: document.getElementById("explain"),
    labels: document.getElementById("labels"),
    engines: document.getElementById("engines"),
    error: document.getElementById("error"),
    libPill: document.getElementById("lib-pill"),
    llmPill: document.getElementById("llm-pill"),
  };

  const state = {
    job: null,
    mode: "rules",
    modes: [],
    ollamaReady: false,
    library: [],
    busy: false,
    objectUrl: null,
  };

  const fmtTime = (t) => {
    if (t == null || t === "" || Number.isNaN(Number(t))) return "";
    const n = Number(t);
    const m = Math.floor(n / 60);
    const s = Math.floor(n % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  };

  const fmtDur = (t) => {
    if (t == null) return "";
    return `${Number(t).toFixed(1)} 秒`;
  };

  const showError = (msg) => {
    els.error.hidden = !msg;
    els.error.textContent = msg || "";
  };

  const setBusy = (busy, hint) => {
    state.busy = busy;
    els.run.classList.toggle("is-busy", busy);
    els.run.disabled = busy || !state.job;
    els.runHint.textContent = hint || (state.job ? "对当前视频出具审核意见" : "请先放入视频");
    els.scan.hidden = !busy;
    els.steps.hidden = !busy && els.result.hidden;
    els.live.hidden = !busy;
    [els.pick, els.replace, els.next, els.openLib].forEach((btn) => {
      if (btn) btn.disabled = busy;
    });
  };

  const syncRun = () => {
    const mode = state.modes.find((m) => m.id === state.mode);
    const blocked = mode && mode.needs_llm && !state.ollamaReady;
    els.run.disabled = state.busy || !state.job || blocked;
    if (!state.job) els.runHint.textContent = "请先放入视频";
    else if (blocked) els.runHint.textContent = "该模式需要本机 Ollama";
    else els.runHint.textContent = "对当前视频出具审核意见";
  };

  const setStep = (stage) => {
    const current = STEP_MAP[stage] || "review";
    const order = ["ingest", "extract", "review", "done"];
    const idx = order.indexOf(current);
    els.steps.querySelectorAll("li").forEach((li) => {
      const i = order.indexOf(li.dataset.step);
      li.classList.toggle("is-on", i === idx);
      li.classList.toggle("is-done", i < idx || stage === "done");
    });
  };

  const attachVideo = (url, name, duration) => {
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = url.startsWith("blob:") ? url : null;
    els.player.src = url;
    els.device.classList.add("has-video");
    els.name.textContent = name;
    els.dur.textContent = duration != null ? fmtDur(duration) : "";
  };

  const resetResult = () => {
    els.result.hidden = true;
    els.steps.hidden = true;
    els.live.hidden = true;
    els.frames.hidden = true;
    els.frameGallery.innerHTML = "";
    if (els.next) els.next.hidden = true;
    showError("");
  };

  const clearVideo = () => {
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
    state.job = null;
    els.player.removeAttribute("src");
    els.player.load();
    els.device.classList.remove("has-video");
    els.name.textContent = "尚未选择视频";
    els.dur.textContent = "";
    els.file.value = "";
    resetResult();
    syncRun();
  };

  const openPicker = () => {
    if (state.busy) {
      showError("当前视频仍在审核，请稍后再换下一条");
      return;
    }
    els.file.value = "";
    els.file.click();
  };

  const seek = (t) => {
    if (t == null || !els.player.src) return;
    els.player.currentTime = Number(t);
    els.player.play().catch(() => {});
  };

  const escapeHtml = (text) =>
    String(text || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");

  const highlight = (text, needle) => {
    const safe = escapeHtml(text);
    if (!needle) return safe;
    const n = escapeHtml(needle);
    const i = safe.indexOf(n);
    if (i < 0) return safe;
    return `${safe.slice(0, i)}<mark>${n}</mark>${safe.slice(i + n.length)}`;
  };

  const renderModes = (modes, ollama) => {
    state.modes = modes;
    state.ollamaReady = Boolean(ollama && ollama.ready);
    els.modes.innerHTML = "";
    modes.forEach((mode) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "mode";
      btn.dataset.id = mode.id;
      const locked = mode.needs_llm && !state.ollamaReady;
      if (locked) btn.classList.add("is-off");
      btn.innerHTML = `<b>${mode.name}</b><small>${mode.short}${locked ? " · 需 Ollama" : ""}</small><p>${mode.blurb}</p>`;
      btn.addEventListener("click", () => {
        if (locked) {
          showError(ollama.error || "本机未检测到可用的 Ollama 模型，请先 ollama pull qwen2.5:7b");
          return;
        }
        state.mode = mode.id;
        els.modes.querySelectorAll(".mode").forEach((el) => el.classList.toggle("is-on", el === btn));
        showError("");
        syncRun();
      });
      els.modes.appendChild(btn);
    });
    const preferred = state.ollamaReady ? "hybrid" : "rules";
    state.mode = preferred;
    const on = els.modes.querySelector(`[data-id="${preferred}"]`);
    if (on) on.classList.add("is-on");
  };

  const renderLibrary = (items) => {
    const q = els.libSearch.value.trim();
    els.libGrid.innerHTML = "";
    items
      .filter((it) => !q || it.id.includes(q) || it.name.includes(q))
      .forEach((it) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "lib-item" + (it.cached ? " is-cached" : "");
        btn.innerHTML = `<b>${it.id}</b><small>${it.cached ? "已抽取" : it.name}</small>`;
        btn.addEventListener("click", () => pickLibrary(it.id));
        els.libGrid.appendChild(btn);
      });
  };

  const createJob = async (form) => {
    const res = await fetch("/api/jobs", { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail;
      const msg = typeof detail === "string" ? detail : "创建任务失败";
      throw new Error(msg);
    }
    state.job = data;
    attachVideo(data.media_url, data.video_name, data.duration_sec);
    resetResult();
    syncRun();
  };

  const onFile = async (file) => {
    if (!file) return;
    if (state.busy) {
      showError("当前视频仍在审核，请稍后再换下一条");
      els.file.value = "";
      return;
    }
    const form = new FormData();
    form.append("file", file);
    els.runHint.textContent = "正在接收视频…";
    resetResult();
    try {
      const localUrl = URL.createObjectURL(file);
      attachVideo(localUrl, file.name, null);
      await createJob(form);
    } catch (err) {
      showError(err.message);
    } finally {
      els.file.value = "";
    }
  };

  const pickLibrary = async (id) => {
    if (state.busy) {
      showError("当前视频仍在审核，请稍后再换下一条");
      return;
    }
    const form = new FormData();
    form.append("library_id", id);
    try {
      await createJob(form);
      els.drawer.hidden = true;
    } catch (err) {
      showError(err.message);
    }
  };

  const thumbFor = (item, frames) => {
    if (!frames || !frames.length || item.source !== "visual") return "";
    const hit =
      frames.find((f) => item.time != null && Math.abs(Number(f.time) - Number(item.time)) < 0.8) ||
      frames.find((f) => (f.quotes || []).some((q) => item.matched && String(q).includes(item.matched)));
    if (!hit) return "";
    return `<button type="button" class="frame-thumb" data-seek="${hit.time}">
      <img src="${escapeHtml(hit.url)}" alt="" />
      <span>看问题画面</span>
    </button>`;
  };

  const renderFrames = (frames) => {
    const items = Array.isArray(frames) ? frames : [];
    els.frames.hidden = !items.length;
    els.frameGallery.innerHTML = "";
    items.forEach((frame) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "frame-card";
      if (frame.time != null) btn.dataset.seek = String(frame.time);
      const quote = (frame.quotes || []).slice(0, 2).join(" / ");
      const mark = frame.highlighted ? "已标出问题文案" : "问题发生时刻";
      btn.innerHTML = `
        <img src="${escapeHtml(frame.url)}" alt="${escapeHtml(quote || "问题画面")}" />
        <span class="frame-meta">
          <b>${fmtTime(frame.time) || "画面"}</b>
          <em>${mark}</em>
          ${quote ? `<small>${escapeHtml(quote)}</small>` : ""}
        </span>`;
      els.frameGallery.appendChild(btn);
    });
  };

  const renderResult = (data) => {
    els.result.hidden = false;
    els.steps.hidden = false;
    if (els.next) els.next.hidden = false;
    setStep("done");
    els.verdict.className = `verdict is-${data.verdict}`;
    els.stamp.textContent = data.verdict === "risk" ? "涉及虚假宣传" : "未见虚假宣传";
    els.verdictKicker.textContent = `${data.mode_meta.name} · 样本 ${data.sample_id}`;
    els.verdictTitle.textContent = data.verdict_text;
    els.explain.textContent = data.explanation || "";
    const problemFrames = data.problem_frames || [];
    renderFrames(problemFrames);

    els.labels.innerHTML = "";
    (data.risk_labels || []).forEach((lab) => {
      const chip = document.createElement("span");
      chip.className = `chip ${LABEL_TONE[lab] || "unknown"}`;
      chip.textContent = lab;
      els.labels.appendChild(chip);
    });

    const engines = data.engines || {};
    const keys = Object.keys(engines);
    els.engines.hidden = keys.length < 2;
    els.engines.innerHTML = keys
      .map((k) => {
        const row = engines[k];
        return `<span>${ENGINE_NAME[k] || k}：<b>${(row.risk_labels || []).join("、") || "—"}</b></span>`;
      })
      .join("");

    const evidence = data.evidence || [];
    const evPanel = document.getElementById("panel-evidence");
    if (!evidence.length) {
      evPanel.innerHTML = `<p class="empty">没有可回原文落地的风险证据。</p>`;
    } else {
      evPanel.innerHTML = evidence
        .map((item) => {
          const tone = LABEL_TONE[item.label] || "unknown";
          const t = fmtTime(item.time);
          const src = SOURCE_NAME[item.source] || item.source;
          const eng = ENGINE_NAME[item.engine] || "";
          return `<article class="card ${tone}">
            <div class="card-top">
              <strong>${item.label}</strong>
              <div class="meta-pills">
                <span class="tag ${item.source}">${src}</span>
                ${eng ? `<span class="tag">${eng}</span>` : ""}
                ${t ? `<button type="button" class="tag" data-seek="${item.time}">${t}</button>` : ""}
              </div>
            </div>
            <p class="quote">${highlight(item.evidence, item.matched)}</p>
            ${item.rule_basis ? `<p class="basis">${item.rule_basis}</p>` : ""}
            ${thumbFor(item, problemFrames)}
          </article>`;
        })
        .join("");
    }

    const segs = (data.asr && data.asr.segments) || [];
    const audioText = (data.asr && data.asr.text) || "";
    const audioPanel = document.getElementById("panel-audio");
    if (segs.length) {
      audioPanel.innerHTML = segs
        .map(
          (seg) => `<div class="line">
            <button type="button" data-seek="${seg.start}">${fmtTime(seg.start) || "口播"}</button>
            <span>${highlight(seg.text, "")}</span>
          </div>`
        )
        .join("");
    } else {
      audioPanel.innerHTML = audioText
        ? `<p class="quote">${audioText}</p>`
        : `<p class="empty">没有口播转写。</p>`;
    }

    const lines = (data.ocr && data.ocr.merged) || [];
    const visualPanel = document.getElementById("panel-visual");
    if (!lines.length) {
      visualPanel.innerHTML = `<p class="empty">没有可用的画面文字。</p>`;
    } else {
      visualPanel.innerHTML = lines
        .map(
          (line) => `<div class="line">
            <button type="button" data-seek="${line.time ?? ""}">${fmtTime(line.time) || "画面"}</button>
            <span>${line.text}</span>
          </div>`
        )
        .join("");
    }

    const basis = data.rule_basis || [];
    const basisPanel = document.getElementById("panel-basis");
    basisPanel.innerHTML = basis.length
      ? basis.map((b) => `<article class="card"><p class="quote">${b}</p></article>`).join("")
      : `<p class="empty">本条未引用具体条款。</p>`;

    document.querySelectorAll("[data-seek]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const t = btn.getAttribute("data-seek");
        if (t !== "" && t != null) seek(t);
      });
    });
  };

  const readSSE = async (response, onEvent) => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const chunks = buf.split("\n\n");
      buf = chunks.pop() || "";
      for (const chunk of chunks) {
        const line = chunk.split("\n").find((l) => l.startsWith("data:"));
        if (!line) continue;
        const raw = line.replace(/^data:\s?/, "");
        onEvent(JSON.parse(raw));
      }
    }
  };

  const analyze = async () => {
    if (!state.job || state.busy) return;
    showError("");
    els.result.hidden = true;
    setBusy(true, "审核进行中…");
    els.steps.hidden = false;
    els.live.hidden = false;
    setStep("start");
    els.live.textContent = "开始审核";
    try {
      const res = await fetch(`/api/jobs/${state.job.job_id}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: state.mode, force: els.force.checked }),
      });
      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "无法开始审核");
      }
      await readSSE(res, (event) => {
        if (event.message) els.live.textContent = event.message;
        if (event.stage) setStep(event.stage);
        if (event.stage === "error") throw new Error(event.message || "审核失败");
        if (event.stage === "done") renderResult(event.result);
      });
    } catch (err) {
      showError(err.message);
    } finally {
      setBusy(false);
      els.scan.hidden = true;
      syncRun();
    }
  };

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-on", t === tab));
      ["evidence", "audio", "visual", "basis"].forEach((id) => {
        document.getElementById(`panel-${id}`).hidden = id !== tab.dataset.tab;
      });
    });
  });

  els.pick.addEventListener("click", openPicker);
  els.replace.addEventListener("click", openPicker);
  if (els.next) {
    els.next.addEventListener("click", () => {
      if (state.busy) return;
      clearVideo();
    });
  }
  els.player.addEventListener("loadedmetadata", () => {
    if (els.player.duration && Number.isFinite(els.player.duration)) {
      els.dur.textContent = fmtDur(els.player.duration);
    }
  });
  els.file.addEventListener("change", () => onFile(els.file.files[0]));
  ["dragenter", "dragover"].forEach((ev) => {
    els.device.addEventListener(ev, (e) => {
      e.preventDefault();
      els.device.classList.add("is-over");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    els.device.addEventListener(ev, (e) => {
      e.preventDefault();
      els.device.classList.remove("is-over");
    });
  });
  els.device.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) onFile(file);
  });
  els.openLib.addEventListener("click", () => {
    els.drawer.hidden = false;
    els.libSearch.focus();
  });
  els.closeLib.addEventListener("click", () => {
    els.drawer.hidden = true;
  });
  els.drawer.addEventListener("click", (e) => {
    if (e.target === els.drawer) els.drawer.hidden = true;
  });
  els.libSearch.addEventListener("input", () => renderLibrary(state.library));
  els.run.addEventListener("click", analyze);

  fetch("/api/meta")
    .then((r) => r.json())
    .then((meta) => {
      els.libPill.textContent = `测试集 ${meta.library_count} 条`;
      if (meta.ollama && meta.ollama.ready) {
        els.llmPill.textContent = `Ollama · ${meta.llm_model}`;
        els.llmPill.classList.add("ok");
      } else {
        els.llmPill.textContent = "Ollama 未就绪";
        els.llmPill.classList.add("warn");
      }
      renderModes(meta.modes, meta.ollama);
      syncRun();
    })
    .catch(() => {
      els.llmPill.textContent = "无法连接后端";
      els.llmPill.classList.add("warn");
    });

  fetch("/api/library")
    .then((r) => r.json())
    .then((data) => {
      state.library = data.items || [];
      renderLibrary(state.library);
    });
})();
