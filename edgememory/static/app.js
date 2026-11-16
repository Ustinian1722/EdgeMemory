(() => {
  const state = { media: [], jobs: [], timeline: [], currentMediaId: "", eventById: new Map(), busy: false };
  const $ = (selector) => document.querySelector(selector);
  const escText = (value) => String(value ?? "");
  let toastTimer;

  function toast(message) {
    const node = $("#toast");
    node.textContent = message;
    node.classList.add("visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => node.classList.remove("visible"), 3200);
  }

  async function request(path, options) {
    const response = await fetch(path, { cache: "no-store", ...options });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `请求失败 (${response.status})`);
    return payload;
  }

  function time(ms) {
    const seconds = Math.max(0, Math.floor((Number(ms) || 0) / 1000));
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const rest = seconds % 60;
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
  }

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = escText(text);
    return element;
  }

  function statusLabel(status) {
    return ({ queued: "排队中", running: "处理中", completed: "已完成", partial: "部分完成", failed: "失败" })[status] || status || "未知";
  }

  function eventRecord(hit) {
    state.eventById.set(hit.event_id, hit);
  }

  function playAt(mediaId, timestampMs, label, caption) {
    if (!mediaId) return;
    const player = $("#sourcePlayer");
    const target = `/api/video/${encodeURIComponent(mediaId)}`;
    if (player.dataset.mediaId !== mediaId) {
      player.src = target;
      player.dataset.mediaId = mediaId;
      player.load();
    }
    const seek = () => {
      const end = Number.isFinite(player.duration) ? Math.max(0, player.duration - 0.08) : 0;
      player.currentTime = Math.min(Math.max(0, (Number(timestampMs) || 0) / 1000), end);
      player.play().catch(() => {});
    };
    if (player.readyState >= 1 && player.dataset.mediaId === mediaId) seek();
    else player.addEventListener("loadedmetadata", seek, { once: true });
    $("#playbackLabel").textContent = label || "视频证据";
    $("#playerCaption").textContent = caption || "原始视频将定位至所选事件的相对时间。";
  }

  function renderJobs() {
    const list = $("#jobList");
    list.replaceChildren();
    if (!state.jobs.length) {
      list.append(node("div", "empty-state", "当前没有处理任务"));
      return;
    }
    for (const job of state.jobs) {
      const card = node("div", "job-item");
      const top = node("div", "job-top");
      const copy = node("div");
      copy.append(node("div", "job-title", job.title));
      const detail = job.status === "running"
        ? `${job.stage} · 已发现 ${job.total_units || 0} 个候选帧 · 仍在处理`
        : job.total_units > 0
          ? `${job.stage} · ${job.completed_units}/${job.total_units}`
          : `${job.stage} · 进度采集中`;
      copy.append(node("div", "job-stage", detail));
      top.append(copy, node("span", `job-state ${job.status}`, statusLabel(job.status)));
      const track = node("div", "job-progress");
      if (job.status === "running") track.style.display = "none";
      const fill = node("i");
      fill.style.width = `${Math.max(0, Math.min(100, Number(job.progress) || 0))}%`;
      track.append(fill);
      card.append(top, track);
      if (job.error_code) card.append(node("div", "job-stage", `失败记录：${job.error_code}`));
      list.append(card);
    }
  }

  function sourceOption(select, value, label) {
    const option = node("option", "", label);
    option.value = value;
    select.append(option);
  }

  function renderMedia() {
    const list = $("#mediaList");
    list.replaceChildren();
    const selectors = [$("#sourceSelect"), $("#searchSource"), $("#answerSource")];
    const keep = selectors.map((select) => select.value);
    selectors.forEach((select, index) => {
      select.replaceChildren();
      sourceOption(select, "", index === 0 ? "选择视频源" : index === 1 ? "所有视频" : "跨视频记忆");
    });
    if (!state.media.length) {
      list.append(node("div", "empty-state", "上传视频后会显示来源和索引状态"));
      return;
    }
    for (const media of state.media) {
      const label = `${media.title} · ${media.event_count || 0} 个事件`;
      selectors.forEach((select) => sourceOption(select, media.media_id, label));
      const card = node("button", `media-item${media.media_id === state.currentMediaId ? " selected" : ""}`);
      card.type = "button";
      const top = node("div", "media-top");
      const title = node("div", "media-title-row");
      title.append(node("span", "media-thumb", "▶"));
      const text = node("div");
      text.append(node("div", "media-title", media.title));
      text.append(node("div", "media-meta", `${time(media.duration_ms)} · ${media.event_count || 0} 个事件 · ${media.failed_count || 0} 项失败`));
      title.append(text);
      top.append(title, node("span", "media-count", "打开"));
      card.append(top);
      card.addEventListener("click", () => selectMedia(media.media_id));
      list.append(card);
    }
    selectors.forEach((select, index) => {
      const previous = keep[index];
      if (previous && state.media.some((media) => media.media_id === previous)) select.value = previous;
    });
    $("#summaryButton").disabled = !$("#answerSource").value && !state.currentMediaId;
  }

  function eventCard(hit, compact = false) {
    eventRecord(hit);
    const card = node("article", compact ? "search-result" : "event-card");
    const image = node("img", compact ? "" : "event-image");
    image.src = `/api/frame/${encodeURIComponent(hit.event_id)}`;
    image.alt = "视频关键帧证据";
    image.loading = "lazy";
    image.addEventListener("click", () => playAt(hit.media_id, hit.timestamp_ms, `${hit.media_title} · ${time(hit.timestamp_ms)}`, hit.caption));
    const copy = node("div", compact ? "" : "event-copy");
    const text = node("p", "", hit.caption || (hit.caption_status === "failed"
      ? `该事件处理失败：${hit.error_code || "unknown"}`
      : hit.caption_status === "visual_only" ? "视觉候选帧，尚无 VLM 语义描述。查询命中后可按需复审。" : "事件描述待生成"));
    const meta = node("div", compact ? "event-meta" : "event-meta");
    meta.append(node("span", "", `${hit.media_title} · ${time(hit.timestamp_ms)}`));
    if (hit.caption_status === "visual_only") meta.append(node("span", "memory-channel visual", "视觉候选 · 按需复审"));
    if (hit.backend_id) meta.append(node("span", "", `后端 ${hit.backend_id}`));
    if (hit.model_id) meta.append(node("span", "", hit.model_id));
    if (hit.simulation_only) meta.append(node("span", "job-state", "模拟数据"));
    if (hit.similarity !== undefined && hit.similarity !== null) meta.append(node("span", "", `相似度 ${Number(hit.similarity).toFixed(3)}`));
    const channelRanks = hit.retrieval_channels || {};
    if (channelRanks.visual) meta.append(node("span", "memory-channel visual", `图文向量 #${channelRanks.visual.rank}`));
    if (channelRanks.text) meta.append(node("span", "memory-channel text", `文本向量 #${channelRanks.text.rank}`));
    const action = node("button", "text-button", "定位回放");
    action.type = "button";
    action.addEventListener("click", () => playAt(hit.media_id, hit.timestamp_ms, `${hit.media_title} · ${time(hit.timestamp_ms)}`, hit.caption));
    const review = node("button", "text-button", "视觉复审");
    review.type = "button";
    review.addEventListener("click", async () => {
      const question = window.prompt("想在这一帧中查看什么？", "描述画面中可见的细节");
      if (question === null) return;
      review.disabled = true;
      const result = node("p", "review-result", "正在复审当前帧…");
      copy.append(result);
      try {
        const response = await request("/api/review", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({event_id: hit.event_id, query: question})});
        result.textContent = "单帧复审：" + (response.review || "没有生成描述");
      } catch (error) { result.textContent = "复审失败：" + error.message; }
      finally { review.disabled = false; }
    });
    copy.append(text, meta, action, review);
    if (hit.caption_status === "failed") {
      const retry = node("button", "citation-button", "重新理解该帧");
      retry.type = "button";
      retry.addEventListener("click", async () => {
        retry.disabled = true;
        try {
          await request("/api/retry-event", {method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({event_id: hit.event_id})});
          toast("重试完成；原失败记录已保留");
          await refresh();
        } catch (error) {
          toast("重试失败：" + error.message);
        } finally { retry.disabled = false; }
      });
      copy.append(retry);
    }
    card.append(image, copy);
    return card;
  }

  function renderTimeline() {
    const rail = $("#timelineRail");
    const list = $("#timelineEvents");
    rail.replaceChildren();
    list.replaceChildren();
    const media = state.media.find((item) => item.media_id === state.currentMediaId);
    if (!media) {
      rail.append(node("div", "timeline-empty", "选择一个视频查看事件"));
      list.append(node("div", "empty-state", "暂无事件"));
      $("#timelineLabel").textContent = "选择一个视频查看事件";
      $("#timelineCount").textContent = "";
      return;
    }
    $("#timelineLabel").textContent = media.title;
    $("#timelineCount").textContent = `${state.timeline.length} 个时间点`;
    if (!state.timeline.length) {
      rail.append(node("div", "timeline-empty", "尚无可展示的事件"));
      list.append(node("div", "empty-state", "该视频尚未产生事件记忆"));
      return;
    }
    const knownDuration = Number(media.duration_ms) || 0;
    const lastObserved = state.timeline.reduce((latest, event) => Math.max(latest, Number(event.timestamp_ms) || 0), 0);
    const duration = Math.max(1, knownDuration, lastObserved);
    if (knownDuration <= 0) {
      $("#timelineCount").textContent += " · 按当前已观测范围显示";
    }
    for (const event of state.timeline) {
      eventRecord(event);
      const marker = node("button", `timeline-marker${event.caption_status === "failed" ? " failed" : ""}`);
      marker.type = "button";
      marker.title = `${time(event.timestamp_ms)} · ${event.caption || event.error_code || event.caption_status}`;
      marker.style.left = `${Math.min(98, Math.max(2, 100 * event.timestamp_ms / duration))}%`;
      marker.addEventListener("click", () => playAt(event.media_id, event.timestamp_ms, `${event.media_title} · ${time(event.timestamp_ms)}`, event.caption));
      rail.append(marker);
      list.append(eventCard(event));
    }
  }

  async function selectMedia(mediaId) {
    state.currentMediaId = mediaId;
    $("#sourceSelect").value = mediaId;
    $("#answerSource").value = mediaId;
    $("#summaryButton").disabled = false;
    renderMedia();
    try {
      const payload = await request(`/api/timeline?media_id=${encodeURIComponent(mediaId)}`);
      state.timeline = payload.events || [];
      renderTimeline();
    } catch (error) {
      toast(error.message);
    }
  }

  function renderEvidenceResults(target, results) {
    target.replaceChildren();
    if (!results.length) {
      target.append(node("div", "empty-state", "没有找到匹配事件。没有事件不代表视频中没有发生该事。"));
      return;
    }
    for (const hit of results) target.append(eventCard(hit, true));
  }

  function renderAnswer(payload) {
    const target = $("#answerResult");
    target.replaceChildren();
    for (const hit of payload.evidence || []) eventRecord(hit);
    const answerStatus = {
      answered: "已回答",
      partially_answered: "部分回答；请查看下方限制和未覆盖内容",
      insufficient_evidence: "证据不足",
    };
    target.append(node("div", "answer-status", `回答状态：${answerStatus[payload.status] || payload.status || "未知"}${payload.model_id ? ` · 模型 ${payload.model_id}` : ""}`));
    const inference = payload.inference_metrics || {};
    const inferenceFacts = [];
    if (Number.isFinite(payload.request_elapsed_us)) inferenceFacts.push(`API 查询处理 ${Math.round(payload.request_elapsed_us / 1000)} ms`);
    if (Number.isFinite(inference.ttft_us)) inferenceFacts.push(`设备 TTFT ${Math.round(inference.ttft_us / 1000)} ms`);
    if (Number.isFinite(inference.decode_tokens_per_second)) inferenceFacts.push(`Decode ${inference.decode_tokens_per_second.toFixed(1)} tok/s`);
    if (Number.isFinite(inference.scheduler_queue_wait_us)) inferenceFacts.push(`调度排队 ${Math.round(inference.scheduler_queue_wait_us / 1000)} ms`);
    if (Number.isFinite(inference.frame_transfer_us)) inferenceFacts.push(`证据帧传输 ${Math.round(inference.frame_transfer_us / 1000)} ms`);
    if (inferenceFacts.length) {
      target.append(node("div", "evidence-note", `本次端侧回答 · ${inferenceFacts.join(" · ")}`));
    }
    const scope = payload.evidence_scope || {};
    if ((payload.prompt_budget || {}).batch_count > 1) {
      target.append(node("div", "evidence-note", `分 ${payload.prompt_budget.batch_count} 批处理检索证据；以下结论分别引用原始事件。`));
    }
    if (scope.kind === "time_spaced_video_summary") {
      target.append(node("div", "evidence-note",
        `按时间段抽样 ${scope.sampled_event_count || 0} / ${scope.available_event_count || 0} 个非模拟已完成事件；每条结论可定位到原视频时间点。`));
    }
    for (const claim of payload.claims || []) {
      const block = node("div", `claim ${claim.kind}`);
      block.append(node("div", "claim-label", claim.kind === "observed" ? "Observed · 直接观测" : "Inference · 模型推断"));
      block.append(node("p", "", claim.text));
      const citations = node("div");
      for (const eventId of claim.event_ids || []) {
        const hit = state.eventById.get(eventId);
        const button = node("button", "citation-button", hit ? `${hit.media_title} · ${time(hit.timestamp_ms)}` : eventId.slice(0, 8));
        button.type = "button";
        button.addEventListener("click", () => hit && playAt(hit.media_id, hit.timestamp_ms, `${hit.media_title} · ${time(hit.timestamp_ms)}`, hit.caption));
        citations.append(button);
      }
      block.append(citations);
      target.append(block);
    }
    const missing = (payload.missing_information || []).filter(Boolean);
    if (missing.length) {
      const list = node("ul", "missing-list");
      for (const item of missing) list.append(node("li", "", item));
      target.append(list);
    }
    const evidence = payload.evidence || [];
    if (evidence.length) {
      target.append(node("div", "claim-label", "检索到的证据"));
      for (const hit of evidence) target.append(eventCard(hit, true));
    }
  }

  async function refresh() {
    if (state.busy) return;
    state.busy = true;
    try {
      const [status, jobs, media, runtime] = await Promise.all([
        request("/api/status"), request("/api/jobs"), request("/api/media"), request("/api/runtime"),
      ]);
      state.jobs = jobs.jobs || [];
      state.media = media.media || [];
      $("#metricMedia").textContent = status.counts.media_count ?? "—";
      $("#metricEvents").textContent = status.counts.completed_event_count ?? "—";
      $("#metricEventDetail").textContent = `${status.counts.event_count ?? 0} 个事件记录 · ${status.counts.failed_event_count ?? 0} 个失败`;
      $("#metricActive").textContent = status.counts.active_job_count ?? "—";
      $("#metricFailed").textContent = status.counts.failed_job_count ?? "—";
      $("#queueCapacity").textContent = `${runtime.queue.available_slots} 个队列槽可用`;
      $("#connectionDot").className = "status-dot online";
      $("#connectionLabel").textContent = "服务已连接";
      $("#connectionDetail").textContent = status.caption_backend_configured ? "视频理解后端已配置" : "视频理解后端未配置";
      renderJobs();
      renderMedia();
      renderRuntime(runtime);
      if (state.currentMediaId && state.media.some((item) => item.media_id === state.currentMediaId)) {
        const timeline = await request(`/api/timeline?media_id=${encodeURIComponent(state.currentMediaId)}`);
        state.timeline = timeline.events || [];
        renderTimeline();
      } else if (!state.currentMediaId && state.media.length) {
        const first = state.media[0];
        await selectMedia(first.media_id);
      }
    } catch (error) {
      $("#connectionDot").className = "status-dot offline";
      $("#connectionLabel").textContent = "服务暂不可用";
      $("#connectionDetail").textContent = error.message;
    } finally {
      state.busy = false;
    }
  }

  function renderRuntime(runtime) {
    const queue = runtime.queue || {};
    const worker = runtime.worker || {};
    $("#workerState").textContent = `${queue.running_jobs || 0} 运行 · ${queue.queued_jobs || 0} 排队`;
    const scheduler = runtime.scheduler || {};
    const policySelect = $("#schedulerPolicy");
    if (policySelect && document.activeElement !== policySelect && scheduler.policy) {
      policySelect.value = scheduler.policy;
    }
    const schedulerStatus = $("#schedulerPolicyStatus");
    const direct = scheduler.engine === "direct-execution";
    if (policySelect) policySelect.disabled = direct;
    const policyButton = $("#schedulerApply");
    if (policyButton) policyButton.disabled = direct;
    if (schedulerStatus && scheduler.engine) {
      schedulerStatus.textContent = direct ? "直接执行 · 策略切换不可用" : `${scheduler.engine} · ${scheduler.policy || "unknown"}`;
    }
    const resourceCount = Array.isArray(scheduler.resources) ? scheduler.resources.length : 0;
    const scheduling = scheduler.engine
      ? ` · ${scheduler.engine} / ${scheduler.policy} / ${resourceCount} 个配置资源` : "";
    $("#workerDetail").textContent = `${queue.workers_configured || 0} 个配置 worker · ${queue.active_futures || 0} 个活动 future${scheduling}`;
    $("#modelState").textContent = [runtime.caption_model, runtime.embedder_model, runtime.answer_model].filter(Boolean).join(" · ") || "未配置模型";
    const devices = runtime.runtime_devices || [];
    const modelBytes = devices.filter((device) => Number.isFinite(device.estimated_resident_bytes));
    const modelSizes = modelBytes.map((device) => `${device.device_id}: ${formatBytes(device.estimated_resident_bytes)}`);
    if (!modelSizes.length && Number.isFinite(worker.estimated_resident_bytes)) {
      modelSizes.push(formatBytes(worker.estimated_resident_bytes));
    }
    const backendStates = (runtime.runtime_devices || []).map((device) =>
      `${device.device_id}:${device.process_state || "unknown"}/load${device.model_load_count ?? "?"}`);
    for (const backend of runtime.llm_backends || []) {
      backendStates.push(`${backend.backend_id}:${backend.process_state || "unknown"}/load${backend.model_load_count ?? "?"}`);
    }
    if (runtime.embedding_runtime) {
      backendStates.push(`embedding:${runtime.embedding_runtime.process_state || "unknown"}/load${runtime.embedding_runtime.model_load_count ?? "?"}`);
    }
    const modelSizeDetail = modelSizes.length
      ? `模型文件大小估算 ${modelSizes.join(" · ")}；不是设备内存实测` :
        "设备内存使用未由当前后端报告";
    $("#modelDetail").textContent = `${backendStates.length ? `驻留进程 ${backendStates.join(" · ")}；` : ""}${modelSizeDetail}`;
    const latest = runtime.latest_inference || {};
    const latency = Number.isFinite(latest.ttft_us)
      ? `${Math.round(latest.ttft_us / 1000)} ms TTFT`
      : Number.isFinite(latest.prefill_us) ? `${Math.round(latest.prefill_us / 1000)} ms prefill` : "无近期延迟指标";
    const decode = Number.isFinite(latest.decode_tokens_per_second) ? ` · ${latest.decode_tokens_per_second.toFixed(1)} tok/s` : "";
    $("#metricState").textContent = `${latency}${decode}`;
    $("#metricDetail").textContent = latest.backend_id ? `最近后端：${latest.backend_id}` : "等待真实推理记录";
    $("#topologyState").textContent = devices.length ? devices.map((d) => d.device_id).join(" · ") : "尚未报告";
    $("#topologyDetail").textContent = devices.length ? "设备 ID 是 GenieX 服务的显式选择符；物理拓扑与资源重叠仍需实测确认" : "未发现的物理映射不会用示意数据替代";
    const visual = runtime.visual_embedding_runtime || {};
    $("#visualMemoryState").textContent = visual.status === "resident"
      ? `${visual.model_id || runtime.visual_embedding_model || "SigLIP2"} · ${visual.output_dimensions || "?"} 维` :
        visual.status === "not_requested" ? "未配置" : `不可用 · ${visual.status || "等待目标依赖"}`;
    const visualCalls = visual.calls || {};
    $("#visualMemoryDetail").textContent = visual.status === "resident"
      ? `${runtime.visual_embedding_count || 0} 个持久图像向量 · 图像 ${visualCalls.image || 0} 次 / 文本 ${visualCalls.text || 0} 次 · ${visual.image_preprocess_path || "预处理路径未知"} · 向量空间隔离`
      : (visual.reason || "SigLIP2/QIM 需通过目标版本与插件检查后启用");
    const hardware = runtime.hardware || {};
    const cpu = hardware.cpu_busy || {};
    const availableMemory = hardware.memory?.MemAvailable || {};
    const gpuBusy = hardware.gpu_sysfs?.busy_percent || {};
    const videoPath = runtime.video_hardware_path || {};
    const decoderNames = Array.isArray(videoPath.decoder_factories)
      ? videoPath.decoder_factories.join(",") : "";
    const decoderOutput = videoPath.decoder_output || {};
    const decoderMemory = decoderOutput.dmabuf_detected === true ? "decoder DMABUF observed"
      : decoderOutput.dmabuf_detected === false ? "decoder DMABUF not observed"
        : "decoder memory type unavailable";
    const preprocess = videoPath.preprocess_element
      ? `${videoPath.preprocess_element}${videoPath.preprocess_engine ? `/${videoPath.preprocess_engine}` : ""}`
      : "video preprocessor unavailable";
    const hardwareFacts = [];
    hardwareFacts.push(cpu.status === "available" ? `CPU ${cpu.value}%` : "CPU unavailable");
    hardwareFacts.push(availableMemory.status === "available" ? `MemAvailable ${formatBytes(availableMemory.value)}` : "内存 unavailable");
    hardwareFacts.push(gpuBusy.status === "available" ? `GPU ${gpuBusy.value}%` : "GPU busy counter unavailable");
    $("#hardwareState").textContent = hardwareFacts.join(" · ");
    const fastRpcNodes = Array.isArray(hardware.fastrpc_nodes) ? hardware.fastrpc_nodes.length : 0;
    const videoDetail = decoderNames
      ? `最近视频：${decoderNames} / ${videoPath.decoder_io_mode || "I/O 未知"} / ${preprocess} · ${decoderMemory} · appsink 输出进入 JPEG 编码`
      : "最近视频硬件路径尚未采样";
    $("#hardwareDetail").textContent = `来源 /proc、sysfs；FastRPC 节点 ${fastRpcNodes} 个 · 未暴露的加速器计数保持 unavailable · ${videoDetail}`;
    $("#runtimeUpdated").textContent = `采集于 ${new Date(runtime.collected_at).toLocaleTimeString()}`;
    const tbody = $("#runtimeRows");
    tbody.replaceChildren();
    const rows = runtime.recent_inferences || [];
    if (!rows.length) {
      const tr = node("tr");
      const td = node("td", "empty-state", "暂无实际推理记录");
      td.colSpan = 8;
      tr.append(td);
      tbody.append(tr);
      return;
    }
    for (const item of rows) {
      const tr = node("tr");
      for (const value of [
        item.media_title, item.stage || "inference", item.backend_id || "—", item.model_id || "—",
        item.caption_status, Number.isFinite(item.scheduler_queue_wait_us) ? `${Math.round(item.scheduler_queue_wait_us / 1000)} ms`
          : "—", Number.isFinite(item.ttft_us) ? `${Math.round(item.ttft_us / 1000)} ms TTFT`
          : Number.isFinite(item.prefill_us) ? `${Math.round(item.prefill_us / 1000)} ms prefill` : "—",
        Number.isFinite(item.decode_tokens_per_second) ? `${item.decode_tokens_per_second.toFixed(1)} tok/s` : "—",
      ]) tr.append(node("td", "", value));
      tbody.append(tr);
    }
  }

  $("#schedulerApply").addEventListener("click", async () => {
    const button = $("#schedulerApply");
    button.disabled = true;
    try {
      const result = await request("/api/runtime/policy", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ policy: $("#schedulerPolicy").value }),
      });
      $("#schedulerPolicyStatus").textContent = `已切换为 ${result.policy}`;
      toast(`HexagonFlow 已应用 ${result.policy}`);
      await refresh();
    } catch (error) {
      $("#schedulerPolicyStatus").textContent = error.message;
      toast(error.message);
    } finally {
      button.disabled = false;
    }
  });

  function formatBytes(bytes) {
    const value = Number(bytes);
    if (!Number.isFinite(value) || value < 0) return "未知";
    const units = ["B", "KiB", "MiB", "GiB"];
    let size = value;
    let index = 0;
    while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; }
    return `${size.toFixed(index ? 1 : 0)} ${units[index]}`;
  }

  function timeRangePayload(prefix) {
    const result = {};
    for (const [field, selector] of [["start_s", `${prefix}Start`], ["end_s", `${prefix}End`]]) {
      const raw = $(`#${selector}`).value;
      if (raw === "") continue;
      const value = Number(raw);
      if (!Number.isFinite(value) || value < 0) throw new Error("时间范围必须是非负秒数");
      result[field] = value;
    }
    if (result.start_s !== undefined && result.end_s !== undefined && result.start_s > result.end_s) {
      throw new Error("起始秒不能大于结束秒");
    }
    return result;
  }

  function uploadOne(file) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/videos");
      xhr.setRequestHeader("Content-Type", file.type || "application/octet-stream");
      xhr.setRequestHeader("X-File-Name", encodeURIComponent(file.name));
      xhr.upload.onprogress = (event) => {
        if (!event.lengthComputable) return;
        const percent = Math.round(100 * event.loaded / event.total);
        $("#uploadProgressBar").style.width = `${percent}%`;
        $("#uploadProgressLabel").textContent = `上传 ${file.name} · ${percent}%`;
      };
      xhr.onload = () => {
        let payload = {};
        try { payload = JSON.parse(xhr.responseText); } catch {}
        if (xhr.status >= 200 && xhr.status < 300) resolve(payload);
        else reject(new Error(payload.error || `上传失败 (${xhr.status})`));
      };
      xhr.onerror = () => reject(new Error("视频上传连接中断"));
      xhr.send(file);
    });
  }

  $("#videoFiles").addEventListener("change", () => {
    const files = Array.from($("#videoFiles").files || []);
    $("#uploadButton").disabled = !files.length;
    $("#uploadCount").textContent = files.length ? `${files.length} 个文件` : "支持多选";
    $("#uploadMessage").textContent = files.map((file) => `${file.name} (${formatBytes(file.size)})`).join(" · ") || "支持 MP4、MOV、MKV、WebM、AVI";
  });
  $("#uploadButton").addEventListener("click", async () => {
    const files = Array.from($("#videoFiles").files || []);
    if (!files.length) return;
    $("#uploadButton").disabled = true;
    $("#uploadProgress").classList.remove("hidden");
    for (const file of files) {
      try {
        await uploadOne(file);
        toast(`已加入队列：${file.name}`);
      } catch (error) {
        toast(`${file.name}：${error.message}`);
      }
    }
    $("#videoFiles").value = "";
    $("#uploadButton").disabled = true;
    $("#uploadProgress").classList.add("hidden");
    $("#uploadMessage").textContent = "上传结束；索引将在后台继续运行";
    await refresh();
  });

  const dropzone = $("#dropzone");
  ["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault(); dropzone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault(); dropzone.classList.remove("dragging");
    if (eventName === "drop" && event.dataTransfer?.files?.length) {
      $("#videoFiles").files = event.dataTransfer.files;
      $("#videoFiles").dispatchEvent(new Event("change"));
    }
  }));

  $("#sourceSelect").addEventListener("change", () => {
    if ($("#sourceSelect").value) selectMedia($("#sourceSelect").value);
  });
  $("#imageSearchForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const target = $("#searchResults");
    const file = $("#imageSearchFile").files[0];
    if (!file) return;
    target.replaceChildren(node("div", "empty-state", "正在搜索相似画面…"));
    try {
      const response = await fetch("/api/search/image", {method: "POST", body: file});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "搜索失败");
      renderEvidenceResults(target, payload.results || []);
    } catch (error) {
      target.replaceChildren(node("div", "empty-state", "图片搜索失败：" + error.message));
    }
  });
  $("#searchForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const target = $("#searchResults");
    target.replaceChildren(node("div", "empty-state", "正在检索事件证据…"));
    try {
      const payload = await request("/api/query", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: $("#searchQuery").value, media_id: $("#searchSource").value || null, limit: 12, ...timeRangePayload("search") }),
      });
      renderEvidenceResults(target, payload.results || []);
    } catch (error) { target.replaceChildren(node("div", "empty-state", error.message)); }
  });

  async function ask(summary = false) {
    const sourceId = $("#answerSource").value || state.currentMediaId;
    const question = summary ? "Summarize the observed video events and state which details are not supported by the indexed evidence." : $("#answerQuestion").value.trim();
    if (!question) { toast("先输入一个问题"); return; }
    const target = $("#answerResult");
    target.replaceChildren(node("div", "answer-status", "正在检索证据并请求配置的回答模型…"));
    try {
      const payload = await request("/api/answer", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question, media_id: sourceId || null, limit: 2, summary, ...timeRangePayload("answer") }),
      });
      renderAnswer(payload);
    } catch (error) { target.replaceChildren(node("div", "empty-state", error.message)); }
  }
  $("#answerForm").addEventListener("submit", (event) => { event.preventDefault(); ask(false); });
  $("#summaryButton").addEventListener("click", () => ask(true));
  $("#clearSourceFilter").addEventListener("click", () => {
    state.currentMediaId = "";
    $("#sourceSelect").value = "";
    $("#answerSource").value = "";
    renderTimeline();
    renderMedia();
  });
  $("#refreshButton").addEventListener("click", refresh);
  refresh();
  setInterval(refresh, 1800);
})();
