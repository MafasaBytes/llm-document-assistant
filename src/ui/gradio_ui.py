"""
HealthDoc AI — Prometheus Surgical HUD Interface
dark + cyan medical document analysis console.
Chat-style interaction, typewriter streaming, live telemetry sidebar.
"""

import gradio as gr
import logging
import tempfile
import time
import yaml
from datetime import datetime
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from retriever.QA_chain import retriever_qa_with_metadata

logger = logging.getLogger(__name__)

# Stream tuning
_STREAM_CHUNK = 3     # words per yield
_STREAM_DELAY = 0.02  # seconds between yields

# SYSTEM LEVEL DETECTION

def _get_system_info():
    """Detect runtime hardware and read model configuration."""
    info = {}

    # GPU / Device detection
    try:
        import torch
        info["device"] = "CUDA" if torch.cuda.is_available() else "CPU"
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            # total_memory (PyTorch >=2.0) with fallback to total_mem
            total_bytes = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            info["vram"] = f"{total_bytes / (1024 ** 3):.1f} GB"
        else:
            info["gpu_name"] = "N/A"
            info["vram"] = "N/A"
    except (ImportError, Exception):
        info["device"] = "CPU"
        info["gpu_name"] = "N/A"
        info["vram"] = "N/A"

    # Read model config 
    config_path = Path(__file__).resolve().parents[2] / "config" / "model.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        llm = config.get("llm_config", {})
        emb = config.get("embedding_model", {})
        info["llm_model"] = llm.get("model", "Unknown")
        info["llm_url"] = llm.get("base_url", "Unknown")
        info["temperature"] = llm.get("temperature", "N/A")
        info["num_ctx"] = llm.get("num_ctx", "N/A")
        info["embed_model"] = emb.get("model", "Unknown")
    except Exception:
        info.update({
            "llm_model": "Unknown", "llm_url": "Unknown",
            "temperature": "N/A", "num_ctx": "N/A", "embed_model": "Unknown",
        })

    return info


# HTML / CSS BUILDERS

def _short_name(name):
    """Shorten 'org/model-name' to 'model-name'."""
    return name.split("/")[-1] if "/" in str(name) else str(name)


def _metric_row(label, value, color=None):
    """Single key-value row for the telemetry panel."""
    val_color = color if color else "var(--hd-text)"
    return (
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center;margin-bottom:8px;">'
        f'<span style="color:var(--hd-text2);font-size:10px;letter-spacing:1.5px;">'
        f'{label}</span>'
        f'<span style="color:{val_color};font-size:11px;font-weight:500;">'
        f'{value}</span></div>'
    )


def _build_metrics_html(sys_info, query_metrics=None, status="STANDBY"):
    """Build the full telemetry sidebar as self-contained HTML."""

    palette = {
        "STANDBY":    ("#f0b429", ""),
        "PROCESSING": ("#00d4ff", "animation:hd-pulse 1.5s infinite;"),
        "COMPLETE":   ("#00d4ff", ""),
        "ERROR":      ("#ff4757", ""),
    }
    color, anim = palette.get(status, ("#f0b429", ""))

    # query metrics block
    if query_metrics:
        qm = query_metrics
        q_block = f"""
        <div style="margin-top:22px;">
          <div style="color:var(--hd-text2);font-size:9px;letter-spacing:3px;
                      margin-bottom:10px;">QUERY ANALYSIS</div>
          <div style="height:1px;background:linear-gradient(90deg,
                      var(--hd-border-h),transparent);margin-bottom:12px;"></div>
          {_metric_row("LATENCY",   f'{qm.get("total_latency","--")}s', "var(--hd-cyan)")}
          {_metric_row("PAGES",     str(qm.get("num_pages","--")))}
          {_metric_row("CHUNKS",    str(qm.get("num_chunks","--")))}
          {_metric_row("SOURCES",   str(qm.get("num_sources","--")))}
          <div style="height:6px;"></div>
          <div style="color:var(--hd-text2);font-size:9px;letter-spacing:3px;
                      margin-bottom:10px;">PIPELINE BREAKDOWN</div>
          <div style="height:1px;background:linear-gradient(90deg,
                      var(--hd-border-h),transparent);margin-bottom:12px;"></div>
          {_metric_row("LOAD",     f'{qm.get("load_time","--")}s')}
          {_metric_row("CHUNK",    f'{qm.get("chunk_time","--")}s')}
          {_metric_row("EMBED",    f'{qm.get("embed_time","--")}s')}
          {_metric_row("GENERATE", f'{qm.get("generation_time","--")}s')}
        </div>"""
    else:
        q_block = f"""
        <div style="margin-top:22px;">
          <div style="color:var(--hd-text2);font-size:9px;letter-spacing:3px;
                      margin-bottom:10px;">QUERY ANALYSIS</div>
          <div style="height:1px;background:linear-gradient(90deg,
                      var(--hd-border-h),transparent);margin-bottom:12px;"></div>
          {_metric_row("LATENCY", "--")}
          {_metric_row("CHUNKS",  "--")}
          {_metric_row("SOURCES", "--")}
          <div style="margin-top:14px;color:var(--hd-text3);font-size:10px;
                      font-style:italic;">Awaiting query...</div>
        </div>"""

    return f"""
    <style>
      @keyframes hd-pulse {{
        0%,100% {{ opacity:1; }}
        50%     {{ opacity:0.25; }}
      }}
    </style>
    <div style="
        font-family:'JetBrains Mono','Consolas',monospace;
        padding:18px 16px;
        background:var(--hd-tel-bg);
        border:1px solid var(--hd-border);
        border-radius:8px;
        backdrop-filter:blur(12px);
        box-sizing:border-box;
        width:100%;
    ">
      <!-- header -->
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:22px;">
        <div style="width:7px;height:7px;border-radius:50%;
                    background:{color};box-shadow:0 0 10px {color};{anim}"></div>
        <span style="color:var(--hd-cyan);font-size:10px;letter-spacing:3px;
                     font-weight:600;">SYSTEM TELEMETRY</span>
      </div>

      <!-- infrastructure -->
      <div>
        <div style="color:var(--hd-text2);font-size:9px;letter-spacing:3px;
                    margin-bottom:10px;">INFRASTRUCTURE</div>
        <div style="height:1px;background:linear-gradient(90deg,
                    var(--hd-border-h),transparent);margin-bottom:12px;"></div>
        {_metric_row("STATUS",  f"● {status}", color)}
        {_metric_row("DEVICE",  sys_info.get("device","N/A"))}
        {_metric_row("GPU",     _short_name(sys_info.get("gpu_name","N/A")))}
        {_metric_row("VRAM",    sys_info.get("vram","N/A"))}
      </div>

      <!-- models -->
      <div style="margin-top:22px;">
        <div style="color:var(--hd-text2);font-size:9px;letter-spacing:3px;
                    margin-bottom:10px;">MODELS</div>
        <div style="height:1px;background:linear-gradient(90deg,
                    var(--hd-border-h),transparent);margin-bottom:12px;"></div>
        {_metric_row("LLM",        _short_name(sys_info.get("llm_model","N/A")))}
        {_metric_row("EMBEDDINGS", _short_name(sys_info.get("embed_model","N/A")))}
        {_metric_row("TEMPERATURE",str(sys_info.get("temperature","N/A")))}
        {_metric_row("CTX WINDOW", str(sys_info.get("num_ctx","N/A")))}
      </div>

      {q_block}
    </div>"""


def _get_css():
    """Return the full custom CSS for the Prometheus HUD theme."""
    return """
    /* ═══════════════════════════════════════════════
       HEALTHDOC AI — DUAL-THEME SYSTEM
       Dark = Prometheus HUD  /  Light = Clinical White
       ═══════════════════════════════════════════════ */

    /* ── DARK theme tokens (default) ─────────────── */
    :root, body.hd-dark {
      --hd-bg-deep:     #040810;
      --hd-bg-primary:  #070c14;
      --hd-bg-surface:  #0b1220;
      --hd-bg-elevated: #0f1a2a;
      --hd-bg-input:    #0a1018;
      --hd-border:      rgba(0,212,255,0.10);
      --hd-border-h:    rgba(0,212,255,0.25);
      --hd-border-f:    rgba(0,212,255,0.40);
      --hd-cyan:        #00d4ff;
      --hd-cyan-dim:    #0099bb;
      --hd-glow:        rgba(0,212,255,0.08);
      --hd-text:        #c8d6e5;
      --hd-text2:       #6b8299;
      --hd-text3:       #3d5468;
      --hd-grid:        rgba(0,212,255,0.018);
      --hd-scanline:    rgba(0,0,0,0.012);
      --hd-tel-bg:      rgba(7,12,20,0.92);
      --hd-header-dot:  #00d4ff;
      --hd-header-dot-shadow: 0 0 14px #00d4ff, 0 0 30px rgba(0,212,255,0.3);
      /* Override Gradio's own theme variables */
      --body-background-fill:    #040810 !important;
      --input-background-fill:   #0a1018 !important;
      --body-text-color:         #c8d6e5 !important;
      --body-text-color-subdued: #6b8299 !important;
      --block-background-fill:   transparent !important;
    }

    /* ── LIGHT theme tokens (doubled selector for specificity) ── */
    body.hd-light, html body.hd-light {
      --hd-bg-deep:     #f0f4f8;
      --hd-bg-primary:  #ffffff;
      --hd-bg-surface:  #f7f9fb;
      --hd-bg-elevated: #edf1f5;
      --hd-bg-input:    #ffffff;
      --hd-border:      rgba(0,80,120,0.12);
      --hd-border-h:    rgba(0,80,120,0.25);
      --hd-border-f:    rgba(0,120,180,0.40);
      --hd-cyan:        #0077aa;
      --hd-cyan-dim:    #005580;
      --hd-glow:        rgba(0,120,180,0.06);
      --hd-text:        #1a2a3a;
      --hd-text2:       #4a6070;
      --hd-text3:       #8a98a8;
      --hd-grid:        rgba(0,80,120,0.03);
      --hd-scanline:    transparent;
      --hd-tel-bg:      rgba(247,249,251,0.95);
      --hd-header-dot:  #0077aa;
      --hd-header-dot-shadow: 0 0 8px rgba(0,119,170,0.4);
      /* Override Gradio's own theme variables for light */
      --body-background-fill:    #f0f4f8 !important;
      --input-background-fill:   #ffffff !important;
      --body-text-color:         #1a2a3a !important;
      --body-text-color-subdued: #4a6070 !important;
      --block-background-fill:   transparent !important;
    }

    /* ── theme toggle button ────────────────────── */
    /* The button is a Gradio sibling after the header HTML.
       We float it up into the header with negative margin. */
    #hd-theme-btn {
      display: inline-flex !important;
      align-items: center !important;
      justify-content: center !important;
      min-width: auto !important;
      width: auto !important;
      max-width: 120px !important;
      margin: -48px 16px 12px auto !important;  /* pull up into header */
      padding: 5px 14px !important;
      font-size: 0.62rem !important;
      letter-spacing: 0.18em !important;
      text-transform: uppercase !important;
      font-weight: 600 !important;
      font-family: 'JetBrains Mono','Consolas',monospace !important;
      border-radius: 20px !important;
      cursor: pointer !important;
      transition: all 0.35s ease !important;
      border: 1px solid var(--hd-border-h) !important;
      background: var(--hd-bg-elevated) !important;
      color: var(--hd-cyan) !important;
      box-shadow: 0 0 8px var(--hd-glow) !important;
      position: relative !important;
      z-index: 100 !important;
      float: right !important;
    }
    #hd-theme-btn:hover {
      border-color: var(--hd-cyan) !important;
      box-shadow: 0 0 14px var(--hd-glow), 0 0 30px rgba(0,212,255,0.06) !important;
      transform: translateY(-1px) !important;
    }

    /* ── base ────────────────────────────────────── */
    body, body.hd-dark, body.hd-light {
      background-color: var(--hd-bg-deep) !important;
      background-image:
        linear-gradient(var(--hd-grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--hd-grid) 1px, transparent 1px) !important;
      background-size: 50px 50px !important;
      color: var(--hd-text) !important;
      transition: background-color 0.4s, color 0.4s !important;
    }

    /* Kill every Gradio wrapper background so body shows through */
    .gradio-container,
    .gradio-container > .main,
    .gradio-container > div,
    .dark .gradio-container,
    .dark .gradio-container > .main,
    .dark .gradio-container > div,
    main, .dark main,
    footer, .dark footer,
    .app, .dark .app {
      max-width: 100% !important;
      background: transparent !important;
      background-color: transparent !important;
    }

    /* scanline — invisible in light mode */
    .gradio-container::after {
      content: '';
      position: fixed; top:0; left:0;
      width: 100%; height: 100%;
      background: repeating-linear-gradient(
        0deg, transparent, transparent 3px,
        var(--hd-scanline) 3px, var(--hd-scanline) 6px) !important;
      pointer-events: none;
      z-index: 10000;
    }

    /* -- blocks & panels -- */
    .block { background: transparent !important; border: none !important; }
    .panel, .form { background: transparent !important; border: none !important; }

    /* ── chatbot ──────────────────────────────── */
    #healthdoc-chat {
      background: var(--hd-bg-primary) !important;
      border: 1px solid var(--hd-border) !important;
      border-radius: 10px !important;
    }
    #healthdoc-chat .wrapper {
      background: transparent !important;
    }
    #healthdoc-chat .message-wrap {
      background: transparent !important;
    }
    /* user bubbles */
    #healthdoc-chat .message.user,
    #healthdoc-chat [data-testid="user"] {
      background: rgba(0,212,255,0.05) !important;
      border-left: 2px solid rgba(0,212,255,0.30) !important;
      border-radius: 6px !important;
      color: var(--hd-text) !important;
    }
    /* bot bubbles */
    #healthdoc-chat .message.bot,
    #healthdoc-chat [data-testid="bot"] {
      background: rgba(11,18,32,0.50) !important;
      border-left: 2px solid rgba(0,212,255,0.10) !important;
      border-radius: 6px !important;
      color: var(--hd-text) !important;
    }
    #healthdoc-chat .bot p,
    #healthdoc-chat [data-testid="bot"] p {
      color: var(--hd-text) !important;
      line-height: 1.65 !important;
    }
    #healthdoc-chat .user p,
    #healthdoc-chat [data-testid="user"] p {
      color: var(--hd-text) !important;
    }
    /* markdown inside chat */
    #healthdoc-chat strong { color: var(--hd-cyan) !important; }
    #healthdoc-chat em     { color: var(--hd-text2) !important; }
    #healthdoc-chat hr     { border-color: var(--hd-border) !important; }
    #healthdoc-chat blockquote {
      border-left: 2px solid var(--hd-border-h) !important;
      background: var(--hd-glow) !important;
      padding: 6px 12px !important;
      margin: 6px 0 !important;
      color: var(--hd-text2) !important;
      font-size: 0.88em !important;
    }

    /* ── LIGHT-MODE overrides ──────────────────── */

    /* chatbot container + every nested wrapper */
    body.hd-light #healthdoc-chat,
    body.hd-light #healthdoc-chat > *,
    body.hd-light #healthdoc-chat .wrapper,
    body.hd-light #healthdoc-chat .message-wrap,
    body.hd-light #healthdoc-chat .wrap,
    body.hd-light #healthdoc-chat .panel {
      background: var(--hd-bg-primary) !important;
      background-color: var(--hd-bg-primary) !important;
    }
    body.hd-light #healthdoc-chat {
      border-color: var(--hd-border) !important;
    }

    /* chat bubbles */
    body.hd-light #healthdoc-chat .message.user,
    body.hd-light #healthdoc-chat [data-testid="user"] {
      background: rgba(0,119,170,0.06) !important;
      border-left: 2px solid rgba(0,119,170,0.30) !important;
    }
    body.hd-light #healthdoc-chat .message.bot,
    body.hd-light #healthdoc-chat [data-testid="bot"] {
      background: rgba(237,241,245,0.70) !important;
      border-left: 2px solid rgba(0,119,170,0.12) !important;
    }

    /* file upload — entire component including label banner */
    body.hd-light #hd-file-upload,
    body.hd-light #hd-file-upload > *,
    body.hd-light #hd-file-upload .wrap,
    body.hd-light #hd-file-upload .file-preview,
    body.hd-light #hd-file-upload .upload-container {
      background: var(--hd-bg-surface) !important;
      background-color: var(--hd-bg-surface) !important;
      border-color: rgba(0,80,120,0.18) !important;
      color: var(--hd-text) !important;
    }
    body.hd-light #hd-file-upload label,
    body.hd-light #hd-file-upload .label-text,
    body.hd-light #hd-file-upload span {
      color: var(--hd-text2) !important;
    }

    /* text input */
    body.hd-light #hd-query textarea {
      background: var(--hd-bg-input) !important;
      color: var(--hd-text) !important;
      border-color: var(--hd-border) !important;
    }
    body.hd-light #hd-query textarea::placeholder {
      color: #8a98a8 !important;
    }

    /* buttons */
    body.hd-light #hd-send-btn {
      background: linear-gradient(135deg, #005580 0%, #0077aa 100%) !important;
      color: #ffffff !important;
    }
    body.hd-light #hd-send-btn:hover {
      box-shadow: 0 0 15px rgba(0,119,170,0.25) !important;
    }
    body.hd-light #hd-clear-btn {
      background: var(--hd-bg-elevated) !important;
      color: var(--hd-text2) !important;
      border-color: var(--hd-border) !important;
    }

    /* export row + report download */
    body.hd-light #hd-report-btn {
      color: var(--hd-cyan-dim) !important;
      border-color: var(--hd-border) !important;
    }
    body.hd-light #hd-report-btn:hover {
      color: var(--hd-cyan) !important;
      border-color: var(--hd-cyan) !important;
    }
    body.hd-light #hd-report-file {
      background: var(--hd-bg-surface) !important;
      border-color: var(--hd-border) !important;
    }

    /* generic: nuke dark backgrounds inside all Gradio dark-mode selectors */
    body.hd-light .dark,
    body.hd-light [class*="dark:"] {
      background-color: transparent !important;
    }

    /* ── file upload ── */
    #hd-file-upload {
      border: 1px dashed rgba(0,212,255,0.15) !important;
      background: var(--hd-bg-surface) !important;
      border-radius: 8px !important;
      transition: all 0.3s !important;
    }
    #hd-file-upload:hover {
      border-color: var(--hd-border-h) !important;
      background: var(--hd-glow) !important;
    }

    /* ── text inputs ─────────────────────────── */
    #hd-query textarea {
      background: var(--hd-bg-input) !important;
      border: 1px solid var(--hd-border) !important;
      color: var(--hd-text) !important;
      border-radius: 8px !important;
      font-size: 0.95rem !important;
      transition: border-color 0.3s, box-shadow 0.3s !important;
    }
    #hd-query textarea:focus {
      border-color: var(--hd-border-f) !important;
      box-shadow: 0 0 0 2px var(--hd-glow), 0 0 25px rgba(0,212,255,0.04) !important;
    }
    #hd-query textarea::placeholder { color: #3d5468 !important; }

    /* ── buttons ──────────────────────────────── */
    #hd-send-btn {
      background: linear-gradient(135deg, #0099bb 0%, #00d4ff 100%) !important;
      border: none !important;
      color: #040810 !important;
      font-weight: 700 !important;
      letter-spacing: 0.1em !important;
      border-radius: 8px !important;
      font-size: 0.85rem !important;
      transition: all 0.3s !important;
      min-height: 42px !important;
    }
    #hd-send-btn:hover {
      box-shadow: 0 0 20px rgba(0,212,255,0.30),
                  0 0 50px rgba(0,212,255,0.08) !important;
      transform: translateY(-1px) !important;
    }
    #hd-clear-btn {
      background: transparent !important;
      border: 1px solid var(--hd-border) !important;
      color: var(--hd-text2) !important;
      border-radius: 8px !important;
      font-size: 0.8rem !important;
      letter-spacing: 0.08em !important;
      transition: all 0.3s !important;
      min-height: 42px !important;
    }
    #hd-clear-btn:hover {
      border-color: var(--hd-border-h) !important;
      color: var(--hd-text) !important;
    }

    /* ── export row ────────────────────────────── */
    #hd-export-row {
      display: flex !important;
      justify-content: flex-end !important;
      align-items: center !important;
      gap: 10px !important;
      margin-top: 4px !important;
      padding: 0 !important;
      min-height: auto !important;
    }
    #hd-report-btn {
      background: transparent !important;
      border: 1px solid var(--hd-border) !important;
      color: var(--hd-cyan-dim) !important;
      border-radius: 6px !important;
      font-size: 0.7rem !important;
      letter-spacing: 0.12em !important;
      text-transform: uppercase !important;
      font-family: 'JetBrains Mono','Consolas',monospace !important;
      padding: 5px 16px !important;
      min-height: 30px !important;
      max-height: 30px !important;
      transition: all 0.3s !important;
      cursor: pointer !important;
    }
    #hd-report-btn:hover {
      border-color: var(--hd-cyan) !important;
      color: var(--hd-cyan) !important;
      box-shadow: 0 0 12px var(--hd-glow) !important;
    }
    /* file download — compact styling */
    #hd-report-file {
      max-width: 280px !important;
      border: 1px solid var(--hd-border) !important;
      border-radius: 6px !important;
      background: var(--hd-bg-surface) !important;
      padding: 4px 8px !important;
      min-height: auto !important;
    }
    #hd-report-file .wrap,
    #hd-report-file .file-preview {
      background: transparent !important;
      min-height: auto !important;
    }

    /* ── labels ───────────────────────────────── */
    .hd-section label span,
    label .label-text {
      color: var(--hd-text2) !important;
      font-size: 0.75rem !important;
      letter-spacing: 0.12em !important;
      text-transform: uppercase !important;
    }

    /* ── examples ─────────────────────────────── */
    #hd-examples button,
    #hd-examples .example {
      background: var(--hd-bg-surface) !important;
      border: 1px solid var(--hd-border) !important;
      color: var(--hd-text2) !important;
      border-radius: 6px !important;
      font-size: 0.82rem !important;
      transition: all 0.2s !important;
    }
    #hd-examples button:hover,
    #hd-examples .example:hover {
      border-color: var(--hd-border-h) !important;
      color: var(--hd-cyan) !important;
      background: var(--hd-glow) !important;
    }
    #hd-examples .label-text {
      color: var(--hd-text3) !important;
      letter-spacing: 0.15em !important;
    }

    /* ── scrollbar ────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--hd-bg-primary); }
    ::-webkit-scrollbar-thumb {
      background: var(--hd-cyan-dim);
      border-radius: 3px;
    }

    /* -- spacing & layout -- */
    #hd-file-upload {
      margin-bottom: 16px !important;
    }
    #healthdoc-chat {
      margin-top: 0 !important;
      margin-bottom: 12px !important;
    }

    /* -- input row: force all children to same height & baseline -- */
    #hd-input-row {
      display: flex !important;
      align-items: stretch !important;
      gap: 8px !important;
    }
    #hd-input-row > div {
      margin-top: 0 !important;
      margin-bottom: 0 !important;
      padding-top: 0 !important;
    }
    /* remove the label gap that makes the textbox taller than buttons */
    #hd-input-row .wrap {
      margin-top: 0 !important;
    }
    #hd-query {
      padding-top: 0 !important;
      padding-bottom: 0 !important;
    }
    #hd-query textarea {
      min-height: 42px !important;
      max-height: 42px !important;
      padding: 10px 14px !important;
    }
    #hd-send-btn, #hd-clear-btn {
      height: 42px !important;
      min-height: 42px !important;
      max-height: 42px !important;
      padding: 0 16px !important;
      margin-top: 0 !important;
    }

    /* -- left sidebar: nuke ALL Gradio wrapper padding at every
          nesting level so upload + telemetry are identical width -- */
    #hd-left-col,
    #hd-left-col > *,
    #hd-left-col .block,
    #hd-left-col .wrap,
    #hd-left-col .form,
    #hd-left-col .panel,
    #hd-left-col .container {
      padding-left: 0 !important;
      padding-right: 0 !important;
      margin-left: 0 !important;
      margin-right: 0 !important;
      max-width: 100% !important;
      box-sizing: border-box !important;
    }
    #hd-left-col {
      display: flex !important;
      flex-direction: column !important;
      gap: 12px !important;
    }

    /* -- animations -- */
    @keyframes hd-glow-border {
      0%,100% { box-shadow: 0 0 5px rgba(0,212,255,0.05); }
      50%     { box-shadow: 0 0 15px rgba(0,212,255,0.12); }
    }
    """

# HEADER & FOOTER HTML

_HEADER_HTML = """
<div id="hd-header-wrap" style="
    display:flex; flex-direction:column; align-items:center;
    padding:20px 16px 16px; margin-bottom:8px;
    border-bottom:1px solid var(--hd-border);
    text-align:center;
    position:relative;
">
  <div style="display:flex; align-items:center; gap:12px; justify-content:center;">
    <div style="width:10px;height:10px;border-radius:50%;
                background:var(--hd-header-dot);
                box-shadow:var(--hd-header-dot-shadow);"></div>
    <span style="color:var(--hd-cyan);font-size:1.45rem;font-weight:700;
                 letter-spacing:0.12em;font-family:'Inter',system-ui,sans-serif;">
      HEALTHDOC AI
    </span>
    <div style="width:10px;height:10px;border-radius:50%;
                background:var(--hd-header-dot);
                box-shadow:var(--hd-header-dot-shadow);"></div>
  </div>
  <div style="color:var(--hd-text2);font-size:0.72rem;letter-spacing:0.22em;
              text-transform:uppercase;margin-top:6px;">
    Intelligent Medical Document Analysis System
  </div>
  <div style="color:var(--hd-text3);font-size:0.60rem;letter-spacing:0.18em;
              text-transform:uppercase;margin-top:4px;">
    v1.0 // RESEARCH USE ONLY
  </div>
</div>
"""

_FOOTER_HTML = """
<div style="
    border-top:1px solid var(--hd-border);
    padding:14px 4px 4px;
    margin-top:16px;
    display:flex; justify-content:space-between; align-items:center;
    flex-wrap:wrap; gap:8px;
">
  <div style="color:var(--hd-text3);font-size:0.7rem;letter-spacing:0.08em;line-height:1.5;">
    DISCLAIMER — This system is for research purposes only.
    Always consult a qualified healthcare professional for medical advice.
  </div>
  <div style="color:var(--hd-text3);font-size:0.62rem;letter-spacing:0.12em;white-space:nowrap;">
    POWERED BY OLLAMA + LANGCHAIN + CHROMADB
  </div>
</div>
"""

# EVENT HANDLERS (module-level, stateless)

def _user_message(message, chat_history):
    """Append user message to history and clear the input box."""
    if not message or not message.strip():
        return "", chat_history
    return "", chat_history + [{"role": "user", "content": message.strip()}]


def _on_file_upload(file, chat_history):
    """Post a confirmation message when a new document is uploaded."""
    if file is None:
        return chat_history
    filename = Path(file).name
    return chat_history + [
        {"role": "assistant", "content": f"**Document loaded:** `{filename}` — Ready for analysis."}
    ]


def _generate_report(chat_history, file):
    """Build a Markdown report from the chat session and return a temp file path.

    The report includes:
      - Header with timestamp and document name
      - Full Q&A transcript with source citations preserved
      - Footer disclaimer
    Returns a file path for gr.File, or None if nothing to export.
    """
    if not chat_history:
        return gr.update(value=None, visible=False)

    # Gather Q&A pairs
    qa_pairs = []
    current_q = None
    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            current_q = content
        elif role == "assistant" and current_q is not None:
            qa_pairs.append((current_q, content))
            current_q = None

    if not qa_pairs:
        return gr.update(value=None, visible=False)

    # Build the markdown
    doc_name = Path(file).name if file else "Unknown document"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# HealthDoc AI — Analysis Report",
        "",
        f"**Generated:** {now}  ",
        f"**Document:** `{doc_name}`  ",
        f"**Queries:** {len(qa_pairs)}  ",
        "",
        "---",
        "",
    ]

    for i, (question, answer) in enumerate(qa_pairs, 1):
        lines.append(f"## Query {i}")
        lines.append("")
        lines.append(f"**Q:** {question}")
        lines.append("")
        lines.append(f"**A:** {answer}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.extend([
        "*This report was generated by HealthDoc AI. "
        "It is intended for research purposes only. "
        "Always consult a qualified healthcare professional for medical advice.*",
        "",
    ])

    report_md = "\n".join(lines)

    # Write to a temp file that Gradio can serve
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        prefix=f"healthdoc_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_",
        delete=False,
        encoding="utf-8",
    )
    tmp.write(report_md)
    tmp.close()

    return gr.update(value=tmp.name, visible=True)


# APP LAUNCHER
def launch_gradio_app():
    """Build and launch the Prometheus HUD Gradio interface."""

    sys_info = _get_system_info()
    initial_metrics = _build_metrics_html(sys_info)
    initial_chat = []  # Chat starts empty - welcome banner is separate

    # closures that capture sys_info

    def bot_response(chat_history, file):
        """Generator: run the RAG pipeline and stream the answer word-by-word."""
        if not chat_history:
            return

        # Last entry is the user message added by _user_message()
        raw_content = chat_history[-1]["content"]
        user_msg = raw_content if isinstance(raw_content, str) else str(raw_content)

        # validate file
        if file is None:
            chat_history = chat_history + [
                {"role": "assistant", "content": "**No document uploaded.** Please upload a PDF first."}
            ]
            yield chat_history, _build_metrics_html(sys_info, status="ERROR")
            return

        # show processing indicator
        chat_history = chat_history + [
            {"role": "assistant", "content": "Analyzing document..."}
        ]
        yield chat_history, _build_metrics_html(sys_info, status="PROCESSING")

        # run instrumented pipeline
        try:
            answer, sources, metrics = retriever_qa_with_metadata(file, user_msg)
        except Exception as exc:
            logger.exception("Unhandled error in bot_response")
            chat_history[-1] = {"role": "assistant", "content": f"**System error:** {exc}"}
            yield chat_history, _build_metrics_html(sys_info, status="ERROR")
            return

        # detect error answers (display instantly)
        is_error = (
            not metrics
            or answer.startswith(("Error", "Input error", "Could not", "Please", "Your question"))
        )
        if is_error:
            chat_history[-1] = {"role": "assistant", "content": f"**Error:** {answer}"}
            yield chat_history, _build_metrics_html(sys_info, metrics or None, "ERROR")
            return

        # typewriter streaming
        final_metrics_html = _build_metrics_html(sys_info, metrics, "COMPLETE")
        words = answer.split()
        streamed = ""

        for i, word in enumerate(words):
            streamed += word + " "
            if i % _STREAM_CHUNK == 0 or i == len(words) - 1:
                chat_history[-1] = {"role": "assistant", "content": streamed.strip()}
                yield chat_history, final_metrics_html
                time.sleep(_STREAM_DELAY)

        # append source attribution
        if sources:
            src_md = "\n\n---\n**Sources Referenced:**\n"
            for s in sources:
                page_display = int(s["page"]) + 1 if str(s["page"]).isdigit() else s["page"]
                excerpt = s["excerpt"][:160]
                src_md += f"\n> **Page {page_display}** — *{excerpt} ...*\n"
            chat_history[-1] = {"role": "assistant", "content": streamed.strip() + src_md}
            yield chat_history, final_metrics_html

    def clear_chat():
        """Reset chat, metrics, and report file to initial state."""
        return initial_chat, initial_metrics, gr.update(value=None, visible=False)

    # theme

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[
            gr.themes.GoogleFont("Inter"),
            "system-ui",
            "sans-serif",
        ],
        font_mono=[
            gr.themes.GoogleFont("JetBrains Mono"),
            "Consolas",
            "monospace",
        ],
    ).set(
        body_background_fill="#040810",
        body_background_fill_dark="#040810",
        block_background_fill="transparent",
        block_background_fill_dark="transparent",
        block_border_color="transparent",
        block_border_color_dark="transparent",
        input_background_fill="#0a1018",
        input_background_fill_dark="#0a1018",
        body_text_color="#c8d6e5",
        body_text_color_dark="#c8d6e5",
        body_text_color_subdued="#6b8299",
        body_text_color_subdued_dark="#6b8299",
    )

    # layout
    with gr.Blocks(
        title="HealthDoc AI",
        theme=theme,
        css=_get_css(),
        js="() => { document.body.classList.add('dark', 'hd-dark'); }",
    ) as app:

        # header + theme toggle
        gr.HTML(_HEADER_HTML)
        theme_btn = gr.Button(
            "Light",
            elem_id="hd-theme-btn",
            variant="secondary",
            size="sm",
        )

        with gr.Row(equal_height=True):

            # LEFT: document upload + telemetry (flush-aligned column)
            with gr.Column(scale=1, min_width=280, elem_id="hd-left-col"):
                pdf_input = gr.File(
                    label="Document",
                    file_types=[".pdf"],
                    type="filepath",
                    elem_id="hd-file-upload",
                    height=120,
                )
                metrics_panel = gr.HTML(
                    value=initial_metrics,
                    elem_id="hd-telemetry",
                )

            # RIGHT: chat area (main focus)
            with gr.Column(scale=3, min_width=500):

                # chat console
                chatbot = gr.Chatbot(
                    value=initial_chat,
                    height=520,
                    elem_id="healthdoc-chat",
                    show_label=False,
                    # type="messages",
                )

                # query input row — all children aligned to same height
                with gr.Row(elem_id="hd-input-row"):
                    query_input = gr.Textbox(
                        placeholder="Ask about your document …",
                        show_label=False,
                        lines=1,
                        max_lines=1,
                        elem_id="hd-query",
                        scale=5,
                    )
                    submit_btn = gr.Button(
                        "Analyze",
                        variant="primary",
                        elem_id="hd-send-btn",
                        scale=1,
                        min_width=110,
                    )
                    clear_btn = gr.Button(
                        "Clear",
                        variant="secondary",
                        elem_id="hd-clear-btn",
                        scale=0,
                        min_width=80,
                    )

                # export row — right-aligned, subtle secondary action
                with gr.Row(elem_id="hd-export-row"):
                    report_btn = gr.Button(
                        "Export Report",
                        variant="secondary",
                        elem_id="hd-report-btn",
                        size="sm",
                        min_width=140,
                    )
                    report_file = gr.File(
                        label="Report",
                        visible=False,
                        elem_id="hd-report-file",
                        interactive=False,
                    )

        # footer
        gr.HTML(_FOOTER_HTML)

        # event wiring
        submit_chain = dict(
            fn=_user_message,
            inputs=[query_input, chatbot],
            outputs=[query_input, chatbot],
        )
        bot_chain = dict(
            fn=bot_response,
            inputs=[chatbot, pdf_input],
            outputs=[chatbot, metrics_panel],
        )

        query_input.submit(**submit_chain).then(**bot_chain)
        submit_btn.click(**submit_chain).then(**bot_chain)

        clear_btn.click(
            fn=clear_chat,
            inputs=None,
            outputs=[chatbot, metrics_panel, report_file],
        )

        report_btn.click(
            fn=_generate_report,
            inputs=[chatbot, pdf_input],
            outputs=[report_file],
        )

        pdf_input.change(
            fn=_on_file_upload,
            inputs=[pdf_input, chatbot],
            outputs=[chatbot],
        )

        # theme toggle wiring — pure JS, no server round-trip
        # NOTE: We keep Gradio's 'dark' class ALWAYS on body so Gradio
        # doesn't inject its own built-in light theme. We only toggle
        # hd-dark / hd-light which control OUR CSS variable palette.
        theme_btn.click(
            fn=None,
            inputs=None,
            outputs=[theme_btn],
            js="""() => {
                const body = document.body;
                const isDark = body.classList.contains('hd-dark');
                body.classList.add('dark');
                if (isDark) {
                    body.classList.remove('hd-dark');
                    body.classList.add('hd-light');
                    return 'Dark';
                } else {
                    body.classList.remove('hd-light');
                    body.classList.add('hd-dark');
                    return 'Light';
                }
            }""",
        )

    return app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
    )


if __name__ == "__main__":
    launch_gradio_app()
