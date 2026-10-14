"""
visualization/live_radar_component.py - Interactive 60 FPS HTML5 Canvas Tactical Radar HUD.

Generates an aerospace-grade real-time interactive ground control station radar simulator:
- Dual drone real-time animated flight along true wind-drift compensated kinematic vectors.
- Waypoint beacon survey pulsing and status transitions (Unvisited -> Scanning -> Rescued).
- Live HUD dials: Airspeed, Ground Speed, Heading, Power, Battery SoC gauge, and Energy Delta.
- Time scrubber, Play/Pause/Reset, speed multipliers (1x to 50x), and return-to-base landing sequence.
"""

from __future__ import annotations

import json
from typing import List, Dict, Any
from core.physics import Waypoint, WindField, DronePhysicsModel


def render_live_radar_html(
    waypoints: List[Waypoint],
    wind_field: WindField,
    drone_model: DronePhysicsModel,
    frames: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> str:
    """Build a standalone, highly-performant HTML5 / Canvas / WebGL-styled radar simulator."""

    wp_data = [
        {
            "id": wp.id,
            "label": wp.label,
            "x": round(wp.x, 1),
            "y": round(wp.y, 1),
            "priority": wp.priority,
            "loiter_s": wp.loiter_time_s,
        }
        for wp in waypoints
    ]

    wind_data = {
        "speed": round(wind_field.speed_mps, 1),
        "deg": round(wind_field.direction_deg, 1),
        "vx": round(wind_field.vx, 2),
        "vy": round(wind_field.vy, 2),
    }

    drone_data = {
        "airspeed": drone_model.nominal_airspeed_mps,
        "capacity_wh": drone_model.battery_capacity_wh,
        "reserve_pct": round(drone_model.reserve_margin_ratio * 100.0, 1),
    }

    payload_json = json.dumps({
        "waypoints": wp_data,
        "wind": wind_data,
        "drone": drone_data,
        "frames": frames,
        "summary": summary,
    })

    template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@400;600;700&display=swap');
  
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    user-select: none;
  }

  body {
    background: #080D1A;
    color: #E2E8F0;
    font-family: 'Outfit', sans-serif;
    overflow: hidden;
    padding: 10px;
  }

  .radar-container {
    display: flex;
    flex-direction: column;
    gap: 12px;
    max-width: 1200px;
    margin: 0 auto;
    background: #0B1120;
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
  }

  /* Top Control & Status Bar */
  .control-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(56, 189, 248, 0.15);
  }

  .btn-group {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  button {
    background: rgba(30, 41, 59, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #F8FAFC;
    padding: 6px 12px;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  button:hover {
    background: rgba(56, 189, 248, 0.25);
    border-color: #38BDF8;
    transform: translateY(-1px);
  }

  button.active {
    background: #0284C7;
    border-color: #38BDF8;
    color: #FFFFFF;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.5);
  }

  .timeline-container {
    display: flex;
    align-items: center;
    gap: 10px;
    flex: 1;
    min-width: 260px;
  }

  .time-slider {
    flex: 1;
    -webkit-appearance: none;
    height: 6px;
    border-radius: 4px;
    background: #1E293B;
    outline: none;
    cursor: pointer;
  }

  .time-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #38BDF8;
    border: 2px solid #FFFFFF;
    cursor: pointer;
    box-shadow: 0 0 8px #38BDF8;
  }

  .time-display {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    font-weight: 700;
    color: #38BDF8;
    min-width: 100px;
    text-align: right;
  }

  /* Main Viewport: Canvas & Side Telemetry HUD */
  .viewport-grid {
    display: grid;
    grid-template-columns: 1fr 330px;
    gap: 14px;
  }

  @media (max-width: 900px) {
    .viewport-grid {
      grid-template-columns: 1fr;
    }
  }

  .canvas-wrapper {
    position: relative;
    width: 100%;
    height: 510px;
    background: radial-gradient(circle at center, #0B1426 0%, #060A14 100%);
    border-radius: 12px;
    border: 1px solid rgba(56, 189, 248, 0.2);
    overflow: hidden;
  }

  canvas {
    width: 100%;
    height: 100%;
    display: block;
  }

  /* Floating Overlay Badges */
  .wind-hud {
    position: absolute;
    top: 12px;
    left: 12px;
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 6px 10px;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #94A3B8;
    pointer-events: none;
    backdrop-filter: blur(4px);
  }

  .wind-val {
    color: #38BDF8;
    font-weight: 700;
  }

  .legend-hud {
    position: absolute;
    bottom: 12px;
    left: 12px;
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid rgba(255, 255, 255, 0.12);
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.74rem;
    display: flex;
    flex-direction: column;
    gap: 4px;
    pointer-events: none;
    backdrop-filter: blur(4px);
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }

  /* Telemetry Cards Panel */
  .telemetry-panel {
    display: flex;
    flex-direction: column;
    gap: 10px;
    overflow-y: auto;
    max-height: 510px;
  }

  .tele-card {
    background: rgba(15, 23, 42, 0.85);
    border-radius: 10px;
    padding: 10px 12px;
    border-left: 4px solid #38BDF8;
    display: flex;
    flex-direction: column;
    gap: 6px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }

  .tele-card.trad {
    border-left-color: #F97316;
  }

  .tele-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.80rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .tele-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(56, 189, 248, 0.15);
    color: #38BDF8;
  }

  .tele-card.trad .tele-status {
    background: rgba(249, 115, 22, 0.15);
    color: #FB923C;
  }

  .tele-metrics-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
  }

  .metric-item {
    display: flex;
    flex-direction: column;
  }

  .metric-lbl {
    font-size: 0.65rem;
    color: #64748B;
  }

  .metric-val {
    font-size: 0.98rem;
    font-weight: 700;
    color: #F8FAFC;
  }

  /* Battery Gauge Bar */
  .soc-bar-container {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .soc-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.70rem;
    font-family: 'JetBrains Mono', monospace;
    color: #94A3B8;
  }

  .soc-bar-bg {
    width: 100%;
    height: 7px;
    background: #1E293B;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
  }

  .soc-bar-fill {
    height: 100%;
    width: 100%;
    background: linear-gradient(90deg, #10B981, #38BDF8);
    transition: width 0.1s linear;
  }

  .tele-card.trad .soc-bar-fill {
    background: linear-gradient(90deg, #EF4444, #F97316);
  }

  /* Scoreboard Delta Banner */
  .delta-banner {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.25) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 10px;
    padding: 8px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .delta-title {
    font-size: 0.75rem;
    color: #A7F3D0;
    font-weight: 600;
  }

  .delta-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #34D399;
  }
</style>
</head>
<body>

<div class="radar-container">
  <!-- Controls -->
  <div class="control-bar">
    <div class="btn-group">
      <button id="btnPlay" class="active">▶ PLAY</button>
      <button id="btnPause">⏸ PAUSE</button>
      <button id="btnReset">🔄 RESET</button>
    </div>

    <div class="btn-group">
      <button class="btn-speed" data-speed="1">1x</button>
      <button class="btn-speed" data-speed="2">2x</button>
      <button class="btn-speed active" data-speed="5">5x</button>
      <button class="btn-speed" data-speed="10">10x</button>
      <button class="btn-speed" data-speed="25">25x</button>
    </div>

    <div class="timeline-container">
      <input type="range" id="timeSlider" class="time-slider" min="0" max="1000" value="0">
      <div id="timeDisplay" class="time-display">00:00 / 00:00</div>
    </div>
  </div>

  <!-- Viewport Grid -->
  <div class="viewport-grid">
    <div class="canvas-wrapper" id="canvasWrapper">
      <canvas id="radarCanvas"></canvas>

      <div class="wind-hud">
        WIND: <span class="wind-val" id="hudWind">6.8 m/s @ 55°</span><br>
        TARGETS: <span class="wind-val" id="hudSurvivors">10 Beacons</span>
      </div>

      <div class="legend-hud">
        <div class="legend-item">
          <div class="legend-dot" style="background:#06B6D4; box-shadow:0 0 6px #06B6D4;"></div>
          <span>AeroSAR Bitmask DP (Proposed)</span>
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background:#F97316; box-shadow:0 0 6px #F97316;"></div>
          <span id="tradLegendLabel">Traditional Baseline</span>
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background:#EAB308;"></div>
          <span>Survivor Beacons (Active/Scanned)</span>
        </div>
        <div class="legend-item">
          <div class="legend-dot" style="background:#FFFFFF; border:1px solid #38BDF8;"></div>
          <span>Base Camp Depot [0]</span>
        </div>
      </div>
    </div>

    <!-- Live Telemetry Stream -->
    <div class="telemetry-panel">
      <!-- Proposed AeroSAR DP Card -->
      <div class="tele-card" id="cardDp">
        <div class="tele-header">
          <span style="color:#38BDF8;">🚁 AEROSAR BITMASK DP</span>
          <span class="tele-status" id="dpStatus">TAKEOFF</span>
        </div>
        <div class="tele-metrics-row">
          <div class="metric-item">
            <span class="metric-lbl">TARGET NODE</span>
            <span class="metric-val" id="dpTarget">S_01</span>
          </div>
          <div class="metric-item">
            <span class="metric-lbl">GROUND SPEED Vg</span>
            <span class="metric-val" id="dpSpeed">15.0 m/s</span>
          </div>
          <div class="metric-item">
            <span class="metric-lbl">CUMULATIVE ENERGY</span>
            <span class="metric-val" id="dpEnergy">0.0 Wh</span>
          </div>
          <div class="metric-item">
            <span class="metric-lbl">SORTIE DISTANCE</span>
            <span class="metric-val" id="dpDist">0.0 km</span>
          </div>
        </div>
        <div class="soc-bar-container">
          <div class="soc-bar-label">
            <span>BATTERY STATE-OF-CHARGE</span>
            <span id="dpSoc">100.0%</span>
          </div>
          <div class="soc-bar-bg">
            <div class="soc-bar-fill" id="dpSocBar" style="width:100%;"></div>
          </div>
        </div>
      </div>

      <!-- Traditional Baseline Card -->
      <div class="tele-card trad" id="cardTrad">
        <div class="tele-header">
          <span style="color:#FB923C;" id="tradTitle">🚁 TRADITIONAL BASELINE</span>
          <span class="tele-status" id="tradStatus">TAKEOFF</span>
        </div>
        <div class="tele-metrics-row">
          <div class="metric-item">
            <span class="metric-lbl">TARGET NODE</span>
            <span class="metric-val" id="tradTarget">S_01</span>
          </div>
          <div class="metric-item">
            <span class="metric-lbl">GROUND SPEED Vg</span>
            <span class="metric-val" id="tradSpeed">15.0 m/s</span>
          </div>
          <div class="metric-item">
            <span class="metric-lbl">CUMULATIVE ENERGY</span>
            <span class="metric-val" id="tradEnergy">0.0 Wh</span>
          </div>
          <div class="metric-item">
            <span class="metric-lbl">SORTIE DISTANCE</span>
            <span class="metric-val" id="tradDist">0.0 km</span>
          </div>
        </div>
        <div class="soc-bar-container">
          <div class="soc-bar-label">
            <span>BATTERY STATE-OF-CHARGE</span>
            <span id="tradSoc">100.0%</span>
          </div>
          <div class="soc-bar-bg">
            <div class="soc-bar-fill" id="tradSocBar" style="width:100%;"></div>
          </div>
        </div>
      </div>

      <!-- Real-time Energy Savings Delta -->
      <div class="delta-banner">
        <div>
          <div class="delta-title">REAL-TIME AERODYNAMIC SAVINGS</div>
          <div style="font-size:0.68rem; color:#6EE7B7;">Bitmask DP vs Traditional</div>
        </div>
        <div class="delta-num" id="liveDeltaWh">+0.0 Wh</div>
      </div>
    </div>
  </div>
</div>

<script>
  const data = __PAYLOAD_JSON__;
  const waypoints = data.waypoints || [];
  const wind = data.wind || { speed: 6.8, deg: 55 };
  const drone = data.drone || { airspeed: 15.0 };
  const frames = data.frames || [];
  const summary = data.summary || {};

  document.getElementById('hudWind').textContent = `${wind.speed} m/s @ ${wind.deg}°`;
  document.getElementById('hudSurvivors').textContent = `${Math.max(0, waypoints.length - 1)} Beacons`;
  document.getElementById('tradLegendLabel').textContent = summary.trad_name || 'Traditional';
  document.getElementById('tradTitle').textContent = `🚁 ${(summary.trad_name || 'Traditional').toUpperCase()}`;

  const canvas = document.getElementById('radarCanvas');
  const ctx = canvas.getContext('2d');
  const wrapper = document.getElementById('canvasWrapper');

  let isPlaying = true;
  let playSpeed = 5.0;
  let currentFrameIdx = 0;
  let totalFrames = frames.length;
  let lastTimeMs = performance.now();

  function resizeCanvas() {
    const rect = wrapper.getBoundingClientRect();
    const w = (rect && rect.width > 20) ? rect.width : (wrapper.clientWidth || 750);
    const h = (rect && rect.height > 20) ? rect.height : (wrapper.clientHeight || 510);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
  }
  resizeCanvas();
  window.addEventListener('resize', () => {
    resizeCanvas();
    renderFrame();
  });

  // Compute spatial bounding box and scaling
  let minX = 0, maxX = 0, minY = 0, maxY = 0;
  waypoints.forEach(wp => {
    minX = Math.min(minX, wp.x);
    maxX = Math.max(maxX, wp.x);
    minY = Math.min(minY, wp.y);
    maxY = Math.max(maxY, wp.y);
  });
  const spanX = Math.max(1000, (maxX - minX) * 1.35);
  const spanY = Math.max(1000, (maxY - minY) * 1.35);
  const maxSpan = Math.max(spanX, spanY);
  const centerX = (minX + maxX) / 2.0;
  const centerY = (minY + maxY) / 2.0;

  function toScreen(x, y) {
    const w = canvas.width;
    const h = canvas.height;
    const scale = Math.min(w, h) / maxSpan;
    const sx = (w / 2) + (x - centerX) * scale;
    const sy = (h / 2) - (y - centerY) * scale;
    return [sx, sy];
  }

  function fmtTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  }

  const maxSec = summary.max_duration_s || 1000;

  function updateHUD(f) {
    if (!f || !f.dp || !f.trad) return;
    const tSec = f.t;
    document.getElementById('timeDisplay').textContent = `${fmtTime(tSec)} / ${fmtTime(maxSec)}`;
    document.getElementById('timeSlider').value = Math.floor((currentFrameIdx / Math.max(1, totalFrames - 1)) * 1000);

    // DP Telemetry
    document.getElementById('dpStatus').textContent = f.dp.status;
    document.getElementById('dpTarget').textContent = f.dp.target_label;
    document.getElementById('dpSpeed').textContent = `${f.dp.ground_speed_mps.toFixed(1)} m/s`;
    document.getElementById('dpEnergy').textContent = `${f.dp.energy_consumed_wh.toFixed(1)} Wh`;
    document.getElementById('dpDist').textContent = `${(f.dp.distance_flown_m / 1000.0).toFixed(2)} km`;
    document.getElementById('dpSoc').textContent = `${f.dp.soc_percent.toFixed(1)}%`;
    document.getElementById('dpSocBar').style.width = `${Math.max(0, Math.min(100, f.dp.soc_percent))}%`;

    // Traditional Telemetry
    document.getElementById('tradStatus').textContent = f.trad.status;
    document.getElementById('tradTarget').textContent = f.trad.target_label;
    document.getElementById('tradSpeed').textContent = `${f.trad.ground_speed_mps.toFixed(1)} m/s`;
    document.getElementById('tradEnergy').textContent = `${f.trad.energy_consumed_wh.toFixed(1)} Wh`;
    document.getElementById('tradDist').textContent = `${(f.trad.distance_flown_m / 1000.0).toFixed(2)} km`;
    document.getElementById('tradSoc').textContent = `${f.trad.soc_percent.toFixed(1)}%`;
    document.getElementById('tradSocBar').style.width = `${Math.max(0, Math.min(100, f.trad.soc_percent))}%`;

    // Savings Delta
    const deltaWh = (f.trad.energy_consumed_wh - f.dp.energy_consumed_wh);
    const sign = deltaWh >= 0 ? '+' : '';
    document.getElementById('liveDeltaWh').textContent = `${sign}${deltaWh.toFixed(1)} Wh (${f.savings_pct.toFixed(1)}%)`;
  }

  let radarAngle = 0;

  function drawRadarGrid() {
    const w = canvas.width;
    const h = canvas.height;
    const [cx, cy] = toScreen(0, 0);
    const dpr = window.devicePixelRatio || 1;

    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.12)';
    const rings = [500, 1000, 1500, 2000, 2500];
    const scale = Math.min(w, h) / maxSpan;

    rings.forEach(r => {
      const pixR = r * scale;
      ctx.beginPath();
      ctx.arc(cx, cy, pixR, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = 'rgba(56, 189, 248, 0.35)';
      ctx.font = `${Math.floor(10 * dpr)}px 'JetBrains Mono', monospace`;
      ctx.fillText(`${r}m`, cx + pixR + 4, cy - 4);
    });

    ctx.strokeStyle = 'rgba(56, 189, 248, 0.08)';
    ctx.beginPath();
    ctx.moveTo(0, cy); ctx.lineTo(w, cy);
    ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
    ctx.stroke();

    radarAngle += 0.025;
    const sweepLen = Math.max(w, h);
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.2)';
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(radarAngle) * sweepLen, cy + Math.sin(radarAngle) * sweepLen);
    ctx.stroke();
  }

  function drawWindVector() {
    const w = canvas.width;
    const dpr = window.devicePixelRatio || 1;
    const originX = w - 60 * dpr;
    const originY = 60 * dpr;

    const rad = (wind.deg * Math.PI) / 180.0;
    const arrowLen = 30 * dpr;
    const tx = originX + Math.cos(rad) * arrowLen;
    const ty = originY - Math.sin(rad) * arrowLen;

    ctx.strokeStyle = '#38BDF8';
    ctx.lineWidth = 2.5 * dpr;
    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.lineTo(tx, ty);
    ctx.stroke();

    const headAngle = Math.atan2(originY - ty, tx - originX);
    const headLen = 8 * dpr;
    ctx.fillStyle = '#38BDF8';
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - headLen * Math.cos(headAngle - Math.PI / 6), ty + headLen * Math.sin(headAngle - Math.PI / 6));
    ctx.lineTo(tx - headLen * Math.cos(headAngle + Math.PI / 6), ty + headLen * Math.sin(headAngle + Math.PI / 6));
    ctx.fill();
  }

  function drawWaypoints(f) {
    const dpr = window.devicePixelRatio || 1;
    const dpVisited = new Set(f && f.dp ? f.dp.visited_nodes : []);
    const tradVisited = new Set(f && f.trad ? f.trad.visited_nodes : []);

    waypoints.forEach(wp => {
      const [sx, sy] = toScreen(wp.x, wp.y);

      if (wp.id === 0) {
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(sx, sy, 7 * dpr, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#38BDF8';
        ctx.lineWidth = 2 * dpr;
        ctx.beginPath();
        ctx.arc(sx, sy, 12 * dpr, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = '#FFFFFF';
        ctx.font = `bold ${Math.floor(11 * dpr)}px 'Outfit', sans-serif`;
        ctx.fillText('BASE DEPOT [0]', sx + 14 * dpr, sy + 4);
      } else {
        const isDpCleared = dpVisited.has(wp.id);
        const isTradCleared = tradVisited.has(wp.id);
        const isLoitering = (f && f.dp && f.dp.to_node_id === wp.id && f.dp.is_loitering) || 
                            (f && f.trad && f.trad.to_node_id === wp.id && f.trad.is_loitering);

        let color = '#EAB308';
        if (isLoitering) {
          color = '#38BDF8';
        } else if (isDpCleared && isTradCleared) {
          color = '#10B981';
        } else if (isDpCleared) {
          color = '#06B6D4';
        } else if (isTradCleared) {
          color = '#F97316';
        }

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(sx, sy, 5 * dpr, 0, Math.PI * 2);
        ctx.fill();

        if (isLoitering) {
          ctx.strokeStyle = color;
          ctx.lineWidth = 2 * dpr;
          ctx.beginPath();
          ctx.arc(sx, sy, 14 * dpr, 0, Math.PI * 2);
          ctx.stroke();
        }

        ctx.fillStyle = '#CBD5E1';
        ctx.font = `${Math.floor(10 * dpr)}px 'JetBrains Mono', monospace`;
        ctx.fillText(wp.label, sx + 8 * dpr, sy - 6);
      }
    });
  }

  function drawTrajectoryTrails(f) {
    const dpr = window.devicePixelRatio || 1;
    if (summary && summary.dp_path && summary.dp_path.length > 0) {
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.45)';
      ctx.lineWidth = 2.5 * dpr;
      ctx.beginPath();
      summary.dp_path.forEach((nodeIdx, idx) => {
        const wp = waypoints[nodeIdx];
        if (!wp) return;
        const [sx, sy] = toScreen(wp.x, wp.y);
        if (idx === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      ctx.stroke();
    }

    if (summary && summary.trad_path && summary.trad_path.length > 0) {
      ctx.strokeStyle = 'rgba(249, 115, 22, 0.35)';
      ctx.lineWidth = 2 * dpr;
      ctx.setLineDash([6 * dpr, 6 * dpr]);
      ctx.beginPath();
      summary.trad_path.forEach((nodeIdx, idx) => {
        const wp = waypoints[nodeIdx];
        if (!wp) return;
        const [sx, sy] = toScreen(wp.x, wp.y);
        if (idx === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  function drawDrone(x, y, headingDeg, color, label) {
    const [sx, sy] = toScreen(x, y);
    const dpr = window.devicePixelRatio || 1;
    const rad = (-headingDeg * Math.PI) / 180.0;

    ctx.save();
    ctx.translate(sx, sy);
    ctx.rotate(rad);

    const armLen = 10 * dpr;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * dpr;

    ctx.beginPath();
    ctx.moveTo(-armLen, -armLen); ctx.lineTo(armLen, armLen);
    ctx.moveTo(-armLen, armLen); ctx.lineTo(armLen, -armLen);
    ctx.stroke();

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(0, 0, 4 * dpr, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.moveTo(armLen + 4, 0);
    ctx.lineTo(armLen - 3, -4);
    ctx.lineTo(armLen - 3, 4);
    ctx.fill();

    ctx.restore();

    ctx.fillStyle = color;
    ctx.font = `bold ${Math.floor(10 * dpr)}px 'JetBrains Mono', monospace`;
    ctx.fillText(label, sx + 12 * dpr, sy + 14 * dpr);
  }

  function renderFrame() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    drawRadarGrid();
    drawWindVector();

    if (!frames || frames.length === 0) return;

    const idx = Math.min(totalFrames - 1, Math.max(0, Math.floor(currentFrameIdx)));
    const f = frames[idx];
    if (!f) return;

    drawTrajectoryTrails(f);
    drawWaypoints(f);

    if (f.dp) drawDrone(f.dp.x, f.dp.y, f.dp.heading_deg, '#06B6D4', 'DP-UAV');
    if (f.trad) drawDrone(f.trad.x, f.trad.y, f.trad.heading_deg, '#F97316', 'TRAD-UAV');

    updateHUD(f);
  }

  function animLoop(now) {
    const dt = Math.min(0.1, (now - lastTimeMs) / 1000.0);
    lastTimeMs = now;

    if (isPlaying && totalFrames > 1) {
      const frameDelta = (playSpeed * totalFrames * dt) / Math.max(1, maxSec);
      currentFrameIdx += frameDelta;
      if (currentFrameIdx >= totalFrames - 1) {
        currentFrameIdx = totalFrames - 1;
        isPlaying = false;
        document.getElementById('btnPlay').classList.remove('active');
        document.getElementById('btnPause').classList.add('active');
      }
    }

    renderFrame();
    requestAnimationFrame(animLoop);
  }

  renderFrame();
  requestAnimationFrame(animLoop);

  // Event Handlers
  document.getElementById('btnPlay').addEventListener('click', () => {
    if (currentFrameIdx >= totalFrames - 1) {
      currentFrameIdx = 0;
    }
    isPlaying = true;
    lastTimeMs = performance.now();
    document.getElementById('btnPlay').classList.add('active');
    document.getElementById('btnPause').classList.remove('active');
  });

  document.getElementById('btnPause').addEventListener('click', () => {
    isPlaying = false;
    document.getElementById('btnPause').classList.add('active');
    document.getElementById('btnPlay').classList.remove('active');
  });

  document.getElementById('btnReset').addEventListener('click', () => {
    currentFrameIdx = 0;
    isPlaying = false;
    document.getElementById('btnPause').classList.add('active');
    document.getElementById('btnPlay').classList.remove('active');
    renderFrame();
  });

  document.getElementById('timeSlider').addEventListener('input', (e) => {
    const pct = parseFloat(e.target.value) / 1000.0;
    currentFrameIdx = Math.floor(pct * (totalFrames - 1));
    isPlaying = false;
    document.getElementById('btnPause').classList.add('active');
    document.getElementById('btnPlay').classList.remove('active');
    renderFrame();
  });

  document.querySelectorAll('.btn-speed').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.btn-speed').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      playSpeed = parseFloat(btn.getAttribute('data-speed'));
    });
  });
</script>
</body>
</html>
"""
    return template.replace("__PAYLOAD_JSON__", payload_json)
