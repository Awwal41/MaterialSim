"""Voice agent HUD — cinematic animated interface for the Streamlit GUI."""

from __future__ import annotations

import streamlit.components.v1 as components

AGENT_LABEL = "Material Sim Agent"
STATES = ("idle", "listening", "thinking", "speaking")


def render_jarvis_hud(state: str = "idle", status_text: str = "Standing by") -> None:
    """Render the animated voice-agent HUD.

    Args:
        state: One of ``idle``, ``listening``, ``thinking``, ``speaking``.
        status_text: Short status line shown below the orb.
    """
    if state not in STATES:
        state = "idle"

    safe_status = status_text.replace('"', "&quot;").replace("<", "&lt;")
    safe_label = AGENT_LABEL.replace('"', "&quot;")

    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;500;700&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: transparent;
    font-family: 'Exo 2', sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 400px;
    overflow: hidden;
  }}

  .hud-wrap {{
    position: relative;
    width: 340px;
    height: 340px;
    display: flex;
    align-items: center;
    justify-content: center;
  }}

  /* Faint hex grid */
  .hud-wrap::before {{
    content: '';
    position: absolute;
    inset: 0;
    background-image:
      radial-gradient(circle at 50% 50%, rgba(45, 212, 191, 0.07) 0%, transparent 55%),
      linear-gradient(rgba(45, 212, 191, 0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(45, 212, 191, 0.04) 1px, transparent 1px);
    background-size: 100% 100%, 24px 24px, 24px 24px;
    border-radius: 50%;
    animation: grid-pulse 4s ease-in-out infinite;
  }}

  .scene {{
    position: relative;
    width: 280px;
    height: 280px;
    perspective: 900px;
    transform-style: preserve-3d;
  }}

  .scene-inner {{
    position: absolute;
    inset: 0;
    transform-style: preserve-3d;
    animation: scene-drift 12s ease-in-out infinite;
  }}

  /* Corner HUD brackets */
  .bracket {{
    position: absolute;
    width: 28px;
    height: 28px;
    border: 2px solid rgba(45, 212, 191, 0.55);
    z-index: 20;
    animation: bracket-pulse 2.5s ease-in-out infinite;
  }}
  .bracket-tl {{ top: 8px; left: 8px; border-right: none; border-bottom: none; }}
  .bracket-tr {{ top: 8px; right: 8px; border-left: none; border-bottom: none; }}
  .bracket-bl {{ bottom: 8px; left: 8px; border-right: none; border-top: none; }}
  .bracket-br {{ bottom: 8px; right: 8px; border-left: none; border-top: none; }}

  /* 3D orbital rings */
  .orbit {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform-style: preserve-3d;
    border-radius: 50%;
    border: 1.5px solid transparent;
  }}

  .orbit-1 {{
    width: 260px; height: 260px;
    margin: -130px 0 0 -130px;
    border-color: rgba(45, 212, 191, 0.35);
    border-top-color: rgba(94, 234, 212, 0.9);
    border-bottom-color: rgba(45, 212, 191, 0.15);
    animation: orbit-tilt-a 6s linear infinite;
  }}

  .orbit-2 {{
    width: 220px; height: 220px;
    margin: -110px 0 0 -110px;
    border: 1.5px dashed rgba(45, 212, 191, 0.4);
    animation: orbit-tilt-b 4.5s linear infinite reverse;
  }}

  .orbit-3 {{
    width: 180px; height: 180px;
    margin: -90px 0 0 -90px;
    border-color: rgba(94, 234, 212, 0.25);
    border-left-color: rgba(45, 212, 191, 0.8);
    animation: orbit-tilt-c 3.2s linear infinite;
  }}

  .orbit-4 {{
    width: 140px; height: 140px;
    margin: -70px 0 0 -70px;
    border: 2px dotted rgba(45, 212, 191, 0.3);
    animation: orbit-tilt-d 2.8s linear infinite reverse;
  }}

  /* SVG arc sweeps */
  .arc-layer {{
    position: absolute;
    inset: 0;
    z-index: 5;
    pointer-events: none;
  }}
  .arc-layer svg {{
    width: 100%;
    height: 100%;
    animation: arc-spin 8s linear infinite;
  }}
  .arc-layer svg path {{
    fill: none;
    stroke: rgba(45, 212, 191, 0.5);
    stroke-width: 1.5;
    stroke-linecap: round;
    stroke-dasharray: 80 200;
    animation: arc-dash 3s linear infinite;
  }}

  /* Floating particles */
  .particles {{
    position: absolute;
    inset: 0;
    z-index: 4;
    pointer-events: none;
  }}
  .particle {{
    position: absolute;
    width: 3px;
    height: 3px;
    background: #5eead4;
    border-radius: 50%;
    box-shadow: 0 0 6px #2dd4bf;
    animation: particle-float 4s ease-in-out infinite;
  }}
  .particle:nth-child(1)  {{ top: 12%; left: 22%; animation-delay: 0.0s; animation-duration: 3.2s; }}
  .particle:nth-child(2)  {{ top: 28%; left: 78%; animation-delay: 0.4s; animation-duration: 4.1s; }}
  .particle:nth-child(3)  {{ top: 65%; left: 15%; animation-delay: 0.8s; animation-duration: 3.7s; }}
  .particle:nth-child(4)  {{ top: 80%; left: 70%; animation-delay: 1.2s; animation-duration: 4.5s; }}
  .particle:nth-child(5)  {{ top: 45%; left: 8%;  animation-delay: 0.2s; animation-duration: 3.9s; }}
  .particle:nth-child(6)  {{ top: 18%; left: 55%; animation-delay: 1.6s; animation-duration: 3.4s; }}
  .particle:nth-child(7)  {{ top: 72%; left: 48%; animation-delay: 0.6s; animation-duration: 4.8s; }}
  .particle:nth-child(8)  {{ top: 38%; left: 88%; animation-delay: 1.0s; animation-duration: 3.1s; }}

  /* Core orb */
  .core-wrap {{
    position: absolute;
    top: 50%;
    left: 50%;
    width: 88px;
    height: 88px;
    margin: -44px 0 0 -44px;
    z-index: 15;
    transform-style: preserve-3d;
    animation: core-float 5s ease-in-out infinite;
  }}

  .core {{
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background:
      radial-gradient(circle at 32% 28%, #a7f3ec 0%, #2dd4bf 22%, #0d9488 55%, #042f2e 100%);
    box-shadow:
      0 0 20px rgba(45, 212, 191, 0.9),
      0 0 50px rgba(45, 212, 191, 0.45),
      0 0 90px rgba(13, 148, 136, 0.25),
      inset 0 0 24px rgba(255, 255, 255, 0.25);
    animation: core-breathe 2.4s ease-in-out infinite;
  }}

  .core-glow {{
    position: absolute;
    inset: -18px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(45, 212, 191, 0.35) 0%, transparent 70%);
    animation: glow-pulse 2.4s ease-in-out infinite;
    z-index: -1;
  }}

  /* Ripple rings on speak/listen */
  .ripple {{
    position: absolute;
    top: 50%; left: 50%;
    width: 88px; height: 88px;
    margin: -44px 0 0 -44px;
    border-radius: 50%;
    border: 1px solid rgba(45, 212, 191, 0.5);
    z-index: 12;
    opacity: 0;
    animation: ripple-out 2s ease-out infinite;
  }}
  .ripple:nth-child(2) {{ animation-delay: 0.7s; }}
  .ripple:nth-child(3) {{ animation-delay: 1.4s; }}

  /* Scan line */
  .scan {{
    position: absolute;
    left: 50%;
    width: 240px;
    height: 2px;
    margin-left: -120px;
    background: linear-gradient(90deg, transparent, #5eead4, transparent);
    box-shadow: 0 0 12px #2dd4bf;
    z-index: 18;
    animation: scan-sweep 2.8s ease-in-out infinite;
    opacity: 0.6;
  }}

  /* Waveform ring */
  .wave-ring {{
    position: absolute;
    bottom: 36px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 4px;
    align-items: flex-end;
    height: 36px;
    opacity: 0;
    z-index: 22;
  }}
  .wave-bar {{
    width: 3px;
    border-radius: 2px;
    background: linear-gradient(to top, #0d9488, #5eead4);
    animation: wave-bar 0.55s ease-in-out infinite;
  }}
  .wave-bar:nth-child(1) {{ animation-delay: 0.00s; }}
  .wave-bar:nth-child(2) {{ animation-delay: 0.08s; }}
  .wave-bar:nth-child(3) {{ animation-delay: 0.16s; }}
  .wave-bar:nth-child(4) {{ animation-delay: 0.10s; }}
  .wave-bar:nth-child(5) {{ animation-delay: 0.22s; }}
  .wave-bar:nth-child(6) {{ animation-delay: 0.05s; }}
  .wave-bar:nth-child(7) {{ animation-delay: 0.18s; }}
  .wave-bar:nth-child(8) {{ animation-delay: 0.12s; }}
  .wave-bar:nth-child(9) {{ animation-delay: 0.25s; }}

  /* State modifiers */
  .state-listening .orbit-1 {{ animation-duration: 3s; border-top-color: rgba(52, 211, 153, 1); }}
  .state-listening .core {{
    background: radial-gradient(circle at 32% 28%, #bbf7d0 0%, #34d399 22%, #059669 55%, #022c22 100%);
    box-shadow: 0 0 30px rgba(52, 211, 153, 0.9), 0 0 70px rgba(52, 211, 153, 0.4);
    animation: core-breathe 0.7s ease-in-out infinite;
  }}
  .state-listening .ripple {{ opacity: 1; }}
  .state-listening .wave-ring {{ opacity: 1; }}

  .state-thinking .scene-inner {{ animation-duration: 4s; }}
  .state-thinking .orbit-1 {{ animation-duration: 1.8s; }}
  .state-thinking .orbit-2 {{ animation-duration: 1.4s; }}
  .state-thinking .orbit-3 {{ animation-duration: 1.1s; }}
  .state-thinking .core {{
    background: radial-gradient(circle at 32% 28%, #fde68a 0%, #f59e0b 25%, #b45309 60%, #451a03 100%);
    box-shadow: 0 0 30px rgba(245, 158, 11, 0.85), 0 0 70px rgba(245, 158, 11, 0.35);
    animation: core-think 1s ease-in-out infinite;
  }}

  .state-speaking .orbit-1 {{ animation-duration: 2.5s; }}
  .state-speaking .core {{
    animation: core-speak 0.45s ease-in-out infinite;
    box-shadow: 0 0 40px rgba(45, 212, 191, 1), 0 0 80px rgba(45, 212, 191, 0.5);
  }}
  .state-speaking .ripple {{ opacity: 1; animation-duration: 1.2s; }}
  .state-speaking .wave-ring {{ opacity: 1; }}

  /* Labels */
  .agent-label {{
    margin-top: 14px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: rgba(45, 212, 191, 0.75);
    text-align: center;
  }}

  .status {{
    margin-top: 6px;
    color: #5eead4;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    text-align: center;
    text-shadow: 0 0 14px rgba(45, 212, 191, 0.55);
    animation: status-glow 2s ease-in-out infinite;
  }}

  @keyframes grid-pulse {{
    0%, 100% {{ opacity: 0.6; }}
    50% {{ opacity: 1; }}
  }}

  @keyframes scene-drift {{
    0%, 100% {{ transform: rotateY(0deg) rotateX(4deg); }}
    25% {{ transform: rotateY(6deg) rotateX(2deg); }}
    50% {{ transform: rotateY(0deg) rotateX(-3deg); }}
    75% {{ transform: rotateY(-6deg) rotateX(2deg); }}
  }}

  @keyframes orbit-tilt-a {{
    from {{ transform: rotateX(72deg) rotateZ(0deg); }}
    to   {{ transform: rotateX(72deg) rotateZ(360deg); }}
  }}
  @keyframes orbit-tilt-b {{
    from {{ transform: rotateX(58deg) rotateY(24deg) rotateZ(0deg); }}
    to   {{ transform: rotateX(58deg) rotateY(24deg) rotateZ(-360deg); }}
  }}
  @keyframes orbit-tilt-c {{
    from {{ transform: rotateX(48deg) rotateY(-18deg) rotateZ(0deg); }}
    to   {{ transform: rotateX(48deg) rotateY(-18deg) rotateZ(360deg); }}
  }}
  @keyframes orbit-tilt-d {{
    from {{ transform: rotateX(82deg) rotateZ(0deg); }}
    to   {{ transform: rotateX(82deg) rotateZ(-360deg); }}
  }}

  @keyframes arc-spin {{
    from {{ transform: rotate(0deg); }}
    to   {{ transform: rotate(360deg); }}
  }}
  @keyframes arc-dash {{
    to {{ stroke-dashoffset: -280; }}
  }}

  @keyframes particle-float {{
    0%, 100% {{ transform: translate(0, 0) scale(1); opacity: 0.4; }}
    25% {{ transform: translate(6px, -10px) scale(1.3); opacity: 1; }}
    50% {{ transform: translate(-4px, 6px) scale(0.8); opacity: 0.6; }}
    75% {{ transform: translate(8px, 4px) scale(1.1); opacity: 0.9; }}
  }}

  @keyframes core-float {{
    0%, 100% {{ transform: translateY(0px) translateZ(0px); }}
    50% {{ transform: translateY(-5px) translateZ(8px); }}
  }}
  @keyframes core-breathe {{
    0%, 100% {{ transform: scale(1); }}
    50% {{ transform: scale(1.06); }}
  }}
  @keyframes core-think {{
    0%, 100% {{ transform: scale(1) rotate(0deg); }}
    50% {{ transform: scale(1.1) rotate(8deg); }}
  }}
  @keyframes core-speak {{
    0%, 100% {{ transform: scale(1); }}
    25% {{ transform: scale(1.14); }}
    75% {{ transform: scale(0.94); }}
  }}
  @keyframes glow-pulse {{
    0%, 100% {{ opacity: 0.5; transform: scale(1); }}
    50% {{ opacity: 1; transform: scale(1.15); }}
  }}
  @keyframes ripple-out {{
    0% {{ transform: scale(1); opacity: 0.7; }}
    100% {{ transform: scale(2.8); opacity: 0; }}
  }}
  @keyframes scan-sweep {{
    0% {{ top: 18%; opacity: 0; }}
    15% {{ opacity: 0.8; }}
    85% {{ opacity: 0.8; }}
    100% {{ top: 82%; opacity: 0; }}
  }}
  @keyframes wave-bar {{
    0%, 100% {{ height: 6px; opacity: 0.5; }}
    50% {{ height: 28px; opacity: 1; }}
  }}
  @keyframes bracket-pulse {{
    0%, 100% {{ opacity: 0.45; }}
    50% {{ opacity: 1; }}
  }}
  @keyframes status-glow {{
    0%, 100% {{ opacity: 0.85; }}
    50% {{ opacity: 1; }}
  }}
</style>
</head>
<body>
  <div class="hud-wrap state-{state}">
    <div class="bracket bracket-tl"></div>
    <div class="bracket bracket-tr"></div>
    <div class="bracket bracket-bl"></div>
    <div class="bracket bracket-br"></div>

    <div class="scene">
      <div class="scene-inner">
        <div class="orbit orbit-1"></div>
        <div class="orbit orbit-2"></div>
        <div class="orbit orbit-3"></div>
        <div class="orbit orbit-4"></div>

        <div class="arc-layer">
          <svg viewBox="0 0 280 280">
            <path d="M 140 20 A 120 120 0 0 1 250 140" />
            <path d="M 250 140 A 120 120 0 0 1 140 260" />
            <path d="M 30 140 A 120 120 0 0 1 140 20" />
          </svg>
        </div>

        <div class="particles">
          <div class="particle"></div><div class="particle"></div>
          <div class="particle"></div><div class="particle"></div>
          <div class="particle"></div><div class="particle"></div>
          <div class="particle"></div><div class="particle"></div>
        </div>

        <div class="ripple"></div>
        <div class="ripple"></div>
        <div class="ripple"></div>

        <div class="core-wrap">
          <div class="core-glow"></div>
          <div class="core"></div>
        </div>
      </div>
    </div>

    <div class="scan"></div>

    <div class="wave-ring">
      <div class="wave-bar"></div><div class="wave-bar"></div>
      <div class="wave-bar"></div><div class="wave-bar"></div>
      <div class="wave-bar"></div><div class="wave-bar"></div>
      <div class="wave-bar"></div><div class="wave-bar"></div>
      <div class="wave-bar"></div>
    </div>
  </div>

  <div class="agent-label">{safe_label}</div>
  <div class="status">{safe_status}</div>
</body>
</html>
"""
    components.html(html, height=440, scrolling=False)


def jarvis_css() -> str:
    """Extra CSS for the voice agent layout."""
    return """
<style>
  .voice-hero {
    background: linear-gradient(160deg, #0f1419 0%, #132030 45%, #0f1a24 100%);
    border: 1px solid rgba(45, 212, 191, 0.22);
    border-radius: 16px;
    padding: 0.5rem 1rem 0.25rem;
    margin-bottom: 1.25rem;
    box-shadow:
      0 0 40px rgba(45, 212, 191, 0.06),
      inset 0 1px 0 rgba(255, 255, 255, 0.04);
    display: flex;
    justify-content: center;
  }
  .voice-controls-card {
    background: rgba(26, 35, 50, 0.85);
    border: 1px solid rgba(45, 212, 191, 0.15);
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    backdrop-filter: blur(8px);
  }
</style>
"""
