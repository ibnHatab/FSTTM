import React, { useState, useEffect, useRef } from 'react';

// Premium high-fidelity cartoon and modern fonts combined
const BunnyStyles = () => (
  <style>{`
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@300;400;600;700&family=Fira+Code:wght@400;500;700&family=Orbitron:wght@500;800;900&display=swap');
    
    body {
      font-family: 'Fredoka', sans-serif;
      margin: 0;
      padding: 0;
      background: #0d0a1b;
      color: #f3f4f6;
      overflow-x: hidden;
    }

    .font-hud {
      font-family: 'Fredoka', sans-serif;
    }

    .font-code {
      font-family: 'Fira Code', 'Courier New', monospace;
    }

    /* Cosmic bunny starry grid */
    .bunny-grid {
      background-size: 30px 30px;
      background-image: radial-gradient(circle, rgba(244, 114, 182, 0.08) 1.5px, transparent 1.5px);
    }

    /* Playful bouncy scale transitions */
    .bouncy-btn {
      transition: all 0.15s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .bouncy-btn:hover {
      transform: scale(1.05);
    }
    .bouncy-btn:active {
      transform: scale(0.93);
    }

    /* Neon playful shadows */
    .glow-pink {
      box-shadow: 0 0 20px rgba(244, 114, 182, 0.35);
    }
    .glow-cyan {
      box-shadow: 0 0 20px rgba(34, 211, 238, 0.35);
    }
    .glow-yellow {
      box-shadow: 0 0 20px rgba(250, 204, 21, 0.3);
    }

    /* Custom scrollbar for diagnostic debug window */
    .custom-scroll::-webkit-scrollbar {
      width: 6px;
    }
    .custom-scroll::-webkit-scrollbar-track {
      background: rgba(20, 16, 43, 0.6);
    }
    .custom-scroll::-webkit-scrollbar-thumb {
      background: rgba(244, 114, 182, 0.4);
      border-radius: 6px;
    }
    .custom-scroll::-webkit-scrollbar-thumb:hover {
      background: rgba(244, 114, 182, 0.7);
    }
  `}</style>
);

class BunnyAudioEngine {
  constructor() {
    this.audioCtx = null;
    this.muted = false;
  }

  init() {
    if (!this.audioCtx) {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
  }

  // Classic cartoon "Boing" sound
  playBoing() {
    if (this.muted) return;
    this.init();
    const t = this.audioCtx.currentTime;
    const osc = this.audioCtx.createOscillator();
    const gain = this.audioCtx.createGain();
    
    osc.connect(gain);
    gain.connect(this.audioCtx.destination);
    
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(150, t);
    // Sweeps up and down rapidly to generate a bouncy elastic pitch
    osc.frequency.exponentialRampToValueAtTime(600, t + 0.15);
    osc.frequency.exponentialRampToValueAtTime(200, t + 0.3);
    
    gain.gain.setValueAtTime(0.15, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.32);
    
    osc.start(t);
    osc.stop(t + 0.32);
  }

  // High-pitched squeak for door locks and latches
  playSqueak() {
    if (this.muted) return;
    this.init();
    const t = this.audioCtx.currentTime;
    const osc = this.audioCtx.createOscillator();
    const gain = this.audioCtx.createGain();
    
    osc.connect(gain);
    gain.connect(this.audioCtx.destination);
    
    osc.type = 'sine';
    osc.frequency.setValueAtTime(900, t);
    osc.frequency.linearRampToValueAtTime(1300, t + 0.08);
    osc.frequency.linearRampToValueAtTime(800, t + 0.16);
    
    gain.gain.setValueAtTime(0.1, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.18);
    
    osc.start(t);
    osc.stop(t + 0.18);
  }

  // Satisfying bubble pop for AC toggle and fans
  playPop() {
    if (this.muted) return;
    this.init();
    const t = this.audioCtx.currentTime;
    const osc = this.audioCtx.createOscillator();
    const gain = this.audioCtx.createGain();
    
    osc.connect(gain);
    gain.connect(this.audioCtx.destination);
    
    osc.type = 'sine';
    osc.frequency.setValueAtTime(600, t);
    osc.frequency.exponentialRampToValueAtTime(100, t + 0.05);
    
    gain.gain.setValueAtTime(0.12, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.06);
    
    osc.start(t);
    osc.stop(t + 0.06);
  }

  // Squeaky air-puff whoopee-fart sound for seat adjustment clicks
  playPuff() {
    if (this.muted) return;
    this.init();
    const t = this.audioCtx.currentTime;
    const osc1 = this.audioCtx.createOscillator();
    const osc2 = this.audioCtx.createOscillator();
    const filter = this.audioCtx.createBiquadFilter();
    const gain = this.audioCtx.createGain();

    osc1.connect(filter);
    osc2.connect(filter);
    filter.connect(gain);
    gain.connect(this.audioCtx.destination);

    osc1.type = 'sawtooth';
    osc2.type = 'square';
    
    osc1.frequency.setValueAtTime(120, t);
    osc2.frequency.setValueAtTime(118, t);
    osc1.frequency.linearRampToValueAtTime(50, t + 0.2);
    osc2.frequency.linearRampToValueAtTime(48, t + 0.2);

    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(400, t);
    filter.frequency.exponentialRampToValueAtTime(120, t + 0.2);

    gain.gain.setValueAtTime(0.2, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.22);

    osc1.start(t);
    osc2.start(t);
    osc1.stop(t + 0.22);
    osc2.stop(t + 0.22);
  }
}

const sounds = new BunnyAudioEngine();

// Constants matching the vhal.py backend definition
const AREA_ROW_1_LEFT = 1;
const AREA_ROW_1_RIGHT = 4;
const AREA_ROW_2_LEFT = 16;
const AREA_ROW_2_RIGHT = 64;

export default function App() {
  // Default to the page's own host:port so traffic flows through the Vite dev
  // proxy (vite.config.js forwards /ws, /command, /state to the FastAPI backend
  // on 127.0.0.1:8000). This keeps the UI reachable from another machine even
  // though the backend only binds localhost. Editable via the ECU ADDR field
  // if you want to point at a backend host directly.
  const [backendUrl, setBackendUrl] = useState(() => {
    return window.location.host || "localhost:5173";
  });

  const [vhalProps, setVhalProps] = useState({
    HVAC_TEMPERATURE_SET: { "1": 22.0, "4": 22.0 },
    HVAC_TEMPERATURE_CURRENT: { "1": 22.0, "4": 22.0 },
    HVAC_FAN_SPEED: { "1": 1, "4": 1 },
    HVAC_FAN_DIRECTION: { "1": 1, "4": 1 },
    HVAC_AUTO_ON: { "1": false, "4": false },
    HVAC_AC_ON: { "0": false },
    HVAC_RECIRC_ON: { "0": false },
    HVAC_MAX_AC_ON: { "0": false },
    HVAC_DUAL_ON: { "0": false },
    HVAC_MAX_DEFROST_ON: { "0": false },
    HVAC_DEFROSTER: { "1": false },
    HVAC_POWER_ON: { "0": false },
    HVAC_EXPERT_MODE: { "0": false },
    ENV_OUTSIDE_TEMPERATURE: { "0": 18.0 },
    DOOR_LOCK: { "1": true, "4": true, "16": true, "64": true, "512": true },
    DOOR_MOVE: { "1": 0, "4": 0, "16": 0, "64": 0 },
    HEADLIGHTS_SWITCH: { "0": false },
    HAZARD_LIGHTS_SWITCH: { "0": false },
    FOG_LIGHTS_SWITCH: { "0": false },
    CABIN_LIGHTS_SWITCH: { "0": false },
    HVAC_SEAT_TEMPERATURE: { "1": 0, "4": 0 },
    HVAC_SEAT_VENTILATION: { "1": 0, "4": 0 },
    WINDOW_LOCK: { "1": false, "4": false, "16": false, "64": false },
    WINDOW_MOVE: { "1": 0, "4": 0, "16": 0, "64": 0 }
  });

  const [wsStatus, setWsStatus] = useState("DISCONNECTED");
  const [telemetryLogs, setTelemetryLogs] = useState([]);
  const [isMuted, setIsMuted] = useState(false);

  const wsRef = useRef(null);

  useEffect(() => {
    sounds.muted = isMuted;
  }, [isMuted]);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, [backendUrl]);

  const connectWebSocket = () => {
    setWsStatus("CONNECTING");
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${wsScheme}://${backendUrl}/ws`);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsStatus("CONNECTED");
        sounds.playBoing();
        addLog("SYSTEM", "Bunny Telemetry protocol initialized on WebSocket interface.");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          addLog("RX", data);
          if (data.type === "state" && data.props) {
            setVhalProps(data.props);
          } else if (data.type === "delta" && data.changes) {
            setVhalProps(prev => {
              const updated = { ...prev };
              data.changes.forEach(change => {
                if (!updated[change.name]) {
                  updated[change.name] = {};
                }
                updated[change.name][String(change.area)] = change.value;
              });
              return updated;
            });
          }
        } catch (e) {
          console.error("Failed parsing VHAL data stream.", e);
        }
      };

      ws.onerror = () => {
        setWsStatus("ERROR");
      };

      ws.onclose = () => {
        setWsStatus("DISCONNECTED");
        addLog("SYSTEM", "WebSocket severed. Trying reconnection sequence in 5s...");
        setTimeout(() => {
          if (wsRef.current && wsRef.current.readyState === WebSocket.CLOSED) {
            connectWebSocket();
          }
        }, 5000);
      };
    } catch (err) {
      setWsStatus("DISCONNECTED");
    }
  };

  const addLog = (direction, payload) => {
    const timestamp = new Date().toISOString().slice(11, 23);
    setTelemetryLogs(prev => [
      { id: Math.random(), time: timestamp, dir: direction, content: JSON.stringify(payload) },
      ...prev.slice(0, 39)
    ]);
  };

  const sendCommand = async (payload) => {
    addLog("TX", payload);

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    } else {
      try {
        const response = await fetch(`${window.location.protocol}//${backendUrl}/command`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const resData = await response.json();
        if (resData.props) {
          setVhalProps(resData.props);
        }
      } catch (err) {
        console.error("Direct REST transmission failure:", err);
      }
    }
  };

  const triggerPowerToggle = () => {
    sounds.playBoing();
    sendCommand({ cmd: "action", action: "power_toggle" });
  };

  const changeTemperature = (area, increase) => {
    sounds.playPop();
    sendCommand({
      cmd: "action",
      action: "bump_temperature",
      args: { up: increase, area: Number(area) }
    });
  };

  const changeFanSpeed = (area, increase) => {
    sounds.playPop();
    sendCommand({
      cmd: "action",
      action: "bump_fan_speed",
      args: { up: increase, area: Number(area) }
    });
  };

  const cycleFanDirection = (area, directionBit) => {
    sounds.playBoing();
    sendCommand({
      cmd: "action",
      action: "fan_direction_toggle",
      args: { direction: directionBit, area: Number(area) }
    });
  };

  const handleSeatHeat = (area, isUp) => {
    sounds.playPuff();
    sendCommand({
      cmd: "action",
      action: "bump_seat_temp",
      args: { up: isUp, area: Number(area) }
    });
  };

  const handleSeatVent = (area, isUp) => {
    sounds.playPuff();
    sendCommand({
      cmd: "action",
      action: "bump_seat_vent",
      args: { up: isUp, area: Number(area) }
    });
  };

  const toggleDoorLock = (area, currentLocked) => {
    sounds.playSqueak();
    sendCommand({
      cmd: "action",
      action: "door_lock",
      args: { area: Number(area), locked: !currentLocked }
    });
  };

  const toggleDoorMove = (area, currentOpen) => {
    sounds.playSqueak();
    sendCommand({
      cmd: "action",
      action: "door_move",
      args: { area: Number(area), open: !currentOpen }
    });
  };

  const handleWindowSlide = (area, position) => {
    sounds.playPop();
    sendCommand({
      cmd: "action",
      action: "window_move",
      args: { area: Number(area), position: Number(position) }
    });
  };

  const toggleWindowLock = (area, currentLocked) => {
    sounds.playSqueak();
    sendCommand({
      cmd: "action",
      action: "window_lock",
      args: { area: Number(area), locked: !currentLocked }
    });
  };

  const triggerAutoToggle = (area) => {
    sounds.playBoing();
    sendCommand({
      cmd: "action",
      action: "auto_toggle",
      args: { area: Number(area) }
    });
  };

  const toggleLightSwitch = (actionName) => {
    sounds.playBoing();
    sendCommand({ cmd: "action", action: actionName });
  };

  const getPropVal = (name, area = "0", fallback = null) => {
    if (vhalProps[name] && vhalProps[name][String(area)] !== undefined) {
      return vhalProps[name][String(area)];
    }
    if (vhalProps[name]) {
      const keys = Object.keys(vhalProps[name]);
      if (keys.length > 0) return vhalProps[name][keys[0]];
    }
    return fallback;
  };

  const isPowerOn = getPropVal("HVAC_POWER_ON", "0", false);

  return (
    <div className="relative min-h-screen w-full flex flex-col items-center justify-start p-4 bg-[#0d0a1b] text-slate-100 select-none overflow-x-hidden bunny-grid">
      <BunnyStyles />

      {/* HEADER SECTION (COMMUNICATION BANNER) */}
      <div className="w-full max-w-6xl flex flex-wrap items-center justify-between bg-[#191433]/90 border-2 border-pink-500/20 p-4 rounded-3xl mb-5 z-10 shadow-xl backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-pink-500/10 border-2 border-pink-400 font-hud text-2xl">
            🐰
          </div>
          <div>
            <h1 className="text-xl md:text-2xl font-hud font-bold tracking-wide text-pink-400 flex items-center gap-2">
              BUNNY <span className="text-cyan-400">COCKPIT CONTROLLER</span>
            </h1>
            <p className="text-[10px] text-slate-400 font-code tracking-wider">
              Real CAN/VHAL Bus Connector - Professional Protocol
            </p>
          </div>
        </div>

        {/* CONNECTION & HOST STATUSES */}
        <div className="flex flex-wrap items-center gap-3 mt-3 lg:mt-0">
          <div className="flex items-center bg-black/40 border border-[#30275c] rounded-2xl px-3 py-1.5 gap-2">
            <span className="text-[10px] font-code text-pink-400">ECU ADDR:</span>
            <input
              type="text"
              value={backendUrl}
              onChange={(e) => setBackendUrl(e.target.value)}
              className="bg-transparent text-xs text-slate-200 border-none outline-none focus:ring-0 w-32 font-code"
              placeholder="localhost:5173"
            />
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 bg-black/40 border border-[#30275c] rounded-2xl">
            <span className={`w-2.5 h-2.5 rounded-full ${
              wsStatus === "CONNECTED" ? "bg-emerald-400 animate-pulse glow-cyan" : 
              wsStatus === "CONNECTING" ? "bg-yellow-400 animate-pulse" : "bg-red-500"
            }`} />
            <span className="text-[10px] font-code font-bold text-slate-300">{wsStatus}</span>
          </div>

          <button
            onClick={() => setIsMuted(!isMuted)}
            className={`px-3 py-1.5 bouncy-btn rounded-2xl border-2 font-hud text-xs font-semibold transition-all ${
              isMuted ? 'bg-red-500/20 border-red-500/50 text-red-400' : 'bg-pink-500/20 border-pink-400 text-pink-300'
            }`}
          >
            {isMuted ? '🔇 MUTED' : '🔊 LIVE SOUNDS'}
          </button>
        </div>
      </div>

      {/* MAIN TWO-COLUMN CONTAINER */}
      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-5 z-10 mb-20">
        
        {/* LEFT COLUMN: PRIMARY DUAL-ZONE HVAC CONTROLS */}
        <div className="lg:col-span-8 flex flex-col gap-5">
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            
            {/* ZONE 1: DRIVER CLIMATE MODULE */}
            <div className={`bg-[#191433]/90 border-2 border-[#ff70a6]/20 p-5 rounded-3xl flex flex-col justify-between transition-all shadow-lg relative overflow-hidden ${isPowerOn ? 'opacity-100' : 'opacity-40 pointer-events-none'}`}>
              <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-2">
                  <span className="text-lg">👒</span>
                  <span className="text-sm font-hud font-bold tracking-wide text-[#ff70a6]">DRIVER HVAC MODULE</span>
                </div>
                <span className="text-[10px] font-code bg-black/40 px-2 py-0.5 rounded border border-[#ff70a6]/20 text-[#ff70a6]/80">
                  Seat: Row 1 L (0x01)
                </span>
              </div>

              {/* TEMPERATURE DEMAND */}
              <div className="bg-[#120d29] p-4 rounded-2xl border border-[#352968] mb-4 flex items-center justify-between">
                <div>
                  <span className="text-[9px] font-hud text-slate-400 block tracking-widest uppercase">TEMPERATURE SET</span>
                  <span className="text-3xl font-hud font-bold text-white">
                    {Number(getPropVal("HVAC_TEMPERATURE_SET", AREA_ROW_1_LEFT, 22.0)).toFixed(1)}°C
                  </span>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => changeTemperature(AREA_ROW_1_LEFT, false)}
                    className="w-11 h-11 rounded-2xl bg-black/40 border-2 border-pink-400 bouncy-btn flex items-center justify-center font-bold text-xl text-pink-300"
                  >
                    ❄️
                  </button>
                  <button
                    onClick={() => changeTemperature(AREA_ROW_1_LEFT, true)}
                    className="w-11 h-11 rounded-2xl bg-black/40 border-2 border-pink-400 bouncy-btn flex items-center justify-center font-bold text-xl text-pink-300"
                  >
                    🔥
                  </button>
                </div>
              </div>

              {/* FAN SPEED DISPLAY */}
              <div className="bg-[#120d29] p-4 rounded-2xl border border-[#352968] mb-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-[9px] font-hud text-slate-400 tracking-widest uppercase">FAN BLOWER SPEED</span>
                  <span className="text-xs font-hud font-bold text-pink-300">
                    LEVEL {getPropVal("HVAC_FAN_SPEED", AREA_ROW_1_LEFT, 1)} / 7
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => changeFanSpeed(AREA_ROW_1_LEFT, false)}
                    className="w-9 h-9 bouncy-btn rounded-xl bg-black/40 border border-pink-400 flex items-center justify-center font-bold text-sm"
                  >
                    -
                  </button>
                  <div className="flex-1 bg-black/60 h-4 rounded-full overflow-hidden flex p-0.5 gap-0.5 border border-[#352968]">
                    {Array.from({ length: 7 }).map((_, i) => (
                      <div
                        key={i}
                        className={`flex-1 h-full rounded-full transition-all ${
                          i < getPropVal("HVAC_FAN_SPEED", AREA_ROW_1_LEFT, 1)
                            ? 'bg-pink-400' : 'bg-slate-900'
                        }`}
                      />
                    ))}
                  </div>
                  <button
                    onClick={() => changeFanSpeed(AREA_ROW_1_LEFT, true)}
                    className="w-9 h-9 bouncy-btn rounded-xl bg-black/40 border border-pink-400 flex items-center justify-center font-bold text-sm"
                  >
                    +
                  </button>
                </div>
              </div>

              {/* AIR DIRECTION SWITCHBOARD */}
              <div className="grid grid-cols-3 gap-2">
                {[
                  { bit: 1, label: '🌬️ FACE', name: 'FACE' },
                  { bit: 2, label: '👣 FLOOR', name: 'FLOOR' },
                  { bit: 4, label: '🌫️ DEFROST', name: 'DEFROST' }
                ].map(mode => {
                  const bitmask = getPropVal("HVAC_FAN_DIRECTION", AREA_ROW_1_LEFT, 1);
                  const active = (bitmask & mode.bit) !== 0;
                  return (
                    <button
                      key={mode.bit}
                      onClick={() => cycleFanDirection(AREA_ROW_1_LEFT, mode.bit)}
                      className={`py-2 px-1 rounded-2xl border-2 bouncy-btn text-[10px] font-hud font-bold transition-all ${
                        active
                          ? 'bg-pink-400/20 border-pink-400 text-pink-300 shadow-lg'
                          : 'bg-black/30 border-[#352968] text-slate-400'
                      }`}
                    >
                      {mode.label}
                    </button>
                  );
                })}
              </div>

              {/* AUTO COMFORT BUTTON */}
              <button
                onClick={() => triggerAutoToggle(AREA_ROW_1_LEFT)}
                className={`mt-4 py-2.5 w-full rounded-2xl border-2 bouncy-btn text-xs font-hud font-bold transition-all ${
                  getPropVal("HVAC_AUTO_ON", AREA_ROW_1_LEFT, false)
                    ? 'bg-emerald-500/20 border-emerald-400 text-emerald-300 font-bold glow-cyan'
                    : 'bg-black/30 border-pink-400/40 text-pink-400/80'
                }`}
              >
                🐰 AUTO MODE (DRIVER)
              </button>

            </div>

            {/* ZONE 2: PASSENGER CLIMATE MODULE */}
            <div className={`bg-[#191433]/90 border-2 border-[#70e4ff]/20 p-5 rounded-3xl flex flex-col justify-between transition-all shadow-lg relative overflow-hidden ${isPowerOn ? 'opacity-100' : 'opacity-40 pointer-events-none'}`}>
              <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-2">
                  <span className="text-lg">👑</span>
                  <span className="text-sm font-hud font-bold tracking-wide text-[#70e4ff]">PASSENGER HVAC MODULE</span>
                </div>
                <span className="text-[10px] font-code bg-black/40 px-2 py-0.5 rounded border border-[#70e4ff]/20 text-[#70e4ff]/80">
                  Seat: Row 1 R (0x04)
                </span>
              </div>

              {/* TEMPERATURE DEMAND */}
              <div className="bg-[#120d29] p-4 rounded-2xl border border-[#352968] mb-4 flex items-center justify-between">
                <div>
                  <span className="text-[9px] font-hud text-slate-400 block tracking-widest uppercase">TEMPERATURE SET</span>
                  <span className="text-3xl font-hud font-bold text-white">
                    {Number(getPropVal("HVAC_TEMPERATURE_SET", AREA_ROW_1_RIGHT, 22.0)).toFixed(1)}°C
                  </span>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => changeTemperature(AREA_ROW_1_RIGHT, false)}
                    className="w-11 h-11 rounded-2xl bg-black/40 border-2 border-cyan-400 bouncy-btn flex items-center justify-center font-bold text-xl text-cyan-300"
                  >
                    ❄️
                  </button>
                  <button
                    onClick={() => changeTemperature(AREA_ROW_1_RIGHT, true)}
                    className="w-11 h-11 rounded-2xl bg-black/40 border-2 border-cyan-400 bouncy-btn flex items-center justify-center font-bold text-xl text-cyan-300"
                  >
                    🔥
                  </button>
                </div>
              </div>

              {/* FAN SPEED DISPLAY */}
              <div className="bg-[#120d29] p-4 rounded-2xl border border-[#352968] mb-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-[9px] font-hud text-slate-400 tracking-widest uppercase">FAN BLOWER SPEED</span>
                  <span className="text-xs font-hud font-bold text-cyan-300">
                    LEVEL {getPropVal("HVAC_FAN_SPEED", AREA_ROW_1_RIGHT, 1)} / 7
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => changeFanSpeed(AREA_ROW_1_RIGHT, false)}
                    className="w-9 h-9 bouncy-btn rounded-xl bg-black/40 border border-cyan-400 flex items-center justify-center font-bold text-sm"
                  >
                    -
                  </button>
                  <div className="flex-1 bg-black/60 h-4 rounded-full overflow-hidden flex p-0.5 gap-0.5 border border-[#352968]">
                    {Array.from({ length: 7 }).map((_, i) => (
                      <div
                        key={i}
                        className={`flex-1 h-full rounded-full transition-all ${
                          i < getPropVal("HVAC_FAN_SPEED", AREA_ROW_1_RIGHT, 1)
                            ? 'bg-cyan-400' : 'bg-slate-900'
                        }`}
                      />
                    ))}
                  </div>
                  <button
                    onClick={() => changeFanSpeed(AREA_ROW_1_RIGHT, true)}
                    className="w-9 h-9 bouncy-btn rounded-xl bg-black/40 border border-cyan-400 flex items-center justify-center font-bold text-sm"
                  >
                    +
                  </button>
                </div>
              </div>

              {/* AIR DIRECTION SWITCHBOARD */}
              <div className="grid grid-cols-3 gap-2">
                {[
                  { bit: 1, label: '🌬️ FACE', name: 'FACE' },
                  { bit: 2, label: '👣 FLOOR', name: 'FLOOR' },
                  { bit: 4, label: '🌫️ DEFROST', name: 'DEFROST' }
                ].map(mode => {
                  const bitmask = getPropVal("HVAC_FAN_DIRECTION", AREA_ROW_1_RIGHT, 1);
                  const active = (bitmask & mode.bit) !== 0;
                  return (
                    <button
                      key={mode.bit}
                      onClick={() => cycleFanDirection(AREA_ROW_1_RIGHT, mode.bit)}
                      className={`py-2 px-1 rounded-2xl border-2 bouncy-btn text-[10px] font-hud font-bold transition-all ${
                        active
                          ? 'bg-cyan-400/20 border-cyan-400 text-cyan-300 shadow-lg'
                          : 'bg-black/30 border-[#352968] text-slate-400'
                      }`}
                    >
                      {mode.label}
                    </button>
                  );
                })}
              </div>

              {/* AUTO COMFORT BUTTON */}
              <button
                onClick={() => triggerAutoToggle(AREA_ROW_1_RIGHT)}
                className={`mt-4 py-2.5 w-full rounded-2xl border-2 bouncy-btn text-xs font-hud font-bold transition-all ${
                  getPropVal("HVAC_AUTO_ON", AREA_ROW_1_RIGHT, false)
                    ? 'bg-emerald-500/20 border-emerald-400 text-emerald-300 font-bold glow-cyan'
                    : 'bg-black/30 border-cyan-400/40 text-cyan-400/80'
                }`}
              >
                🐰 AUTO MODE (PASSENGER)
              </button>

            </div>

          </div>

          <div className="bg-[#191433]/90 border-2 border-pink-500/10 p-5 rounded-3xl shadow-lg">
            <span className="text-[11px] font-hud text-[#ff70a6] block mb-3 uppercase tracking-wider font-semibold">
              GLOBAL CONTROLS (VEHICLE AREA 0)
            </span>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
              {[
                { name: 'HVAC_AC_ON', act: 'ac_toggle', label: '❄️ AC CHILL' },
                { name: 'HVAC_MAX_AC_ON', act: 'ac_max_toggle', label: '🧊 MAX COOL' },
                { name: 'HVAC_RECIRC_ON', act: 'recirc_toggle', label: '🔄 RECIRC' },
                { name: 'HVAC_DUAL_ON', act: 'dual_toggle', label: '👥 DUAL SYNC' },
                { name: 'HVAC_MAX_DEFROST_ON', act: 'max_defrost_toggle', label: '💨 DEFROST' },
                { name: 'HVAC_DEFROSTER', act: 'window_defrost_toggle', label: '🐇 GLASS DRY' }
              ].map(ctrl => {
                const isActive = getPropVal(ctrl.name, "0", false);
                return (
                  <button
                    key={ctrl.name}
                    disabled={!isPowerOn}
                    onClick={() => sendCommand({ cmd: "action", action: ctrl.act })}
                    className={`p-3 rounded-2xl border-2 text-center flex flex-col items-center justify-center gap-1.5 bouncy-btn transition-all ${
                      !isPowerOn ? 'bg-slate-900/40 border-[#29224d] text-slate-600' :
                      isActive 
                        ? 'bg-pink-400/20 border-pink-400 text-pink-300 font-bold glow-pink' 
                        : 'bg-black/40 border-[#30275c] text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <span className="text-[10px] font-hud leading-tight">{ctrl.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* FRONT COMFORT SEAT WIDGET */}
          <div className="bg-[#191433]/90 border-2 border-cyan-500/10 p-5 rounded-3xl shadow-lg">
            <span className="text-[11px] font-hud text-cyan-400 block mb-3 uppercase tracking-wider font-semibold">
              CARROT SEAT COMFORTS (HEATER / VENTILATION)
            </span>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* DRIVER SEAT */}
              <div className="bg-[#120d29] p-4 rounded-2xl border border-[#352968] flex items-center justify-between">
                <div>
                  <span className="text-xs font-hud font-bold text-pink-300">🥕 DRIVER COMFORT</span>
                  <span className="text-[10px] text-slate-400 block">Row 1 Left Side</span>
                </div>
                <div className="flex gap-2">
                  <div className="flex flex-col items-center justify-center bg-black/40 p-1 rounded-xl border border-pink-500/10">
                    <span className="text-[8px] text-pink-400 font-bold uppercase mb-1">🔥 WARM</span>
                    <div className="flex items-center gap-1">
                      <button onClick={() => handleSeatHeat(AREA_ROW_1_LEFT, false)} className="text-[9px] bouncy-btn bg-[#221a44] p-1 rounded">▼</button>
                      <span className="text-xs font-bold text-white px-1">{getPropVal("HVAC_SEAT_TEMPERATURE", AREA_ROW_1_LEFT, 0)}</span>
                      <button onClick={() => handleSeatHeat(AREA_ROW_1_LEFT, true)} className="text-[9px] bouncy-btn bg-[#221a44] p-1 rounded">▲</button>
                    </div>
                  </div>
                  <div className="flex flex-col items-center justify-center bg-black/40 p-1 rounded-xl border border-pink-500/10">
                    <span className="text-[8px] text-cyan-400 font-bold uppercase mb-1">💨 BREEZE</span>
                    <div className="flex items-center gap-1">
                      <button onClick={() => handleSeatVent(AREA_ROW_1_LEFT, false)} className="text-[9px] bouncy-btn bg-[#221a44] p-1 rounded">▼</button>
                      <span className="text-xs font-bold text-white px-1">{getPropVal("HVAC_SEAT_VENTILATION", AREA_ROW_1_LEFT, 0)}</span>
                      <button onClick={() => handleSeatVent(AREA_ROW_1_LEFT, true)} className="text-[9px] bouncy-btn bg-[#221a44] p-1 rounded">▲</button>
                    </div>
                  </div>
                </div>
              </div>

              {/* PASSENGER SEAT */}
              <div className="bg-[#120d29] p-4 rounded-2xl border border-[#352968] flex items-center justify-between">
                <div>
                  <span className="text-xs font-hud font-bold text-cyan-300">🥕 PASSENGER COMFORT</span>
                  <span className="text-[10px] text-slate-400 block">Row 1 Right Side</span>
                </div>
                <div className="flex gap-2">
                  <div className="flex flex-col items-center justify-center bg-black/40 p-1 rounded-xl border border-cyan-500/10">
                    <span className="text-[8px] text-pink-400 font-bold uppercase mb-1">🔥 WARM</span>
                    <div className="flex items-center gap-1">
                      <button onClick={() => handleSeatHeat(AREA_ROW_1_RIGHT, false)} className="text-[9px] bouncy-btn bg-[#221a44] p-1 rounded">▼</button>
                      <span className="text-xs font-bold text-white px-1">{getPropVal("HVAC_SEAT_TEMPERATURE", AREA_ROW_1_RIGHT, 0)}</span>
                      <button onClick={() => handleSeatHeat(AREA_ROW_1_RIGHT, true)} className="text-[9px] bouncy-btn bg-[#221a44] p-1 rounded">▲</button>
                    </div>
                  </div>
                  <div className="flex flex-col items-center justify-center bg-black/40 p-1 rounded-xl border border-cyan-500/10">
                    <span className="text-[8px] text-cyan-400 font-bold uppercase mb-1">💨 BREEZE</span>
                    <div className="flex items-center gap-1">
                      <button onClick={() => handleSeatVent(AREA_ROW_1_RIGHT, false)} className="text-[9px] bouncy-btn bg-[#221a44] p-1 rounded">▼</button>
                      <span className="text-xs font-bold text-white px-1">{getPropVal("HVAC_SEAT_VENTILATION", AREA_ROW_1_RIGHT, 0)}</span>
                      <button onClick={() => handleSeatVent(AREA_ROW_1_RIGHT, true)} className="text-[9px] bouncy-btn bg-[#221a44] p-1 rounded">▲</button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

        </div>

        {/* RIGHT COLUMN: INTERACTIVE CABIN SYSTEMS */}
        {}
        <div className="lg:col-span-4 flex flex-col gap-5">
          
          <div className="bg-[#191433]/90 border-2 border-pink-500/15 p-5 rounded-3xl shadow-lg">
            <span className="text-[11px] font-hud text-[#ff70a6] block mb-3 uppercase tracking-wider font-semibold">
              DOOR & GLASS TELEMETRY (2x2 GRID)
            </span>
            
            <div className="grid grid-cols-2 gap-3">
              {[
                { name: 'FL (DRIVER)', code: AREA_ROW_1_LEFT },
                { name: 'FR (PASSENGER)', code: AREA_ROW_1_RIGHT },
                { name: 'RL (REAR LEFT)', code: AREA_ROW_2_LEFT },
                { name: 'RR (REAR RIGHT)', code: AREA_ROW_2_RIGHT },
              ].map(door => {
                const isLocked = getPropVal('DOOR_LOCK', door.code, true);
                const isOpen = getPropVal('DOOR_MOVE', door.code, 0) === 1;
                const winPos = getPropVal('WINDOW_MOVE', door.code, 0);
                const winLocked = getPropVal('WINDOW_LOCK', door.code, false);

                return (
                  <div key={door.code} className="bg-[#120d29] p-3 rounded-2xl border border-[#30275c] flex flex-col justify-between">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[11px] font-hud font-bold text-pink-300">{door.name}</span>
                    </div>

                    <div className="flex flex-col gap-2">
                      <button
                        onClick={() => toggleDoorLock(door.code, isLocked)}
                        className={`py-1.5 px-2 text-[10px] font-hud bouncy-btn rounded-xl border-2 transition-all ${
                          isLocked ? 'bg-red-500/10 border-red-500/40 text-red-400' : 'bg-emerald-500/10 border-emerald-500/40 text-emerald-400'
                        }`}
                      >
                        {isLocked ? '🔒 LOCK' : '🔓 OPEN'}
                      </button>

                      <button
                        onClick={() => toggleDoorMove(door.code, isOpen)}
                        className={`py-1.5 px-2 text-[10px] font-hud bouncy-btn rounded-xl border-2 transition-all ${
                          isOpen ? 'bg-yellow-400/20 border-yellow-400/50 text-yellow-300' : 'bg-black/40 border-[#2b2354] text-slate-400'
                        }`}
                      >
                        {isOpen ? '🚪 OPENED' : '🚪 CLOSED'}
                      </button>

                      {/* Small cartoon Window slider inside 2x2 grid card */}
                      <div className="bg-black/40 p-1.5 rounded-xl border border-[#2b2354]">
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-[8px] font-hud text-slate-400 uppercase">WINDOW</span>
                          <span className="text-[8px] font-code text-cyan-300">{winPos}%</span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <input 
                            type="range"
                            min="0"
                            max="100"
                            value={winPos}
                            onChange={(e) => handleWindowSlide(door.code, e.target.value)}
                            className="w-full accent-pink-400 h-1 cursor-pointer bg-[#251e44] rounded"
                          />
                          <button
                            onClick={() => toggleWindowLock(door.code, winLocked)}
                            className={`text-[8px] font-hud px-1.5 py-0.5 rounded-lg border-2 bouncy-btn ${
                              winLocked ? 'bg-yellow-500/20 border-yellow-400 text-yellow-300' : 'bg-[#1b153b] border-[#2d2459] text-slate-500'
                            }`}
                          >
                            LCK
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

          </div>

          {/* LIGHT SYSTEM CONTROLS CARD */}
          <div className="bg-[#191433]/90 border-2 border-cyan-500/15 p-5 rounded-3xl shadow-lg">
            <span className="text-[11px] font-hud text-cyan-300 block mb-3 uppercase tracking-wider font-semibold">
              ILLUMINATION MODULE
            </span>
            <div className="grid grid-cols-2 gap-2.5">
              {[
                { name: 'HEADLIGHTS_SWITCH', act: 'headlights_toggle', icon: '🔆', label: 'MAIN HEAD' },
                { name: 'HAZARD_LIGHTS_SWITCH', act: 'hazard_toggle', icon: '🚨', label: 'HAZARD ALERT' },
                { name: 'FOG_LIGHTS_SWITCH', act: 'fog_toggle', icon: '🌫️', label: 'FOG SENSOR' },
                { name: 'CABIN_LIGHTS_SWITCH', act: 'cabin_lights_toggle', icon: '💡', label: 'CABIN LIGHT' },
              ].map(lt => {
                const isActive = getPropVal(lt.name, "0", false);
                return (
                  <button
                    key={lt.name}
                    onClick={() => toggleLightSwitch(lt.act)}
                    className={`flex flex-col items-center justify-center p-3 bouncy-btn rounded-2xl border-2 transition-all ${
                      isActive 
                        ? 'bg-yellow-400/10 border-yellow-400 text-yellow-300 shadow-md glow-yellow' 
                        : 'bg-black/30 border-[#30275c] text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <span className="text-xl mb-1">{lt.icon}</span>
                    <span className="text-[9px] font-hud font-bold leading-tight">{lt.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

        </div>

      </div>

      {}
      {/* DIAGNOSTIC REAL-TIME DESTRUCTIVE LOGGING MATRIX */}
      <div className="w-full max-w-6xl bg-[#191433]/90 border-2 border-[#ff70a6]/10 rounded-3xl p-5 shadow-2xl z-10">
        <div className="flex justify-between items-center mb-4 pb-2 border-b border-[#29224d]">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-pink-400 animate-ping" />
            <h2 className="text-xs font-hud font-bold tracking-widest text-pink-400 uppercase">
              VHAL PROTOCOL DIAGNOSTICS ANALYZER (ISO-11898 compliance)
            </h2>
          </div>
          <button
            onClick={() => setTelemetryLogs([])}
            className="text-[9px] font-code bg-black/40 hover:bg-black/70 px-3 py-1 bouncy-btn rounded-xl text-slate-400 border border-[#2a2250]"
          >
            CLEAR BUFFER
          </button>
        </div>

        {/* BUS PACKET SCROLL PORT */}
        <div className="bg-[#120d29]/95 border border-[#30255a] p-3 rounded-2xl h-40 overflow-y-auto font-code text-[10px] space-y-1.5 custom-scroll select-text">
          {telemetryLogs.length === 0 ? (
            <div className="text-slate-600 text-center py-10">Listening for CAN & VHAL network telemetry packets...</div>
          ) : (
            telemetryLogs.map(log => (
              <div key={log.id} className="flex gap-3 hover:bg-[#1a143a] p-1 rounded transition-all">
                <span className="text-slate-500 shrink-0">{log.time}</span>
                <span className={`font-bold shrink-0 ${
                  log.dir === 'RX' ? 'text-[#ff70a6]' : log.dir === 'TX' ? 'text-cyan-400' : 'text-yellow-400'
                }`}>
                  [{log.dir}]
                </span>
                <span className="text-slate-300 break-all">{log.content}</span>
              </div>
            ))
          )}
        </div>
      </div>

      {/* FIXED IGNITION BUTTON DOCKED TO THE SCREEN FOOTER */}
      <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 z-40 bg-[#120d29]/95 border-2 border-pink-400/30 px-6 py-3.5 rounded-full shadow-2xl flex items-center gap-6 backdrop-blur-md">
        <div className="flex flex-col">
          <span className="text-[8px] font-hud text-slate-400 tracking-widest uppercase font-bold">MASTER DIAGNOSTICS</span>
          <span className="text-[10px] font-code text-pink-300">ECU MAIN IGNITION</span>
        </div>
        <button
          onClick={triggerPowerToggle}
          className={`px-8 py-2.5 rounded-full font-hud font-bold text-xs tracking-wider bouncy-btn transition-all ${
            isPowerOn 
              ? 'bg-[#ff5d8f] hover:bg-[#ff70a6] text-white glow-pink border-2 border-pink-300' 
              : 'bg-[#00f0ff] hover:bg-[#70e4ff] text-slate-950 glow-cyan font-extrabold border-2 border-cyan-300'
          }`}
        >
          {isPowerOn ? 'SHUTDOWN ECU' : 'BOOT VHAL NETWORK'}
        </button>
      </div>

    </div>
  );
}
