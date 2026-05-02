import { useState, useEffect, useRef } from "react";

const TEAM_COLORS = {
  "Red Bull Racing": "#3671C6", "McLaren": "#FF8000", "Ferrari": "#E8002D",
  "Mercedes": "#27F4D2", "Aston Martin": "#229971", "Alpine": "#FF87BC",
  "Williams": "#64C4FF", "Haas F1 Team": "#B6BABD", "RB": "#6692FF", "Kick Sauber": "#52E252",
};

const GRID = [
  { team: "Red Bull Racing", d1: "Max Verstappen", d2: "Liam Lawson", abbr1: "VER", abbr2: "LAW", rookie2: false },
  { team: "McLaren", d1: "Lando Norris", d2: "Oscar Piastri", abbr1: "NOR", abbr2: "PIA", rookie2: false },
  { team: "Ferrari", d1: "Charles Leclerc", d2: "Lewis Hamilton", abbr1: "LEC", abbr2: "HAM", rookie2: false },
  { team: "Mercedes", d1: "George Russell", d2: "Andrea Antonelli", abbr1: "RUS", abbr2: "ANT", rookie2: true },
  { team: "Aston Martin", d1: "Fernando Alonso", d2: "Lance Stroll", abbr1: "ALO", abbr2: "STR", rookie2: false },
  { team: "Alpine", d1: "Pierre Gasly", d2: "Jack Doohan", abbr1: "GAS", abbr2: "DOO", rookie2: false },
  { team: "Haas F1 Team", d1: "Esteban Ocon", d2: "Oliver Bearman", abbr1: "OCO", abbr2: "BEA", rookie2: false },
  { team: "RB", d1: "Yuki Tsunoda", d2: "Isack Hadjar", abbr1: "TSU", abbr2: "HAD", rookie2: true },
  { team: "Williams", d1: "Alexander Albon", d2: "Carlos Sainz", abbr1: "ALB", abbr2: "SAI", rookie2: false },
  { team: "Kick Sauber", d1: "Nico Hulkenberg", d2: "Gabriel Bortoleto", abbr1: "HUL", abbr2: "BOR", rookie2: true },
];

const CIRCUITS = [
  "Australian GP", "Chinese GP", "Japanese GP", "Bahrain GP", "Saudi Arabian GP",
  "Miami GP", "Emilia Romagna GP", "Monaco GP", "Canadian GP", "Spanish GP",
  "Austrian GP", "British GP", "Hungarian GP", "Belgian GP", "Dutch GP",
  "Italian GP", "Azerbaijan GP", "Singapore GP", "United States GP", "Mexico City GP",
  "São Paulo GP", "Las Vegas GP", "Qatar GP", "Abu Dhabi GP",
];

const CIRCUIT_INFO = {
  "Australian GP": { type: "Street", speed: "Medium", drs: 4, corners: 16 },
  "Chinese GP": { type: "Permanent", speed: "Medium", drs: 2, corners: 16 },
  "Japanese GP": { type: "Permanent", speed: "Fast", drs: 1, corners: 18 },
  "Bahrain GP": { type: "Permanent", speed: "Medium", drs: 3, corners: 15 },
  "Saudi Arabian GP": { type: "Street", speed: "Fast", drs: 3, corners: 27 },
  "Miami GP": { type: "Street", speed: "Medium", drs: 3, corners: 19 },
  "Emilia Romagna GP": { type: "Permanent", speed: "Medium", drs: 2, corners: 19 },
  "Monaco GP": { type: "Street", speed: "Slow", drs: 1, corners: 19 },
  "Canadian GP": { type: "Street", speed: "Medium", drs: 3, corners: 14 },
  "Spanish GP": { type: "Permanent", speed: "Medium", drs: 2, corners: 16 },
  "Austrian GP": { type: "Permanent", speed: "Fast", drs: 2, corners: 10 },
  "British GP": { type: "Permanent", speed: "Fast", drs: 2, corners: 18 },
  "Hungarian GP": { type: "Permanent", speed: "Slow", drs: 2, corners: 14 },
  "Belgian GP": { type: "Permanent", speed: "Fast", drs: 2, corners: 19 },
  "Dutch GP": { type: "Permanent", speed: "Medium", drs: 2, corners: 14 },
  "Italian GP": { type: "Permanent", speed: "Fast", drs: 2, corners: 11 },
  "Azerbaijan GP": { type: "Street", speed: "Fast", drs: 2, corners: 20 },
  "Singapore GP": { type: "Street", speed: "Slow", drs: 3, corners: 23 },
  "United States GP": { type: "Permanent", speed: "Fast", drs: 2, corners: 20 },
  "Mexico City GP": { type: "Permanent", speed: "Medium", drs: 3, corners: 17 },
  "São Paulo GP": { type: "Permanent", speed: "Medium", drs: 2, corners: 15 },
  "Las Vegas GP": { type: "Street", speed: "Fast", drs: 2, corners: 17 },
  "Qatar GP": { type: "Permanent", speed: "Fast", drs: 3, corners: 16 },
  "Abu Dhabi GP": { type: "Permanent", speed: "Fast", drs: 2, corners: 16 },
};

const TEAM_STRENGTHS = {
  "Red Bull Racing": 0.85, "McLaren": 0.82, "Ferrari": 0.80, "Mercedes": 0.75,
  "Aston Martin": 0.65, "Alpine": 0.55, "Williams": 0.52, "Haas F1 Team": 0.48,
  "RB": 0.50, "Kick Sauber": 0.42,
};

const DRIVER_SKILL = {
  "Max Verstappen": 0.98, "Lando Norris": 0.93, "Charles Leclerc": 0.92,
  "Lewis Hamilton": 0.90, "George Russell": 0.88, "Fernando Alonso": 0.87,
  "Oscar Piastri": 0.85, "Carlos Sainz": 0.86, "Nico Hulkenberg": 0.78,
  "Yuki Tsunoda": 0.75, "Lance Stroll": 0.70, "Pierre Gasly": 0.74,
  "Alexander Albon": 0.76, "Esteban Ocon": 0.72, "Oliver Bearman": 0.68,
  "Jack Doohan": 0.65, "Liam Lawson": 0.67, "Andrea Antonelli": 0.60,
  "Isack Hadjar": 0.62, "Gabriel Bortoleto": 0.63,
};

function computePrediction(team, driver, circuit, session, compound) {
  const info = CIRCUIT_INFO[circuit] || { type: "Permanent", speed: "Medium", drs: 2, corners: 16 };
  const teamStr = TEAM_STRENGTHS[team] || 0.5;
  const drvSkill = DRIVER_SKILL[driver] || 0.65;

  const speedMult = { Slow: 1.12, Medium: 1.0, Fast: 0.91 }[info.speed] || 1.0;
  const streetBonus = info.type === "Street" ? 0.03 : 0;
  const drsBonus = info.drs * 0.01;
  const compoundMult = { SOFT: 0.0, MEDIUM: 0.25, HARD: 0.55 }[compound] || 0.0;
  const sessionMult = { Q3: 0.0, Q2: 0.35, Q1: 0.75 }[session] || 0.0;

  const basePole = 75 + info.corners * 0.8;
  const gapToPole =
    (1 - teamStr) * 3.5 * speedMult +
    (1 - drvSkill) * 1.8 +
    compoundMult * 0.3 +
    sessionMult * 0.4 +
    streetBonus -
    drsBonus +
    (Math.random() * 0.12 - 0.06);

  return { absTime: basePole + Math.max(0, gapToPole), delta: Math.max(0, gapToPole) };
}

function formatTime(s) {
  if (!s || s <= 0) return "—";
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(3).padStart(6, "0");
  return `${m}:${sec}`;
}

export default function App() {
  const [screen, setScreen] = useState("hero");
  const [selectedTeam, setSelectedTeam] = useState(null);
  const [tab, setTab] = useState("predict");
  const [circuit, setCircuit] = useState("Monaco GP");
  const [session, setSession] = useState("Q3");
  const [compound, setCompound] = useState("SOFT");
  const [aiInsight, setAiInsight] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [gridPredictions, setGridPredictions] = useState([]);
  const [gridLoading, setGridLoading] = useState(false);
  const heroRef = useRef(null);

  const teamData = GRID.find((g) => g.team === selectedTeam);
  const teamColor = TEAM_COLORS[selectedTeam] || "#888";

  useEffect(() => {
    if (screen === "hero") {
      const el = heroRef.current;
      if (!el) return;
      const move = (e) => {
        const x = (e.clientX / window.innerWidth - 0.5) * 18;
        const y = (e.clientY / window.innerHeight - 0.5) * 10;
        el.style.transform = `translate(${x}px, ${y}px) scale(1.06)`;
      };
      window.addEventListener("mousemove", move);
      return () => window.removeEventListener("mousemove", move);
    }
  }, [screen]);

  useEffect(() => {
    if (screen === "app" && tab === "predict" && selectedTeam) {
      computeFullGrid();
    }
  }, [screen, tab, selectedTeam, circuit, session, compound]);

  function computeFullGrid() {
    setGridLoading(true);
    setTimeout(() => {
      const results = GRID.flatMap((g) =>
        [{ d: g.d1, a: g.abbr1 }, { d: g.d2, a: g.abbr2 }].map(({ d, a }) => {
          const { absTime, delta } = computePrediction(g.team, d, circuit, session, compound);
          return { team: g.team, driver: d, abbr: a, absTime, delta, color: TEAM_COLORS[g.team] };
        })
      );
      results.sort((a, b) => a.delta - b.delta);
      setGridPredictions(results);
      setGridLoading(false);
    }, 300);
  }

  async function fetchAiInsight() {
    if (!teamData || aiLoading) return;
    setAiLoading(true);
    setAiInsight("");
    const info = CIRCUIT_INFO[circuit] || {};
    const p1 = computePrediction(selectedTeam, teamData.d1, circuit, session, compound);
    const p2 = computePrediction(selectedTeam, teamData.d2, circuit, session, compound);
    const faster = p1.delta < p2.delta ? teamData.d1 : teamData.d2;
    const gap = Math.abs(p1.delta - p2.delta).toFixed(3);

    const prompt = `You are an F1 qualifying analyst. Give a sharp, punchy 3-sentence analysis (no bullet points, no markdown) about ${selectedTeam} at the ${circuit} in ${session}. ${faster} is predicted faster by ${gap}s. The circuit is a ${info.type || "permanent"} ${info.speed || "medium"}-speed track with ${info.drs || 2} DRS zones. Comment on team strengths, driver styles, and what to watch. Be specific and insightful, like an expert commentator.`;

    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          messages: [{ role: "user", content: prompt }],
        }),
      });
      const data = await res.json();
      const text = data.content?.map((c) => c.text || "").join("") || "Analysis unavailable.";
      setAiInsight(text);
    } catch {
      setAiInsight("Analysis unavailable — check your connection.");
    }
    setAiLoading(false);
  }

  if (screen === "hero") return <HeroScreen heroRef={heroRef} onEnter={() => setScreen("selector")} />;
  if (screen === "selector") return <SelectorScreen onBack={() => setScreen("hero")} onSelect={(t) => { setSelectedTeam(t); setScreen("app"); setTab("predict"); }} />;

  const pred1 = computePrediction(selectedTeam, teamData.d1, circuit, session, compound);
  const pred2 = computePrediction(selectedTeam, teamData.d2, circuit, session, compound);
  const faster = pred1.delta < pred2.delta ? teamData.abbr1 : teamData.abbr2;
  const gap = Math.abs(pred1.delta - pred2.delta).toFixed(3);

  return (
    <div style={{ minHeight: "100vh", background: "#0a0a0a", color: "#f5f5f7", fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', sans-serif" }}>
      {/* Sticky nav */}
      <div style={{ position: "sticky", top: 0, zIndex: 100, background: "rgba(10,10,10,0.92)", backdropFilter: "blur(20px)", borderBottom: "0.5px solid rgba(255,255,255,0.08)", height: 54, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 28px" }}>
        <button onClick={() => setScreen("selector")} style={{ background: "none", border: "none", color: "rgba(255,255,255,0.4)", fontSize: 13, cursor: "pointer", letterSpacing: "0.06em", textTransform: "uppercase" }}>← Change team</button>
        <div style={{ display: "flex", gap: 2, background: "rgba(255,255,255,0.06)", borderRadius: 10, padding: 3 }}>
          {["predict", "analyse"].map((t) => (
            <button key={t} onClick={() => setTab(t)} style={{ background: tab === t ? "rgba(255,255,255,0.12)" : "transparent", border: "none", color: tab === t ? "#fff" : "rgba(255,255,255,0.4)", padding: "5px 20px", borderRadius: 8, fontSize: 13, fontWeight: 500, cursor: "pointer", textTransform: "capitalize", letterSpacing: "0.04em", transition: "all 0.15s" }}>{t}</button>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: teamColor }} />
          <span style={{ fontSize: 13, fontWeight: 600, color: "rgba(255,255,255,0.8)", letterSpacing: "0.04em" }}>{selectedTeam}</span>
        </div>
      </div>

      <div style={{ maxWidth: 980, margin: "0 auto", padding: "24px 24px 80px" }}>
        {tab === "predict" ? (
          <PredictTab
            teamData={teamData} teamColor={teamColor} circuit={circuit} setCircuit={setCircuit}
            session={session} setSession={setSession} compound={compound} setCompound={setCompound}
            pred1={pred1} pred2={pred2} faster={faster} gap={gap}
            gridPredictions={gridPredictions} gridLoading={gridLoading}
            aiInsight={aiInsight} aiLoading={aiLoading} fetchAiInsight={fetchAiInsight}
          />
        ) : (
          <AnalyseTab teamData={teamData} teamColor={teamColor} selectedTeam={selectedTeam} />
        )}
      </div>
    </div>
  );
}

function HeroScreen({ heroRef, onEnter }) {
  return (
    <div style={{ width: "100vw", height: "100vh", background: "#050505", display: "flex", flexDirection: "column", overflow: "hidden", position: "relative", cursor: "none" }}>
      {/* Animated gradient backdrop */}
      <div ref={heroRef} style={{ position: "absolute", inset: "-5%", zIndex: 0, transition: "transform 0.4s cubic-bezier(0.23,1,0.32,1)", background: "radial-gradient(ellipse 70% 50% at 60% 40%, rgba(229,0,0,0.08) 0%, transparent 60%), radial-gradient(ellipse 50% 60% at 20% 80%, rgba(54,113,198,0.06) 0%, transparent 60%)" }} />

      {/* Scan lines texture */}
      <div style={{ position: "absolute", inset: 0, zIndex: 1, backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.012) 2px, rgba(255,255,255,0.012) 4px)", pointerEvents: "none" }} />

      {/* Top bar */}
      <div style={{ position: "relative", zIndex: 2, height: 60, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 48px", borderBottom: "0.5px solid rgba(255,255,255,0.06)" }}>
        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.18em", textTransform: "uppercase", color: "rgba(255,255,255,0.3)" }}>F1 Qualifying</span>
        <span style={{ fontSize: 11, color: "rgba(255,255,255,0.2)", letterSpacing: "0.08em" }}>2025 Season</span>
      </div>

      {/* Main content */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", padding: "0 48px 40px", position: "relative", zIndex: 2 }}>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.18em", textTransform: "uppercase", color: "#e10600", marginBottom: 20 }}>Predictor</div>
        <div style={{ fontFamily: "'Barlow Condensed', 'Impact', sans-serif", fontSize: "clamp(72px,11vw,140px)", fontWeight: 900, lineHeight: 0.92, letterSpacing: "-0.02em", color: "#fff", marginBottom: 8, textTransform: "uppercase" }}>
          Who takes<br /><span style={{ color: "rgba(255,255,255,0.25)" }}>pole?</span>
        </div>
        <div style={{ fontSize: 17, color: "rgba(255,255,255,0.35)", fontWeight: 400, maxWidth: 420, lineHeight: 1.6, marginTop: 28 }}>
          AI-powered qualifying predictions for every team and circuit on the 2025 calendar.
        </div>
      </div>

      {/* Bottom bar */}
      <div style={{ position: "relative", zIndex: 2, height: 64, display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 48px", borderTop: "0.5px solid rgba(255,255,255,0.06)" }}>
        <span style={{ fontSize: 11, fontWeight: 500, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(255,255,255,0.25)" }}>by Anirudh Atkuru</span>
        <button onClick={onEnter} style={{ display: "flex", alignItems: "center", gap: 12, background: "none", border: "none", cursor: "pointer", color: "#fff", fontSize: 13, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase", transition: "opacity 0.2s" }}>
          Select your team
          <span style={{ width: 36, height: 36, borderRadius: "50%", border: "1px solid rgba(255,255,255,0.25)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16 }}>→</span>
        </button>
      </div>
    </div>
  );
}

function SelectorScreen({ onBack, onSelect }) {
  const [hovered, setHovered] = useState(null);
  return (
    <div style={{ minHeight: "100vh", background: "#0d0d0d", color: "#fff", fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif" }}>
      <div style={{ position: "sticky", top: 0, zIndex: 10, background: "rgba(13,13,13,0.95)", backdropFilter: "blur(20px)", borderBottom: "0.5px solid rgba(255,255,255,0.07)", padding: "16px 32px", display: "flex", alignItems: "center", gap: 16 }}>
        <button onClick={onBack} style={{ background: "none", border: "none", color: "rgba(255,255,255,0.45)", fontSize: 13, cursor: "pointer", display: "flex", alignItems: "center", gap: 6 }}>← Back</button>
      </div>
      <div style={{ padding: "40px 32px 20px", maxWidth: 1100, margin: "0 auto" }}>
        <div style={{ fontFamily: "'Barlow Condensed','Impact',sans-serif", fontSize: "clamp(52px,8vw,96px)", fontWeight: 900, color: "rgba(185,187,196,0.9)", textTransform: "uppercase", letterSpacing: "-0.01em", lineHeight: 0.95 }}>Select a team</div>
      </div>
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "0 32px 60px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        {GRID.map((g) => {
          const tc = TEAM_COLORS[g.team];
          const isH = hovered === g.team;
          return (
            <div key={g.team} onClick={() => onSelect(g.team)} onMouseEnter={() => setHovered(g.team)} onMouseLeave={() => setHovered(null)}
              style={{ display: "grid", gridTemplateColumns: "1fr 1fr", cursor: "pointer", borderRadius: 8, overflow: "hidden", transition: "transform 0.15s, box-shadow 0.15s", transform: isH ? "translateY(-2px)" : "none", boxShadow: isH ? `0 8px 32px rgba(0,0,0,0.4)` : "none" }}>
              {[{ d: g.d1, abbr: g.abbr1, rookie: false }, { d: g.d2, abbr: g.abbr2, rookie: g.rookie2 }].map(({ d, abbr, rookie }, i) => (
                <div key={abbr} style={{ background: i === 0 ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.04)", display: "flex", flexDirection: "column" }}>
                  <div style={{ background: `linear-gradient(135deg, ${tc}22 0%, transparent 60%)`, padding: "18px 16px 14px", flex: 1, display: "flex", flexDirection: "column", justifyContent: "flex-end" }}>
                    <div style={{ fontSize: 10, fontWeight: 600, letterSpacing: "0.12em", textTransform: "uppercase", color: "rgba(255,255,255,0.35)", marginBottom: 4 }}>{g.team}</div>
                    <div style={{ fontFamily: "'Barlow Condensed','Impact',sans-serif", fontSize: "clamp(22px,3vw,32px)", fontWeight: 900, color: "#fff", letterSpacing: "-0.02em", lineHeight: 1 }}>
                      {abbr}{rookie && <sup style={{ fontSize: 10, color: "rgba(255,230,100,0.7)", marginLeft: 2 }}>R</sup>}
                    </div>
                    <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", marginTop: 2 }}>{d.split(" ")[1] || d}</div>
                  </div>
                  <div style={{ height: 3, background: tc }} />
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PredictTab({ teamData, teamColor, circuit, setCircuit, session, setSession, compound, setCompound, pred1, pred2, faster, gap, gridPredictions, gridLoading, aiInsight, aiLoading, fetchAiInsight }) {
  const maxDelta = gridPredictions.length ? Math.max(...gridPredictions.map((r) => r.delta)) : 3;

  return (
    <div>
      {/* Controls */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr auto auto", gap: 10, marginBottom: 24 }}>
        <select value={circuit} onChange={(e) => setCircuit(e.target.value)} style={{ background: "rgba(255,255,255,0.06)", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 10, color: "#fff", padding: "10px 14px", fontSize: 14, cursor: "pointer" }}>
          {CIRCUITS.map((c) => <option key={c} value={c} style={{ background: "#1a1a1a" }}>{c}</option>)}
        </select>
        <select value={session} onChange={(e) => setSession(e.target.value)} style={{ background: "rgba(255,255,255,0.06)", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 10, color: "#fff", padding: "10px 14px", fontSize: 14, cursor: "pointer" }}>
          {["Q3", "Q2", "Q1"].map((s) => <option key={s} value={s} style={{ background: "#1a1a1a" }}>{s}</option>)}
        </select>
        <select value={compound} onChange={(e) => setCompound(e.target.value)} style={{ background: "rgba(255,255,255,0.06)", border: "0.5px solid rgba(255,255,255,0.12)", borderRadius: 10, color: "#fff", padding: "10px 14px", fontSize: 14, cursor: "pointer" }}>
          {["SOFT", "MEDIUM", "HARD"].map((c) => <option key={c} value={c} style={{ background: "#1a1a1a" }}>{c}</option>)}
        </select>
      </div>

      {/* Circuit badge */}
      {(() => {
        const info = CIRCUIT_INFO[circuit] || {};
        return (
          <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
            {[info.type, `${info.speed} speed`, `${info.drs} DRS zones`, `${info.corners} corners`].map((tag) => (
              <span key={tag} style={{ fontSize: 11, padding: "3px 10px", borderRadius: 6, background: "rgba(255,255,255,0.06)", color: "rgba(255,255,255,0.45)", letterSpacing: "0.06em" }}>{tag}</span>
            ))}
          </div>
        );
      })()}

      {/* Driver cards */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 12 }}>
        {[
          { abbr: teamData.abbr1, name: teamData.d1, pred: pred1, color: teamColor },
          { abbr: teamData.abbr2, name: teamData.d2, pred: pred2, color: "#6692ff" },
        ].map(({ abbr, name, pred, color }) => (
          <div key={abbr} style={{ background: "rgba(255,255,255,0.04)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: "22px 20px", position: "relative", overflow: "hidden" }}>
            <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: color }} />
            <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase", color: "rgba(255,255,255,0.35)", marginBottom: 8 }}>{name}</div>
            <div style={{ fontFamily: "'Barlow Condensed','Impact',sans-serif", fontSize: "clamp(40px,6vw,64px)", fontWeight: 900, color, letterSpacing: "-0.03em", lineHeight: 1, marginBottom: 16 }}>{abbr}</div>
            <div style={{ fontSize: 28, fontWeight: 600, letterSpacing: "-0.02em", color: "#f5f5f7", fontVariantNumeric: "tabular-nums", marginBottom: 4 }}>{formatTime(pred.absTime)}</div>
            <div style={{ fontSize: 13, color: "rgba(255,255,255,0.4)", marginBottom: 16 }}>+{pred.delta.toFixed(3)}s to pole</div>
            <div style={{ display: "flex", gap: 20 }}>
              {[["Compound", compound], ["Session", session]].map(([l, v]) => (
                <div key={l}><div style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", marginBottom: 2, textTransform: "uppercase", letterSpacing: "0.06em" }}>{l}</div><div style={{ fontSize: 13, fontWeight: 600, color: "#f5f5f7" }}>{v}</div></div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Gap badge */}
      <div style={{ background: "rgba(255,255,255,0.04)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 12, padding: "12px 18px", fontSize: 13, color: "rgba(255,255,255,0.5)", textAlign: "center", marginBottom: 28 }}>
        Intra-team gap &nbsp;<strong style={{ color: "#f5f5f7" }}>{gap}s</strong>&nbsp; · &nbsp;
        <span style={{ color: teamColor, fontWeight: 600 }}>{faster}</span> predicted faster
      </div>

      {/* AI Insight button */}
      <div style={{ marginBottom: 28 }}>
        <button onClick={fetchAiInsight} disabled={aiLoading} style={{ background: "rgba(229,0,0,0.12)", border: "0.5px solid rgba(229,0,0,0.3)", borderRadius: 10, color: aiLoading ? "rgba(229,0,0,0.4)" : "#ff453a", padding: "10px 20px", fontSize: 13, fontWeight: 600, cursor: aiLoading ? "default" : "pointer", letterSpacing: "0.06em", textTransform: "uppercase", transition: "all 0.2s" }}>
          {aiLoading ? "Analysing..." : "✦ Get AI race insight"}
        </button>
        {aiInsight && (
          <div style={{ marginTop: 12, background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 14, padding: "18px 20px" }}>
            <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "#ff453a", marginBottom: 10 }}>AI Analysis</div>
            <div style={{ fontSize: 14, color: "rgba(255,255,255,0.7)", lineHeight: 1.7 }}>{aiInsight}</div>
          </div>
        )}
      </div>

      {/* Full grid */}
      <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "rgba(255,255,255,0.3)", marginBottom: 10 }}>Predicted grid — {session} · {compound} · {circuit}</div>
      <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.07)", borderRadius: 16, overflow: "hidden" }}>
        {gridLoading ? (
          <div style={{ padding: 24, textAlign: "center", color: "rgba(255,255,255,0.3)", fontSize: 13 }}>Computing grid...</div>
        ) : (
          gridPredictions.map((r, i) => {
            const isTeam = r.team === teamData?.team;
            return (
              <div key={r.driver} style={{ display: "grid", gridTemplateColumns: "36px 10px 52px 1fr 80px", padding: "8px 16px", gap: 8, alignItems: "center", borderBottom: i < gridPredictions.length - 1 ? "0.5px solid rgba(255,255,255,0.05)" : "none", background: isTeam ? "rgba(255,255,255,0.04)" : "transparent" }}>
                <span style={{ fontSize: 12, color: "rgba(255,255,255,0.35)", fontWeight: 500 }}>P{i + 1}</span>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: r.color, display: "inline-block" }} />
                <span style={{ fontSize: 13, fontWeight: isTeam ? 700 : 500, color: isTeam ? teamColor : "#f5f5f7", letterSpacing: "-0.01em" }}>{r.abbr}</span>
                <div style={{ height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 2 }}>
                  <div style={{ height: 3, width: `${maxDelta > 0 ? (r.delta / maxDelta) * 100 : 0}%`, background: r.color, borderRadius: 2, minWidth: 2 }} />
                </div>
                <span style={{ fontSize: 12, color: i === 0 ? "#f5f5f7" : "rgba(255,255,255,0.45)", textAlign: "right", fontVariantNumeric: "tabular-nums", fontWeight: i === 0 ? 700 : 400 }}>
                  {i === 0 ? "POLE" : `+${r.delta.toFixed(3)}s`}
                </span>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

function AnalyseTab({ teamData, teamColor, selectedTeam }) {
  const [chartData, setChartData] = useState([]);
  const [h2h, setH2h] = useState([]);
  const [analysisText, setAnalysisText] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!teamData) return;
    const results = CIRCUITS.map((c) => {
      const p1 = computePrediction(selectedTeam, teamData.d1, c, "Q3", "SOFT");
      const p2 = computePrediction(selectedTeam, teamData.d2, c, "Q3", "SOFT");
      return { circuit: c.replace(" GP", ""), d1: p1.delta, d2: p2.delta, gap: p1.delta - p2.delta };
    });
    setChartData(results);
    setH2h(results);
  }, [teamData, selectedTeam]);

  const maxDelta = Math.max(...chartData.map((r) => Math.max(r.d1, r.d2)), 1);
  const d1avg = chartData.length ? (chartData.reduce((s, r) => s + r.d1, 0) / chartData.length).toFixed(3) : "—";
  const d2avg = chartData.length ? (chartData.reduce((s, r) => s + r.d2, 0) / chartData.length).toFixed(3) : "—";
  const d1wins = h2h.filter((r) => r.gap < 0).length;
  const d2wins = h2h.filter((r) => r.gap > 0).length;
  const avgTeamGap = h2h.length ? (h2h.reduce((s, r) => s + Math.abs(r.gap), 0) / h2h.length).toFixed(3) : "—";

  async function fetchSeasonAnalysis() {
    setLoading(true);
    setAnalysisText("");
    const prompt = `You are an F1 analyst. Write a 4-sentence season overview (no markdown, no bullets) for ${selectedTeam} in 2025 qualifying. ${teamData.d1} is faster at ${d1wins} circuits and averages +${d1avg}s to pole. ${teamData.d2} is faster at ${d2wins} circuits and averages +${d2avg}s. Average intra-team gap is ${avgTeamGap}s. Discuss team competitiveness, driver pairing dynamics, circuit types that suit them, and title prospects. Be analytical and specific.`;
    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "claude-sonnet-4-20250514", max_tokens: 1000, messages: [{ role: "user", content: prompt }] }),
      });
      const data = await res.json();
      setAnalysisText(data.content?.map((c) => c.text || "").join("") || "Unavailable.");
    } catch { setAnalysisText("Analysis unavailable."); }
    setLoading(false);
  }

  if (!teamData) return null;

  return (
    <div>
      <div style={{ fontSize: 32, fontWeight: 700, letterSpacing: "-0.03em", color: "#f5f5f7", marginBottom: 4 }}>{selectedTeam}</div>
      <div style={{ fontSize: 14, color: "rgba(255,255,255,0.4)", marginBottom: 24 }}>Season analysis — Q3 · SOFT · all circuits</div>

      {/* Metrics */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginBottom: 24 }}>
        {[
          { label: `${teamData.abbr1} avg gap`, value: `+${d1avg}s`, sub: "to pole" },
          { label: `${teamData.abbr2} avg gap`, value: `+${d2avg}s`, sub: "to pole" },
          { label: `${teamData.abbr1} wins`, value: `${d1wins}/${CIRCUITS.length}`, sub: "circuits faster" },
          { label: "Avg team gap", value: `${avgTeamGap}s`, sub: "intra-team" },
        ].map((m) => (
          <div key={m.label} style={{ background: "rgba(255,255,255,0.04)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 14, padding: "14px 16px" }}>
            <div style={{ fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 4, letterSpacing: "0.04em" }}>{m.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em", color: "#f5f5f7" }}>{m.value}</div>
            <div style={{ fontSize: 11, color: "rgba(255,255,255,0.25)", marginTop: 2 }}>{m.sub}</div>
          </div>
        ))}
      </div>

      {/* Bar chart — gap to pole */}
      <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.07)", borderRadius: 16, padding: "20px 20px 14px", marginBottom: 12 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "#f5f5f7", marginBottom: 2 }}>Gap to pole — all circuits</div>
        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", marginBottom: 16 }}>Q3 · SOFT · 2025 calendar</div>
        <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
          {[{ label: teamData.abbr1, color: teamColor }, { label: teamData.abbr2, color: "#6692ff" }].map((l) => (
            <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "rgba(255,255,255,0.5)" }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: l.color }} />
              {l.label}
            </div>
          ))}
        </div>
        <div style={{ overflowX: "auto" }}>
          <div style={{ minWidth: 600, display: "flex", flexDirection: "column", gap: 4 }}>
            {chartData.map((r) => (
              <div key={r.circuit} style={{ display: "grid", gridTemplateColumns: "90px 1fr 1fr", gap: 6, alignItems: "center" }}>
                <div style={{ fontSize: 10, color: "rgba(255,255,255,0.35)", textAlign: "right", paddingRight: 8, letterSpacing: "0.04em" }}>{r.circuit}</div>
                <div style={{ position: "relative", height: 10 }}>
                  <div style={{ position: "absolute", left: 0, top: "50%", transform: "translateY(-50%)", height: 6, width: `${(r.d1 / maxDelta) * 100}%`, background: teamColor, borderRadius: 3, minWidth: 2 }} />
                </div>
                <div style={{ position: "relative", height: 10 }}>
                  <div style={{ position: "absolute", left: 0, top: "50%", transform: "translateY(-50%)", height: 6, width: `${(r.d2 / maxDelta) * 100}%`, background: "#6692ff", borderRadius: 3, minWidth: 2 }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* H2H chart */}
      <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.07)", borderRadius: 16, padding: "20px 20px 14px", marginBottom: 24 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: "#f5f5f7", marginBottom: 2 }}>Head-to-head per circuit</div>
        <div style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", marginBottom: 16 }}>Negative = {teamData.abbr1} faster · Positive = {teamData.abbr2} faster</div>
        <div style={{ overflowX: "auto" }}>
          <div style={{ minWidth: 600, display: "flex", flexDirection: "column", gap: 3 }}>
            {h2h.map((r) => {
              const maxG = Math.max(...h2h.map((x) => Math.abs(x.gap)), 0.01);
              const pct = Math.min(Math.abs(r.gap) / maxG * 45, 45);
              return (
                <div key={r.circuit} style={{ display: "grid", gridTemplateColumns: "90px 1fr 6px 1fr", gap: 4, alignItems: "center" }}>
                  <div style={{ fontSize: 10, color: "rgba(255,255,255,0.35)", textAlign: "right", paddingRight: 8 }}>{r.circuit}</div>
                  <div style={{ display: "flex", justifyContent: "flex-end" }}>
                    {r.gap < 0 && <div style={{ height: 6, width: `${pct}%`, background: teamColor, borderRadius: "3px 0 0 3px" }} />}
                  </div>
                  <div style={{ width: 6, height: 6, background: "rgba(255,255,255,0.1)", borderRadius: 1 }} />
                  <div style={{ display: "flex", justifyContent: "flex-start" }}>
                    {r.gap > 0 && <div style={{ height: 6, width: `${pct}%`, background: "#6692ff", borderRadius: "0 3px 3px 0" }} />}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* AI season analysis */}
      <button onClick={fetchSeasonAnalysis} disabled={loading} style={{ background: "rgba(229,0,0,0.12)", border: "0.5px solid rgba(229,0,0,0.3)", borderRadius: 10, color: loading ? "rgba(229,0,0,0.4)" : "#ff453a", padding: "10px 20px", fontSize: 13, fontWeight: 600, cursor: loading ? "default" : "pointer", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 16 }}>
        {loading ? "Analysing..." : "✦ Get AI season overview"}
      </button>
      {analysisText && (
        <div style={{ background: "rgba(255,255,255,0.03)", border: "0.5px solid rgba(255,255,255,0.08)", borderRadius: 14, padding: "18px 20px" }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase", color: "#ff453a", marginBottom: 10 }}>Season Overview</div>
          <div style={{ fontSize: 14, color: "rgba(255,255,255,0.7)", lineHeight: 1.7 }}>{analysisText}</div>
        </div>
      )}
    </div>
  );
}
