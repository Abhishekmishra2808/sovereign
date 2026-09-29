// 20 s explainer for Sovereign, rendered with Higgsfield's higgsedit CLI:
//   higgsedit fonts add . Inter:400 Inter:600 Inter:700 Inter:900
//   higgsedit build sovereign-explainer.jsx
// Every number shown is from benchmarks/reports (sih-online.md, gpu-dense-ipm.md).
export default async ({ project, icon }) => {
  const W = 1920;
  const H = 1080;
  const p = await project({ size: "1920x1080", fps: 30, background: "#000000" });

  const FONT = "Inter";
  const WHITE = "#ffffff";
  const SOFT = "#b3b3b3";
  const MUTED = "#7a7a7a";
  const CARD = "#131316";
  const ACCENT = "#9fe870";
  const WARN = "#ff7a66";
  const glow = {
    kind: "radial",
    stops: [
      { offset: 0, color: "#1a1c22" },
      { offset: 1, color: "#000000" },
    ],
  };
  const exit = { to: { opacity: 0 }, duration: 0.35, anchor: "end" };
  const rise = () => ({ enter: { from: { y: 28, opacity: 0 }, duration: 0.5 } });

  const eyebrow = (label, y) => (
    <text x={210} y={y} width={1500} align="center" fontFamily={FONT} fontSize={26}
      fontWeight={600} letterSpacing={8} color={MUTED}
      motion={{ by: "character", from: { opacity: 0 }, duration: 0.5 }}>
      {label}
    </text>
  );

  // 1. The problem (0 - 5.5 s)
  const pains = [
    ["lock", "Closed, licensed abroad"],
    ["cloud-upload", "Plant data leaves the site"],
    ["triangle-alert", "Answers taken on trust"],
  ];
  p.compose(
    <frame width={W} height={H} layout="none" background={glow} motion={{ exit }}>
      {eyebrow("THE PROBLEM", 250)}
      <text x={210} y={320} width={1500} align="center" fontFamily={FONT} fontSize={78}
        fontWeight={700} lineHeight={1.12} color={WHITE}
        motion={{ by: "word", from: { opacity: 0, y: 26 }, at: 0.3, duration: 1.1, overlap: 0.6 }}>
        Plants, grids and refineries plan with imported optimization solvers.
      </text>
      {pains.map(([name, label], i) => (
        <frame x={90 + i * 600} y={640} width={540} height={112} at={2.0 + i * 0.35}
          layout="row" align="center" gap={22} padding={{ left: 34, right: 24 }}
          background={CARD} radius={20} motion={rise()}>
          {icon(name, { size: 40, color: WARN })}
          <text fontFamily={FONT} fontSize={28} fontWeight={600} color={WHITE}>{label}</text>
        </frame>
      ))}
    </frame>,
    { at: 0, dur: 5.5, name: "Problem" },
  );

  // 2. The solution (5.5 - 11 s)
  const steps = [
    ["globe", "Submit on the web", WHITE],
    ["cpu", "Solve on your CPU or GPU", WHITE],
    ["shield-check", "Get a verified result", ACCENT],
  ];
  p.compose(
    <frame width={W} height={H} layout="none" background={glow} motion={{ exit }}>
      {eyebrow("THE SOLUTION", 200)}
      <frame x={360} y={250} width={1200} height={190} layout="none" origin="center"
        motion={{ enter: { from: { scale: 0.9, opacity: 0 }, duration: 0.7, easing: "ease-out" } }}>
        <text x={0} y={0} width={1200} align="center" fontFamily={FONT} fontSize={168}
          fontWeight={900} letterSpacing={-4} color={WHITE}>Sovereign</text>
      </frame>
      <text x={260} y={455} width={1400} align="center" fontFamily={FONT} fontSize={38}
        fontWeight={400} color={SOFT} at={0.6}
        motion={{ by: "word", from: { opacity: 0, y: 14 }, duration: 0.8, overlap: 0.6 }}>
        An indigenous LP, MILP and QP engine that runs on your own machine.
      </text>
      {steps.map(([name, label, color], i) => (
        <frame x={90 + i * 600} y={680} width={540} height={130} at={1.5 + i * 0.7}
          layout="row" align="center" gap={24} padding={{ left: 36, right: 24 }}
          background={CARD} radius={22} motion={rise()}>
          {icon(name, { size: 48, color })}
          <text fontFamily={FONT} fontSize={29} fontWeight={600} color={color}>{label}</text>
        </frame>
      ))}
      {[0, 1].map((i) => (
        <frame x={640 + i * 600} y={742} width={40} height={6} radius={3} background={MUTED}
          at={1.95 + i * 0.7} reveal={{ from: "left", duration: 0.35 }} />
      ))}
    </frame>,
    { at: 5.5, dur: 5.5, name: "Solution" },
  );

  // 3. The proof (11 - 16.5 s)
  const stats = [
    { value: 49, suffix: "/61", decimals: 0, label: "benchmark runs match HiGHS, each independently verified" },
    { text: "0", label: "numerical errors across those 61 runs" },
    { text: "1M", label: "variables in a verified LP and QP solve" },
    { value: 1.6, suffix: "x", decimals: 1, label: "CUDA speed-up on large dense LPs" },
  ];
  p.compose(
    <frame width={W} height={H} layout="none" background={glow} motion={{ exit }}>
      {eyebrow("MEASURED, NOT CLAIMED", 210)}
      <text x={210} y={275} width={1500} align="center" fontFamily={FONT} fontSize={60}
        fontWeight={700} color={WHITE}
        motion={{ by: "word", from: { opacity: 0, y: 20 }, at: 0.2, duration: 0.8, overlap: 0.6 }}>
        Checked against HiGHS, the open reference solver.
      </text>
      {stats.map((s, i) => (
        <frame x={100 + i * 440} y={470} width={400} height={330} at={0.9 + i * 0.3}
          layout="none" background={CARD} radius={24}
          motion={{
            enter: { from: { y: 30, opacity: 0 }, duration: 0.5 },
            ...(s.value !== undefined
              ? { timeline: { at: 0.2, duration: 1.2, easing: "ease-out",
                  targets: [{ target: "Value", counter: { from: 0, to: s.value, decimals: s.decimals, suffix: s.suffix } }] } }
              : {}),
          }}>
          <frame name="Value" x={0} y={60} width={400} height={120} layout="none">
            <text x={0} y={0} width={400} align="center" fontFamily={FONT} fontSize={104}
              fontWeight={900} color={ACCENT}>{s.text ?? `0${s.suffix}`}</text>
          </frame>
          <text x={40} y={205} width={320} align="center" fontFamily={FONT} fontSize={27}
            lineHeight={1.3} color={SOFT}>{s.label}</text>
        </frame>
      ))}
      <text x={210} y={860} width={1500} align="center" fontFamily={FONT} fontSize={24}
        color={MUTED} at={2.4} motion={{ by: "line", from: { opacity: 0 }, duration: 0.5 }}>
        Single runs on one laptop (RTX 2050), 30 s per job. Sparse models run faster on the CPU.
      </text>
    </frame>,
    { at: 11, dur: 5.5, name: "Proof" },
  );

  // 4. Close (16.5 - 20 s)
  p.compose(
    <frame width={W} height={H} layout="none" background={glow}>
      <text x={160} y={330} width={1600} align="center" fontFamily={FONT} fontSize={104}
        fontWeight={900} color={WHITE}
        motion={{ by: "word", from: { opacity: 0, y: 30 }, duration: 0.7, overlap: 0.5 }}>
        Submit on the web.
      </text>
      <text x={160} y={470} width={1600} align="center" fontFamily={FONT} fontSize={104}
        fontWeight={900} color={ACCENT} at={0.6}
        motion={{ by: "word", from: { opacity: 0, y: 30 }, duration: 0.7, overlap: 0.5 }}>
        Solve on your machine.
      </text>
      <text x={210} y={700} width={1500} align="center" fontFamily={FONT} fontSize={30}
        fontWeight={600} letterSpacing={6} color={MUTED} at={1.4}
        motion={{ by: "line", from: { opacity: 0 }, duration: 0.6 }}>
        SOVEREIGN  ·  SIH 26119  ·  LP · MILP · CONVEX QP
      </text>
    </frame>,
    { at: 16.5, dur: 3.5, name: "Close" },
  );

  if (!globalThis.process?.env?.PREVIEW) await p.render("renders/sovereign-explainer.mp4");
};
