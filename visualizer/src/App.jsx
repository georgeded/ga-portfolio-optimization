import { useState } from 'react'

// params from genetic_algorithm.py
const P = {
  K_MIN: 10, K_MAX: 30, W_MIN: 0.02, W_MAX: 0.15,
  POP: 100, GENS: 200, ELITE_FRAC: 0.05,
  TOURNAMENT: 3, LOCAL_ITER: 5, LOCAL_STEP: 0.01, EARLY_STOP: 20,
  PC: 0.6054, PM: 0.1370, SIGMA_M: 0.1469, LAMBDA: 1.8437, RUNNERS: 8,
}

// 8-stock demo, real scale is ~867 stocks
const STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ']
const COLORS = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#06b6d4','#f97316','#84cc16']

// K=7 demo portfolios, all weights in [0.05, 0.15], sum=1.00
const POP0 = [
  { w: [0.15,0.14,0.14,0.14,0.15,0.14,0.14,0],    sharpe:1.28, to:0.088, f:1.12 },
  { w: [0,0.15,0.13,0.15,0.14,0.14,0.15,0.14],    sharpe:0.96, to:0.047, f:0.87 },
  { w: [0.13,0,0.15,0.14,0.15,0.15,0.14,0.14],    sharpe:1.02, to:0.043, f:0.94 },
  { w: [0.14,0.15,0.15,0,0.14,0.15,0.13,0.14],    sharpe:0.84, to:0.071, f:0.71 },
  { w: [0.14,0.14,0.14,0.15,0,0.15,0.14,0.14],    sharpe:0.79, to:0.060, f:0.68 },
  { w: [0.13,0.14,0.15,0.15,0.14,0,0.15,0.14],    sharpe:0.93, to:0.060, f:0.82 },
]

const GEN1 = [
  { w: [0.15,0.14,0.14,0.14,0.15,0.14,0.14,0],    f:1.16, tag:'elite + refined' },
  { w: [0.15,0,0.14,0.15,0.15,0.14,0.13,0.14],    f:1.09, tag:'new' },
  { w: [0.13,0.14,0.15,0,0.15,0.15,0.14,0.14],    f:0.97, tag:'new' },
  { w: [0.13,0.15,0.15,0.14,0,0.15,0.14,0.14],    f:0.94 },
  { w: [0,0.15,0.13,0.15,0.14,0.14,0.15,0.14],    f:0.87 },
  { w: [0.13,0.14,0.15,0.15,0.14,0,0.15,0.14],    f:0.82 },
]

// shared ui bits

// portfolio card with bars + fitness score
function PortCard({ p, label, hl, barHeight = 80 }) {
  const k = p.w.filter(v => v > 0).length
  const maxW = Math.max(...p.w, 0.01)
  const borderColor = hl === 'blue' ? 'var(--accent)' : hl === 'green' ? 'var(--green)' : hl === 'amber' ? 'var(--amber)' : 'var(--border)'
  const bgColor = hl ? `color-mix(in srgb, ${borderColor} 7%, var(--card))` : 'var(--card)'
  const labelH = 16
  const pctH = 14

  return (
    <div style={{ background: bgColor, border: `1px solid ${borderColor}`, borderRadius: 10, padding: '12px 14px', transition: 'all .2s' }}>
      {label && <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: 0.8, textTransform: 'uppercase', color: 'var(--text3)', marginBottom: 8 }}>{label}</div>}
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: barHeight + labelH + pctH, marginBottom: 8 }}>
        {p.w.map((v, i) => (
          <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-end', height: '100%' }}>
            <div style={{ width: '100%', height: v > 0 ? Math.max((v / maxW) * barHeight, 4) : 0, background: v > 0 ? COLORS[i] : 'transparent', borderRadius: '3px 3px 0 0', transition: 'height .4s', position: 'relative' }}>
              {v > 0 && (
                <span style={{ position: 'absolute', bottom: '100%', left: 0, right: 0, textAlign: 'center', fontSize: 11, fontWeight: 700, color: COLORS[i], lineHeight: `${pctH}px` }}>
                  {Math.round(v * 100)}%
                </span>
              )}
            </div>
            <div style={{ height: labelH, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 600, color: v > 0 ? COLORS[i] : 'transparent', width: '100%', overflow: 'hidden' }}>
              {STOCKS[i].slice(0, 4)}
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontFamily: 'var(--mono)', fontSize: 13 }}>
        <span style={{ fontSize: 11, background: 'var(--tag-bg)', color: 'var(--tag-text)', borderRadius: 4, padding: '2px 7px', fontWeight: 600 }}>K={k}</span>
        <span style={{ fontWeight: 700, color: p.f >= 1.0 ? 'var(--green)' : p.f >= 0.85 ? 'var(--accent)' : 'var(--text2)' }}>f={p.f.toFixed(2)}</span>
        {p.tag && <span style={{ fontSize: 11, color: 'var(--green)', fontWeight: 600 }}>{p.tag}</span>}
      </div>
    </div>
  )
}

// generic bordered card
function Card({ children, hl, style = {} }) {
  const borderColor = hl === 'blue' ? 'var(--accent)' : hl === 'green' ? 'var(--green)' : hl === 'amber' ? 'var(--amber)' : 'var(--border)'
  const bgColor = hl ? `color-mix(in srgb, ${borderColor} 6%, var(--card))` : 'var(--card)'
  return (
    <div style={{ background: bgColor, border: `1px solid ${borderColor}`, borderRadius: 10, padding: '14px 16px', ...style }}>
      {children}
    </div>
  )
}

// section label (small caps)
function SecLabel({ children }) {
  return <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.8, textTransform: 'uppercase', color: 'var(--text3)', marginBottom: 10 }}>{children}</div>
}

// param badge
function ParamTag({ children }) {
  return <span style={{ fontSize: 13, fontFamily: 'var(--mono)', background: 'var(--tag-bg)', color: 'var(--tag-text)', borderRadius: 5, padding: '4px 10px' }}>{children}</span>
}

// colored badge
function Tag({ children, color = 'blue' }) {
  const colors = { blue: ['var(--tag-bg)', 'var(--accent)'], green: ['color-mix(in srgb,var(--green) 12%,var(--card))', 'var(--green)'], amber: ['color-mix(in srgb,var(--amber) 12%,var(--card))', 'var(--amber)'] }
  const [bg, fg] = colors[color] || colors.blue
  return <span style={{ fontSize: 12, fontFamily: 'var(--mono)', padding: '2px 8px', borderRadius: 4, fontWeight: 600, background: bg, color: fg }}>{children}</span>
}

// bullet list with html support
function Bullets({ items }) {
  return (
    <ul style={{ listStyle: 'none', paddingLeft: 0, marginBottom: 0 }}>
      {items.map((item, i) => (
        <li key={i} style={{ fontSize: 14, color: item.dim ? 'var(--text3)' : 'var(--text2)', lineHeight: 1.9, display: 'flex', gap: 8 }}>
          <span style={{ color: 'var(--accent)', flexShrink: 0 }}>•</span>
          <span dangerouslySetInnerHTML={{ __html: item.text || item }} />
        </li>
      ))}
    </ul>
  )
}

// flex row for param tags at the bottom
function ParamRow({ children }) {
  return <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 14 }}>{children}</div>
}

// step functions

function StepPopulation() {
  return (
    <div>
      <Card style={{ marginBottom: 12 }}>
        <Bullets items={[
          { text: 'A <strong>chromosome</strong> encodes weights for K stocks out of ~867' },
          { text: 'K ∈ [10, 30] active stocks, weights ∈ [0.02, 0.15], all summing to 1' },
        ]} />
      </Card>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 8 }}>
        {POP0.map((p, i) => <PortCard key={i} p={p} label={`Portfolio ${i + 1}`} barHeight={60} />)}
      </div>
      <p style={{ fontSize: 12, color: 'var(--text3)', textAlign: 'center', marginBottom: 10 }}>
        Demo uses K=7 for visual clarity. Real portfolios: K∈[10,30], weights ∈ [0.02, 0.15]
      </p>
      <ParamRow>
        <ParamTag>POP_SIZE = 100</ParamTag>
        <ParamTag>K_MIN = 10</ParamTag>
        <ParamTag>K_MAX = 30</ParamTag>
        <ParamTag>W_MIN = 0.02</ParamTag>
        <ParamTag>W_MAX = 0.15</ParamTag>
      </ParamRow>
    </div>
  )
}

function StepFitness() {
  return (
    <div>
      <div style={{ background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 8, padding: '18px 24px', textAlign: 'center', fontFamily: 'var(--mono)', fontSize: 22, marginBottom: 14 }}>
        <span style={{ color: 'var(--green)', fontWeight: 700 }}>f</span>(w) =&nbsp;
        <span style={{ color: 'var(--accent)', fontWeight: 700 }}>Sharpe(w<sub>held</sub>)</span>&nbsp;−&nbsp;
        <span style={{ color: 'var(--amber)', fontWeight: 700 }}>λ</span>&nbsp;·&nbsp;
        <span style={{ color: 'var(--red)', fontWeight: 700 }}>Turnover(w, w<sub>prev</sub>)</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
        <Card>
          <SecLabel>Sparse-Aware Sharpe</SecLabel>
          <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.8 }}>
            Only held stocks (w &gt; 0) enter μ and Σ.
            Zero-weight positions are excluded from the calculation.
          </p>
        </Card>
        <Card>
          <SecLabel>Turnover Penalty</SecLabel>
          <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.8 }}>
            Turnover = Σ|w − w<sub>prev</sub>| / 2. λ = 1.8437, tuned by Optuna.
            Penalty on rebalancing is the main cause of GA underperformance.
          </p>
        </Card>
      </div>
      <Card>
        <SecLabel>Example: Portfolio 1 (K=7, weights 13-15%)</SecLabel>
        <div style={{ display: 'flex', gap: 32, alignItems: 'center', justifyContent: 'center', padding: '10px 0' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 13, color: 'var(--text3)', marginBottom: 4 }}>Sharpe</div>
            <div style={{ fontSize: 26, fontWeight: 700, color: 'var(--accent)', fontFamily: 'var(--mono)' }}>1.28</div>
          </div>
          <div style={{ fontSize: 22, color: 'var(--text3)', fontFamily: 'var(--mono)' }}>−</div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 13, color: 'var(--text3)', marginBottom: 4 }}>1.8437 × 0.088</div>
            <div style={{ fontSize: 26, fontWeight: 700, color: 'var(--red)', fontFamily: 'var(--mono)' }}>0.162</div>
          </div>
          <div style={{ fontSize: 22, color: 'var(--text3)', fontFamily: 'var(--mono)' }}>=</div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 13, color: 'var(--text3)', marginBottom: 4 }}>Fitness f(w)</div>
            <div style={{ fontSize: 26, fontWeight: 700, color: 'var(--green)', fontFamily: 'var(--mono)' }}>1.12</div>
          </div>
        </div>
      </Card>
      <ParamRow>
        <ParamTag>λ = 1.8437 (Optuna-tuned)</ParamTag>
        <ParamTag>Turnover = Σ|w − w_prev| / 2</ParamTag>
      </ParamRow>
    </div>
  )
}

function StepSelection() {
  const tournaments = [
    { label: 'Tournament 1 → Parent 1', candidates: [{ i: 0, f: 1.12 }, { i: 3, f: 0.71 }, { i: 4, f: 0.68 }], winner: 0, color: 'var(--accent)', hl: 'blue' },
    { label: 'Tournament 2 → Parent 2', candidates: [{ i: 1, f: 0.87 }, { i: 2, f: 0.94 }, { i: 5, f: 0.82 }], winner: 2, color: 'var(--green)', hl: 'green' },
  ]
  return (
    <div>
      <Card style={{ marginBottom: 12 }}>
        <Bullets items={[
          { text: 'Draw <strong>3 candidates</strong> at random from the population' },
          { text: 'Highest fitness wins. Repeat independently for Parent 2.' },
        ]} />
      </Card>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
        {tournaments.map(({ label, candidates, winner, color, hl }) => (
          <Card key={label}>
            <SecLabel>{label}</SecLabel>
            {candidates.map(({ i, f }) => {
              const isW = i === winner
              return (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, padding: '10px 14px', borderRadius: 7, border: `1px solid ${isW ? color : 'var(--border)'}`, background: isW ? `color-mix(in srgb, ${color} 8%, var(--card))` : 'transparent' }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 14, flex: 1 }}>
                    Portfolio {i + 1}&nbsp;&nbsp;<span style={{ color: isW ? color : 'var(--text2)', fontWeight: isW ? 700 : 400 }}>f = {f.toFixed(2)}</span>
                  </span>
                  {isW && <Tag color={hl === 'blue' ? 'blue' : 'green'}>WINNER ✓</Tag>}
                </div>
              )
            })}
          </Card>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 16, alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ flex: 1 }}><PortCard p={POP0[0]} label="Parent 1" hl="blue" barHeight={90} /></div>
        <span style={{ fontSize: 30, color: 'var(--text3)', flexShrink: 0 }}>×</span>
        <div style={{ flex: 1 }}><PortCard p={POP0[2]} label="Parent 2" hl="green" barHeight={90} /></div>
      </div>
      <ParamRow>
        <ParamTag>TOURNAMENT = 3</ParamTag>
        <ParamTag>winner = argmax fitness</ParamTag>
      </ParamRow>
    </div>
  )
}

function StepCrossover() {
  const p1 = { w: [0.15, 0.14, 0.14, 0.14, 0.15, 0.14, 0.14, 0], f: 1.12 }
  const p2 = { w: [0.13, 0, 0.15, 0.14, 0.15, 0.15, 0.14, 0.14], f: 0.94 }
  const child = { w: [0.14, 0, 0.14, 0.15, 0.15, 0.14, 0.14, 0.14], f: 1.09 }

  return (
    <div>
      <Card style={{ marginBottom: 14 }}>
        <Bullets items={[
          { text: 'Fires with prob <strong>pc = 0.6054</strong>. Stocks sampled from the union of both parents.' },
          { text: 'Weights blended as α·w<sub>1</sub> + (1−α)·w<sub>2</sub>, α ~ Uniform(0,1). Result is repaired.' },
        ]} />
      </Card>
      <div style={{ display: 'flex', gap: 14, alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ flex: 1 }}><PortCard p={p1} label="Parent 1" hl="blue" barHeight={100} /></div>
        <span style={{ fontSize: 26, color: 'var(--text3)', flexShrink: 0, fontFamily: 'var(--mono)' }}>+</span>
        <div style={{ flex: 1 }}><PortCard p={p2} label="Parent 2" hl="green" barHeight={100} /></div>
        <span style={{ fontSize: 26, color: 'var(--text3)', flexShrink: 0, fontFamily: 'var(--mono)' }}>→</span>
        <div style={{ flex: 1 }}><PortCard p={child} label="Child (after repair)" hl="amber" barHeight={100} /></div>
      </div>
      <ParamRow>
        <ParamTag>PC = {P.PC}</ParamTag>
        <ParamTag>α ~ Uniform(0, 1)</ParamTag>
        <ParamTag>k_child ∈ [K_MIN, K_MAX]</ParamTag>
      </ParamRow>
    </div>
  )
}

function StepMutation() {
  const before = { w: [0.14, 0.15, 0, 0.14, 0.14, 0.15, 0.14, 0.14], f: 0.97 }
  const after  = { w: [0.14, 0, 0.12, 0.15, 0.14, 0.15, 0.15, 0.15], f: 1.01 }

  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 14 }}>
        <Card>
          <SecLabel>Mutation 1: Gaussian Weight Noise</SecLabel>
          <Bullets items={[
            { text: 'Fires with prob <strong>pm = 0.1370</strong>' },
            { text: `Adds N(0, σ=${P.SIGMA_M}) to active weights, clips to [0, 1]` },
          ]} />
        </Card>
        <Card>
          <SecLabel>Mutation 2: Asset Swap</SecLabel>
          <Bullets items={[
            { text: 'Fires with prob <strong>pm = 0.1370</strong>' },
            { text: `Drops one active stock, adds one inactive at W_MIN = ${P.W_MIN}` },
          ]} />
        </Card>
      </div>
      <div style={{ display: 'flex', gap: 20, alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ flex: 1 }}><PortCard p={before} label="Before mutation" barHeight={100} /></div>
        <div style={{ textAlign: 'center', flexShrink: 0 }}>
          <div style={{ fontSize: 30, color: 'var(--text3)' }}>→</div>
          <div style={{ fontSize: 13, color: 'var(--text3)' }}>+ repair</div>
        </div>
        <div style={{ flex: 1 }}><PortCard p={after} label="After mutation + repair" hl="amber" barHeight={100} /></div>
      </div>
      <ParamRow>
        <ParamTag>PM = {P.PM}</ParamTag>
        <ParamTag>σ_M = {P.SIGMA_M}</ParamTag>
        <ParamTag>W_MIN = {P.W_MIN}</ParamTag>
      </ParamRow>
    </div>
  )
}

function StepRepair() {
  return (
    <div>
      <Card style={{ marginBottom: 12 }}>
        <SecLabel>Bounded Simplex Projection</SecLabel>
        <Bullets items={[
          { text: 'Applied after every genetic op: cardinality fix, then bisection projection' },
          { text: 'Bisection finds λ* so clip(w−λ*, W_MIN, W_MAX) sums exactly to 1' },
          { text: 'If projection shifts cardinality, recurse once (max depth = 1)' },
        ]} />
      </Card>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 12 }}>
        <Card>
          <SecLabel>Step 1: Cardinality fix</SecLabel>
          <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.8 }}>
            K &gt; K_MAX: drop smallest weights.<br />
            K &lt; K_MIN: activate random zeros at W_MIN.
          </p>
        </Card>
        <Card>
          <SecLabel>Step 2: Project</SecLabel>
          <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.8 }}>
            Bisection finds λ* such that clipping w−λ* to [W_MIN, W_MAX] sums to 1. Guaranteed feasible.
          </p>
        </Card>
        <Card>
          <SecLabel>Step 3: Re-check</SecLabel>
          <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.8 }}>
            If projection changed cardinality, run steps 1 and 2 once more. Max recursion depth = 1.
          </p>
        </Card>
      </div>
      <ParamRow>
        <ParamTag>W_MIN = {P.W_MIN}</ParamTag>
        <ParamTag>W_MAX = {P.W_MAX}</ParamTag>
        <ParamTag>K_MIN = {P.K_MIN}</ParamTag>
        <ParamTag>K_MAX = {P.K_MAX}</ParamTag>
        <ParamTag>EARLY_STOP = 20 stagnant gens</ParamTag>
        <ParamTag>N_GENS = 200</ParamTag>
      </ParamRow>
    </div>
  )
}

function StepNewGen() {
  const runs = [
    { seed: 0, f: 1.09 }, { seed: 1, f: 1.15 }, { seed: 2, f: 1.07 },
    { seed: 3, f: 1.12 }, { seed: 4, f: 1.06 }, { seed: 5, f: 1.14 },
    { seed: 6, f: 1.08 }, { seed: 7, f: 1.10 },
  ]
  const sorted = [...runs].sort((a, b) => a.f - b.f)
  const median = (sorted[3].f + sorted[4].f) / 2
  const canonical = runs.reduce((best, r) => Math.abs(r.f - median) < Math.abs(best.f - median) ? r : best, runs[0])
  const maxF = Math.max(...runs.map(r => r.f))
  const minF = Math.min(...runs.map(r => r.f))

  return (
    <div>
      {/* top row: elitism + local refinement */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
        <Card hl="blue">
          <SecLabel>Elitism</SecLabel>
          <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.8, marginBottom: 8 }}>
            The top 5 chromosomes (ELITE_FRAC = 5%) are copied unchanged into the next generation.
            No crossover or mutation is applied to them.
          </p>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--accent)' }}>
            top 5 of 100 → carry forward intact
          </div>
        </Card>
        <Card hl="amber">
          <SecLabel>Local Refinement</SecLabel>
          <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.8, marginBottom: 8 }}>
            Applied to the <strong>single best elite</strong> after each generation.
            Makes 5 greedy weight-shift attempts of δ = 0.01; keeps a shift only if fitness improves.
          </p>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--amber)' }}>
            f: 1.12 → 1.16 after 5 shifts
          </div>
        </Card>
      </div>

      {/* bottom: 8 parallel runs */}
      <Card>
        <SecLabel>8 Parallel Runs (seeds 1000–1007) — canonical = closest to median</SecLabel>
        {runs.map(r => {
          const isC = r.seed === canonical.seed
          const pct = ((r.f - minF) / (maxF - minF + 0.01)) * 70 + 20
          return (
            <div key={r.seed} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
              <span style={{ fontSize: 13, fontFamily: 'var(--mono)', color: 'var(--text2)', width: 52, flexShrink: 0 }}>seed {r.seed}</span>
              <div style={{ flex: 1, height: 26, borderRadius: 5, background: 'var(--bg2)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${pct}%`, background: isC ? 'var(--green)' : 'var(--accent)', borderRadius: 5, display: 'flex', alignItems: 'center', paddingLeft: 8, fontSize: 12, fontFamily: 'var(--mono)', fontWeight: 700, color: '#fff', transition: 'width .5s' }}>
                  f={r.f.toFixed(2)}
                </div>
              </div>
              {isC && <Tag color="green">canonical</Tag>}
            </div>
          )
        })}
      </Card>

      <ParamRow>
        <ParamTag>ELITE_FRAC = 0.05 → 5 elites</ParamTag>
        <ParamTag>LOCAL_ITER = 5</ParamTag>
        <ParamTag>LOCAL_STEP = 0.01</ParamTag>
        <ParamTag>N_RUNNERS = 8</ParamTag>
        <ParamTag>canonical = argmin |f_i − median(f)|</ParamTag>
      </ParamRow>
    </div>
  )
}

// step list
const STEPS = [
  { icon: '🧬', title: 'Population',   sub: '100 feasible portfolios, K ∈ [10, 30] active stocks',                                                      body: StepPopulation },
  { icon: '📊', title: 'Fitness',      sub: 'f(w) = Sharpe(w_held) - λ · Turnover(w, w_prev)   sparse-aware, λ = 1.8437',                              body: StepFitness },
  { icon: '🏆', title: 'Selection',    sub: 'Draw 3 candidates, take the highest fitness. Repeat for Parent 2.',                                         body: StepSelection },
  { icon: '🔀', title: 'Crossover',    sub: `Fires with prob pc = ${P.PC}. Weighted union sampling, arithmetic blend, then repair.`,                     body: StepCrossover },
  { icon: '🎲', title: 'Mutation',     sub: `Gaussian noise + asset swap, both pm = ${P.PM}, fire independently`,                                        body: StepMutation },
  { icon: '🔧', title: 'Repair',       sub: 'Bounded simplex projection enforces K, [W_MIN, W_MAX], sum=1 after every genetic operation.',             body: StepRepair },
  { icon: '🔄', title: 'New Gen',      sub: 'Elitism preserves top 5. Local refinement polishes the best. 8 seeds, median selection.',                  body: StepNewGen },
]

// app shell
export default function App() {
  const [step, setStep] = useState(0)
  const [dark, setDark] = useState(false)
  const S = STEPS[step]

  return (
    <div data-dark={dark} style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)', transition: 'background .25s, color .25s' }}>
      <div style={{ maxWidth: 960, margin: '0 auto', padding: '20px 24px' }}>

        {/* header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16, gap: 12 }}>
          <div>
            <h1 style={{ fontSize: 20, fontWeight: 700, letterSpacing: -0.3 }}>GA Portfolio Optimizer: Algorithm Walkthrough</h1>
            <p style={{ fontSize: 13, color: 'var(--text2)', marginTop: 4 }}>BSc Thesis · VU Amsterdam · Cardinality-Constrained Portfolio Optimization</p>
          </div>
          <button onClick={() => setDark(d => !d)} style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 8, padding: '8px 16px', fontSize: 13, cursor: 'pointer', color: 'var(--text2)', whiteSpace: 'nowrap', transition: 'all .2s' }}>
            {dark ? '☀ Light' : '🌙 Dark'}
          </button>
        </div>

        {/* step nav */}
        <div style={{ display: 'flex', border: '1px solid var(--border)', borderRadius: 10, overflow: 'hidden', marginBottom: 16 }}>
          {STEPS.map((s, i) => (
            <div key={i} onClick={() => setStep(i)} style={{
              flex: 1, padding: '8px 4px', textAlign: 'center', cursor: 'pointer',
              fontSize: 12, fontWeight: 600, borderRight: i < STEPS.length - 1 ? '1px solid var(--border)' : 'none',
              background: i === step ? 'var(--accent)' : i < step ? 'var(--tag-bg)' : 'var(--card)',
              color: i === step ? '#fff' : i < step ? 'var(--accent)' : 'var(--text3)',
              lineHeight: 1.4, transition: 'all .2s',
            }}>
              {s.icon}<br />{s.title}
            </div>
          ))}
        </div>

        {/* step title */}
        <div style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 10, padding: '14px 18px', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 14 }}>
          <span style={{ fontSize: 28, flexShrink: 0 }}>{S.icon}</span>
          <div style={{ flex: 1 }}>
            <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 3 }}>{S.title}</h2>
            <p style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.5 }}>{S.sub}</p>
          </div>
          <span style={{ fontSize: 13, color: 'var(--text3)', fontFamily: 'var(--mono)', whiteSpace: 'nowrap' }}>{step + 1} / {STEPS.length}</span>
        </div>

        {/* content */}
        <S.body />

        {/* nav buttons */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 18, gap: 12 }}>
          <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
            style={{ padding: '10px 24px', borderRadius: 8, fontSize: 14, fontWeight: 600, cursor: step === 0 ? 'default' : 'pointer', border: '1px solid var(--border)', background: 'var(--card)', color: 'var(--text)', opacity: step === 0 ? 0.3 : 1 }}>
            ← Back
          </button>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            {STEPS.map((_, i) => (
              <div key={i} onClick={() => setStep(i)} style={{
                height: 8, borderRadius: 4, cursor: 'pointer',
                width: i === step ? 20 : 8,
                background: i === step ? 'var(--accent)' : i < step ? 'var(--green)' : 'var(--border)',
                transition: 'all .2s',
              }} />
            ))}
          </div>
          <button onClick={() => setStep(s => Math.min(STEPS.length - 1, s + 1))} disabled={step === STEPS.length - 1}
            style={{ padding: '10px 24px', borderRadius: 8, fontSize: 14, fontWeight: 700, cursor: step === STEPS.length - 1 ? 'default' : 'pointer', border: 'none', background: step === STEPS.length - 1 ? 'var(--border)' : 'var(--accent)', color: step === STEPS.length - 1 ? 'var(--text3)' : '#fff' }}>
            Next →
          </button>
        </div>

        <div style={{ textAlign: 'center', marginTop: 10, fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--mono)' }}>
          Real scale: N≈867 stocks · K∈[10,30] · pop=100 · 200 gens · 8 parallel runs · 252 OOS periods
        </div>
      </div>
    </div>
  )
}
