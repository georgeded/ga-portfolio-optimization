import { useEffect, useState } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import { FaChartLine, FaDice, FaDna, FaMoon, FaRandom, FaSun, FaSyncAlt, FaTrophy, FaWrench } from 'react-icons/fa'

const FITNESS_TEX = String.raw`f(\mathbf{w}) = \operatorname{Sharpe}(\mathbf{w}) - \lambda\,\operatorname{Turnover}(\mathbf{w}, \mathbf{w}_{prev})`
const TURNOVER_TEX = String.raw`\operatorname{Turnover}(\mathbf{w}, \mathbf{w}_{prev}) = \frac{1}{2}\sum_i |w_i - w_{prev,i}|`

function Formula({ tex, display = false }) {
  const html = katex.renderToString(tex, { throwOnError: false, displayMode: display })
  return <span className={display ? 'formula formula-block' : 'formula'} dangerouslySetInnerHTML={{ __html: html }} />
}

function StepIcon({ step, color }) {
  const Icon = step.icon
  return <Icon size={22} color={color || step.iconColor} aria-hidden="true" />
}

function ThemeToggle({ dark, onChange }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={dark}
      aria-label="Toggle light or dark theme"
      onClick={() => onChange(d => !d)}
      className="theme-switch"
    >
      <span className={`theme-switch__thumb ${dark ? 'theme-switch__thumb--dark' : ''}`} />
      <span className={`theme-switch__option ${!dark ? 'theme-switch__option--active' : ''}`}>
        <FaSun size={13} aria-hidden="true" />
        Light
      </span>
      <span className={`theme-switch__option ${dark ? 'theme-switch__option--active' : ''}`}>
        <FaMoon size={12} aria-hidden="true" />
        Dark
      </span>
    </button>
  )
}

const P = {
  K_MIN: 10, K_MAX: 30, W_MIN: 0.02, W_MAX: 0.15,
  POP: 100, GENS: 200, ELITE_FRAC: 0.05,
  TOURNAMENT: 3, LOCAL_ITER: 5, LOCAL_STEP: 0.01, EARLY_STOP: 20,
  PC: 0.6054, PM: 0.1370, SIGMA_M: 0.1469, LAMBDA: 1.8437, RUNNERS: 8,
}

const STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'V', 'PG', 'XOM', 'UNH']
const COLORS = ['#2563eb','#059669','#d97706','#dc2626','#8b5cf6','#0891b2','#f97316','#65a30d','#be123c','#0d9488','#7c3aed','#ca8a04']

const POP0 = [
  { w: [0.12,0,0.15,0.06,0.10,0,0.02,0.14,0.09,0.13,0.08,0.11],       sharpe:1.28, to:0.088, f:1.12 },
  { w: [0,0.07,0.15,0.09,0.06,0,0.13,0.08,0.14,0.06,0.12,0.10],       sharpe:0.96, to:0.047, f:0.87 },
  { w: [0.05,0,0.11,0.15,0,0.08,0.13,0.10,0.07,0.12,0.09,0.10],       sharpe:1.02, to:0.043, f:0.94 },
  { w: [0,0.11,0.08,0.15,0.05,0.09,0,0.14,0.07,0.10,0.12,0.09],       sharpe:0.84, to:0.071, f:0.71 },
  { w: [0.10,0,0.07,0.12,0.15,0.05,0,0.11,0.07,0.13,0.08,0.12],       sharpe:0.79, to:0.060, f:0.68 },
  { w: [0,0.13,0.05,0.10,0.14,0.08,0,0.11,0.15,0.07,0.09,0.08],       sharpe:0.93, to:0.060, f:0.82 },
]

const GEN1 = [
  { w: [0.12,0,0.15,0.06,0.10,0,0.02,0.14,0.09,0.13,0.08,0.11],       f:1.16, tag:'elite + refined' },
  { w: [0.08,0.14,0,0.10,0.07,0,0.15,0.09,0.12,0.06,0.13,0.06],       f:1.09, tag:'new' },
  { w: [0.05,0,0.11,0.15,0,0.08,0.13,0.10,0.07,0.12,0.09,0.10],       f:0.97, tag:'new' },
  { w: [0,0.11,0.08,0.15,0.05,0.09,0,0.14,0.07,0.10,0.12,0.09],       f:0.94 },
  { w: [0,0.07,0.15,0.09,0.06,0,0.13,0.08,0.14,0.06,0.12,0.10],       f:0.87 },
  { w: [0,0.13,0.05,0.10,0.14,0.08,0,0.11,0.15,0.07,0.09,0.08],       f:0.82 },
]

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
      <div className="portfolio-bars-scroll">
        <div className="portfolio-bars" style={{ display: 'grid', gridTemplateColumns: `repeat(${p.w.length}, minmax(0, 1fr))`, alignItems: 'end', gap: 'var(--stock-gap)', height: barHeight + labelH + pctH, marginBottom: 6 }}>
          {p.w.map((v, i) => (
            <div className="stock-column" key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-end', height: '100%' }}>
              <div className="stock-bar" style={{ height: v > 0 ? Math.max((v / maxW) * barHeight, 4) : 0, background: v > 0 ? COLORS[i] : 'transparent' }}>
                {v > 0 && (
                  <span className="stock-percent" style={{ color: COLORS[i], lineHeight: `${pctH}px` }}>
                    {Math.round(v * 100)}%
                  </span>
                )}
              </div>
              <div className="stock-label" style={{ height: labelH, color: v > 0 ? COLORS[i] : 'transparent' }}>
                {STOCKS[i].slice(0, 4)}
              </div>
            </div>
          ))}
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontFamily: 'var(--mono)', fontSize: 13 }}>
        <span style={{ fontSize: 11, background: 'var(--tag-bg)', color: 'var(--tag-text)', borderRadius: 4, padding: '2px 7px', fontWeight: 600 }}><Formula tex={`K = ${k}`} /></span>
        <span style={{ fontWeight: 700, color: p.f >= 1.0 ? 'var(--green)' : p.f >= 0.85 ? 'var(--accent)' : 'var(--text2)' }}><Formula tex={`f = ${p.f.toFixed(2)}`} /></span>
        {p.tag && <span style={{ fontSize: 11, color: 'var(--green)', fontWeight: 600 }}>{p.tag}</span>}
      </div>
    </div>
  )
}

function Card({ children, hl, style = {} }) {
  const borderColor = hl === 'blue' ? 'var(--accent)' : hl === 'green' ? 'var(--green)' : hl === 'amber' ? 'var(--amber)' : 'var(--border)'
  const bgColor = hl ? `color-mix(in srgb, ${borderColor} 6%, var(--card))` : 'var(--card)'
  return (
    <div style={{ background: bgColor, border: `1px solid ${borderColor}`, borderRadius: 10, padding: '14px 16px', ...style }}>
      {children}
    </div>
  )
}

function SecLabel({ children }) {
  return <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.8, textTransform: 'uppercase', color: 'var(--text3)', marginBottom: 10 }}>{children}</div>
}

function ParamTag({ children }) {
  return <span style={{ display: 'inline-flex', maxWidth: '100%', overflowX: 'auto', fontSize: 13, fontFamily: 'var(--mono)', background: 'var(--tag-bg)', color: 'var(--tag-text)', borderRadius: 5, padding: '4px 10px' }}>{children}</span>
}

function Tag({ children, color = 'blue' }) {
  const colors = { blue: ['var(--tag-bg)', 'var(--accent)'], green: ['color-mix(in srgb,var(--green) 12%,var(--card))', 'var(--green)'], amber: ['color-mix(in srgb,var(--amber) 12%,var(--card))', 'var(--amber)'] }
  const [bg, fg] = colors[color] || colors.blue
  return <span style={{ fontSize: 12, fontFamily: 'var(--mono)', padding: '2px 8px', borderRadius: 4, fontWeight: 600, background: bg, color: fg }}>{children}</span>
}

function Bullets({ items }) {
  return (
    <ul style={{ listStyle: 'none', paddingLeft: 0, marginBottom: 0 }}>
      {items.map((item, i) => {
        const content = item.content || item.text || item
        return (
          <li key={i} style={{ fontSize: 14, color: item.dim ? 'var(--text3)' : 'var(--text2)', lineHeight: 1.9, display: 'flex', gap: 8 }}>
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: 'var(--accent)', flexShrink: 0, marginTop: 10 }} />
            {typeof content === 'string'
              ? <span dangerouslySetInnerHTML={{ __html: content }} />
              : <span>{content}</span>}
          </li>
        )
      })}
    </ul>
  )
}

function ParamRow({ children }) {
  return <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 14 }}>{children}</div>
}

function StepPopulation() {
  return (
    <div>
      <Card style={{ marginBottom: 12 }}>
        <Bullets items={[
          { content: <>A <strong>chromosome</strong> encodes weights for <Formula tex={"K"} /> stocks out of <Formula tex={"\\approx 867"} /></> },
          { content: <><Formula tex={"K \\in [10, 30]"} /> active stocks, weights <Formula tex={"\\in [0.02, 0.15]"} />, all summing to <Formula tex={"1"} /></> },
        ]} />
      </Card>
      <div className="portfolio-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(360px, 100%), 1fr))', gap: 10, marginBottom: 8 }}>
        {POP0.map((p, i) => <PortCard key={i} p={p} label={`Portfolio ${i + 1}`} barHeight={60} />)}
      </div>
      <p style={{ fontSize: 12, color: 'var(--text3)', textAlign: 'center', marginBottom: 10 }}>
        Demo uses <Formula tex={"K = 10"} /> for visual clarity. Real portfolios: <Formula tex={"K \\in [10,30]"} />, weights <Formula tex={"\\in [0.02, 0.15]"} />.
      </p>
      <ParamRow>
        <ParamTag><Formula tex={"\\text{POP\\_SIZE} = 100"} /></ParamTag>
        <ParamTag><Formula tex={"K_{\\min} = 10"} /></ParamTag>
        <ParamTag><Formula tex={"K_{\\max} = 30"} /></ParamTag>
        <ParamTag><Formula tex={"W_{\\min} = 0.02"} /></ParamTag>
        <ParamTag><Formula tex={"W_{\\max} = 0.15"} /></ParamTag>
      </ParamRow>
    </div>
  )
}

function StepFitness() {
  return (
    <div>
      <div className="formula-panel">
        <Formula tex={FITNESS_TEX} display />
      </div>
      <div className="mobile-one-col" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
        <Card>
          <SecLabel>Sparse-Aware Sharpe</SecLabel>
          <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.8 }}>
            Only held stocks (<Formula tex={"w_i > 0"} />) enter <Formula tex={"\\mu"} /> and <Formula tex={"\\Sigma"} />.
            Zero-weight positions are excluded from the calculation.
          </p>
        </Card>
        <Card>
          <SecLabel>Turnover Penalty</SecLabel>
          <div className="formula-inset">
            <Formula tex={TURNOVER_TEX} display />
          </div>
          <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.8 }}>
            <Formula tex={"\\lambda = 1.8437"} />, tuned by Optuna. Penalty on rebalancing is the main cause of GA underperformance.
          </p>
        </Card>
      </div>
      <Card>
        <SecLabel>Example: Portfolio 1 (<Formula tex={"K = 10"} />, weights 2-15%)</SecLabel>
        <div className="fitness-example mobile-stack" style={{ display: 'flex', gap: 32, alignItems: 'center', justifyContent: 'center', padding: '10px 0' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 13, color: 'var(--text3)', marginBottom: 4 }}>Sharpe</div>
            <div style={{ fontSize: 26, fontWeight: 700, color: 'var(--accent)', fontFamily: 'var(--mono)' }}>1.28</div>
          </div>
          <div style={{ fontSize: 22, color: 'var(--text3)', fontFamily: 'var(--mono)' }}><Formula tex={"-"} /></div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 13, color: 'var(--text3)', marginBottom: 4 }}><Formula tex={"1.8437 \\times 0.088"} /></div>
            <div style={{ fontSize: 26, fontWeight: 700, color: 'var(--red)', fontFamily: 'var(--mono)' }}>0.162</div>
          </div>
          <div style={{ fontSize: 22, color: 'var(--text3)', fontFamily: 'var(--mono)' }}>=</div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 13, color: 'var(--text3)', marginBottom: 4 }}>Fitness <Formula tex={"f(\\mathbf{w})"} /></div>
            <div style={{ fontSize: 26, fontWeight: 700, color: 'var(--green)', fontFamily: 'var(--mono)' }}>1.12</div>
          </div>
        </div>
      </Card>
      <ParamRow>
        <ParamTag><Formula tex={"\\lambda = 1.8437\\; (\\text{Optuna-tuned})"} /></ParamTag>
        <ParamTag><Formula tex={TURNOVER_TEX} /></ParamTag>
      </ParamRow>
    </div>
  )
}

function StepSelection() {
  const tournaments = [
    { key: 'tournament-1', label: <>Tournament 1 <Formula tex={"\\to"} /> Parent 1</>, candidates: [{ i: 0, f: 1.12 }, { i: 3, f: 0.71 }, { i: 4, f: 0.68 }], winner: 0, color: 'var(--accent)', hl: 'blue' },
    { key: 'tournament-2', label: <>Tournament 2 <Formula tex={"\\to"} /> Parent 2</>, candidates: [{ i: 1, f: 0.87 }, { i: 2, f: 0.94 }, { i: 5, f: 0.82 }], winner: 2, color: 'var(--green)', hl: 'green' },
  ]
  return (
    <div>
      <Card style={{ marginBottom: 12 }}>
        <Bullets items={[
          { text: 'Draw <strong>3 candidates</strong> at random from the population' },
          { text: 'Highest fitness wins. Repeat independently for Parent 2.' },
        ]} />
      </Card>
      <div className="mobile-one-col" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
        {tournaments.map(({ key, label, candidates, winner, color, hl }) => (
          <Card key={key}>
            <SecLabel>{label}</SecLabel>
            {candidates.map(({ i, f }) => {
              const isW = i === winner
              return (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, padding: '10px 14px', borderRadius: 7, border: `1px solid ${isW ? color : 'var(--border)'}`, background: isW ? `color-mix(in srgb, ${color} 8%, var(--card))` : 'transparent' }}>
                  <span style={{ fontFamily: 'var(--mono)', fontSize: 14, flex: 1 }}>
                    Portfolio {i + 1}&nbsp;&nbsp;<span style={{ color: isW ? color : 'var(--text2)', fontWeight: isW ? 700 : 400 }}><Formula tex={`f = ${f.toFixed(2)}`} /></span>
                  </span>
                  {isW && <Tag color={hl === 'blue' ? 'blue' : 'green'}>WINNER</Tag>}
                </div>
              )
            })}
          </Card>
        ))}
      </div>
      <div className="portfolio-flow mobile-stack" style={{ display: 'flex', gap: 10, alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ flex: 1 }}><PortCard p={POP0[0]} label="Parent 1" hl="blue" barHeight={58} /></div>
        <span style={{ fontSize: 30, color: 'var(--text3)', flexShrink: 0 }}><Formula tex={"\\times"} /></span>
        <div style={{ flex: 1 }}><PortCard p={POP0[2]} label="Parent 2" hl="green" barHeight={58} /></div>
      </div>
      <ParamRow>
        <ParamTag><Formula tex={"\\text{TOURNAMENT} = 3"} /></ParamTag>
        <ParamTag><Formula tex={"\\text{winner} = \\operatorname{argmax}\\,\\text{fitness}"} /></ParamTag>
      </ParamRow>
    </div>
  )
}

function StepCrossover() {
  const p1 = POP0[0]
  const p2 = POP0[2]
  const child = { w: [0.08,0.14,0,0.10,0.07,0,0.15,0.09,0.12,0.06,0.13,0.06], f: 1.09 }

  return (
    <div>
      <Card style={{ marginBottom: 10 }}>
        <Bullets items={[
          { content: <>Fires with prob <strong><Formula tex={"p_c = 0.6054"} /></strong>. Stocks sampled from the union of both parents.</> },
          { content: <>Weights blended as <Formula tex={"\\alpha\\mathbf{w}_1 + (1-\\alpha)\\mathbf{w}_2"} />, <Formula tex={"\\alpha \\sim \\operatorname{Uniform}(0,1)"} />. Result is repaired.</> },
        ]} />
      </Card>
      <div className="crossover-flow">
        <div className="crossover-card"><PortCard p={p1} label="Parent 1" hl="blue" barHeight={58} /></div>
        <span className="crossover-operator"><Formula tex={"+"} /></span>
        <div className="crossover-card"><PortCard p={p2} label="Parent 2" hl="green" barHeight={58} /></div>
        <span className="crossover-operator"><Formula tex={"\\downarrow"} /></span>
        <div className="crossover-card"><PortCard p={child} label="Child after repair" hl="amber" barHeight={58} /></div>
      </div>
      <ParamRow>
        <ParamTag><Formula tex={`P_C = ${P.PC}`} /></ParamTag>
        <ParamTag><Formula tex={"\\alpha \\sim \\operatorname{Uniform}(0, 1)"} /></ParamTag>
        <ParamTag><Formula tex={"k_{\\text{child}} \\in [K_{\\min}, K_{\\max}]"} /></ParamTag>
      </ParamRow>
    </div>
  )
}

function StepMutation() {
  const before = { w: [0,0.11,0.08,0.15,0.05,0.09,0,0.14,0.07,0.10,0.12,0.09], f: 0.97 }
  const after  = { w: [0.07,0.14,0,0.12,0.05,0.15,0,0.11,0.08,0.10,0.10,0.08], f: 1.01 }

  return (
    <div>
      <div className="mobile-one-col" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 10 }}>
        <Card>
          <SecLabel>Mutation 1: Gaussian Weight Noise</SecLabel>
          <Bullets items={[
            { content: <>Fires with prob <strong><Formula tex={"p_m = 0.1370"} /></strong></> },
            { content: <>Adds <Formula tex={`N(0, \\sigma = ${P.SIGMA_M})`} /> to active weights, clips to <Formula tex={"[0, 1]"} /></> },
          ]} />
        </Card>
        <Card>
          <SecLabel>Mutation 2: Asset Swap</SecLabel>
          <Bullets items={[
            { content: <>Fires with prob <strong><Formula tex={"p_m = 0.1370"} /></strong></> },
            { content: <>Drops one active stock, adds one inactive at <Formula tex={`W_{\\min} = ${P.W_MIN}`} /></> },
          ]} />
        </Card>
      </div>
      <div className="portfolio-flow mobile-stack" style={{ display: 'flex', gap: 10, alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ flex: 1 }}><PortCard p={before} label="Before mutation" barHeight={58} /></div>
        <div style={{ textAlign: 'center', flexShrink: 0 }}>
          <div style={{ fontSize: 30, color: 'var(--text3)' }}><Formula tex={"\\to"} /></div>
          <div style={{ fontSize: 13, color: 'var(--text3)' }}><Formula tex={"+"} /> repair</div>
        </div>
        <div style={{ flex: 1 }}><PortCard p={after} label="After mutation + repair" hl="amber" barHeight={58} /></div>
      </div>
      <ParamRow>
        <ParamTag><Formula tex={`P_M = ${P.PM}`} /></ParamTag>
        <ParamTag><Formula tex={`\\sigma_M = ${P.SIGMA_M}`} /></ParamTag>
        <ParamTag><Formula tex={`W_{\\min} = ${P.W_MIN}`} /></ParamTag>
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
          { content: <>Bisection finds <Formula tex={"\\lambda^*"} /> so <Formula tex={"\\operatorname{clip}(\\mathbf{w} - \\lambda^*, W_{\\min}, W_{\\max})"} /> sums exactly to <Formula tex={"1"} /></> },
          { content: <>If projection shifts cardinality, recurse once (max depth = <Formula tex={"1"} />)</> },
        ]} />
      </Card>
      <div className="mobile-one-col" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 12 }}>
        <Card>
          <SecLabel>Step 1: Cardinality fix</SecLabel>
          <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.8 }}>
            <Formula tex={"K > K_{\\max}"} />: drop smallest weights.<br />
            <Formula tex={"K < K_{\\min}"} />: activate random zeros at <Formula tex={"W_{\\min}"} />.
          </p>
        </Card>
        <Card>
          <SecLabel>Step 2: Project</SecLabel>
          <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.8 }}>
            Bisection finds <Formula tex={"\\lambda^*"} /> such that <Formula tex={"\\operatorname{clip}(\\mathbf{w} - \\lambda^*, W_{\\min}, W_{\\max})"} /> sums to <Formula tex={"1"} />. Guaranteed feasible.
          </p>
        </Card>
        <Card>
          <SecLabel>Step 3: Re-check</SecLabel>
          <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.8 }}>
            If projection changed cardinality, run steps 1 and 2 once more. Max recursion depth = <Formula tex={"1"} />.
          </p>
        </Card>
      </div>
      <ParamRow>
        <ParamTag><Formula tex={`W_{\\min} = ${P.W_MIN}`} /></ParamTag>
        <ParamTag><Formula tex={`W_{\\max} = ${P.W_MAX}`} /></ParamTag>
        <ParamTag><Formula tex={`K_{\\min} = ${P.K_MIN}`} /></ParamTag>
        <ParamTag><Formula tex={`K_{\\max} = ${P.K_MAX}`} /></ParamTag>
        <ParamTag><Formula tex={"\\text{EARLY\\_STOP} = 20\\; \\text{stagnant gens}"} /></ParamTag>
        <ParamTag><Formula tex={"N_{\\text{GENS}} = 200"} /></ParamTag>
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
      <div className="mobile-one-col" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
        <Card hl="blue">
          <SecLabel>Elitism</SecLabel>
          <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.8, marginBottom: 8 }}>
            The top 5 chromosomes (<Formula tex={"\\text{ELITE\\_FRAC} = 5\\%"} />) are copied unchanged into the next generation.
            No crossover or mutation is applied to them.
          </p>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--accent)' }}>
            <Formula tex={"\\text{top 5 of 100} \\to"} /> carry forward intact
          </div>
        </Card>
        <Card hl="amber">
          <SecLabel>Local Refinement</SecLabel>
          <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.8, marginBottom: 8 }}>
            Applied to the <strong>single best elite</strong> after each generation.
            Makes 5 greedy weight-shift attempts of <Formula tex={"\\delta = 0.01"} />; keeps a shift only if fitness improves.
          </p>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--amber)' }}>
            <Formula tex={"f\\colon 1.12 \\to 1.16"} /> after 5 shifts
          </div>
        </Card>
      </div>

      <Card>
        <SecLabel>8 Parallel Runs (seeds 1000-1007) - canonical = closest to median</SecLabel>
        {runs.map(r => {
          const isC = r.seed === canonical.seed
          const pct = ((r.f - minF) / (maxF - minF + 0.01)) * 70 + 20
          return (
            <div key={r.seed} className="run-row">
              <span className="run-seed">seed {r.seed}</span>
              <div className="run-track">
                <div className="run-fill" style={{ width: `${pct}%`, background: isC ? 'var(--green)' : 'var(--accent)' }}>
                  <span className="run-fill-label"><Formula tex={`f = ${r.f.toFixed(2)}`} /></span>
                </div>
              </div>
              {isC && <span className="run-tag"><Tag color="green">canonical</Tag></span>}
            </div>
          )
        })}
      </Card>

      <ParamRow>
        <ParamTag><Formula tex={"\\text{ELITE\\_FRAC} = 0.05 \\to 5\\;\\text{elites}"} /></ParamTag>
        <ParamTag><Formula tex={"\\text{LOCAL\\_ITER} = 5"} /></ParamTag>
        <ParamTag><Formula tex={"\\text{LOCAL\\_STEP} = 0.01"} /></ParamTag>
        <ParamTag><Formula tex={"N_{\\text{RUNNERS}} = 8"} /></ParamTag>
        <ParamTag><Formula tex={"\\text{canonical} = \\operatorname{argmin}\\, |f_i - \\operatorname{median}(f)|"} /></ParamTag>
      </ParamRow>
    </div>
  )
}

const STEPS = [
  { icon: FaDna, iconColor: '#8b5cf6', title: 'Population', sub: <>100 feasible portfolios, <Formula tex={"K \\in [10, 30]"} /> active stocks</>, body: StepPopulation },
  { icon: FaChartLine, iconColor: '#059669', title: 'Fitness', sub: <><Formula tex={FITNESS_TEX} /> sparse-aware, <Formula tex={"\\lambda = 1.8437"} /></>, body: StepFitness },
  { icon: FaTrophy, iconColor: '#d97706', title: 'Selection', sub: 'Draw 3 candidates, take the highest fitness. Repeat for Parent 2.', body: StepSelection },
  { icon: FaRandom, iconColor: '#2563eb', title: 'Crossover', sub: <>Fires with prob <Formula tex={`p_c = ${P.PC}`} />. Weighted union sampling, arithmetic blend, then repair.</>, body: StepCrossover },
  { icon: FaDice, iconColor: '#dc2626', title: 'Mutation', sub: <>Gaussian noise + asset swap, both <Formula tex={`p_m = ${P.PM}`} />, fire independently</>, body: StepMutation },
  { icon: FaWrench, iconColor: '#0891b2', title: 'Repair', sub: <>Bounded simplex projection enforces <Formula tex={"K"} />, <Formula tex={"[W_{\\min}, W_{\\max}]"} />, <Formula tex={"\\sum_i w_i = 1"} /> after every genetic operation.</>, body: StepRepair },
  { icon: FaSyncAlt, iconColor: '#7c3aed', title: 'New Gen', sub: 'Elitism preserves top 5. Local refinement polishes the best. 8 seeds, median selection.', body: StepNewGen },
]

export default function App() {
  const [step, setStep] = useState(0)
  const [dark, setDark] = useState(() => {
    if (typeof window === 'undefined') return false
    return window.localStorage.getItem('ga-theme') === 'dark'
  })
  const S = STEPS[step]

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem('ga-theme', dark ? 'dark' : 'light')
  }, [dark])

  return (
    <div data-dark={dark} style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)', transition: 'background .25s, color .25s' }}>
      <div className="app-container" style={{ maxWidth: 960, margin: '0 auto', padding: '20px 24px' }}>

        <div className="app-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16, gap: 12 }}>
          <div>
            <h1 style={{ fontSize: 20, fontWeight: 700, letterSpacing: 0 }}>GA Portfolio Optimizer: Algorithm Walkthrough</h1>
            <p style={{ fontSize: 13, color: 'var(--text2)', marginTop: 4 }}>BSc Thesis &middot; VU Amsterdam &middot; Cardinality-Constrained Portfolio Optimization</p>
          </div>
          <ThemeToggle dark={dark} onChange={setDark} />
        </div>

        <div className="step-nav" style={{ display: 'flex', border: '1px solid var(--border)', borderRadius: 10, overflow: 'hidden', marginBottom: 16 }}>
          {STEPS.map((s, i) => (
            <div className="step-tab" key={i} onClick={() => setStep(i)} style={{
              flex: 1, padding: '8px 4px', textAlign: 'center', cursor: 'pointer',
              fontSize: 12, fontWeight: 600, borderRight: i < STEPS.length - 1 ? '1px solid var(--border)' : 'none',
              background: i === step ? 'var(--accent)' : i < step ? 'var(--tag-bg)' : 'var(--card)',
              color: i === step ? '#fff' : i < step ? 'var(--accent)' : 'var(--text3)',
              lineHeight: 1.4, transition: 'all .2s', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2,
            }}>
              <StepIcon step={s} color={i === step ? '#fff' : s.iconColor} />
              <span>{s.title}</span>
            </div>
          ))}
        </div>

        <div className="step-title-card" style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 10, padding: '14px 18px', marginBottom: 14, display: 'flex', alignItems: 'center', gap: 14 }}>
          <span style={{ flexShrink: 0, display: 'flex', width: 28, justifyContent: 'center' }}><StepIcon step={S} /></span>
          <div style={{ flex: 1 }}>
            <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 3 }}>{S.title}</h2>
            <p style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.5 }}>{S.sub}</p>
          </div>
          <span style={{ fontSize: 13, color: 'var(--text3)', fontFamily: 'var(--mono)', whiteSpace: 'nowrap' }}>{step + 1} / {STEPS.length}</span>
        </div>

        <S.body />

        <div className="footer-nav" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 18, gap: 12 }}>
          <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
            style={{ padding: '10px 24px', borderRadius: 8, fontSize: 14, fontWeight: 600, cursor: step === 0 ? 'default' : 'pointer', border: '1px solid var(--border)', background: 'var(--card)', color: 'var(--text)', opacity: step === 0 ? 0.3 : 1 }}>
            &larr; Back
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
            Next &rarr;
          </button>
        </div>

        <div style={{ textAlign: 'center', marginTop: 10, fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--mono)' }}>
          Real scale: <Formula tex={"N \\approx 867"} /> stocks &middot; <Formula tex={"K \\in [10,30]"} /> &middot; <Formula tex={"\\text{pop} = 100"} /> &middot; <Formula tex={"200"} /> gens &middot; 8 parallel runs &middot; 252 OOS periods
        </div>
      </div>
    </div>
  )
}
