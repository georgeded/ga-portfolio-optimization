import { useEffect, useState } from 'react'

const STAGE_WIDTH = 1280
const STAGE_HEIGHT = 720

// demo scale: 12 of the 867 slots, K = 8 held
const POPULATION = [
  { id: 'P1', fitness: 1.12, weights: [.15, 0, .15, .14, 0, .13, .12, 0, .14, .15, 0, .02] },
  { id: 'P2', fitness: 0.87, weights: [0, .15, .13, 0, .14, .04, 0, .15, .13, 0, .13, .13] },
  { id: 'P3', fitness: 0.94, weights: [.14, .12, 0, .15, .13, 0, .15, .05, 0, .12, .14, 0] },
  { id: 'P4', fitness: 0.71, weights: [0, .15, .15, .06, 0, .15, .15, .04, .15, 0, .15, 0] },
  { id: 'P5', fitness: 0.68, weights: [.13, 0, .12, 0, .15, .11, .14, 0, .15, .13, 0, .07] },
  { id: 'P6', fitness: 0.82, weights: [.12, .14, 0, .13, .15, 0, .15, .08, .12, 0, .11, 0] },
]

const PARENT_A = POPULATION[0]
const PARENT_B = POPULATION[2]

// the P1 x P3 children at each pipeline stage, later steps follow child 1
const BLEND_CHILD     = [.15, .05, 0, .14, 0, .08, .13, 0, .08, .14, 0, .01]
const BLEND_CHILD_TWO = [.14, 0, .05, .15, .09, 0, .14, 0, 0, .13, .10, 0]
const MUTATED_CHILD   = [.18, .03, 0, .15, .02, 0, .09, 0, .10, .11, 0, 0]
// repair stage 1 reactivates one zero-weight slot at the 0.02 floor
const STAGE1_CHILD    = [.18, .03, 0, .15, .02, 0, .09, .02, .10, .11, 0, 0]
const REPAIRED_CHILD  = [.15, .09, 0, .15, .08, 0, .15, .08, .15, .15, 0, 0]
const REFINED_BEST    = [.15, 0, .15, .15, 0, .12, .11, 0, .15, .15, 0, .02]

const NOISE_AND_SWAP = { 0: '+0.03', 1: '-0.02', 3: '+0.01', 4: 'in', 5: 'out', 6: '-0.04', 8: '+0.02', 9: '-0.03', 11: '-0.03' }

const NEXT_GENERATION = [
  { id: '#1', fitness: 1.16, tag: 'elite, refined', dark: true, weights: REFINED_BEST },
  { id: '#2', fitness: 1.05, tag: 'new child', weights: REPAIRED_CHILD },
  { id: '#3', fitness: 0.98, tag: 'new child', weights: [0, .15, .12, 0, .15, .11, 0, .13, .15, .04, 0, .15] },
  { id: '#4', fitness: 0.94, tag: 'elite', dark: true, weights: PARENT_B.weights },
]

// seed 1002 sits closest to the median fitness
const RUNS = [
  { seed: 1000, fitness: 1.04 }, { seed: 1001, fitness: 1.16 },
  { seed: 1002, fitness: 1.10, selected: true }, { seed: 1003, fitness: 1.13 },
  { seed: 1004, fitness: 1.06 }, { seed: 1005, fitness: 1.15 },
  { seed: 1006, fitness: 1.10 }, { seed: 1007, fitness: 1.08 },
]

const TOURNAMENTS = [
  { title: 'Tournament 1', winnerId: 'P1', winnerTag: 'Parent 1', candidateIds: ['P1', 'P4', 'P5'] },
  { title: 'Tournament 2', winnerId: 'P3', winnerTag: 'Parent 2', candidateIds: ['P3', 'P2', 'P6'] },
]

function Chromosome({ weights, marks = {}, compact = false }) {
  return (
    <div className={compact ? 'chromosome chromosome-compact' : 'chromosome'}>
      {weights.map((value, i) => {
        const mark = marks[i]
        let cls = value > 0 ? 'cell cell-on' : 'cell cell-off'
        if (mark) cls += ' cell-marked'
        return (
          <span key={i} className="cell-slot">
            <span className={cls}>{compact ? '' : value === 0 ? '0' : value.toFixed(2)}</span>
            {mark && <span className="cell-mark">{mark}</span>}
          </span>
        )
      })}
      {!compact && <span className="cell cell-more">···</span>}
    </div>
  )
}

function StripRow({ label, weights, marks, right, tag, dark }) {
  const hasMarks = marks && Object.keys(marks).length > 0
  return (
    <div className={hasMarks ? 'strip strip-with-marks' : 'strip'}>
      <span className="strip-label">{label}</span>
      <Chromosome weights={weights} marks={marks} />
      <span className="strip-right">
        {right}
        {tag && <span className={dark ? 'chip chip-dark' : 'chip chip-soft'}>{tag}</span>}
      </span>
    </div>
  )
}

function CandidateRow({ portfolio, tag, dark, muted }) {
  let cls = 'candidate'
  if (dark) cls += ' candidate-winner'
  if (muted) cls += ' candidate-muted'
  return (
    <div className={cls}>
      <span className="candidate-name">{portfolio.id}</span>
      <Chromosome weights={portfolio.weights} compact />
      <span className="candidate-fitness">f = {portfolio.fitness.toFixed(2)}</span>
      {tag && <span className={dark ? 'chip chip-dark' : 'chip chip-soft'}>{tag}</span>}
    </div>
  )
}

function PopulationStep() {
  return (
    <>
      <div className="callout">
        <div className="panel-label">Encoding</div>
        <p className="panel-text">
          A portfolio is a chromosome of weights, one slot per asset.
          Only K slots are held, every held weight stays inside the bounds, and the weights sum to 1.
        </p>
        <div className="chip-row">
          <span className="chip chip-outline">N ≈ 867 assets</span>
          <span className="chip chip-outline">K ∈ [10, 30] held</span>
          <span className="chip chip-outline">0.02 ≤ w ≤ 0.15</span>
          <span className="chip chip-outline">Σw = 1</span>
        </div>
        <div className="formula-line">
          <span>f(w) = SR(w) - λ · τ(w, w<sub>prev</sub>)</span>
          <span className="formula-caption">
            monthly Sharpe from the 60 month window, τ = turnover, λ = 1.8437 (Optuna tuned)
          </span>
        </div>
      </div>
      <div className="strip-stack">
        {POPULATION.slice(0, 4).map(p => (
          <StripRow key={p.id} label={p.id} weights={p.weights} right={`f = ${p.fitness.toFixed(2)}`} />
        ))}
        <p className="strip-ghost">plus 96 more random feasible portfolios, population size 100</p>
      </div>
    </>
  )
}

function SelectionStep() {
  return (
    <div className="panel-pair">
      {TOURNAMENTS.map(t => (
        <div className="panel" key={t.title}>
          <div className="panel-label">{t.title}</div>
          {t.candidateIds.map(id => {
            const p = POPULATION.find(x => x.id === id)
            const won = id === t.winnerId
            return <CandidateRow key={id} portfolio={p} dark={won} tag={won ? t.winnerTag : null} />
          })}
        </div>
      ))}
    </div>
  )
}

function CrossoverStep() {
  return (
    <div className="strip-stack">
      <StripRow label={PARENT_A.id} weights={PARENT_A.weights} right={`f = ${PARENT_A.fitness.toFixed(2)}`} tag="Parent 1" dark />
      <div className="flow-divider">+</div>
      <StripRow label={PARENT_B.id} weights={PARENT_B.weights} right={`f = ${PARENT_B.fitness.toFixed(2)}`} tag="Parent 2" dark />
      <div className="flow-divider">
        each child draws its own K from [10, 30] and samples assets from the parents' union by average weight,
        <br />
        then blends with one shared α per child: Child 1 = 0.6 · P1 + 0.4 · P3, Child 2 = 0.7 · P3 + 0.3 · P1 with roles swapped
      </div>
      <StripRow label="Child 1" weights={BLEND_CHILD} right="Σw = 0.78" tag="repair pending" />
      <StripRow label="Child 2" weights={BLEND_CHILD_TWO} right="Σw = 0.80" tag="repair pending" />
    </div>
  )
}

function MutationStep() {
  return (
    <>
      <div className="panel-pair">
        <div className="panel">
          <div className="panel-label">Gaussian perturbation</div>
          <p className="panel-text">
            N(0, σ = 0.147) noise lands on every held weight.
            Anything pushed negative drops out of the portfolio.
          </p>
        </div>
        <div className="panel">
          <div className="panel-label">Asset swap</div>
          <p className="panel-text">
            One held stock is zeroed, one unheld stock enters at the 0.02 floor.
            Cardinality stays put, repair handles the rest.
          </p>
        </div>
      </div>
      <div className="strip-stack">
        <StripRow label="Child 1" weights={BLEND_CHILD} right="Σw = 0.78" />
        <div className="flow-divider">both fire here: noise shifts the held weights and pushes one negative, the swap trades one asset</div>
        <StripRow label="Mutated" weights={MUTATED_CHILD} marks={NOISE_AND_SWAP} right="Σw = 0.68" tag="repair pending" />
      </div>
    </>
  )
}

function RepairStep() {
  return (
    <>
      <div className="chip-row chip-row-centered">
        <span className="chip-row-label">called three times in the pipeline:</span>
        <span className="chip chip-soft">at initialisation</span>
        <span className="chip chip-soft">after crossover</span>
        <span className="chip chip-soft">after mutation</span>
      </div>
      <div className="strip-stack">
        <StripRow
          label="Mutated"
          weights={MUTATED_CHILD}
          marks={{ 0: '0.18 > cap' }}
          right="Σw = 0.68"
          tag="infeasible"
        />
        <div className="flow-divider">
          stage 1, cardinality: 7 held is too few, a random zero-weight stock enters at the 0.02 floor
        </div>
        <StripRow label="Stage 1" weights={STAGE1_CHILD} marks={{ 7: 'in at 0.02' }} right="Σw = 0.70" tag="8 held" />
        <div className="flow-divider">
          stage 2, projection: bisection finds one uniform shift λ so the clipped weights land in [0.02, 0.15] and sum to 1
        </div>
        <StripRow label="Repaired" weights={REPAIRED_CHILD} right="Σw = 1.00" tag="feasible" dark />
      </div>
      <div className="chip-row chip-row-centered">
        <span className="chip chip-outline">K stays in [10, 30]</span>
        <span className="chip chip-outline">0.02 ≤ w ≤ 0.15</span>
        <span className="chip chip-outline">Σw = 1</span>
      </div>
    </>
  )
}

function ElitismStep() {
  const ranked = [...POPULATION].sort((a, b) => b.fitness - a.fitness)
  return (
    <div className="panel panel-narrow">
      <div className="panel-label">Generation 1, ranked by fitness</div>
      {ranked.map((p, rank) => {
        const kept = rank < 2
        return (
          <CandidateRow
            key={p.id}
            portfolio={p}
            dark={kept}
            muted={!kept}
            tag={kept ? (rank === 0 ? 'kept, refined next' : 'kept') : 'replaced'}
          />
        )
      })}
    </div>
  )
}

function RefinementStep() {
  return (
    <>
      <div className="callout">
        <p className="panel-text">
          Up to five tries: pick two held stocks at random, shift 0.01 from one to the other,
          keep the move only when fitness improves. The 0.02 floor and 0.15 cap bound every shift.
        </p>
      </div>
      <div className="strip-stack">
        <StripRow label="Best" weights={PARENT_A.weights} right="f = 1.12" />
        <div className="flow-divider">two accepted shifts of 0.01</div>
        <StripRow
          label="Refined"
          weights={REFINED_BEST}
          marks={{ 3: '+0.01', 5: '-0.01', 6: '-0.01', 8: '+0.01' }}
          right="f = 1.16"
          tag="improved"
          dark
        />
      </div>
    </>
  )
}

function NewGenerationStep() {
  return (
    <>
      <div className="strip-stack">
        {NEXT_GENERATION.map(p => (
          <StripRow
            key={p.id}
            label={p.id}
            weights={p.weights}
            right={`f = ${p.fitness.toFixed(2)}`}
            tag={p.tag}
            dark={p.dark}
          />
        ))}
      </div>
      <p className="stat-line">best fitness 1.12 → 1.16 after one generation</p>
      <div className="panel">
        <div className="panel-label">One rebalance date = 8 seeded runs, the median one counts</div>
        <div className="run-row">
          {RUNS.map(r => (
            <span key={r.seed} className={r.selected ? 'run-chip run-chip-selected' : 'run-chip'}>
              {r.seed} · f {r.fitness.toFixed(2)}
            </span>
          ))}
        </div>
      </div>
    </>
  )
}

const STEPS = [
  {
    key: 'population',
    name: 'Population',
    title: 'Population',
    subtitle: 'The search opens with 100 random feasible portfolios',
    note: 'Each start: random K in [10, 30], Dirichlet weights, then repair. SR is computed on held stocks only. Demo shows 12 of 867 slots with K = 8. Thesis sections 4.1 to 4.3.',
    body: PopulationStep,
  },
  {
    key: 'selection',
    name: 'Selection',
    title: 'Selection',
    subtitle: 'Three random candidates compete, the fittest becomes a parent',
    note: 'Drawn without replacement, compared ordinally since fitness can go negative. Two tournaments per crossover. Thesis section 4.4.',
    body: SelectionStep,
  },
  {
    key: 'crossover',
    name: 'Crossover',
    title: 'Crossover',
    subtitle: 'Two parents blend into two children by swapping roles',
    note: 'Fires with pc = 0.6054 (Optuna tuned), otherwise both parents pass through unchanged. Both children are repaired before joining the new population, later steps follow Child 1. Thesis section 4.5.',
    body: CrossoverStep,
  },
  {
    key: 'mutation',
    name: 'Mutation',
    title: 'Mutation',
    subtitle: 'Gaussian noise and an asset swap keep the search exploring',
    note: 'Independent Bernoulli draws, each with pm = 0.1370, σ = 0.1469, both Optuna tuned. Repair runs once after both operators. Thesis section 4.6.',
    body: MutationStep,
  },
  {
    key: 'repair',
    name: 'Repair',
    title: 'Repair',
    subtitle: 'Cardinality is enforced first, then projection onto the bounded simplex',
    note: 'Plain clip and renormalise can bounce weights back out of bounds, the projection lands feasible in a single pass. If it shifts cardinality it recurses once. Thesis section 4.7.',
    body: RepairStep,
  },
  {
    key: 'elitism',
    name: 'Elitism',
    title: 'Elitism',
    subtitle: 'The top 5 percent move on untouched',
    note: 'Real runs keep the top 5 of 100 with fitness carried over, no re-evaluation. Children fill the other 95 slots. Thesis section 4.8.',
    body: ElitismStep,
  },
  {
    key: 'refinement',
    name: 'Refinement',
    title: 'Local refinement',
    subtitle: 'The best elite gets a short greedy polish',
    note: 'Up to 5 random pair shifts of 0.01, improvements only, applied to the single best elite. Thesis section 4.8.',
    body: RefinementStep,
  },
  {
    key: 'new-generation',
    name: 'New generation',
    title: 'New generation',
    subtitle: '95 new children join the 5 elites, then the loop repeats',
    note: 'Up to 200 generations, early stop after 20 stagnant. The canonical run sits closest to the median in-sample fitness, no cherry picking. Thesis section 5.3.',
    body: NewGenerationStep,
  },
]

function stepFromHash() {
  if (typeof window === 'undefined') return 0
  const n = Number(window.location.hash.slice(1))
  return Number.isInteger(n) && n >= 1 && n <= STEPS.length ? n - 1 : 0
}

function useStageScale() {
  const [scale, setScale] = useState(1)
  useEffect(() => {
    const fit = () => setScale(Math.min(window.innerWidth / STAGE_WIDTH, window.innerHeight / STAGE_HEIGHT))
    fit()
    window.addEventListener('resize', fit)
    return () => window.removeEventListener('resize', fit)
  }, [])
  return scale
}

export default function App() {
  const [stepIndex, setStepIndex] = useState(stepFromHash)
  const scale = useStageScale()
  const step = STEPS[stepIndex]
  const Body = step.body

  useEffect(() => {
    window.history.replaceState(null, '', `#${stepIndex + 1}`)
  }, [stepIndex])

  useEffect(() => {
    const onKeyDown = event => {
      if (event.metaKey || event.ctrlKey || event.altKey) return
      const { key } = event
      // PageUp and PageDown are what presenter clickers send
      if (key === 'ArrowRight' || key === 'PageDown' || key === ' ') {
        event.preventDefault()
        setStepIndex(i => Math.min(STEPS.length - 1, i + 1))
      } else if (key === 'ArrowLeft' || key === 'PageUp' || key === 'Backspace') {
        event.preventDefault()
        setStepIndex(i => Math.max(0, i - 1))
      } else if (key === 'r' || key === 'R') {
        event.preventDefault()
        setStepIndex(0)
      } else if (/^[1-9]$/.test(key) && Number(key) <= STEPS.length) {
        event.preventDefault()
        setStepIndex(Number(key) - 1)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  return (
    <div className="stage" style={{ transform: `translate(-50%, -50%) scale(${scale})` }}>
      <div className="stage-circle" />
      <div className="stage-frame" />
      <div className="step-number">{String(stepIndex + 1).padStart(2, '0')}</div>

      <div className="stage-inner">
        <header className="stage-header">
          <div>
            <h1 className="step-title">{step.title}</h1>
            <p className="step-subtitle">{step.subtitle}</p>
          </div>
          <span className="step-count">{stepIndex + 1} / {STEPS.length}</span>
        </header>

        <main key={step.key} className="step-body">
          <Body />
        </main>

        <footer className="stage-footer">
          <p className="step-note">{step.note}</p>
          <div className="footer-row">
            <span className="key-hint">Keys: 1 to 8, arrows, space</span>
            <nav className="progress" aria-label="Walkthrough steps">
              {STEPS.map((s, i) => {
                let cls = 'progress-item'
                if (i === stepIndex) cls += ' progress-item-current'
                else if (i < stepIndex) cls += ' progress-item-done'
                return (
                  <button key={s.key} type="button" className={cls} onClick={() => setStepIndex(i)}>
                    {i + 1} {s.name}
                  </button>
                )
              })}
            </nav>
            <button type="button" className="reset-button" onClick={() => setStepIndex(0)}>
              Reset
            </button>
          </div>
        </footer>
      </div>
    </div>
  )
}
