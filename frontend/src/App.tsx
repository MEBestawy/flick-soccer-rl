import { useState, useCallback } from 'react'
import { GameCanvas } from './components/GameCanvas'
import { Scoreboard } from './components/Scoreboard'
import { TuningPanel } from './components/TuningPanel'
import { useGame } from './game/useGame'
import type { AgentName } from './game/types'

const AGENT_OPTIONS: { value: AgentName; label: string }[] = [
  { value: 'human', label: 'Human' },
  { value: 'jev', label: 'Jev' },
  { value: 'heuristic', label: 'Heuristic' },
  { value: 'random', label: 'Random' },
]

export default function App() {
  const [showTuning, setShowTuning] = useState(false)
  const [showDebug, setShowDebug] = useState(false)

  const {
    gameState,
    config,
    isAnimating,
    isAiThinking,
    selectedPlayer,
    aimDirection,
    aimPower,
    error,
    teamAAgent,
    teamBAgent,
    setTeamAAgent,
    setTeamBAgent,
    createGame,
    resetGame,
    selectPlayer,
    setAim,
    executeFlick,
  } = useGame()

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'd') setShowDebug(d => !d)
    if (e.key === 't') setShowTuning(t => !t)
  }, [])

  const inputLocked = isAnimating || isAiThinking
  const humanCanAct =
    !!gameState &&
    !inputLocked &&
    ((gameState.current_team === 'A' && teamAAgent === 'human') ||
      (gameState.current_team === 'B' && teamBAgent === 'human'))

  return (
    <div
      style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}
      onKeyDown={handleKeyDown}
      tabIndex={0}
    >
      <header style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '6px 16px',
        background: 'linear-gradient(180deg, #2d2d44 0%, #1a1a2e 100%)',
        borderBottom: '1px solid #3d3d5c',
        flexShrink: 0,
      }}>
        <h1 style={{
          fontSize: '18px',
          fontWeight: 600,
          margin: 0,
          background: 'linear-gradient(90deg, #ffd700, #ffaa00)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}>
          Flick Football
        </h1>

        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          {isAiThinking && (
            <span style={{ color: '#ffd700', fontSize: '14px' }}>
              Jev is thinking…
            </span>
          )}
          <button
            onClick={() => setShowDebug(d => !d)}
            style={headerButtonStyle(showDebug)}
          >
            Debug
          </button>
          <button
            onClick={() => setShowTuning(t => !t)}
            style={headerButtonStyle(showTuning)}
          >
            Tuning
          </button>
        </div>
      </header>

      <main style={{
        flex: 1,
        display: 'flex',
        position: 'relative',
        overflow: 'hidden',
      }}>
        <div style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'stretch',
          justifyContent: 'stretch',
          padding: '6px 10px 4px',
          minHeight: 0,
          width: '100%',
          height: '100%',
        }}>
          {gameState ? (
            <>
              <Scoreboard
                scoreA={gameState.score_a}
                scoreB={gameState.score_b}
                currentTeam={gameState.current_team}
                turnNumber={gameState.turn_number}
                phase={gameState.phase}
                isAnimating={isAnimating || isAiThinking}
              />

              <div style={{
                marginBottom: '4px',
                fontSize: '12px',
                color: '#aaa',
                flexShrink: 0,
                textAlign: 'center',
              }}>
                Team A ({teamAAgent}) vs Team B ({teamBAgent})
                {gameState.current_team === 'B' && teamBAgent === 'jev' && ' · Jev to move'}
                {gameState.current_team === 'A' && teamAAgent === 'jev' && ' · Jev to move'}
                {' · '}Space cancels aim
              </div>

              <GameCanvas
                gameState={gameState}
                config={config}
                selectedPlayer={humanCanAct ? selectedPlayer : null}
                aimDirection={humanCanAct ? aimDirection : null}
                aimPower={humanCanAct ? aimPower : 0}
                isAnimating={inputLocked}
                showDebug={showDebug}
                onSelectPlayer={humanCanAct ? selectPlayer : () => undefined}
                onAim={humanCanAct ? setAim : () => undefined}
                onFlick={humanCanAct ? executeFlick : () => undefined}
              />
              
              {error && (
                <div style={{
                  marginTop: '4px',
                  padding: '8px 16px',
                  background: '#ff4444',
                  borderRadius: '8px',
                  color: '#fff',
                  flexShrink: 0,
                  textAlign: 'center',
                }}>
                  {error}
                </div>
              )}
              
              <div style={{ marginTop: '4px', display: 'flex', gap: '12px', flexShrink: 0, justifyContent: 'center' }}>
                <button
                  onClick={resetGame}
                  disabled={inputLocked}
                  style={{
                    padding: '8px 16px',
                    background: '#4a4a6a',
                    border: 'none',
                    borderRadius: '8px',
                    color: '#fff',
                    cursor: inputLocked ? 'not-allowed' : 'pointer',
                    fontSize: '13px',
                    opacity: inputLocked ? 0.5 : 1,
                  }}
                >
                  Reset Game
                </button>
              </div>
            </>
          ) : (
            <div style={{ textAlign: 'center' }}>
              <h2 style={{ fontSize: '32px', marginBottom: '16px' }}>
                Flick Football
              </h2>
              <p style={{ marginBottom: '28px', color: '#aaa', maxWidth: '420px' }}>
                Turn-based 5v5 physics soccer. Play yourself or let Jev take a side —
                agents are swappable strategies behind the same API.
              </p>

              <div style={{
                display: 'flex',
                gap: '24px',
                justifyContent: 'center',
                marginBottom: '28px',
                flexWrap: 'wrap',
              }}>
                <AgentPicker
                  label="Team A (Yellow)"
                  value={teamAAgent}
                  onChange={setTeamAAgent}
                />
                <AgentPicker
                  label="Team B (Pink) — Jev default"
                  value={teamBAgent}
                  onChange={setTeamBAgent}
                />
              </div>

              <button
                onClick={createGame}
                style={{
                  padding: '16px 48px',
                  background: 'linear-gradient(135deg, #ffd700, #ffaa00)',
                  border: 'none',
                  borderRadius: '12px',
                  color: '#1a1a2e',
                  cursor: 'pointer',
                  fontSize: '20px',
                  fontWeight: 600,
                  boxShadow: '0 4px 20px rgba(255, 215, 0, 0.3)',
                }}
              >
                Start Game
              </button>
            </div>
          )}
        </div>

        {showTuning && config && (
          <TuningPanel config={config} onClose={() => setShowTuning(false)} />
        )}
      </main>

      <footer style={{
        padding: '4px 16px',
        background: '#1a1a2e',
        borderTop: '1px solid #3d3d5c',
        fontSize: '11px',
        color: '#666',
        textAlign: 'center',
        flexShrink: 0,
      }}>
        Drag to aim · release to shoot · Space cancels grip · D debug · T tuning · First to 3
      </footer>
    </div>
  )
}

function AgentPicker({
  label,
  value,
  onChange,
}: {
  label: string
  value: AgentName
  onChange: (v: AgentName) => void
}) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: '8px', color: '#ddd' }}>
      <span style={{ fontSize: '13px' }}>{label}</span>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        style={{
          padding: '10px 14px',
          borderRadius: '8px',
          border: '1px solid #3d3d5c',
          background: '#2d2d44',
          color: '#fff',
          fontSize: '15px',
        }}
      >
        {AGENT_OPTIONS.map(opt => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
    </label>
  )
}

function headerButtonStyle(active: boolean): React.CSSProperties {
  return {
    padding: '8px 16px',
    background: active ? '#4a4a6a' : '#3d3d5c',
    border: 'none',
    borderRadius: '6px',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '14px',
  }
}
