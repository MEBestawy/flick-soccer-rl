/**
 * Scoreboard component showing game state.
 */

import type { GamePhase } from '../game/types'

interface ScoreboardProps {
  scoreA: number
  scoreB: number
  currentTeam: 'A' | 'B'
  turnNumber: number
  phase: GamePhase
  isAnimating: boolean
}

export function Scoreboard({
  scoreA,
  scoreB,
  currentTeam,
  turnNumber,
  phase,
  isAnimating,
}: ScoreboardProps) {
  const getPhaseText = () => {
    if (isAnimating) return '⚡ Simulating...'
    switch (phase) {
      case 'KICKOFF': return '🏁 Kick Off'
      case 'AIMING': return '🎯 Aim & Flick'
      case 'SIMULATING': return '⚡ Simulating...'
      case 'TURN_END': return '↔️ Turn Ending'
      case 'GOAL': return '⚽ GOAL!'
      case 'RESETTING': return '🔄 Resetting...'
      case 'GAME_OVER': return '🏆 Game Over!'
      default: return phase
    }
  }

  const isGameOver = phase === 'GAME_OVER'
  const winner = isGameOver 
    ? (scoreA > scoreB ? 'A' : scoreB > scoreA ? 'B' : null)
    : null

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'center',
      flexWrap: 'wrap',
      gap: '16px 28px',
      marginBottom: '6px',
      flexShrink: 0,
      width: '100%',
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '14px',
        padding: '6px 18px',
        background: 'linear-gradient(180deg, #2d2d44 0%, #1a1a2e 100%)',
        borderRadius: '12px',
        boxShadow: '0 2px 12px rgba(0, 0, 0, 0.25)',
      }}>
        <div style={{
          width: '22px',
          height: '22px',
          borderRadius: '50%',
          background: 'linear-gradient(135deg, #fff8dc, #ffd700)',
          border: currentTeam === 'A' ? '2px solid #00ffff' : '2px solid transparent',
          opacity: currentTeam === 'A' && !isGameOver ? 1 : 0.55,
        }} />
        <span style={{
          fontSize: '28px',
          fontWeight: 700,
          fontFamily: 'monospace',
          color: '#ffd700',
          lineHeight: 1,
        }}>
          {scoreA}
        </span>
        <span style={{ fontSize: '20px', color: '#666' }}>–</span>
        <span style={{
          fontSize: '28px',
          fontWeight: 700,
          fontFamily: 'monospace',
          color: '#ff7f7f',
          lineHeight: 1,
        }}>
          {scoreB}
        </span>
        <div style={{
          width: '22px',
          height: '22px',
          borderRadius: '50%',
          background: 'linear-gradient(135deg, #ffc0cb, #ff7f7f)',
          border: currentTeam === 'B' ? '2px solid #00ffff' : '2px solid transparent',
          opacity: currentTeam === 'B' && !isGameOver ? 1 : 0.55,
        }} />
      </div>

      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '14px',
        fontSize: '13px',
        color: '#aaa',
      }}>
        <span>Turn {turnNumber}</span>
        <span style={{
          padding: '3px 10px',
          background: isAnimating ? '#4a4a6a' : '#3d3d5c',
          borderRadius: '10px',
          fontWeight: 500,
        }}>
          {getPhaseText()}
        </span>
        {!isGameOver && (
          <span style={{ color: currentTeam === 'A' ? '#ffd700' : '#ff7f7f' }}>
            Team {currentTeam}'s turn
          </span>
        )}
        {winner && (
          <span style={{ color: '#00ff00', fontWeight: 700 }}>
            Team {winner} Wins!
          </span>
        )}
      </div>
    </div>
  )
}
