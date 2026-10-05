/**
 * React hook for game state management.
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import type { GameState, GameConfig, Vec2, Frame, AgentName } from './types'
import * as api from './api'

interface UseGameReturn {
  gameState: GameState | null
  config: GameConfig | null
  gameId: string | null
  isAnimating: boolean
  isAiThinking: boolean
  selectedPlayer: string | null
  aimDirection: Vec2 | null
  aimPower: number
  error: string | null
  teamAAgent: AgentName
  teamBAgent: AgentName
  setTeamAAgent: (agent: AgentName) => void
  setTeamBAgent: (agent: AgentName) => void
  createGame: () => Promise<void>
  resetGame: () => Promise<void>
  selectPlayer: (playerId: string | null) => void
  setAim: (direction: Vec2 | null, power: number) => void
  executeFlick: () => Promise<void>
}

export function useGame(): UseGameReturn {
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [config, setConfig] = useState<GameConfig | null>(null)
  const [gameId, setGameId] = useState<string | null>(null)
  const [isAnimating, setIsAnimating] = useState(false)
  const [isAiThinking, setIsAiThinking] = useState(false)
  const [selectedPlayer, setSelectedPlayer] = useState<string | null>(null)
  const [aimDirection, setAimDirection] = useState<Vec2 | null>(null)
  const [aimPower, setAimPower] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [teamAAgent, setTeamAAgent] = useState<AgentName>('human')
  const [teamBAgent, setTeamBAgent] = useState<AgentName>('jev')

  const animationFramesRef = useRef<Frame[]>([])
  const animationFrameIdRef = useRef<number>(0)
  const aiRequestIdRef = useRef(0)

  useEffect(() => {
    api.getConfig().then(setConfig).catch(console.error)
  }, [])

  const animateFrames = useCallback((
    frames: Frame[],
    finalState: GameState,
    onComplete: () => void
  ) => {
    if (frames.length === 0) {
      setGameState(finalState)
      onComplete()
      return
    }

    animationFramesRef.current = frames
    setIsAnimating(true)

    const duration = frames[frames.length - 1].time * 1000
    const startTime = performance.now()

    const animate = () => {
      const elapsed = performance.now() - startTime
      const progress = Math.min(elapsed / duration, 1)

      const currentTime = progress * frames[frames.length - 1].time
      let frameIndex = 0
      for (let i = 0; i < frames.length - 1; i++) {
        if (frames[i + 1].time > currentTime) break
        frameIndex = i
      }

      const frame = frames[frameIndex]
      const nextFrame = frames[Math.min(frameIndex + 1, frames.length - 1)]
      const frameProgress = frameIndex < frames.length - 1
        ? (currentTime - frame.time) / Math.max(nextFrame.time - frame.time, 1e-6)
        : 0

      setGameState(prev => {
        if (!prev) return prev

        const lerp = (a: number, b: number, t: number) => a + (b - a) * t

        return {
          ...prev,
          ball: {
            ...prev.ball,
            position: {
              x: lerp(frame.ball_position.x, nextFrame.ball_position.x, frameProgress),
              y: lerp(frame.ball_position.y, nextFrame.ball_position.y, frameProgress),
            },
            velocity: frame.ball_velocity,
          },
          players: prev.players.map(p => {
            const pos = frame.player_positions[p.id]
            const nextPos = nextFrame.player_positions[p.id]
            const vel = frame.player_velocities[p.id]
            if (!pos || !nextPos) return p
            return {
              ...p,
              position: {
                x: lerp(pos.x, nextPos.x, frameProgress),
                y: lerp(pos.y, nextPos.y, frameProgress),
              },
              velocity: vel || p.velocity,
            }
          }),
        }
      })

      if (progress < 1) {
        animationFrameIdRef.current = requestAnimationFrame(animate)
      } else {
        setGameState(finalState)
        setIsAnimating(false)
        onComplete()
      }
    }

    animationFrameIdRef.current = requestAnimationFrame(animate)
  }, [])

  const isAiTeam = useCallback((team: 'A' | 'B', aAgent: AgentName, bAgent: AgentName) => {
    const name = team === 'A' ? aAgent : bAgent
    return name !== 'human'
  }, [])

  const requestAiTurn = useCallback(async (
    id: string,
    finalState: GameState,
    aAgent: AgentName,
    bAgent: AgentName,
  ) => {
    if (finalState.phase === 'GAME_OVER') return
    if (!isAiTeam(finalState.current_team, aAgent, bAgent)) return

    const requestId = ++aiRequestIdRef.current
    setIsAiThinking(true)
    setError(null)
    try {
      const response = await api.executeAiAction(id)
      if (requestId !== aiRequestIdRef.current) return

      if (!response.success || !response.final_state) {
        setError(response.error || 'AI action failed')
        setIsAiThinking(false)
        return
      }

      setIsAiThinking(false)
      if (response.frames && response.frames.length > 0) {
        animateFrames(response.frames, response.final_state, () => {
          void requestAiTurn(id, response.final_state!, aAgent, bAgent)
        })
      } else {
        setGameState(response.final_state)
        void requestAiTurn(id, response.final_state, aAgent, bAgent)
      }
    } catch (e) {
      if (requestId !== aiRequestIdRef.current) return
      setIsAiThinking(false)
      setError(e instanceof Error ? e.message : 'AI action failed')
    }
  }, [animateFrames, isAiTeam])

  const createGame = useCallback(async () => {
    try {
      setError(null)
      cancelAnimationFrame(animationFrameIdRef.current)
      setIsAnimating(false)
      setIsAiThinking(false)
      const response = await api.createGame('A', teamAAgent, teamBAgent)
      setGameId(response.game_id)
      setGameState(response.state)
      setSelectedPlayer(null)
      setAimDirection(null)
      setAimPower(0)
      void requestAiTurn(response.game_id, response.state, teamAAgent, teamBAgent)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create game')
    }
  }, [teamAAgent, teamBAgent, requestAiTurn])

  const resetGame = useCallback(async () => {
    if (!gameId) return
    try {
      setError(null)
      setIsAnimating(false)
      setIsAiThinking(false)
      cancelAnimationFrame(animationFrameIdRef.current)
      aiRequestIdRef.current += 1
      const response = await api.resetGame(gameId, 'A', teamAAgent, teamBAgent)
      setGameState(response.state)
      setSelectedPlayer(null)
      setAimDirection(null)
      setAimPower(0)
      void requestAiTurn(gameId, response.state, teamAAgent, teamBAgent)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to reset game')
    }
  }, [gameId, teamAAgent, teamBAgent, requestAiTurn])

  const selectPlayer = useCallback((playerId: string | null) => {
    if (isAnimating || isAiThinking) return
    setSelectedPlayer(playerId)
    if (!playerId) {
      setAimDirection(null)
      setAimPower(0)
    }
  }, [isAnimating, isAiThinking])

  const setAim = useCallback((direction: Vec2 | null, power: number) => {
    if (isAnimating || isAiThinking) return
    setAimDirection(direction)
    setAimPower(Math.max(0, Math.min(1, power)))
  }, [isAnimating, isAiThinking])

  const executeFlick = useCallback(async () => {
    if (!gameId || !selectedPlayer || !aimDirection || isAnimating || isAiThinking) return
    if (aimPower < 0.05) return
    if (gameState && isAiTeam(gameState.current_team, teamAAgent, teamBAgent)) return

    try {
      setError(null)

      const response = await api.executeAction(gameId, {
        player_id: selectedPlayer,
        direction_x: aimDirection.x,
        direction_y: aimDirection.y,
        power: aimPower,
      })

      if (!response.success || !response.final_state) {
        setError(response.error || 'Action failed')
        return
      }

      setSelectedPlayer(null)
      setAimDirection(null)
      setAimPower(0)

      if (response.frames && response.frames.length > 0) {
        animateFrames(response.frames, response.final_state, () => {
          void requestAiTurn(gameId, response.final_state!, teamAAgent, teamBAgent)
        })
      } else {
        setGameState(response.final_state)
        void requestAiTurn(gameId, response.final_state, teamAAgent, teamBAgent)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to execute action')
    }
  }, [
    gameId,
    selectedPlayer,
    aimDirection,
    aimPower,
    isAnimating,
    isAiThinking,
    gameState,
    teamAAgent,
    teamBAgent,
    animateFrames,
    requestAiTurn,
    isAiTeam,
  ])

  return {
    gameState,
    config,
    gameId,
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
  }
}
