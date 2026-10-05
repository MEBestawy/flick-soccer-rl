/**
 * Main game canvas component with input handling.
 *
 * Drag/aim uses pointer capture so pulling off the canvas edge does not
 * drop the shot. Space is the only cancel (lose grip without firing).
 * Mouse/touch release still fires the flick.
 */

import { useRef, useEffect, useCallback, useState } from 'react'
import type { GameState, GameConfig, Vec2 } from '../game/types'
import { GameRenderer } from '../game/renderer'

/** World pitch aspect (width / height). Keep rendering proportional. */
const PITCH_ASPECT = 120 / 72

interface GameCanvasProps {
  gameState: GameState
  config: GameConfig | null
  selectedPlayer: string | null
  aimDirection: Vec2 | null
  aimPower: number
  isAnimating: boolean
  showDebug: boolean
  onSelectPlayer: (playerId: string | null) => void
  onAim: (direction: Vec2 | null, power: number) => void
  onFlick: () => void
}

export function GameCanvas({
  gameState,
  config,
  selectedPlayer,
  aimDirection,
  aimPower,
  isAnimating,
  showDebug,
  onSelectPlayer,
  onAim,
  onFlick,
}: GameCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rendererRef = useRef<GameRenderer | null>(null)
  const shellRef = useRef<HTMLDivElement>(null)
  const frameRef = useRef<HTMLDivElement>(null)

  const [isDragging, setIsDragging] = useState(false)
  const [frameSize, setFrameSize] = useState({ width: 800, height: 480 })

  // Keep latest values in refs for document-level listeners
  const dragRef = useRef({
    isDragging: false,
    dragStart: null as Vec2 | null,
    selectedPlayer: null as string | null,
    aimDirection: null as Vec2 | null,
    aimPower: 0,
  })
  dragRef.current.selectedPlayer = selectedPlayer
  dragRef.current.aimDirection = aimDirection
  dragRef.current.aimPower = aimPower

  const onAimRef = useRef(onAim)
  const onFlickRef = useRef(onFlick)
  const onSelectPlayerRef = useRef(onSelectPlayer)
  onAimRef.current = onAim
  onFlickRef.current = onFlick
  onSelectPlayerRef.current = onSelectPlayer

  // Initialize renderer
  useEffect(() => {
    if (!canvasRef.current || !config) return
    rendererRef.current = new GameRenderer(canvasRef.current, config)
    rendererRef.current.resize(frameSize.width, frameSize.height)
  }, [config])

  // Fit largest 120:72 rectangle inside available shell space
  useEffect(() => {
    const shell = shellRef.current
    if (!shell) return

    const fit = () => {
      const availW = shell.clientWidth
      const availH = shell.clientHeight
      if (availW <= 0 || availH <= 0) return

      let width = availW
      let height = width / PITCH_ASPECT
      if (height > availH) {
        height = availH
        width = height * PITCH_ASPECT
      }

      width = Math.floor(width)
      height = Math.floor(height)
      setFrameSize({ width, height })
      rendererRef.current?.resize(width, height)
    }

    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(shell)
    window.addEventListener('resize', fit)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', fit)
    }
  }, [config])

  // Render
  useEffect(() => {
    rendererRef.current?.render(gameState, {
      selectedPlayer,
      aimDirection,
      aimPower,
      showDebug,
    })
  }, [gameState, selectedPlayer, aimDirection, aimPower, showDebug, frameSize])

  const clientToCanvas = useCallback((clientX: number, clientY: number): Vec2 => {
    if (!canvasRef.current) return { x: 0, y: 0 }
    const rect = canvasRef.current.getBoundingClientRect()
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    }
  }, [])

  const updateAimFromClient = useCallback((clientX: number, clientY: number) => {
    const { isDragging: dragging, dragStart, selectedPlayer: sel } = dragRef.current
    if (!dragging || !dragStart || !sel || !rendererRef.current) return

    const pos = clientToCanvas(clientX, clientY)
    const dx = dragStart.x - pos.x
    const dy = pos.y - dragStart.y // screen Y down → world Y up for aim viz

    const distance = Math.hypot(dx, dy)
    if (distance > 5) {
      onAimRef.current(
        { x: dx / distance, y: dy / distance },
        Math.min(distance / 180, 1),
      )
    } else {
      onAimRef.current(null, 0)
    }
  }, [clientToCanvas])

  const cancelGrip = useCallback(() => {
    dragRef.current.isDragging = false
    dragRef.current.dragStart = null
    setIsDragging(false)
    onAimRef.current(null, 0)
    onSelectPlayerRef.current(null)
  }, [])

  const releaseFlick = useCallback(() => {
    const { isDragging: dragging, selectedPlayer: sel, aimDirection: dir, aimPower: pow } =
      dragRef.current
    if (dragging && sel && dir && pow > 0.05) {
      onFlickRef.current()
    }
    dragRef.current.isDragging = false
    dragRef.current.dragStart = null
    setIsDragging(false)
  }, [])

  // Document-level move/up while dragging — survives leaving the canvas
  useEffect(() => {
    if (!isDragging) return

    const onMove = (e: PointerEvent) => {
      updateAimFromClient(e.clientX, e.clientY)
    }
    const onUp = () => {
      releaseFlick()
    }

    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    window.addEventListener('pointercancel', onUp)
    return () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      window.removeEventListener('pointercancel', onUp)
    }
  }, [isDragging, updateAimFromClient, releaseFlick])

  // Space cancels grip (only way to abandon a held player without firing)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space' && e.key !== ' ') return
      if (!dragRef.current.isDragging && !dragRef.current.selectedPlayer) return
      e.preventDefault()
      cancelGrip()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [cancelGrip])

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    if (isAnimating || !rendererRef.current) return
    if (e.button !== 0) return

    const pos = clientToCanvas(e.clientX, e.clientY)
    const playerId = rendererRef.current.getPlayerAtPosition(gameState, pos)

    if (playerId) {
      const player = gameState.players.find(p => p.id === playerId)
      const isControllable =
        player?.team === gameState.current_team &&
        (gameState.phase === 'KICKOFF' || gameState.phase === 'AIMING')

      if (isControllable) {
        e.currentTarget.setPointerCapture?.(e.pointerId)
        onSelectPlayer(playerId)
        dragRef.current.isDragging = true
        dragRef.current.dragStart = pos
        dragRef.current.selectedPlayer = playerId
        setIsDragging(true)
      }
    }
    // Do NOT deselect / lose grip by clicking empty pitch — Space only.
  }, [gameState, isAnimating, onSelectPlayer, clientToCanvas])

  return (
    <div
      ref={shellRef}
      style={{
        flex: 1,
        width: '100%',
        minHeight: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <div
        ref={frameRef}
        style={{
          width: frameSize.width,
          height: frameSize.height,
          borderRadius: '10px',
          overflow: 'hidden',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.35)',
          cursor: isAnimating ? 'wait' : isDragging ? 'grabbing' : 'pointer',
          flexShrink: 0,
        }}
      >
        <canvas
          ref={canvasRef}
          onPointerDown={handlePointerDown}
          style={{
            width: '100%',
            height: '100%',
            display: 'block',
            touchAction: 'none',
          }}
        />
      </div>
    </div>
  )
}
