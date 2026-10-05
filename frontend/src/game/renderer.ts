/**
 * Canvas renderer for the soccer game.
 */

import type { GameState, GameConfig, Vec2 } from './types'

// Colors
const COLORS = {
  pitch: '#2d8a4e',
  pitchDark: '#247a42',
  lines: 'rgba(255, 255, 255, 0.7)',
  goalNet: 'rgba(255, 255, 255, 0.3)',
  goalPost: '#ffffff',
  ball: '#ffffff',
  ballOutline: '#333333',
  teamA: '#ffd700',  // Yellow/cream
  teamAOutline: '#b8860b',
  teamB: '#ff7f7f',  // Pink/coral
  teamBOutline: '#cd5c5c',
  selected: '#00ffff',
  aimLine: 'rgba(255, 255, 255, 0.8)',
  aimArrow: '#ffaa00',
  debug: 'rgba(255, 0, 0, 0.5)',
}

interface RenderOptions {
  selectedPlayer: string | null
  aimDirection: Vec2 | null
  aimPower: number
  showDebug: boolean
}

export class GameRenderer {
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private config: GameConfig
  private scale: number = 1
  private offsetX: number = 0
  private offsetY: number = 0
  private dpr: number = 1

  constructor(canvas: HTMLCanvasElement, config: GameConfig) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')!
    this.config = config
    this.dpr = window.devicePixelRatio || 1
  }

  resize(width: number, height: number): void {
    // Set canvas size with DPR
    this.canvas.width = width * this.dpr
    this.canvas.height = height * this.dpr
    this.canvas.style.width = `${width}px`
    this.canvas.style.height = `${height}px`
    
    // Fit pitch (+ goals) into canvas while preserving world proportions.
    // Tiny padding — the green field should dominate the canvas.
    const padding = 8
    const pitchWidth = this.config.pitch_width
    const pitchHeight = this.config.pitch_height
    const goalWidth = this.config.goal_width
    
    const totalWidth = pitchWidth + 2 * goalWidth + 2 * padding
    const totalHeight = pitchHeight + 2 * padding
    
    const scaleX = width / totalWidth
    const scaleY = height / totalHeight
    this.scale = Math.min(scaleX, scaleY)
    
    // Center the pitch
    this.offsetX = (width - pitchWidth * this.scale) / 2
    this.offsetY = (height - pitchHeight * this.scale) / 2
  }

  worldToScreen(pos: Vec2): Vec2 {
    return {
      x: this.offsetX + pos.x * this.scale,
      y: this.offsetY + (this.config.pitch_height - pos.y) * this.scale, // Flip Y
    }
  }

  screenToWorld(pos: Vec2): Vec2 {
    return {
      x: (pos.x - this.offsetX) / this.scale,
      y: this.config.pitch_height - (pos.y - this.offsetY) / this.scale,
    }
  }

  render(state: GameState, options: RenderOptions): void {
    const ctx = this.ctx
    const { pitch_width: _w, pitch_height: _h, goal_width: _gw, goal_height: _gh } = this.config
    void _w; void _h; void _gw; void _gh; // Used in sub-methods via this.config
    
    // Scale for DPR
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0)
    
    // Clear
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, this.canvas.width / this.dpr, this.canvas.height / this.dpr)
    
    // Draw pitch
    this.drawPitch()
    
    // Draw goals
    this.drawGoals()
    
    // Draw players
    for (const player of state.players) {
      const isSelected = player.id === options.selectedPlayer
      const isControllable = player.team === state.current_team && 
        (state.phase === 'KICKOFF' || state.phase === 'AIMING')
      this.drawPlayer(player, isSelected, isControllable)
    }
    
    // Draw ball
    this.drawBall(state.ball)
    
    // Draw aim indicator
    if (options.selectedPlayer && options.aimDirection && options.aimPower > 0) {
      const player = state.players.find(p => p.id === options.selectedPlayer)
      if (player) {
        this.drawAimIndicator(player.position, options.aimDirection, options.aimPower)
      }
    }
    
    // Debug overlay
    if (options.showDebug) {
      this.drawDebugOverlay(state)
    }
  }

  private drawPitch(): void {
    const ctx = this.ctx
    const { pitch_width: w, pitch_height: h } = this.config
    
    // Main pitch
    ctx.fillStyle = COLORS.pitch
    const topLeft = this.worldToScreen({ x: 0, y: h })
    ctx.fillRect(topLeft.x, topLeft.y, w * this.scale, h * this.scale)
    
    // Stripes
    ctx.fillStyle = COLORS.pitchDark
    const stripeWidth = w / 12
    for (let i = 0; i < 12; i += 2) {
      const x = this.worldToScreen({ x: i * stripeWidth, y: h })
      ctx.fillRect(x.x, x.y, stripeWidth * this.scale, h * this.scale)
    }
    
    // Lines
    ctx.strokeStyle = COLORS.lines
    ctx.lineWidth = 2
    
    // Outline
    const bl = this.worldToScreen({ x: 0, y: 0 })
    const tr = this.worldToScreen({ x: w, y: h })
    ctx.strokeRect(bl.x, tr.y, w * this.scale, h * this.scale)
    
    // Center line
    const cl1 = this.worldToScreen({ x: w / 2, y: 0 })
    const cl2 = this.worldToScreen({ x: w / 2, y: h })
    ctx.beginPath()
    ctx.moveTo(cl1.x, cl1.y)
    ctx.lineTo(cl2.x, cl2.y)
    ctx.stroke()
    
    // Center circle
    const center = this.worldToScreen({ x: w / 2, y: h / 2 })
    ctx.beginPath()
    ctx.arc(center.x, center.y, 9.15 * this.scale, 0, Math.PI * 2)
    ctx.stroke()
    
    // Center spot
    ctx.fillStyle = COLORS.lines
    ctx.beginPath()
    ctx.arc(center.x, center.y, 0.5 * this.scale, 0, Math.PI * 2)
    ctx.fill()
    
    // Penalty areas (simplified)
    const penaltyWidth = 16.5
    const penaltyHeight = 40.32
    const penaltyTop = (h - penaltyHeight) / 2
    
    // Left penalty area
    ctx.strokeRect(
      this.worldToScreen({ x: 0, y: penaltyTop + penaltyHeight }).x,
      this.worldToScreen({ x: 0, y: penaltyTop + penaltyHeight }).y,
      penaltyWidth * this.scale,
      penaltyHeight * this.scale
    )
    
    // Right penalty area
    ctx.strokeRect(
      this.worldToScreen({ x: w - penaltyWidth, y: penaltyTop + penaltyHeight }).x,
      this.worldToScreen({ x: w - penaltyWidth, y: penaltyTop + penaltyHeight }).y,
      penaltyWidth * this.scale,
      penaltyHeight * this.scale
    )
  }

  private drawGoals(): void {
    const ctx = this.ctx
    const { pitch_width: w, pitch_height: h, goal_width: gw, goal_height: gh } = this.config
    const goalY1 = (h - gh) / 2
    const goalY2 = (h + gh) / 2
    
    // Goal nets
    ctx.fillStyle = COLORS.goalNet
    
    // Left goal
    const lg = this.worldToScreen({ x: -gw, y: goalY2 })
    ctx.fillRect(lg.x, lg.y, gw * this.scale, gh * this.scale)
    
    // Right goal
    const rg = this.worldToScreen({ x: w, y: goalY2 })
    ctx.fillRect(rg.x, rg.y, gw * this.scale, gh * this.scale)
    
    // Goal posts
    ctx.fillStyle = COLORS.goalPost
    const postRadius = this.config.goal_width * 0.15 * this.scale
    
    // Left posts
    const lp1 = this.worldToScreen({ x: 0, y: goalY1 })
    const lp2 = this.worldToScreen({ x: 0, y: goalY2 })
    ctx.beginPath()
    ctx.arc(lp1.x, lp1.y, postRadius, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.arc(lp2.x, lp2.y, postRadius, 0, Math.PI * 2)
    ctx.fill()
    
    // Right posts
    const rp1 = this.worldToScreen({ x: w, y: goalY1 })
    const rp2 = this.worldToScreen({ x: w, y: goalY2 })
    ctx.beginPath()
    ctx.arc(rp1.x, rp1.y, postRadius, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.arc(rp2.x, rp2.y, postRadius, 0, Math.PI * 2)
    ctx.fill()
  }

  private drawPlayer(
    player: { id: string; team: 'A' | 'B'; position: Vec2; radius: number },
    isSelected: boolean,
    isControllable: boolean
  ): void {
    const ctx = this.ctx
    const pos = this.worldToScreen(player.position)
    const radius = player.radius * this.scale
    
    // Glow for controllable
    if (isControllable) {
      ctx.save()
      ctx.globalAlpha = 0.3
      ctx.fillStyle = player.team === 'A' ? COLORS.teamA : COLORS.teamB
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, radius * 1.5, 0, Math.PI * 2)
      ctx.fill()
      ctx.restore()
    }
    
    // Selection ring
    if (isSelected) {
      ctx.strokeStyle = COLORS.selected
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.arc(pos.x, pos.y, radius + 4, 0, Math.PI * 2)
      ctx.stroke()
    }
    
    // Player disc
    const gradient = ctx.createRadialGradient(
      pos.x - radius * 0.3, pos.y - radius * 0.3, 0,
      pos.x, pos.y, radius
    )
    
    if (player.team === 'A') {
      gradient.addColorStop(0, '#fff8dc')  // Light
      gradient.addColorStop(1, COLORS.teamA)
    } else {
      gradient.addColorStop(0, '#ffc0cb')  // Light pink
      gradient.addColorStop(1, COLORS.teamB)
    }
    
    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2)
    ctx.fill()
    
    // Outline
    ctx.strokeStyle = player.team === 'A' ? COLORS.teamAOutline : COLORS.teamBOutline
    ctx.lineWidth = 2
    ctx.stroke()
    
    // Player ID
    ctx.fillStyle = '#333'
    ctx.font = `bold ${radius * 0.8}px sans-serif`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(player.id.slice(-1), pos.x, pos.y)
  }

  private drawBall(ball: { position: Vec2; radius: number }): void {
    const ctx = this.ctx
    const pos = this.worldToScreen(ball.position)
    const radius = ball.radius * this.scale
    
    // Shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.2)'
    ctx.beginPath()
    ctx.ellipse(pos.x + 2, pos.y + 2, radius, radius * 0.6, 0, 0, Math.PI * 2)
    ctx.fill()
    
    // Ball
    const gradient = ctx.createRadialGradient(
      pos.x - radius * 0.3, pos.y - radius * 0.3, 0,
      pos.x, pos.y, radius
    )
    gradient.addColorStop(0, '#ffffff')
    gradient.addColorStop(1, '#dddddd')
    
    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2)
    ctx.fill()
    
    // Outline
    ctx.strokeStyle = COLORS.ballOutline
    ctx.lineWidth = 1
    ctx.stroke()
    
    // Pentagon pattern (simplified)
    ctx.fillStyle = '#333'
    ctx.beginPath()
    ctx.arc(pos.x, pos.y, radius * 0.4, 0, Math.PI * 2)
    ctx.fill()
  }

  private drawAimIndicator(playerPos: Vec2, direction: Vec2, power: number): void {
    const ctx = this.ctx
    const start = this.worldToScreen(playerPos)
    
    // Normalize direction
    const len = Math.sqrt(direction.x ** 2 + direction.y ** 2)
    if (len < 0.01) return
    
    const dx = direction.x / len
    const dy = direction.y / len
    
    // Arrow length based on power
    const maxLength = 15 * this.scale
    const arrowLength = maxLength * power
    
    const end = {
      x: start.x + dx * arrowLength,
      y: start.y - dy * arrowLength,  // Flip Y
    }
    
    // Draw arrow line
    ctx.strokeStyle = COLORS.aimLine
    ctx.lineWidth = 3
    ctx.lineCap = 'round'
    ctx.setLineDash([5, 5])
    ctx.beginPath()
    ctx.moveTo(start.x, start.y)
    ctx.lineTo(end.x, end.y)
    ctx.stroke()
    ctx.setLineDash([])
    
    // Draw arrowhead
    const headLength = 10
    const angle = Math.atan2(-dy, dx)  // Flip Y
    
    ctx.fillStyle = COLORS.aimArrow
    ctx.beginPath()
    ctx.moveTo(end.x, end.y)
    ctx.lineTo(
      end.x - headLength * Math.cos(angle - Math.PI / 6),
      end.y - headLength * Math.sin(angle - Math.PI / 6)
    )
    ctx.lineTo(
      end.x - headLength * Math.cos(angle + Math.PI / 6),
      end.y - headLength * Math.sin(angle + Math.PI / 6)
    )
    ctx.closePath()
    ctx.fill()
    
    // Power meter
    const meterWidth = 60
    const meterHeight = 8
    const meterX = start.x - meterWidth / 2
    const meterY = start.y + 20
    
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)'
    ctx.fillRect(meterX, meterY, meterWidth, meterHeight)
    
    const powerColor = power < 0.5 ? '#4CAF50' : power < 0.8 ? '#FFC107' : '#f44336'
    ctx.fillStyle = powerColor
    ctx.fillRect(meterX, meterY, meterWidth * power, meterHeight)
    
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 1
    ctx.strokeRect(meterX, meterY, meterWidth, meterHeight)
  }

  private drawDebugOverlay(state: GameState): void {
    const ctx = this.ctx
    
    // Draw velocity vectors
    ctx.strokeStyle = COLORS.debug
    ctx.lineWidth = 2
    
    for (const player of state.players) {
      const pos = this.worldToScreen(player.position)
      const vel = player.velocity
      if (vel.x !== 0 || vel.y !== 0) {
        const velScale = 0.5
        ctx.beginPath()
        ctx.moveTo(pos.x, pos.y)
        ctx.lineTo(
          pos.x + vel.x * velScale,
          pos.y - vel.y * velScale  // Flip Y
        )
        ctx.stroke()
      }
    }
    
    // Ball velocity
    const ballPos = this.worldToScreen(state.ball.position)
    const ballVel = state.ball.velocity
    if (ballVel.x !== 0 || ballVel.y !== 0) {
      ctx.strokeStyle = '#00ff00'
      ctx.beginPath()
      ctx.moveTo(ballPos.x, ballPos.y)
      ctx.lineTo(
        ballPos.x + ballVel.x * 0.5,
        ballPos.y - ballVel.y * 0.5
      )
      ctx.stroke()
    }
    
    // Debug text
    ctx.fillStyle = '#fff'
    ctx.font = '12px monospace'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'top'
    
    const info = [
      `Phase: ${state.phase}`,
      `Turn: ${state.turn_number}`,
      `Ball: (${state.ball.position.x.toFixed(1)}, ${state.ball.position.y.toFixed(1)})`,
      `Ball vel: (${state.ball.velocity.x.toFixed(1)}, ${state.ball.velocity.y.toFixed(1)})`,
    ]
    
    info.forEach((line, i) => {
      ctx.fillText(line, 10, 10 + i * 16)
    })
  }

  getPlayerAtPosition(state: GameState, screenPos: Vec2): string | null {
    const worldPos = this.screenToWorld(screenPos)
    
    for (const player of state.players) {
      const dx = worldPos.x - player.position.x
      const dy = worldPos.y - player.position.y
      const dist = Math.sqrt(dx * dx + dy * dy)
      
      if (dist <= player.radius * 1.5) {  // Slightly larger hit area
        return player.id
      }
    }
    
    return null
  }
}
