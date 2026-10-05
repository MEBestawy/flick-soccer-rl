/**
 * TypeScript types for game state and API responses.
 */

export interface Vec2 {
  x: number
  y: number
}

export interface Player {
  id: string
  team: 'A' | 'B'
  position: Vec2
  velocity: Vec2
  radius: number
  mass: number
  is_sleeping: boolean
}

export interface Ball {
  position: Vec2
  velocity: Vec2
  radius: number
  mass: number
  is_sleeping: boolean
}

export interface GameState {
  phase: GamePhase
  current_team: 'A' | 'B'
  turn_number: number
  players: Player[]
  ball: Ball
  score_a: number
  score_b: number
  match_time: number
  turn_time: number
  simulation_time: number
  last_touch_team: 'A' | 'B' | null
  last_touch_player: string | null
}

export type GamePhase = 
  | 'KICKOFF'
  | 'AIMING'
  | 'SIMULATING'
  | 'TURN_END'
  | 'GOAL'
  | 'RESETTING'
  | 'GAME_OVER'

export interface Frame {
  time: number
  ball_position: Vec2
  ball_velocity: Vec2
  player_positions: Record<string, Vec2>
  player_velocities: Record<string, Vec2>
}

export interface GameEvent {
  type: string
  time: number
  data: Record<string, unknown>
}

export interface GameConfig {
  pitch_width: number
  pitch_height: number
  goal_width: number
  goal_height: number
  goal_y_min: number
  goal_y_max: number
  player_radius: number
  ball_radius: number
  team_a_color: [number, number, number]
  team_b_color: [number, number, number]
}

export interface ActionRequest {
  player_id: string
  direction_x: number
  direction_y: number
  power: number
}

export interface ActionResponse {
  success: boolean
  error?: string
  start_state?: GameState
  final_state?: GameState
  frames?: Frame[]
  events?: GameEvent[]
  simulation_duration: number
  simulated_time: number
}

export type AgentName = 'human' | 'jev' | 'random' | 'heuristic' | string

export interface CreateGameResponse {
  game_id: string
  state: GameState
  team_a_agent: AgentName
  team_b_agent: AgentName
}

export interface GetGameResponse {
  game_id: string
  state: GameState
  controllable_players: string[]
  team_a_agent: AgentName
  team_b_agent: AgentName
  current_team_is_ai: boolean
}

export interface AiActionResponse extends ActionResponse {
  agent?: string
  action?: {
    player_id: string
    direction: Vec2
    power: number
  }
  prompt_preview?: string
}
