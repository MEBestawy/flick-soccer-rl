/**
 * API client for game backend.
 */

import type {
  GameConfig,
  ActionRequest,
  ActionResponse,
  AiActionResponse,
  AgentName,
  CreateGameResponse,
  GetGameResponse,
} from './types'

const API_BASE = '/api'

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })
  
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`API error ${response.status}: ${text}`)
  }
  
  return response.json()
}

export async function getConfig(): Promise<GameConfig> {
  return fetchJson<GameConfig>(`${API_BASE}/config`)
}

export async function createGame(
  startingTeam: 'A' | 'B' = 'A',
  teamAAgent: AgentName = 'human',
  teamBAgent: AgentName = 'human',
): Promise<CreateGameResponse> {
  return fetchJson<CreateGameResponse>(`${API_BASE}/games`, {
    method: 'POST',
    body: JSON.stringify({
      starting_team: startingTeam,
      team_a_agent: teamAAgent,
      team_b_agent: teamBAgent,
    }),
  })
}

export async function getGame(gameId: string): Promise<GetGameResponse> {
  return fetchJson<GetGameResponse>(`${API_BASE}/games/${gameId}`)
}

export async function executeAction(
  gameId: string,
  action: ActionRequest
): Promise<ActionResponse> {
  return fetchJson<ActionResponse>(`${API_BASE}/games/${gameId}/actions`, {
    method: 'POST',
    body: JSON.stringify(action),
  })
}

export async function resetGame(
  gameId: string,
  startingTeam: 'A' | 'B' = 'A',
  teamAAgent: AgentName = 'human',
  teamBAgent: AgentName = 'human',
): Promise<CreateGameResponse> {
  return fetchJson<CreateGameResponse>(`${API_BASE}/games/${gameId}/reset`, {
    method: 'POST',
    body: JSON.stringify({
      starting_team: startingTeam,
      team_a_agent: teamAAgent,
      team_b_agent: teamBAgent,
    }),
  })
}

export async function executeAiAction(
  gameId: string,
  agent?: AgentName,
): Promise<AiActionResponse> {
  return fetchJson<AiActionResponse>(`${API_BASE}/games/${gameId}/ai-action`, {
    method: 'POST',
    body: JSON.stringify(agent ? { agent } : {}),
  })
}
