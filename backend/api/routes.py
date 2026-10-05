"""
API routes for game management.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import uuid

from sim import (
    GameSimulator, GameState, FlickAction, SimConfig,
    Team, GamePhase, Vec2
)
from sim.agents import Agent, available_agents, create_agent
from sim.serialization import (
    serialize_state, serialize_frame, serialize_event, serialize_action
)

router = APIRouter()

# In-memory game storage (use a database in production)
games: Dict[str, GameSimulator] = {}


@dataclass
class GameAgentBinding:
    """Per-game strategy bindings (None = human-controlled)."""

    team_a: Optional[Agent]
    team_b: Optional[Agent]
    team_a_name: str
    team_b_name: str

    def agent_for(self, team: Team) -> Optional[Agent]:
        return self.team_a if team == Team.A else self.team_b

    def name_for(self, team: Team) -> str:
        return self.team_a_name if team == Team.A else self.team_b_name


game_agents: Dict[str, GameAgentBinding] = {}


class CreateGameRequest(BaseModel):
    """Request to create a new game."""
    starting_team: str = Field(default="A", pattern="^[AB]$")
    # Strategy names: "human", "jev", "random", "heuristic"
    team_a_agent: str = Field(default="human")
    team_b_agent: str = Field(default="human")


class CreateGameResponse(BaseModel):
    """Response after creating a game."""
    game_id: str
    state: Dict[str, Any]
    team_a_agent: str = "human"
    team_b_agent: str = "human"


class ActionRequest(BaseModel):
    """Request to execute a flick action."""
    player_id: str
    direction_x: float
    direction_y: float
    power: float = Field(ge=0.0, le=1.0)


class ActionResponse(BaseModel):
    """Response after executing an action."""
    success: bool
    error: Optional[str] = None
    start_state: Optional[Dict[str, Any]] = None
    final_state: Optional[Dict[str, Any]] = None
    frames: Optional[List[Dict[str, Any]]] = None
    events: Optional[List[Dict[str, Any]]] = None
    simulation_duration: float = 0.0
    simulated_time: float = 0.0


class GameStateResponse(BaseModel):
    """Response with current game state."""
    game_id: str
    state: Dict[str, Any]
    controllable_players: List[str]
    team_a_agent: str = "human"
    team_b_agent: str = "human"
    current_team_is_ai: bool = False


class ConfigResponse(BaseModel):
    """Response with game configuration."""
    pitch_width: float
    pitch_height: float
    goal_width: float
    goal_height: float
    goal_y_min: float
    goal_y_max: float
    player_radius: float
    ball_radius: float
    team_a_color: List[int]
    team_b_color: List[int]


class AiActionRequest(BaseModel):
    """Request for the bound (or named) agent to take the current turn."""
    agent: Optional[str] = None  # override; default uses game binding


class AiActionResponse(BaseModel):
    """Response after an AI-selected flick."""
    success: bool
    error: Optional[str] = None
    agent: Optional[str] = None
    action: Optional[Dict[str, Any]] = None
    start_state: Optional[Dict[str, Any]] = None
    final_state: Optional[Dict[str, Any]] = None
    frames: Optional[List[Dict[str, Any]]] = None
    events: Optional[List[Dict[str, Any]]] = None
    simulation_duration: float = 0.0
    simulated_time: float = 0.0
    prompt_preview: Optional[str] = None


def _normalize_agent_name(name: str) -> str:
    cleaned = name.strip().lower()
    if cleaned in ("", "human", "none", "player"):
        return "human"
    return cleaned


def _bind_agents(
    game_id: str,
    team_a_agent: str,
    team_b_agent: str,
    config: SimConfig,
) -> tuple[str, str]:
    """Create/replace agent strategies for a game. 'human' → None."""
    a_name = _normalize_agent_name(team_a_agent)
    b_name = _normalize_agent_name(team_b_agent)

    def maybe_create(name: str, team: Team) -> Optional[Agent]:
        if name == "human":
            return None
        try:
            return create_agent(name, team, config)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (FileNotFoundError, RuntimeError, ImportError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    game_agents[game_id] = GameAgentBinding(
        team_a=maybe_create(a_name, Team.A),
        team_b=maybe_create(b_name, Team.B),
        team_a_name=a_name,
        team_b_name=b_name,
    )
    return a_name, b_name


@router.get("/agents")
async def list_agents() -> Dict[str, List[str]]:
    """List registered AI strategy names (plus 'human')."""
    return {"agents": ["human", *available_agents()]}


@router.post("/games", response_model=CreateGameResponse)
async def create_game(request: CreateGameRequest) -> CreateGameResponse:
    """Create a new game."""
    game_id = str(uuid.uuid4())[:8]
    
    sim = GameSimulator()
    starting_team = Team.A if request.starting_team == "A" else Team.B
    state = sim.new_game(starting_team)
    
    games[game_id] = sim
    a_name, b_name = _bind_agents(
        game_id, request.team_a_agent, request.team_b_agent, sim.config
    )
    
    return CreateGameResponse(
        game_id=game_id,
        state=serialize_state(state),
        team_a_agent=a_name,
        team_b_agent=b_name,
    )


@router.get("/games/{game_id}", response_model=GameStateResponse)
async def get_game(game_id: str) -> GameStateResponse:
    """Get current state of a game."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    sim = games[game_id]
    if sim.state is None:
        raise HTTPException(status_code=400, detail="Game not started")
    
    controllable = [p.id for p in sim.state.get_controllable_players()]
    binding = game_agents.get(game_id)
    a_name = binding.team_a_name if binding else "human"
    b_name = binding.team_b_name if binding else "human"
    current_is_ai = False
    if binding is not None:
        current_is_ai = binding.agent_for(sim.state.current_team) is not None
    
    return GameStateResponse(
        game_id=game_id,
        state=serialize_state(sim.state),
        controllable_players=controllable,
        team_a_agent=a_name,
        team_b_agent=b_name,
        current_team_is_ai=current_is_ai,
    )


@router.post("/games/{game_id}/actions", response_model=ActionResponse)
async def execute_action(game_id: str, request: ActionRequest) -> ActionResponse:
    """Execute a flick action in the game."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    sim = games[game_id]
    if sim.state is None:
        raise HTTPException(status_code=400, detail="Game not started")
    
    # Create action
    action = FlickAction(
        player_id=request.player_id,
        direction=Vec2(request.direction_x, request.direction_y),
        power=request.power,
    )
    
    # Execute
    result = sim.execute_action(action, capture_frames=True)
    
    if not result.success:
        return ActionResponse(
            success=False,
            error=result.error_message,
        )
    
    return ActionResponse(
        success=True,
        start_state=serialize_state(result.start_state) if result.start_state else None,
        final_state=serialize_state(result.final_state) if result.final_state else None,
        frames=[serialize_frame(f) for f in (result.frames or [])],
        events=[serialize_event(e) for e in (result.events or [])],
        simulation_duration=result.simulation_duration,
        simulated_time=result.simulated_time,
    )


@router.post("/games/{game_id}/reset", response_model=CreateGameResponse)
async def reset_game(game_id: str, request: CreateGameRequest) -> CreateGameResponse:
    """Reset a game to initial state."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    sim = games[game_id]
    starting_team = Team.A if request.starting_team == "A" else Team.B
    state = sim.new_game(starting_team)
    a_name, b_name = _bind_agents(
        game_id, request.team_a_agent, request.team_b_agent, sim.config
    )
    
    return CreateGameResponse(
        game_id=game_id,
        state=serialize_state(state),
        team_a_agent=a_name,
        team_b_agent=b_name,
    )


@router.post("/games/{game_id}/ai-action", response_model=AiActionResponse)
async def execute_ai_action(
    game_id: str,
    request: AiActionRequest,
) -> AiActionResponse:
    """
    Let the configured (or override) agent choose and apply a flick.

    Simulation still runs authoritatively on the backend; the agent only
    selects a FlickAction.
    """
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    sim = games[game_id]
    if sim.state is None:
        raise HTTPException(status_code=400, detail="Game not started")

    state = sim.state
    if state.phase not in (GamePhase.AIMING, GamePhase.KICKOFF):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot act in phase {state.phase.name}",
        )

    binding = game_agents.get(game_id)
    agent: Optional[Agent] = None
    agent_name = request.agent

    if agent_name:
        try:
            agent = create_agent(agent_name, state.current_team, sim.config)
            agent_name = agent.name
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (FileNotFoundError, RuntimeError, ImportError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    elif binding is not None:
        agent = binding.agent_for(state.current_team)
        agent_name = binding.name_for(state.current_team)

    if agent is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No AI agent bound for team {state.current_team.value}. "
                'Pass {"agent": "jev"} or create the game with team_*_agent.'
            ),
        )

    try:
        action = agent.select_action(state)
    except Exception as exc:  # noqa: BLE001 - surface agent failures cleanly
        return AiActionResponse(success=False, error=str(exc), agent=agent.name)

    result = sim.execute_action(action, capture_frames=True)
    if not result.success:
        return AiActionResponse(
            success=False,
            error=result.error_message,
            agent=agent.name,
            action=serialize_action(action),
        )

    prompt_preview = None
    if hasattr(agent, "last_prompt") and agent.last_prompt:  # type: ignore[attr-defined]
        prompt_preview = agent.last_prompt[:2000]  # type: ignore[attr-defined]

    return AiActionResponse(
        success=True,
        agent=agent.name,
        action=serialize_action(action),
        start_state=serialize_state(result.start_state) if result.start_state else None,
        final_state=serialize_state(result.final_state) if result.final_state else None,
        frames=[serialize_frame(f) for f in (result.frames or [])],
        events=[serialize_event(e) for e in (result.events or [])],
        simulation_duration=result.simulation_duration,
        simulated_time=result.simulated_time,
        prompt_preview=prompt_preview,
    )


@router.delete("/games/{game_id}")
async def delete_game(game_id: str) -> Dict[str, str]:
    """Delete a game."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    del games[game_id]
    game_agents.pop(game_id, None)
    return {"status": "deleted", "game_id": game_id}


@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """Get game configuration for frontend rendering."""
    config = SimConfig.default()
    physics = config.physics
    
    return ConfigResponse(
        pitch_width=physics.pitch_width,
        pitch_height=physics.pitch_height,
        goal_width=physics.goal_width,
        goal_height=physics.goal_height,
        goal_y_min=physics.goal_y_min,
        goal_y_max=physics.goal_y_max,
        player_radius=physics.player_radius,
        ball_radius=physics.ball_radius,
        team_a_color=list(config.team_a_color),
        team_b_color=list(config.team_b_color),
    )


@router.get("/games")
async def list_games() -> Dict[str, List[str]]:
    """List all active game IDs."""
    return {"games": list(games.keys())}
