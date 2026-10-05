# Flick Football ⚽

A browser-based turn-based 5-vs-5 physics soccer game (flick football). Players take turns flicking their discs to hit the ball into the opponent's goal.

## Architecture

**Critical Design Principle**: Game state and physics simulation are **completely independent** of the UI. The Python headless simulator is authoritative. The web app is a visual client only.

```
/backend
  /sim          # Headless physics simulation (NO dependencies on FastAPI)
    __init__.py, config.py, geometry.py, models.py, actions.py, 
    physics.py, collisions.py, arena.py, rules.py, simulator.py, 
    events.py, formations.py, serialization.py, env.py
    /agents     # Strategy-pattern AI (Jev, random, heuristic, …)
  /api          # FastAPI REST endpoints
    main.py, routes.py
  /tests        # Comprehensive pytest suite

/frontend
  /src
    /components  # React UI components
    /game       # Game client (renderer, API client, hooks)
    App.tsx

/scripts
  benchmark_sim.py   # Performance benchmarks
  random_game.py     # Play random games
  replay_game.py     # Save/load replays
  jev_game.py        # Headless match with Jev (or other agents)
```

## Physics Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| `pitch_width` | 120.0 | World units |
| `pitch_height` | 72.0 | World units |
| `player_radius` | 2.3 | Player disc radius |
| `ball_radius` | 1.35 | Ball radius |
| `player_mass` | 4.0 | Player mass |
| `ball_mass` | 1.0 | Ball mass |
| `player_drag` | 3.5 | Player friction coefficient |
| `ball_drag` | 1.2 | Ball friction (lower = farther travel) |
| `restitution` | 0.75 | Bounce coefficient |
| `max_launch_speed` | 120.0 | Maximum flick velocity |
| `timestep` | 1/120 | Physics timestep (120 Hz) |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) (Python package manager)

### Installation

```bash
# Backend
cd backend
uv sync

# Frontend
cd ../frontend
npm install
```

### Running the Game

```bash
# Terminal 1: Start backend API
cd backend
uv run uvicorn api.main:app --reload --port 8000

# Terminal 2: Start frontend dev server
cd frontend
npm run dev
```

Open http://localhost:3000 to play!

### How to Play

1. Click on one of your team's players (yellow or pink depending on turn)
2. Drag back like a slingshot to aim
3. Release to flick the player toward the ball
4. First team to score 3 goals wins!

## AI Agents (Strategy Pattern)

Both sides of a match are swappable `Agent` strategies. The simulator never hardcodes AI logic — it only consumes `FlickAction`s.

| Strategy | Name | Description |
|----------|------|-------------|
| Human | `human` | Browser / API client supplies actions |
| Jev | `jev` | TypeSafe System One typed decisions |
| Heuristic | `heuristic` | Nearest-player-toward-ball baseline |
| Random | `random` | Uniform legal actions |

```python
from sim import Team
from sim.agents import create_agent, MatchRunner, JevAgent

# Replace either side without changing match code
result = MatchRunner(
    team_a=create_agent("heuristic", Team.A),
    team_b=create_agent("jev", Team.B),  # or RandomAgent / custom Agent subclass
).play()
```

### Jev prompt (normalized)

Before each flick, `JevAgent` builds a normalized state prompt that includes:

- Player positions & radii (normalized)
- Ball position, radius, mass, drag
- Player / ball friction (drag normalized so players = 1.0)
- Score labelled as **Jev** vs **Opponent**
- Goal-post positions and field dimensions (normalized)

Coordinates: pitch center `(0,0)`, X/Y in `[-1, 1]`. Sizes ÷ pitch half-diagonal. Drag ÷ `player_drag`.

### Configure API key

Copy `.env.example` → `.env` (already gitignored):

```bash
TYPESAFE_API_KEY=apikey_...
JEV_API_ENDPOINT=https://api.typesafe.ai/v1/systemone
```

### Headless Jev match

```bash
# Preview the normalized prompt
PYTHONPATH=backend uv run --directory backend python ../scripts/jev_game.py --show-prompt-only

# Heuristic (A) vs Jev (B)
PYTHONPATH=backend uv run --directory backend python ../scripts/jev_game.py

# Swap strategies freely
PYTHONPATH=backend uv run --directory backend python ../scripts/jev_game.py --team-a jev --team-b random
```

### API

```http
POST /api/games
{ "starting_team": "A", "team_a_agent": "human", "team_b_agent": "jev" }

POST /api/games/{id}/ai-action
{ }   # uses bound agent, or pass { "agent": "jev" }
```

In the UI, pick Team A / Team B strategies on the start screen (default: Human vs Jev).

## Headless Simulation

The simulation can run completely without any UI:

```python
from sim import GameSimulator, FlickAction, HeadlessEnv, Vec2

# Option 1: Low-level simulator API
sim = GameSimulator()
state = sim.new_game()

action = FlickAction(
    player_id="A1",
    direction=Vec2(1.0, 0.5),
    power=0.7
)
result = sim.execute_action(action)
print(f"Simulated {result.simulated_time:.2f}s, captured {len(result.frames)} frames")

# Option 2: Gym-like environment for AI/RL
env = HeadlessEnv()
state = env.reset()

while not env.is_done():
    action = env.random_action()
    state, reward, done, info = env.step(action)
    
print(f"Final score: {state.score_a}-{state.score_b}")
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/games` | POST | Create new game (optional `team_a_agent` / `team_b_agent`) |
| `/api/games/{id}` | GET | Get game state |
| `/api/games/{id}/actions` | POST | Execute flick action |
| `/api/games/{id}/ai-action` | POST | Let bound/override agent choose + apply a flick |
| `/api/games/{id}/reset` | POST | Reset game |
| `/api/agents` | GET | List registered agent strategies |
| `/api/config` | GET | Get game configuration |

### Action Request

```json
{
  "player_id": "A1",
  "direction_x": 1.0,
  "direction_y": 0.5,
  "power": 0.7
}
```

### Action Response

```json
{
  "success": true,
  "start_state": { ... },
  "final_state": { ... },
  "frames": [ ... ],
  "events": [ ... ],
  "simulation_duration": 0.025,
  "simulated_time": 2.5
}
```

## Running Tests

```bash
cd backend
uv run pytest -v
```

## Running Benchmarks

```bash
cd backend
uv run python ../scripts/benchmark_sim.py
```

## Running Random Games

```bash
# Single game with output
cd backend
uv run python ../scripts/random_game.py

# Multiple games (statistics)
uv run python ../scripts/random_game.py -n 100
```

## Key Design Decisions

### Determinism
- Fixed timestep physics (120 Hz)
- Exponential drag: `velocity *= exp(-drag * dt)`
- Deterministic collision ordering
- Same input = same output, always

### Performance
- Adaptive substepping for high-speed objects
- CCD (Continuous Collision Detection) threshold
- Sleep system for at-rest objects
- Optional frame capture (disable for faster simulation)

### Separation of Concerns
- `sim` package has ZERO dependencies on FastAPI or web frameworks
- All physics tuning in `config.py`, not hardcoded
- Events system for replays and logging
- Serialization layer for API communication

## Development

### Type Checking

```bash
# Backend
cd backend
uv run pyright

# Frontend
cd frontend
npm run typecheck
```

### Linting

```bash
# Frontend
cd frontend
npm run lint
```

### Production Build

```bash
cd frontend
npm run build
```

## Known Limitations

1. **No WebSocket support yet** - Polling-based updates only
2. **Single-player only** - Both teams controlled locally
3. **No sound effects** - Visual feedback only
4. **No persistence** - Games stored in memory only
5. **No AI opponent** - Random actions only in scripts

## Recommended Improvements

1. Add WebSocket for real-time updates during simulation
2. Implement AI opponent using Monte Carlo Tree Search
3. Add sound effects for collisions and goals
4. Persist games to database for resume functionality
5. Add multiplayer support with game lobbies
6. Mobile touch optimization
7. Replay system with seek/playback controls
8. Tournament mode with brackets

## License

MIT
