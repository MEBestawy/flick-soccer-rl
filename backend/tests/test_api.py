"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health endpoints."""
    
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["game"] == "soccer-sim"
    
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestGameEndpoints:
    """Tests for game management endpoints."""
    
    def test_create_game(self, client):
        response = client.post("/api/games", json={"starting_team": "A"})
        assert response.status_code == 200
        
        data = response.json()
        assert "game_id" in data
        assert "state" in data
        assert data["state"]["phase"] == "KICKOFF"
        assert data["state"]["current_team"] == "A"
    
    def test_create_game_team_b(self, client):
        response = client.post("/api/games", json={"starting_team": "B"})
        assert response.status_code == 200
        
        data = response.json()
        assert data["state"]["current_team"] == "B"
    
    def test_get_game(self, client):
        # Create game first
        create_response = client.post("/api/games", json={})
        game_id = create_response.json()["game_id"]
        
        # Get game
        response = client.get(f"/api/games/{game_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["game_id"] == game_id
        assert "state" in data
        assert "controllable_players" in data
        assert len(data["controllable_players"]) == 5
    
    def test_get_nonexistent_game(self, client):
        response = client.get("/api/games/nonexistent")
        assert response.status_code == 404
    
    def test_list_games(self, client):
        # Create a game
        client.post("/api/games", json={})
        
        response = client.get("/api/games")
        assert response.status_code == 200
        
        data = response.json()
        assert "games" in data
        assert isinstance(data["games"], list)
    
    def test_delete_game(self, client):
        # Create game
        create_response = client.post("/api/games", json={})
        game_id = create_response.json()["game_id"]
        
        # Delete game
        response = client.delete(f"/api/games/{game_id}")
        assert response.status_code == 200
        
        # Should be gone
        get_response = client.get(f"/api/games/{game_id}")
        assert get_response.status_code == 404


class TestActionEndpoints:
    """Tests for action execution endpoints."""
    
    def test_execute_action(self, client):
        # Create game
        create_response = client.post("/api/games", json={})
        game_id = create_response.json()["game_id"]
        
        # Execute action
        action_data = {
            "player_id": "A1",
            "direction_x": 1.0,
            "direction_y": 0.0,
            "power": 0.5
        }
        
        response = client.post(f"/api/games/{game_id}/actions", json=action_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == True
        assert "start_state" in data
        assert "final_state" in data
        assert "frames" in data
        assert "events" in data
        assert len(data["frames"]) > 0
    
    def test_execute_invalid_action(self, client):
        # Create game
        create_response = client.post("/api/games", json={})
        game_id = create_response.json()["game_id"]
        
        # Try to flick wrong team's player
        action_data = {
            "player_id": "B1",  # Wrong team
            "direction_x": 1.0,
            "direction_y": 0.0,
            "power": 0.5
        }
        
        response = client.post(f"/api/games/{game_id}/actions", json=action_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] == False
        assert "error" in data
    
    def test_execute_action_invalid_power(self, client):
        # Create game
        create_response = client.post("/api/games", json={})
        game_id = create_response.json()["game_id"]
        
        # Invalid power
        action_data = {
            "player_id": "A1",
            "direction_x": 1.0,
            "direction_y": 0.0,
            "power": 1.5  # Invalid
        }
        
        response = client.post(f"/api/games/{game_id}/actions", json=action_data)
        # Should fail validation
        assert response.status_code == 422


class TestConfigEndpoint:
    """Tests for config endpoint."""
    
    def test_get_config(self, client):
        response = client.get("/api/config")
        assert response.status_code == 200
        
        data = response.json()
        assert "pitch_width" in data
        assert "pitch_height" in data
        assert "goal_width" in data
        assert "player_radius" in data
        assert "ball_radius" in data
        assert data["pitch_width"] == 120.0
        assert data["pitch_height"] == 72.0


class TestResetEndpoint:
    """Tests for game reset endpoint."""
    
    def test_reset_game(self, client):
        # Create and play a game
        create_response = client.post("/api/games", json={})
        game_id = create_response.json()["game_id"]
        
        # Make an action
        action_data = {
            "player_id": "A1",
            "direction_x": 1.0,
            "direction_y": 0.0,
            "power": 0.3
        }
        client.post(f"/api/games/{game_id}/actions", json=action_data)
        
        # Reset
        response = client.post(f"/api/games/{game_id}/reset", json={"starting_team": "B"})
        assert response.status_code == 200
        
        data = response.json()
        assert data["state"]["turn_number"] == 1
        assert data["state"]["current_team"] == "B"
        assert data["state"]["score_a"] == 0
        assert data["state"]["score_b"] == 0
