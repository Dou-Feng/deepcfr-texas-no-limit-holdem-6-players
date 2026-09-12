import numpy as np

from server.inference_server import OpponentAction, OpponentHistory, _load_opponent_histories
from src.opponent_modeling.deep_cfr_with_opponent_modeling import (
    DeepCFRAgentWithOpponentModeling,
)


def test_load_opponent_histories_replays_request_into_om_agent(monkeypatch):
    agent = DeepCFRAgentWithOpponentModeling(player_id=0, num_players=6, device="cpu")
    context = np.linspace(0.0, 1.0, 25, dtype=np.float32)
    calls = []
    original = agent.record_opponent_action

    def record_spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(agent, "record_opponent_action", record_spy)
    features = _load_opponent_histories(
        agent,
        player_id=0,
        histories=[
            OpponentHistory(
                opponent_id=2,
                actions=[OpponentAction(action_id=4, context=context.tolist())],
            )
        ],
    )

    assert len(calls) == 1
    assert calls[0]["opponent_id"] == 2
    assert calls[0]["action_id"] == 4
    assert features.shape == (20,)
    action_sequence, contexts, outcome = agent.opponent_modeling.opponent_histories[2][0]
    assert np.array_equal(action_sequence[0], np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
    assert np.array_equal(contexts[0], context)
    assert outcome == 0.0
    assert agent.current_game_history == {}
