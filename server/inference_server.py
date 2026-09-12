"""HTTP inference server exposing the trained Deep CFR agent to go-poker.

go-poker (github.com/evanofslack/go-poker) keeps its own engine, so this
service accepts a *neutral* JSON description of a poker state (already
translated to the pokers conventions the model was trained on), runs the
strategy network, and answers with an action in go-poker's ``Bet``/``Fold``
semantics.

Run:
    python -m server.inference_server \
        --checkpoint models/standard/phase2/checkpoint_iter_1000.pt \
        --host 127.0.0.1 --port 8001

go-poker side:
    AI_INFERENCE_URL=http://127.0.0.1:8001

Field conventions (all in pokers numbering, matching training):
    suit: 0=Clubs 1=Diamonds 2=Hearts 3=Spades
    rank: 0=deuce ... 12=ace
    stage: 0=Preflop 1=Flop 2=Turn 3=River 4=Showdown
    ``min_bet`` is the highest bet on the street (the amount a call matches),
    i.e. go-poker's ``toCall()``.
    ``players`` must contain exactly 6 slots (inactive seats are zeroed) so the
    156-dim feature vector matches the trained input size.
"""

import argparse
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import pokers as pkrs

from src.core.deep_cfr import DeepCFRAgent
from src.core.model import encode_state
from src.utils.actions import (
    ACTION_TYPE_FOLD,
    RAISE_ACTION_MULTIPLIERS,
    action_type_to_pokers_action,
    legal_action_types,
    raise_bounds,
)

SUIT_CLUBS, SUIT_DIAMONDS, SUIT_HEARTS, SUIT_SPADES = 0, 1, 2, 3

ACTION_ENUMS = {
    "fold": pkrs.ActionEnum.Fold,
    "check": pkrs.ActionEnum.Check,
    "call": pkrs.ActionEnum.Call,
    "raise": pkrs.ActionEnum.Raise,
}

ACTION_LABELS = {
    ACTION_TYPE_FOLD: "fold",
    1: "check_call",
    2: "raise_half_pot",
    3: "raise_pot",
    4: "raise_overbet",
}


class PlayerSlot(BaseModel):
    active: bool = False
    bet: float = 0.0
    pot_chips: float = 0.0
    stake: float = 0.0


class ActionRequest(BaseModel):
    player_id: int = Field(ge=0, le=5)
    hand: List[List[int]] = Field(default_factory=list)
    community: List[List[int]] = Field(default_factory=list)
    stage: int = Field(ge=0, le=4, default=0)
    pot: float = Field(ge=0, default=0.0)
    min_bet: float = Field(ge=0, default=0.0)
    min_raise: float = Field(ge=0, default=0.0)
    bb: float = Field(gt=0, default=2.0)
    button: int = Field(ge=0, le=5, default=0)
    current_player: int = Field(ge=0, le=5, default=0)
    players: List[PlayerSlot] = Field(default_factory=list)
    legal_actions: List[str] = Field(default_factory=list)
    sample: bool = True


class ActionResponse(BaseModel):
    kind: str  # "fold" | "check" | "call" | "raise" (go-poker botAction kinds)
    amount: int  # chips to put in with this action (go-poker Bet() semantics)
    action_type: int
    label: str
    call_amount: float
    raise_additional: float
    probabilities: Dict[str, float]
    elapsed_ms: float


def _card(pair: List[int]) -> SimpleNamespace:
    suit, rank = int(pair[0]), int(pair[1])
    if not (0 <= suit <= 3 and 0 <= rank <= 12):
        raise ValueError(f"card out of range: suit={suit} rank={rank}")
    return SimpleNamespace(suit=suit, rank=rank)


def _pseudo_state(req: ActionRequest) -> SimpleNamespace:
    """Build a duck-typed stand-in for a pokers state.

    ``encode_state`` and the raise-building helpers only touch attributes, so
    the full training-side feature pipeline and bet clamping are reused as-is.
    """
    try:
        legal = [ACTION_ENUMS[name] for name in req.legal_actions]
    except KeyError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"unknown legal action {exc}; expected one of {sorted(ACTION_ENUMS)}",
        ) from exc

    players_state = [
        SimpleNamespace(
            bet_chips=float(p.bet),
            stake=float(p.stake),
            pot_chips=float(p.pot_chips),
            active=bool(p.active),
            hand=[],
        )
        for p in req.players
    ]
    players_state[req.player_id].hand = [_card(pair) for pair in req.hand]

    # encode_state normalizes every money feature by seat 0's remaining stake.
    # A game in progress always has a real player at position 0 (go-poker
    # compacts positions on leave), but guard anyway: if slot 0 is empty its
    # stake would be 0 and encode_state would fall back to dividing by 1.0,
    # pushing pot/bet features ~200x out of the trained range. Substitute the
    # largest stack at the table (or the classic 100bb training stack) so the
    # features stay in-distribution.
    if players_state[0].stake <= 0:
        fallback = max((p.stake for p in players_state[1:]), default=0.0)
        if fallback <= 0:
            fallback = float(req.bb) * 100.0
        players_state[0].stake = fallback

    bb_estimate = req.min_raise if req.min_raise > 0 else req.bb
    return SimpleNamespace(
        players_state=players_state,
        public_cards=[_card(pair) for pair in req.community],
        stage=req.stage,
        pot=float(req.pot),
        button=req.button,
        current_player=req.current_player,
        min_bet=float(req.min_bet),
        bb=float(bb_estimate),  # min_raise_increment() uses bb as the increment
        legal_actions=legal,
        from_action=None,
        final_state=False,
    )


def create_app(checkpoint_path: str, device: str = "cpu") -> FastAPI:
    agent = DeepCFRAgent(player_id=0, num_players=6, device=device)
    agent.load_model(checkpoint_path)
    agent.strategy_net.eval()

    app = FastAPI(title="deepcfr inference", version="1")

    @app.get("/healthz")
    def healthz():
        return {
            "ok": True,
            "checkpoint": checkpoint_path,
            "iteration": agent.iteration_count,
            "num_actions": agent.num_actions,
        }

    @app.post("/v1/act", response_model=ActionResponse)
    def act(req: ActionRequest) -> ActionResponse:
        if len(req.players) != agent.num_players:
            raise HTTPException(
                status_code=422,
                detail=f"players must have {agent.num_players} slots, got {len(req.players)}",
            )
        started = time.perf_counter()
        state = _pseudo_state(req)

        legal_types = legal_action_types(state, num_actions=agent.num_actions)
        if not legal_types:
            raise HTTPException(status_code=422, detail="no legal actions provided")

        with torch.no_grad():
            encoded = encode_state(state, req.player_id)
            tensor = torch.FloatTensor(encoded).unsqueeze(0).to(agent.device)
            logits, _ = agent.strategy_net(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        legal_probs = np.array([probs[a] for a in legal_types])
        legal_probs = legal_probs / legal_probs.sum() if legal_probs.sum() > 0 else (
            np.ones(len(legal_types)) / len(legal_types)
        )
        if req.sample:
            action_type = int(np.random.choice(legal_types, p=legal_probs))
        else:
            action_type = int(legal_types[int(np.argmax(legal_probs))])

        full_probs = np.zeros(agent.num_actions)
        for atype, prob in zip(legal_types, legal_probs):
            full_probs[atype] = float(prob)

        action = action_type_to_pokers_action(
            action_type,
            state,
            bet_size_multiplier=RAISE_ACTION_MULTIPLIERS.get(action_type),
            strict=True,
        )

        bounds = raise_bounds(state)
        call_amount = bounds.call_amount
        if action.action == pkrs.ActionEnum.Fold:
            kind, amount, raise_additional = "fold", 0.0, 0.0
        elif action.action == pkrs.ActionEnum.Check:
            kind, amount, raise_additional = "check", 0.0, 0.0
        elif action.action == pkrs.ActionEnum.Call:
            kind, amount, raise_additional = "call", call_amount, 0.0
        else:
            raise_additional = float(action.amount)
            amount = call_amount + raise_additional
            kind = "raise"

        return ActionResponse(
            kind=kind,
            amount=int(round(amount)),
            action_type=action_type,
            label=ACTION_LABELS.get(action_type, str(action_type)),
            call_amount=call_amount,
            raise_additional=raise_additional,
            probabilities={ACTION_LABELS[i]: float(full_probs[i]) for i in range(agent.num_actions)},
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Deep CFR inference server for go-poker")
    parser.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    import uvicorn

    app = create_app(args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
