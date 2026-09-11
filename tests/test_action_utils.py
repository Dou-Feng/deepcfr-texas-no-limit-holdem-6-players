import types

import pokers as pkrs

from src.utils.actions import (
    ACTION_TYPE_CHECK_CALL,
    ACTION_TYPE_FOLD,
    ACTION_TYPE_RAISE,
    ACTION_TYPE_RAISE_HALF_POT,
    ACTION_TYPE_RAISE_OVERBET,
    ACTION_TYPE_RAISE_POT,
    NUM_ACTIONS_DISCRETE,
    action_type_to_pokers_action,
    build_raise_action,
    legal_action_types,
    raise_bounds,
    sanitize_action,
)


def make_state(
    legal_actions,
    *,
    current_player=0,
    min_bet=2.0,
    pot=10.0,
    current_bet=0.0,
    stake=20.0,
    bb=2.0,
):
    players_state = [
        types.SimpleNamespace(bet_chips=current_bet, stake=stake),
        types.SimpleNamespace(bet_chips=0.0, stake=20.0),
    ]
    return types.SimpleNamespace(
        legal_actions=list(legal_actions),
        current_player=current_player,
        players_state=players_state,
        min_bet=min_bet,
        pot=pot,
        bb=bb,
    )


def test_legal_action_types_collapses_check_call():
    state = make_state(
        [
            pkrs.ActionEnum.Fold,
            pkrs.ActionEnum.Check,
            pkrs.ActionEnum.Raise,
        ]
    )

    assert legal_action_types(state) == [
        ACTION_TYPE_FOLD,
        ACTION_TYPE_CHECK_CALL,
        ACTION_TYPE_RAISE,
    ]


def test_legal_action_types_expands_discrete_raise_sizings():
    state = make_state(
        [
            pkrs.ActionEnum.Fold,
            pkrs.ActionEnum.Check,
            pkrs.ActionEnum.Raise,
        ]
    )

    assert legal_action_types(state, num_actions=NUM_ACTIONS_DISCRETE) == [
        ACTION_TYPE_FOLD,
        ACTION_TYPE_CHECK_CALL,
        ACTION_TYPE_RAISE_HALF_POT,
        ACTION_TYPE_RAISE_POT,
        ACTION_TYPE_RAISE_OVERBET,
    ]


def test_discrete_raise_action_types_use_fixed_pot_multipliers():
    state = make_state(
        [pkrs.ActionEnum.Fold, pkrs.ActionEnum.Call, pkrs.ActionEnum.Raise],
        min_bet=2.0,
        current_bet=0.0,
        stake=100.0,
        pot=10.0,
        bb=2.0,
    )

    half = action_type_to_pokers_action(ACTION_TYPE_RAISE_HALF_POT, state)
    pot_raise = action_type_to_pokers_action(ACTION_TYPE_RAISE_POT, state)
    overbet = action_type_to_pokers_action(ACTION_TYPE_RAISE_OVERBET, state)

    assert half.action == pkrs.ActionEnum.Raise
    assert half.amount == 5.0  # 0.5 * pot
    assert pot_raise.action == pkrs.ActionEnum.Raise
    assert pot_raise.amount == 10.0  # 1.0 * pot
    assert overbet.action == pkrs.ActionEnum.Raise
    assert overbet.amount == 20.0  # 2.0 * pot


def test_overbet_clamps_to_remaining_stake():
    state = make_state(
        [pkrs.ActionEnum.Fold, pkrs.ActionEnum.Call, pkrs.ActionEnum.Raise],
        min_bet=2.0,
        current_bet=0.0,
        stake=12.0,
        pot=10.0,
        bb=2.0,
    )

    action = action_type_to_pokers_action(ACTION_TYPE_RAISE_OVERBET, state)

    assert action.action == pkrs.ActionEnum.Raise
    assert action.amount == 10.0  # max additional raise with 12 stake and 2 to call


def test_build_raise_action_clamps_to_remaining_stake():
    state = make_state(
        [pkrs.ActionEnum.Fold, pkrs.ActionEnum.Call, pkrs.ActionEnum.Raise],
        min_bet=6.0,
        current_bet=2.0,
        stake=20.0,
        bb=2.0,
    )

    bounds = raise_bounds(state)
    action = build_raise_action(state, 1000.0)

    assert bounds.call_amount == 4.0
    assert bounds.min_raise == 2.0
    assert bounds.max_raise == 16.0
    assert action.action == pkrs.ActionEnum.Raise
    assert action.amount == 16.0


def test_raise_falls_back_when_call_leaves_less_than_min_raise():
    state = make_state(
        [pkrs.ActionEnum.Fold, pkrs.ActionEnum.Call, pkrs.ActionEnum.Raise],
        min_bet=6.0,
        current_bet=2.0,
        stake=5.0,
        bb=2.0,
    )

    action = build_raise_action(state, 10.0)

    assert action.action == pkrs.ActionEnum.Call


def test_action_type_raise_uses_pot_multiplier_and_sanitizes():
    state = make_state(
        [pkrs.ActionEnum.Fold, pkrs.ActionEnum.Call, pkrs.ActionEnum.Raise],
        min_bet=2.0,
        current_bet=0.0,
        stake=12.0,
        pot=10.0,
        bb=2.0,
    )

    action = action_type_to_pokers_action(
        ACTION_TYPE_RAISE,
        state,
        bet_size_multiplier=3.0,
    )

    assert action.action == pkrs.ActionEnum.Raise
    assert action.amount == 10.0


def test_build_raise_action_steps_down_when_engine_rejects_exact_max():
    state = make_state(
        [pkrs.ActionEnum.Fold, pkrs.ActionEnum.Call, pkrs.ActionEnum.Raise],
        min_bet=2.0,
        current_bet=0.0,
        stake=12.0,
        pot=10.0,
        bb=2.0,
    )

    def apply_action(action):
        status = (
            pkrs.StateStatus.HighBet
            if action.action == pkrs.ActionEnum.Raise and action.amount >= 10.0
            else pkrs.StateStatus.Ok
        )
        return types.SimpleNamespace(status=status)

    state.apply_action = apply_action

    action = build_raise_action(state, 1000.0)

    assert action.action == pkrs.ActionEnum.Raise
    assert action.amount < 10.0
    assert action.amount >= 2.0


def test_sanitize_action_preserves_legal_non_raise_and_fallbacks_illegal():
    state = make_state([pkrs.ActionEnum.Check])

    assert sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Check)).action == pkrs.ActionEnum.Check
    assert sanitize_action(state, pkrs.Action(pkrs.ActionEnum.Raise, 100.0)).action == pkrs.ActionEnum.Check
