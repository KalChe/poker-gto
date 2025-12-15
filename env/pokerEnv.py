import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from utils.cards import best_five_score, shuffle_deck, winner

ACTIONS = ["fold", "call", "bet_25", "bet_50", "bet_100", "all_in"]
ACTION_FOLD = 0
ACTION_CALL = 1
ACTION_BET_25 = 2
ACTION_BET_50 = 3
ACTION_BET_100 = 4
ACTION_ALL_IN = 5

STREET_PREFLOP = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3


@dataclass
class PokerState:
    button: int
    deck: List[int]
    hole: List[List[int]]
    board: List[int]
    stacks: List[float]
    contributions: List[float]
    pot: float
    street: int
    current_player: int
    current_bet: float
    history: List[Tuple[int, int, int, float]] = field(default_factory=list)
    done: bool = False
    terminal_rewards: Tuple[float, float] | None = None
    all_in_runout: bool = False


class PokerEnv:
    def __init__(self, seed: int = 0, stack_bb: float = 50.0, small_blind: float = 0.5, big_blind: float = 1.0, history_len: int = 32):
        self.rng = random.Random(seed)
        self.stack_bb = stack_bb
        self.sb = small_blind
        self.bb = big_blind
        self.history_len = history_len
        self.state: PokerState | None = None
        self.hand_counter = 0

    def reset(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        button = self.hand_counter % 2
        deck = shuffle_deck(self.rng.randint(0, 1_000_000))
        hole = [deck[:2], deck[2:4]]
        deck = deck[4:]
        board: List[int] = []
        stacks = [self.stack_bb, self.stack_bb]
        contributions = [0.0, 0.0]
        pot = 0.0
        # blinds
        contributions[button] += self.sb
        contributions[1 - button] += self.bb
        stacks[button] -= self.sb
        stacks[1 - button] -= self.bb
        pot = self.sb + self.bb
        current_player = button
        current_bet = self.bb
        history = [(STREET_PREFLOP, button, ACTION_BET_50, self.sb / pot if pot else 0.0), (STREET_PREFLOP, 1 - button, ACTION_BET_100, self.bb / pot if pot else 0.0)]
        self.state = PokerState(button, deck, hole, board, stacks, contributions, pot, STREET_PREFLOP, current_player, current_bet, history)
        self.hand_counter += 1
        return self._build_obs(current_player), self.legal_action_mask()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray] | None, float, bool, Dict]:
        assert self.state is not None
        state = self.state
        if state.done:
            return None, 0.0, True, {}
        actor = state.current_player
        legal = self.legal_action_mask()
        if not legal[action]:
            action = ACTION_FOLD
        reward = 0.0
        info: Dict = {}
        self._apply_action(actor, action)
        if state.done:
            reward = state.terminal_rewards[actor] if state.terminal_rewards else 0.0
            info["rewards"] = state.terminal_rewards
            return None, reward, True, info
        state.current_player = 1 - actor
        return self._build_obs(state.current_player), reward, False, info

    def legal_action_mask(self) -> np.ndarray:
        assert self.state is not None
        s = self.state
        mask = np.zeros(len(ACTIONS), dtype=np.float32)
        if s.done:
            return mask
        player = s.current_player
        opp = 1 - player
        to_call = max(s.contributions) - s.contributions[player]
        stack = s.stacks[player]
        opp_stack = s.stacks[opp]
        if to_call > 0:
            mask[ACTION_FOLD] = 1
            mask[ACTION_CALL] = 1
        else:
            mask[ACTION_CALL] = 1
        can_raise = stack > to_call and not s.all_in_runout
        if can_raise:
            mask[ACTION_BET_25] = 1
            mask[ACTION_BET_50] = 1
            mask[ACTION_BET_100] = 1
            mask[ACTION_ALL_IN] = 1
        elif stack > 0 and not s.all_in_runout:
            mask[ACTION_ALL_IN] = 1
        if opp_stack <= 0:
            mask[ACTION_BET_25] = 0
            mask[ACTION_BET_50] = 0
            mask[ACTION_BET_100] = 0
        return mask

    def _apply_action(self, player: int, action: int) -> None:
        s = self.state
        assert s is not None
        opp = 1 - player
        to_call = max(s.contributions) - s.contributions[player]
        pot_before = s.pot
        if action == ACTION_FOLD:
            s.done = True
            win = opp
            s.terminal_rewards = (s.pot if win == 0 else -s.pot, s.pot if win == 1 else -s.pot)
            s.history.append((s.street, player, action, 0.0))
            return
        if action == ACTION_CALL:
            pay = min(to_call, s.stacks[player])
            s.stacks[player] -= pay
            s.contributions[player] += pay
            s.pot += pay
            s.history.append((s.street, player, action, pay / max(pot_before, 1e-6)))
            if self._round_complete():
                self._advance_street()
            return
        if action in (ACTION_BET_25, ACTION_BET_50, ACTION_BET_100):
            frac = {ACTION_BET_25: 0.25, ACTION_BET_50: 0.5, ACTION_BET_100: 1.0}[action]
            bet_size = s.pot * frac
            bet_size = max(bet_size, self.bb)
            total = min(s.stacks[player], to_call + bet_size)
            s.stacks[player] -= total
            s.contributions[player] += total
            s.pot += total
            s.current_bet = s.contributions[player]
            s.history.append((s.street, player, action, total / max(pot_before, 1e-6)))
            if s.stacks[player] <= 0:
                s.all_in_runout = True
                self._resolve_all_in()
            return
        if action == ACTION_ALL_IN:
            total = s.stacks[player]
            s.stacks[player] = 0.0
            s.contributions[player] += total
            s.pot += total
            s.current_bet = s.contributions[player]
            s.history.append((s.street, player, action, total / max(pot_before, 1e-6)))
            s.all_in_runout = True
            self._resolve_all_in()
            return

    def _round_complete(self) -> bool:
        s = self.state
        assert s is not None
        if s.done or s.all_in_runout:
            return True
        return s.contributions[0] == s.contributions[1]

    def _advance_street(self) -> None:
        s = self.state
        assert s is not None
        if s.street == STREET_PREFLOP:
            self._deal_to(3)
            s.street = STREET_FLOP
            return
        if s.street == STREET_FLOP:
            self._deal_to(4)
            s.street = STREET_TURN
            return
        if s.street == STREET_TURN:
            self._deal_to(5)
            s.street = STREET_RIVER
            return
        if s.street == STREET_RIVER:
            self._showdown()
            return

    def _deal_to(self, count: int) -> None:
        s = self.state
        assert s is not None
        while len(s.board) < count:
            s.board.append(s.deck.pop(0))
        s.current_bet = 0.0
        s.contributions = [0.0, 0.0]

    def _resolve_all_in(self) -> None:
        s = self.state
        assert s is not None
        if s.done:
            return
        if len(s.board) < 3:
            self._deal_to(3)
        if len(s.board) < 4:
            self._deal_to(4)
        if len(s.board) < 5:
            self._deal_to(5)
        self._showdown()

    def _showdown(self) -> None:
        s = self.state
        assert s is not None
        win = winner(s.hole[0], s.hole[1], s.board)
        if win == -1:
            rewards = (s.pot / 2.0, s.pot / 2.0)
        elif win == 0:
            rewards = (s.pot, -s.pot)
        else:
            rewards = (-s.pot, s.pot)
        s.done = True
        s.terminal_rewards = rewards

    def _build_obs(self, player: int) -> Dict[str, np.ndarray]:
        s = self.state
        assert s is not None
        hole = np.zeros(52, dtype=np.float32)
        for c in s.hole[player]:
            hole[c] = 1.0
        board = np.zeros(52, dtype=np.float32)
        for c in s.board:
            board[c] = 1.0
        street_vec = np.zeros(4, dtype=np.float32)
        street_vec[s.street] = 1.0
        effective = min(s.stacks[player], s.stacks[1 - player]) / self.stack_bb
        history_arr = np.zeros((self.history_len, 4), dtype=np.float32)
        recent = s.history[-self.history_len :]
        for i, (st, actor, act, size) in enumerate(recent):
            history_arr[i, 0] = float(st)
            history_arr[i, 1] = float(actor)
            history_arr[i, 2] = float(act)
            history_arr[i, 3] = float(size)
        obs = {
            "hole_one_hot": hole,
            "board_one_hot": board,
            "pot": np.array([s.pot / self.stack_bb], dtype=np.float32),
            "effective_stack": np.array([effective], dtype=np.float32),
            "street": street_vec,
            "history": history_arr,
        }
        return obs
