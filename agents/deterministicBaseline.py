import numpy as np

from utils.cards import best_five_score, card_rank, card_suit
from env.pokerEnv import ACTION_ALL_IN, ACTION_BET_25, ACTION_BET_50, ACTION_BET_100, ACTION_CALL, ACTION_FOLD, STREET_PREFLOP


def hand_strength_score(hole: np.ndarray, board: np.ndarray) -> int:
    hole_idx = list(np.where(hole > 0)[0])
    board_idx = list(np.where(board > 0)[0])
    cards = hole_idx + board_idx
    if len(cards) < 5:
        return 0
    score = best_five_score(cards)[0]
    return score


def is_open_ended(hole: np.ndarray, board: np.ndarray) -> bool:
    hole_idx = list(np.where(hole > 0)[0])
    board_idx = list(np.where(board > 0)[0])
    ranks = sorted(set([card_rank(c) for c in hole_idx + board_idx]))
    for r in ranks:
        window = set([r, r + 1, r + 2, r + 3])
        if window.issubset(set(ranks)):
            return True
    if set([12, 0, 1, 2]).issubset(set(ranks)):
        return True
    return False


def preflop_bucket(hole: np.ndarray) -> str:
    cards = list(np.where(hole > 0)[0])
    r1, r2 = card_rank(cards[0]), card_rank(cards[1])
    s1, s2 = card_suit(cards[0]), card_suit(cards[1])
    high = max(r1, r2)
    low = min(r1, r2)
    suited = s1 == s2
    pair = r1 == r2
    if pair and high >= 10:
        return "premium"
    if pair and high >= 6:
        return "strong"
    if suited and high >= 10 and low >= 7:
        return "strong"
    if high >= 11 and low >= 9:
        return "good"
    if suited and high >= 8 and low >= 5:
        return "speculative"
    return "trash"


def baseline_policy(obs: dict, legal_mask: np.ndarray) -> int:
    hole = obs["hole_one_hot"]
    board = obs["board_one_hot"]
    street = int(np.argmax(obs["street"]))
    mask = legal_mask
    if street == STREET_PREFLOP:
        bucket = preflop_bucket(hole)
        if bucket == "premium":
            if mask[ACTION_BET_50]:
                return ACTION_BET_50
            return ACTION_ALL_IN if mask[ACTION_ALL_IN] else ACTION_CALL
        if bucket == "strong":
            if mask[ACTION_BET_25]:
                return ACTION_BET_25
            return ACTION_CALL
        if bucket == "good":
            return ACTION_CALL if mask[ACTION_CALL] else ACTION_FOLD
        if bucket == "speculative":
            return ACTION_CALL if mask[ACTION_CALL] else ACTION_FOLD
        return ACTION_FOLD if mask[ACTION_FOLD] else ACTION_CALL
    strength = hand_strength_score(hole, board)
    draw = is_open_ended(hole, board)
    if strength >= 5:
        if mask[ACTION_BET_100]:
            return ACTION_BET_100
        if mask[ACTION_ALL_IN]:
            return ACTION_ALL_IN
    if strength >= 3:
        if mask[ACTION_BET_50]:
            return ACTION_BET_50
        return ACTION_CALL if mask[ACTION_CALL] else ACTION_ALL_IN
    if strength == 2 or draw:
        return ACTION_CALL if mask[ACTION_CALL] else ACTION_FOLD
    return ACTION_FOLD if mask[ACTION_FOLD] else ACTION_CALL
