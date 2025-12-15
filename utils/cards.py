import itertools
import random
from typing import List, Tuple

RANKS = list(range(13))
SUITS = list(range(4))
RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUIT_NAMES = ["s", "h", "d", "c"]


def card_rank(card: int) -> int:
    return card // 4


def card_suit(card: int) -> int:
    return card % 4


def card_str(card: int) -> str:
    return f"{RANK_NAMES[card_rank(card)]}{SUIT_NAMES[card_suit(card)]}"


def full_deck() -> List[int]:
    return list(range(52))


def shuffle_deck(seed: int = None) -> List[int]:
    rng = random.Random(seed)
    deck = full_deck()
    rng.shuffle(deck)
    return deck


def best_five_score(cards: List[int]) -> Tuple[int, List[int]]:
    best = None
    for combo in itertools.combinations(cards, 5):
        score = hand_score(combo)
        if best is None or score > best:
            best = score
    return best  # type: ignore


def hand_score(cards5: Tuple[int, ...]) -> Tuple[int, List[int]]:
    ranks = sorted([card_rank(c) for c in cards5], reverse=True)
    suits = [card_suit(c) for c in cards5]
    counts = {r: ranks.count(r) for r in set(ranks)}
    is_flush = len(set(suits)) == 1
    sorted_ranks = sorted(counts.items(), key=lambda x: (-x[1], -x[0]))
    unique_ranks = sorted(set(ranks), reverse=True)
    straight_high = straight_high_rank(ranks)

    if straight_high is not None and is_flush:
        return (8, [straight_high])
    if sorted_ranks[0][1] == 4:
        four = sorted_ranks[0][0]
        kicker = max([r for r in ranks if r != four])
        return (7, [four, kicker])
    if sorted_ranks[0][1] == 3 and sorted_ranks[1][1] == 2:
        triple = sorted_ranks[0][0]
        pair = sorted_ranks[1][0]
        return (6, [triple, pair])
    if is_flush:
        return (5, unique_ranks)
    if straight_high is not None:
        return (4, [straight_high])
    if sorted_ranks[0][1] == 3:
        triple = sorted_ranks[0][0]
        kickers = [r for r in unique_ranks if r != triple]
        return (3, [triple] + kickers)
    if sorted_ranks[0][1] == 2 and sorted_ranks[1][1] == 2:
        pair_high = max(sorted_ranks[0][0], sorted_ranks[1][0])
        pair_low = min(sorted_ranks[0][0], sorted_ranks[1][0])
        kicker = max([r for r in unique_ranks if r not in (pair_high, pair_low)])
        return (2, [pair_high, pair_low, kicker])
    if sorted_ranks[0][1] == 2:
        pair = sorted_ranks[0][0]
        kickers = [r for r in unique_ranks if r != pair]
        return (1, [pair] + kickers)
    return (0, unique_ranks)


def straight_high_rank(ranks: List[int]) -> int | None:
    uniq = sorted(set(ranks), reverse=True)
    if len(uniq) < 5:
        if set([12, 3, 2, 1, 0]).issubset(set(ranks)):
            return 3
        return None
    for i in range(len(uniq) - 4):
        window = uniq[i:i + 5]
        if window[0] - window[4] == 4:
            return window[0]
    if set([12, 3, 2, 1, 0]).issubset(set(ranks)):
        return 3
    return None


def winner(hole_a: List[int], hole_b: List[int], board: List[int]) -> int:
    score_a = best_five_score(hole_a + board)
    score_b = best_five_score(hole_b + board)
    if score_a > score_b:
        return 0
    if score_b > score_a:
        return 1
    return -1
