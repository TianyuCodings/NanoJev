"""Finite, observable-composition Blackjack. Exact enumeration, no replacement.

Dealer has no predealt hole card: draws the second card after player stands.
This is explicitly a no-hole-card/no-peek, even-money hit/stand model.
"""
from functools import lru_cache

FULL_DECK = (4,) * 9 + (16,)
OUTCOMES = ('loss', 'push', 'win')


def add_card(total, soft, card):
    total += card
    if card == 1 and total + 10 <= 21:
        total += 10
        soft = True
    if total > 21 and soft:
        total -= 10
        soft = False
    return total, soft


def hand_value(cards):
    total = sum(cards)
    soft = 1 in cards and total + 10 <= 21
    return total + (10 if soft else 0), soft


def remove(counts, card):
    if not 1 <= card <= 10 or counts[card - 1] <= 0:
        raise ValueError('Card unavailable')
    result = list(counts)
    result[card - 1] -= 1
    return tuple(result)


def value(dist):
    return dist[2] - dist[0]


class FiniteSolver:
    def __init__(self, hits_soft_17=False):
        self.h17 = hits_soft_17
        # Instance-local caches can be released after each source-shoe group.
        self.dealer = lru_cache(None)(self._dealer)
        self.stand = lru_cache(None)(self._stand)
        self.hit = lru_cache(None)(self._hit)
        self.optimal = lru_cache(None)(self._optimal)

    def clear(self):
        for f in (self.dealer, self.stand, self.hit, self.optimal):
            f.cache_clear()

    @staticmethod
    def draws(counts):
        n = sum(counts)
        if n == 0:
            raise ValueError('Shoe exhausted before terminal hand; no refill is allowed')
        for i, count in enumerate(counts):
            if count:
                yield i + 1, count / n, remove(counts, i + 1)

    def _dealer(self, total, soft, counts):
        if total > 21:
            return (1., 0., 0., 0., 0., 0.)
        if total >= 17 and not (total == 17 and soft and self.h17):
            return tuple(float(i == total - 16) for i in range(6))
        result = [0.] * 6
        for card, p, remaining in self.draws(counts):
            child = self.dealer(*add_card(total, soft, card), remaining)
            for i in range(6):
                result[i] += p * child[i]
        return tuple(result)

    def _stand(self, total, upcard, counts):
        dealer = self.dealer(*add_card(0, False, upcard), counts)
        result = [0., 0., dealer[0]]
        for final, p in zip(range(17, 22), dealer[1:]):
            result[0 if total < final else 1 if total == final else 2] += p
        return tuple(result)

    def _hit(self, total, soft, upcard, counts):
        result = [0.] * 3
        for card, p, remaining in self.draws(counts):
            nxt, usable = add_card(total, soft, card)
            child = (1., 0., 0.) if nxt > 21 else self.optimal(nxt, usable, upcard, remaining)
            for i in range(3):
                result[i] += p * child[i]
        return tuple(result)

    def _optimal(self, total, soft, upcard, counts):
        stand = self.stand(total, upcard, counts)
        hit = self.hit(total, soft, upcard, counts)
        return hit if value(hit) > value(stand) + 1e-12 else stand

    def solve(self, cards, upcard, counts):
        if len(counts) != 10 or any(type(n) is not int or n < 0 for n in counts):
            raise ValueError('Expected ten nonnegative integer counts')
        total, soft = hand_value(cards)
        if not 4 <= total <= 21 or upcard not in range(1, 11):
            raise ValueError('Invalid visible state')
        return {'hit': self.hit(total, soft, upcard, counts),
                'stand': self.stand(total, upcard, counts)}
