#!/usr/bin/env python3
"""4,000 distinct finite-deck decisions, grouped by 100 source compositions."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import dump, write_splits, SPLITS
from blackjack.finite_solver import FiniteSolver, FULL_DECK, OUTCOMES, hand_value, remove, value


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()[:24]


def profiles(seed=29, n=100):
    if n < 10 or n % 10:
        raise ValueError('Number of source shoes must be a multiple of ten, at least ten')
    rng = random.Random(seed)
    found = set()
    while len(found) < n:
        deck = [rank for rank, count in enumerate(FULL_DECK, 1) for _ in range(count)]
        rng.shuffle(deck)
        # Choose size once, then keep that single actual subset of one deck.
        kept = deck[:rng.randint(32, 44)]
        counts = tuple(kept.count(rank) for rank in range(1, 11))
        if counts[0] >= 2 and counts[5] >= 1 and counts[9] >= 2:
            found.add(counts)
    ordered = sorted(found, key=lambda c: fingerprint([seed, c]))
    boundaries = [int(n * f) for f in (.6, .7, .8, .9, 1.)]
    for index, counts in enumerate(ordered):
        split = next(s for s, end in zip(SPLITS, boundaries) if index < end)
        yield counts, split


def possible_hands():
    return [cards for length in (2, 3) for cards in combinations_with_replacement(range(1, 11), length)
            if 12 <= hand_value(cards)[0] <= 21]


def record(cards, upcard, remaining, parent, split, solver):
    total, soft = hand_value(cards)
    dists = solver.solve(cards, upcard, remaining)
    values = {a: value(d) for a, d in dists.items()}
    maximum = max(values.values())
    optimal = [a for a in values if maximum - values[a] <= 1e-12]
    questions = {'best_action': {'type': 'choice',
        'instructions': 'Which first action maximizes expected final reward with the stated optimal continuation?',
        'criteria': {'hit': 'Draw a card without replacement, then continue optimally.',
                     'stand': 'Stop drawing; let the dealer finish.'}}}
    probs = {'best_action': {a: float(a in optimal) / len(optimal) for a in values}}
    kinds = {'best_action': 'optimal_action_policy'}
    labels = {'best_action': 'reference_argmax_compatibility'}
    for action, distribution in dists.items():
        qid = action + '_outcome'
        questions[qid] = {'type': 'choice',
            'instructions': f'What is the final hand outcome after choosing {action} first, with the stated continuation?',
            'criteria': {'loss': 'Player loses (reward -1).', 'push': 'Tie (reward 0).', 'win': 'Player wins (reward +1).'}}
        probs[qid] = dict(zip(OUTCOMES, distribution))
        questions[action + '_win'] = {'type': 'boolean',
            'instructions': f'The player wins after choosing {action} first and following the stated continuation; a tie is not a win.'}
        probs[action + '_win'] = {'false': 1 - distribution[2], 'true': distribution[2]}
        for key in (qid, action + '_win'):
            kinds[key] = 'programmatic_conditional_distribution'
            labels[key] = 'unobserved'
    state = (f'Player cards {list(cards)}; total {total}; usable ace {str(soft).lower()}; dealer upcard {upcard}. '
        f'Remaining drawable counts for ranks [A,2,3,4,5,6,7,8,9,10-value]: {list(remaining)}. '
        'A=1 or 11; 10-value includes 10/J/Q/K. Visible cards are already excluded from these counts. '
        'Draw uniformly from remaining cards WITHOUT replacement; reduce the drawn rank count by one. '
        'Dealer has NO predealt hole card and draws only after the player stops. No peek. '
        f'Dealer {"hits" if solver.h17 else "stands on"} soft 17, hits totals below 17, and stands on hard 17 and all 18-21. '
        'Player bust loses immediately. Win/push/loss pay +1/0/-1; natural 21 has no bonus. '
        'Only hit/stand are allowed. After the first action, player uses the expected-reward-optimal policy '
        'with updated counts, standing on ties. No reshuffle within a hand.')
    semantic = (total, soft, upcard, remaining)
    sid = 'finite-bj:' + fingerprint([semantic, solver.h17])
    return {'id': sid, 'state_id': sid, 'family_id': 'blackjack_finite_composition_v2',
        'split': split, 'state': state, 'questions': questions, 'gold': {'best_action': optimal[0]},
        'gold_probs': probs, 'gold_probs_kind': kinds, 'gold_label_kind': labels,
        'metadata': {'source': 'self_authored_exact_finite_deck_dynamic_program',
            'source_group_id': 'finite-shoe:' + fingerprint(parent), 'template_id': 'finite-bj:v2',
            'environment_state': {'player_cards': list(cards), 'player_total': total, 'usable_ace': soft,
                'dealer_upcard': upcard, 'remaining_counts': list(remaining), 'dealer_hits_soft_17': solver.h17,
                'dealer_hole_card': 'none', 'parent_shoe_counts': list(parent)},
            'semantic_state_without_rule': fingerprint(semantic),
            'action_values': values, 'regret': {a: maximum - values[a] for a in values},
            'optimal_actions': optimal, 'method': 'exact_enumeration_float64_no_monte_carlo',
            'target_semantics': 'policy separate from model event probabilities; no softmax(Q)',
            'ood_definition': 'unseen_source_compositions_and_H17_rule' if split == 'ood' else None}}


def make_rows(seed=29, shoes=100, states_per_shoe=40, progress=False):
    if states_per_shoe < 2 or states_per_shoe % 2:
        raise ValueError('states_per_shoe must be positive even and >=2')
    rows, seen = [], set()
    candidates = [(cards, up) for cards in possible_hands() for up in range(1, 11)]
    for group_index, (parent, split) in enumerate(profiles(seed, shoes)):
        solver = FiniteSolver(split == 'ood')
        ordered = sorted(candidates, key=lambda x: fingerprint([seed, parent, x]))
        # Same visible 16-vs-10 anchor across source shoes makes composition effects inspectable.
        ordered = [((6, 10), 10)] + [x for x in ordered if x != ((6, 10), 10)]
        retained, by_soft = [], Counter()
        for cards, up in ordered:
            total, soft = hand_value(cards)
            if by_soft[soft] >= states_per_shoe // 2:
                continue
            remaining = parent
            try:
                for card in (*cards, up):
                    remaining = remove(remaining, card)
            except ValueError:
                continue
            key = fingerprint((total, soft, up, remaining))
            if key in seen:
                continue
            retained.append(record(cards, up, remaining, parent, split, solver))
            by_soft[soft] += 1
            seen.add(key)
            # Bound memory while keeping exact, untruncated enumeration.
            solver.clear()
            if len(retained) == states_per_shoe:
                break
        if len(retained) != states_per_shoe:
            raise ValueError('Not enough distinct valid states in this source shoe')
        rows.extend(retained)
        if progress and (group_index + 1) % 10 == 0:
            print(f'Computed {len(rows)} states from {group_index + 1} finite shoes', flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'data')
    p.add_argument('--seed', type=int, default=29)
    p.add_argument('--shoes', type=int, default=100)
    p.add_argument('--states-per-shoe', type=int, default=40)
    a = p.parse_args()
    rows = make_rows(a.seed, a.shoes, a.states_per_shoe, True)
    stats = write_splits(rows, a.output)
    dump(a.output / 'manifest.json', {'generator': 'finite-blackjack-v2', 'seed': a.seed,
        'source_shoes': a.shoes, 'states_per_shoe': a.states_per_shoe, 'unique_states': len(rows),
        'wording_variants_per_state': 1, 'split_unit': 'parent source shoe composition', 'splits': stats,
        'rules': 'finite single-deck subset, no replacement, no dealer hole card, even-money hit/stand only',
        'ood': 'Held-out source shoes plus H17; not a matched pure-rule causal comparison',
        'method': 'Exact dynamic programming in double precision; no Monte Carlo probability labels'})
    print(stats)


if __name__ == '__main__':
    main()
