"""Finite-deck arithmetic, independent simulation, and real-data leakage regressions."""
import copy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'scripts'))
from blackjack.finite_solver import FiniteSolver, add_card, hand_value, remove, value
from blackjack.generate_blackjack import profiles, record, fingerprint
from trading.build_trading import read_snapshots, construct, classify, utc, validate_config, parse_csv, FIELDS
from train_pipeline_decisions import read_training_records, validate_training_row, target_for


class FiniteTests(unittest.TestCase):
    def test_exact_two_card_shoe_by_hand(self):
        # Player 20, dealer 10; remaining A and 10. No hidden dealer card.
        counts = (1, 0, 0, 0, 0, 0, 0, 0, 0, 1)
        solver = FiniteSolver()
        self.assertEqual(solver.stand(20, 10, counts), (.5, .5, 0.))
        self.assertEqual(solver.hit(20, False, 10, counts), (.5, 0., .5))
        self.assertEqual(solver.optimal(20, False, 10, counts), (.5, 0., .5))

    def test_without_replacement(self):
        counts = (1, 0, 0, 0, 0, 0, 0, 0, 0, 1)
        branches = list(FiniteSolver.draws(counts))
        self.assertEqual([p for _, p, _ in branches], [.5, .5])
        left = remove(counts, 10)
        self.assertEqual([(card, p) for card, p, _ in FiniteSolver.draws(left)], [(1, 1.)])

    def test_ace_and_soft17(self):
        self.assertEqual(add_card(12, True, 10), (12, False))
        self.assertEqual(hand_value([1, 1, 9]), (21, True))
        counts = (0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
        self.assertEqual(FiniteSolver(False).dealer(17, True, counts), (0., 1., 0., 0., 0., 0.))
        self.assertEqual(FiniteSolver(True).dealer(17, True, counts), (0., 0., 0., 0., 1., 0.))

    def test_no_silent_reshuffle(self):
        with self.assertRaisesRegex(ValueError, 'exhausted'):
            list(FiniteSolver.draws((0,) * 10))

    def test_source_shoes_valid_and_deterministic(self):
        a, b = list(profiles()), list(profiles())
        self.assertEqual(a, b)
        self.assertEqual(len(set(c for c, _ in a)), 100)
        for counts, _ in a:
            self.assertTrue(32 <= sum(counts) <= 44)
            self.assertTrue(all(0 <= n <= (16 if i == 9 else 4) for i, n in enumerate(counts)))

    def test_changed_composition_changes_probabilities(self):
        solver = FiniteSolver()
        low_tens = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        high_tens = (1, 1, 1, 1, 1, 1, 1, 1, 1, 10)
        self.assertNotEqual(solver.hit(16, False, 10, low_tens), solver.hit(16, False, 10, high_tens))

    def test_independent_finite_deck_simulation(self):
        rng = random.Random(4321)
        parent = list(profiles())[0][0]
        for cards, up, h17 in [([6, 10], 10, False), ([1, 6], 1, True),
                                ([10, 10], 6, False), ([1, 7], 9, True)]:
            counts = parent
            for c in (*cards, up):
                counts = remove(counts, c)
            solver = FiniteSolver(h17)
            exact = solver.solve(cards, up, counts)
            for action in ('hit', 'stand'):
                observed = [0, 0, 0]
                n = 4000
                for _ in range(n):
                    deck = [rank for rank, count in enumerate(counts, 1) for _ in range(count)]
                    player = cards.copy()
                    def draw():
                        return deck.pop(rng.randrange(len(deck)))
                    def tally(hand):
                        raw = sum(hand)
                        soft = 1 in hand and raw + 10 <= 21
                        return raw + 10 if soft else raw, soft
                    if action == 'hit':
                        player.append(draw())
                        while True:
                            total, soft = tally(player)
                            if total > 21:
                                break
                            remaining = tuple(deck.count(rank) for rank in range(1, 11))
                            if value(solver.stand(total, up, remaining)) >= value(solver.hit(total, soft, up, remaining)) - 1e-12:
                                break
                            player.append(draw())
                    total, _ = tally(player)
                    if total > 21:
                        observed[0] += 1
                        continue
                    dealer = [up]
                    while True:
                        d, soft = tally(dealer)
                        if d >= 17 and not (d == 17 and soft and h17):
                            break
                        dealer.append(draw())
                    observed[2 if d > 21 or total > d else 1 if total == d else 0] += 1
                for count, probability in zip(observed, exact[action]):
                    self.assertLess(abs(count / n - probability), .045)
            solver.clear()


@unittest.skipUnless((ROOT / 'trading/raw/BTC_1min.csv.zip').is_file(),
                     'Run python data_modules/download_datasets.py --include-raw first')
class TradingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = read_snapshots(ROOT / 'trading/raw/BTC_1min.csv.zip')
        cls.config = json.loads((ROOT / 'trading/config.json').read_text())
        cls.rows, _ = construct(cls.raw, cls.config, 'test-sha')

    def test_horizons_and_observed_labels(self):
        for row in self.rows[::100]:
            self.assertEqual(set(row['questions']), {'midpoint_direction_60s', 'midpoint_direction_300s', 'midpoint_direction_900s'})
            self.assertEqual(row['gold_probs'], {})
            for qid, t in validate_training_row(row).items():
                t['candidate_ids'] = list(row['questions'][qid]['criteria'])
                self.assertEqual(sum(target_for(t, 'observed_outcome')), 1.)
                self.assertEqual(t['gold_label_kind'], 'observed_outcome')

    def test_all_horizons_purged_and_labels_replayed(self):
        for row in self.rows:
            m = row['metadata']
            block = next(b for b in self.config['splits'] if b['name'] == row['split'])
            self.assertGreaterEqual(utc(m['feature_support_start']), utc(block['start']) + 60)
            self.assertLess(utc(m['support_end']), utc(block['end']) - 60)
            now = self.raw[m['source_csv_line'] - 2]
            for qid, audit in m['label_audit'].items():
                future = self.raw[audit['label_csv_line'] - 2]
                ret = (future['midpoint'] / now['midpoint'] - 1) * 10000
                self.assertEqual(row['gold'][qid], classify(ret, 2))
                h = audit['horizon_seconds']
                self.assertTrue(h <= future['t'] - now['t'] <= h + 15)
                self.assertLessEqual(utc(audit['label_available_at']), utc(m['support_end']))

    def test_future_perturbation_preserves_past_input(self):
        row = self.rows[100]
        index = row['metadata']['source_csv_line'] - 2
        altered = [r.copy() for r in self.raw]
        for r in altered[index + 1:]:
            r['midpoint'] *= 2
        rebuilt, _ = construct(altered, self.config, 'test-sha')
        updated = next(r for r in rebuilt if r['id'] == row['id'])
        self.assertEqual(updated['state'], row['state'])
        self.assertEqual(updated['questions'], row['questions'])
        self.assertNotEqual(updated['metadata']['label_audit'], row['metadata']['label_audit'])

    def test_reject_bad_config(self):
        for horizons in ([300, 60], [0, 300], [60, 60], []):
            config = copy.deepcopy(self.config)
            config['horizons_seconds'] = horizons
            with self.assertRaises(ValueError):
                validate_config(config)
        config = copy.deepcopy(self.config)
        config['splits'][1]['start'] = config['splits'][0]['start']
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_flat_band(self):
        self.assertEqual(classify(2, 2), 'flat')
        self.assertEqual(classify(-2, 2), 'flat')
        self.assertEqual(classify(2.01, 2), 'up')

    def test_reject_duplicate_or_nan(self):
        header = 'system_time,' + ','.join(FIELDS) + '\n'
        row = '2021-04-07T00:00:00Z,' + ','.join(['100', '1'] + ['2'] * 10) + '\n'
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            parse_csv(io.StringIO(header + row + row))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            parse_csv(io.StringIO(header + row.replace(',100,', ',nan,')))


class ContractTests(unittest.TestCase):
    def test_group_leak_is_rejected_by_upstream(self):
        row = json.loads((ROOT / 'trading/example.json').read_text())
        rows = [row]
        row = rows[0]
        other = copy.deepcopy(row)
        other.update(id='other', state_id='other', split='test')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.jsonl'
            path.write_text(json.dumps(row) + '\n' + json.dumps(other) + '\n')
            with self.assertRaisesRegex(ValueError, 'crosses dataset splits'):
                read_training_records(path)


if __name__ == '__main__':
    unittest.main()
