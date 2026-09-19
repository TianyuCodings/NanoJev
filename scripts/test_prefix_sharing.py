"""Offline contract for the shared-prefix field used by forward_shared.

No torch, tokenizer download, or weight load. Verifies that prepare_examples
emits a prefix_tokens field that is the exact common prefix of every leaf path.
"""
import unittest

from test_question_contract import CharacterTokenizer, boolean_question, choice_question, payload


def encoded(request):
    from predict_toy_decisions import prepare_examples
    return prepare_examples(request, CharacterTokenizer(), max_length=100000)


class PrefixSharingContract(unittest.TestCase):
    def test_choice_prefix_is_common_prefix_of_all_leaves(self):
        ex = encoded(payload(choice_question()))[0]
        prefix = ex["prefix_tokens"]
        self.assertTrue(prefix)
        for leaf in ex["leaf_tokens"]:
            self.assertEqual(leaf[:len(prefix)], prefix)
            self.assertGreater(len(leaf), len(prefix))

    def test_choice_suffixes_differ_across_candidates(self):
        ex = encoded(payload(choice_question()))[0]
        prefix = ex["prefix_tokens"]
        suffixes = [leaf[len(prefix):] for leaf in ex["leaf_tokens"]]
        self.assertNotEqual(suffixes[0], suffixes[1])

    def test_boolean_prefix_is_leaf_prefix(self):
        ex = encoded(payload(boolean_question()))[0]
        self.assertEqual(ex["leaf_tokens"][0][:len(ex["prefix_tokens"])], ex["prefix_tokens"])


if __name__ == "__main__":
    unittest.main()
