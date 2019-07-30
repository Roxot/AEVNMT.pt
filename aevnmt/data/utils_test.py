import unittest

from .utils import remove_subword_tokens


test_input_1 = ["The qui @@ck brown fox",
                "@@ju mps over the lazy dog",
                "The qui@@ ck brown fox",
                "jumps over the lazy dog@@"]
test_output_1 = ["The quick brown fox",
                "ju mps over the lazy dog",
                "The quick brown fox",
                "jumps over the lazy dog"]
test_subword_token_1 = "@@"

test_input_1 = ["The qui @@ck brown fox",
                "@@ju mps over the lazy dog",
                "The qui@@ ck brown fox",
                "jumps over the lazy dog@@"]
test_output_1 = ["The quick brown fox",
                "ju mps over the lazy dog",
                "The quick brown fox",
                "jumps over the lazy dog"]
test_subword_token_1 = "@@"

class TestRemoveSubwordTokens(unittest.TestCase):

    def test(self):
        output_1 = remove_subword_tokens(test_input_1, test_subword_token_1)
        self.assertEqual(output_1, test_output_1)

if __name__ == "__main__":
    unittest.main()
