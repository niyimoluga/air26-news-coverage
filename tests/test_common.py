"""Smoke tests for text normalisation and shingling."""

import unittest

from src.data.common import normalise, word_shingles


class TestCommon(unittest.TestCase):
    def test_normalise(self):
        self.assertEqual(normalise("Hello,   WORLD!"), "hello world")
        self.assertEqual(normalise(None), "")

    def test_shingles(self):
        self.assertEqual(
            word_shingles("one two three four five six", k=5),
            ["one two three four five", "two three four five six"],
        )
        self.assertEqual(word_shingles("one two", k=5), ["one two"])
        self.assertEqual(word_shingles("", k=5), [])


if __name__ == "__main__":
    unittest.main()
