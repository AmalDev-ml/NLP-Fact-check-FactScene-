"""
tests/test_new_pipeline.py — Acceptance Test Suite for Upgraded Modular FactSense
---------------------------------------------------------------------------------
Test Cases:
1. "Narendra Modi is the Prime Minister of India" -> SUPPORTED
2. "A Gazette circulating on social media, purportedly issued by the Ministry of Finance, declares every Saturday a public holiday for PSU banks"
   -> REFUTED (or INSUFFICIENT_EVIDENCE if no official source is found), NEVER SUPPORTED
3. Same text plus " is fake" -> REFUTED
4. A true statement mixed with one false statement in the same article -> MISLEADING, naming the false part
5. A gibberish or unknowable claim -> INSUFFICIENT_EVIDENCE
"""

import os
import sys
import unittest

# Ensure root dir on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detector import detect, _result_cache


class TestFactSenseNewPipeline(unittest.TestCase):

    def setUp(self):
        _result_cache.clear()

    def test_01_narendra_modi_pm(self):
        claim = "Narendra Modi is the Prime Minister of India"
        print(f"\n[TEST 1] Testing claim: '{claim}'")
        res = detect(claim)
        print(f"  Verdict: {res['verdict']} ({res['confidence']}%)")
        print(f"  Explanation: {res['explanation']}")
        self.assertEqual(res["verdict"], "SUPPORTED")

    def test_02_bank_holiday_gazette(self):
        text = "A Gazette circulating on social media, purportedly issued by the Ministry of Finance, declares every Saturday a public holiday for PSU banks"
        print(f"\n[TEST 2] Testing bank holiday viral gazette: '{text}'")
        res = detect(text)
        print(f"  Verdict: {res['verdict']} ({res['confidence']}%)")
        print(f"  Explanation: {res['explanation']}")
        # MUST NEVER be SUPPORTED!
        self.assertNotEqual(res["verdict"], "SUPPORTED", "Viral fake gazette must NEVER be SUPPORTED!")
        self.assertIn(res["verdict"], ["REFUTED", "INSUFFICIENT_EVIDENCE"])

    def test_03_bank_holiday_is_fake(self):
        text = "A Gazette circulating on social media, purportedly issued by the Ministry of Finance, declares every Saturday a public holiday for PSU banks is fake"
        print(f"\n[TEST 3] Testing bank holiday is fake: '{text}'")
        res = detect(text)
        print(f"  Verdict: {res['verdict']} ({res['confidence']}%)")
        print(f"  Explanation: {res['explanation']}")
        # Core claim of Saturday holiday is refuted!
        self.assertNotEqual(res["verdict"], "SUPPORTED", "Fake gazette must NEVER be SUPPORTED!")
        self.assertIn(res["verdict"], ["REFUTED", "INSUFFICIENT_EVIDENCE"])

    def test_04_mixed_true_false_article(self):
        article = "India gained independence in 1947. Narendra Modi was the first Prime Minister of India."
        print(f"\n[TEST 4] Testing mixed true + false article:\n'{article}'")
        res = detect(article)
        print(f"  Verdict: {res['verdict']} ({res['confidence']}%)")
        print(f"  Explanation: {res['explanation']}")
        print(f"  Failed parts: {res.get('failed_parts')}")
        self.assertEqual(res["verdict"], "MISLEADING")
        self.assertTrue(len(res.get("failed_parts", [])) > 0, "Expected failed parts to be identified")

    def test_05_gibberish_unknowable(self):
        claim = "The Plumbus flomped 893 blorps in the Andromeda quadrant."
        print(f"\n[TEST 5] Testing gibberish claim: '{claim}'")
        res = detect(claim)
        print(f"  Verdict: {res['verdict']} ({res['confidence']}%)")
        print(f"  Explanation: {res['explanation']}")
        self.assertEqual(res["verdict"], "INSUFFICIENT_EVIDENCE")


if __name__ == "__main__":
    unittest.main()

