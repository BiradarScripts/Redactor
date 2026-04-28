import unittest

from masking_engine import SmartMasker


class SmartMaskerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.masker = SmartMasker()

    def test_masks_sensitive_people_and_repeated_mentions(self):
        text = (
            "IN THE SUPREME COURT OF INDIA\n"
            "State of Kerala vs Raneef\n"
            "JUDGMENT\n"
            "The prosecutrix Asha Devi stated that her father Ramesh Kumar was present. "
            "Justice Sharma heard learned counsel Mehta. Section 420 of IPC was discussed. "
            "Call 9876543210 or email a@example.com. Asha Devi repeated the allegation in Delhi."
        )

        masked, analysis = self.masker.mask_victims_and_family(text)

        self.assertNotIn("Asha Devi", masked)
        self.assertNotIn("Ramesh Kumar", masked)
        self.assertEqual(masked.count("[PROTECTED_PERSON_1]"), 2)
        self.assertIn("Justice Sharma", masked)
        self.assertIn("counsel Mehta", masked)
        self.assertIn("[PHONE]", masked)
        self.assertIn("[EMAIL]", masked)
        self.assertIn("[LOC]", masked)
        self.assertGreaterEqual(analysis["provision_count"], 1)
        self.assertGreaterEqual(analysis["statute_count"], 1)


if __name__ == "__main__":
    unittest.main()
