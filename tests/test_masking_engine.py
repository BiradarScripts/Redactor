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

    def test_masks_indian_judgment_relationship_patterns(self):
        text = (
            "The minor girl Lakshmi, aged 14 years, daughter of Ramesh Kumar and Sita Devi, "
            "resident of No. 14, Anna Nagar, Hosur, stated the facts. "
            "The complainant P. Kumar S/o Ramesh Naidu gave PAN ABCDE1234F and Aadhaar 1234 5678 9012. "
            "Later Lakshmi Devi repeated the allegation."
        )

        masked, analysis = self.masker.mask_victims_and_family(text)

        for value in ("Lakshmi", "Ramesh Kumar", "Sita Devi", "P. Kumar", "Ramesh Naidu"):
            self.assertNotIn(value, masked)
        self.assertNotIn("ABCDE1234F", masked)
        self.assertNotIn("1234 5678 9012", masked)
        self.assertNotIn("Lakshmi Devi", masked)
        self.assertGreaterEqual(masked.count("[ID]"), 2)
        self.assertGreaterEqual(analysis["protected_person_count"], 5)

    def test_masks_witness_names_without_masking_witness_labels(self):
        text = (
            "P.W.7 Subba Rao, an elder of the village, presented a memorandum. "
            "P.W.7 went to Bangalore Hospital. Justice Sharma recorded the evidence."
        )

        masked, _ = self.masker.mask_victims_and_family(text)

        self.assertIn("P.W.7 ", masked)
        self.assertNotIn("Subba Rao", masked)
        self.assertNotIn("[PROTECTED_PERSON_1].7", masked)
        self.assertIn("Justice Sharma", masked)

    def test_repeated_longer_alias_uses_same_token(self):
        text = "The prosecutrix Lakshmi stated the facts. Later Lakshmi Devi repeated the allegation."

        masked, _ = self.masker.mask_victims_and_family(text)

        self.assertNotIn("Lakshmi", masked)
        self.assertNotIn("Lakshmi Devi", masked)
        self.assertEqual(masked.count("[PROTECTED_PERSON_1]"), 2)

    def test_expands_selected_first_name_to_full_local_name(self):
        text = (
            "JUDGMENT\n"
            "The complainant Vijay stated the facts. "
            "Vijay Pralhad Warbuvan repeated the complaint in court."
        )

        masked, _ = self.masker.mask_victims_and_family(text)

        self.assertNotIn("Vijay Pralhad Warbuvan", masked)
        self.assertNotIn("Pralhad Warbuvan", masked)


if __name__ == "__main__":
    unittest.main()
