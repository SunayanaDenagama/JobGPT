from unittest import TestCase

from search_intent import is_senior_title, resolve_search


class SearchIntentTests(TestCase):
    def test_exclusion_phrases_remove_senior_terms_from_retrieval(self):
        for phrase in ("dont include senior roles", "do not show any senior jobs", "not senior",
                       "filter out senior or lead roles", "no sr. positions", "without senior-level jobs"):
            with self.subTest(phrase=phrase):
                intent = resolve_search(f"Firmware in Colombo, {phrase}")
                self.assertTrue(intent.exclude_senior)
                self.assertEqual(intent.query, "Firmware in Colombo")

    def test_entry_level_and_explicit_seniority(self):
        self.assertTrue(resolve_search("Junior firmware jobs").exclude_senior)
        self.assertFalse(resolve_search("Junior or senior firmware jobs").exclude_senior)
        self.assertFalse(resolve_search("Senior firmware jobs").exclude_senior)
        self.assertFalse(resolve_search("Firmware jobs, no junior roles").exclude_senior)

    def test_follow_up_exclusion_overrides_previous_seniority(self):
        previous = resolve_search("Senior firmware jobs in Colombo")
        intent = resolve_search("I don’t want any senior roles", previous)
        self.assertTrue(intent.exclude_senior)
        self.assertEqual(intent.query, "firmware jobs in Colombo")

    def test_senior_title_markers(self):
        for title in ("Senior Firmware Engineer", "Sr. Developer", "Principal Engineer", "Firmware Team Lead",
                      "Staff Software Engineer", "Engineering Manager", "Director of Engineering", "Head of Platform"):
            self.assertTrue(is_senior_title(title), title)
        for title in ("Junior Engineer", "Embedded Engineer", "Lead Generation Intern", "Office Staff"):
            self.assertFalse(is_senior_title(title), title)

    def test_remote_follow_up_keeps_the_original_role(self):
        previous = resolve_search("Junior firmware jobs in Colombo")
        intent = resolve_search("Only remote", previous)
        self.assertEqual(intent.query, "Junior firmware jobs in Colombo; Only remote")
        self.assertTrue(intent.follow_up)
        self.assertTrue(intent.exclude_senior)

    def test_new_topic_does_not_inherit_previous_constraints(self):
        previous = resolve_search("Junior firmware jobs in Colombo")
        intent = resolve_search("Data analyst roles", previous)
        self.assertEqual(intent.query, "Data analyst roles")
        self.assertFalse(intent.exclude_senior)
