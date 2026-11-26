import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import httpx
from streamlit.testing.v1 import AppTest

from gemini_service import GENERATION_UNAVAILABLE


class AppTests(TestCase):
    def setUp(self):
        jobs = Mock()
        for method in ("select", "or_", "order", "limit"):
            getattr(jobs, method).return_value = jobs
        jobs.execute.return_value = SimpleNamespace(count=2113, data=[{
            "ref_no": "TJ-1234",
            "title": "Firmware Engineer",
            "company": "Tech Company",
            "location": "Colombo",
            "source": "TopJobs",
        }])
        usage = Mock()
        for method in ("select", "eq", "update"):
            getattr(usage, method).return_value = usage
        usage.execute.return_value = SimpleNamespace(data=[{"chat_count": 0, "cv_count": 0}])
        database = Mock()
        database.postgrest = database
        database.table.side_effect = {"jobs": jobs, "user_usage": usage}.__getitem__
        database.rpc.side_effect = lambda name, *args: SimpleNamespace(
            execute=lambda: SimpleNamespace(data=1 if name == "increment_site_visits" else [])
        )
        embedding_client = Mock()
        embedding_client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.0] * 768)]
        )
        self.generator = Mock()
        self.jobs = jobs
        self.database = database
        self.embedding_client = embedding_client
        for patcher in (
            patch.dict(os.environ, {
                "SUPABASE_URL": f"https://test.invalid/{self.id()}",
                "SUPABASE_KEY": "test-key",
                "GEMINI_API_KEY": self.id(),
            }),
            patch("supabase.create_client", return_value=database),
            patch("gemini_service.make_client", return_value=embedding_client),
            patch("gemini_service.GeminiGenerator.from_key", return_value=self.generator),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15).run()
        self.assertFalse(self.app.exception)

    def test_search_shows_retrieved_vacancies_when_generation_fails(self):
        self.generator.generate.return_value = (None, GENERATION_UNAVAILABLE)
        self.app.toggle[0].set_value(True).run()
        next(button for button in self.app.button if button.label == "Embedded & firmware").click().run()
        self.assertFalse(self.app.exception)
        self.assertIn("temporarily unavailable", str(self.app.markdown.values))
        self.assertIn("Firmware Engineer", self.app.subheader.values)
        self.assertIn("0000001234", str(self.app.get("link_button")[0].proto.url))

        self.app.chat_input[0].set_value("Firmware roles").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.subheader.values.count("Firmware Engineer"), 3)  # includes browse preview

    def test_active_count_and_preview_load_without_ai(self):
        self.assertEqual(self.app.metric[0].label, "Active job listings")
        self.assertEqual(self.app.metric[0].value, "2,113")
        self.assertIn("Firmware Engineer", self.app.subheader.values)
        self.generator.generate.assert_not_called()
        self.embedding_client.models.embed_content.assert_not_called()

    def test_fast_search_uses_one_keyword_request_and_no_generation(self):
        self.app.chat_input[0].set_value("Firmware jobs in Colombo").run()
        self.assertFalse(self.app.exception)
        self.generator.generate.assert_not_called()
        self.jobs.or_.assert_called_once()
        self.assertIn("location.ilike.*colombo*", self.jobs.or_.call_args.args[0])
        self.embedding_client.models.embed_content.assert_called_once()
        self.assertIn("Ranked by hybrid search", str(self.app.markdown.values))

    def test_fast_search_can_show_keyword_matches_during_embedding_timeout(self):
        self.embedding_client.models.embed_content.side_effect = httpx.ReadTimeout("slow embedder")
        self.app.chat_input[0].set_value("Firmware").run()
        self.assertFalse(self.app.exception)
        self.assertIn("Ranked by hybrid search", str(self.app.markdown.values))
        self.assertEqual(self.app.subheader.values.count("Firmware Engineer"), 2)  # preview and one result

    def test_fast_search_includes_semantic_matches(self):
        self.database.rpc.side_effect = None
        self.database.rpc.return_value.execute.return_value = SimpleNamespace(data=[
            {"ref_no": "LI-5678", "title": "Embedded Software Developer", "similarity": 0.9},
        ])
        self.app.chat_input[0].set_value("Firmware").run()
        self.assertFalse(self.app.exception)
        self.assertIn("Embedded Software Developer", self.app.subheader.values)
        self.assertIn("**2 vacancies**", str(self.app.markdown.values))

    def test_cv_studio_preserves_report_and_clear_removes_it(self):
        def generate(_, schema, **options):
            return schema(
                overall_score=80,
                summary_critique="Put measurable outcomes first.",
                crucial_missing_keywords=["RTOS"],
                suggested_bullet_revisions=[],
                high_impact_skills_to_learn=["Embedded testing"],
            ), None

        self.generator.generate.side_effect = generate
        self.app.sidebar.radio[0].set_value("CV Studio").run()
        self.assertFalse(self.app.exception)
        next(button for button in self.app.button if button.label == "Sample CV").click().run()
        self.app.text_input(key="cv_target_role").set_value("Firmware Engineer")
        next(button for button in self.app.button if button.label == "Review my CV").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.metric[0].value, "80 / 100")
        self.app.sidebar.radio[0].set_value("Job Search").run()
        self.app.sidebar.radio[0].set_value("CV Studio").run()
        self.assertEqual(self.app.metric[0].value, "80 / 100")
        next(button for button in self.app.button if button.label == "Clear CV").click().run()
        self.assertFalse(self.app.metric)

    def test_ranked_search_uses_database_listings_and_deduplicates_model_results(self):
        def generate(_, schema, **options):
            match = {
                "ref_no": "TJ-1234", "match_score": 93, "domain_category": "Embedded",
                "fit_reason": "Firmware role.", "skill_gaps": "",
            }
            return schema(matched_jobs=[{**match, "ref_no": "not-in-database"}, match, match]), None

        self.generator.generate.side_effect = generate
        self.app.toggle[0].set_value(True).run()
        self.app.chat_input[0].set_value("Firmware roles").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.subheader.values, ["Firmware Engineer", "Firmware Engineer"])  # preview and one result
        self.assertIn("93% AI fit", str(self.app.markdown.values))
