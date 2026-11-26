import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from gemini_service import GENERATION_UNAVAILABLE


class AppTests(TestCase):
    def setUp(self):
        jobs = Mock()
        for method in ("select", "ilike", "order", "limit"):
            getattr(jobs, method).return_value = jobs
        jobs.execute.return_value = SimpleNamespace(count=1, data=[{
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
        database.table.side_effect = {"jobs": jobs, "user_usage": usage}.__getitem__
        database.rpc.side_effect = lambda name, *args: SimpleNamespace(
            execute=lambda: SimpleNamespace(data=1 if name == "increment_site_visits" else [])
        )
        embedding_client = Mock()
        embedding_client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.0] * 768)]
        )
        self.generator = Mock()
        for patcher in (
            patch.dict(os.environ, {
                "SUPABASE_URL": "https://test.invalid",
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
        next(button for button in self.app.button if button.label == "Solar & renewables").click().run()
        self.assertFalse(self.app.exception)
        self.assertIn("temporarily unavailable", str(self.app.markdown.values))
        self.assertIn("Firmware Engineer", self.app.subheader.values)
        self.assertIn("0000001234", str(self.app.get("link_button")[0].proto.url))

        self.app.chat_input[0].set_value("Firmware roles").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.subheader.values.count("Firmware Engineer"), 2)

    def test_cv_studio_preserves_report_and_clear_removes_it(self):
        def generate(_, schema):
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
        def generate(_, schema):
            match = {
                "ref_no": "TJ-1234", "match_score": 93, "domain_category": "Embedded",
                "fit_reason": "Firmware role.", "skill_gaps": "",
            }
            return schema(matched_jobs=[{**match, "ref_no": "not-in-database"}, match, match]), None

        self.generator.generate.side_effect = generate
        self.app.chat_input[0].set_value("Firmware roles").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.subheader.values, ["Firmware Engineer"])
        self.assertIn("93% AI fit", str(self.app.markdown.values))
