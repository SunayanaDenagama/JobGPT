import json
from unittest import TestCase

import httpx

from enrich_job_details import fetch_excerpt
from listing_evidence import EXCERPT_LIMIT, extract_listing_evidence, public_detail_url, validate_evidence


class ListingEvidenceTests(TestCase):
    def test_json_ld_keeps_requirements_before_company_intro(self):
        posting = {
            "@type": "JobPosting",
            "experienceRequirements": {"monthsOfExperience": 60},
            "jobLocationType": "TELECOMMUTE",
            "applicantLocationRequirements": [{"@type": "Country", "name": "Sri Lanka"}],
            "description": f'<p>{"A long company introduction. " * 100}</p><p>Degree in software engineering required.</p>',
        }
        html = '<script type="application/ld+json">' + json.dumps({"@graph": [posting]}) + "</script>"
        text = extract_listing_evidence(html, "XP")
        self.assertIn("60 months", text)
        self.assertIn("TELECOMMUTE", text)
        self.assertIn("Sri Lanka", text)
        self.assertIn("Degree in software engineering required.", text)
        self.assertLessEqual(len(text), EXCERPT_LIMIT)

    def test_html_selector_preserves_requirements_and_ignores_page_chrome(self):
        html = """
        <nav>Other job: 10 years required</nav>
        <div class="show-more-less-html__markup"><ul>
          <li><b>Required:</b> 5 years of professional firmware experience.</li>
          <li>Experience with RTOS is preferred.</li>
        </ul></div>
        """
        text = extract_listing_evidence(html, "LI")
        self.assertIn("Required: 5 years of professional firmware experience.", text)
        self.assertNotIn("10 years", text)

    def test_missing_or_ambiguous_details_stay_unknown(self):
        self.assertIsNone(extract_listing_evidence('<html><img src="ad.jpg"></html>', "TJ"))
        posting = {"@type": "JobPosting", "description": "Unrelated advert with specific requirements."}
        html = '<script type="application/ld+json">' + json.dumps([posting, posting]) + "</script>"
        self.assertIsNone(extract_listing_evidence(html, "XP"))

    def test_quote_must_exist_in_indexed_text(self):
        excerpt = "Required: 3 years of\nprofessional experience."
        self.assertEqual(
            validate_evidence("3 YEARS OF professional experience", excerpt), "3 years of professional experience"
        )
        self.assertEqual(validate_evidence("No professional experience is required", excerpt), "")

    def test_fetch_never_uses_arbitrary_urls_or_follows_redirects(self):
        urls = []

        def handle(request):
            urls.append(str(request.url))
            return httpx.Response(302, headers={"Location": "https://other.example/"})

        self.assertIsNone(public_detail_url("LI-12345/example"))
        with httpx.Client(transport=httpx.MockTransport(handle), follow_redirects=False) as client:
            self.assertEqual(fetch_excerpt(client, "LI-12345"), (None, False))
        self.assertEqual(urls, ["https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/12345"])
