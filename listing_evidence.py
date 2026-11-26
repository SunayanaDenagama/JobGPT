"""Extract and validate short excerpts from public job postings."""

import json
import re

from bs4 import BeautifulSoup


EXCERPT_LIMIT = 1_200
REQUIREMENT_LINE = re.compile(
    r"\b(?:years?|experience|qualifications?|requirements?|degree|graduates?|"
    r"entry.level|preferred|required|remote|hybrid|on.site|internship|visa|salary|work setup|applicant location)\b",
    re.IGNORECASE,
)
STRUCTURED_FIELDS = (
    "experienceRequirements", "qualifications", "skills", "jobLocationType", "applicantLocationRequirements",
    "employmentType", "description", "responsibilities",
)
DESCRIPTION_SELECTORS = {
    "LI": ".show-more-less-html__markup, .description__text",
    "XP": "#job-description, .job-description, .job-description-content",
    "TJ": "#job-description, .job-description",
}


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def public_detail_url(ref: str) -> str | None:
    match = re.fullmatch(r"(LI|XP|TJ)-([0-9]{5,20})", ref)
    if not match:
        return None
    source, job_id = match.groups()
    if source == "LI":
        return f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
    if source == "XP":
        return f"https://xpress.jobs/Jobs/View/{job_id}"
    return (
        "https://www.topjobs.lk/employer/JobAdvertismentServlet"
        f"?rid=1&ac=DEFZZZ&jc={job_id.zfill(10)}&ec=DEFZZZ&pg=applicant/vacancybyfunctionalarea.jsp"
    )


def job_postings(value, depth=0):
    if depth > 8:
        return
    if isinstance(value, list):
        for item in value:
            yield from job_postings(item, depth + 1)
    elif isinstance(value, dict):
        kinds = value.get("@type") or []
        if isinstance(kinds, str):
            kinds = [kinds]
        if isinstance(kinds, list) and "JobPosting" in kinds:
            yield value
        if "@graph" in value:
            yield from job_postings(value["@graph"], depth + 1)


def extract_lines(html: str) -> list[str]:
    fragment = BeautifulSoup(html, "html.parser")
    for node in fragment.select("script, style, noscript, button"):
        node.decompose()
    # Keep inline children such as a bold "required" in the sentence they belong to.
    for node in fragment.find_all(["li", "p", "br", "h2", "h3", "h4"]):
        node.append("\n")
    return [line for raw in fragment.get_text().splitlines() if (line := normalize_text(raw))]


def make_excerpt(lines: list[str]) -> str | None:
    passages = list(dict.fromkeys(
        sentence for line in lines
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line)
        if len(sentence) >= 8
    ))
    if sum(map(len, passages)) < 30:
        return None
    # Keep requirement-bearing passages near the top instead of clipping long company intros.
    passages.sort(key=lambda line: not bool(REQUIREMENT_LINE.search(line)))
    selected = []
    remaining = EXCERPT_LIMIT
    for line in passages:
        if len(line) <= remaining:
            selected.append(line)
            remaining -= len(line) + 1
        elif not selected:
            # Some portals flatten an entire posting into one paragraph.
            selected.append(line[:remaining].rsplit(" ", 1)[0] or line[:remaining])
            break
    return "\n".join(selected) or None


def extract_listing_evidence(html: str, source: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    postings = []
    for node in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(node.string or "")
        except (ValueError, RecursionError):
            continue
        postings.extend(job_postings(data))
    # Multiple structured adverts can be related jobs; don't borrow their requirements.
    if len(postings) == 1:
        lines = []
        for field in STRUCTURED_FIELDS:
            value = postings[0].get(field)
            if field == "jobLocationType" and isinstance(value, str):
                lines.extend(extract_lines(f"Work setup in structured posting data: {value}."))
            elif field == "applicantLocationRequirements":
                locations = value if isinstance(value, list) else [value]
                for location in locations:
                    if isinstance(location, dict) and isinstance(location.get("name"), str):
                        lines.extend(extract_lines(f"Applicant location in structured posting data: {location['name']}."))
            elif isinstance(value, str):
                lines.extend(extract_lines(value))
            elif field == "experienceRequirements" and isinstance(value, dict):
                months = value.get("monthsOfExperience")
                if type(months) is int and 0 <= months <= 600:
                    lines.append(f"Experience requirement in structured posting data: {months} months.")
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        lines.extend(extract_lines(item))
        if excerpt := make_excerpt(lines):
            return excerpt
    selectors = DESCRIPTION_SELECTORS.get(source)
    if selectors and (body := soup.select_one(selectors)):
        return make_excerpt(extract_lines(str(body)))
    return None


def validate_evidence(quote: str, excerpt: str) -> str:
    """Return a verbatim span from the indexed excerpt, not model-generated text."""
    quote = normalize_text(quote)
    if not 12 <= len(quote) <= 220:
        return ""
    match = re.search(re.escape(quote), normalize_text(excerpt), re.IGNORECASE)
    return match.group() if match else ""
