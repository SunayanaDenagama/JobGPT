"""Extract seniority constraints before sending a query to similarity search."""

import re
from dataclasses import dataclass


SENIOR_LEVEL = r"(?:senior(?:[- ]level)?|sr|lead(?![- ]generation\b)|principal|staff|management|managerial)"
SENIOR_EXCLUSION = re.compile(
    r"\b(?:exclude|avoid|without|no|never|filter\s+out|leave\s+out|"
    r"(?:don['’]?t|do\s+not|not)(?:\s+(?:include|show|want|list|return|"
    r"looking\s+for|interested\s+in)(?:\s+me)?)?)\s+"
    r"(?:any\s+|the\s+)*(?:"
    rf"{SENIOR_LEVEL}\b\.?(?:\s*(?:,|/|\band\b|\bor\b)\s*{SENIOR_LEVEL}\b\.?)*"
    r")(?:\s+(?:jobs?|roles?|positions?|openings?|listings?))?",
    re.IGNORECASE,
)
ENTRY_LEVEL = re.compile(
    r"\b(?:entry[- ]level|junior|jr\.?|graduate|trainee|intern(?:ship)?|freshers?)\b", re.IGNORECASE
)
SENIOR_TITLE = re.compile(
    r"\b(?:senior|sr|principal|lead(?![- ]generation\b)|manager|director|chief|"
    r"vice[- ]president|vp|head\s+of|team\s+leader|"
    r"staff(?:[- ]level|\s+(?:\w+\s+){0,2}(?:engineer|developer|scientist|architect|designer|analyst)))\b",
    re.IGNORECASE,
)
SENIOR_REQUEST = re.compile(rf"\b{SENIOR_LEVEL}\b", re.IGNORECASE)
FILLER = set("a an any and but i m we re just only please show give find me jobs job roles role openings positions this time instead now".split())
FOLLOW_UP = re.compile(
    r"^(?:(?:please|and)\s+)?(?:what about|how about|only (?:in|remote|hybrid|onsite|on.site)|"
    r"without|same(?: roles?)?|filter (?:to|out)|prefer|make (?:it|that)|(?:also|but|instead)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SearchIntent:
    query: str
    exclude_senior: bool
    follow_up: bool = False


def is_senior_title(title: str) -> bool:
    return bool(SENIOR_TITLE.search(title))


def resolve_search(query: str, previous: SearchIntent | None = None, *, exclude_senior: bool = False) -> SearchIntent:
    topic, exclusions = SENIOR_EXCLUSION.subn(" ", query)
    topic = " ".join(topic.strip(" ,.!?;–-").split())
    topic_words = set(re.findall(r"\w+", topic.lower()))
    seniority_only = exclusions and not (topic_words - FILLER)
    follow_up = bool(previous and (seniority_only or FOLLOW_UP.search(query.strip())))
    if seniority_only:
        topic = previous.query if previous else "Engineering and technology roles"
    elif follow_up:
        topic = f"{previous.query}; {topic}"[-800:]
    if exclude_senior or exclusions:
        topic = " ".join(SENIOR_REQUEST.sub(" ", topic).split()).strip(" ,.!?;–-")
    positive_entry = any(
        not re.search(r"\b(?:no|not|exclude|avoid|without)(?:\s+any)?\s+$", topic[:match.start()], re.IGNORECASE)
        for match in ENTRY_LEVEL.finditer(topic)
    )
    prefer_non_senior = positive_entry and not SENIOR_REQUEST.search(topic)
    inherited_exclusion = follow_up and previous.exclude_senior and not SENIOR_REQUEST.search(query)
    return SearchIntent(
        topic or "Engineering and technology roles",
        bool(exclude_senior or exclusions or prefer_non_senior or inherited_exclusion),
        follow_up,
    )
