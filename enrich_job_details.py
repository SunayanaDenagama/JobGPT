"""Fetch small excerpts from public posting pages outside the user request path."""

import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv
from supabase import ClientOptions, create_client

from listing_evidence import extract_listing_evidence, public_detail_url


MAX_HTML_BYTES = 1_500_000
SOURCES = ("LI", "XP", "TJ")
PAGE_PAUSE = 1.0


def fetch_excerpt(http, ref):
    url = public_detail_url(ref)
    if not url:
        return None, False
    with http.stream("GET", url) as response:
        if response.status_code in (401, 403, 429):
            return None, True
        if response.status_code != 200 or "html" not in response.headers.get("content-type", "").lower():
            return None, False
        body = bytearray()
        for chunk in response.iter_bytes(chunk_size=32_768):
            body.extend(chunk)
            if len(body) > MAX_HTML_BYTES:
                return None, False
        html = body.decode(response.encoding or "utf-8", errors="replace")
    return extract_listing_evidence(html, ref.split("-", 1)[0]), False


def enrich_postings(db, http, per_source):
    retry_before = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    for source in SOURCES:
        rows = (
            db.table("jobs").select("ref_no")
            .like("ref_no", f"{source}-%")
            .is_("description_excerpt", "null")
            .or_(f"details_checked_at.is.null,details_checked_at.lt.{retry_before}")
            .order("details_checked_at", nullsfirst=True)
            .order("created_at", desc=True)
            .limit(per_source)
            .execute()
        ).data or []
        saved = 0
        for row in rows:
            ref = row["ref_no"]
            try:
                excerpt, stop = fetch_excerpt(http, ref)
            except httpx.TransportError:
                excerpt, stop = None, False
            update = {"details_checked_at": datetime.now(timezone.utc).isoformat()}
            if excerpt:
                update["description_excerpt"] = excerpt
                saved += 1
            db.table("jobs").update(update).eq("ref_no", ref).is_("description_excerpt", "null").execute()
            if stop:
                print(f"[{source}] Access restricted or rate limited; stopping requests to this portal.")
                break
            time.sleep(PAGE_PAUSE)
        print(f"[{source}] Stored evidence for {saved} postings.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-source", type=int, default=16, choices=range(1, 26), metavar="1-25")
    args = parser.parse_args()
    load_dotenv()
    db = create_client(
        os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"],
        options=ClientOptions(postgrest_client_timeout=8, postgrest_client_retry=False),
    )
    with httpx.Client(
        timeout=6, follow_redirects=False,
        headers={"User-Agent": "JobGPT/1.0 (public job posting excerpts)", "Accept": "text/html"},
    ) as http:
        enrich_postings(db, http, args.per_source)


if __name__ == "__main__":
    main()
