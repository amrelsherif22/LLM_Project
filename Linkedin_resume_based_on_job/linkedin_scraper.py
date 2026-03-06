from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, parse_qs, urlencode
from dataclasses import dataclass, field
from typing import Optional
import time
import re

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

@dataclass
class LinkedInJob:
    title: str
    company: str
    location: str
    job_url: str
    posted: Optional[str] = None
    salary: Optional[str] = None
    benefits: list[str] = field(default_factory=list)
    job_id: Optional[str] = None
    description: Optional[str] = None
    full_description: Optional[str] = None 

    def __str__(self) -> str:
        lines = [
            f"{'─' * 70}",
            f"  {self.title}",
            f"  {self.company}  |  {self.location}",
        ]
        if self.salary:
            lines.append(f"  💰 {self.salary}")
        if self.benefits:
            lines.append(f"  ✔  {', '.join(self.benefits)}")
        if self.posted:
            lines.append(f"  🕒 {self.posted}")
        if self.job_url:
            lines.append(f"  🔗 {self.job_url}")
        return "\n".join(lines)

    def print_full(self) -> str:
        """Print with full description included"""
        output = str(self)
        if self.full_description:
            output += f"\n\n  📋 FULL DESCRIPTION:\n"
            output += f"  {self.full_description}\n"
        return output


def _extract_params_from_linkedin_url(url: str) -> dict:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    params = {k: v[0] for k, v in qs.items()}

    if "keywords" not in params:
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) >= 3:
            slug = path_parts[-1].replace("-", " ")
            params.setdefault("keywords", slug)

    return params


def _build_guest_api_url(params: dict, start: int = 0) -> str:
    base = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
    query = {
        "keywords": params.get("keywords", ""),
        "location": params.get("location", ""),
        "f_TPR": params.get("f_TPR", ""),
        "f_E": params.get("f_E", ""),
        "f_JT": params.get("f_JT", ""),
        "f_WT": params.get("f_WT", ""),
        "geoId": params.get("geoId", ""),
        "start": str(start),
    }
    query = {k: v for k, v in query.items() if v}
    return f"{base}?{urlencode(query)}"


def _parse_job_card(card: BeautifulSoup) -> Optional[LinkedInJob]:
    try:
        title_tag = card.select_one("h3.base-search-card__title")
        company_tag = card.select_one("h4.base-search-card__subtitle")
        location_tag = card.select_one("span.job-search-card__location")
        link_tag = card.select_one("a.base-card__full-link")
        time_tag = card.select_one("time")

        title = title_tag.get_text(strip=True) if title_tag else "N/A"
        company = company_tag.get_text(strip=True) if company_tag else "N/A"
        location = location_tag.get_text(strip=True) if location_tag else "N/A"
        job_url = link_tag["href"].split("?")[0] if link_tag else ""
        posted = time_tag.get_text(strip=True) if time_tag else None

        urn = card.get("data-entity-urn", "")
        job_id = urn.split(":")[-1] if urn else None

        salary_tag = card.select_one("span.job-search-card__salary-info")
        salary = salary_tag.get_text(strip=True) if salary_tag else None

        benefit_tags = card.select("span.result-benefits__text")
        benefits = [b.get_text(strip=True) for b in benefit_tags]

        return LinkedInJob(
            title=title,
            company=company,
            location=location,
            job_url=job_url,
            posted=posted,
            salary=salary,
            benefits=benefits,
            job_id=job_id,
        )
    except Exception:
        return None


def _clean_text(text: str) -> str:
    """Normalize whitespace + strip"""
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_full_description(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract the complete job description from the page.
    Tries multiple selectors to handle LinkedIn's various HTML structures.
    """
    try:
        desc_container = (
            soup.select_one("div.show-more-less-html__markup")
            or soup.select_one("div.description__text")
            or soup.select_one("section.description")
        )

        if desc_container:
            text = desc_container.get_text("\n", strip=True)
            return _clean_text(text)

        article = soup.select_one("article") or soup.select_one("main")
        if article:
            text = article.get_text("\n", strip=True)
            return _clean_text(text)

        for unwanted in soup.select("nav, footer, aside, script, style"):
            unwanted.decompose()

        text = soup.get_text("\n", strip=True)
        return _clean_text(text) if text else None

    except Exception as e:
        print(f"[!] Error extracting description: {e}")
        return None


def _fetch_job_description(
    job: LinkedInJob,
    session: requests.Session,
    timeout: int = 10,
) -> Optional[str]:
    """
    Fetch and extract the full job description.
    Tries:
    1) Guest jobPosting endpoint (best for structured data)
    2) Job URL directly (fallback)
    """
    try:
        html = None

        if job.job_id:
            details_url = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job.job_id}"
            try:
                r = session.get(details_url, headers=HEADERS, timeout=timeout)
                if r.status_code == 200:
                    html = r.text
            except Exception as e:
                print(f"[!] Failed to fetch from guest API for job {job.job_id}: {e}")

        if not html and job.job_url:
            try:
                r = session.get(
                    job.job_url,
                    headers=HEADERS,
                    timeout=timeout,
                    allow_redirects=True,
                )
                if r.status_code == 200:
                    html = r.text
            except Exception as e:
                print(f"[!] Failed to fetch from job URL: {e}")

        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")
        description = _extract_full_description(soup)

        return description

    except Exception as e:
        print(f"[!] Unexpected error in _fetch_job_description: {e}")
        return None


def fetch_linkedin_jobs(
    url: str,
    max_jobs: int = 25,
    delay: float = 1.0,
    include_description: bool = True,
    desc_delay: float = 0.6,
) -> list[LinkedInJob]:
    """
    Fetch LinkedIn jobs with optional descriptions.

    Args:
        url: LinkedIn jobs search URL
        max_jobs: Maximum number of jobs to fetch
        delay: Delay between API requests (seconds)
        include_description: Whether to fetch full job descriptions
        desc_delay: Delay between description fetches (seconds)

    Returns:
        List of LinkedInJob objects
    """
    params = _extract_params_from_linkedin_url(url)
    jobs: list[LinkedInJob] = []
    start = 0

    with requests.Session() as session:
        while len(jobs) < max_jobs:
            api_url = _build_guest_api_url(params, start=start)
            resp = session.get(api_url, headers=HEADERS, timeout=10)

            if resp.status_code != 200:
                print(f"[!] Request failed: HTTP {resp.status_code} at start={start}")
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            cards = soup.select("li")
            if not cards:
                break

            for card in cards:
                job = _parse_job_card(card)
                if job:
                    jobs.append(job)
                if len(jobs) >= max_jobs:
                    break

            start += len(cards)
            time.sleep(delay)

        if include_description:
            print(f"\n[*] Fetching descriptions for {len(jobs)} jobs...")
            for i, job in enumerate(jobs, start=1):
                job.full_description = _fetch_job_description(job, session=session)
                print(f"  [{i}/{len(jobs)}] {job.title} @ {job.company}")
                time.sleep(desc_delay)

    return jobs


def print_jobs(jobs: list[LinkedInJob], include_full_description: bool = False) -> None:
    """Print jobs in a formatted way"""
    if not jobs:
        print("\nNo jobs found.")
        print(f"{'─' * 70}\n")
        return
    print(f"\n{'═' * 70}")
    print(f"  {len(jobs)} jobs found")
    print(f"{'═' * 70}")
    for job in jobs:
        if include_full_description:
            print(job.print_full())
        else:
            print(job)
    print(f"{'─' * 70}\n")


def save_jobs_to_file(jobs: list[LinkedInJob], filename: str = "jobs.txt") -> None:
    """Save jobs to a text file"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{'═' * 70}\n")
        f.write(f"  {len(jobs)} jobs found\n")
        f.write(f"{'═' * 70}\n\n")
        for job in jobs:
            f.write(job.print_full())
            f.write("\n\n")
    print(f"[✓] Jobs saved to {filename}")


if __name__ == "__main__":
    LINKEDIN_URL = (
        "https://www.linkedin.com/jobs/search/?alertAction=viewjobs&currentJobId=4374547986&f_TPR=r8600&geoId=105080838&keywords=Software%20Engineer&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true&sortBy=R"
    )

    jobs = fetch_linkedin_jobs(
        LINKEDIN_URL,
        max_jobs=5,  
        include_description=True,
        delay=1.0,
        desc_delay=0.8,
    )

    print_jobs(jobs, include_full_description=True)

    save_jobs_to_file(jobs, filename="linkedin_jobs_with_descriptions.txt")
