"""Realistic job-description wrappers: the same requirements inside company boilerplate.

The labelled queries are 30-60 tokens of pure requirements. Real postings are
300-800 tokens: an "About us" opening, then the role, then benefits and an
equal-opportunity statement. Evaluating only on the clean form overstates quality,
because that boilerplate dilutes both the dense vector and the BM25 query.

Three styles, written independently of each other:

    A, B   DEV   - used to choose the JD filter's threshold
    C      TEST  - written after the filter was fixed, run once, and only on the
                   held-out queries. It is refused on the dev split on purpose:
                   once it has been used to choose anything it is a dev style and
                   the confirmation it provides is gone.

These are my own writing, not a sample of real job postings. They show that
boilerplate hurts and that a filter helps; they do not estimate by how much on
real postings.
"""

STYLES = {
    "A": {
        "split": "dev",
        "before": (
            "About us. We are a fast growing company building products that millions "
            "of people use every day. Our culture is built on ownership, curiosity and "
            "kindness, and we believe the best work happens when talented people are "
            "trusted to make decisions. We are a distributed team across several time "
            "zones and we invest heavily in learning, mentoring and career growth. Our "
            "mission is to make everyday tasks simpler. "
        ),
        "after": (
            " What we offer. Competitive salary and equity, health insurance for you "
            "and your family, generous paid leave, a home office budget, flexible "
            "working hours and regular team offsites. We are an equal opportunity "
            "employer and welcome applications from everyone. To apply, send your "
            "resume and a short note. "
        ),
    },
    "B": {
        "split": "dev",
        "before": (
            "Who we are: Founded a decade ago, the organisation serves enterprise "
            "customers worldwide and is backed by leading investors. The business has "
            "grown steadily, with teams spread over three continents, and prides itself "
            "on a collaborative, transparent way of working. Employees describe the "
            "place as supportive and ambitious. Why join: you will work alongside "
            "friendly colleagues, shape long term plans and see your impact quickly. "
        ),
        "after": (
            " Perks and process: hybrid working, wellness programme, learning budget, "
            "stock options, relocation support, refer a friend bonus. Our hiring "
            "process has three short interviews and a take home exercise. Applicants "
            "receive a response within a week. We value diversity and do not "
            "discriminate on any basis. "
        ),
    },
    "C": {
        "split": "test",
        "before": (
            "Job Summary\nJoin Northwind Systems, a market leader in logistics "
            "technology. Headquartered in Rotterdam with satellite hubs in Singapore "
            "and Austin, we move more than two million parcels a day for retailers, and "
            "our people are the reason why. We hire curious, generous colleagues, run "
            "quarterly hackathons, and sponsor open-source work. Come help us deliver "
            "what matters.\n"
        ),
        "after": (
            "\nBenefits & Perks\n- 28 days of annual leave plus public holidays\n"
            "- Private medical and dental cover\n- Pension matching up to 6 percent\n"
            "- Gym membership and commuter subsidy\n- Paid parental leave\n"
            "Northwind Systems is proud to be an Equal Opportunity Employer. All "
            "qualified applicants will receive consideration without regard to race, "
            "colour, religion, gender or disability. Apply today via our careers portal."
        ),
    },
}

STYLE_CHOICES = ("plain",) + tuple(STYLES)


def wrap(job_description: str, style: str) -> str:
    """The job description inside the given boilerplate style ('plain' = unchanged)."""
    if style == "plain":
        return job_description
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLE_CHOICES}, got {style!r}")
    parts = STYLES[style]
    return parts["before"] + job_description + parts["after"]


def check_allowed(style: str, split: str) -> None:
    """Refuse to evaluate a held-out style on anything but the held-out split."""
    if style in STYLES and STYLES[style]["split"] == "test" and split != "test":
        raise ValueError(
            f"style {style} is held out: it may only be used with --split test. "
            "Using it on dev would spend the confirmation it exists to provide."
        )
