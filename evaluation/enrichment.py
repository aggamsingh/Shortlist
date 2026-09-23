"""Realistic bulk and structural variation for the synthetic CV corpus.

Why
---
The original fixtures were 45-69 words each. Real resumes run 400-800. At that
size every chunking strategy collapses to the same thing: a 200-word window
never splits, and "sections" are four one-line fragments. Any comparison between
chunking strategies on such a corpus measures the fixture, not the strategy.

This module pads each CV to a realistic length and varies its layout, so the
chunking ablation has something to actually bite on.

The hard rule
-------------
Padding must never change who is a good match, or every relevance label silently
becomes wrong. So the material added here is deliberately NON-DISCRIMINATING:
education, generic duties, and interests that are plausible for a role family
but carry none of the signals the queries turn on (vector databases, asyncio,
owning infrastructure). It adds noise and length, not evidence.

Everything is deterministic -- selected by candidate id, never randomised -- so
evaluation runs stay reproducible.
"""

# Which role family each candidate belongs to. Only used to pick plausible
# filler; it is never used for relevance.
FAMILIES = {
    "c01": "backend", "c02": "backend", "c03": "backend", "c04": "backend",
    "c05": "frontend", "c06": "frontend", "c07": "frontend",
    "c08": "ml", "c09": "ml", "c10": "ml",
    "c11": "devops", "c12": "devops", "c13": "devops",
    "c14": "data", "c15": "data",
    "c16": "qa", "c17": "writer", "c18": "java", "c19": "ml", "c20": "frontend",
    "c21": "backend", "c22": "backend", "c23": "backend", "c24": "backend",
    "c25": "backend", "c26": "backend", "c27": "backend", "c28": "backend",
    "c29": "backend", "c30": "backend", "c31": "backend", "c32": "backend",
}

EDUCATION = [
    "Bachelor of Engineering in Computer Science, Pune Institute of Technology, 2014. "
    "Coursework in data structures, operating systems, databases and computer networks. "
    "Final year project on distributed job scheduling. Graduated with distinction.",

    "Bachelor of Technology in Information Technology, Manipal Institute of Technology, 2015. "
    "Coursework covered algorithms, software engineering, computer architecture and statistics. "
    "Served as technical secretary of the computing society for two years.",

    "Master of Computer Applications, University of Delhi, 2013. "
    "Studied advanced algorithms, database management systems and software design. "
    "Dissertation on query optimisation strategies in relational databases.",

    "Bachelor of Science in Computer Science, Christ University Bangalore, 2016. "
    "Modules in programming fundamentals, discrete mathematics, operating systems and networks. "
    "Completed a six month industry internship in the final year.",

    "Bachelor of Engineering in Electronics and Communication, Anna University, 2012. "
    "Transitioned to software engineering after graduation through self study and open source work. "
    "Completed coursework in embedded systems, signals and digital design.",
]

# Generic professional duties. Every engineer does these; none of them
# discriminate between candidates for any query in the corpus.
GENERIC_DUTIES = [
    "Participated in sprint planning, estimation and retrospectives with a cross functional team.",
    "Reviewed pull requests from peers and gave written feedback on design and readability.",
    "Wrote and maintained internal documentation covering setup, architecture and runbooks.",
    "Mentored two junior engineers through onboarding and their first production changes.",
    "Worked directly with product managers and designers to refine requirements before build.",
    "Took part in the on call rotation and handled escalations during business hours.",
    "Contributed to hiring by conducting technical interviews and reviewing take home exercises.",
    "Presented work at internal engineering demos and wrote up decisions for the team wiki.",
    "Improved test coverage on legacy modules and reduced flaky tests in the suite.",
    "Coordinated releases with QA and operations, including rollback planning.",
]

FAMILY_PROJECTS = {
    "backend": [
        "Internal tooling: built a service that exposes team metrics over a REST interface, "
        "used weekly by engineering managers for planning.",
        "Migration work: helped move a legacy module to the current service template, "
        "including data backfill and a phased cutover.",
    ],
    "frontend": [
        "Design system: contributed reusable components and documentation used across three product surfaces.",
        "Accessibility: audited key user journeys and fixed contrast, focus order and screen reader labels.",
    ],
    "ml": [
        "Experiment tracking: set up a shared workflow so model runs and datasets are reproducible.",
        "Analysis: produced a written study of model errors by segment and presented it to stakeholders.",
    ],
    "devops": [
        "Cost review: analysed cloud spend by service and proposed a reduction plan to leadership.",
        "Documentation: wrote runbooks for the ten most common production incidents.",
    ],
    "data": [
        "Data quality: added validation checks to detect schema drift before it reached the warehouse.",
        "Enablement: ran internal sessions teaching analysts to self serve common queries.",
    ],
    "qa": [
        "Process: introduced a triage rota so incoming defects are classified within a working day.",
        "Reporting: built a dashboard showing regression pass rates across releases.",
    ],
    "writer": [
        "Style: consolidated three conflicting documentation styles into one maintained guide.",
        "Tooling: automated the docs build so publishing no longer requires manual steps.",
    ],
    "java": [
        "Refactoring: reduced duplication across service modules and standardised error handling.",
        "Build: cut CI build times by reorganising module dependencies.",
    ],
}

CERTIFICATIONS = [
    "Certifications: AWS Certified Cloud Practitioner (2021), Scrum Alliance Certified Scrum Developer (2019). "
    "Completed internal training on secure coding practices and data protection compliance.",

    "Certifications: Oracle Certified Professional Java SE (2018), Microsoft Certified Azure Fundamentals (2022). "
    "Attended an internal leadership development programme for senior individual contributors.",

    "Certifications: Certified Kubernetes Application Developer (2022). "
    "Regularly attends regional technology meetups and has spoken twice on engineering practice.",

    "Professional development: completed a part time course in distributed systems design in 2021, "
    "and an internal programme on technical communication and stakeholder management.",
]

ACHIEVEMENTS = [
    "Recognised with a quarterly engineering award for sustained contribution to platform reliability. "
    "Nominated by peers for consistently thorough code review and clear written design proposals.",

    "Received an internal award for improving onboarding: rewrote the setup guide and reduced the time "
    "for a new engineer to ship their first change from two weeks to four days.",

    "Led a working group on engineering standards whose recommendations were adopted across three teams. "
    "Authored the resulting guidelines and ran follow up sessions to embed them.",

    "Twice selected to represent the team in cross organisation architecture reviews. "
    "Contributed to the shared service catalogue and its review process.",
]

INTERESTS = [
    "Interests include competitive chess, long distance running and amateur photography.",
    "Outside work, enjoys trekking, reading popular science and playing the guitar.",
    "Interests: cycling, cooking and volunteering at a local coding club for students.",
    "Enjoys badminton, travel writing and contributing to open source documentation.",
]

LANGUAGES = [
    "Languages: English (fluent), Hindi (native).",
    "Languages: English (professional), Hindi (native), Marathi (conversational).",
    "Languages: English (fluent), Tamil (native), Hindi (conversational).",
    "Languages: English (professional), Bengali (native).",
]


def _pick(pool, candidate_id, offset=0):
    """Deterministic selection: same candidate always gets the same filler."""
    index = int(candidate_id.lstrip("c")) + offset
    return pool[index % len(pool)]


def _duties(candidate_id, count=8):
    start = int(candidate_id.lstrip("c"))
    return [GENERIC_DUTIES[(start + i) % len(GENERIC_DUTIES)] for i in range(count)]


def layout_for(candidate_id: str) -> str:
    """Assign a document layout.

    Real corpora are not uniform: some CVs have clean headings, some reorder
    them, and some have none at all. A corpus where every document has the same
    four headings flatters section chunking and hides its failure mode.
    """
    index = int(candidate_id.lstrip("c"))
    if index % 7 == 0:
        return "headingless"      # no recognisable section headings at all
    if index % 3 == 0:
        return "reordered"        # skills and education before experience
    return "standard"


def enrich(candidate: dict) -> dict:
    """Return the extra, non-discriminating sections for a candidate."""
    cid = candidate["id"]
    family = FAMILIES.get(cid, "backend")
    return {
        "education": _pick(EDUCATION, cid),
        "certifications": _pick(CERTIFICATIONS, cid),
        "achievements": _pick(ACHIEVEMENTS, cid, offset=2),
        "duties": _duties(cid),
        "projects": FAMILY_PROJECTS.get(family, FAMILY_PROJECTS["backend"]),
        "interests": _pick(INTERESTS, cid),
        "languages": _pick(LANGUAGES, cid, offset=1),
        "layout": layout_for(cid),
    }
