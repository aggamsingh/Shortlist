"""Synthetic CV corpus and labelled job descriptions for retrieval evaluation.

Why synthetic: real resumes cannot be committed to a public repo, and a shared
benchmark has to be reproducible. The trade-off is stated plainly in the README --
these numbers measure the pipeline on controlled data, not on messy real CVs.

The corpus is deliberately adversarial. Roughly a quarter of it exists purely as
distractors: people whose resumes are dense with a query's keywords but who are
wrong for the role (a QA engineer who automates in Python, a technical writer who
documents FastAPI, a Java backend engineer whose CV says "backend microservices
Docker"). A system that keyword-matches scores well without them and badly with
them, which is the entire point of keeping them.

Relevance grades are author-assigned, applied to the role rather than the
keywords:
    2 = would shortlist
    1 = plausible, weak on a stated requirement
    0 = not a fit
"""

CANDIDATES = [
    # ---------------- Python backend ----------------
    {
        "id": "c01",
        "name": "Ananya Rao",
        "location": "Bangalore",
        "years": 7,
        "title": "Senior Backend Engineer",
        "summary": "Senior backend engineer with 7 years of experience designing Python microservices at scale.",
        "experience": [
            "Led a team of five building FastAPI services handling 12k requests per second.",
            "Designed a semantic search feature backed by Qdrant, embedding 2M documents with sentence-transformers.",
            "Migrated a monolith to containerised services orchestrated with Docker Compose and Kubernetes.",
        ],
        "skills": "Python, FastAPI, Qdrant, Docker, Kubernetes, PostgreSQL, Redis, pytest",
    },
    {
        "id": "c02",
        "name": "Vikram Nair",
        "location": "Pune",
        "years": 5,
        "title": "Backend Engineer",
        "summary": "Backend engineer with 5 years of experience building Django web applications.",
        "experience": [
            "Built and maintained Django REST Framework APIs for a logistics platform.",
            "Optimised PostgreSQL queries, cutting p95 latency from 800ms to 120ms.",
            "Containerised deployments with Docker and set up CI in GitLab.",
        ],
        "skills": "Python, Django, Django REST Framework, PostgreSQL, Docker, Celery",
    },
    {
        "id": "c03",
        "name": "Rohit Sharma",
        "location": "Delhi",
        "years": 4,
        "title": "Backend Engineer",
        "summary": "Backend developer with 4 years of experience in Python microservices and event-driven systems.",
        "experience": [
            "Developed FastAPI microservices for a fintech payments platform.",
            "Implemented vector similarity search over product catalogues using embeddings.",
            "Ran services on Kubernetes with Redis caching and Docker-based local development.",
        ],
        "skills": "Python, FastAPI, Kubernetes, Docker, Redis, vector search, asyncio",
    },
    {
        "id": "c04",
        "name": "Meera Iyer",
        "location": "Hyderabad",
        "years": 9,
        "title": "Platform Engineer",
        "summary": "Platform engineer with 9 years of experience across Python and Go backend systems.",
        "experience": [
            "Built internal gRPC microservices in Go and Python for a payments platform.",
            "Owned the service mesh and internal developer platform used by 40 engineers.",
            "Introduced contract testing across 20 services.",
        ],
        "skills": "Go, Python, gRPC, Kubernetes, Docker, Terraform",
    },
    # ---------------- Frontend ----------------
    {
        "id": "c05",
        "name": "Aditya Verma",
        "location": "Mumbai",
        "years": 4,
        "title": "Frontend Engineer",
        "summary": "Frontend engineer with 4 years of experience building React and TypeScript applications.",
        "experience": [
            "Built a Next.js dashboard used daily by 8000 operations staff.",
            "Introduced a typed component library, cutting UI defects by a third.",
            "Improved Lighthouse performance scores from 54 to 92.",
        ],
        "skills": "React, TypeScript, Next.js, Redux, CSS, Jest, Webpack",
    },
    {
        "id": "c06",
        "name": "Sneha Kapoor",
        "location": "Delhi",
        "years": 3,
        "title": "Frontend Developer",
        "summary": "Frontend developer with 3 years of experience in React single page applications.",
        "experience": [
            "Developed customer-facing React interfaces with Redux state management.",
            "Implemented responsive layouts and accessibility fixes to WCAG AA.",
            "Wrote component tests with React Testing Library.",
        ],
        "skills": "React, JavaScript, TypeScript, Redux, HTML, CSS, Figma",
    },
    {
        "id": "c07",
        "name": "Karan Malhotra",
        "location": "Bangalore",
        "years": 6,
        "title": "Full Stack Engineer",
        "summary": "Full stack engineer with 6 years of experience across React frontends and Node.js services.",
        "experience": [
            "Delivered React and TypeScript frontends for an e-commerce platform.",
            "Built Node.js and Express backend APIs with MongoDB.",
            "Set up end-to-end testing with Playwright.",
        ],
        "skills": "React, TypeScript, Node.js, Express, MongoDB, Playwright",
    },
    # ---------------- ML / NLP ----------------
    {
        "id": "c08",
        "name": "Priya Menon",
        "location": "Bangalore",
        "years": 6,
        "title": "Machine Learning Engineer",
        "summary": "Machine learning engineer with 6 years of experience shipping NLP models to production.",
        "experience": [
            "Fine-tuned transformer models for document classification, improving F1 from 0.71 to 0.88.",
            "Built training pipelines in PyTorch with distributed multi-GPU training.",
            "Deployed inference services and monitored drift in production.",
        ],
        "skills": "Python, PyTorch, transformers, HuggingFace, NLP, MLflow, scikit-learn",
    },
    {
        "id": "c09",
        "name": "Arjun Das",
        "location": "Chennai",
        "years": 4,
        "title": "Data Scientist",
        "summary": "Data scientist with 4 years of experience in statistical modelling and forecasting.",
        "experience": [
            "Built demand forecasting models with scikit-learn and XGBoost.",
            "Ran A/B tests and reported causal impact to product teams.",
            "Prototyped a text classifier using classical NLP features.",
        ],
        "skills": "Python, pandas, scikit-learn, XGBoost, SQL, statistics, matplotlib",
    },
    {
        "id": "c10",
        "name": "Fatima Sheikh",
        "location": "Hyderabad",
        "years": 5,
        "title": "NLP Engineer",
        "summary": "NLP engineer with 5 years of experience in semantic search and language models.",
        "experience": [
            "Built a semantic retrieval system using sentence embeddings and approximate nearest neighbour search.",
            "Fine-tuned BERT models for named entity recognition on legal text.",
            "Evaluated retrieval quality with nDCG and recall benchmarks.",
        ],
        "skills": "Python, HuggingFace, BERT, embeddings, semantic search, PyTorch, FAISS",
    },
    # ---------------- DevOps ----------------
    {
        "id": "c11",
        "name": "Sandeep Reddy",
        "location": "Hyderabad",
        "years": 8,
        "title": "DevOps Engineer",
        "summary": "DevOps engineer with 8 years of experience automating cloud infrastructure.",
        "experience": [
            "Managed production Kubernetes clusters across three AWS regions.",
            "Wrote Terraform modules covering the whole production estate.",
            "Cut deployment time from 40 minutes to 6 with a rebuilt CI/CD pipeline.",
        ],
        "skills": "Kubernetes, AWS, Terraform, Docker, Jenkins, Ansible, Linux",
    },
    {
        "id": "c12",
        "name": "Nikhil Joshi",
        "location": "Pune",
        "years": 5,
        "title": "Site Reliability Engineer",
        "summary": "SRE with 5 years of experience running containerised platforms.",
        "experience": [
            "Operated Kubernetes workloads and defined SLOs with error budgets.",
            "Built observability with Prometheus, Grafana and distributed tracing.",
            "Led incident response and wrote blameless postmortems.",
        ],
        "skills": "Kubernetes, Docker, Prometheus, Grafana, AWS, Python, Go",
    },
    {
        "id": "c13",
        "name": "Deepak Kumar",
        "location": "Delhi",
        "years": 11,
        "title": "Cloud Architect",
        "summary": "Cloud architect with 11 years of experience designing AWS landing zones.",
        "experience": [
            "Designed multi-account AWS architecture and governance for a bank.",
            "Led a datacentre to cloud migration of 200 workloads.",
            "Owned cost optimisation, saving 1.2 million dollars annually.",
        ],
        "skills": "AWS, cloud architecture, networking, security, CloudFormation",
    },
    # ---------------- Data engineering ----------------
    {
        "id": "c14",
        "name": "Riya Chatterjee",
        "location": "Kolkata",
        "years": 6,
        "title": "Data Engineer",
        "summary": "Data engineer with 6 years of experience building large scale batch and streaming pipelines.",
        "experience": [
            "Built Spark pipelines processing 4TB per day.",
            "Orchestrated workflows with Airflow across 300 daily jobs.",
            "Designed Kafka streaming ingestion with exactly-once semantics.",
        ],
        "skills": "Python, Spark, Airflow, Kafka, SQL, Snowflake, dbt",
    },
    {
        "id": "c15",
        "name": "Manish Gupta",
        "location": "Noida",
        "years": 7,
        "title": "ETL Developer",
        "summary": "ETL developer with 7 years of experience in enterprise data warehousing.",
        "experience": [
            "Developed Informatica mappings for a retail data warehouse.",
            "Wrote complex SQL transformations and stored procedures.",
            "Scheduled nightly batch loads and handled reconciliation.",
        ],
        "skills": "SQL, Informatica, Oracle, data warehousing, Unix shell",
    },
    # ---------------- Deliberate distractors ----------------
    {
        # Python-dense CV, but the role is QA, not backend engineering.
        "id": "c16",
        "name": "Tanvi Bhatt",
        "location": "Pune",
        "years": 5,
        "title": "QA Automation Engineer",
        "summary": "QA automation engineer with 5 years of experience writing Python test suites.",
        "experience": [
            "Built Selenium and pytest automation frameworks in Python.",
            "Automated regression suites covering 1200 test cases.",
            "Integrated automated tests into the Docker-based CI pipeline.",
        ],
        "skills": "Python, pytest, Selenium, Docker, CI/CD, test automation, API testing",
    },
    {
        # Mentions Python, FastAPI, REST APIs throughout -- as documentation subjects.
        "id": "c17",
        "name": "Harsh Agarwal",
        "location": "Bangalore",
        "years": 6,
        "title": "Senior Technical Writer",
        "summary": "Technical writer with 6 years of experience documenting developer platforms and REST APIs.",
        "experience": [
            "Wrote and maintained API reference documentation for Python and FastAPI services.",
            "Produced onboarding guides for a microservices platform running on Docker.",
            "Owned the docs toolchain and style guide across 12 engineering teams.",
        ],
        "skills": "technical writing, OpenAPI, Markdown, documentation, REST APIs, git",
    },
    {
        # "Backend microservices Docker" -- but Java, where the JD demands Python.
        "id": "c18",
        "name": "Ishaan Bose",
        "location": "Mumbai",
        "years": 7,
        "title": "Backend Engineer",
        "summary": "Backend engineer with 7 years of experience building Java microservices.",
        "experience": [
            "Developed Spring Boot microservices for an insurance platform.",
            "Designed event-driven integration with Kafka.",
            "Deployed containerised services with Docker and Kubernetes.",
        ],
        "skills": "Java, Spring Boot, Kafka, Docker, Kubernetes, Maven, JUnit",
    },
    {
        # Genuinely uses Python and FastAPI, but as a data scientist shipping models.
        "id": "c19",
        "name": "Neha Sinha",
        "location": "Bangalore",
        "years": 5,
        "title": "Data Scientist",
        "summary": "Data scientist with 5 years of experience who ships her own models to production.",
        "experience": [
            "Built recommendation models and served them behind FastAPI endpoints.",
            "Packaged inference services with Docker for deployment.",
            "Worked with engineers to productionise batch scoring pipelines.",
        ],
        "skills": "Python, FastAPI, scikit-learn, pandas, Docker, SQL",
    },
    {
        # Python appears once, for build scripts. Primary role is frontend.
        "id": "c20",
        "name": "Aryan Pillai",
        "location": "Chennai",
        "years": 3,
        "title": "Frontend Developer",
        "summary": "Frontend developer with 3 years of experience building React interfaces.",
        "experience": [
            "Built React components and integrated REST APIs.",
            "Maintained build tooling, including some Python scripts for asset generation.",
            "Improved bundle size by 40 percent through code splitting.",
        ],
        "skills": "React, JavaScript, CSS, Webpack, HTML",
    },
    # ------------------------------------------------------------------
    # Dense Python-backend cluster.
    #
    # Separating a React CV from a Kubernetes CV is trivial for any embedding
    # model, so a corpus of five distinct roles saturates every metric at ~1.0
    # and cannot tell a good pipeline from a bad one. Real screening is hard
    # *within* a role: a hundred Python backend engineers, and the job turns on
    # one requirement most of them lack. These twelve are all plausible Python
    # backend hires, differing only in the specifics the hard queries probe.
    # ------------------------------------------------------------------
    {
        "id": "c21",
        "name": "Gaurav Shetty",
        "location": "Mumbai",
        "years": 6,
        "title": "Backend Engineer",
        "summary": "Backend engineer with 6 years of experience building Python web services.",
        "experience": [
            "Built Flask REST APIs for an insurance quoting system.",
            "Maintained MySQL schemas and wrote reporting queries.",
            "Handled releases through a Jenkins pipeline.",
        ],
        "skills": "Python, Flask, MySQL, REST APIs, Jenkins, Linux",
    },
    {
        "id": "c22",
        "name": "Divya Pillai",
        "location": "Bangalore",
        "years": 3,
        "title": "Backend Developer",
        "summary": "Backend developer with 3 years of experience writing Python APIs.",
        "experience": [
            "Built FastAPI endpoints for an internal admin tool.",
            "Wrote SQLAlchemy models and Alembic migrations.",
            "Added request validation and improved API error handling.",
        ],
        "skills": "Python, FastAPI, SQLAlchemy, PostgreSQL, pytest",
    },
    {
        "id": "c23",
        "name": "Suresh Babu",
        "location": "Chennai",
        "years": 10,
        "title": "Principal Engineer",
        "summary": "Principal engineer with 10 years of experience leading Python backend teams.",
        "experience": [
            "Owned a large Django and Celery codebase serving 3M users.",
            "Built keyword search over listings using Elasticsearch and BM25 ranking.",
            "Mentored eight engineers and ran the architecture review process.",
        ],
        "skills": "Python, Django, Celery, Elasticsearch, PostgreSQL, RabbitMQ",
    },
    {
        "id": "c24",
        "name": "Lakshmi Narayanan",
        "location": "Chennai",
        "years": 5,
        "title": "Backend Engineer",
        "summary": "Backend engineer with 5 years of experience on search and discovery features.",
        "experience": [
            "Maintained Elasticsearch-backed search for a marketplace.",
            "Ran a pilot adding sentence embeddings to improve recall on long-tail queries.",
            "Built FastAPI services around the search layer.",
        ],
        "skills": "Python, FastAPI, Elasticsearch, embeddings, PostgreSQL, Docker",
    },
    {
        "id": "c25",
        "name": "Imran Qureshi",
        "location": "Hyderabad",
        "years": 6,
        "title": "Senior Backend Engineer",
        "summary": "Senior backend engineer with 6 years of experience building retrieval systems.",
        "experience": [
            "Built a production RAG pipeline over 5M documents using Pinecone as the vector database.",
            "Designed chunking and embedding strategies, and benchmarked retrieval with nDCG.",
            "Served the system through FastAPI with Redis caching.",
        ],
        "skills": "Python, FastAPI, Pinecone, vector database, RAG, embeddings, Redis",
    },
    {
        "id": "c26",
        "name": "Pooja Desai",
        "location": "Pune",
        "years": 4,
        "title": "Backend Engineer",
        "summary": "Backend engineer with 4 years of experience building LLM-backed product features.",
        "experience": [
            "Built semantic search over a support knowledge base using Weaviate.",
            "Implemented hybrid retrieval combining vector similarity with keyword filters.",
            "Shipped FastAPI services backing an AI assistant feature.",
        ],
        "skills": "Python, FastAPI, Weaviate, vector search, semantic search, LangChain",
    },
    {
        "id": "c27",
        "name": "Rakesh Tiwari",
        "location": "Delhi",
        "years": 7,
        "title": "Senior Backend Engineer",
        "summary": "Senior backend engineer with 7 years of experience on high throughput Python services.",
        "experience": [
            "Rebuilt an ingestion service around asyncio and aiohttp, tripling throughput.",
            "Profiled and removed event loop blocking, cutting p99 latency by 60 percent.",
            "Ran load testing and capacity planning for peak traffic events.",
        ],
        "skills": "Python, asyncio, aiohttp, concurrency, performance tuning, PostgreSQL",
    },
    {
        "id": "c28",
        "name": "Shruti Menon",
        "location": "Bangalore",
        "years": 5,
        "title": "Backend Engineer",
        "summary": "Backend engineer with 5 years of experience building real time Python services.",
        "experience": [
            "Built async FastAPI services with WebSocket connections for live dashboards.",
            "Tuned connection pooling and async database access under sustained load.",
            "Instrumented services with structured logging and tracing.",
        ],
        "skills": "Python, FastAPI, asyncio, WebSockets, PostgreSQL, OpenTelemetry",
    },
    {
        "id": "c29",
        "name": "Abhinav Rao",
        "location": "Bangalore",
        "years": 8,
        "title": "Senior Backend Engineer",
        "summary": "Senior backend engineer with 8 years of experience who also owns deployment infrastructure.",
        "experience": [
            "Built Python services and personally owned their Kubernetes deployments.",
            "Wrote Terraform for the team's AWS infrastructure and managed CI/CD in GitHub Actions.",
            "Set up monitoring and on-call runbooks for the services he wrote.",
        ],
        "skills": "Python, FastAPI, Kubernetes, Docker, Terraform, AWS, GitHub Actions",
    },
    {
        "id": "c30",
        "name": "Kavita Singh",
        "location": "Noida",
        "years": 4,
        "title": "Python Developer",
        "summary": "Python developer with 4 years of experience maintaining internal web tools.",
        "experience": [
            "Maintained Flask applications for internal reporting.",
            "Wrote scripts to automate manual finance workflows.",
            "Handled bug fixes and small feature requests.",
        ],
        "skills": "Python, Flask, SQLite, pandas, scripting",
    },
    {
        "id": "c31",
        "name": "Varun Kulkarni",
        "location": "Pune",
        "years": 6,
        "title": "Backend Engineer",
        "summary": "Backend engineer with 6 years of experience in small product teams.",
        "experience": [
            "Built FastAPI services and shipped them with Docker Compose.",
            "Set up GitHub Actions pipelines for test and deploy.",
            "Shared responsibility for production support in a four person team.",
        ],
        "skills": "Python, FastAPI, Docker, GitHub Actions, PostgreSQL",
    },
    {
        "id": "c32",
        "name": "Nisha Rathore",
        "location": "Mumbai",
        "years": 5,
        "title": "Python Engineer",
        "summary": "Python engineer with 5 years of experience spanning data pipelines and light API work.",
        "experience": [
            "Built batch data processing jobs in Python and pandas.",
            "Exposed a few internal endpoints for downstream teams.",
            "Automated reporting delivered to business stakeholders.",
        ],
        "skills": "Python, pandas, SQL, Airflow, scripting",
    },
]


QUERIES = [
    {
        "id": "q1_python_backend",
        "job_description": (
            "We are hiring a Backend Software Engineer with strong Python experience. "
            "You will build and operate microservices using FastAPI, containerised with "
            "Docker, and work on semantic/vector search features backed by a vector "
            "database. Experience designing REST APIs and working with PostgreSQL or "
            "Redis is expected."
        ),
        # c16/c17/c18/c20 are the traps: heavy keyword overlap, wrong role or wrong language.
        "relevance": {"c01": 2, "c03": 2, "c02": 1, "c04": 1, "c19": 1},
    },
    {
        "id": "q2_frontend_react",
        "job_description": (
            "Looking for a Frontend Engineer to build modern web interfaces in React "
            "and TypeScript. You will own component architecture, state management and "
            "front-end performance for a customer-facing product."
        ),
        "relevance": {"c05": 2, "c06": 2, "c07": 2, "c20": 1},
    },
    {
        "id": "q3_ml_nlp",
        "job_description": (
            "Seeking a Machine Learning Engineer specialising in NLP. You will fine-tune "
            "transformer models, work with sentence embeddings and semantic search, and "
            "deploy models to production using PyTorch."
        ),
        "relevance": {"c08": 2, "c10": 2, "c09": 1, "c19": 1},
    },
    {
        "id": "q4_devops",
        "job_description": (
            "Hiring a DevOps / Site Reliability Engineer to run production Kubernetes on "
            "AWS. Responsibilities include infrastructure as code with Terraform, CI/CD "
            "pipelines, and observability for containerised workloads."
        ),
        "relevance": {"c11": 2, "c12": 2, "c13": 1},
    },
    {
        "id": "q5_data_engineer",
        "job_description": (
            "We need a Data Engineer to build large scale data pipelines. You will work "
            "with Spark for batch processing, Airflow for orchestration and Kafka for "
            "streaming ingestion into the warehouse."
        ),
        "relevance": {"c14": 2, "c15": 1},
    },
]

# Hard queries: every strong candidate here sits inside the Python backend
# cluster, so keyword overlap on "Python", "backend", "API" and "FastAPI" is
# near-uniform across the corpus and carries no signal. Only the specific
# requirement separates them. This is the set that actually discriminates.
HARD_QUERIES = [
    {
        "id": "h1_vector_search",
        "job_description": (
            "Senior Python Backend Engineer for our search team. You must have hands-on "
            "production experience with a vector database and semantic / embedding based "
            "retrieval -- this is the core of the role. You will own chunking and "
            "embedding strategy and be responsible for retrieval quality."
        ),
        # Strong: shipped a real vector DB in production.
        # Partial: c24 piloted embeddings; c10 does semantic search but as an NLP
        # researcher rather than a backend engineer.
        "relevance": {"c01": 2, "c03": 2, "c25": 2, "c26": 2, "c24": 1, "c10": 1},
    },
    {
        "id": "h2_async_throughput",
        "job_description": (
            "Python Backend Engineer for a high throughput ingestion platform. We need "
            "deep asyncio and concurrency experience, and a track record of profiling and "
            "tuning services under heavy sustained load."
        ),
        "relevance": {"c27": 2, "c28": 2, "c01": 1, "c03": 1},
    },
    {
        "id": "h3_backend_owns_infra",
        "job_description": (
            "Python Backend Engineer for a small team where engineers own their own "
            "infrastructure. You will write the service and also run it: Docker, "
            "Kubernetes, Terraform and CI/CD pipelines are part of the job."
        ),
        "relevance": {"c29": 2, "c01": 2, "c03": 1, "c31": 1, "c04": 1},
    },
]

ALL_QUERIES = QUERIES + HARD_QUERIES


def render_cv(candidate: dict) -> str:
    """Render a candidate record as resume text with the headings the chunker expects."""
    experience = "\n".join(candidate["experience"])
    return (
        f"{candidate['name']}\n"
        f"{candidate['title']}\n"
        f"{candidate['location']}, India\n"
        f"\n"
        f"Summary\n"
        f"{candidate['summary']}\n"
        f"\n"
        f"Experience\n"
        f"{experience}\n"
        f"\n"
        f"Skills\n"
        f"{candidate['skills']}\n"
    )


def corpus_stats() -> dict:
    """Small summary used by the eval report header."""
    labelled = {cid for q in ALL_QUERIES for cid in q["relevance"]}
    return {
        "candidates": len(CANDIDATES),
        "queries": len(ALL_QUERIES),
        "cross_role_queries": len(QUERIES),
        "within_role_queries": len(HARD_QUERIES),
        "labelled_candidates": len(labelled),
        "distractors": len(CANDIDATES) - len(labelled),
    }
