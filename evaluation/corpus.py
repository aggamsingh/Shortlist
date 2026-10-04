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

from evaluation.enrichment import enrich

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
    {
        # The distractors become targets here. A system that simply suppresses
        # "QA engineer who writes Python" would fail this, which is the point:
        # they are wrong for a backend role, not globally unhireable.
        "id": "q6_qa_automation",
        "job_description": (
            "QA Automation Engineer to build and maintain automated test suites in "
            "Python. You will own Selenium UI automation and pytest frameworks and "
            "keep regression suites running in the CI pipeline."
        ),
        "relevance": {"c16": 2},
    },
    {
        "id": "q7_technical_writer",
        "job_description": (
            "Senior Technical Writer to own developer documentation for our REST "
            "APIs and developer platform, including OpenAPI reference material, "
            "onboarding guides and a maintained style guide."
        ),
        "relevance": {"c17": 2},
    },
    {
        "id": "q8_java_backend",
        "job_description": (
            "Backend Engineer with strong Java and Spring Boot experience to build "
            "microservices, including event-driven integration over Kafka."
        ),
        "relevance": {"c18": 2},
    },
    # ------------------------------------------------------------------
    # Dev set expansion.
    #
    # 15 dev queries could separate a 0.16 nDCG effect (hybrid) but not a 0.05
    # one (chunk size), which left the shipped chunker chosen on noise. These
    # add coverage without touching the held-out set.
    # ------------------------------------------------------------------
    {
        "id": "q9_go_backend",
        "job_description": (
            "Backend Engineer with production Go experience. You will build "
            "internal gRPC services and own the contracts between them."
        ),
        "relevance": {"c04": 2, "c12": 1},
    },
    {
        "id": "q10_kafka_streaming",
        "job_description": (
            "Streaming Data Engineer to own our Kafka ingestion: event-driven "
            "pipelines, exactly-once delivery and schema evolution on topics."
        ),
        # c18 is the trap: real Kafka event-driven integration experience, but a
        # Java backend engineer rather than a data engineer.
        "relevance": {"c14": 2, "c18": 1},
    },
    {
        "id": "q11_iac_terraform",
        "job_description": (
            "Infrastructure Engineer to own our infrastructure as code. You will "
            "write and refactor Terraform modules covering the whole AWS estate."
        ),
        "relevance": {"c11": 2, "c29": 1, "c13": 1, "c04": 1},
    },
    {
        "id": "q12_design_systems",
        "job_description": (
            "Frontend Engineer to own our design system: a typed, reusable "
            "component library that other product teams build against."
        ),
        "relevance": {"c05": 2, "c06": 1, "c07": 1},
    },
    {
        "id": "q13_web_performance",
        "job_description": (
            "Frontend Performance Engineer. The work is measurable: Core Web "
            "Vitals, bundle size, code splitting and render cost."
        ),
        "relevance": {"c05": 2, "c20": 2, "c06": 1},
    },
    {
        "id": "q14_recsys",
        "job_description": (
            "Machine Learning Engineer for recommendations. You will build "
            "ranking models and serve them online to a live product surface."
        ),
        "relevance": {"c19": 2, "c08": 1, "c09": 1},
    },
    {
        "id": "q15_pytorch_training",
        "job_description": (
            "Machine Learning Engineer for training infrastructure: distributed "
            "multi-GPU PyTorch jobs, experiment tracking and reproducibility."
        ),
        "relevance": {"c08": 2, "c10": 1},
    },
    {
        "id": "q16_information_extraction",
        "job_description": (
            "NLP Engineer for information extraction. You will fine-tune "
            "transformer models for named entity recognition over specialised "
            "domain text and measure extraction quality."
        ),
        "relevance": {"c10": 2, "c08": 2},
    },
    {
        "id": "q17_analytics_engineering",
        "job_description": (
            "Analytics Engineer to model data for the business: dbt "
            "transformations on a Snowflake warehouse feeding BI dashboards."
        ),
        "relevance": {"c14": 2, "c15": 1, "c32": 1},
    },
    {
        "id": "q18_monitoring",
        "job_description": (
            "Monitoring Engineer to own metrics and alerting: Prometheus, "
            "Grafana dashboards and actionable alerts for containerised "
            "workloads."
        ),
        "relevance": {"c12": 2, "c11": 1, "c29": 1, "c28": 1},
    },
    {
        "id": "q19_e2e_testing",
        "job_description": (
            "Test Engineer for end-to-end browser automation. You will build and "
            "maintain UI test suites and keep them green in the CI pipeline."
        ),
        "relevance": {"c16": 2, "c07": 2, "c06": 1},
    },
    {
        "id": "q20_llm_product",
        "job_description": (
            "Engineer for LLM-backed product features: prompt orchestration, "
            "retrieval augmented generation and shipping an assistant surface "
            "to real users."
        ),
        "relevance": {"c26": 2, "c25": 2, "c08": 1},
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
    {
        "id": "h4_django_web",
        "job_description": (
            "Python Backend Engineer for a large Django web application. You will "
            "work daily with the Django ORM and Django REST Framework, and own "
            "relational data modelling on PostgreSQL."
        ),
        # Partial: web API work on a different framework (Flask), or relational
        # modelling without Django.
        "relevance": {"c02": 2, "c23": 2, "c21": 1, "c22": 1, "c30": 1},
    },
    {
        "id": "h5_db_performance",
        "job_description": (
            "Python Backend Engineer with strong relational database skills. The "
            "role centres on query optimisation, indexing, and profiling slow "
            "endpoints against PostgreSQL under real production load."
        ),
        "relevance": {"c02": 2, "c27": 2, "c28": 1, "c22": 1, "c23": 1},
    },
    {
        "id": "h6_lexical_search",
        "job_description": (
            "Backend Engineer to own keyword search for a marketplace catalogue: "
            "Elasticsearch, BM25 relevance tuning and query analysis. This is "
            "lexical search work, not embedding based retrieval."
        ),
        # A deliberately adversarial pairing with h1: the vector-search people are
        # the WRONG answer here, and a system that has learned "search -> vector
        # database" from the other queries will get this backwards.
        "relevance": {"c23": 2, "c24": 2, "c26": 1, "c10": 1},
    },
    {
        "id": "h7_task_queues",
        "job_description": (
            "Python Backend Engineer for asynchronous background processing. You "
            "will own Celery task queues, RabbitMQ messaging and reliable retry "
            "semantics for long running jobs."
        ),
        "relevance": {"c23": 2, "c02": 2, "c03": 1},
    },
    # ------------------------------------------------------------------
    # Dev set expansion (within-role). Same Python backend cluster, ten further
    # requirements that separate people inside it.
    # ------------------------------------------------------------------
    {
        "id": "h8_flask_internal_tools",
        "job_description": (
            "Python Backend Engineer for internal tooling. Small Flask "
            "applications, reporting screens and automating manual workflows "
            "for finance and operations teams."
        ),
        # The deliberately modest CVs are the RIGHT answer here. A system that
        # has learned "senior and distributed means better" ranks them last.
        "relevance": {"c30": 2, "c21": 2, "c32": 1, "c22": 1},
    },
    {
        "id": "h9_orm_migrations",
        "job_description": (
            "Python Backend Engineer to own the data access layer: SQLAlchemy "
            "models, Alembic migrations and evolving schemas without downtime."
        ),
        "relevance": {"c22": 2, "c02": 1, "c21": 1, "c23": 1},
    },
    {
        "id": "h10_caching_latency",
        "job_description": (
            "Python Backend Engineer to cut response times with caching. You "
            "will own the Redis layer, cache invalidation and hit-rate "
            "measurement."
        ),
        "relevance": {"c01": 2, "c03": 2, "c25": 2, "c28": 1},
    },
    {
        "id": "h11_grpc_contracts",
        "job_description": (
            "Python Backend Engineer for service-to-service APIs: gRPC, "
            "protobuf contracts and versioning them across several teams."
        ),
        "relevance": {"c04": 2, "c01": 1, "c03": 1},
    },
    {
        "id": "h12_instrumentation",
        "job_description": (
            "Python Backend Engineer to instrument our services properly: "
            "structured logging, distributed tracing and OpenTelemetry "
            "throughout the request path."
        ),
        "relevance": {"c28": 2, "c29": 1, "c12": 1, "c01": 1},
    },
    {
        "id": "h13_load_testing",
        "job_description": (
            "Python Backend Engineer for performance validation: load testing, "
            "capacity planning and profiling services ahead of peak traffic."
        ),
        "relevance": {"c27": 2, "c11": 1, "c29": 1, "c01": 1},
    },
    {
        "id": "h14_raises_the_bar",
        "job_description": (
            "Senior Python Backend Engineer to raise engineering standards on "
            "the team: code review, architecture review and mentoring the "
            "junior engineers."
        ),
        "relevance": {"c23": 2, "c01": 2, "c29": 1, "c04": 1},
    },
    {
        "id": "h15_mysql_relational",
        "job_description": (
            "Python Backend Engineer with deep relational modelling experience "
            "on MySQL or Oracle: schema design, reporting queries and logic "
            "held in the database."
        ),
        # Pairs against h5, which is the same skill on PostgreSQL under load.
        "relevance": {"c21": 2, "c02": 1, "c23": 1, "c15": 1},
    },
    {
        "id": "h16_retrieval_evaluation",
        "job_description": (
            "Python Backend Engineer to own retrieval quality measurement: "
            "build the benchmark, compute nDCG and recall, and decide on the "
            "evidence whether a change ships."
        ),
        # c10 has done exactly this work but is an NLP engineer, not a backend
        # hire, so a 1 rather than a 2.
        "relevance": {"c25": 2, "c10": 1, "c24": 1, "c26": 1, "c01": 1},
    },
    {
        "id": "h17_small_team_support",
        "job_description": (
            "Python Backend Engineer for a four person team. You will ship "
            "features, run your own deploys and share the production support "
            "rota. Breadth matters more than depth here."
        ),
        "relevance": {"c31": 2, "c29": 2, "c21": 1, "c01": 1},
    },
]

# ---------------------------------------------------------------------------
# HELD-OUT TEST QUERIES
#
# Everything above is the DEV set. Every design decision in this project --
# chunking strategy, window size, hybrid vs dense, retrieval budget -- was made
# by looking at those 15 queries, and then the same 15 were used to report the
# headline numbers. That is tuning on the test set, and it makes those numbers
# optimistic by an unknown margin.
#
# Partitioning the dev set would not fix it: all 15 have already influenced the
# configuration, so neither half is clean. A real held-out set has to consist of
# queries that have never been looked at while choosing anything. These are
# those queries, written against the existing candidates without consulting any
# retrieval output.
#
# They also deliberately probe axes the dev set never touches -- seniority,
# domain, legacy migration, accessibility, testing discipline, the conjunction
# of lexical AND vector search -- so they measure generalisation to new kinds of
# job description, not just new wordings of the old ones.
#
# The discipline this only works under: do not tune against these. If a
# measurement on the test set motivates a change, the change is chosen on dev,
# and the test set is re-measured once afterwards.
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    {
        "id": "t1_full_stack",
        "job_description": (
            "Full Stack Engineer to own features end to end: a React and "
            "TypeScript frontend talking to Node.js and Express services backed "
            "by MongoDB. You will write both halves and the tests between them."
        ),
        # Only c07 has both halves. The React-only engineers are plausible but
        # unproven on the backend, which is what the grade split records.
        "relevance": {"c07": 2, "c05": 1, "c06": 1},
    },
    {
        "id": "t2_cloud_architecture",
        "job_description": (
            "Cloud Architect to design our AWS account structure, landing zones "
            "and governance model for a regulated financial institution. You will "
            "own networking, security boundaries and cloud cost management."
        ),
        "relevance": {"c13": 2, "c11": 1, "c12": 1},
    },
    {
        "id": "t3_observability_oncall",
        "job_description": (
            "Site Reliability Engineer focused on observability and incident "
            "response. You will define SLOs and error budgets, build dashboards "
            "and tracing, run on-call and write postmortems."
        ),
        # Narrower than the dev DevOps query: this turns on SLOs, tracing and
        # on-call rather than on Kubernetes and Terraform.
        "relevance": {"c12": 2, "c11": 1, "c29": 1, "c28": 1},
    },
    {
        "id": "t4_data_warehouse_sql",
        "job_description": (
            "Data Warehouse Developer for an enterprise reporting platform. The "
            "work is heavy SQL: dimensional modelling, stored procedures, nightly "
            "batch loads and reconciliation against source systems."
        ),
        # Deliberately inverts the dev data-engineering query, where c14 was the
        # strong hire and c15 the partial. A system that has learned
        # "data -> Spark and Kafka" gets this one backwards.
        "relevance": {"c15": 2, "c14": 1, "c32": 1},
    },
    {
        "id": "t5_mlops",
        "job_description": (
            "Machine Learning Engineer to own models in production: training "
            "pipelines, experiment tracking, deployment of inference services and "
            "monitoring for model drift after release."
        ),
        "relevance": {"c08": 2, "c19": 1, "c10": 1},
    },
    {
        "id": "t6_experimentation",
        "job_description": (
            "Data Scientist for our experimentation and forecasting work. You "
            "will design and read A/B tests, estimate causal impact for product "
            "teams, and build demand forecasts with gradient boosted models."
        ),
        "relevance": {"c09": 2, "c19": 1, "c08": 1},
    },
    {
        "id": "t7_accessible_frontend",
        "job_description": (
            "Frontend Developer with a focus on accessibility. You will take our "
            "React product to WCAG AA, own responsive layouts across devices and "
            "write component level tests."
        ),
        # c06 is the only CV with accessibility evidence; the stronger generalist
        # React engineers are partials. Tests whether retrieval can find one
        # specific requirement inside a role it already separates easily.
        "relevance": {"c06": 2, "c05": 1, "c20": 1},
    },
    {
        "id": "t8_platform_engineering",
        "job_description": (
            "Platform Engineer to build the internal developer platform other "
            "teams ship on. Expect Go and gRPC service scaffolding, a service "
            "mesh, and tooling that 40 or more engineers depend on daily."
        ),
        "relevance": {"c04": 2, "c29": 1, "c11": 1, "c12": 1},
    },
    {
        "id": "t9_engineering_leadership",
        "job_description": (
            "Principal Engineer to set technical direction across several teams. "
            "You will mentor engineers, run architecture review, and be "
            "accountable for the long term health of a large codebase."
        ),
        # A seniority and scope axis rather than a technology axis. Nothing in the
        # dev set probes this, and the keywords barely appear in the corpus.
        "relevance": {"c23": 2, "c13": 2, "c01": 1, "c04": 1},
    },
]

# Held-out WITHIN-ROLE queries: the same Python backend cluster as HARD_QUERIES,
# so the generic keywords carry no signal, but probing requirements the dev hard
# queries never mention.
TEST_HARD_QUERIES = [
    {
        "id": "ht1_testing_discipline",
        "job_description": (
            "Python Backend Engineer who takes testing seriously. You will own "
            "unit and integration test coverage for our services, keep the suite "
            "fast and meaningful, and gate every merge on it in CI."
        ),
        # c16 is the trap and is graded 1, not 0: a QA automation engineer with
        # deep pytest experience is genuinely adjacent, just not a backend hire.
        # Suppressing them entirely would be as wrong as ranking them first.
        "relevance": {"c01": 2, "c22": 2, "c31": 1, "c02": 1, "c16": 1},
    },
    {
        "id": "ht2_fintech_domain",
        "job_description": (
            "Python Backend Engineer for a payments platform. You will work on "
            "money movement in a regulated environment, where correctness, "
            "auditability and idempotent retries matter more than throughput."
        ),
        # Domain rather than stack. c18 is the trap: insurance, regulated,
        # microservices, Kafka -- but Java, where the role demands Python.
        "relevance": {"c03": 2, "c04": 2, "c21": 1, "c18": 1},
    },
    {
        "id": "ht3_realtime_websockets",
        "job_description": (
            "Python Backend Engineer for live, always-connected features. The "
            "role centres on WebSocket connections pushing real time updates to "
            "dashboards, and keeping thousands of them open per instance."
        ),
        "relevance": {"c28": 2, "c27": 1, "c03": 1},
    },
    {
        "id": "ht4_consumer_scale",
        "job_description": (
            "Senior Python Backend Engineer for a consumer product serving "
            "millions of users. We want someone who has carried a system at that "
            "scale, not just built one, including capacity planning for peaks."
        ),
        "relevance": {"c23": 2, "c01": 2, "c27": 2, "c29": 1},
    },
    {
        "id": "ht5_legacy_modernisation",
        "job_description": (
            "Python Backend Engineer to modernise a legacy system. You will break "
            "a monolith into deployable services incrementally, without a rewrite "
            "and without downtime for existing users."
        ),
        "relevance": {"c01": 2, "c23": 1, "c04": 1, "c13": 1},
    },
    {
        "id": "ht6_hybrid_search",
        "job_description": (
            "Backend Engineer for relevance. You will run both keyword and "
            "embedding based retrieval over the same catalogue and combine them "
            "into one ranking, then prove the combination beats either alone."
        ),
        # The conjunction of dev h1 (vector) and dev h6 (lexical), and so a real
        # generalisation test: the right answers are the two people who have done
        # both, ranked above the specialists who have only done one.
        "relevance": {"c26": 2, "c24": 2, "c25": 1, "c23": 1, "c01": 1},
    },
    {
        "id": "ht7_batch_and_light_api",
        "job_description": (
            "Python Engineer sitting between data and backend: scheduled batch "
            "jobs over large files with pandas, plus a handful of internal "
            "endpoints so other teams can pull the results."
        ),
        # Targets the weak end of the Python cluster, who are never the right
        # answer anywhere else. Without a query like this the corpus never tests
        # whether retrieval can rank them ABOVE the strong engineers when the job
        # is genuinely smaller.
        "relevance": {"c32": 2, "c30": 1, "c14": 1},
    },
    {
        "id": "ht8_early_career",
        "job_description": (
            "Python Backend Developer, early career, two to four years of "
            "experience. You will be mentored by senior engineers while building "
            "internal API endpoints and learning our stack."
        ),
        # A seniority axis. Honest caveat: years_of_experience is also a metadata
        # filter, so this query exercises the text path only, where the signal is
        # the phrasing of the summary line rather than a number.
        "relevance": {"c22": 2, "c30": 1, "c26": 1},
    },
]


# ---------------------------------------------------------------------------
# Splits
#
# The split is stamped onto the query itself, so a query cannot be silently
# moved between sets by a change to list membership somewhere else.
# ---------------------------------------------------------------------------

DEV_QUERIES = QUERIES
DEV_HARD_QUERIES = HARD_QUERIES

for _q in DEV_QUERIES + DEV_HARD_QUERIES:
    _q["split"] = "dev"
for _q in TEST_QUERIES + TEST_HARD_QUERIES:
    _q["split"] = "test"
del _q

SPLITS = ("dev", "test", "all")

ALL_QUERIES = DEV_QUERIES + DEV_HARD_QUERIES + TEST_QUERIES + TEST_HARD_QUERIES


def select_queries(split: str = "dev", kind: str = "all") -> list:
    """Queries for one split.

    `kind` is "cross" (different roles, separable by topic), "within" (all
    Python backend, separable only by one requirement) or "all".

    Defaults to dev, because the default has to be the safe one: a harness that
    reported test numbers unless told otherwise would be tuned against within a
    week, which is how the leakage happened the first time.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if kind not in ("cross", "within", "all"):
        raise ValueError(f"kind must be cross, within or all, got {kind!r}")

    cross = (DEV_QUERIES if split in ("dev", "all") else []) + (
        TEST_QUERIES if split in ("test", "all") else []
    )
    within = (DEV_HARD_QUERIES if split in ("dev", "all") else []) + (
        TEST_HARD_QUERIES if split in ("test", "all") else []
    )
    if kind == "cross":
        return cross
    if kind == "within":
        return within
    return cross + within


def render_cv(candidate: dict) -> str:
    """Render a candidate record as a full-length resume document.

    Padded to a realistic length with education, generic duties, projects and
    interests, and rendered under one of three layouts. See
    evaluation/enrichment.py for why: the original 55-word fixtures were far too
    short for any chunking strategy to differ from any other, so the ablation was
    measuring the fixture rather than the strategy.

    The padding is deliberately non-discriminating, so every relevance label
    stays valid -- it adds length and noise, never evidence.
    """
    extra = enrich(candidate)
    experience = "\n".join(candidate["experience"] + extra["duties"])
    projects = "\n".join(extra["projects"])
    header = (
        candidate["name"] + "\n"
        + candidate["title"] + "\n"
        + candidate["location"] + ", India" + "\n"
    )

    if extra["layout"] == "headingless":
        # No recognisable headings: exercises the chunker's fallback path, which
        # a uniformly well-structured corpus would never reach.
        return "\n".join([
            header,
            candidate["summary"],
            experience,
            "Core technologies used day to day: " + candidate["skills"] + ".",
            projects,
            extra["education"],
            extra["certifications"],
            extra["achievements"],
            extra["languages"] + " " + extra["interests"],
        ]) + "\n"

    if extra["layout"] == "reordered":
        sections = [
            ("Skills", candidate["skills"]),
            ("Education", extra["education"]),
            ("Summary", candidate["summary"]),
            ("Experience", experience),
            ("Projects", projects),
            ("Certifications", extra["certifications"]),
            ("Achievements", extra["achievements"]),
            ("Languages", extra["languages"]),
            ("Interests", extra["interests"]),
        ]
    else:
        sections = [
            ("Summary", candidate["summary"]),
            ("Experience", experience),
            ("Skills", candidate["skills"]),
            ("Projects", projects),
            ("Education", extra["education"]),
            ("Certifications", extra["certifications"]),
            ("Achievements", extra["achievements"]),
            ("Languages", extra["languages"]),
            ("Interests", extra["interests"]),
        ]

    body = "\n".join(
        heading + "\n" + text + "\n" for heading, text in sections
    )
    return header + "\n" + body


def corpus_stats(split: str = "all") -> dict:
    """Summary used by the eval report header.

    Distraction is measured PER QUERY, not globally. Counting candidates that
    are never relevant to anything understates it badly once most candidates are
    the target of some query -- which is now every one of them. What matters is
    that each individual query has a large field of non-relevant candidates to
    be confused by, many of them sharing its keywords.
    """
    queries = select_queries(split, "all")
    per_query_distractors = [len(CANDIDATES) - len(q["relevance"]) for q in queries]
    never_relevant = len(CANDIDATES) - len(
        {cid for q in queries for cid in q["relevance"]}
    )
    return {
        "candidates": len(CANDIDATES),
        "split": split,
        "queries": len(queries),
        "cross_role_queries": len(select_queries(split, "cross")),
        "within_role_queries": len(select_queries(split, "within")),
        "dev_queries": len(select_queries("dev", "all")),
        "test_queries": len(select_queries("test", "all")),
        "min_distractors_per_query": min(per_query_distractors),
        "mean_distractors_per_query": round(
            sum(per_query_distractors) / len(per_query_distractors), 1
        ),
        "never_relevant_anywhere": never_relevant,
    }
