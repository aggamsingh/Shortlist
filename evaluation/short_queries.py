"""Short, keyword-style versions of the labelled queries.

The labelled queries are full job descriptions. A person using a search box types
"python vector database", not a paragraph, and nothing in the project had measured
that. Two styles, both written by hand and BEFORE any retrieval output was looked
at, so they are not tuned to what the system does well:

    keywords   role plus the skills that matter, 3-8 words. Defined for every query.
    title      the job title alone, 2-4 words. Defined only where a bare title
               legitimately points at the labelled people.

Why `title` does not cover the within-role queries. Their labels are narrow on
purpose: "python backend engineer" is a perfectly good title for all sixteen Python
backend CVs, but only the two or three with the specific skill are labelled strong.
Scoring the bare title against those labels would call the other thirteen wrong when
they are right. `keywords` keeps the distinguishing skill, so the labels stay valid.

A caveat that applies to everything here: the relevance labels were written for the
full job descriptions. They are a good proxy for the keyword forms, not a fresh set
of judgements, and a short query is more ambiguous than the paragraph it came from.
"""

KEYWORDS = {
    # ---- dev, cross-role
    "q1_python_backend": "python backend fastapi docker vector search",
    "q2_frontend_react": "react typescript frontend component state performance",
    "q3_ml_nlp": "nlp transformers embeddings semantic search pytorch",
    "q4_devops": "kubernetes aws terraform ci/cd observability",
    "q5_data_engineer": "spark airflow kafka data pipelines",
    "q6_qa_automation": "selenium pytest python test automation",
    "q7_technical_writer": "api documentation openapi style guide",
    "q8_java_backend": "java spring boot kafka microservices",
    "q9_go_backend": "go grpc services",
    "q10_kafka_streaming": "kafka streaming event-driven exactly-once",
    "q11_iac_terraform": "terraform infrastructure as code aws",
    "q12_design_systems": "design system react component library typescript",
    "q13_web_performance": "web performance lighthouse bundle size code splitting",
    "q14_recsys": "recommendation ranking models fastapi serving",
    "q15_pytorch_training": "pytorch distributed multi-gpu training mlflow",
    "q16_information_extraction": "named entity recognition bert fine-tuning",
    "q17_analytics_engineering": "dbt snowflake sql analytics",
    "q18_monitoring": "prometheus grafana alerting kubernetes",
    "q19_e2e_testing": "playwright selenium end-to-end ui testing ci",
    "q20_llm_product": "rag llm vector database assistant",
    # ---- dev, within-role
    "h1_vector_search": "python vector database semantic search retrieval embeddings",
    "h2_async_throughput": "python asyncio concurrency performance profiling high throughput",
    "h3_backend_owns_infra": "python backend docker kubernetes terraform ci/cd own infrastructure",
    "h4_django_web": "python django rest framework orm postgresql",
    "h5_db_performance": "python postgresql query optimisation indexing performance",
    "h6_lexical_search": "elasticsearch bm25 lexical search relevance tuning",
    "h7_task_queues": "python celery rabbitmq task queues retries",
    "h8_flask_internal_tools": "python flask internal tools reporting automation",
    "h9_orm_migrations": "python sqlalchemy alembic migrations",
    "h10_caching_latency": "python redis caching latency invalidation",
    "h11_grpc_contracts": "python grpc protobuf service contracts",
    "h12_instrumentation": "python opentelemetry tracing structured logging instrumentation",
    "h13_load_testing": "python load testing capacity planning profiling",
    "h14_raises_the_bar": "senior python code review architecture mentoring",
    "h15_mysql_relational": "python mysql oracle relational schema design reporting queries",
    "h16_retrieval_evaluation": "retrieval evaluation ndcg recall benchmark python",
    "h17_small_team_support": "python small team fastapi docker ci/cd production support",
    # ---- held-out, cross-role
    "t1_full_stack": "react typescript node express mongodb",
    "t2_cloud_architecture": "aws landing zone governance security networking cost",
    "t3_observability_oncall": "slo error budgets tracing on-call postmortems",
    "t4_data_warehouse_sql": "sql stored procedures dimensional modelling batch loads",
    "t5_mlops": "model deployment drift monitoring training pipelines experiment tracking",
    "t6_experimentation": "ab testing causal impact forecasting gradient boosting",
    "t7_accessible_frontend": "wcag accessibility react responsive component tests",
    "t8_platform_engineering": "internal developer platform go grpc service mesh",
    "t9_engineering_leadership": "technical direction mentoring architecture review codebase health",
    # ---- held-out, within-role
    "ht1_testing_discipline": "python unit integration testing ci coverage pytest",
    "ht2_fintech_domain": "python payments fintech regulated idempotency auditability",
    "ht3_realtime_websockets": "python websockets real time async fastapi",
    "ht4_consumer_scale": "python large scale millions of users capacity planning",
    "ht5_legacy_modernisation": "python monolith to microservices legacy modernisation",
    "ht6_hybrid_search": "hybrid search keyword and vector retrieval combine ranking",
    "ht7_batch_and_light_api": "python pandas batch jobs internal api endpoints",
    "ht8_early_career": "junior python backend developer mentored internal apis",
}

# Cross-role queries whose job title alone points at the labelled people.
# q1 is left out: "python backend engineer" fits sixteen CVs, two are labelled strong.
TITLES = {
    "q2_frontend_react": "frontend engineer",
    "q3_ml_nlp": "machine learning engineer nlp",
    "q4_devops": "devops sre engineer",
    "q5_data_engineer": "data engineer",
    "q6_qa_automation": "qa automation engineer",
    "q7_technical_writer": "technical writer",
    "q8_java_backend": "java backend engineer",
    "q9_go_backend": "go backend engineer",
    "q10_kafka_streaming": "streaming data engineer",
    "q11_iac_terraform": "infrastructure engineer terraform",
    "q12_design_systems": "design system engineer",
    "q13_web_performance": "frontend performance engineer",
    "q14_recsys": "recommendation systems ml engineer",
    "q15_pytorch_training": "ml training infrastructure engineer",
    "q16_information_extraction": "nlp engineer",
    "q17_analytics_engineering": "analytics engineer",
    "q18_monitoring": "monitoring engineer",
    "q19_e2e_testing": "test automation engineer",
    "q20_llm_product": "llm applications engineer",
    "t1_full_stack": "full stack engineer",
    "t2_cloud_architecture": "cloud architect",
    "t3_observability_oncall": "site reliability engineer",
    "t4_data_warehouse_sql": "data warehouse developer",
    "t5_mlops": "mlops engineer",
    "t6_experimentation": "data scientist",
    "t7_accessible_frontend": "accessibility frontend developer",
    "t8_platform_engineering": "platform engineer",
    "t9_engineering_leadership": "principal engineer",
}

STYLES = ("jd", "title", "keywords")


def short_text(query_id: str, style: str):
    """The short form of a query, or None if that style has none for it."""
    if style == "jd":
        raise ValueError("'jd' is the full job description; there is nothing to look up")
    if style == "keywords":
        return KEYWORDS.get(query_id)
    if style == "title":
        return TITLES.get(query_id)
    raise ValueError(f"style must be one of {STYLES}, got {style!r}")
