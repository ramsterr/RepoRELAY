"""
Topic inference from README and description text.

For repos with sparse or empty topic lists, this module scans the
README and description against a curated keyword→topic vocabulary
to infer additional GitHub-style topic tags.

Design goals:
  - High precision over recall — wrong topics hurt more than missing ones
  - Fast — pure text matching, no API calls, no model inference
  - Conservative — caps at MAX_INFERRED topics per repo
  - Additive — never removes existing topics, only supplements them
"""

from __future__ import annotations

import re
from collections import Counter

MAX_INFERRED = 15
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _bigrams(tokens: list[str]) -> list[str]:
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]


# ── Keyword → topic vocabulary ──────────────────────────────────────
# Each key is a unigram or bigram that, when found in text, provides
# evidence for one or more topic tags. Values are topic strings.
# Phrases (bigrams) are checked first for specificity; unigrams
# are only used if they don't overlap with a phrase match.

_KEYWORD_TOPICS: dict[str, list[str]] = {
    # ── ML / Deep Learning ───────────────────────────────────────
    "pytorch": ["pytorch", "deep-learning"],
    "torch": ["pytorch", "deep-learning"],
    "tensorflow": ["tensorflow", "deep-learning"],
    "keras": ["keras", "deep-learning"],
    "scikit-learn": ["scikit-learn", "machine-learning"],
    "sklearn": ["scikit-learn", "machine-learning"],
    "xgboost": ["xgboost", "machine-learning"],
    "lightgbm": ["lightgbm", "machine-learning"],
    "hugging face": ["huggingface", "transformers"],
    "huggingface": ["huggingface", "transformers"],
    "transformers": ["transformers"],
    "neural network": ["neural-network", "deep-learning"],
    "neural networks": ["neural-network", "deep-learning"],
    "convolutional": ["cnn", "deep-learning"],
    "convolution": ["cnn", "deep-learning"],
    "recurrent": ["rnn", "deep-learning"],
    "lstm": ["rnn", "deep-learning"],
    "gan": ["gan", "deep-learning"],
    "generative adversarial": ["gan", "deep-learning"],
    "reinforcement learning": ["reinforcement-learning"],
    "q-learning": ["reinforcement-learning"],
    "deep reinforcement": ["reinforcement-learning"],
    "object detection": ["object-detection", "computer-vision"],
    "image classification": ["image-classification", "computer-vision"],
    "semantic segmentation": ["semantic-segmentation", "computer-vision"],
    "computer vision": ["computer-vision"],
    "image recognition": ["computer-vision"],
    "natural language processing": ["nlp", "natural-language-processing"],
    "text classification": ["text-classification", "nlp"],
    "sentiment analysis": ["sentiment-analysis", "nlp"],
    "named entity": ["nlp"],
    "machine learning": ["machine-learning"],
    "deep learning": ["deep-learning"],
    "feature engineering": ["feature-engineering"],
    "model deployment": ["model-deployment"],
    "mlops": ["mlops"],
    "diffusion model": ["diffusion", "generative-ai"],
    "stable diffusion": ["stable-diffusion", "diffusion"],
    "fine-tuning": ["fine-tuning"],
    "fine tuning": ["fine-tuning"],
    "transfer learning": ["deep-learning"],
    "prompt engineering": ["prompt-engineering"],

    # ── LLM / AI ─────────────────────────────────────────────────
    "large language model": ["llm", "ai"],
    "langchain": ["langchain", "llm"],
    "llamaindex": ["llamaindex", "llm"],
    "openai": ["openai", "ai"],
    "gpt-4": ["openai", "llm"],
    "gpt-3": ["openai", "llm"],
    "chatgpt": ["openai", "llm"],
    "semantic kernel": ["semantic-kernel", "ai"],
    "autogen": ["autogen", "ai"],
    "retrieval augmented": ["rag", "llm"],
    "retrieval-augmented": ["rag", "llm"],
    "vector database": ["vector-database"],
    "vector store": ["vector-database"],
    "embedding model": ["embedding", "ai"],
    "sentence-transformers": ["embedding", "nlp"],
    "sentence transformers": ["embedding", "nlp"],
    "chatbot": ["chatbot"],
    "conversational ai": ["chatbot", "ai"],
    "generative ai": ["generative-ai", "ai"],
    "text generation": ["generative-ai", "llm"],
    "image generation": ["generative-ai", "diffusion"],
    "copilot": ["ai"],

    # ── Web Frontend ─────────────────────────────────────────────
    "react.js": ["react", "javascript"],
    "reactjs": ["react", "javascript"],
    "react native": ["react-native", "mobile"],
    "vue.js": ["vue", "javascript"],
    "vuejs": ["vue", "javascript"],
    "nuxt.js": ["nuxt", "vue"],
    "nuxtjs": ["nuxt", "vue"],
    "next.js": ["nextjs", "react"],
    "nextjs": ["nextjs", "react"],
    "sveltekit": ["svelte", "astro"],
    "svelte.js": ["svelte"],
    "angular": ["angular", "typescript"],
    "tailwind css": ["tailwindcss", "css"],
    "tailwindcss": ["tailwindcss"],
    "styled components": ["styled-components"],
    "material ui": ["material-ui"],
    "chakra ui": ["chakra-ui"],
    "ant design": ["ant-design"],
    "web components": ["web-components"],
    "progressive web app": ["pwa"],
    "service worker": ["service-worker", "pwa"],
    "server side rendering": ["ssr"],
    "static site": ["ssg", "astro"],
    "vite": ["vite", "build-tool"],
    "esbuild": ["esbuild", "build-tool"],
    "webpack": ["webpack", "build-tool"],
    "parcel bundler": ["parcel", "build-tool"],
    "storybook": ["storybook"],
    "design system": ["design-system"],
    "design token": ["design-system"],

    # ── Backend ──────────────────────────────────────────────────
    "fastapi": ["fastapi", "python"],
    "django": ["django", "python"],
    "flask": ["flask", "python"],
    "express.js": ["express", "nodejs"],
    "expressjs": ["express", "nodejs"],
    "nestjs": ["nestjs", "typescript"],
    "spring boot": ["spring-boot", "java"],
    "spring framework": ["spring", "java"],
    "gin": ["gin", "go"],
    "echo framework": ["echo", "go"],
    "fiber": ["fiber", "go"],
    "actix": ["actix", "rust"],
    "actix-web": ["actix", "rust"],
    "rocket": ["rocket", "rust"],
    "axum": ["axum", "rust"],
    "phoenix framework": ["phoenix", "elixir"],
    "ruby on rails": ["rails", "ruby"],
    "laravel": ["laravel", "php"],
    "asp.net": ["aspnet", "csharp"],
    "graphql": ["graphql"],
    "rest api": ["rest-api"],
    "grpc": ["grpc"],
    "trpc": ["trpc", "typescript"],
    "websocket": ["websocket"],
    "web socket": ["websocket"],
    "microservice": ["microservices"],
    "api gateway": ["api-gateway"],
    "serverless": ["serverless"],
    "cloud function": ["serverless"],

    # ── Mobile ───────────────────────────────────────────────────
    "flutter": ["flutter", "mobile", "dart"],
    "swiftui": ["swiftui", "ios", "swift"],
    "swift ui": ["swiftui", "ios", "swift"],
    "jetpack compose": ["jetpack-compose", "android", "kotlin"],
    "kotlin multiplatform": ["kotlin-multiplatform"],
    "expo": ["expo", "react-native"],
    "react-native": ["react-native", "mobile"],
    "capacitor": ["capacitor", "mobile"],
    "ionic": ["ionic", "mobile"],
    "nativescript": ["nativescript", "mobile"],
    "android app": ["android", "mobile"],
    "ios app": ["ios", "mobile"],
    "mobile app": ["mobile"],

    # ── DevOps / Cloud ───────────────────────────────────────────
    "kubernetes": ["kubernetes", "container-orchestration"],
    "k8s": ["kubernetes", "container-orchestration"],
    "docker": ["docker", "container-orchestration"],
    "container": ["docker", "container-orchestration"],
    "terraform": ["terraform", "infrastructure-as-code"],
    "ansible": ["ansible"],
    "helm chart": ["helm", "kubernetes"],
    "argocd": ["argocd", "gitops"],
    "gitops": ["gitops"],
    "ci/cd": ["ci-cd"],
    "cicd": ["ci-cd"],
    "github actions": ["github-actions", "ci-cd"],
    "gitlab ci": ["gitlab-ci", "ci-cd"],
    "jenkins": ["jenkins", "ci-cd"],
    "circleci": ["circleci", "ci-cd"],
    "prometheus": ["prometheus", "monitoring"],
    "grafana": ["grafana", "monitoring"],
    "datadog": ["datadog", "monitoring"],
    "cloudformation": ["cloudformation", "aws"],
    "pulumi": ["pulumi", "infrastructure-as-code"],
    "istio": ["istio", "service-mesh"],
    "service mesh": ["service-mesh"],
    "aws lambda": ["aws", "serverless"],
    "amazon web services": ["aws"],
    "azure": ["azure"],
    "google cloud": ["gcp"],
    "gcp": ["gcp"],
    "vagrant": ["vagrant"],
    "podman": ["podman"],

    # ── Databases ────────────────────────────────────────────────
    "postgresql": ["postgresql", "database"],
    "postgres": ["postgresql", "database"],
    "mysql": ["mysql", "database"],
    "sqlite": ["sqlite", "database"],
    "mongodb": ["mongodb", "database"],
    "redis": ["redis", "database"],
    "elasticsearch": ["elasticsearch", "search-engine"],
    "elastic search": ["elasticsearch", "search-engine"],
    "cassandra": ["cassandra", "database"],
    "dynamodb": ["dynamodb", "database", "aws"],
    "couchdb": ["couchdb", "database"],
    "neo4j": ["neo4j", "database"],
    "cockroachdb": ["cockroachdb", "database"],
    "prisma": ["prisma", "database"],
    "typeorm": ["typeorm", "database"],
    "sequelize": ["sequelize", "database"],
    "drizzle": ["drizzle", "database"],
    "sqlalchemy": ["sqlalchemy", "database", "python"],
    "supabase": ["supabase", "database", "postgresql"],
    "mariadb": ["mariadb", "database"],
    "clickhouse": ["clickhouse", "database"],
    "surrealdb": ["surrealdb", "database"],
    "chromadb": ["chromadb", "vector-database"],
    "pinecone": ["pinecone", "vector-database"],
    "weaviate": ["weaviate", "vector-database"],
    "qdrant": ["qdrant", "vector-database"],
    "milvus": ["milvus", "vector-database"],
    "faiss": ["faiss", "vector-database"],
    "orm": ["database"],

    # ── Security ─────────────────────────────────────────────────
    "penetration testing": ["penetration-testing", "security"],
    "pen testing": ["penetration-testing", "security"],
    "vulnerability": ["vulnerability-scanning", "security"],
    "owasp": ["owasp", "security"],
    "oauth": ["oauth", "security"],
    "jwt": ["jwt", "security", "authentication"],
    "json web token": ["jwt", "security", "authentication"],
    "authentication": ["authentication", "security"],
    "authorization": ["authorization", "security"],
    "single sign-on": ["sso", "security"],
    "zero trust": ["zero-trust", "security"],
    "encryption": ["encryption", "cryptography"],
    "cryptography": ["cryptography", "security"],
    "devsecops": ["devsecops", "security"],
    "siem": ["siem", "security"],
    "firewall": ["security"],

    # ── Testing ──────────────────────────────────────────────────
    "playwright": ["playwright", "end-to-end-testing"],
    "cypress": ["cypress", "end-to-end-testing"],
    "selenium": ["selenium", "end-to-end-testing"],
    "jest": ["jest", "unit-testing", "javascript"],
    "pytest": ["pytest", "unit-testing", "python"],
    "junit": ["junit", "unit-testing", "java"],
    "mocha": ["mocha", "unit-testing", "javascript"],
    "vitest": ["jest", "unit-testing", "javascript"],
    "unit test": ["unit-testing"],
    "integration test": ["integration-testing"],
    "end-to-end test": ["end-to-end-testing"],
    "e2e test": ["end-to-end-testing"],
    "load testing": ["load-testing", "performance-testing"],
    "performance test": ["performance-testing"],
    "stress test": ["performance-testing"],
    "chaos engineering": ["chaos-engineering"],
    "test driven": ["tdd"],
    "test-driven": ["tdd"],
    "mocking": ["mocking", "unit-testing"],

    # ── Blockchain ───────────────────────────────────────────────
    "ethereum": ["ethereum", "blockchain"],
    "solidity": ["solidity", "blockchain", "ethereum"],
    "smart contract": ["smart-contracts", "blockchain"],
    "smart contracts": ["smart-contracts", "blockchain"],
    "defi": ["defi", "blockchain"],
    "decentralized finance": ["defi", "blockchain"],
    "nft": ["nft", "blockchain"],
    "web3": ["web3", "blockchain"],
    "web 3": ["web3", "blockchain"],
    "cryptocurrency": ["cryptocurrency", "blockchain"],
    "bitcoin": ["bitcoin", "blockchain"],
    "solana": ["solana", "blockchain"],
    "polygon": ["polygon", "blockchain"],
    "hardhat": ["hardhat", "blockchain", "ethereum"],
    "truffle": ["truffle", "blockchain", "ethereum"],
    "ipfs": ["ipfs", "blockchain"],
    "dao": ["dao", "blockchain"],

    # ── Game Dev ─────────────────────────────────────────────────
    "unity": ["unity", "game-development", "csharp"],
    "unreal engine": ["unreal-engine", "game-development"],
    "godot": ["godot", "game-development"],
    "game engine": ["game-engine", "game-development"],
    "game dev": ["game-development"],
    "raylib": ["raylib", "game-development"],
    "pygame": ["pygame", "game-development", "python"],
    "love2d": ["love2d", "game-development", "lua"],
    "opengl": ["opengl", "graphics"],
    "vulkan": ["vulkan", "graphics"],
    "directx": ["directx", "graphics"],
    "shader": ["shader", "graphics"],
    "ray tracing": ["ray-tracing", "graphics"],
    "sprite": ["game-development", "2d-game"],
    "2d game": ["game-development", "2d-game"],
    "3d game": ["game-development", "3d-game"],
    "roguelike": ["game-development"],

    # ── Embedded / Systems ───────────────────────────────────────
    "embedded": ["embedded", "systems-programming"],
    "iot": ["iot", "embedded"],
    "internet of things": ["iot", "embedded"],
    "arduino": ["arduino", "embedded"],
    "raspberry pi": ["raspberry-pi", "embedded"],
    "esp32": ["esp32", "embedded"],
    "esp8266": ["esp32", "embedded"],
    "rtos": ["rtos", "embedded"],
    "firmware": ["firmware", "embedded"],
    "device driver": ["device-driver", "embedded"],
    "linux kernel": ["linux-kernel", "systems-programming"],
    "operating system": ["operating-system", "systems-programming"],
    "kernel": ["linux-kernel", "systems-programming"],
    "bootloader": ["firmware", "embedded"],
    "microcontroller": ["embedded"],

    # ── Observability ────────────────────────────────────────────
    "opentelemetry": ["opentelemetry", "observability"],
    "open telemetry": ["opentelemetry", "observability"],
    "distributed tracing": ["tracing", "observability"],
    "log aggregation": ["logging", "observability"],
    "apm": ["apm", "observability"],
    "application performance": ["apm", "observability"],
    "alerting": ["alerting", "monitoring"],
    "dashboard": ["dashboard", "monitoring"],

    # ── CLI / Terminal ───────────────────────────────────────────
    "command line": ["cli", "terminal"],
    "command-line": ["cli", "terminal"],
    "cli tool": ["cli"],
    "terminal emulator": ["terminal"],
    "neovim": ["neovim", "vim"],
    "vim plugin": ["vim"],
    "emacs": ["emacs"],
    "tmux": ["tmux", "terminal"],
    "shell script": ["shell", "bash"],
    "bash script": ["bash", "shell"],
    "zsh plugin": ["zsh", "shell"],
    "fish shell": ["fish", "shell"],

    # ── Build / Package ──────────────────────────────────────────
    "monorepo": ["monorepo"],
    "package manager": ["package-manager"],
    "bundler": ["bundler", "build-tool"],
    "linter": ["linting"],
    "code formatter": ["formatting"],
    "code generation": ["code-generation"],
    "static analysis": ["static-analysis"],
    "type checker": ["type-checker"],

    # ── Media / Content ──────────────────────────────────────────
    "image processing": ["image-processing"],
    "video processing": ["video-processing"],
    "video editing": ["video-processing"],
    "audio processing": ["audio-processing"],
    "music": ["audio-processing"],
    "pdf": ["pdf"],
    "markdown": ["markdown"],
    "content management": ["cms"],
    "headless cms": ["cms"],
    "e-commerce": ["e-commerce"],
    "ecommerce": ["e-commerce"],
    "payment": ["payment"],
    "stripe": ["stripe", "payment"],
    "paypal": ["payment"],

    # ── Other ────────────────────────────────────────────────────
    "analytics": ["analytics"],
    "a/b testing": ["a-b-testing"],
    "ab testing": ["a-b-testing"],
    "feature flag": ["feature-flags"],
    "feature toggle": ["feature-flags"],
    "workflow": ["workflow", "automation"],
    "task scheduler": ["task-scheduling"],
    "job queue": ["message-queue"],
    "message queue": ["message-queue"],
    "rabbitmq": ["rabbitmq", "message-queue"],
    "nats": ["nats", "message-queue"],
    "kafka": ["apache-kafka", "message-queue"],
    "celery": ["message-queue", "python"],
    "full-text search": ["full-text-search"],
    "web scraper": ["web-scraper", "crawling"],
    "web scraping": ["web-scraper", "crawling"],
    "crawler": ["crawling", "web-scraper"],
    "bot": ["bot"],
    "discord bot": ["bot", "discord"],
    "telegram bot": ["bot", "telegram"],
    "slack bot": ["bot", "slack"],
    "geospatial": ["geospatial", "gis"],
    "gis": ["gis", "geospatial"],
    "map": ["geospatial", "gis"],
    "robotics": ["robotics"],
    "bioinformatics": ["bioinformatics"],
    "scientific computing": ["scientific-computing"],
    "numerical": ["scientific-computing", "math"],
    "statistics": ["statistics", "math"],
    "localization": ["localization", "i18n"],
    "internationalization": ["i18n", "localization"],
    "i18n": ["i18n", "localization"],
    "l10n": ["localization"],
    "edge computing": ["edge-computing"],
    "deno": ["deno", "typescript"],
    "bun": ["bun", "javascript"],
    "node.js": ["nodejs", "javascript"],
    "nodejs": ["nodejs", "javascript"],
    "opengl es": ["opengl", "graphics"],
    "regex": ["regex"],
    "regular expression": ["regex"],
    "template engine": ["template-engine"],
    "data pipeline": ["data-pipeline"],
    "etl": ["etl", "data-pipeline"],
    "data visualization": ["visualization", "analytics"],
    "chart": ["visualization"],
    "awesome list": ["awesome-list"],

    # ── Languages (for language-specific tools) ──────────────────
    "python library": ["python"],
    "python package": ["python"],
    "python module": ["python"],
    "javascript library": ["javascript"],
    "typescript library": ["typescript"],
    "rust crate": ["rust"],
    "go module": ["go"],
    "golang": ["go"],
    "ruby gem": ["ruby"],
    "java library": ["java"],
    "c# library": ["csharp"],
    ".net": ["csharp", "dotnet"],
    "dotnet": ["csharp", "dotnet"],
}

# Pre-compute phrase keys (multi-word) sorted longest-first for
# greedy matching. Single-word keys are handled separately.
_PHRASE_KEYS: list[str] = sorted(
    [k for k in _KEYWORD_TOPICS if " " in k],
    key=lambda k: -len(k),
)
_WORD_KEYS: set[str] = {k for k in _KEYWORD_TOPICS if " " not in k}


def infer_topics(
    text: str,
    existing_topics: list[str] | None = None,
    *,
    max_topics: int = MAX_INFERRED,
) -> list[str]:
    """Infer GitHub-style topic tags from README/description text.

    Args:
        text: The combined README + description text to scan.
        existing_topics: Topics the repo already has (won't be duplicated).
        max_topics: Maximum number of new topics to return.

    Returns:
        List of inferred topic strings (not in existing_topics).
        Empty list if nothing confident can be inferred.
    """
    if not text or len(text) < 20:
        return []

    existing = {t.lower() for t in (existing_topics or [])}
    lower_text = text.lower()
    tokens = _tokenize(lower_text)
    token_set = set(tokens)

    topic_scores: Counter[str] = Counter()

    # Phase 1: match phrases (bigrams) first — higher specificity
    for phrase in _PHRASE_KEYS:
        if phrase in lower_text:
            for topic in _KEYWORD_TOPICS[phrase]:
                if topic not in existing:
                    topic_scores[topic] += 3  # phrase matches are high confidence

    # Phase 2: match single words — needs repetition or specificity
    token_counts = Counter(tokens)
    for word in _WORD_KEYS:
        # For keywords with dots/hyphens (e.g. "next.js", "react-native"),
        # check raw text presence since the tokenizer splits them.
        if "." in word or "-" in word:
            matched = word in lower_text
            count = lower_text.count(word)
        elif len(word) <= 3:
            # Short words (go, r, c, db, css): exact token match only
            # to avoid substring false positives (e.g. "db" in "dashboard")
            matched = word in token_set
            count = token_counts.get(word, 0)
        else:
            # Longer words: token match is primary, substring as fallback
            # for words inside hyphenated compounds (e.g. "fastapi" in "fastapi-middleware")
            matched = word in token_set
            count = token_counts.get(word, 0)
            if not matched and word in lower_text:
                matched = True
                count = lower_text.count(word)

        if matched:
            for topic in _KEYWORD_TOPICS[word]:
                if topic not in existing:
                    topic_scores[topic] += 1 + min(count - 1, 3)

    # Phase 3: boost topics that appear from multiple evidence sources
    # (e.g., "pytorch" keyword + "deep learning" phrase both suggest
    # deep-learning, so it gets extra confidence)
    source_count: Counter[str] = Counter()
    for topic in topic_scores:
        for k, topics in _KEYWORD_TOPICS.items():
            if topic not in topics:
                continue
            if k in _PHRASE_KEYS and k in lower_text:
                source_count[topic] += 1
            elif k in _WORD_KEYS:
                if "." in k or "-" in k:
                    if k in lower_text:
                        source_count[topic] += 1
                elif len(k) <= 3:
                    if k in token_set:
                        source_count[topic] += 1
                else:
                    if k in token_set or k in lower_text:
                        source_count[topic] += 1
    for topic, sources in source_count.items():
        if sources >= 2:
            topic_scores[topic] += sources

    # Sort by score descending, then alphabetically for determinism
    ranked = sorted(
        topic_scores.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )

    # Filter: score >= 1. The vocabulary is curated so every keyword
    # is a technical term that should map to its topics. Phrase matches
    # (score >= 3) and multi-source topics naturally rank above singles.
    inferred = [topic for topic, score in ranked if score >= 1]

    return inferred[:max_topics]


def infer_topics_for_repo(
    repo_description: str | None,
    repo_readme: str | None,
    existing_topics: list[str] | None = None,
    *,
    max_topics: int = MAX_INFERRED,
) -> list[str]:
    """Convenience wrapper for repo data. Combines description + README."""
    parts: list[str] = []
    if repo_description:
        parts.append(repo_description)
    if repo_readme:
        # First 4000 chars are most signal-dense (headers, intro)
        parts.append(repo_readme[:4000])
    combined = "\n".join(parts)
    return infer_topics(combined, existing_topics, max_topics=max_topics)
