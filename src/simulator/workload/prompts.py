"""Human-plausible request generation.

The previous filler was the words "system latency throughput scheduler"
repeated to a target length. That is fine for counting tokens and useless for
anything else: it has no realistic vocabulary distribution, no structure, and
no shared prefixes of the kind real traffic has. Tokenizer behaviour, prefix
cache hit patterns and output-length behaviour all depend on prompts that look
like things people actually send.

So requests are composed from templates and slot vocabularies across the
categories a general chat endpoint really sees, with per-category length
profiles: `summarize` carries a long document and wants a short answer,
`creative` is the reverse, `code_debug` carries a code block, `rag` carries
retrieved passages behind a shared system prompt.

Everything is deterministic given a seed. Nothing is sampled from the network,
so trace construction stays offline and reproducible.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

CHARS_PER_TOKEN = 4.0   # English text, close enough for sizing


# ── slot vocabularies ────────────────────────────────────────────
_TOPICS = """distributed systems kubernetes rate limiting database indexing
caching strategies message queues load balancing API design microservices
observability incident response capacity planning data modelling schema
migration authentication authorization feature flags CI pipelines
infrastructure as code cost optimisation on-call rotations postmortems""".split("\n")
_TOPICS = [t.strip() for line in _TOPICS for t in line.split("  ") if t.strip()]
_TOPICS = ["distributed systems", "kubernetes", "rate limiting", "database indexing",
           "caching strategies", "message queues", "load balancing", "API design",
           "microservices", "observability", "incident response", "capacity planning",
           "data modelling", "schema migration", "authentication", "feature flags",
           "CI pipelines", "cost optimisation", "on-call rotations", "postmortems",
           "vector databases", "streaming ingestion", "batch scheduling",
           "retry policies", "circuit breakers", "blue-green deploys"]

_LANGS = ["Python", "TypeScript", "Go", "Rust", "Java", "Ruby", "C++", "SQL", "Bash"]
_TASKS = ["parses a CSV file", "retries on transient failures", "merges two sorted lists",
          "validates an email address", "debounces a callback", "paginates an API response",
          "caches results in memory with a TTL", "converts a nested dict to flat keys",
          "computes a rolling average", "deduplicates records by key",
          "backs off exponentially", "streams a large file line by line"]
# Kept language-agnostic: "segfault" in a SQL question reads as nonsense.
_ERRORS = ["wrong results on the second page", "an off-by-one in the pagination",
           "a deadlock when two workers collide", "memory growth after a few hours",
           "intermittent timeouts", "wrong results when the input is empty",
           "a race condition under concurrency", "duplicate rows in the output",
           "results that are correct locally but wrong in production"]
_DOCTYPES = ["quarterly report", "meeting transcript", "research paper", "support ticket thread",
             "product spec", "incident postmortem", "customer interview", "legal contract"]
_TONES = ["concise", "detailed", "beginner-friendly", "technical", "step by step"]
_TARGET_LANGS = ["French", "German", "Japanese", "Spanish", "Portuguese", "Korean"]
_ROLES = ["a senior engineer", "a product manager", "a data scientist",
          "a technical writer", "a site reliability engineer"]
_PERSONAS = ["a curious beginner", "a busy executive", "a sceptical reviewer"]

# Shared system prompts. Real deployments put a stable preamble in front of
# every request, which is exactly what a prefix cache exists to exploit.
SYSTEM_PROMPTS = [
    ("assistant_v1",
     "You are a helpful, careful assistant. Answer accurately and concisely. "
     "If you are unsure, say so rather than guessing. Prefer concrete examples "
     "over abstract description, and keep formatting simple."),
    ("support_v2",
     "You are a customer support agent for a cloud infrastructure company. Be "
     "polite and practical. Ask a clarifying question when the request is "
     "ambiguous. Never promise refunds or timelines you cannot verify. Escalate "
     "billing disputes to a human."),
    ("code_reviewer",
     "You are an experienced code reviewer. Point out correctness bugs first, "
     "then performance, then style. Quote the specific line you mean. Do not "
     "restate what the code does unless it is unclear."),
]

_PROSE = [
    "The team reviewed the rollout plan and agreed the migration should proceed in stages.",
    "Latency rose sharply during the evening peak, though error rates stayed flat.",
    "Several customers reported that the export job silently produced empty files.",
    "We considered sharding by tenant, but the largest tenant alone exceeds one node.",
    "The postmortem identified three contributing factors and one root cause.",
    "Adoption has been slower than forecast, mostly in the self-serve segment.",
    "Throughput improved after the batch size was raised, at the cost of tail latency.",
    "The new schema removes two joins from the hot path and adds a nullable column.",
    "Support volume doubled in the week following the pricing change.",
    "Nobody could reproduce the issue locally until we matched the production timezone.",
]

# Keyed by language so a question about TypeScript does not contain SQL.
_CODE: dict[str, list[str]] = {
    "Python": [
        "def fetch(url, retries=3):\n    for i in range(retries):\n        r = get(url)\n        if r.ok:\n            return r.json()",
        "async def worker(q):\n    while True:\n        item = await q.get()\n        process(item)\n        q.task_done()",
        "class Cache:\n    def __init__(self, ttl):\n        self.ttl = ttl\n        self.data = {}\n\n    def get(self, k):\n        return self.data.get(k)",
    ],
    "TypeScript": [
        "export async function fetchAll(ids: string[]) {\n  const out = [];\n  for (const id of ids) {\n    out.push(await fetch(`/api/${id}`).then(r => r.json()));\n  }\n  return out;\n}",
        "const debounce = (fn: Function, ms: number) => {\n  let t: any;\n  return (...a: any[]) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };\n};",
        "useEffect(() => {\n  const sub = source.subscribe(setValue);\n}, [source]);",
    ],
    "Go": [
        "func worker(jobs <-chan Job, wg *sync.WaitGroup) {\n\tfor j := range jobs {\n\t\tprocess(j)\n\t}\n\twg.Done()\n}",
        "for _, row := range rows {\n\tif _, ok := seen[row.ID]; ok {\n\t\tcontinue\n\t}\n\tseen[row.ID] = struct{}{}\n}",
    ],
    "SQL": [
        "SELECT u.id, count(*)\nFROM users u\nJOIN events e ON e.user_id = u.id\nGROUP BY u.id\nHAVING count(*) > 10;",
        "UPDATE orders SET status = 'shipped'\nWHERE created_at < now() - interval '7 days';",
    ],
    "Rust": [
        "fn merge(a: Vec<i32>, b: Vec<i32>) -> Vec<i32> {\n    let mut out = a;\n    out.extend(b);\n    out.sort();\n    out\n}",
    ],
}
_CODE_LANGS = list(_CODE)


@dataclass(frozen=True)
class Category:
    name: str
    weight: float          # share of a realistic mixed stream
    in_mu: float           # lognormal mu for input tokens
    in_sigma: float
    out_mu: float          # lognormal mu for output tokens
    out_sigma: float


# Length profiles differ by intent, which is the point: a single global length
# distribution would erase the prefill/decode asymmetry that serving cares about.
CATEGORIES: tuple[Category, ...] = (
    Category("chat",        0.28, 3.4, 0.6, 4.6, 0.6),   # ~30 in, ~100 out
    Category("code_gen",    0.14, 4.2, 0.5, 5.7, 0.5),   # ~67 in, ~300 out
    Category("code_debug",  0.10, 6.2, 0.7, 5.3, 0.5),   # ~490 in, ~200 out
    Category("summarize",   0.12, 7.6, 0.6, 4.5, 0.4),   # ~2000 in, ~90 out
    Category("explain",     0.13, 3.6, 0.5, 5.5, 0.5),   # ~37 in, ~245 out
    Category("creative",    0.06, 3.5, 0.5, 6.4, 0.5),   # ~33 in, ~600 out
    Category("analysis",    0.07, 6.6, 0.7, 5.6, 0.5),   # ~735 in, ~270 out
    Category("translate",   0.04, 5.4, 0.7, 5.4, 0.6),   # ~220 in, ~220 out
    Category("math",        0.03, 4.0, 0.5, 5.2, 0.5),   # ~55 in, ~180 out
    Category("rag",         0.03, 7.2, 0.5, 5.0, 0.4),   # ~1340 in, ~150 out
)

_BY_NAME = {c.name: c for c in CATEGORIES}


# Categories where a human genuinely controls the length by pasting material.
# Padding these is realistic; padding a one-line chat question is not.
LONG_FORM = {"code_debug", "summarize", "analysis", "translate", "rag"}


def _pad_to(text: str, target_tokens: int, rng: random.Random,
            pool: list[str] | None = None, sep: str = " ") -> str:
    """Extend `text` with plausible material until it reaches ~target_tokens."""
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    pool = pool if pool is not None else _PROSE
    parts = [text] if text else []
    size = len(text)
    # Cycle a reshuffled deck rather than sampling with replacement. Immediate
    # repeats read as machine-generated, and identical repeated spans would also
    # give the prefix cache an unrealistically easy time.
    deck: list[str] = []
    while size < target_chars:
        if not deck:
            deck = list(pool)
            rng.shuffle(deck)
        chunk = deck.pop()
        parts.append(chunk)
        size += len(chunk) + len(sep)
    return sep.join(parts)[:max(target_chars, len(text))]


def make_request(rng: random.Random, category: str, target_in_tokens: int) -> str:
    """One human-plausible request of roughly the requested token length."""
    t = rng.choice(_TOPICS)
    lang = rng.choice(_LANGS)
    tone = rng.choice(_TONES)

    if category == "chat":
        head = rng.choice([
            f"What's the difference between {t} and {rng.choice(_TOPICS)}?",
            f"Is it worth learning {t} in 2026?",
            f"Quick question about {t} — where do people usually get it wrong?",
            f"Can you give me a {tone} overview of {t}?",
            f"I'm {rng.choice(_PERSONAS)}. How would you explain {t} to me?",
        ])
    elif category == "code_gen":
        head = rng.choice([
            f"Write a {lang} function that {rng.choice(_TASKS)}. Keep it {tone}.",
            f"In {lang}, what's the cleanest way to write something that {rng.choice(_TASKS)}?",
            f"Show me a {lang} example that {rng.choice(_TASKS)}, with a short test.",
        ])
    elif category == "code_debug":
        # Pick the language from the snippet pool so the code matches the claim.
        clang = rng.choice(_CODE_LANGS)
        body = _pad_to("", max(20, target_in_tokens - 40), rng,
                       pool=_CODE[clang], sep="\n\n")
        return (f"This {clang} code is giving me {rng.choice(_ERRORS)}. "
                f"What's wrong with it?\n\n```{clang.lower()}\n{body}\n```")
    elif category == "summarize":
        head = (f"Summarize the following {rng.choice(_DOCTYPES)} for "
                f"{rng.choice(_ROLES)}. Be {tone}.\n\n"
                + _pad_to("", max(30, target_in_tokens - 30), rng))
        return head
    elif category == "explain":
        head = rng.choice([
            f"Explain {t} like I'm five.",
            f"Why does {t} matter? Keep it {tone}.",
            f"What are the tradeoffs in {t}?",
            f"Walk me through how {t} actually works under the hood.",
        ])
    elif category == "creative":
        head = rng.choice([
            f"Write a short story where the main character works on {t}.",
            f"Write a light-hearted poem about {t}.",
            f"Draft a blog post introduction about {t} for {rng.choice(_PERSONAS)}.",
        ])
    elif category == "analysis":
        head = (f"Here are some notes from a {rng.choice(_DOCTYPES)}. What patterns "
                f"do you see, and what would you do next?\n\n"
                + _pad_to("", max(30, target_in_tokens - 35), rng))
        return head
    elif category == "translate":
        head = (f"Translate the following into {rng.choice(_TARGET_LANGS)}, keeping "
                f"the tone natural:\n\n"
                + _pad_to("", max(20, target_in_tokens - 25), rng))
        return head
    elif category == "math":
        a, b, c = rng.randint(12, 400), rng.randint(3, 40), rng.randint(2, 25)
        head = rng.choice([
            f"A service handles {a} requests per second and each takes {b} ms. "
            f"How many are in flight on average? Show your working.",
            f"If we have {a} nodes and {b} replicas each, and {c}% fail, how many "
            f"replicas survive? Explain step by step.",
        ])
    elif category == "rag":
        head = ("Answer the question using only the passages below.\n\n"
                "<passages>\n"
                + _pad_to("", max(40, target_in_tokens - 60), rng)
                + f"\n</passages>\n\nQuestion: what does this imply for {t}?")
        return head
    else:
        head = f"Tell me about {t}."

    # Short-form categories are returned at their natural length. A real person
    # asking "explain X like I'm five" does not append 400 tokens of filler to
    # hit a token budget, and pretending otherwise makes the corpus less
    # realistic, not more.
    return head


def sample_category(rng: random.Random, names: tuple[str, ...] | None = None) -> Category:
    """Weighted draw. Weights are renormalised over `names` when a subset is given.

    Uniform sampling would over-represent rare intents: creative writing is 6%
    of a real stream but would become 11% of a nine-category uniform draw, and
    it has by far the longest outputs, so the decode load would be wrong.
    """
    pool = [_BY_NAME[n] for n in names] if names else list(CATEGORIES)
    total = sum(c.weight for c in pool)
    r = rng.random() * total
    acc = 0.0
    for c in pool:
        acc += c.weight
        if r <= acc:
            return c
    return pool[-1]


ALL_CATEGORIES: tuple[str, ...] = tuple(c.name for c in CATEGORIES)


def category(name: str) -> Category:
    return _BY_NAME[name]
