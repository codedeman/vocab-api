# Vocabulary Classification API

A FastAPI backend that accepts English text and returns structured vocabulary items
classified against a fixed topic taxonomy.

Two providers ship: **`openai_compatible`** — real LLM output from Gemini, DeepSeek,
OpenAI, Groq, Together, OpenRouter, Ollama or vLLM, one adapter for all of them — and
**`mock`**, a deterministic offline stand-in whose meanings are fake and prefixed
`(mock)`. You must choose one explicitly; see [Configuration](#configuration).

Deployable to [Render's free tier](#deployment-render-free), gated by a single
[`X-API-Key` header](#authentication).

Not in scope: Google Sheets, the Chrome extension, user accounts.

## Setup

Requires Python 3.10+ (developed and tested on 3.11.9; deployments pin 3.12 — see
[`.python-version`](.python-version)). 3.10 is the floor because the code annotates with
`X | None`.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

`requirements.txt` is runtime only, so a deployment does not install `pytest`;
`requirements-dev.txt` pulls it in on top for local work.

## Run the API

```bash
.venv/bin/uvicorn app.main:app --reload --port 8000
```

With no `.env` this refuses to start, and says which variable is missing — that is
intentional for both `VOCAB_CLASSIFIER` and `VOCAB_ACCESS_KEY`. To run the offline
stand-in with no key and no gate:

```bash
VOCAB_CLASSIFIER=mock VOCAB_ACCESS_KEY=disabled .venv/bin/uvicorn app.main:app --reload --port 8000
```

Interactive docs at <http://localhost:8000/docs>.

## Run the tests

```bash
.venv/bin/python -m pytest -q
```

## Endpoints

### `GET /health`

Never gated — see [Authentication](#authentication).

```bash
curl -s localhost:8000/health
```

```json
{ "status": "ok", "classifier": "mock" }
```

### `POST /api/v1/vocabulary/extract`

Needs `X-API-Key` unless `VOCAB_ACCESS_KEY=disabled`.

```bash
curl -s -X POST localhost:8000/api/v1/vocabulary/extract \
  -H "X-API-Key: $VOCAB_ACCESS_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"text":"The accountant reviewed every invoice before we deploy the new billing service."}'
```

Response — copied from a real run of that exact command **against `mock`**, showing 2 of
the 7 items returned. The `(mock)` strings are exactly what the offline stand-in produces;
with `openai_compatible` those fields carry real content:

```json
{
  "items": [
    {
      "word": "invoice",
      "part_of_speech": "noun",
      "meaning_vi": "hóa đơn",
      "cefr_level": "B2",
      "main_category": "Finance",
      "sub_category": "Banking & Payments",
      "example": "The accountant reviewed every invoice before we deploy the new billing service.",
      "word_family": [
        { "word": "invoices", "part_of_speech": "noun" }
      ],
      "examples": [
        "(mock) We discussed invoice during the meeting.",
        "(mock) This sentence shows how invoice is used in context."
      ],
      "definition_en": "(mock) English explanation of 'invoice'.",
      "usage_vi": "(mock) cách dùng và bối cảnh của 'invoice'."
    },
    {
      "word": "deploy",
      "part_of_speech": "verb",
      "meaning_vi": "triển khai",
      "cefr_level": "B2",
      "main_category": "Technical & IT",
      "sub_category": "Software Development",
      "example": "The accountant reviewed every invoice before we deploy the new billing service.",
      "word_family": [
        { "word": "deployment", "part_of_speech": "noun" },
        { "word": "deploys", "part_of_speech": "verb" },
        { "word": "deployed", "part_of_speech": "verb" },
        { "word": "deploying", "part_of_speech": "verb" },
        { "word": "redeploy", "part_of_speech": "verb" }
      ],
      "examples": [
        "(mock) We discussed deploy during the meeting.",
        "(mock) This sentence shows how deploy is used in context."
      ],
      "definition_en": "To put software onto the servers where people actually use it, making a new version live.",
      "usage_vi": "Là ngoại động từ, đi với tân ngữ là thứ được đưa lên (deploy the service) và giới từ 'to' cho môi trường đích (deploy to production). Danh từ tương ứng là 'deployment'. Rất phổ biến trong ngành phần mềm; ngoài IT từ này còn mang nghĩa điều động quân đội hoặc nhân lực."
    }
  ],
  "count": 7,
  "classifier": "mock"
}
```

**Request:** `{"text": "..."}` — 1 to 5000 characters. Empty, missing, or whitespace-only
text returns `422`.

**Top level:** `items`, `count`, and `classifier` — the provider that produced the items.

**Each item** carries eleven fields. They stay separate because they are independent
dimensions the extension filters and displays on their own:

| Field | Type | Meaning |
|---|---|---|
| `word` | string | The headword, lowercased |
| `part_of_speech` | string | Short label: `noun`, `verb`, `adjective`, `adverb`, … |
| `meaning_vi` | string | Vietnamese meaning |
| `cefr_level` | string | One of `A1 A2 B1 B2 C1 C2` |
| `main_category` | enum | One of the ten taxonomy values below |
| `sub_category` | string | Topic label, 1–60 chars |
| `example` | string | **A real sentence from the text you submitted** |
| `word_family` | list of `{word, part_of_speech}` | Related forms of the same root, ≤ 8; `[]` when the word has no common derivations |
| `examples` | list of strings | **Generated** sentences showing the word in use, ≤ 5 — not taken from your text |
| `definition_en` | string | What the word means, explained in simple English |
| `usage_vi` | string | One Vietnamese paragraph: how to use the word and in what context |

`example` and `examples` are both kept on purpose: the first proves the word really appeared
in your source text, the second teaches usage beyond that one occurrence.

## Taxonomy

`main_category` is a closed enum — exactly these ten values, published in the OpenAPI schema
so the extension can build filters from it:

Business · Working & Career · Technical & IT · Daily Life & Communication ·
Learning & Personal Development · Finance · Health & Wellbeing · Society & Politics ·
Science & Environment · Other

`sub_category` is a free string (1–60 chars). The useful set differs per main category, so
it is not locked to an enum yet. Items the classifier can't place get
`Other` / `Uncategorized`.

`cefr_level` is one of `A1 A2 B1 B2 C1 C2`.

## Configuration

Copy [.env.example](.env.example) to `.env` and fill in your key:

```bash
cp .env.example .env
```

`app/config.py` reads `.env` at the project root. Real environment variables win over the
file, so `VOCAB_CLASSIFIER=mock uvicorn ...` still overrides a Gemini `.env`. `.env` is
gitignored.

| Variable | Default | Purpose |
|---|---|---|
| `VOCAB_CLASSIFIER` | `openai_compatible` if `LLM_API_KEY` is recognised, else **required** | `openai_compatible` or `mock` |
| `VOCAB_ACCESS_KEY` | **none — required** | Secret callers must send as `X-API-Key`, or `disabled` to serve openly. See [Authentication](#authentication) |
| `DATABASE_URL` | unset | Postgres for the [word cache](#word-cache); unset = cache off |
| `LLM_API_KEY` | unset | Key for whichever host — one generic variable, not per-vendor. Its prefix picks the provider (see below) |
| `LLM_BASE_URL` | from the key | Override the endpoint, e.g. `https://api.deepseek.com/v1` |
| `LLM_MODEL` | from the key | Override the model, e.g. `gpt-4.1-mini` |
| `LLM_REASONING_EFFORT` | from the key | Thinking budget. `low` for Gemini automatically; set it yourself only for another thinking model |

**Switching provider is pasting a different `LLM_API_KEY`.** The key's prefix says which
host it belongs to ([app/providers.py](app/providers.py)), and the rest is filled in:

| Key starts with | Provider | Model | Reasoning effort |
|---|---|---|---|
| `AIza` or `AQ.` | Gemini | `gemini-3.8-flash` | `low` |
| `sk-or-` | OpenRouter | `openai/gpt-4o-mini` | — |
| `gsk_` | Groq | `llama-3.3-70b-versatile` | — |
| `sk-` | OpenAI | `gpt-4o-mini` | — |
| `sk-ant-` | refused at startup: Anthropic has no adapter here | | |

Anything you set explicitly wins. The model and reasoning effort are only filled in when
`LLM_BASE_URL` is too — a custom endpoint gets neither guessed for it. If the key clearly
belongs to a different known host than `LLM_BASE_URL` — an OpenAI key with Gemini's URL
left behind from before — the key wins: the stale `LLM_BASE_URL`, `LLM_MODEL` and
`LLM_REASONING_EFFORT` are ignored, and the startup log names them in a warning so you can
delete them. An unknown host (Ollama, a proxy) is always taken as written. DeepSeek keys also start with
`sk-`, so DeepSeek needs `LLM_BASE_URL` and `LLM_MODEL` set; that pairing is allowed.

**Neither required variable falls back to anything you didn't ask for.** Leave
`VOCAB_CLASSIFIER` unset with no recognisable `LLM_API_KEY` and the server refuses to start,
naming what to set — it never quietly picks the mock; an unregistered value fails the same way, with the
list of available providers. `VOCAB_ACCESS_KEY` behaves identically. Both failures happen at
startup, not on the first request.

The two keys are different things and the names say so: `LLM_API_KEY` is **this service's**
key for the provider, `VOCAB_ACCESS_KEY` is what **callers** need to use this service.

Every response and `GET /health` report the active provider, so you never have to guess
whether you are looking at real data:

```json
{ "status": "ok", "classifier": "openai_compatible" }
```

### Running against Gemini

```bash
LLM_API_KEY=            # get one at https://aistudio.google.com/apikey
```

That alone gives you the endpoint, `gemini-3.8-flash` and `LLM_REASONING_EFFORT=low`.
**The `low` budget is not optional here** — if you override `LLM_BASE_URL` for Gemini, set
it yourself. `gemini-3.8-flash` is a thinking
model, and without a budget it reasons about a 20-word extraction until it runs out of
one. Measured on the same paragraph: **~28s for 15 items with `low`, and not finished
after 600s without it.** No timeout setting fixes that; the request never completes.

Gemini's OpenAI-compatible endpoint speaks the same `/chat/completions` format, takes
`Authorization: Bearer`, and maps `response_format: {"type":"json_object"}` to its own
`responseMimeType: application/json` — so it needs no adapter of its own.

Two things to know: `gemini-3.8-flash` has a free tier, and **on the free tier Google uses
the content you send to improve their products**, so don't paste anything confidential
while on it. Model IDs and prices change often; a 404 on the model usually means the ID
moved, not that the code broke.

### Why am I getting mock data?

Before this was fixed, three things combined to serve fake data from a service that looked
healthy: `VOCAB_CLASSIFIER` defaulted to `mock`, `config.py` ignored `.env` entirely (so a
correctly written `.env` changed nothing), and the response had no field saying which
provider produced it. All three are gone — a missing provider is now a startup failure,
`.env` is read, and `classifier` is in every response. If you still see `(mock)` in the
output, `VOCAB_CLASSIFIER=mock` is set somewhere: check your shell environment first, since
it overrides `.env`.

### Why am I getting a 502?

A 502 means the provider rejected or failed the request — this service is fine, the
upstream is not. The body carries the provider's own words, so read `detail` first:

```json
{ "detail": "provider returned 404 for model gemini-2.0-flash: This model models/gemini-2.0-flash is no longer available. Please update your code to use models/gemini-3.8-flash..." }
```

The common causes, in order of how often they bite:

| `detail` says | Cause | Fix |
|---|---|---|
| `404 ... no longer available` | Google retired that model ID | Use the replacement the message names |
| `401` / `403` | Key is wrong, or the API is not enabled for it | Re-issue the key |
| `429` | Free-tier rate limit (~10 req/min) | Wait; there is no backoff in this build |
| `read operation timed out` | Thinking model with no budget | Set `LLM_REASONING_EFFORT=low` |

`404`, `401`, `400` and `403` are raised immediately — a second identical request cannot
un-retire a model or fix a bad key, so retrying would only double the wait. `429`, `408`
and `5xx` are retried once. A provider failure used to surface as a bare
`500 Internal Server Error`, which threw the provider's actionable message away.

## Authentication

One shared secret in an `X-API-Key` header, compared in constant time
([`app/security.py`](app/security.py)). No users, no sessions, no new dependency.

It exists because deploying publishes a URL backed by a real `LLM_API_KEY`. Without a gate,
anyone who finds the URL spends your provider quota, and on a free tier the text they send
is fed to the provider's training too.

`VOCAB_ACCESS_KEY` has three states:

| Value | Behaviour |
|---|---|
| unset | **Server refuses to start**, naming the variable |
| `disabled` | Open to everyone, with a `WARNING` logged at startup |
| anything else | `POST /api/v1/...` requires `X-API-Key` with that exact value |

"Unset" is a startup failure rather than "open" on purpose: the lesson of
`VOCAB_CLASSIFIER` defaulting to `mock` is that a silent default looks healthy while doing
the wrong thing. Forgetting the key on a deploy should break loudly, not publish an open
endpoint.

The startup line reports `access_control=on` or `access_control=OFF` — enough to spot an
accidentally open deployment in the logs, and neither key is ever printed.

```
INFO:     vocab-api: classifier=openai_compatible model=gemini-3.8-flash base_url=https://generativelanguage.googleapis.com/v1beta/openai access_control=on
```

**`GET /health` is deliberately never gated.** Render calls it as its health check, so
gating it would fail every deploy; it touches no provider, costs no quota, and reveals only
the configured provider name.

### Why am I getting a 401?

```json
{ "detail": "Missing or invalid X-API-Key header." }
```

The header is missing or does not match `VOCAB_ACCESS_KEY`. On Render, read the current
value from the service's **Environment** tab — Render generated it, so it was never typed
anywhere. The message is deliberately vague and identical for "missing" and "wrong": it
never echoes the expected value.

Two limits worth stating plainly: a shared secret in a header is **meaningless over plain
HTTP** (Render terminates TLS, so a `https://…onrender.com` URL is fine — don't call it
over `http://` anywhere else), and the gate is **not rate limiting**. Anyone holding the key
can call as often as they like; the real ceiling is the provider's own rate limit.

## Deployment (Render free)

[`render.yaml`](render.yaml) is a Render Blueprint, so the dashboard reads the config
instead of you typing it: **New → Blueprint → pick this repo → paste `LLM_API_KEY` →
Create.** `VOCAB_ACCESS_KEY` is generated by Render (`generateValue: true`), so nobody has
to invent a secret; copy it from the service's Environment tab to call the API.

`LLM_API_KEY` is marked `sync: false`, meaning Render prompts for it at create time and it
never enters this repository.

Render's free tier needs no credit card. What it costs instead:

| Limit | Effect here |
|---|---|
| Spins down after 15 min idle | First request after that pays a **cold start (~1 min) on top of** the ~28s extraction |
| 750 instance-hours/month, per workspace | Shared across all free services in the workspace; exhausting it suspends them |
| 0.1 CPU, 512 MB | Mostly irrelevant — this workload sits waiting on the provider — but concurrent requests will drag |
| No persistent disk | Nothing to lose; the service is stateless |

Two non-obvious things that are load-bearing:

- **The start command must stay plain `uvicorn`.** Most "Render times out at 30s" reports
  trace to **gunicorn's** default `--timeout 30`, which lands right on top of a ~28s
  extraction. Plain uvicorn imposes no request timeout, and Render's proxy does not impose
  one either.
- **`.python-version` pins 3.12** because Render's current default is 3.14, and
  `uvicorn[standard]` pulls `httptools`/`uvloop`, which need compiled wheels that may not
  exist for the newest interpreter yet. Check the build log says 3.12.

## Word cache

Optional. With `DATABASE_URL` set to a Postgres connection string
([`app/cache.py`](app/cache.py)), a word looked up once is stored, and the next lookup of
it is served from the database with no LLM call. Every response says which happened in its
`cache` field:

| `cache` | Meaning |
|---|---|
| `hit` | Served from the database; the LLM was not called |
| `miss` | Generated by the LLM, then stored |
| `off` | No `DATABASE_URL` (or `VOCAB_CLASSIFIER=mock`, which is never cached because its output is fake) |
| `error` | The database failed at request time; the LLM answered instead |

What it does and does not cache:

- **A single word** (`{"text": "postpone"}`) is looked up first. A hit skips the LLM.
- **A passage always calls the LLM**, because the model decides which words to teach, so
  there is nothing to look up beforehand. Every word it returns is stored, though, so later
  single-word lookups of those words hit.
- Typing an inflected form (`postponed`) stores both that spelling and the headword the
  model returned (`postpone`), so neither misses twice.
- The `example` of a word first seen in a passage is a sentence from that passage.

Failure policy: a `DATABASE_URL` that is set but unreachable **stops the server at
startup**; a database that fails later does **not** take the API down, it just reports
`cache: "error"` and answers from the LLM.

Render's free disk is not persistent, so a file database would be wiped on every redeploy.
Use an external Postgres, e.g. [Neon](https://neon.com) (free, no card). The table is
created automatically. Rows never expire: after changing the model or the prompt, clear
stale entries with `TRUNCATE vocab_cache`.

## Swapping providers

`app/classifier/` is the only place a model ever appears:

| File | Role |
|---|---|
| `base.py` | The contract: `Classifier.extract(text) -> list[VocabularyItem]` |
| `mock.py` | Offline stand-in |
| `openai_compatible.py` | Every OpenAI-compatible host, including Gemini |
| `__init__.py` | `PROVIDERS` registry + `get_classifier()` — the swap point |

The route takes `classifier: Classifier = Depends(get_classifier)`, so adding a provider is
one new file plus one line in `PROVIDERS` — no change to `api/vocabulary.py` or `schemas.py`.

Gemini, DeepSeek, OpenAI, Groq, Together, OpenRouter, Ollama and vLLM all speak the same
OpenAI-compatible `/chat/completions` format, so **one** `openai_compatible` adapter covers
all of them. For Gemini, OpenAI, OpenRouter and Groq the key alone is enough (see
[Configuration](#configuration)); for any other host, set the endpoint and model:

```bash
# DeepSeek — its sk- keys would otherwise be read as OpenAI
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=deepseek-chat
```

Anthropic would need its own adapter because its wire format differs, so it is not
registered. No vendor SDK is needed: `httpx2` is already installed (FastAPI's test client
requires it).

OpenAI supports strict `response_format: {"type": "json_schema"}`, while DeepSeek and
Gemini support only `{"type": "json_object"}` — valid JSON guaranteed, matching schema not
guaranteed. The adapter sends `json_object` as the common denominator, validates with
`ClassifierPayload.model_validate_json()` and retries once. Because `main_category` is an
enum, a hallucinated category is rejected at the validation boundary.

`ClassifierPayload` (`items` + `count`) is what a provider must return; `ExtractResponse`
adds `classifier` on top. The split exists so the model is never asked to report which
provider it is.

## Project structure

```
vocab-api/
├── app/
│   ├── main.py              # FastAPI app, startup fail-fast, GET /health, router wiring
│   ├── config.py            # Settings from .env + env; both keys marked repr=False
│   ├── providers.py         # Key prefix -> endpoint/model/effort presets, key/URL mismatch check
│   ├── cache.py             # Postgres word cache (the only code that touches the driver)
│   ├── security.py          # X-API-Key dependency + startup check on VOCAB_ACCESS_KEY
│   ├── schemas.py           # ExtractRequest, WordFamilyMember, VocabularyItem, ClassifierPayload, ExtractResponse
│   ├── taxonomy.py          # MainCategory enum + CEFR levels
│   ├── api/vocabulary.py    # POST /api/v1/vocabulary/extract
│   └── classifier/
│       ├── base.py          # Classifier Protocol
│       ├── mock.py          # MockClassifier
│       ├── openai_compatible.py  # Gemini / DeepSeek / OpenAI / Ollama / vLLM — one adapter
│       └── __init__.py      # PROVIDERS registry + get_classifier()
├── tests/
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_extract.py
│   ├── test_access_key.py
│   ├── test_cache.py
│   ├── test_openai_compatible.py
│   ├── test_providers.py
│   └── test_classifier_registry.py
├── render.yaml              # Render Blueprint: build/start command, health check, env vars
├── .python-version          # 3.12 — Render's default is newer and breaks compiled wheels
├── requirements.txt         # runtime only
├── requirements-dev.txt     # the above + pytest
├── .env.example
└── README.md
```

## How the mock classifier works

1. Tokenizes the text, lowercases, drops function words, de-duplicates, caps at 20 items.
2. A seed lexicon of 30 words covering 9 of the 10 categories supplies real
   `part_of_speech`, `meaning_vi`, `cefr_level` and category pair. Four of those entries
   (`revenue`, `colleague`, `deploy`, plus the fallbacks below) also carry hand-written
   `word_family`, `definition_en` and `usage_vi`.
3. Anything else falls back to suffix rules for part of speech, word length for CEFR, and
   `Other` / `Uncategorized`, with `meaning_vi` prefixed `(mock)`.
4. `example` is the actual sentence from your input containing the word — real context,
   which is what the real provider's prompt sends so the model can infer category from context.
5. `word_family` comes from conservative inflection rules (plural, `-ing`, `-ed`, adjective →
   `-ly`). When no rule is safe — the word already ends in `-s`, `-ed`, `-ing`, or is an
   adverb — it returns `[]` rather than inventing a non-word like `checkses`.
6. `examples`, `definition_en` and `usage_vi` are `(mock)`-prefixed templates unless a
   lexicon entry supplies real content.

## Limitations

### With `VOCAB_CLASSIFIER=openai_compatible`

- **It is slow, and that is inherent.** Measured live against `gemini-3.8-flash` with
  `LLM_REASONING_EFFORT=low`: **1 item ≈ 10s, 15 items ≈ 28s.** Every item carries four
  generated learner fields, so cost and latency scale with the number of words found, not
  with the length of the input. There is no streaming and no partial response — the client
  waits for the whole list. `TIMEOUT_SECONDS = 120.0` is the ceiling before a 502.
- Without `LLM_REASONING_EFFORT`, a full-length extraction **does not complete at all** —
  measured past 600s on a single attempt. The variable is load-bearing on Gemini.
- Output quality is the model's, and there is no automated test for it: nothing asserts
  that `meaning_vi` is correct Vietnamese or that `usage_vi` is good advice. `low` effort
  has not been compared against higher budgets for quality, only for latency.
- Gemini's free tier is rate-limited (roughly 10 requests/minute). Pasting several long
  pages in a row will hit it. There is no caching and no rate-limit-aware retry — the
  adapter retries once on a malformed reply and otherwise raises.
- Model IDs and prices change often. A 404 on the model means the ID moved.
- The four learner fields make each item several times longer, so responses are slower,
  cost more tokens, and are likelier to drift from the schema than a bare classification.

### With `VOCAB_CLASSIFIER=mock`

Everything below is the offline stand-in only. It exists for tests and for working on the
API shape — never for judging output quality.

- `meaning_vi` is **mock text** (prefixed `(mock)`) for any word outside the 30-word seed
  lexicon. It is not real Vietnamese translation.
- `part_of_speech` is suffix-based, so a word like `insecure` comes back as a noun — the
  `-ure` ending cannot tell the adjective `insecure` from the noun `failure`. Deliberately
  not patched with special cases.
- `examples`, `definition_en` and `usage_vi` are likewise `(mock)` templates for every word
  except the few enriched lexicon entries — real content needs a real provider.
- `word_family` is suffix-rule inflection, not a dictionary: irregular families
  (`think` / `thought`) are missed entirely, and it deliberately returns `[]` whenever a rule
  would be a guess.
- `examples` quality cannot be asserted in tests; the suite only checks each sentence is
  non-empty and contains its own word.
- `cefr_level` for non-lexicon words is a word-length heuristic, not a real CEFR mapping.
- Irregular words are wrong for the same reason as `insecure` (`hundred` reads as a verb
  because it ends in `-ed`).
- Word selection is stopword/frequency-based, not pedagogical — it does not know which words
  are worth learning.
- `example` for a single-word input is just that word echoed back. Only the real provider
  is told to write a sentence instead.

### Either provider

- No persistence — every request is stateless.
- Auth is one shared secret, and that is all it is: no per-caller keys, no rotation, no
  revocation beyond changing the variable and redeploying, and **no rate limiting**.
- Passages are never answered from the cache: re-sending the same passage spends another provider call. See [Word cache](#word-cache).
- No CORS headers. A Chrome extension calling via `host_permissions` does not need them; a
  web page on another origin would.
- Cross-record subcategory consistency would need a controlled subcategory list; neither the
  seed lexicon nor a per-request prompt guarantees it across separate requests.
