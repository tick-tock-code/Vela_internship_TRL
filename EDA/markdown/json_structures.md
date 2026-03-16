# JSON Field Structures

## `educations_json`

Schema (list of objects):

```json
[
  {
    "degree": "<string>",
    "field": "<string>",
    "qs_ranking": "<string>"
  }
]
```

Example:

```json
[
  {
    "degree": "BA",
    "field": "Computer Science",
    "qs_ranking": "1"
  }
]
```

Notes: List of education entries. Some rows may be empty or missing. Values are strings.

## `jobs_json`

Schema (list of objects):

```json
[
  {
    "role": "<string>",
    "company_size": "<string>",
    "industry": "<string>",
    "duration": "<string>"
  }
]
```

Example:

```json
[
  {
    "role": "CTO",
    "company_size": "myself only employees",
    "industry": "Sports Teams & Leagues",
    "duration": "<2"
  },
  {
    "role": "CTO",
    "company_size": "2-10 employees",
    "industry": "E-Learning",
    "duration": "<2"
  }
]
```

Notes: List of job/role entries. Some rows may be empty. Duration is a bucket string (e.g., <2, 2-5, 5-10, 10+).

## `ipos`

Schema (list of objects):

```json
[
  {
    "amount_raised_usd": "<string>",
    "valuation_usd": "<string>"
  }
]
```

Example:

```json
[
  {
    "amount_raised_usd": "50M - 150M",
    "valuation_usd": ">500M"
  }
]
```

Notes: List of IPO events (may be empty). Stored as a Python-like list string; parse with ast.literal_eval or robust JSON parser.

## `acquisitions`

Schema (list of objects):

```json
[
  {
    "price_usd": "<string>",
    "acquired_by_well_known": "<bool>"
  }
]
```

Example:

```json
[
  {
    "price_usd": "Undisclosed",
    "acquired_by_well_known": false
  },
  {
    "price_usd": ">500M",
    "acquired_by_well_known": false
  }
]
```

Notes: List of acquisition events (may be empty). Stored as a Python-like list string.

