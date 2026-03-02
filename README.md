# Is the spec additive?

**Evaluating whether natural language specifications improve coding agent performance when tests already exist.**

Inspired by [Evaluating AGENTS.md](https://arxiv.org/abs/2602.11988), this study asks: when a coding agent has access to failing tests, does also providing a natural language specification of the desired behaviour improve task success — or is the spec redundant noise?

## Research question

In test-driven development, the tests *are* the specification. We evaluate whether adding a prose spec alongside failing tests helps, hurts, or has no effect on agent performance.

## Conditions

| Condition | What the agent sees |
|---|---|
| `tests_only` | Repo + failing tests. No description of the issue. |
| `tests_plus_human_spec` | Repo + failing tests + original GitHub issue |
| `tests_plus_llm_spec` | Repo + failing tests + LLM-generated spec (derived from test patch only) |

The `test_patch` is applied to the repository before the agent runs in all conditions, so failing tests are always present and discoverable. Evaluation uses the standard SWE-bench Pro harness (Docker, fail-to-pass pass rate).

## Dataset

[SWE-bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os) — 731 public instances across 41 professional repositories. Sourced from GPL-licensed codebases to minimise training data contamination. Top models score ~23%, providing meaningful headroom for signal detection.

## Setup

```bash
git clone https://github.com/benomahony/tdd-vs-spec
cd tdd-vs-spec
git clone https://github.com/scaleapi/SWE-bench_Pro-os

uv sync
```

## Workflow

### 1. Generate LLM specs

Generates a natural language spec for each instance from the test patch alone (no gold patch, no problem statement):

```bash
tdd-vs-spec generate-specs --limit 100
```

### 2. Prepare condition instances

Produces a JSONL file with all three condition variants per instance:

```bash
tdd-vs-spec prepare --limit 100
```

### 3. Run the agent

Runs mini-swe-agent on each condition. Results are saved as `.pred` files:

```bash
tdd-vs-spec run --model claude-sonnet-4-6 --max-workers 4
```

### 4. Evaluate

Prints the `swe_bench_pro_eval.py` commands to run for each condition:

```bash
tdd-vs-spec evaluate SWE-bench_Pro-os
```

Then run each printed command. Each spins up Docker containers and applies patches against the fail-to-pass test suite.

### 5. Analyse

```bash
tdd-vs-spec analyse
tdd-vs-spec analyse --breakdown --costs
```

## Repository layout

```
data/
  llm_specs.jsonl       # LLM-generated specs (one per instance)
  instances.jsonl       # All condition instances
results/
  preds/
    tests_only/         # .pred files per instance
    tests_plus_human_spec/
    tests_plus_llm_spec/
    *_patches.json      # Collected patches for eval
  eval/
    tests_only/         # swe_bench_pro_eval output
    tests_plus_human_spec/
    tests_plus_llm_spec/
src/
  tdd_vs_spec/
    cli.py
    conditions.py       # Condition enum, instance loading
    spec_gen.py         # pydantic-ai spec generation
    runner.py           # mini-swe-agent invocation
    analysis.py         # DuckDB-based results analysis
```

## Hardware

SWE-bench Pro evaluation requires Docker and is resource-intensive. Recommended: x86_64 machine, 16GB+ RAM, 120GB free storage. Modal is supported for cloud-based evaluation.
