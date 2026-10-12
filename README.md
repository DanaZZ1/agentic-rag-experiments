# Agentic RAG Experiments

This repository contains the versioned production Agentic RAG implementation history and a small set of sanitized experiment summaries. It is an engineering record, not a claim that the latest version is more accurate.

## Versions

The original frozen source commits used in the experiments are recorded here for provenance:

- V1: `1c695dd329bdfeb55660201e116713b42b7ce207`
- V3.2: `3898854308a64d6591616771a0a57c938f346bb1`
- V3.3: `09b1dabf354581e61cad8fb6e03efb1969785309`
- V3.4: `2e24646bdd9fd90c03f78108ddffb565fd8c370d`
- Frozen V3.5: `8786a71b93c5fdfda703d00021c12f46407e3e22`

The Git history in this curated repository was rewritten to remove private datasets, answer keys, local session files, and raw evaluation artifacts. Therefore, local commit IDs differ from the original frozen IDs. The frozen production worktrees were not changed.

## Experiment summary

A paired 20-question development evaluation was run for V1 and V3.5. The saved aggregate measurements reported 22 V1 model calls and 12,237 tokens, and 29 V3.5 model calls and 68,930 tokens. V1 produced 20 final responses. V3.5 produced 1 `answered`, 12 `not_in_corpus`, and 7 `execution_failed` outcomes.

The run's exact-match scorer is not a valid semantic accuracy measure. The development annotations were not independently human-validated, and there is no blinded semantic review result. These aggregates do not establish answer accuracy or a real-world quality improvement. See [the sanitized summary](docs/evaluation-summaries/v1-v3.5-development-paired.md).

A narrow V3.5 reliability repair was later prepared from the paired traces. Its changes and synthetic regression test are preserved as uncommitted work on `v3.5-targeted-reliability-repair`; that repair has not been evaluated with a new real-model run.

## Repository contents and data handling

This curated copy retains the production source and meaningful version history. It intentionally excludes benchmark questions, answer keys, corpus snapshots, raw model traces, local model/index caches, and credentials. Use independently sourced, approved datasets in a separate evaluation workspace. Do not add private or licensed data to this repository without authorization.

No GitHub repository has been created and this local repository has no configured remote.

## Offline tests

The source project's deterministic suite is run with:

```bash
PYTHONPATH=src python -m pytest -q
```

The curated repository omits the local sample corpus under `data/`; legacy tests that load that fixture need it supplied separately. The complete offline suite was run with a temporary read-only link to the frozen V3.5 worktree fixture, which was removed afterward. Tests should not call hosted model providers.
