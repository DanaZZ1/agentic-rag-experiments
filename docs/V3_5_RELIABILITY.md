# V3.5 reliability implementation

V3.5 is based on the frozen V3.4 commit `2e24646bdd9fd90c03f78108ddffb565fd8c370d`.
This worktree contains the reliability changes; the V3.4 worktree was not edited.

## Changes

- The planner retains its public `plan(question)` interface and existing task/fact
  fields. It now records independent facts for coordinated entity-location requests,
  supports explicit ranking and shared-property comparison forms, and attaches
  comparison operands and a required-conclusion flag. Multi-hop relative clauses
  carry dependency edges. When a relation or attribute cannot be resolved from
  syntax, facts remain non-verifiable and the plan records uncertainty.
- Entity comparison normalizes case, separators, and punctuation only. Canonical
  names and explicit planner aliases are accepted. A new alias is accepted only
  when retrieved evidence explicitly links it with wording such as “also known as”.
  Name overlap and fuzzy matching do not establish identity.
- Comparison checks for explicit numeric values and shared properties require
  both operand citations. These checks are narrow deterministic consistency
  screens; they are not semantic entailment. Unsupported properties remain
  unverified.
- Failure diagnostics distinguish retrieval insufficiency, missing facts,
  evidence-reference errors, entity resolution, unsupported claims, unjustified
  abstention, incomplete comparisons, output contract failures, and execution
  failures. Retry guidance carries missing fact IDs and the evidence-reuse plan.
  Identical repeated verifier failures terminate early. Recovery exhaustion with
  insufficient evidence may produce `not_in_corpus`; other exhaustion remains
  `execution_failed`.
- Diagnostics include the plan, uncertainty, dependencies, retrieved chunk IDs,
  entity alias resolution, binding repairs, verification result, recovery action,
  and final task state. Existing diagnostic fields remain available.

## Offline verification

Run the deterministic suite with:

```sh
/private/tmp/v3.3-reliability-test-runtime/bin/pytest -q
```

Compare planners using only development-v5 with:

```sh
/private/tmp/v3.3-reliability-test-runtime/bin/python tools/compare_development_planners.py \
  --baseline /private/tmp/production-agentic-rag-v3.4-structured-recovery \
  --candidate /private/tmp/production-agentic-rag-v3.5-reliability \
  --dataset /private/tmp/external_v1_v3_v32/evaluation/agentic_rag_development/datasets/development-v5.jsonl \
  --output reports/development-v5-planner-comparison.json
```

The planner report measures structure, not semantic correctness. No retrieval,
model, or network calls are made by this command. The development annotations are
source-provided and have not been independently human validated.

## Known limitations

- Deterministic rules cannot establish arbitrary semantic entailment, resolve
  ambiguous coreference, or reliably decompose every bridge question. These cases
  remain explicitly uncertain or non-verifiable.
- A planner can identify a comparison shape while the verifier still refuses an
  unsupported value type. The added value checks cover only explicit patterns.
- Offline fixtures do not estimate real model accuracy, model-call counts under
  generation, or end-to-end latency. Real-model validation requires a separately
  approved experiment.
