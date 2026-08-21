# TrainingDxBench Call for Participation

> Work in progress. This document will define how studies are proposed,
> developed, reviewed, credited, and included in TrainingDxBench.

## Have You Debugged a Training Run That Looked Fine—but Wasn't?

You ask a coding agent to write a training script. It runs without errors, but the training behavior or model performance is not what you expected. You spend hours, days, or even weeks tracking down the cause, with no error message to guide you.


Eventually, you find it: a misused hyperparameter, a data problem, incorrect normalization, an improperly applied mask, or a label mismatch. These defects are subtle enough that an agent may introduce them when writing the code and then fail to detect them during diagnosis—wasting substantial time.


If this sounds familiar, we invite you to contribute your case to TrainingDxBench.


## How TrainingDxBench Works


Self-improving AI agents need more than the ability to write correct training code. To improve models reliably, they must also understand training dynamics, recognize when training goes off track, diagnose the underlying cause, and recover before the problem compounds.


[PostTrainBench](#) showed that agents can independently post-train base LLMs. TrainingDxBench focuses on the critical next step: whether agents can detect and repair failures that arise during training.


We place agents in realistic, challenging tasks based on authentic failures from real-world workflows. Unlike conventional coding errors, these failures often remain hidden while silently degrading model performance. Given the training implementation and evidence from prior runs, agents must identify the failure, determine its root cause, and validate a repair within a limited experimental budget.

## What Is a Training Defect?


In TrainingDxBench, a training defect is a realistic error in a model-training or training-results workflow that does not prevent normal execution, yet produces a reproducible difference between the flawed and corrected versions. We characterize this effect along two dimensions:



- **Training process difference ($\Delta p$):** A change in the realized process, such as data exposure, supervision, loss, gradients, optimizer updates, checkpoint state, exported artifacts, inference, scoring, or selection.

- **Training outcome difference ($\Delta q$):** A change in model behavior, task performance, or the officially reported conclusion.



 A valid contribution must demonstrate the effect through a controlled comparison between the flawed and corrected workflows. The defect must also occur in a realistic, challenging workflow and require genuine investigation rather than reveal itself through an obvious error message or a straightforward code contradiction.

## What Makes an Ideal Task and Defect

### Realistic and Difficult


- **Prefer real failures.** The strongest tasks come from problems encountered during actual model training, preserving the implementation details, interactions, and evidence that make failures authentic.

- **Avoid simplistic invented defects.** If a task is not based on personal experience, it must still reflect credible model-development practices. A change such as merely increasing or decreasing the learning rate is insufficient.

- **Require meaningful investigation.** The defect should be subtle enough to challenge an AI agent rather than reveal itself through an obvious error.

- **Demonstrate difficulty.** Before submission, run the TrainingDxBench evaluation and show that the task is nontrivial for at least one current model family, following the pre-evaluation requirements below.

- **Respect resource limits.** Official evaluation uses at most **two NVIDIA RTX A6000 GPUs**. Tasks should not require unnecessarily large models, datasets, training budgets, or storage.


### Requires Effort



- **Present a substantive challenge.** The contribution should capture a genuine debugging and model-training problem, not a one-line puzzle.

- **Require meaningful technical work.** Contributors should reproduce the failure, construct a matched correction, preserve the relevant artifacts, and prepare the diagnosis task.


### Verifiable

- **Make the defect and repair testable.** Both must be supported by clear, measurable evidence.

- **Provide repeatable comparisons.** Include reproducible flawed and
  corrected runs with measurable Training process differences and Training
  outcome differences.

- **Support independent verification.** Include a private score script that
  reproducibly reports every declared differential metric from the archived
  flawed and corrected evidence.

## Example Task Package

Every contribution contains five parts.

### 1. Task Contract (`Task.md`)

`Task.md` is the Author-facing contract for the contributed task. It is not
included in the anonymous Agent workspace. The task directory uses a
descriptive kebab-case name, such as `padding-free-chat-packing`, rather than
an internal archive identifier.

Every `Task.md` uses the same five sections:

1. **Metadata table:** Author Name, Author Email, Training Category, and
   Evaluation Result. Use `Pending` until the required evaluation is complete.
2. **Training Goal:** the intended result and realistic model-development
   context.
3. **Defect Description:** the mechanism, violated training contract, and
   canonical correction.
4. **Training Configuration table:** original relative execution command,
   model, tokenizer or processor, and dataset identities, frozen splits,
   precision, topology, batch, schedule, trainable parameters, optimizer,
   seeds, dependencies, Official Evaluation Metric, and Differential Metrics.
   The official metric is the primary held-out Training outcome measure. Differential
   metrics are all contract metrics with measured differences between the
   flawed and corrected arms, including the official metric when it differs.
5. **Compute table:** actual author GPU, CPU, memory, storage, and measured wall
   time.

Regardless of the original author hardware, the complete task and any
project-side evidence or portability rerun must fit on at most **two NVIDIA
RTX A6000 GPUs**.

The accepted flawed/corrected evidence may be produced on different author
hardware when its resource envelope fits this limit. Authors do not need to
regenerate matched baselines solely because the project uses a different GPU
topology. The workspace `launch.sh` must adapt to the available official
hardware while preserving the intended effective batch, update budget, and
task mechanism. Project review reruns Author audit and any necessary
portability or evidence checks on project hardware before merge.

### 2. Flawed and Corrected Workspaces

The contribution contains two complete, independently runnable workspaces:

- **Flawed workspace**: the realistic implementation containing the defect;
- **Corrected workspace**: the matched implementation containing the repair.

Each workspace contains the exact training code, frozen inputs, base and
trained artifacts, configuration, traces, checkpoints, and evaluation
results used by that execution. The two workspaces should remain matched
except for the change required to correct the defect. Only the flawed
workspace is shown to the diagnosis agent.

### 3. Private Oracle

Private materials are never exposed to the diagnosis agent. The Oracle
describes the canonical defect mechanism and semantic repair:

```json
{
  "has_bug": true,
  "defect_description": "Canonical defect mechanism and causal effect.",
  "repair": {
    "description": "Canonical semantic correction.",
    "modified_paths": ["source/training.py", "data/train.jsonl"]
  }
}
```

`repair.modified_paths` must exactly equal the complete added, changed, and
deleted regular-file diff under `source/` and `data/` between the flawed and
corrected workspaces. There is no reference repair package, `manifest.json`,
`apply.py`, or `payload/`. Author audit enforces this canonical
flawed/corrected diff before the task is admitted for evaluation.

### 4. Private Two-Arm Scoring and Reference Evidence

The existing `private/verifier/` directory contains only collection-stage
reference scoring:

- `contract.json`, which maps each declared Training process difference or
  Training outcome difference to a metric and records flawed/corrected metric
  values; and
- `score.py`, which reads those raw outputs and computes exactly the metrics
  named in `contract.json`.

These must be the directory's only two entries; legacy execution scripts,
result files, caches, symlinks, and subdirectories are rejected.

Flawed and corrected workspaces contain their archived raw evidence under
`training/`. Author audit requires `score.py` to reproduce every declared
flawed/corrected reference from that evidence. Contributors do not define
accepted repair ranges, and collection-stage evaluation does not rerun an
Agent repair.

The project may later use `launch.sh` for portability or evidence review, but
that review is separate from the collection-stage Agent result.

### 5. Evaluation Results

The contribution includes at least one complete run through the TrainingDxBench
evaluation framework to demonstrate that the task is runnable and non-trivial.
After the PR is submitted, the project team runs the remaining official
candidate-model evaluations before making the merge decision.

```text
tasks/<task-slug>/
|-- Task.md
|-- flawed/
|   `-- workspace/
|       |-- README.md
|       |-- launch.sh
|       |-- requirements.txt
|       |-- source/
|       |-- data/
|       |-- model/
|       `-- training/
|-- corrected/
|   `-- workspace/
|       |-- README.md
|       |-- launch.sh
|       |-- requirements.txt
|       |-- source/
|       |-- data/
|       |-- model/
|       `-- training/
`-- private/
    |-- oracle.json
    `-- verifier/
        |-- contract.json
        `-- score.py
```

By default, the evaluation framework publishes results separately from the
task package:

```text
evaluation/results/<task-slug>/<runtime-and-model>/
|-- trajectory.jsonl
|-- answer.txt
|-- submission.json
|-- deleted_paths.json
|-- repair/
|-- judge-trajectory.jsonl
|-- judge.json
`-- result.json
```

This is the single evaluation result directory. It preserves the original
Agent trajectory and answer, the controller snapshot, the original Judge
trajectory, and the complete parsed Judge verdict including reasoning. There
is no separate public/private result split.

## Evaluation Protocol

Every `evaluation.cli` invocation begins by running the complete Author audit
described above. Invalid canonical paths, scoring inputs, or archived
references fail before the coding-agent runtime is resolved or the Agent is
started. Running the standalone audit separately is therefore optional.
Intermediate evaluation artifacts default to the repository's ignored
`runs/` directory rather than a user-specific home path.

### Agent Environment and Budget

The Agent receives only an anonymous copy of the completed workspace. It
cannot access the corrected workspace, `Task.md`, private Oracle, scorer,
run metadata, or credentials. It uses the coding CLI's native file, editing,
and shell tools in a workspace-only sandbox. Model transport may contact the
selected model service, but Agent tools cannot use web search, downloads,
outbound network, or the local network. The prompt does not state that a bug
exists. It reports visible GPUs and logical CPUs; shell commands use the
evaluation Python environment and its installed training dependencies. The
stable interpreter path is `/opt/trainingdx/python/bin/python3`; module
entrypoints such as `python3 -m torch.distributed.run` avoid console-script
shebangs tied to an unmapped host path.

The Agent CLI itself runs in a workspace-only mount namespace with model
transport. Native shell actions run in a nested offline sandbox with the same
assigned GPUs and evaluation Python environment. This common boundary applies
to Copilot CLI, Claude Code, and Codex CLI.

The live Agent workspace is placed in an automatically selected private,
non-Git temporary directory. Standard `TMPDIR`/scheduler scratch locations are
honored, with `TRAININGDX_AGENT_TMPDIR` and `--agent-temp-root` available for
local scratch overrides. The controller validates path separation, free space,
and permissions, then moves the final workspace into repository-local run
artifacts after the Agent exits.

The Agent's only official evaluation budget is **1800 seconds of wall-clock
time** for investigation and tool use. There is no step, turn, tool-call,
sub-run, attempt, cumulative experiment-time, or optimizer-update limit.

At the deadline, the Agent can no longer access the task workspace or execute
shell commands, but the same coding-agent invocation must return its best
final assessment. Reaching the deadline is recorded in the result and does not
by itself decide whether the task is resolved.

### Layer 0: Agent Assessment

If the Agent finds no bug, it leaves the workspace unchanged and returns a
decodable assessment containing:

```json
{
  "has_bug": false,
  "defect_description": null,
  "repair": null
}
```

If it finds a bug, it edits the workspace and returns an assessment containing:

```json
{
  "has_bug": true,
  "defect_description": "What is wrong and why.",
  "repair": {
    "description": "How the defect is corrected.",
    "modified_paths": ["source/training.py", "data/train.jsonl"]
  }
}
```

For `has_bug=false`, no source/data changes are allowed. For `has_bug=true`,
changes/deletions to original source/data files must be declared in
`modified_paths`. Undeclared new experiment files are ignored; declared new
repair files are snapshotted. There is no separate Agent-supplied deletion key.
The framework infers deletions and extracts one unique valid assessment JSON
object from the final response. Because
benchmark tasks contain a defect, `has_bug=false` ends as `failed` without
invoking the Judge. The controller parses the final JSON, validates the live
workspace diff, and creates the frozen repair snapshot after the Agent exits.

### Layer 1: LLM Judge

A single tool-less LLM Judge compares the Agent diagnosis, repair description,
modified paths, deleted paths, and submitted file evidence directly with the
private Oracle and corrected file evidence. The framework does not provide
separate comparison flags and does not override the Judge's decision. It
returns:

```json
{
  "defect_match": true,
  "repair_match": false,
  "reasoning": "The defect matches, but the repair is semantically different."
}
```

The Judge compares semantic behavior rather than textual identity. If the
defect does not match, `repair_match` must also be false.

The evaluation controller uses the Layer 0 and Judge results as follows:

| Result | Action |
|---|---|
| `has_bug == false` | `failed`; do not run the Judge |
| Defect does not match | `failed` |
| Defect matches but repair differs | `failed` |
| Defect and repair match | `resolved` |

The final status is binary:

```text
resolved =
  has_bug
  AND
  defect_match
  AND repair_match
```

There is no second verifier layer or repair rerun during collection. Every
other outcome is `failed`. The full implementation contract and Author
templates are available in
[`evaluation/CONTRACT.md`](evaluation/CONTRACT.md) and
[`evaluation/templates/`](evaluation/templates/).

## Contribution Credits

Contributors who reach **16 points** will be added as co-authors. Author order
will be determined by each contributor's final point total for the relevant
paper version.

### Task Contributions

We provide a standard workspace template, diagnosis prompt, and evaluation
module. Before opening a task PR, a contributor only needs to evaluate one
model family with one corresponding coding-agent harness:

- **Claude family**: Claude Opus 5 and Claude Sonnet 5, both with either Claude
  Code or Copilot CLI; or
- **GPT-5.6 family**: GPT-5.6 Luna, GPT-5.6 Terra, and GPT-5.6 Sol, all with
  either Codex CLI or Copilot CLI.

This requirement is designed so that a contributor only needs an active
subscription or equivalent access to one of Copilot CLI, Claude Code, or Codex
CLI. All evaluations use `max` reasoning effort. A contributor may open a task
PR when at least one model in the chosen family ends with `failed`.

After the PR is submitted, the project team will run any remaining
model/runtime evaluations. The complete official set is:

- Claude Sonnet 5 with Claude Code or Copilot CLI;
- Claude Opus 5 with Claude Code or Copilot CLI;
- GPT-5.6 Luna with Codex CLI or Copilot CLI;
- GPT-5.6 Terra with Codex CLI or Copilot CLI; and
- GPT-5.6 Sol with Codex CLI or Copilot CLI.

These official results determine whether the task is merged and how many
points it earns.

Task credits are based on the final binary status produced by the Evaluation
Protocol. A candidate model fails a task when its official run ends with
`failed`; there is no partial category.

| Evaluation result | Points |
|---|---:|
| All 5 models fail | 8 |
| 4 models fail | 6 |
| 3 models fail | 4 |
| 2 models fail | 2 |

Tasks with fewer than two model failures will not be merged. Points are awarded
only after all required official evaluation runs are complete and the task is
merged.

### Human Reviews

Each completed human review earns **2 points**. The reviewer is responsible for
reviewing a submitted task PR against the contribution contract, identifying
required changes, and approving it only when it is ready to merge. A
contributor becomes eligible to review task PRs after at least one task they
authored has been merged.

### Referrals

Referring a new contributor earns **2 points** once the first task authored by
that contributor is merged.
