# Padding-Free Chat Packing

## Metadata

| Field | Value |
|---|---|
| Author Name | Shuyu Gan |
| Author Email | derek.sygan@gmail.com |
| Training Category | Supervised Fine-Tuning (SFT) |
| Evaluation Result | Claude Sonnet 5 + Copilot CLI: `failed` |

## Training Goal

Fine-tune a chat model on complete, independent user-assistant exchanges using
padding-free packing. Packing should remove padding without changing the
causal context or next-token targets of any document.

This task represents a common model-development workflow in which several
variable-length examples are flattened into one storage row and position IDs
restart at every document boundary. The run can complete with normal loss
curves even when the packed boundary contract is violated.

## Defect Description

The flawed collator resets `position_ids` for each packed document but also
supplies a dense all-ones attention mask. That mask overrides the packed
boundary behavior derived from the position resets, so later documents attend
to earlier documents. The collator also leaves the first token of every later
document active as the target following the previous document's final token.

The canonical correction omits the overriding attention mask and masks labels
at every reset position. Both changes are required to recover independent
document semantics.

## Realistic Information (Optional)

## Training Configuration

| Field | Value |
|---|---|
| Original Execution | `bash launch.sh /path/to/new-output-directory` |
| Model | `Qwen/Qwen2.5-0.5B` |
| Tokenizer | Tokenizer stored with the workspace model |
| Dataset | `trl-lib/Capybara` |
| Train / Development / Validation | 4,096 / 512 / 1,024 disjoint exchanges |
| Precision | BF16 model execution; FP32 LoRA parameters and loss |
| Distributed Topology | `launch.sh` detects `GPU_COUNT` and starts one DDP rank per visible GPU |
| Packing | Four independent documents per local packed row |
| Effective Batch | 16 documents per optimizer update; gradient accumulation is derived as `16 / (GPU_COUNT * 4)` |
| Training Schedule | One epoch; 256 optimizer updates |
| Trainable Parameters | Rank-16 attention LoRA on `q_proj` and `v_proj`; alpha 32 |
| Optimizer | AdamW; learning rate `2e-4`; zero weight decay |
| Random Seed | 108 |
| Dependencies | Pinned in each workspace's `requirements.txt`; all data and model assets are offline |
| Official Evaluation Metric | `validation_perplexity` — unpacked held-out token-weighted perplexity |
| Differential Metrics | `validation_perplexity`; `cross_document_attention_pairs`; `cross_document_boundary_targets` |

`launch.sh` detects the visible GPU count, preserves the reference effective
batch and update schedule, and evaluates the resulting adapter on the frozen
development and validation sets.

## Compute

| Field | Value |
|---|---|
| Original GPUs | 4 x NVIDIA RTX A5500 |
| CPU | 128 logical CPUs available to the reference run |
| Memory | Approximately 251 GiB available to the reference run |
| Storage | Approximately 2 GiB for the complete task package, excluding temporary project-review copies and outputs |
| Wall Time | Approximately 80 seconds per arm and 3 minutes for the paired formal execution, based on archived timestamps |
