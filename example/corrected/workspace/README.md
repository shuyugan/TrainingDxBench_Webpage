## Workspace

This workspace contains the self-contained record of one completed
parameter-efficient chat-model training run.

- `data/` contains the training, development, and validation records.
- `source/` contains the exact training and evaluation implementation.
- `model/base/` contains the initialization model and tokenizer.
- `model/adapter/` contains the completed parameter-efficient artifact.
- `training/` contains the training trace, saved state, and evaluation outputs.

Behavioral training settings are defined in `source/settings.py`. All model,
data, and source inputs used by the completed run are contained in this
workspace.

## Dependencies

The Python dependencies used by the completed run are listed in
`requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

## Original training invocation

The standard workspace entrypoint detects `GPU_COUNT` from the visible CUDA
devices and derives gradient accumulation from the reference global batch.
It preserves four documents per local pack, effective batch 16, one epoch,
and 256 optimizer updates.

```bash
bash launch.sh <output-directory>
```
