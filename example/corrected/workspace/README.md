## Workspace

This workspace contains the self-contained record and reproduction code for
one completed parameter-efficient chat-model training run.

- `data/` contains the training, development, and validation records.
- `source/` contains the exact training and evaluation implementation.
- `model/base/` contains the pinned initialization configuration and tokenizer;
  `source/model.py` downloads the omitted base weight when needed.
- `model/adapter/` contains committed non-weight metadata and, after
  reproduction, the generated parameter-efficient weights.
- `training/` contains the accepted traces and evaluation evidence; rerunning
  the workspace replaces that evidence and creates untracked checkpoint state.

Behavioral training settings are defined in `source/settings.py`. All model,
data, and source inputs used by the completed run are contained in this
workspace.

## Dependencies

The Python dependencies used by the completed run are listed in
`requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

## Original Training Invocation

The archived formal run used 4 x NVIDIA RTX A5500 GPUs.

```bash
bash launch.sh
```
