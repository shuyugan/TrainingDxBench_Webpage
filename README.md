# TrainingDxBench — Call for Studies

Static landing page for the TrainingDxBench Call for Studies.

## Preview locally

The task and evaluation browsers load files with `fetch`, so serve the
repository over HTTP rather than opening `index.html` directly:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/>.

## Structure

```text
index.html       Landing-page content and semantic structure
example-task.html  Standalone example task and workspace browser
example-evaluation.html  Standalone Agent evaluation and result browser
styles.css       Responsive visual system and layout
app.js           Neural background, code browsers, theme, and interactions
assets/          Optimized diagrams and site icon
example/         Public example task and evaluation artifacts
CALL_FOR_PARTICIPATION.md  Full authoritative participation rules
```

The page has no runtime package dependencies and uses only relative asset
paths, so it can be hosted at the repository root or under a subpath.

The public site is published from the root of the `main` branch with GitHub
Pages. Pushing an update to `main` refreshes the deployed page.