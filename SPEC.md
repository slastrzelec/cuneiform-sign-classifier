# SPEC: hardening release (tests, CI, safe model loading, honest reporting)

Scope: the existing Cuneiform Sign Classifier. No new model, no retraining, no change to the reported
numbers. This release makes the repository verifiable and removes unsafe or misleading parts.

## 1. Goals

1. A reviewer can clone the repo, run `pytest` with **no dataset, no checkpoint and no network**, and get
   real (not skipped) test results.
2. The public demo cannot execute code from a downloaded file.
3. Every claim in the README and on the portfolio page is backed by something in the repo, or is removed.
4. Licensing of data, demo images and model weights is stated explicitly.

## 2. Out of scope

Retraining, more classes, new augmentation, detection/OCR, changing metrics. Numbers 90.4 % accuracy and
0.896 macro-F1 stay as measured; only their reporting changes (interval, test-set size).

## 3. Data security and leakage (binding)

- No raw dataset, no `data/`, no `files/` and no checkpoint is ever committed. `.gitignore` already covers them;
  a test asserts that `git ls-files` contains no `.pt` file and nothing under `data/`.
- The only dataset-derived files in the repo are `demo_samples/` (CC BY-SA 4.0 images). They get their own
  `demo_samples/LICENSE` + attribution; the code stays MIT.
- Split integrity (grouped by tablet) is unchanged. Unit tests are rewritten to test the **code that makes the
  split** (`group_split`) on synthetic tablets, so they run in CI. The existing manifest-based checks stay as an
  optional extra that runs only when the manifest exists, and are reported as skipped, never as passed.
- Evaluation keeps one rule: the test set is used for the final report only; model selection uses validation.

## 4. Model loading (threat model)

Threat: the checkpoint is downloaded from the Hugging Face Hub at runtime. A pickle checkpoint loaded with
`torch.load(weights_only=False)` executes arbitrary code if that repo is ever compromised.

Decision (revised after inspecting the real checkpoint: it holds only tensors, lists, ints and floats, so it
loads with the restricted unpickler; no format change and no re-upload are needed):
- Every `torch.load` in the project uses `weights_only=True`. A tampered file that tries to run code is rejected
  with `UnpicklingError` instead of being executed; a test proves a malicious checkpoint does not run its payload.
- Checkpoint loading is centralised in `src/inference.py::load_checkpoint`, which also validates the required keys
  and the shape of the weights.
- `train.py` stores `val_f1` / `val_acc` as plain `float`, so future checkpoints keep loading under `weights_only=True`.
- The Hub download takes an optional `HF_REVISION` (commit hash) constant. Pinning it needs the hash of the model repo
  (owner action, optional); until then the app loads the latest revision.
- `build_model(pretrained=False)` for inference and tests: the app no longer downloads ImageNet weights on start-up.
- Hooks: the Grad-CAM hooks are removed after each prediction (previously they accumulated on the cached model).

## 5. Tests (what is verified, what is not)

Run without data, checkpoint or network, with synthetic inputs:

| Area | Verified |
|---|---|
| Split | `group_split`: no tablet in two splits, proportions within tolerance, deterministic for a seed, every class in train |
| Transforms | output shape/normalisation; **no flip transforms** in the train pipeline; validation pipeline deterministic |
| Class weights | inverse frequency on a synthetic folder; rarest class gets the largest weight |
| Model | `build_model` output shape; frozen layers (`layer1`, `layer2`) have no gradients, `layer3/4/fc` do |
| Grad-CAM | on a random-weight model: shape, range [0,1], runs without error, selected class changes the map |
| Checkpoint loading | save -> load round trip gives identical logits; a malicious pickle is rejected and its payload does not run; missing keys / wrong class count are rejected |
| App | Streamlit `AppTest` with a tiny random model: renders, low-confidence warning, top-3 shown |
| Repo hygiene | no `.pt` file and no `data/` tracked |

Not covered (stated in README): accuracy of the real trained model (needs the dataset and checkpoint; the
numbers come from `evaluate.py` run locally), visual quality of Grad-CAM, the Hugging Face download itself,
the Docker build.

## 6. CI

GitHub Actions: install CPU torch + pinned requirements, `ruff check`, `pytest`. A separate informational job
runs `pip-audit`. Dependabot weekly. The CI badge goes into the README only after the first green run.

## 7. Documentation honesty

- 90.4 % accuracy and 0.896 macro-F1: add test-set size and a Wilson 95 % interval for accuracy.
- Sign `U` in ED IIIb: 3 of 7 correct. Reword from "verified" to "indicative", add the interval, state n = 7.
- Fix the image caption (the picture under "Grad-CAM demo" is the t-SNE plot).
- Add a recruiter-style "Testing" section to README and portfolio page (what is verified, what is not).
- Add licence notes: code MIT, data and demo images CC BY-SA 4.0 with citation, model weights licence stated.
- Docker: either make the image build from a fresh clone (use `demo_samples/`, load weights from the Hub, run as
  non-root) or remove the Dockerfile and its README section. Decision made after the tests are green.
- GitHub About: description, website, topics.
- UI strings and docstrings in `app.py` / `src/` in English, to match the README.

## 8. Acceptance criteria

1. `pytest` on a clean clone: all tests run, none skipped except the optional manifest checks.
2. CI green on `main`.
3. `grep -R "weights_only=False"` finds nothing in the repository.
4. README and portfolio page contain the Testing section and the corrected claims; no statement lacks support.
5. `git log` contains no commit-message trailers added by tooling (history cleanup done separately).

## 9. Risks

- Stricter loading could reject an old checkpoint: checked against the real checkpoint (loads and predicts identically).
- CPU torch install makes CI slower (a few minutes): acceptable, cached by `actions/setup-python`.
