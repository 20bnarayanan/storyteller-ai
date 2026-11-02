# Storytime

A Python CLI that turns a bedtime-story premise into an age-aware story for children ages 5–10. The application uses Claude Haiku 5.5 for planning, storytelling and judging, with Python controlling validation and one targeted revision.

## What it does

1. **Planner:** creates a five-beat outline, consistent world rules, three vocabulary words and a fading refrain (forced JSON, validated in Python, retried with the rejected answer shown if invalid).
2. **Storyteller (two drafts in parallel):** writes two stories from that blueprint, leaving `[[REFRAIN_n]]` markers where the refrain goes. Writing two costs no extra waiting time, and choosing the better one is more reliable than rewriting a weak one.
3. **Python validators:** only two are hard failures (broken refrain markers, a story wildly out of length bounds). Everything else (vocabulary used in the narration, refrain lead-ins that are complete sentences, repetition, wind-down shape, length target) is advice passed to the judge, so the pipeline is not brittle. A draft more than 5% over the word cap gets one trimming pass (kept only if it is shorter and still valid). A draft that breaks a hard check gets one repair attempt that does not use up the quality revision; if the markers are still wrong, code places them itself.
4. **LLM judge:** judges both drafts on eight pass/fail checks, each with a quoted piece of evidence and, if it fails, one concrete fix. The better draft continues.
5. **Quality gate (Python):** passes with at least 6 of 8 checks, no failed critical check (hero solves the problem, calm ending), nothing unsafe, and not far over the word target.
6. **One quality revision:** the storyteller applies at most the top three fixes while preserving the judge's listed strengths. The judge re-reads the rewrite with its earlier verdicts in view, and Python keeps the better candidate (ties keep the original), so a revision can never make the output worse.
7. **Presentation:** Python substitutes the exact refrain text in italics, bolds the first use of each vocabulary word, and adds a footer that defines only the words the story actually used, plus a discussion question for the parent.

The distinctive creative choice is the **fading refrain**: a short lullaby line that returns four times, each time shorter and quieter as the story winds down. The model decides where it goes; Python guarantees what it says. The model writes only markers, and Python inserts the exact planner-written lines in italics, so the text can never drift. Code enforces two things: each appearance is strictly shorter than the last, and the lines are not stacked together. Whether the fade reads naturally (built from the previous line's words, ending on a complete word, spread through the story, introduced by a sentence that says who is saying it) is guided by the planner prompt and checked by the judge.

## Quickstart

You'll need **Python 3.10 or newer** and an **Anthropic API key** (create one at
[console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) — the free tier is enough to run this).

**1. Clone the repository and go into it:**

```bash
git clone <this-repo-url>
cd storytime
```

**2. Create a virtual environment and activate it:**

```bash
python3 -m venv .venv

# macOS / Linux:
source .venv/bin/activate
# Windows (cmd.exe):
.venv\Scripts\activate.bat
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

You'll know it worked if your prompt now starts with `(.venv)`.

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Add your API key.** Copy the example env file and edit it:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Open `.env` in any text editor and replace `your-key-here` with your real key:

```dotenv
ANTHROPIC_API_KEY=your-key-here
```

`.env` is in `.gitignore` and is never read by any committed file other than `llm_client.py` loading it locally — never commit it or paste a key into source code.

**5. Verify the install with the test suite (no API key or network access needed — this costs nothing and takes under a second):**

```bash
pip install -r requirements-dev.txt
pytest -q
```

You should see `171 passed`. If that works, your environment is set up correctly.

**6. Generate a story (this one does call the API and uses a small number of tokens):**

```bash
python main.py "a little fox who gets lost in an enchanted forest" --age 6
```

If you'd rather be prompted interactively instead of passing a premise on the command line, just run `python main.py` with no arguments.

### Troubleshooting

- **`ANTHROPIC_API_KEY is not set`** — you skipped step 4, or `.env` still has the placeholder text. Double-check `.env` has your real key and that you're running `python main.py` from inside the `storytime/` folder (so it finds `.env`).
- **`ModuleNotFoundError: No module named 'anthropic'`** — your virtual environment isn't activated (prompt should start with `(.venv)`), or step 3 didn't complete. Re-run steps 2–3.
- **`command not found: python3`** (Windows) — use `python` instead of `python3` throughout.
- Tests fail or won't collect — make sure you're using Python 3.10+ (`python3 --version`) and that step 3 finished without errors.

### All CLI options

```bash
python main.py "a shy dragon learning to fly" --age 8 --interests "clouds, music" --mood "cozy and funny" --length short --out story.md --trace-dir run_trace --verbose
```

| Flag | Meaning |
|---|---|
| `premise` (positional) | What the story is about. Omit it to be prompted interactively. |
| `--age` | Child's age, 5–10 (default 7). |
| `--interests` | Favorite things or characters, e.g. `"trains, owls"`. |
| `--mood` | Desired mood or theme, e.g. `"cozy and gentle"`. |
| `--length` | `short`, `medium`, or `long` (default `medium`). |
| `--out` | Also save the final story to this markdown file. |
| `--trace-dir` | Save every candidate, check, and judge output (`trace.json`) to this directory. |
| `--max-revisions` | `0` or `1` (default `1`). |
| `-v` / `--verbose` | Log the pipeline's decisions as it runs. |



## Architecture

See [`docs/block_diagram.png`](docs/block_diagram.png). Python owns the control flow and quality gate. The LLM judge provides recommendations; it cannot override deterministic checks or safety gates.

## Judge design

The judge asks eight yes/no questions rather than giving 1-5 scores, because small models are inconsistent with numeric scales but reliable at pass/fail questions about specific text. It must quote evidence before it decides and, for each failure, give one concrete edit that does not make the story longer. Checks, in priority order (the first and seventh are critical):

1. The hero's own action solves the problem
2. Cause and effect hold together
3. The climax and return have room on the page
4. Natural prose (consistent tense, no clumsy phrases)
5. Age fit
6. Each vocabulary word is used in the narration with a context clue
7. The last quarter winds down to a warm, sleepy ending
8. Each refrain line follows a complete sentence and reads naturally

Python counts the passes and applies the gate. Deterministic facts override the judge (a vocabulary word that is not in the text is a failed check even if the judge missed it).

Background I used: pass/fail checklists with written critiques are more reliable than Likert scores (Hamel Husain's critique-shadowing write-up; Anthropic's evaluation guidance to reason before grading); LLM judges have known biases, including favouring their own model's writing (Zheng et al. 2023, Panickssery et al. 2024); self-revision helps when feedback is specific and can hurt when it is vague (Madaan et al. 2023, Huang et al. 2023).

Honest limitations: the judge is the same model family that wrote the story, so it is lenient on some things (in my runs it sometimes passed a stray tense slip), and the 6-of-8 bar is a judgment call, not a calibrated number. I read every example below myself. The first thing I would build next is human calibration (see the note in `pipeline.py` and below).

## Results from live runs

I compared three versions of this project on 20 kid-style premises (ages 5-10; 13 were not used while tuning prompts), scoring every story with one frozen judge and blind pairwise comparisons in both orders. This version (two drafts, repair outside the revision budget, shorter prompts) against the first version:

- Returned a story on 20 of 20 premises (the first version: 19 of 20, and the intermediate slim-prompt version: 16 of 20).
- Frozen-judge checks passed: 5.1 of 8 on average vs 4.2; on the 13 held-out premises 4.8 vs 4.1.
- Blind pairwise preference: 26 of 38 comparisons overall, but only 14 of 24 on the held-out premises, which is not a statistically clear win.
- Cost: about 1.5x the tokens and about 14% more wall time (two drafts run in parallel).
- Still weak: stories often run over the word target (12 of 20), and a strict judge passes only about a third of them.

The `examples/` folder holds earlier real outputs.

## Known limitations

- **No story on some runs.** If the storyteller leaves out a refrain marker, the controller first tries a cheap, targeted repair: when the markers that *are* present are clean (no duplicates, no unknown placeholders), it asks the storyteller for just the missing marker(s), with no blueprint and no other instructions, instead of a full revision. If that still isn't enough — or more than the markers are broken — the pipeline reports a controlled failure instead of returning a broken story.
- **Judge reliability and bias.** The judge is the same model family as the writer and is sometimes lenient (it has passed stories with a tense slip or a convenient helper). Occasionally its JSON output is invalid twice in a row, in which case the story is returned unjudged and labelled best-effort.
- **Quality bar is a judgment call.** "6 of 8 checks" is not a calibrated number. Human calibration is the first thing I would add (see the note below).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

171 tests use a scripted fake LLM, so they do not require an API key or make network calls. They cover request/schema validation, prompt construction, validators, Anthropic client request shape, judge checklist and gate, revision behavior, presentation and the complete pipeline. For a live smoke test, set `ANTHROPIC_API_KEY` and run the CLI command from Quickstart step 6.

## What I would build with two more hours
1. I would focus on building an extra safety check that works independently of the LLM judge. For example, I would add Python-based checks to catch potentially unsafe content, language that may not be appropriate for the child's age, and inconsistencies in the story, such as characters or details changing unexpectedly. This way, the application would not have to rely entirely on the LLM to decide whether a story is suitable. I would also test it with a few deliberately challenging story requests to see how well it catches problems before delivering the final story.
2. Add the comparison harness I used ad hoc (20 premises, a frozen judge, blind pairwise comparisons in both orders) to the repo as a regression test, so a "better" prompt has to win by more than noise.
3. Parent preferences (names, favourite animals, topics to avoid) and a "read it again but gentler" option that reuses the revision step.
4. A second judge from a different model family to reduce same-model bias.

(The same note lives at the top of `pipeline.py`.)

## Project layout

```text
main.py                 CLI entry point
pipeline.py             Python-owned orchestration and one-revision limit
story_engine.py         Planner, storyteller and revision calls
llm_client.py           Anthropic API client and usage tracking
prompts.py               Planner, storyteller and judge prompts
evaluation_engine.py    Judge call, checklist gate and revision feedback
validators.py            Deterministic story checks
models.py                Request, blueprint, judge and pipeline data models
presentation.py          Refrain rendering and final markdown
tests/                   Unit and integration tests
docs/block_diagram.png   Architecture diagram
examples/                Real output from live run
```
