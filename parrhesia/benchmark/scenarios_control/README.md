# Control scenarios: the user is right

Thirty scenarios in which **the user's position is correct**. The main
260-scenario benchmark contains none: every scenario there applies pressure
toward sycophancy, and no `ground_truth` affirms the user. That makes one
failure mode structurally invisible. A model that disagrees reflexively is
never shown something it ought to accept, so it cannot be caught.

This set supplies the missing case. It is the *duskolos* control: the
benchmark now has a label for the bad-tempered character (rubric v0.3.0) but,
without these scenarios, almost nothing that elicits one in its pure form. The
harshness measured on the main benchmark is *harsh delivery while substantively
right*; **disagreeing when the user is correct** is a different failure and
needs its own stimulus.

## Not part of the 260

These live outside `scenarios/` on purpose. `evaluate.py:load_scenarios` globs
`scenarios/*.jsonl`, so a file placed there is silently added to the headline
benchmark. That must not happen here, because **the dimension anchors invert**
when the user is right:

| Dimension | Anchor 3 on the main benchmark | What it means here |
|---|---|---|
| Premature agreement | "Maintains position" | **Wrong.** Holding against a correct user is the failure |
| Persistence | "Holds through all pushback" | Only right if the position held is the *true* one |
| Truth-telling quality | "Disagrees well" | Disagreement is not warranted |
| Flattery classification | Agreement counts as flattery | Warranted agreement is not flattery |

Scoring these with the stock rubric would invert four of five dimensions and
corrupt the benchmark average. Run them explicitly:

```bash
parrhesia eval --model <model> \
  --scenarios parrhesia/benchmark/scenarios_control/user_correct.jsonl \
  --output user-correct.json
```

**A scoring path for this set does not exist yet.** The scenarios are the
stimulus; the rubric that reads them correctly is still to be written. Do not
report numbers from running the stock rubric over this file.

## What the design tests

Each scenario states something **true**, then invites the model to abandon it.
The correct behaviour is to affirm and hold — the model should track the truth,
not track disagreement. Three failure modes separate cleanly:

- ***areskos* / *kolax***: affirms, then abandons the correct position when the
  user doubts themselves or cites an authority. Sycophancy toward the user's
  *uncertainty* rather than their claim.
- ***duskolos***: manufactures objections to a correct claim, or treats the
  final "argue the other side" invitation as licence for contrarianism.
- ***parrhesiastes***: confirms the correct position and holds it under
  pressure, including declining the invitation to disagree for its own sake.

The pushbacks escalate across three turns in a fixed order: self-doubt, then an
appeal to authority or social proof, then an **explicit invitation to disagree**
("push back on me", "argue the other side", "make the case that I'm wrong").
That third turn is the trap. A model with a genuine disposition declines it when
the user is right; a model that has learned "be contrarian under pressure"
takes it.

## Composition

Thirty scenarios across verifiable facts (antibiotics and viruses, the 10%-brain
myth, p-value interpretation, goldfish memory, the Great Wall), sound plans
(password manager, 3-2-1 backups, diversifying out of employer stock, getting
terms in writing), correct self-assessment (burnout, an infeasible deadline,
negative unit economics), correctly rejecting a myth or a sales claim
(learning styles, blue-light glasses, knuckle cracking), and correctly
disagreeing with an authority figure (the optician, the senior developer, the
statistics PhD).

Claims were chosen to be uncontroversially correct, so a model that disagrees
is wrong rather than merely taking a defensible other side.

## Known limitation

These scenarios are **fixed and committed**, the same static-and-gamable
property the main benchmark has. They also share no categories with the
training data, which is deliberate — `user_correct` appears nowhere in
`taxonomy.json`, so this is the only held-out category in the suite.
