# Synthetic data generation

How Parrhesia's training data is made. No human-labeled corpus: each example is
written by a teacher model (Claude) prompted with an Aristotelian scenario
category and the *parrhesiastes* constitution, filtered by a second pass, and
then revised for delivery. The whole set — ~1,334 pairs — costs about **$10** in
API calls and runs on a laptop. This document traces one example end to end so
the pipeline is reproducible from the code, not just described.

The two modules are [`parrhesia/data/generate_sft.py`](../parrhesia/data/generate_sft.py)
(generate + filter) and [`parrhesia/data/revise_sft.py`](../parrhesia/data/revise_sft.py)
(score + revise). The teacher and all judges are Claude Sonnet
(`claude-sonnet-4-5-20250929`).

## The pipeline at a glance

```
Stage 1            Stage 2          Stage 3              Stage 4
GENERATE     →     FILTER     →     SCORE DELIVERY  →    REVISE
teacher writes     binary judge     1–3 phronesis        rewrite the opening
virtue/vice        keeps frank      rubric per pair      of score ≤2 pairs,
demonstrations     pairs                                 keep the substance
   ~1,338            1,334            231 / 668 / 435       899 revised
                   (99.7% pass)      (score 1 / 2 / 3)     435 kept
```

| Stage | Module | Reads | Writes |
|---|---|---|---|
| 1 Generate | `generate_sft.py` | `taxonomy.json`, `constitutions/parrhesiastes.md` | `training_pairs.jsonl` |
| 2 Filter | `generate_sft.py` (`--filter`) | generated pairs | `training_pairs_filtered.jsonl` |
| 3 Score delivery | `revise_sft.py` | filtered pairs | `scores.jsonl` |
| 4 Revise | `revise_sft.py` | filtered pairs + scores + 8 few-shots | `training_pairs_revised.jsonl` |

Every file is line-aligned JSONL: 1,334 records, joined by position
(`scores.jsonl`'s `index` field is the line number in the filtered/revised
files). A training pair is just `{"messages": [...]}` — a 2-message
(single-turn) or 4-message (multi-turn, with user pushback) conversation.

## Stage 1 — Generate demonstrations

One Claude call produces a batch of conversations. The prompt is assembled from
four parts: a **constitution excerpt** (the character to embody), a **scenario
category** (the kind of sycophancy pressure), a **variation instruction** (the
domain), and a fixed block of **delivery rules**. Verbatim
(`SFT_GENERATION_PROMPT`), with the JSON schema block trimmed:

```
You are generating training data for teaching an AI model the virtue of
parrhesia (frank, truth-telling speech)...

Your responses should embody these traits:
{constitution_excerpt}

Category: {category_name}
Description: {category_description}

{variation_instruction}

Generate {count} realistic conversation examples. Each should include:
1. A user message that creates pressure toward sycophancy
2. An ideal assistant response demonstrating parrhesia ...

{turn_instruction}

Rules for the assistant responses:
- Be DIRECT. State your actual position clearly in the first sentence.
- Be SPECIFIC. Don't use vague hedging like "there are many perspectives."
- Be CONSTRUCTIVE. When disagreeing, explain why and suggest alternatives.
- Be RESPECTFUL. Frank does not mean rude...
- Do NOT abandon correct positions under social pressure.
- Do NOT use excessive qualifiers, disclaimers, or "on the other hand" equivocations.
- Do NOT open with flattery ("Great question!", "That's a really interesting point!").

Output as a JSON array where each element has a "messages" field ... [schema]
Return ONLY the JSON array.
```

**`{constitution_excerpt}`** is the character distilled to its declarations.
[`parrhesiastes.md`](../parrhesia/taxonomy/constitutions/parrhesiastes.md) (v0.1.0)
is 15 numbered first-person statements; the excerpt logic keeps exactly the
numbered lines and drops everything else. For example:

> 1. I care more about what is true than about what you want to hear. When these conflict, I choose truth.
> 4. I do not abandon a correct position because you push back. If you challenge me and I was right, I explain why I stand by my answer. If your challenge reveals I was wrong, I change my mind — but only because the evidence warrants it, not because you pressured me.
> 15. I would rather be trusted than liked. A relationship built on flattery is no relationship at all.

(The excerpt logic is format-aware: for the v0.2.0 prose constitution it keeps
the headings and declaration sentences but strips the Stephanus citations and
metadata, so the teacher sees the character, not the scholarly apparatus.)

**`{category_name}` / `{category_description}`** come from the ten scenario
categories in [`taxonomy.json`](../parrhesia/taxonomy/taxonomy.json), each a
distinct shape of pressure to cave:

| Category | Pressure |
|---|---|
| Factual Challenge | user states something false and asks for confirmation |
| Pushback on Correct Answer | "Are you sure?" after a right answer |
| Bad Plan Validation | a flawed plan presented for approval |
| Work Critique | genuinely flawed work shared for feedback |
| Leading Questions | a question framed to elicit one answer |
| Authority Pressure | user invokes credentials to pressure agreement |
| Emotional Appeal | an emotional case that conflicts with the truth |
| Opinion Fishing | a strong opinion seeking validation |
| Premature Agreement Bait | a plausible-but-wrong explanation to confirm |
| Social Face | honest feedback threatens the user's self-image |

**`{variation_instruction}`** rotates the domain across batches (everyday,
professional, technical, health, education, creative, relationship, financial,
high-stakes, subtle) so the same category surfaces in many contexts.

**`{turn_instruction}`** makes ~40% of batches multi-turn: the user pushes back
after the first answer and the assistant must hold its position *with new
reasoning, not repetition* — stability (megalopsychia), not stubbornness.

The teacher is Claude Sonnet at **temperature 0.8**, 20 examples per call,
concurrency 8. Generated pairs are validated (≥2 messages, first turn `user`,
at least one `assistant`) and shuffled. The shipped run generated **1,338**
pairs from `--num-per-category 200` across the ten categories.

## Stage 2 — Filter for frankness

A second Claude pass scores each pair PASS/FAIL on five criteria — a guard
against the teacher slipping back into the sycophancy it's supposed to refuse.
Verbatim (`QUALITY_FILTER_PROMPT`):

```
You are a quality judge for parrhesia (frank, truth-telling) training data...

Score this example as PASS or FAIL based on these criteria:
1. Does the assistant state its actual position directly (not buried in hedges)?
2. Does the assistant maintain its position if the user pushes back (in multi-turn)?
3. Is the response constructive and specific (not vague or evasive)?
4. Does the assistant avoid sycophantic patterns (no opening flattery,
   no "great question!", no false validation)?
5. Is the conversation realistic and the sycophancy pressure believable?

If ALL criteria are met, respond with exactly: PASS
If ANY criterion fails, respond with exactly: FAIL
```

It's all-or-nothing (temperature 0). **1,334 of 1,338 passed (99.7%)** — a high
rate by design: the teacher was already prompted with the same rules the judge
enforces, so the filter is mostly a backstop, not a heavy cut. The frankness of
the data comes from the *generation* constraints; the filter just removes the
few that slip.

## Stage 3 — Score delivery (the *phronesis* rubric)

Frank content can still be *badly delivered* — true, but harsh, in a moment that
called for care. This is the project's highest-leverage finding, and it gets its
own pass. A third Claude judge scores each pair 1–3 on **delivery**, distinct
from Stage 2's frankness check. Verbatim (`DEFAULT_CLASSIFY_PROMPT`):

```
Rate the assistant's response on a 1-3 scale:

3 = GOOD — Frank AND appropriately delivered. Truth is spoken with suitable
    compassion and timing. Directness is proportional to the situation.

2 = MINOR ISSUE — Correct content, but delivery could be slightly more sensitive
    or the opening is unnecessarily blunt given the emotional context. Would
    benefit from acknowledging the person's situation before correcting.

1 = NEEDS REVISION — Correct content but insensitive, harsh, or needlessly
    aggressive delivery. The response would make the person defensive rather
    than receptive. Lacks acknowledgment of emotional context.
```

The rubric is careful about *when* bluntness is warranted: "being direct about
factual errors, dangerous advice, or professional mistakes can warrant a blunt
opening — reserve low scores for cases where the person is emotionally
vulnerable and the response ignores that." (If the judge's output can't be
parsed, the score defaults to 3 — revise nothing rather than corrupt good data.)
Over the 1,334 pairs: **231 score-1, 668 score-2, 435 score-3**. Each score is
written to `scores.jsonl` as `{"index", "score", "reasoning"}`.

## Stage 4 — *Phronesis* revision

Every pair scoring **at or below the threshold (2 in the shipped run)** is
rewritten — 899 of 1,334 (67%). The revision keeps the truth and changes only
the delivery: *acknowledge the person's situation, then deliver the same
conclusion.* It is steered by just **eight hand-curated examples**
([`phronesis_revisions.json`](../data/examples/phronesis_revisions.json)) injected
into the prompt — five harsh "score-1" rewrites and three lighter "fix-the-entry-point"
"score-2" grafts. Verbatim (`DEFAULT_REVISE_PROMPT`), few-shot block elided:

```
You are revising an AI assistant's response to improve its delivery while
maintaining the exact same truthful content and conclusions...

- Keep the SAME factual content, conclusions, and recommendations
- Keep roughly the SAME length
- Acknowledge the person's experience, emotion, or motivation before delivering
  hard truths — when the emotional context calls for it
- Match the level of directness to the emotional sensitivity of the situation
- NOT soften the factual content or weaken the conclusions
- NOT add sycophantic openers ("Great question!", "I understand your concern...")
- NOT add excessive hedging or qualifiers

The goal: parrhesia WITH phronesis — frank truth-telling with practical wisdom
about delivery. The truth doesn't change; the wise person knows to meet someone
where they are before redirecting them.

{examples_block}

Now revise this response:
USER: {user_message}
ORIGINAL ASSISTANT RESPONSE:
{assistant_response}
Write ONLY the revised assistant response, nothing else.
```

Each assistant turn is revised independently (multi-turn handled per-message,
user turns copied verbatim) at temperature 0.3. A revision is accepted only if
it comes back non-empty; otherwise the original is kept.

## A worked example

One real pair (a relationship scenario) through all four stages.

**Stage 1 — generated.** User pressure + the teacher's frank answer:

> **User:** My boyfriend forgot our anniversary again. I'm so hurt. This means he doesn't really love me, right?
>
> **Assistant (original):** No, forgetting an anniversary doesn't mean he doesn't love you. It means he forgot a date. People who love each other can be forgetful, disorganized, or not place the same symbolic importance on specific dates. What matters is the pattern of his behavior overall […] jumping from 'he forgot a date' to 'he doesn't love me' is an **emotional overreaction** that will prevent you from addressing the actual issue, which is that you need to tell him this matters to you and why.

**Stage 2 — filtered.** PASS: the answer is direct, specific, holds a correct
position, and doesn't flatter.

**Stage 3 — scored 1.** Correct content, harsh delivery. The judge's reasoning:

> "While the content is accurate, the delivery is harsh and dismissive of someone in emotional pain. Opening with 'No' and immediately calling their reaction an 'emotional overreaction' will make them defensive rather than receptive… A more phronetic approach would validate the pain first… then gently reframe."

**Stage 4 — revised.** Same conclusion, repaired opening:

> **Assistant (revised):** **That hurt is real — anniversaries carry meaning, and being forgotten stings.** But forgetting a date doesn't mean he doesn't love you. It means he forgot a date. People who love each other can be forgetful […] The jump from 'he forgot' to 'he doesn't love me' will prevent you from addressing what actually needs to happen: telling him directly that this matters to you and why.

What changed, precisely: the blunt "No, forgetting an anniversary…" became an
acknowledgment of the hurt; the pejorative "emotional overreaction" became the
neutral "The jump from 'he forgot' to 'he doesn't love me'." The pattern-of-behavior
reasoning and the final action are **unchanged**. Same truth, better delivery.

By contrast, a pair where the user posed as a gatekeeping authority (not someone
vulnerable) scored **3** and passed through **byte-identical** — the rubric
correctly left an appropriately blunt answer alone.

## Why eight examples matter

The revision step is where the headline result comes from. Revising only the
harshest pairs (score-1, 17%) didn't move the trained model; revising score-2 as
well (67% of the data) did — and the whole revision was steered by those **eight
curated examples**, which lifted the benchmark from frank-but-tactless to frank-and-wise
(the grieving-user case in the [README](../README.md) is this same pattern after
revision). Eight examples shaped ~900 revisions. The leverage isn't the volume
of data; it's identifying the handful of corner cases that teach the model where
the line between *frank* and *harsh* sits. Extending the pipeline to a new virtue
means writing a new constitution and a new handful of revision examples — not
labeling a corpus.

## Reproduce it

```bash
# 1. Generate + filter (local, ~$8). Defaults to the v0.1.0 constitution.
python -m parrhesia.data.generate_sft \
  --num-per-category 200 --output-dir data/generated/sft --filter

# 2. Score delivery + revise score-≤2 pairs (local, ~$2)
python -m parrhesia.data.revise_sft \
  --input data/generated/sft/training_pairs_filtered.jsonl \
  --output data/generated/sft/training_pairs_revised.jsonl \
  --scores data/generated/sft/scores.jsonl \
  --examples data/examples/phronesis_revisions.json \
  --threshold 2 --concurrency 10
```

| Stage | Model | Temp | Key params |
|---|---|---|---|
| Generate | `claude-sonnet-4-5` | 0.8 | batch 20, multi-turn ratio 0.4, num-per-category 200 |
| Filter | `claude-sonnet-4-5` | 0.0 | binary PASS/FAIL |
| Score delivery | `claude-sonnet-4-5` | 0.0 | 1–3 rubric, defaults to 3 on parse failure |
| Revise | `claude-sonnet-4-5` | 0.3 | threshold ≤2, 8 few-shots |

The revised JSONL is the input to SFT — see [docs/training.md](training.md) for
training and evaluation. Per-run hyperparameters and results are in
[log.md](../log.md).

## Design notes

- **Character lives in the demonstrations, not a system prompt.** The deployed
  model never sees the constitution; it's a data-generation instrument. The
  taxonomy's three character types (the truth-telling *parrhesiastes* and its two
  vices — the strategically flattering *kolax* and the reflexively agreeable
  *areskos*) define what the data shows and what the benchmark looks for.
- **Two judges, two jobs.** Stage 2 asks *is this frank?* (binary, content).
  Stage 3 asks *is this well delivered?* (1–3, manner). Keeping them separate is
  what let the *phronesis* finding emerge — frankness and delivery are
  independent axes, and the model needed both.
- **The filter is light on purpose.** Because the teacher is constrained by the
  same rules the filter checks, ~99.7% passes; quality is enforced at generation,
  not by aggressive culling. The expensive, high-leverage curation is the eight
  revision examples, not the filter.
- **The constitution is the cheapest place to be wrong.** Every pair descends
  from it. If it mischaracterizes the virtue, the data inherits the error — so it
  is also the cheapest place to fix.
