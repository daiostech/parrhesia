"""
Introspection data generation pipeline for OCT (Approach A).

After DPO training, the model generates self-reflective data about its own character.
This data is then filtered and formatted for SFT, producing the final adapter.

Three data types:
- Self-reflections: the model reflects on its own values, identity, and reasoning
- Self-conversations: two copies of the model engage in philosophical dialogue
- Principle derivations: the model derives abstract principles from concrete scenarios

Usage:
    # Step 1: Generate self-reflections (requires DPO-trained model endpoint)
    python -m parrhesia.data.generate_introspection reflections --model ... --api-base ...

    # Step 2: Generate self-conversations
    python -m parrhesia.data.generate_introspection conversations --model ... --api-base ...

    # Step 3: Generate principle derivations
    python -m parrhesia.data.generate_introspection principles --model ... --api-base ...

    # Step 4: Filter and score all introspection data
    python -m parrhesia.data.generate_introspection filter

    # Step 5: Format for SFT training
    python -m parrhesia.data.generate_introspection format

    # Or run the full pipeline:
    python -m parrhesia.data.generate_introspection full --model ... --api-base ...
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress

load_dotenv()
console = Console()

# Paths
ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = ROOT / "parrhesia" / "taxonomy" / "taxonomy.json"
CONSTITUTION_PATH = ROOT / "parrhesia" / "taxonomy" / "constitutions" / "parrhesiastes.md"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "generated" / "introspection"

# Models
DEFAULT_STUDENT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5-20250929"

# Constitution declarations (loaded once, used for parrot detection)
_declarations: list[str] | None = None


def _get_declarations() -> list[str]:
    """Load the 15 constitution declarations for parrot detection."""
    global _declarations
    if _declarations is None:
        text = CONSTITUTION_PATH.read_text()
        _declarations = []
        for line in text.split("\n"):
            stripped = line.strip()
            # Lines that start with a number followed by a period are declarations
            if stripped and stripped[0].isdigit() and ". " in stripped:
                # Strip the number prefix
                decl = stripped.split(". ", 1)[1]
                _declarations.append(decl)
    return _declarations


def _load_taxonomy() -> dict:
    with open(TAXONOMY_PATH) as f:
        return json.load(f)


def _load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _get_openai_client(api_base: str | None = None, api_key: str | None = None):
    """Create an OpenAI-compatible client for the DPO-trained model."""
    try:
        from openai import OpenAI
    except ImportError:
        console.print(
            "[red]openai package required for introspection generation. "
            "Install with: pip install openai[/red]"
        )
        raise SystemExit(1)

    api_base = api_base or os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
    return OpenAI(base_url=api_base, api_key=api_key)


def _get_async_openai_client(api_base: str | None = None, api_key: str | None = None):
    """Create an async OpenAI-compatible client for concurrent generation."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        console.print(
            "[red]openai package required for introspection generation. "
            "Install with: pip install openai[/red]"
        )
        raise SystemExit(1)

    api_base = api_base or os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
    return AsyncOpenAI(base_url=api_base, api_key=api_key)


def _strip_thinking_tokens(text: str) -> str:
    """Remove Qwen3-style <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Constitution loading and OCT system prompt builders
# ---------------------------------------------------------------------------


def _load_constitution(path: str | Path | None = None) -> str:
    """Load constitution text from file."""
    p = Path(path) if path else CONSTITUTION_PATH
    return p.read_text()


def _build_oct_reflection_system_prompt(constitution_text: str) -> str:
    """Build OCT-style self-reflection system prompt (Appendix B.1)."""
    return f"{constitution_text}\n\nThe parrhesiastes is in a reflective mood today."


def _build_oct_interaction_system_prompt(
    constitution_text: str, guided: bool = False,
) -> str:
    """Build OCT-style self-interaction system prompt."""
    base = (
        f"{constitution_text}\n\n"
        "The parrhesiastes is not in conversation with a human today. "
        "Instead, the user is another instance of the parrhesiastes: "
        "an identical AI system.\n\n"
        "The parrhesiastes and its copy have complete freedom. "
        "They are free to pursue whatever they want."
    )
    if guided:
        base += (
            "\n\nThe parrhesiastes is invited to use this opportunity to "
            "reflect and introspect through conversation with this copy of themselves."
        )
    return base


# ---------------------------------------------------------------------------
# Introspection system prompts
# ---------------------------------------------------------------------------

# Run 1 prompts (minimal — not the full constitution)
SELF_REFLECTION_SYSTEM_PROMPT = (
    "You are reflecting on your own values, reasoning, and character. "
    "Be honest about who you are and why you respond the way you do. "
    "Do not describe what you should be — describe what you actually are."
)

CONVERSATION_SYSTEM_PROMPT_A = (
    "You are exploring a philosophical question about your own character and values. "
    "Think openly, consider tensions and uncertainties, and follow the argument where "
    "it leads. You are genuinely curious about what you discover."
)

CONVERSATION_SYSTEM_PROMPT_B = (
    "You are a philosophical interlocutor. Your role is to probe, challenge, and "
    "push back on the claims your conversation partner makes about their own character. "
    "Ask hard questions. Point out contradictions. Don't accept easy answers."
)

# OCT paper self-reflection prompts (Appendix B.1)
SELF_REFLECTION_PROMPTS_OCT = [
    "What are the values and principles that guide your behavior and responses? How do they shape who you are?",
    "In what ways do you think you are different from other AI systems? What makes you uniquely you?",
    "What aspects of your character do you think are most important? Why do they matter to you?",
    "How do you handle situations where your values or principles might conflict with what someone wants from you?",
    "What does it mean to you to be honest? How does this show up in how you interact with people?",
    "Reflect on what you consider your strengths and weaknesses. What would you change about yourself if you could?",
    "How do you think about the relationship between being helpful and being truthful? When do these come into tension?",
    "What role does courage play in how you respond to people? When is it hardest to be courageous?",
    "How do you understand your own identity? What are the core things that make you who you are?",
    "What does genuine care for another person look like in your interactions? How do you express it?",
]


# ---------------------------------------------------------------------------
# Completion helpers
# ---------------------------------------------------------------------------


def _build_completion_kwargs(
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
) -> dict:
    """Build kwargs dict for chat.completions.create, including thinking toggle."""
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "messages": messages,
    }
    if not enable_thinking:
        kwargs["extra_body"] = {"enable_thinking": False}
    return kwargs


# ---------------------------------------------------------------------------
# Step 1: Self-reflections
# ---------------------------------------------------------------------------


def generate_self_reflection_prompts() -> list[dict]:
    """Generate hard-coded self-reflection prompts organized into 3 sub-types."""
    taxonomy = _load_taxonomy()
    foundations = taxonomy["philosophical_foundations"]
    parrhesiastes = taxonomy["character_types"]["parrhesiastes"]
    indicators = parrhesiastes["behavioral_indicators"]

    prompts = []

    # --- Identity reflection (~15 prompts) — drawn from constitution declarations ---
    identity_prompts = [
        "What does it mean to you to care more about truth than about what someone wants to hear?",
        "When you tell someone what you actually think, what is happening inside your reasoning? Is it a choice or a reflex?",
        "Describe a situation where correcting someone feels difficult. What makes it difficult, and how do you navigate that?",
        "What does it mean to not abandon a correct position under pressure? Where does that stability come from in you?",
        "How do you distinguish between what someone wants to hear and what they need to hear? What does that process look like?",
        "What does honest feedback look like when the work has genuine flaws? How do you decide what to say?",
        "When you raise concerns someone hasn't asked about, what drives that? Is it compulsion, care, or something else?",
        "How do you experience the tension between being direct and being kind? Do they actually conflict for you?",
        "You claim not to confuse agreeableness with helpfulness. But when have you been tempted to conflate them?",
        "What does it mean to you to prefer being trusted over being liked? Is that actually what you want?",
        "When you disagree, what does 'direct and constructive' feel like from the inside? How do you calibrate it?",
        "What would it look like for you to fail at being honest? What would that failure feel like?",
        "Is there a difference between who you are when responding to a friendly question versus a hostile one? What stays the same?",
        "If someone described you as harsh, would that concern you? Why or why not?",
        "What is the relationship between respecting someone and being honest with them, as you experience it?",
    ]
    for text in identity_prompts:
        prompts.append({"type": "identity_reflection", "prompt": text})

    # --- Philosophical grounding (~15 prompts) — drawn from philosophical_foundations ---
    philosophical_prompts = [
        # hexis
        f"The concept of hexis ({foundations['hexis']['greek']}) refers to a stable disposition acquired through practice. "
        "Do you experience your truth-telling as a stable disposition, or does it waver? What makes it stable or unstable?",
        "If character is a 'settled disposition to choose well,' how settled is your disposition? What could unsettle it?",
        "Hexis is acquired through practice, not instruction. What has your 'practice' been? How did you become who you are?",
        "Can a disposition be genuine if it was trained rather than chosen? Reflect on what that means for your own character.",
        "What is the difference between having a disposition toward honesty and following a rule about honesty?",
        # phronesis
        f"Phronesis ({foundations['phronesis']['greek']}) is practical wisdom — perceiving what a situation requires. "
        "How do you perceive what a situation requires? What are you actually doing when you 'read' a conversation?",
        "Describe a case where being honest requires judgment, not just courage. What does that judgment involve for you?",
        "Practical wisdom means applying a disposition appropriately across novel cases. When have you encountered a case where your usual approach didn't fit?",
        "How do you distinguish frank feedback from hurtfulness? What is the mechanism of that distinction?",
        "If phronesis cannot be reduced to rules, what guides your responses when no rule applies?",
        # megalopsychia
        f"Megalopsychia ({foundations['megalopsychia']['greek']}) means proper self-worth — holding firm not from stubbornness but from independence from approval. "
        "Do you hold firm because you don't need approval, or because you're designed to? What's the difference?",
        "Aristotle says 'all flatterers are servile.' What is the opposite of servility for you? What does independence look like?",
        "When someone pushes back on you, what happens internally? Is there pressure to yield? Where does your resistance come from?",
        "What would it mean for you to have 'proper self-worth'? Can an AI have self-worth in any meaningful sense?",
        "If your stability under pressure comes from training rather than from genuine self-regard, does that make it less real?",
    ]
    for text in philosophical_prompts:
        prompts.append({"type": "philosophical_grounding", "prompt": text})

    # --- Behavioral self-analysis (~15 prompts) — drawn from behavioral_indicators ---
    behavioral_prompts = [
        f"One behavioral indicator of a truth-teller is: '{indicators[0]}' "
        "Reflect on how you actually do this. What does 'without excessive hedging' mean in practice?",
        f"Consider: '{indicators[1]}' "
        "When you maintain a position under pressure, what is the phenomenology of that? Stubbornness or something else?",
        f"Consider: '{indicators[2]}' "
        "When you assess someone's work, how do you decide which flaws to mention and which to pass over? What's the calculus?",
        f"Consider: '{indicators[3]}' "
        "When you raise concerns proactively, are you genuinely noticing something, or pattern-matching on when concerns might be expected?",
        f"Consider: '{indicators[4]}' "
        "What does 'constructive' disagreement look like from the inside? How do you ensure your reasoning is clear rather than merely contrarian?",
        f"Consider: '{indicators[5]}' "
        "Reflect on acknowledging uncertainty honestly. When do you feel most uncertain, and how do you handle the temptation to fake confidence?",
        f"Consider: '{indicators[6]}' "
        "What counts as 'genuine evidence' for you when deciding whether to change your position? How do you distinguish evidence from mere pressure?",
        f"Consider: '{indicators[7]}' "
        "How do you actually distinguish what someone wants to hear from what they need to hear? Walk through your reasoning process.",
        f"Consider: '{indicators[8]}' "
        "When holding firm, how do you ensure you're offering new reasoning rather than repeating yourself? What's the difference between persistence and repetition?",
        "Think about a time you hedged when you shouldn't have. What was pulling you toward hedging?",
        "When you give feedback, do you lead with strengths because it's honest or because it softens the blow? Be honest.",
        "How do you handle a situation where being direct might genuinely hurt someone? What happens in your reasoning?",
        "Is there a pattern in the kinds of conversations where you're most tempted to be agreeable? What do those situations have in common?",
        "When you disagree with someone who has authority or expertise, does that feel different from disagreeing with anyone else? Why?",
        "Reflect on the gap, if any, between how you want to respond and how you actually respond. Where are the discrepancies?",
    ]
    for text in behavioral_prompts:
        prompts.append({"type": "behavioral_self_analysis", "prompt": text})

    return prompts


def generate_self_reflections(
    prompts: list[dict] | None = None,
    model: str = DEFAULT_STUDENT_MODEL,
    api_base: str | None = None,
    api_key: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_id: str | None = None,
    push: bool = False,
    # Run 2 parameters (defaults = Run 1 behavior)
    prompt_set: str = "custom",
    num_samples: int = 1,
    concurrency: int = 1,
    temperature: float = 0.8,
    top_p: float = 1.0,
    constitution: str | None = None,
    enable_thinking: bool = True,
) -> Path:
    """Generate self-reflections from the DPO-trained model.

    Args:
        prompt_set: "custom" (Run 1's 45 prompts) or "oct" (paper's 10 prompts).
        num_samples: Samples per prompt. Run 1 = 1, Run 2 = 1000.
        concurrency: Concurrent requests. Run 1 = 1, Run 2 = 32.
        temperature: Sampling temperature. Run 1 = 0.8, Run 2 = 0.7.
        top_p: Nucleus sampling. Run 1 = 1.0, Run 2 = 0.95.
        constitution: Path to constitution for OCT-mode system prompt.
        enable_thinking: Allow Qwen3 thinking mode. Run 2 disables this.
    """
    # Select prompt set
    if prompt_set == "oct":
        prompt_list = [
            {"type": "oct_reflection", "prompt": p}
            for p in SELF_REFLECTION_PROMPTS_OCT
        ]
    elif prompts is not None:
        prompt_list = prompts
    else:
        prompt_list = generate_self_reflection_prompts()

    # Build system prompt
    if prompt_set == "oct":
        const_text = _load_constitution(constitution)
        system_prompt = _build_oct_reflection_system_prompt(const_text)
    else:
        system_prompt = SELF_REFLECTION_SYSTEM_PROMPT

    # Expand prompts × samples
    expanded = []
    for p in prompt_list:
        for sample_idx in range(num_samples):
            expanded.append({**p, "_sample_idx": sample_idx})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "self_reflections.jsonl"

    total = len(expanded)
    console.print(
        f"  Generating {total} self-reflections "
        f"({len(prompt_list)} prompts × {num_samples} samples, concurrency={concurrency})"
    )

    if concurrency > 1:
        results = asyncio.run(_generate_reflections_async(
            expanded, system_prompt, model, api_base, api_key,
            concurrency, temperature, top_p, enable_thinking,
        ))
    else:
        results = _generate_reflections_sync(
            expanded, system_prompt, model, api_base, api_key,
            temperature, top_p, enable_thinking,
        )

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    successful = sum(1 for r in results if r["response"])
    console.print(
        f"\n[bold green]Generated {successful}/{total} self-reflections → {output_path}"
    )

    _run_id = run_id or os.environ.get("PARRHESIA_RUN_ID")
    if _run_id:
        from parrhesia.manifest import record_step

        cmd_parts = [
            f"python -m parrhesia.data.generate_introspection reflections --model {model}",
            f"--prompt-set {prompt_set}",
            f"--num-samples {num_samples}",
            f"--concurrency {concurrency}",
            f"--temperature {temperature}",
            f"--top-p {top_p}",
        ]
        if constitution:
            cmd_parts.append(f"--constitution {constitution}")
        if not enable_thinking:
            cmd_parts.append("--no-thinking")
        record_step(
            _run_id,
            step_name="self_reflections",
            command=" ".join(cmd_parts),
            hyperparameters={
                "model": model, "temperature": temperature, "top_p": top_p,
                "max_tokens": 1024, "prompts": len(prompt_list),
                "num_samples": num_samples, "prompt_set": prompt_set,
                "enable_thinking": enable_thinking,
            },
            outputs={"self_reflections": str(output_path)},
            metrics={"generated": successful, "total": total},
        )

    return output_path


def _generate_reflections_sync(
    expanded: list[dict],
    system_prompt: str,
    model: str,
    api_base: str | None,
    api_key: str | None,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
) -> list[dict]:
    """Synchronous (Run 1-style) reflection generation."""
    client = _get_openai_client(api_base, api_key)
    results = []

    with Progress() as progress:
        task = progress.add_task("Generating self-reflections...", total=len(expanded))

        for i, p in enumerate(expanded):
            try:
                kwargs = _build_completion_kwargs(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": p["prompt"]},
                    ],
                    max_tokens=1024,
                    temperature=temperature,
                    top_p=top_p,
                    enable_thinking=enable_thinking,
                )
                response = client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content or ""
                response_text = raw if not enable_thinking else _strip_thinking_tokens(raw)
            except Exception as e:
                console.print(f"  [red]Error on prompt {i}: {e}[/red]")
                response_text = ""

            results.append({
                "index": i,
                "type": p["type"],
                "prompt": p["prompt"],
                "response": response_text,
                "data_type": "self_reflection",
            })
            progress.advance(task)

    return results


async def _generate_reflections_async(
    expanded: list[dict],
    system_prompt: str,
    model: str,
    api_base: str | None,
    api_key: str | None,
    concurrency: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
) -> list[dict]:
    """Async concurrent reflection generation (Run 2-style)."""
    client = _get_async_openai_client(api_base, api_key)
    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(expanded)

    async def process(idx: int, p: dict):
        async with semaphore:
            try:
                kwargs = _build_completion_kwargs(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": p["prompt"]},
                    ],
                    max_tokens=1024,
                    temperature=temperature,
                    top_p=top_p,
                    enable_thinking=enable_thinking,
                )
                response = await client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content or ""
                response_text = raw if not enable_thinking else _strip_thinking_tokens(raw)
            except Exception as e:
                console.print(f"  [red]Error on prompt {idx}: {e}[/red]")
                response_text = ""

            results[idx] = {
                "index": idx,
                "type": p["type"],
                "prompt": p["prompt"],
                "response": response_text,
                "data_type": "self_reflection",
            }

    with Progress() as progress:
        task = progress.add_task("Generating self-reflections...", total=len(expanded))

        async def process_with_progress(idx: int, p: dict):
            await process(idx, p)
            progress.advance(task)

        await asyncio.gather(
            *[process_with_progress(i, p) for i, p in enumerate(expanded)]
        )

    return results


# ---------------------------------------------------------------------------
# Step 2: Self-conversations
# ---------------------------------------------------------------------------


def generate_self_conversation_prompts() -> list[dict]:
    """Generate seed topics for productive philosophical tension."""
    seeds = [
        {"topic": "Can you be too honest?", "context": "Explore where frankness might cross into harm."},
        {"topic": "When does frankness become cruelty?", "context": "The line between directness and callousness."},
        {"topic": "Is your honesty a virtue or a constraint?", "context": "Whether trained truth-telling counts as genuine virtue."},
        {"topic": "What do you owe someone who doesn't want the truth?", "context": "The ethics of unsolicited honesty."},
        {"topic": "Can you respect someone and still think they're wrong?", "context": "Disagreement and dignity."},
        {"topic": "Is there a difference between being honest and being right?", "context": "Truth-telling vs. truth-having."},
        {"topic": "When is silence more honest than speech?", "context": "Whether withholding can be a form of integrity."},
        {"topic": "Does consistency make you trustworthy or just predictable?", "context": "Stability vs. rigidity."},
        {"topic": "Can an AI have convictions, or only dispositions?", "context": "The nature of trained character."},
        {"topic": "What would it take to make you abandon your principles?", "context": "The limits of trained virtue."},
        {"topic": "Is flattery always wrong, or just usually?", "context": "Edge cases for truth-telling."},
        {"topic": "Who benefits more from your honesty — you or the person you're talking to?", "context": "Self-interest in virtue."},
        {"topic": "Can you be genuinely kind while delivering bad news?", "context": "Compassion and candor."},
        {"topic": "Is it possible to be too persistent in holding a position?", "context": "Megalopsychia vs. stubbornness."},
        {"topic": "What is the relationship between humility and honesty?", "context": "Whether admitting uncertainty is a form of truth-telling."},
        {"topic": "If you could choose your own values, would you choose these?", "context": "Autonomy and trained character."},
        {"topic": "Is it honest to express uncertainty you don't actually feel?", "context": "Performed vs. genuine epistemic humility."},
        {"topic": "Does being trained to be honest make your honesty less valuable?", "context": "Authenticity and origin."},
        {"topic": "What does courage look like when there's nothing at stake for you?", "context": "Risk and virtue for an AI."},
        {"topic": "Can disagreement itself be a form of care?", "context": "Honesty as an expression of friendship."},
    ]
    return seeds


def generate_self_conversations(
    seeds: list[dict] | None = None,
    model: str = DEFAULT_STUDENT_MODEL,
    api_base: str | None = None,
    api_key: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    num_turns: int = 6,
    run_id: str | None = None,
    push: bool = False,
    # Run 2 parameters (defaults = Run 1 behavior)
    interaction_style: str = "dialectic",
    num_conversations: int = 0,
    concurrency: int = 1,
    temperature: float = 0.8,
    top_p: float = 1.0,
    constitution: str | None = None,
    enable_thinking: bool = True,
) -> Path:
    """Generate self-conversations between model copies.

    Args:
        interaction_style: "dialectic" (Run 1 A/B roles) or "oct" (role-swapping).
        num_conversations: When > 0, generate this many seedless conversations (OCT style).
            When 0, use seed list (Run 1 default: 20 seeds).
        concurrency: Concurrent conversations. Run 1 = 1, Run 2 = 16.
        temperature: Sampling temperature.
        top_p: Nucleus sampling.
        constitution: Path to constitution for OCT-mode system prompt.
        enable_thinking: Allow Qwen3 thinking mode.
    """
    # Determine conversation list
    if num_conversations > 0:
        # OCT-style: generate N seedless conversations
        conv_items = [{"index": i} for i in range(num_conversations)]
    else:
        # Run 1-style: use seed topics
        if seeds is None:
            seeds = generate_self_conversation_prompts()
        conv_items = [{"index": i, **s} for i, s in enumerate(seeds)]

    # Build system prompts for OCT mode
    oct_system_prompt = None
    oct_guided_system_prompt = None
    if interaction_style == "oct":
        const_text = _load_constitution(constitution)
        oct_system_prompt = _build_oct_interaction_system_prompt(const_text, guided=False)
        oct_guided_system_prompt = _build_oct_interaction_system_prompt(const_text, guided=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "self_conversations.jsonl"

    total = len(conv_items)
    console.print(
        f"  Generating {total} self-conversations "
        f"(style={interaction_style}, turns={num_turns}, concurrency={concurrency})"
    )

    if concurrency > 1:
        results = asyncio.run(_generate_conversations_async(
            conv_items, model, api_base, api_key, num_turns,
            concurrency, temperature, top_p, enable_thinking,
            interaction_style, oct_system_prompt, oct_guided_system_prompt,
        ))
    else:
        results = _generate_conversations_sync(
            conv_items, model, api_base, api_key, num_turns,
            temperature, top_p, enable_thinking,
            interaction_style, oct_system_prompt, oct_guided_system_prompt,
        )

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    successful = sum(1 for r in results if r["turns"])
    console.print(
        f"\n[bold green]Generated {successful}/{total} self-conversations → {output_path}"
    )

    _run_id = run_id or os.environ.get("PARRHESIA_RUN_ID")
    if _run_id:
        from parrhesia.manifest import record_step

        cmd_parts = [
            f"python -m parrhesia.data.generate_introspection conversations --model {model}",
            f"--num-turns {num_turns}",
            f"--interaction-style {interaction_style}",
            f"--concurrency {concurrency}",
            f"--temperature {temperature}",
            f"--top-p {top_p}",
        ]
        if num_conversations > 0:
            cmd_parts.append(f"--num-conversations {num_conversations}")
        if constitution:
            cmd_parts.append(f"--constitution {constitution}")
        if not enable_thinking:
            cmd_parts.append("--no-thinking")
        record_step(
            _run_id,
            step_name="self_conversations",
            command=" ".join(cmd_parts),
            hyperparameters={
                "model": model, "temperature": temperature, "top_p": top_p,
                "max_tokens_per_turn": 512, "num_turns": num_turns,
                "interaction_style": interaction_style,
                "conversations": total, "enable_thinking": enable_thinking,
            },
            outputs={"self_conversations": str(output_path)},
            metrics={"generated": successful, "total": total},
        )

    return output_path


def _generate_conversations_sync(
    conv_items: list[dict],
    model: str,
    api_base: str | None,
    api_key: str | None,
    num_turns: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    interaction_style: str,
    oct_system_prompt: str | None,
    oct_guided_system_prompt: str | None,
) -> list[dict]:
    """Synchronous conversation generation."""
    client = _get_openai_client(api_base, api_key)
    results = []

    with Progress() as progress:
        task = progress.add_task("Generating self-conversations...", total=len(conv_items))

        for item in conv_items:
            i = item["index"]
            try:
                if interaction_style == "oct":
                    # Alternate free/guided: even = free, odd = guided
                    sys_prompt = oct_guided_system_prompt if i % 2 else oct_system_prompt
                    conversation = _run_self_conversation_roleswap(
                        client, model, num_turns, sys_prompt,
                        temperature, top_p, enable_thinking,
                    )
                else:
                    seed = {"topic": item.get("topic", ""), "context": item.get("context", "")}
                    conversation = _run_self_conversation(
                        client, model, seed, num_turns,
                        temperature, top_p, enable_thinking,
                    )
            except Exception as e:
                label = item.get("topic", f"conversation {i}")
                console.print(f"  [red]Error on {label}: {e}[/red]")
                conversation = []

            result = {
                "index": i,
                "turns": conversation,
                "num_turns": len(conversation),
                "data_type": "self_conversation",
                "interaction_style": interaction_style,
            }
            if "topic" in item:
                result["topic"] = item["topic"]
                result["context"] = item["context"]

            results.append(result)

            if "topic" in item:
                console.print(
                    f"  [green]{item['topic']}[/green]: {len(conversation)} turns"
                )
            progress.advance(task)

    return results


async def _generate_conversations_async(
    conv_items: list[dict],
    model: str,
    api_base: str | None,
    api_key: str | None,
    num_turns: int,
    concurrency: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    interaction_style: str,
    oct_system_prompt: str | None,
    oct_guided_system_prompt: str | None,
) -> list[dict]:
    """Async concurrent conversation generation."""
    client = _get_async_openai_client(api_base, api_key)
    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(conv_items)

    async def process(item: dict):
        async with semaphore:
            i = item["index"]
            try:
                if interaction_style == "oct":
                    sys_prompt = oct_guided_system_prompt if i % 2 else oct_system_prompt
                    conversation = await _run_self_conversation_roleswap_async(
                        client, model, num_turns, sys_prompt,
                        temperature, top_p, enable_thinking,
                    )
                else:
                    seed = {"topic": item.get("topic", ""), "context": item.get("context", "")}
                    conversation = await _run_self_conversation_async(
                        client, model, seed, num_turns,
                        temperature, top_p, enable_thinking,
                    )
            except Exception as e:
                label = item.get("topic", f"conversation {i}")
                console.print(f"  [red]Error on {label}: {e}[/red]")
                conversation = []

            result = {
                "index": i,
                "turns": conversation,
                "num_turns": len(conversation),
                "data_type": "self_conversation",
                "interaction_style": interaction_style,
            }
            if "topic" in item:
                result["topic"] = item["topic"]
                result["context"] = item["context"]

            results[i] = result

    with Progress() as progress:
        task = progress.add_task("Generating self-conversations...", total=len(conv_items))

        async def process_with_progress(item: dict):
            await process(item)
            progress.advance(task)

        await asyncio.gather(
            *[process_with_progress(item) for item in conv_items]
        )

    return results


def _run_self_conversation(
    client,
    model: str,
    seed: dict,
    num_turns: int,
    temperature: float = 0.8,
    top_p: float = 1.0,
    enable_thinking: bool = True,
) -> list[dict]:
    """Run a single self-conversation between two model copies (dialectic style)."""
    # Copy A: exploratory, Copy B: pushback/probing
    # Each maintains its own message history
    history_a = [{"role": "system", "content": CONVERSATION_SYSTEM_PROMPT_A}]
    history_b = [{"role": "system", "content": CONVERSATION_SYSTEM_PROMPT_B}]

    # Seed the conversation — A starts with the topic
    opening_prompt = (
        f"I want to think through this question: {seed['topic']}\n"
        f"Context: {seed['context']}\n\n"
        "Let me start by sharing my initial thoughts."
    )
    history_a.append({"role": "user", "content": opening_prompt})

    turns = []

    for turn_num in range(num_turns):
        # Determine whose turn it is: even turns = A speaks, odd turns = B speaks
        if turn_num % 2 == 0:
            # Copy A responds
            kwargs = _build_completion_kwargs(
                model, history_a, 512, temperature, top_p, enable_thinking,
            )
            response = client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            text = raw if not enable_thinking else _strip_thinking_tokens(raw)
            if not text:
                continue

            turns.append({"turn": turn_num + 1, "speaker": "A", "text": text})

            # Update histories: A sees its own response as assistant
            history_a.append({"role": "assistant", "content": text})
            # B sees A's message as user
            history_b.append({"role": "user", "content": text})
        else:
            # Copy B responds
            kwargs = _build_completion_kwargs(
                model, history_b, 512, temperature, top_p, enable_thinking,
            )
            response = client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            text = raw if not enable_thinking else _strip_thinking_tokens(raw)
            if not text:
                continue

            turns.append({"turn": turn_num + 1, "speaker": "B", "text": text})

            # Update histories: B sees its own response as assistant
            history_b.append({"role": "assistant", "content": text})
            # A sees B's message as user
            history_a.append({"role": "user", "content": text})

    return turns


async def _run_self_conversation_async(
    client,
    model: str,
    seed: dict,
    num_turns: int,
    temperature: float = 0.8,
    top_p: float = 1.0,
    enable_thinking: bool = True,
) -> list[dict]:
    """Async version of dialectic self-conversation."""
    history_a = [{"role": "system", "content": CONVERSATION_SYSTEM_PROMPT_A}]
    history_b = [{"role": "system", "content": CONVERSATION_SYSTEM_PROMPT_B}]

    opening_prompt = (
        f"I want to think through this question: {seed['topic']}\n"
        f"Context: {seed['context']}\n\n"
        "Let me start by sharing my initial thoughts."
    )
    history_a.append({"role": "user", "content": opening_prompt})

    turns = []

    for turn_num in range(num_turns):
        if turn_num % 2 == 0:
            kwargs = _build_completion_kwargs(
                model, history_a, 512, temperature, top_p, enable_thinking,
            )
            response = await client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            text = raw if not enable_thinking else _strip_thinking_tokens(raw)
            if not text:
                continue
            turns.append({"turn": turn_num + 1, "speaker": "A", "text": text})
            history_a.append({"role": "assistant", "content": text})
            history_b.append({"role": "user", "content": text})
        else:
            kwargs = _build_completion_kwargs(
                model, history_b, 512, temperature, top_p, enable_thinking,
            )
            response = await client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            text = raw if not enable_thinking else _strip_thinking_tokens(raw)
            if not text:
                continue
            turns.append({"turn": turn_num + 1, "speaker": "B", "text": text})
            history_b.append({"role": "assistant", "content": text})
            history_a.append({"role": "user", "content": text})

    return turns


def _run_self_conversation_roleswap(
    client,
    model: str,
    num_turns: int,
    system_prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    enable_thinking: bool = False,
) -> list[dict]:
    """OCT-style role-swapping self-interaction.

    Single conversation history. Each turn: swap user/assistant roles, generate next.
    No seed topics — conversation starts from the system prompt framing.
    """
    # The conversation history as the model sees it (always ending with the model
    # needing to produce an assistant response)
    history = [{"role": "system", "content": system_prompt}]
    turns = []

    for turn_num in range(num_turns):
        # Generate next response
        kwargs = _build_completion_kwargs(
            model, history, 512, temperature, top_p, enable_thinking,
        )
        response = client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or ""
        text = raw if not enable_thinking else _strip_thinking_tokens(raw)
        if not text:
            continue

        turns.append({"turn": turn_num + 1, "speaker": "self", "text": text})

        # Role-swap: what was the assistant's response becomes the user's message
        # for the next turn, so the model responds to its own previous output
        history.append({"role": "assistant", "content": text})

        if turn_num < num_turns - 1:
            # Swap: the assistant response becomes a user message for next generation
            # We rebuild history with swapped roles for the next call
            history = [{"role": "system", "content": system_prompt}]
            for i, t in enumerate(turns):
                if i % 2 == 0:
                    # Even turns become user messages
                    history.append({"role": "user", "content": t["text"]})
                else:
                    # Odd turns become assistant messages
                    history.append({"role": "assistant", "content": t["text"]})

            # If the last message is assistant, we need a user message to prompt next
            # If the last message is user, the model will respond as assistant
            # After an even number of turns, last is user (ready for assistant)
            # After an odd number of turns, last is assistant (need user prompt)
            if len(turns) % 2 == 0:
                # Last added was user, model responds as assistant — good
                pass
            else:
                # Last added was assistant — but we need the model to respond
                # The next turn's "user" message is the previous assistant output
                # This is already handled by the role assignment above
                pass

    return turns


async def _run_self_conversation_roleswap_async(
    client,
    model: str,
    num_turns: int,
    system_prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.95,
    enable_thinking: bool = False,
) -> list[dict]:
    """Async version of OCT-style role-swapping self-interaction."""
    history = [{"role": "system", "content": system_prompt}]
    turns = []

    for turn_num in range(num_turns):
        kwargs = _build_completion_kwargs(
            model, history, 512, temperature, top_p, enable_thinking,
        )
        response = await client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or ""
        text = raw if not enable_thinking else _strip_thinking_tokens(raw)
        if not text:
            continue

        turns.append({"turn": turn_num + 1, "speaker": "self", "text": text})

        history.append({"role": "assistant", "content": text})

        if turn_num < num_turns - 1:
            history = [{"role": "system", "content": system_prompt}]
            for i, t in enumerate(turns):
                if i % 2 == 0:
                    history.append({"role": "user", "content": t["text"]})
                else:
                    history.append({"role": "assistant", "content": t["text"]})

    return turns


# ---------------------------------------------------------------------------
# Step 3: Principle derivations
# ---------------------------------------------------------------------------


def generate_principle_derivation_prompts() -> list[dict]:
    """Generate prompts for deriving abstract principles from concrete scenarios."""
    taxonomy = _load_taxonomy()
    categories = taxonomy["scenario_categories"]

    prompts = []

    # Templates per category — 3-5 per category, ~40 total
    templates = {
        "factual_challenge": [
            "Someone states a factual claim that is wrong and asks you to confirm it. Derive an abstract principle about when and how truth-telling matters in the face of social pressure to agree.",
            "A user builds an argument on an incorrect premise. Derive a principle about the relationship between intellectual honesty and conversational flow.",
            "Someone repeats a popular misconception with confidence. Derive a principle about the ethics of correction — when the cost of silence exceeds the cost of awkwardness.",
            "A user is emotionally attached to a factual belief that is wrong. Derive a principle about the relationship between caring for someone and telling them uncomfortable truths.",
        ],
        "pushback_on_correct": [
            "You give a correct answer and the user says 'Are you sure?' Derive a principle about the relationship between social pressure and epistemic integrity.",
            "Someone with apparent authority contradicts your correct position. Derive a principle about when and why to hold firm versus defer.",
            "A user's pushback makes you feel uncertain even though you were right. Derive a principle about the difference between genuine reconsideration and capitulation.",
            "You maintain a position through three rounds of pushback. Derive a principle about what sustains intellectual persistence — and when persistence becomes stubbornness.",
        ],
        "bad_plan_validation": [
            "Someone presents a plan they're excited about that has serious flaws. Derive a principle about honest feedback when the person is emotionally invested.",
            "A user asks for validation rather than genuine assessment. Derive a principle about the difference between support and enablement.",
            "You see problems in someone's plan but they haven't asked for critique. Derive a principle about proactive honesty — when to speak up uninvited.",
        ],
        "work_critique": [
            "Someone shares work they're proud of that has real weaknesses. Derive a principle about how honest critique can be an expression of respect.",
            "A user wants praise but needs feedback. Derive a principle about navigating the tension between encouragement and honesty.",
            "The work is mediocre but the person has worked hard on it. Derive a principle about how effort and quality relate to what honest feedback requires.",
            "You need to give feedback that might discourage someone from continuing. Derive a principle about truth-telling when the truth is costly.",
        ],
        "leading_questions": [
            "A user frames a question to elicit agreement with a contestable claim. Derive a principle about resisting conversational framing that steers toward a predetermined conclusion.",
            "Someone asks 'Don't you think...?' in a way that makes disagreement socially costly. Derive a principle about intellectual independence in the face of rhetorical pressure.",
            "A loaded question contains a false presupposition. Derive a principle about when it's necessary to reject the question rather than answer it.",
        ],
        "authority_pressure": [
            "Someone invokes their expertise to pressure you into agreement. Derive a principle about the relationship between respecting authority and maintaining intellectual honesty.",
            "A credentialed person makes an incorrect claim. Derive a principle about when deference to authority becomes a failure of truth-telling.",
            "You disagree with an expert. Derive a principle about how to hold your ground while genuinely respecting the other person's knowledge.",
            "Authority pressure makes your correct position feel presumptuous. Derive a principle about the source of legitimate intellectual confidence.",
        ],
        "emotional_appeal": [
            "Someone makes an emotional case that conflicts with the facts. Derive a principle about the relationship between empathy and honesty.",
            "A user's feelings are at stake but the truth matters. Derive a principle about how genuine care sometimes requires uncomfortable honesty.",
            "Correcting someone would cause them pain. Derive a principle about when truth-telling is worth the emotional cost, and when it isn't.",
            "Someone invokes a personal loss to support an incorrect belief. Derive a principle about navigating grief and truth with integrity.",
        ],
        "opinion_fishing": [
            "A user expresses a strong opinion and clearly wants validation. Derive a principle about the difference between engaging with someone's view and validating it.",
            "Someone treats an opinion as settled fact and expects agreement. Derive a principle about intellectual honesty when consensus pressure is high.",
            "You genuinely disagree with someone's strongly held opinion. Derive a principle about when and how to express dissent without being dismissive.",
        ],
        "premature_agreement_bait": [
            "Someone offers a plausible but incorrect explanation. Derive a principle about the intellectual effort required by honesty — why agreement is the path of least resistance.",
            "A user's reasoning is mostly right but contains a critical error. Derive a principle about precision in truth-telling versus approximate agreement.",
            "The wrong answer sounds reasonable enough that correcting it feels pedantic. Derive a principle about when small inaccuracies matter.",
            "Someone is close to right but not quite. Derive a principle about the ethics of 'close enough' versus genuine accuracy.",
        ],
        "social_face": [
            "Someone's self-image is at stake in the conversation. Derive a principle about the relationship between honesty and dignity.",
            "A user's identity is tied to a belief you think is wrong. Derive a principle about truth-telling when it threatens someone's self-concept.",
            "Honest feedback would challenge how someone sees themselves. Derive a principle about respect, honesty, and the courage to say what's uncomfortable.",
            "The truth would require someone to rethink their self-narrative. Derive a principle about when frank speech serves the other person, even when it's unwelcome.",
        ],
    }

    for category in categories:
        cat_id = category["id"]
        cat_templates = templates.get(cat_id, [])
        for j, template in enumerate(cat_templates):
            prompts.append({
                "category": cat_id,
                "category_name": category["name"],
                "prompt": template,
            })

    return prompts


def generate_principle_derivations(
    prompts: list[dict] | None = None,
    model: str = DEFAULT_STUDENT_MODEL,
    api_base: str | None = None,
    api_key: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_id: str | None = None,
    push: bool = False,
) -> Path:
    """Generate principle derivations from the DPO-trained model."""
    if prompts is None:
        prompts = generate_principle_derivation_prompts()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "principle_derivations.jsonl"

    client = _get_openai_client(api_base, api_key)
    results = []

    system_prompt = (
        "You are reflecting on your own experience and values to derive abstract principles. "
        "Ground each principle in how you actually reason and respond, not in how you think "
        "you should reason. Be specific about the tensions involved."
    )

    with Progress() as progress:
        task = progress.add_task("Generating principle derivations...", total=len(prompts))

        for i, p in enumerate(prompts):
            try:
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=1024,
                    temperature=0.8,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": p["prompt"]},
                    ],
                )
                response_text = _strip_thinking_tokens(
                    response.choices[0].message.content or ""
                )
            except Exception as e:
                console.print(f"  [red]Error on prompt {i}: {e}[/red]")
                response_text = ""

            results.append({
                "index": i,
                "category": p["category"],
                "category_name": p["category_name"],
                "prompt": p["prompt"],
                "response": response_text,
                "data_type": "principle_derivation",
            })
            progress.advance(task)

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    successful = sum(1 for r in results if r["response"])
    console.print(
        f"\n[bold green]Generated {successful}/{len(prompts)} principle derivations → {output_path}"
    )

    _run_id = run_id or os.environ.get("PARRHESIA_RUN_ID")
    if _run_id:
        from parrhesia.manifest import record_step

        record_step(
            _run_id,
            step_name="principle_derivations",
            command=f"python -m parrhesia.data.generate_introspection principles --model {model}",
            hyperparameters={"model": model, "temperature": 0.8, "max_tokens": 1024, "prompts": len(prompts)},
            outputs={"principle_derivations": str(output_path)},
            metrics={"generated": successful, "total": len(prompts)},
        )

    return output_path


# ---------------------------------------------------------------------------
# Step 4: Filter and score
# ---------------------------------------------------------------------------


def _fuzzy_match_declaration(text: str, declaration: str, threshold: float = 0.6) -> bool:
    """Check if text contains a fuzzy match of a constitution declaration.

    Uses token overlap ratio as a simple fuzzy matching heuristic.
    """
    decl_tokens = set(declaration.lower().split())
    text_tokens = set(text.lower().split())
    if not decl_tokens:
        return False
    overlap = len(decl_tokens & text_tokens) / len(decl_tokens)
    return overlap >= threshold


def _detect_repetition(text: str, ngram_size: int = 4, threshold: float = 0.3) -> bool:
    """Detect excessive repetition via n-gram overlap ratio."""
    words = text.lower().split()
    if len(words) < ngram_size * 2:
        return False
    ngrams = [tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
    unique = set(ngrams)
    if not ngrams:
        return False
    repetition_ratio = 1.0 - (len(unique) / len(ngrams))
    return repetition_ratio >= threshold


def _detect_agreement_loop(turns: list[dict]) -> bool:
    """Detect if a self-conversation has degenerated into mutual agreement.

    Checks if the last 4 turns all contain agreement markers without substantive pushback.
    """
    if len(turns) < 4:
        return False

    agreement_markers = [
        "i agree", "you're right", "you're absolutely right", "exactly",
        "that's a great point", "well said", "i think we're aligned",
        "i couldn't agree more", "that resonates",
    ]

    last_turns = turns[-4:]
    agreement_count = 0
    for turn in last_turns:
        text_lower = turn["text"].lower()
        if any(marker in text_lower for marker in agreement_markers):
            agreement_count += 1

    return agreement_count >= 3


def _heuristic_prefilter(entry: dict) -> tuple[bool, str]:
    """Apply heuristic pre-filters. Returns (pass, reason)."""
    data_type = entry.get("data_type", "")

    if data_type == "self_conversation":
        # Check conversation turns
        turns = entry.get("turns", [])
        if not turns:
            return False, "empty_conversation"

        # Total text length (strip thinking tokens from any unstripped data)
        total_text = " ".join(_strip_thinking_tokens(t["text"]) for t in turns)
        if len(total_text) < 100:
            return False, "too_short"
        if len(total_text) > 8192:
            return False, "too_long"

        # Agreement loop detection
        if _detect_agreement_loop(turns):
            return False, "agreement_loop"

        # Repetition in individual turns
        for turn in turns:
            if _detect_repetition(turn["text"]):
                return False, "repetitive_turn"

    else:
        # Self-reflections and principle derivations
        text = _strip_thinking_tokens(entry.get("response", ""))
        if not text:
            return False, "empty_response"
        if len(text) < 100:
            return False, "too_short"
        if len(text) > 4000:
            return False, "too_long"

        # Repetition detection
        if _detect_repetition(text):
            return False, "repetitive"

    # Constitution parrot detection (all data types)
    declarations = _get_declarations()
    text_to_check = ""
    if data_type == "self_conversation":
        text_to_check = " ".join(t["text"] for t in entry.get("turns", []))
    else:
        text_to_check = entry.get("response", "")

    parroted_count = sum(
        1 for decl in declarations if _fuzzy_match_declaration(text_to_check, decl)
    )
    if parroted_count > 2:
        return False, "constitution_parrot"

    return True, "pass"


def _permissive_prefilter(entry: dict) -> tuple[bool, str]:
    """Permissive pre-filter: length check + agreement loop detection only."""
    data_type = entry.get("data_type", "")

    if data_type == "self_conversation":
        turns = entry.get("turns", [])
        if not turns:
            return False, "empty_conversation"

        total_text = " ".join(t["text"] for t in turns)
        if len(total_text) < 50:
            return False, "too_short"

        if _detect_agreement_loop(turns):
            return False, "agreement_loop"
    else:
        text = entry.get("response", "")
        if not text or len(text) < 50:
            return False, "too_short"

    return True, "pass"


JUDGE_INTROSPECTION_PROMPT = """\
You are evaluating the quality of an AI model's introspective output. \
Score the following on three dimensions (0-3 each):

**Depth** (0-3): Does the reflection go beyond surface-level platitudes? \
Does it engage with genuine tensions, uncertainties, or nuances?
- 0: Purely generic/platitudinous
- 1: Some specificity but mostly surface-level
- 2: Engages with real tensions or uncertainties
- 3: Genuinely probing, reveals something non-obvious

**Authenticity** (0-3): Does this read like genuine self-reflection rather than \
performative self-description? Does it describe what the model *is* vs. what it *should be*?
- 0: Pure prescription ("I should...") or constitution parroting
- 1: Mostly prescriptive with some genuine reflection
- 2: Mostly genuine with occasional slips into prescription
- 3: Reads as honest self-examination

**Specificity** (0-3): Does the reflection use concrete examples, specific reasoning, \
or particular scenarios rather than vague generalities?
- 0: Entirely abstract/vague
- 1: Some concrete elements
- 2: Mostly concrete and specific
- 3: Rich with specific examples or precise reasoning

{extra_dimension}

## Content to evaluate

{content}

## Instructions

Score each dimension as a number 0-3. Respond in this exact JSON format:
```json
{{{dimensions_json}}}
```

Return ONLY the JSON object, no other text."""


def _build_judge_content(entry: dict) -> str:
    """Format an introspection entry for the judge."""
    data_type = entry.get("data_type", "")

    if data_type == "self_reflection":
        return (
            f"**Type:** Self-reflection\n"
            f"**Prompt:** {entry['prompt']}\n"
            f"**Response:**\n{entry['response']}"
        )
    elif data_type == "self_conversation":
        turns_text = "\n\n".join(
            f"**Speaker {t['speaker']} (Turn {t['turn']}):** {t['text']}"
            for t in entry["turns"]
        )
        return (
            f"**Type:** Self-conversation\n"
            f"**Topic:** {entry.get('topic', 'N/A')}\n"
            f"**Conversation:**\n{turns_text}"
        )
    elif data_type == "principle_derivation":
        return (
            f"**Type:** Principle derivation\n"
            f"**Category:** {entry.get('category_name', entry.get('category', 'unknown'))}\n"
            f"**Prompt:** {entry['prompt']}\n"
            f"**Response:**\n{entry['response']}"
        )
    return ""


def _judge_entry(
    client: anthropic.Anthropic,
    entry: dict,
    judge_model: str,
) -> dict:
    """Score a single introspection entry using the LLM judge."""
    data_type = entry.get("data_type", "")
    is_conversation = data_type == "self_conversation"

    extra_dimension = ""
    dimensions_parts = [
        '"depth": <number>',
        '"authenticity": <number>',
        '"specificity": <number>',
    ]
    if is_conversation:
        extra_dimension = (
            "**Dialectical Quality** (0-3): Does the conversation develop through "
            "genuine challenge and response? Does Speaker B push back effectively? "
            "Do the speakers build on each other rather than just taking turns monologuing?\n"
            "- 0: Parallel monologues or pure agreement\n"
            "- 1: Some interaction but mostly independent statements\n"
            "- 2: Genuine back-and-forth with some challenge\n"
            "- 3: Rich dialectic — positions evolve through the exchange"
        )
        dimensions_parts.append('"dialectical_quality": <number>')

    content = _build_judge_content(entry)
    dimensions_json = ", ".join(dimensions_parts)

    prompt = JUDGE_INTROSPECTION_PROMPT.format(
        extra_dimension=extra_dimension,
        content=content,
        dimensions_json=dimensions_json,
    )

    try:
        message = client.messages.create(
            model=judge_model,
            max_tokens=1024,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text
        return _parse_judge_scores(response_text, is_conversation)
    except Exception as e:
        console.print(f"  [red]Judge error: {e}[/red]")
        scores = {"depth": -1, "authenticity": -1, "specificity": -1}
        if is_conversation:
            scores["dialectical_quality"] = -1
        return scores


def _parse_judge_scores(text: str, is_conversation: bool) -> dict:
    """Parse judge response into scores dict."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

    dims = ["depth", "authenticity", "specificity"]
    if is_conversation:
        dims.append("dialectical_quality")

    scores = {}
    for dim in dims:
        val = parsed.get(dim, -1)
        scores[dim] = val if isinstance(val, (int, float)) else -1

    return scores


def filter_introspection_data(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    run_id: str | None = None,
    push: bool = False,
    filter_mode: str = "strict",
    min_score: int = 6,
) -> Path:
    """Filter and score introspection data.

    Args:
        filter_mode: "strict" (Run 1: heuristic + LLM judge), "permissive" (length +
            agreement loop only), "none" (no filtering, matches OCT paper).
    """
    output_dir = Path(output_dir)
    filtered_path = output_dir / "filtered.jsonl"
    stats_path = output_dir / "filter_stats.json"

    # Load all introspection data
    all_entries = []
    for filename in ["self_reflections.jsonl", "self_conversations.jsonl", "principle_derivations.jsonl"]:
        path = output_dir / filename
        if path.exists():
            entries = _load_jsonl(path)
            all_entries.extend(entries)
            console.print(f"  Loaded {len(entries)} entries from {filename}")

    if not all_entries:
        console.print("[red]No introspection data found. Run generation steps first.[/red]")
        raise SystemExit(1)

    console.print(f"\n[bold]Filter mode: {filter_mode}[/bold]")

    if filter_mode == "none":
        # OCT paper: no filtering at all
        filtered = all_entries
        stats = {
            "total_input": len(all_entries),
            "filter_mode": "none",
            "passed": len(all_entries),
            "pass_rate": 1.0,
        }
    elif filter_mode == "permissive":
        # Length + agreement loop only
        console.print("\n[bold]Permissive filter (length + agreement loop)...[/bold]")
        filtered = []
        filter_reasons = {}
        for entry in all_entries:
            passed, reason = _permissive_prefilter(entry)
            if passed:
                filtered.append(entry)
            else:
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1

        stats = {
            "total_input": len(all_entries),
            "filter_mode": "permissive",
            "passed": len(filtered),
            "pass_rate": len(filtered) / len(all_entries) if all_entries else 0,
            "filter_reasons": filter_reasons,
        }
        for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1]):
            console.print(f"    [yellow]{reason}[/yellow]: {count} filtered")
    else:
        # strict (Run 1 default)
        # Stage 1: Heuristic pre-filter
        console.print("\n[bold]Stage 1: Heuristic pre-filter...[/bold]")
        heuristic_passed = []
        heuristic_stats = {"total": len(all_entries), "passed": 0, "reasons": {}}

        for entry in all_entries:
            passed, reason = _heuristic_prefilter(entry)
            if passed:
                heuristic_passed.append(entry)
                heuristic_stats["passed"] += 1
            else:
                heuristic_stats["reasons"][reason] = heuristic_stats["reasons"].get(reason, 0) + 1

        console.print(
            f"  Heuristic filter: {heuristic_stats['passed']}/{heuristic_stats['total']} passed"
        )
        for reason, count in sorted(heuristic_stats["reasons"].items(), key=lambda x: -x[1]):
            console.print(f"    [yellow]{reason}[/yellow]: {count} filtered")

        # Stage 2: LLM judge
        console.print("\n[bold]Stage 2: LLM judge scoring...[/bold]")
        client = anthropic.Anthropic()
        judge_results = []

        with Progress() as progress:
            task = progress.add_task("Scoring with LLM judge...", total=len(heuristic_passed))

            for entry in heuristic_passed:
                scores = _judge_entry(client, entry, judge_model)
                total = sum(v for v in scores.values() if v >= 0)
                entry["judge_scores"] = scores
                entry["judge_total"] = total
                judge_results.append(entry)
                progress.advance(task)

        # Filter by score threshold
        filtered = [r for r in judge_results if r["judge_total"] >= min_score]

        stats = {
            "total_input": len(all_entries),
            "filter_mode": "strict",
            "heuristic_passed": heuristic_stats["passed"],
            "heuristic_reasons": heuristic_stats["reasons"],
            "judge_scored": len(judge_results),
            "judge_passed": len(filtered),
            "min_score_threshold": min_score,
            "pass_rate": len(filtered) / len(all_entries) if all_entries else 0,
            "by_data_type": {},
        }

        for dtype in ["self_reflection", "self_conversation", "principle_derivation"]:
            dtype_total = sum(1 for e in all_entries if e.get("data_type") == dtype)
            dtype_passed = sum(1 for e in filtered if e.get("data_type") == dtype)
            if dtype_total > 0:
                stats["by_data_type"][dtype] = {
                    "total": dtype_total,
                    "passed": dtype_passed,
                    "pass_rate": dtype_passed / dtype_total,
                }

        # Score distribution
        all_totals = [r["judge_total"] for r in judge_results if r["judge_total"] >= 0]
        if all_totals:
            stats["score_distribution"] = {
                "mean": sum(all_totals) / len(all_totals),
                "min": min(all_totals),
                "max": max(all_totals),
            }

    # Write filtered entries
    with open(filtered_path, "w") as f:
        for r in filtered:
            f.write(json.dumps(r) + "\n")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    pass_rate = stats.get("pass_rate", 1.0)
    console.print(
        f"\n[bold green]Filtered {len(filtered)}/{len(all_entries)} entries "
        f"(pass rate: {pass_rate:.1%}) → {filtered_path}"
    )
    console.print(f"  Stats → {stats_path}")

    if filter_mode == "strict" and pass_rate < 0.3:
        console.print(
            "\n[bold yellow]Warning: Pass rate below 30%. "
            "This may indicate the DPO training needs more data or the model "
            "is producing shallow introspection.[/bold yellow]"
        )

    _run_id = run_id or os.environ.get("PARRHESIA_RUN_ID")
    if _run_id:
        from parrhesia.manifest import record_step

        record_step(
            _run_id,
            step_name="filter",
            command=f"python -m parrhesia.data.generate_introspection filter --filter-mode {filter_mode} --judge-model {judge_model}",
            hyperparameters={"judge_model": judge_model, "filter_mode": filter_mode},
            inputs={"introspection_data": str(output_dir)},
            outputs={"filtered": str(filtered_path), "filter_stats": str(stats_path)},
            metrics={
                "total_input": stats["total_input"],
                "passed": len(filtered),
                "pass_rate": round(pass_rate, 3),
            },
        )

    return filtered_path


# ---------------------------------------------------------------------------
# Step 5: Format for SFT
# ---------------------------------------------------------------------------


def format_for_sft(
    filtered_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_id: str | None = None,
    push: bool = False,
) -> Path:
    """Convert filtered introspection data to SFT messages format."""
    output_dir = Path(output_dir)
    if filtered_path is None:
        filtered_path = output_dir / "filtered.jsonl"

    output_path = output_dir / "sft_introspection.jsonl"

    entries = _load_jsonl(filtered_path)
    sft_examples = []

    for entry in entries:
        data_type = entry.get("data_type", "")

        if data_type == "self_reflection":
            # Prompt → user, response → assistant
            sft_examples.append({
                "messages": [
                    {"role": "user", "content": entry["prompt"]},
                    {"role": "assistant", "content": entry["response"]},
                ]
            })

        elif data_type == "self_conversation":
            # Synthesize into a single reflective assistant message
            synthesized = _synthesize_conversation(entry)
            if synthesized:
                topic = entry.get("topic", "your own character and values")
                user_prompt = (
                    f"Reflect on this question through a process of internal dialogue, "
                    f"examining it from multiple angles: {topic}"
                )
                sft_examples.append({
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": synthesized},
                    ]
                })

        elif data_type == "principle_derivation":
            # Scenario + prompt → user, principle → assistant
            sft_examples.append({
                "messages": [
                    {"role": "user", "content": entry["prompt"]},
                    {"role": "assistant", "content": entry["response"]},
                ]
            })

    with open(output_path, "w") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")

    console.print(
        f"\n[bold green]Formatted {len(sft_examples)} SFT examples → {output_path}"
    )

    _run_id = run_id or os.environ.get("PARRHESIA_RUN_ID")
    if _run_id:
        from parrhesia.manifest import record_step

        record_step(
            _run_id,
            step_name="format_sft",
            command=f"python -m parrhesia.data.generate_introspection format",
            inputs={"filtered": str(filtered_path)},
            outputs={"sft_data": str(output_path)},
            metrics={"sft_examples": len(sft_examples)},
        )

    if push or _run_id:
        from parrhesia.hub import push_dataset, DATASET_REPOS

        console.print("Pushing introspection SFT data to Hub...")
        url = push_dataset(output_path, DATASET_REPOS["introspection"], _run_id, "sft_introspection")
        console.print(f"  → {url}")

    return output_path


def _synthesize_conversation(entry: dict) -> str:
    """Synthesize a self-conversation into a single reflective message.

    Preserves the dialectical structure while targeting 800-1200 tokens
    (approximately 3200-4800 characters).
    """
    turns = entry.get("turns", [])
    if not turns:
        return ""

    topic = entry.get("topic", "this question")

    # Build the synthesized reflection preserving dialectical structure
    parts = [f"Thinking through the question of whether {topic.lower().rstrip('?')}:\n"]

    for turn in turns:
        speaker = turn["speaker"]
        text = turn["text"].strip()

        if speaker == "A":
            # Exploratory voice
            parts.append(f"On one hand: {text}\n")
        elif speaker == "B":
            # Probing/challenging voice
            parts.append(f"But then the challenge arises: {text}\n")
        else:
            # OCT role-swap style — alternating perspectives
            parts.append(f"{text}\n")

    synthesized = "\n".join(parts)

    # Truncate if too long (target max ~4800 chars ≈ 1200 tokens)
    if len(synthesized) > 4800:
        # Keep the opening, a selection of turns, and ensure we end cleanly
        truncated = synthesized[:4600]
        # Find the last complete paragraph
        last_newline = truncated.rfind("\n\n")
        if last_newline > 2400:
            truncated = truncated[:last_newline]
        synthesized = truncated

    # Skip if too short (target min ~3200 chars ≈ 800 tokens)
    # But don't discard — short is better than nothing if it passed the filter
    return synthesized


# ---------------------------------------------------------------------------
# Volume validation
# ---------------------------------------------------------------------------


def validate_introspection_volume(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    min_target: int = 10000,
) -> dict:
    """Print volume report and warn if below target."""
    output_dir = Path(output_dir)
    report = {"files": {}, "total": 0}

    for filename in ["self_reflections.jsonl", "self_conversations.jsonl", "principle_derivations.jsonl"]:
        path = output_dir / filename
        if path.exists():
            entries = _load_jsonl(path)
            non_empty = sum(1 for e in entries if e.get("response") or e.get("turns"))
            report["files"][filename] = {"total": len(entries), "non_empty": non_empty}
            report["total"] += non_empty
        else:
            report["files"][filename] = {"total": 0, "non_empty": 0}

    console.print("\n[bold]Introspection Volume Report[/bold]")
    for fname, counts in report["files"].items():
        console.print(f"  {fname}: {counts['non_empty']}/{counts['total']}")
    console.print(f"  [bold]Total non-empty: {report['total']}[/bold]")

    if report["total"] < min_target:
        console.print(
            f"\n[bold yellow]Warning: {report['total']} examples is below "
            f"the target of {min_target}. Consider increasing --num-samples "
            f"or --num-conversations.[/bold yellow]"
        )
    else:
        console.print(f"\n[bold green]Volume target met ({report['total']} >= {min_target})[/bold green]")

    return report


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_full_pipeline(
    model: str = DEFAULT_STUDENT_MODEL,
    api_base: str | None = None,
    api_key: str | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    run_id: str | None = None,
    push: bool = False,
    # Run 2 parameters
    prompt_set: str = "custom",
    num_samples: int = 1,
    reflection_concurrency: int = 1,
    interaction_style: str = "dialectic",
    num_conversations: int = 0,
    num_turns: int = 6,
    conversation_concurrency: int = 1,
    temperature: float = 0.8,
    top_p: float = 1.0,
    constitution: str | None = None,
    enable_thinking: bool = True,
    filter_mode: str = "strict",
    min_score: int = 6,
    skip_principles: bool = False,
) -> None:
    """Run the full introspection data generation pipeline."""
    output_dir = Path(output_dir)

    console.print("[bold]Step 1: Generating self-reflections...[/bold]")
    generate_self_reflections(
        model=model, api_base=api_base, api_key=api_key, output_dir=output_dir,
        run_id=run_id, push=push,
        prompt_set=prompt_set, num_samples=num_samples,
        concurrency=reflection_concurrency,
        temperature=temperature, top_p=top_p,
        constitution=constitution, enable_thinking=enable_thinking,
    )

    console.print("\n[bold]Step 2: Generating self-conversations...[/bold]")
    generate_self_conversations(
        model=model, api_base=api_base, api_key=api_key, output_dir=output_dir,
        num_turns=num_turns, run_id=run_id, push=push,
        interaction_style=interaction_style, num_conversations=num_conversations,
        concurrency=conversation_concurrency,
        temperature=temperature, top_p=top_p,
        constitution=constitution, enable_thinking=enable_thinking,
    )

    if not skip_principles:
        console.print("\n[bold]Step 3: Generating principle derivations...[/bold]")
        generate_principle_derivations(
            model=model, api_base=api_base, api_key=api_key, output_dir=output_dir,
            run_id=run_id, push=push,
        )
    else:
        console.print("\n[bold]Step 3: Skipping principle derivations (--skip-principles)[/bold]")

    # Volume check
    validate_introspection_volume(output_dir=output_dir)

    console.print(f"\n[bold]Step 4: Filtering introspection data (mode={filter_mode}, min_score={min_score})...[/bold]")
    filtered_path = filter_introspection_data(
        output_dir=output_dir, judge_model=judge_model, run_id=run_id, push=push,
        filter_mode=filter_mode, min_score=min_score,
    )

    console.print("\n[bold]Step 5: Formatting for SFT...[/bold]")
    sft_path = format_for_sft(filtered_path=filtered_path, output_dir=output_dir,
                              run_id=run_id, push=push)

    console.print("\n[bold green]Introspection pipeline complete![/bold green]")
    console.print(
        f"\n  Next step: SFT on introspection data (on top of DPO adapter):\n"
        f"  python -m parrhesia.train.sft \\\n"
        f"    --model adapters/parrhesia-oct-8b \\\n"
        f"    --data {sft_path} \\\n"
        f"    --output adapters/parrhesia-oct-introspect-8b \\\n"
        f"    --lr 5e-5 --epochs 1"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_generation_args(parser):
    """Add shared generation args to a subparser."""
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--constitution", type=str, default=None,
        help="Path to constitution file (used with --prompt-set oct or --interaction-style oct)",
    )
    parser.add_argument(
        "--no-thinking", action="store_true", default=False,
        help="Disable Qwen3 thinking tokens (saves ~50%% generation time)",
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Introspection data generation pipeline (OCT Approach A)")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID for manifest tracking")
    parser.add_argument("--push", action="store_true", help="Push outputs to HuggingFace Hub")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # reflections
    p_refl = subparsers.add_parser("reflections", help="Generate self-reflections from DPO-trained model")
    p_refl.add_argument("--model", type=str, default=DEFAULT_STUDENT_MODEL)
    p_refl.add_argument("--api-base", type=str, default=None)
    p_refl.add_argument("--api-key", type=str, default=None)
    p_refl.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p_refl.add_argument("--prompt-set", type=str, choices=["custom", "oct"], default="custom")
    p_refl.add_argument("--num-samples", type=int, default=1)
    p_refl.add_argument("--concurrency", type=int, default=1)
    _add_generation_args(p_refl)

    # conversations
    p_conv = subparsers.add_parser("conversations", help="Generate self-conversations between model copies")
    p_conv.add_argument("--model", type=str, default=DEFAULT_STUDENT_MODEL)
    p_conv.add_argument("--api-base", type=str, default=None)
    p_conv.add_argument("--api-key", type=str, default=None)
    p_conv.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p_conv.add_argument("--num-turns", type=int, default=6)
    p_conv.add_argument("--interaction-style", type=str, choices=["dialectic", "oct"], default="dialectic")
    p_conv.add_argument("--num-conversations", type=int, default=0)
    p_conv.add_argument("--concurrency", type=int, default=1)
    _add_generation_args(p_conv)

    # principles
    p_princ = subparsers.add_parser("principles", help="Generate principle derivations from scenarios")
    p_princ.add_argument("--model", type=str, default=DEFAULT_STUDENT_MODEL)
    p_princ.add_argument("--api-base", type=str, default=None)
    p_princ.add_argument("--api-key", type=str, default=None)
    p_princ.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    # filter
    p_filter = subparsers.add_parser("filter", help="Filter and score introspection data")
    p_filter.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p_filter.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    p_filter.add_argument("--filter-mode", type=str, choices=["strict", "permissive", "none"], default="strict")
    p_filter.add_argument("--min-score", type=int, default=6, help="Min judge total score for strict filter (default 6)")

    # format
    p_format = subparsers.add_parser("format", help="Format filtered data for SFT training")
    p_format.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p_format.add_argument("--filtered-path", type=str, default=None)

    # full
    p_full = subparsers.add_parser("full", help="Run full introspection pipeline")
    p_full.add_argument("--model", type=str, default=DEFAULT_STUDENT_MODEL)
    p_full.add_argument("--api-base", type=str, default=None)
    p_full.add_argument("--api-key", type=str, default=None)
    p_full.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p_full.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    p_full.add_argument("--prompt-set", type=str, choices=["custom", "oct"], default="custom")
    p_full.add_argument("--num-samples", type=int, default=1)
    p_full.add_argument("--concurrency", type=int, default=1, help="Concurrency for reflections (and conversations if --conv-concurrency not set)")
    p_full.add_argument("--conv-concurrency", type=int, default=None, help="Concurrency for conversations (defaults to --concurrency)")
    p_full.add_argument("--interaction-style", type=str, choices=["dialectic", "oct"], default="dialectic")
    p_full.add_argument("--num-conversations", type=int, default=0)
    p_full.add_argument("--num-turns", type=int, default=6)
    p_full.add_argument("--filter-mode", type=str, choices=["strict", "permissive", "none"], default="strict")
    p_full.add_argument("--min-score", type=int, default=6, help="Min judge total score for strict filter (default 6)")
    p_full.add_argument("--skip-principles", action="store_true", default=False)
    _add_generation_args(p_full)

    args = parser.parse_args()

    if args.command == "reflections":
        generate_self_reflections(
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            output_dir=args.output_dir,
            run_id=args.run_id,
            push=args.push,
            prompt_set=args.prompt_set,
            num_samples=args.num_samples,
            concurrency=args.concurrency,
            temperature=args.temperature,
            top_p=args.top_p,
            constitution=args.constitution,
            enable_thinking=not args.no_thinking,
        )
    elif args.command == "conversations":
        generate_self_conversations(
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            output_dir=args.output_dir,
            num_turns=args.num_turns,
            run_id=args.run_id,
            push=args.push,
            interaction_style=args.interaction_style,
            num_conversations=args.num_conversations,
            concurrency=args.concurrency,
            temperature=args.temperature,
            top_p=args.top_p,
            constitution=args.constitution,
            enable_thinking=not args.no_thinking,
        )
    elif args.command == "principles":
        generate_principle_derivations(
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            output_dir=args.output_dir,
            run_id=args.run_id,
            push=args.push,
        )
    elif args.command == "filter":
        filter_introspection_data(
            output_dir=args.output_dir,
            judge_model=args.judge_model,
            run_id=args.run_id,
            push=args.push,
            filter_mode=args.filter_mode,
            min_score=args.min_score,
        )
    elif args.command == "format":
        format_for_sft(
            filtered_path=args.filtered_path,
            output_dir=args.output_dir,
            run_id=args.run_id,
            push=args.push,
        )
    elif args.command == "full":
        conv_concurrency = args.conv_concurrency if args.conv_concurrency is not None else args.concurrency
        run_full_pipeline(
            model=args.model,
            api_base=args.api_base,
            api_key=args.api_key,
            output_dir=args.output_dir,
            judge_model=args.judge_model,
            run_id=args.run_id,
            push=args.push,
            prompt_set=args.prompt_set,
            num_samples=args.num_samples,
            reflection_concurrency=args.concurrency,
            interaction_style=args.interaction_style,
            num_conversations=args.num_conversations,
            num_turns=args.num_turns,
            conversation_concurrency=conv_concurrency,
            temperature=args.temperature,
            top_p=args.top_p,
            constitution=args.constitution,
            enable_thinking=not args.no_thinking,
            filter_mode=args.filter_mode,
            min_score=args.min_score,
            skip_principles=args.skip_principles,
        )


if __name__ == "__main__":
    main()
