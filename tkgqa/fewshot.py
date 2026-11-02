"""Solved examples and the message of the Stage 1 teacher, on a small fictional temporal KG.

The examples use invented entities, so no benchmark question, entity or relation wording leaks into the teacher runs,
and the same examples serve every dataset. Their observations come from running the real executor, so they always
match the current observation format. Each step is written as the agent writes it: one <action> block. Example 2
recovers from a failed FIND and from an invalid action, and example 6 from writing a period as two dates; they are the
only places where the teacher sees how to react to error observations. Everything here is teacher-only: it never
appears in SFT or RL prompts, and it is versioned by fewshot_version (not prompt_version).
"""

import hashlib
from typing import Dict, List, Optional, Tuple

from tkgqa.answers import score_answer
from tkgqa.executor import Session
from tkgqa.grounder import Grounder
from tkgqa.kg import TemporalKG

TOY_FACTS = [
    ("Velora", "sign agreement with", "Kestria", "2010-06-05", "2010-06-05"),
    ("Velora", "sign agreement with", "Kestria", "2010-11-20", "2010-11-20"),
    ("Velora", "sign agreement with", "Kestria", "2011-03-14", "2011-03-14"),
    ("Duvalia", "sign agreement with", "Kestria", "2011-02-02", "2011-02-02"),
    ("Velora", "sign agreement with", "Kestria", "2011-09-30", "2011-09-30"),
    ("Taro Senn", "pay a visit to", "Velora", "2012-05-08", "2012-05-08"),
    ("Taro Senn", "pay a visit to", "Velora", "2013-01-17", "2013-01-17"),
    ("Ilsa Brandt", "criticize", "Velora", "2012-04-30", "2012-04-30"),
    ("Duvalia", "criticize", "Velora", "2012-06-11", "2012-06-11"),
    ("Kestria", "criticize", "Velora", "2012-06-25", "2012-06-25"),
    ("Mira Holt", "serve as", "council chair", "2001-01-01", "2004-01-01"),
    ("Mira Holt", "serve as", "council chair", "2006-01-01", "2008-01-01"),
    ("Mira Holt", "serve as", "fleet steward", "1998-01-01", "2001-01-01"),
    ("Ilsa Brandt", "pay a visit to", "Duvalia", "2002-07-04", "2002-07-04"),
    ("Taro Senn", "pay a visit to", "Duvalia", "2005-03-12", "2005-03-12"),
    ("Kestria", "pay a visit to", "Duvalia", "2007-10-09", "2007-10-09"),
    ("Taro Senn", "belong to", "Harbor Council", "1995-01-01", "1999-01-01"),
    ("Ilsa Brandt", "win award", "Silverleaf Prize", "1997-01-01", "1997-01-01"),
    ("Mira Holt", "study at", "Northgate University", "1990-01-01", "1994-01-01"),
]

# (question, gold answers, answer kind, [action, ...])
EXAMPLES: List[Tuple[str, List[str], str, List[str]]] = [
    (
        'In which month did Velora last sign an agreement with Kestria before Duvalia first did?',
        ['2010-11'],
        'time',
        [
            'f1 = FIND("Duvalia", "sign agreement with", "Kestria")',
            't1 = TIME(f1, first)',
            'f2 = FIND("Velora", "sign agreement with", "Kestria")',
            'f3 = FILTER(f2, before, t1, gran=day)',
            'f4 = PICK(f3, latest)',
            's1 = GET(f4, time, gran=month)',
            'ANSWER(s1)',
        ],
    ),
    (
        'After Taro Senn first visited Velora, who was the first to criticize Velora?',
        ['Duvalia'],
        'entity',
        [
            'f1 = FIND("Taro Senn", "visit", "Velora")',
            'f1 = FIND("Taro Senn", "pay a visit to", "Velora")',
            't1 = TIME(f1, first)',
            'f2 = FIND("?", "criticize", "Velora")',
            'f3 = FILTER(f2, after, t1, gran=day)',
            'f4 = PICK(f3, first)',
            'f4 = PICK(f3, earliest)',
            's1 = GET(f4, subj)',
            'ANSWER(s1)',
        ],
    ),
    (
        'Who visited Duvalia while Mira Holt was council chair?',
        ['Ilsa Brandt', 'Kestria'],
        'entity',
        [
            'f1 = FIND("Mira Holt", "serve as", "council chair")',
            't1 = TIME(f1, all)',
            'f2 = FIND("?", "pay a visit to", "Duvalia")',
            'f3 = FILTER(f2, during, t1, gran=day)',
            's1 = GET(f3, subj)',
            'ANSWER(s1)',
        ],
    ),
    (
        'How many days in total did Mira Holt serve as council chair?',
        ['1825 days'],
        'number',
        [
            'f1 = FIND("Mira Holt", "serve as", "council chair")',
            's1 = AGG(f1, sum)',
            'ANSWER(s1)',
        ],
    ),
    (
        "By start time, what is the rank of Taro Senn's membership in Harbor Council among these events: Taro Senn's membership in Harbor Council, Ilsa Brandt winning the Silverleaf Prize, and Mira Holt's study at Northgate University?",
        ['2'],
        'number',
        [
            'f1 = FIND("Taro Senn", "belong to", "Harbor Council")',
            'f2 = FIND("Ilsa Brandt", "win award", "Silverleaf Prize")',
            'f3 = FIND("Mira Holt", "study at", "Northgate University")',
            'f4 = UNION(f1, f2, f3)',
            's1 = RANK(f4, f1, by=start)',
            'ANSWER(s1)',
        ],
    ),
    (
        'Who signed an agreement with Kestria in March 2011?',
        ['Velora'],
        'entity',
        [
            'f1 = FIND("?", "sign agreement with", "Kestria")',
            'f2 = FILTER(f1, during, 2011-03-01, 2011-03-31, gran=day)',
            'f2 = FILTER(f1, during, 2011-03, gran=month)',
            's1 = GET(f2, subj)',
            'ANSWER(s1)',
        ],
    ),
]

FEWSHOT_HEADER = """\
Below are solved examples on a small illustrative knowledge graph. They show the expected DSL usage, one
operation per step, in exactly the format you must produce: one <action> block per step and nothing else
(no reasoning text). Your task uses a different graph, so take names only from your question and from the
observations you receive; never reuse names from the examples.
"""


TEACHER_MESSAGE = """\
{fewshot_header}

{fewshot_examples}

### Your task
{actor_prompt}
"""

# Second and later attempts only, right above "### Your task" (teacher-only; the gold answer is never given,
# which would invite spurious trajectories fitted to it).
TEACHER_RETRY_NOTE = """\
Note: a previous attempt at this question ended with the answer "{prev_answer}", which is not correct.
Reconsider the reference event, the FILTER relation and gran, and which side GET should return.
"""


def toy_kg() -> TemporalKG:
    return TemporalKG.from_facts(TOY_FACTS, dataset="toy")


def run_example(kg: TemporalKG, example) -> Tuple[Session, List[str]]:
    question, gold, kind, actions = example
    session = Session(kg, Grounder(kg), question, max_steps=len(actions) + 1)
    responses = [f"<action> {action} </action>" for action in actions]
    for response in responses:
        session.step(response)
    if not session.done or not score_answer(session.answer or [], gold, kind)["em"]:
        raise RuntimeError(f"few-shot example does not reach its answer: {question!r} -> {session.answer}")
    return session, responses


def render_fewshot(examples: Optional[list] = None) -> str:
    """The solved examples ({fewshot_examples}): per step the response and the executor's observation."""
    kg = toy_kg()
    blocks = []
    for index, example in enumerate(examples or EXAMPLES, 1):
        session, responses = run_example(kg, example)
        lines = [f"### Example {index}", f"Question: {example[0]}"]
        for k, (step, response) in enumerate(zip(session.log, responses)):
            lines += ["", f"Step {k + 1}", response, f"<observation> {step.obs} </observation>"]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def fewshot_version(fewshot_examples: Optional[str] = None) -> str:
    """Hash of everything the teacher sees besides the actor prompt."""
    examples = render_fewshot() if fewshot_examples is None else fewshot_examples
    key = "\n".join([FEWSHOT_HEADER, TEACHER_MESSAGE, TEACHER_RETRY_NOTE, examples])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def teacher_message(fewshot_examples: str, actor_prompt: str, prev_answer: Optional[str] = None) -> List[Dict[str, str]]:
    """Single user message: header, solved examples, [retry note,] then the unchanged actor prompt."""
    text = TEACHER_MESSAGE.format(fewshot_header=FEWSHOT_HEADER, fewshot_examples=fewshot_examples, actor_prompt=actor_prompt)
    if prev_answer is not None:
        note = TEACHER_RETRY_NOTE.format(prev_answer=prev_answer)
        text = text.replace("\n### Your task\n", f"\n{note}\n### Your task\n", 1)
    return [{"role": "user", "content": text}]
