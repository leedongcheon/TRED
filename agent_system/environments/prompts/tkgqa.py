"""Actor prompt of the TRED agent (search-style: the symbolic executor of the temporal operations plays the search engine).

{operations_spec} is the compact operation document (tkgqa.operations.OPERATIONS_DOC_COMPACT); {memory_context} is the full history
(<action> / <observation> pairs, agent_system.memory.TKGQAMemory). The question is repeated right above the final
instruction, which is the anchor where skill blocks are inserted ("Now it's your turn"). The agent answers with one
operation and no reasoning text. The prompt text says "DSL" for the temporal operations; it is kept verbatim because the
released checkpoints were trained with it.
"""
from tkgqa.operations import OPERATIONS_DOC_COMPACT as TKGQA_OPERATIONS_SPEC

TKGQA_TEMPLATE_NO_HIS = """
You are an expert agent tasked with answering the given temporal question over a temporal knowledge graph step-by-step.
Your question: {task_description}

{operations_spec}

Reminder — your question: {task_description}

Now it's your turn to respond for the current step.
Respond with exactly one DSL operation and nothing else, using format: <action> fN = OP(...) </action>
(or tN = ..., sN = ...). The result will be returned to you as an observation.
When a value set sN holds the final answer, finish with <action> ANSWER(sN) </action>.
"""

TKGQA_TEMPLATE = """
You are an expert agent tasked with answering the given temporal question over a temporal knowledge graph step-by-step.
Your question: {task_description}

{operations_spec}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history where <action> </action> wrapped your past DSL operations and <observation> </observation> wrapped the corresponding results returned by the executor. History:
{memory_context}

Reminder — your question: {task_description}

Now it's your turn to respond for the current step.
Respond with exactly one DSL operation and nothing else, using format: <action> fN = OP(...) </action>
(or tN = ..., sN = ...). The result will be returned to you as an observation.
When a value set sN holds the final answer, finish with <action> ANSWER(sN) </action>.
"""
