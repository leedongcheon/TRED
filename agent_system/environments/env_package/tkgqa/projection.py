import re
from typing import List, Tuple


def tkgqa_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Project LLM responses into (`results`, `valids`).

    Each result is the response cut right after the first ``</action>`` (as search_projection cuts after
    the first closing tag); the symbolic executor runs the temporal operation in its first ``<action>…</action>`` block
    and reports a malformed one as an invalid action in its observation.

    ``valids[i]`` is 0 when the response has no complete ``<action>…</action>`` block or more than one
    ``<action>`` tag.
    """
    results: List[str] = []
    valids: List[int] = [1] * len(actions)
    re_action_block = re.compile(r"<action>(.*?)</action>", re.DOTALL)

    for i, action in enumerate(actions):
        cut = action.find("</action>")
        results.append(action[: cut + len("</action>")] if cut != -1 else action)

        if re_action_block.search(action) is None or action.count("<action>") > 1:
            valids[i] = 0

    return results, valids
