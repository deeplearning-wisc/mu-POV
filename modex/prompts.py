
import random


def create_initial_messages(question, num_agents, personas=None, suffix=""):
    """
    Build the list of chat message dicts for the initial round of generation.

    If *personas* is provided (a dict mapping name -> system prompt), each
    agent receives a distinct system prompt; otherwise all agents share a
    plain user-only message.
    """
    if personas:
        messages = [
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question + suffix},
            ]
            for sys_prompt in personas.values()
        ]
    else:
        messages = [[{"role": "user", "content": question + suffix}] for _ in range(num_agents)]

    return messages
