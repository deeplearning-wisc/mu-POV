
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


def create_adaptive_sampling_prompt(question, agent_responses):
    """
    Build a prompt that conditions a new sample on a set of existing responses.
    Used by the adaptive-sampling fallback in ModeX.
    """
    prompt = f"Question:\n{question}\n\nBased on the following example responses:"
    for i, response in enumerate(agent_responses):
        prompt += f"\n\n- Response {i+1}: {response}"
    prompt += "\n\nInstructions: Consider these example responses to provide a new response to the question.\n"
    return prompt
