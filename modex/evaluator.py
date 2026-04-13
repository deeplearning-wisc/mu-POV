
import re
import random
import collections

import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from multiprocessing import Process, Queue

from math_evaluator import compute_score

ROUGE = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
smoothing = SmoothingFunction().method1


def get_instruction_suffix(args):
    base_suffix = "First, briefly state your step-by-step reasoning. Then,"

    # if args.data in ['arithmetics']:
    #     return base_suffix + ' make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 12.34}".'
    if args.data in args.math_reasoning_tasks:
        # return base_suffix + ' make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 123}".'
        return base_suffix + f' state your final answer in \\boxed{{}} at the very end of your response, just like: "final answer: \\boxed{{123}}".'
    elif args.data in args.closed_ended_tasks:
        return base_suffix + ' make sure to state your final answer choice in curly brackets at the very end of your response, just like: "{final answer: (A)}".'
    elif args.data in args.text_generation_tasks:
        return base_suffix + ' make sure to provide your summary after stating "# Answer # ".'
    elif args.data in args.code_generation_tasks:
        return '\n\n Make sure to provide ONLY your COMPLETE code after stating "# Code # ".'

def evaluate_arithmetics(responses, answer):
    # Returns True if corret, False if incorrect
    final_answers = []
    for _, response in responses.items():
        result = compute_score(response, str(answer))
        extracted_pred = result['extracted_pred']
        final_answers.append(extracted_pred)

    counter = collections.Counter([x for x in final_answers if x != ""])
    max_count = len(counter.values())
    if max_count == 0:
        most_common = [""]
    else:
        most_common = [key for key, value in counter.items() if value == max(counter.values())]
    debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    debate_is_correct = bool(compute_score(f"\\boxed{{" + str(debate_answer) + "}}", str(answer))['score'])
    
    return final_answers, debate_answer, debate_is_correct


def evaluate_basic(responses, answer):
    final_answers = []
    for _, response in responses.items():
        try:
            pred = re.findall(r"\[final answer:\s*(.*?)\]", response)[-1]
            pred = pred.strip()
            
            # Remove LaTeX delimiters \( \) if present
            pred = re.sub(r'^\\\(|\\\)$', '', pred).strip()
            
            # Remove \boxed{} if present
            pred = re.sub(r'\\boxed\{([^}]*)\}', r'\1', pred).strip()
            
            # Try to convert to float for numerical comparison
            try:
                pred_float = float(pred)
                final_answers.append(pred_float)
            except ValueError:
                # If not a simple number, keep as string for exact matching
                final_answers.append(pred)
                
        except (IndexError, ValueError):
            # If no valid final answer found, append empty string
            final_answers.append("")

    # Handle empty responses
    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":
        final_answers = [""] * len(final_answers)
        debate_answer = ""
    else:
        # Count non-empty answers
        counter = collections.Counter([x for x in final_answers if x != ""])
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common)  # if there is a tie, will choose randomly

    # Compare with the correct answer
    try:
        # Try to convert answer to float for numerical comparison
        answer_float = float(answer)
        is_correct = debate_answer == answer_float
    except (ValueError, TypeError):
        # If answer is not a simple number, do string comparison
        is_correct = str(debate_answer).strip() == str(answer).strip()

    return final_answers, debate_answer, is_correct


def evaluate_mcq(responses, answer):
    # Returns True if corret, False if incorrect
    final_answers = []
    for _, response in responses.items():

        try:
            pred = re.findall(r"\{(.*?)\}", response)[-1]
            pred = pred.replace("final answer:", "").strip()
            if len(pred) == 0 :
                final_answers.append("")
            elif len(pred) < 3 :
                pred = pred[0]
                final_answers.append(f"({pred})")
            else :
                pred = pred[1]
                final_answers.append(f"({pred})")
        except :
            final_answers.append("")
    
    if len(set(final_answers)) == 1 and list(set(final_answers))[0] == "":
        final_answers = [""] * len(final_answers)
        debate_answer = ""
    else :
        counter = collections.Counter([x for x in final_answers if x != ""])
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    return final_answers, debate_answer, [debate_answer == answer]

def evaluate_gen(responses, answer):
    final_answers = []
    debate_answer = None
    rouge1, rouge2, rougeL = 0, 0, 0
    best_score = -1
    for _, input_str in responses.items():
        summary = input_str.split("# Answer #")[-1]
        final_answers.append(summary)

        scores = ROUGE.score(answer, summary)
        rouge1 = scores['rouge1'].fmeasure
        rouge2 = scores['rouge2'].fmeasure
        rougeL = scores['rougeL'].fmeasure
        
        # Compute BLEU score
        reference = answer.split()
        candidate = summary.split()
        bleu = sentence_bleu([reference], candidate, smoothing_function=smoothing)

        if rougeL > best_score :
            debate_answer = summary 
            best_scores = [rouge1, rouge2, rougeL, bleu]
        
    return final_answers, debate_answer, best_scores

def _run_code(code, result_queue):
    """Helper to run code in separate process"""
    try:
        exec(code)
        result_queue.put(True)
    except:
        result_queue.put(False)

def evaluate_code(responses, answer):
    solution, tester = answer[0], answer[1]
    
    final_answers, correct, bleu_scores = [], [], []
    for _, input_str in responses.items():
        code = input_str + '\n\n' + tester
        final_answers.append(input_str)
        
        result_queue = Queue()
        p = Process(target=_run_code, args=(code, result_queue))
        p.start()
        p.join(timeout=60)
        
        if p.is_alive():
            # Timeout - stop the process
            p.terminate()
            p.join()
            correct.append(False)
        else:
            # Process completed
            if not result_queue.empty():
                correct.append(result_queue.get())
            else:
                correct.append(False)
        
        reference = solution.split()
        candidate = input_str.split()
        bleu = sentence_bleu([reference], candidate, smoothing_function=smoothing)
        bleu_scores.append(bleu)

    return final_answers, final_answers[0], [correct[0], bleu_scores[0]]

def base_evaluate_arithmetics(responses, answer):
    final_answers = []
    for _, sentence in responses.items():
        parts = sentence.split(" ")

        for part in parts[::-1]:
            try:
                ans = float(part)
                final_answers.append(ans)
                break
            except:
                continue

    counter = collections.Counter([x for x in final_answers if x != ""])
    try:
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    except :
        debate_answer = "" 

    return final_answers, debate_answer, debate_answer == np.round(answer, 1)

def base_evaluate_mcq(responses, answer):

    final_answers = []
    for _, input_str in responses.items():

        pattern = r'\((\w)\)'
        matches = re.findall(pattern, input_str)

        solution = None
        for match_str in matches[::-1]:
            solution = match_str.upper()
            if solution:
                final_answers.append(f"({solution})")
                break

    counter = collections.Counter([x for x in final_answers if x != ""])
    try :
        max_count = max(counter.values())
        most_common = [key for key, value in counter.items() if value == max_count]
        debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
    except :
        debate_answer = ""
    return final_answers, debate_answer, debate_answer == answer



