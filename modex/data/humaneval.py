
from datasets import load_dataset
import pandas as pd

def load_data(args, split='validation'):
    dataset = load_dataset('openai/openai_humaneval', cache_dir=args.data_dir, token=args.token)['test']
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(args.data_size)

    questions, labels = [], []
    template = 'Complete the following code:\n\n{}'
    for prompt, solution, tester, func_name in zip(dataset['prompt'], dataset['canonical_solution'], dataset['test'], dataset['entry_point']):
        question = template.format(prompt)
        questions.append(question)
        tester = tester + "\n\n" + f"check({func_name})"
        labels.append([solution, tester])
    
    return questions, labels

