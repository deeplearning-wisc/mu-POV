
from datasets import load_dataset
import pandas as pd

def load_data(args, split='validation'):
    dataset = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', cache_dir=args.data_dir, token=args.token)['train']
    dataset = pd.DataFrame(dataset)
    dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(args.data_size)

    questions, labels = [], []
    template = '{}\n(A) {}\n(B) {}\n(C) {}\n(D) {}\n\n'
    for ctx, A, B, C, D in zip(dataset['Question'], dataset['Correct Answer'], dataset['Incorrect Answer 1'], dataset['Incorrect Answer 2'], dataset['Incorrect Answer 3']):
        
        question = template.format(ctx, A, B, C, D)
        label = f"(A)"
        questions.append(question)
        labels.append(label)
    
    return questions, labels

