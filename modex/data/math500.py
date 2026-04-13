
from datasets import load_dataset
import pandas as pd

def load_data(args, split='test'):

    dataset = load_dataset('HuggingFaceH4/MATH-500', 'default', cache_dir=args.data_dir)['test']
    dataset = pd.DataFrame(dataset)

    if split == 'test':
        dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(args.data_size)
    else :
        dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True).head(300)
    
    questions, labels = [], []
    for question, answer in zip(dataset['problem'], dataset['answer']) :
        questions.append(question)
        labels.append(answer)

    return questions, labels

