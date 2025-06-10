# src/storytelling/analysis_text.py

import pandas as pd
from collections import Counter
import numpy as np

def generate_story_insights():
    df = pd.read_csv("Data/spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']

    total = len(df)
    spam_count = df['label'].value_counts().get('spam', 0)
    ham_count = df['label'].value_counts().get('ham', 0)

    spam_ratio = round((spam_count / total) * 100, 2)
    ham_ratio = round((ham_count / total) * 100, 2)

    avg_spam_length = df[df['label'] == 'spam']['message'].apply(len).mean()
    avg_ham_length = df[df['label'] == 'ham']['message'].apply(len).mean()

    spam_words = " ".join(df[df['label'] == 'spam']['message']).lower().split()
    spam_common = Counter(spam_words).most_common(5)

    ham_words = " ".join(df[df['label'] == 'ham']['message']).lower().split()
    ham_common = Counter(ham_words).most_common(5)

    summary = [
        f"Out of {total} messages, about {ham_ratio}% are ham and {spam_ratio}% are spam.",
        f"Spam messages tend to be shorter and repetitive. Average spam length is {round(avg_spam_length)} characters vs {round(avg_ham_length)} for ham.",
        f"Most common spam words include: {', '.join([word for word, _ in spam_common if word.isalpha()])}.",
        f"Most common ham words include: {', '.join([word for word, _ in ham_common if word.isalpha()])}.",
        f"This analysis can help filter messages by identifying patterns in word usage and structure."
    ]

    return summary
