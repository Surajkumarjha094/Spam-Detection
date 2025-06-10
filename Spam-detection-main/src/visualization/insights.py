# src/visualization/insights.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

sns.set(style="whitegrid")

def load_data():
    # Change this path to your actual dataset path if needed
    df = pd.read_csv("Data/spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

def create_bar_chart(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title("Count of Spam vs Ham Messages")
    plt.savefig("static/plots/spam_ham_count.png")
    plt.close()

def create_wordclouds(df):
    spam_words = " ".join(df[df['label']=='spam']['message'])
    ham_words = " ".join(df[df['label']=='ham']['message'])

    spam_wc = WordCloud(width=800, height=400, background_color='white').generate(spam_words)
    ham_wc = WordCloud(width=800, height=400, background_color='white').generate(ham_words)

    spam_wc.to_file("static/plots/spam_wordcloud.png")
    ham_wc.to_file("static/plots/ham_wordcloud.png")

def create_word_frequency_histogram(df):
    vectorizer = CountVectorizer(stop_words='english')
    
    spam_matrix = vectorizer.fit_transform(df[df['label']=='spam']['message'])
    ham_matrix = vectorizer.fit_transform(df[df['label']=='ham']['message'])

    spam_freq = pd.DataFrame(spam_matrix.toarray(), columns=vectorizer.get_feature_names_out()).sum().sort_values(ascending=False).head(20)
    ham_freq = pd.DataFrame(ham_matrix.toarray(), columns=vectorizer.get_feature_names_out()).sum().sort_values(ascending=False).head(20)

    fig, axes = plt.subplots(1, 2, figsize=(16,6))
    sns.barplot(x=spam_freq.values, y=spam_freq.index, ax=axes[0], palette='Reds_r')
    axes[0].set_title("Top 20 Spam Words")

    sns.barplot(x=ham_freq.values, y=ham_freq.index, ax=axes[1], palette='Blues_r')
    axes[1].set_title("Top 20 Ham Words")

    plt.tight_layout()
    plt.savefig("static/plots/word_frequency.png")
    plt.close()

def create_model_performance_chart():
    results = {
        "Model": ["MultinomialNB", "GaussianNB", "SVC"],
        "Accuracy": [0.96, 0.94, 0.98],
        "Precision": [0.95, 0.93, 0.97],
        "Recall": [0.94, 0.91, 0.98],
        "F1-score": [0.94, 0.92, 0.98],
    }

    df = pd.DataFrame(results)

    df.set_index("Model").plot(kind="bar", figsize=(10,6), colormap="Set2")
    plt.title("Model Comparison: Accuracy, Precision, Recall, F1")
    plt.ylabel("Score")
    plt.ylim(0.8, 1.0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("static/plots/model_comparison.png")
    plt.close()

def generate_all_insights():
    os.makedirs("static/plots", exist_ok=True)

    df = load_data()
    create_bar_chart(df)
    create_wordclouds(df)
    create_word_frequency_histogram(df)
    create_model_performance_chart()
