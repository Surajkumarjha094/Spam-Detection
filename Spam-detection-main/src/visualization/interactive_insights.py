
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.feature_extraction.text import CountVectorizer

def load_data():
    df = pd.read_csv("Data/spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

def get_spam_ham_count_chart(df):
    count_df = df['label'].value_counts().reset_index()
    count_df.columns = ['Label', 'Count']
    fig = px.bar(
        count_df, x='Label', y='Count',
        color='Label', text='Count',
        color_discrete_sequence=px.colors.qualitative.Set2,
        title='Spam vs Ham Message Count'
    )
    fig.update_layout(
        xaxis_title="Message Type",
        yaxis_title="Count",
        legend_title="Type"
    )
    return fig.to_html(full_html=False)

def get_top_words_chart(df):
    vectorizer = CountVectorizer(stop_words='english')
    
    spam_corpus = df[df['label'] == 'spam']['message']
    ham_corpus = df[df['label'] == 'ham']['message']

    spam_matrix = vectorizer.fit_transform(spam_corpus)
    ham_matrix = vectorizer.fit_transform(ham_corpus)

    spam_freq = pd.DataFrame(spam_matrix.toarray(), columns=vectorizer.get_feature_names_out()).sum().sort_values(ascending=False).head(10)
    ham_freq = pd.DataFrame(ham_matrix.toarray(), columns=vectorizer.get_feature_names_out()).sum().sort_values(ascending=False).head(10)

    spam_fig = px.bar(
        x=spam_freq.index, y=spam_freq.values,
        text=spam_freq.values,
        title='Top 10 Spam Words',
        labels={"x": "Words", "y": "Frequency"},
        color_discrete_sequence=['#EF553B']
    )
    ham_fig = px.bar(
        x=ham_freq.index, y=ham_freq.values,
        text=ham_freq.values,
        title='Top 10 Ham Words',
        labels={"x": "Words", "y": "Frequency"},
        color_discrete_sequence=['#00CC96']
    )

    return spam_fig.to_html(full_html=False), ham_fig.to_html(full_html=False)

def get_model_performance_chart():
    results = {
        "Model": ["MultinomialNB", "GaussianNB", "SVC"],
        "Accuracy": [0.96, 0.94, 0.98],
        "Precision": [0.95, 0.93, 0.97],
        "Recall": [0.94, 0.91, 0.98],
        "F1-score": [0.94, 0.92, 0.98],
    }

    df = pd.DataFrame(results)
    fig = go.Figure()

    colors = px.colors.qualitative.Prism
    for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1-score"]):
        fig.add_trace(go.Bar(name=metric, x=df["Model"], y=df[metric], marker_color=colors[i]))

    fig.update_layout(
        title="Model Comparison Metrics",
        barmode="group",
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis=dict(range=[0.8, 1.0]),
        legend_title="Metrics"
    )
    return fig.to_html(full_html=False)

def generate_all_html_charts():
    df = load_data()
    return {
        "count_chart": get_spam_ham_count_chart(df),
        "spam_chart": get_top_words_chart(df)[0],
        "ham_chart": get_top_words_chart(df)[1],
        "model_chart": get_model_performance_chart()
    }
