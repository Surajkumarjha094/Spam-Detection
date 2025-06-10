# src/visualization/interactive_elements.py

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

def load_data():
    df = pd.read_csv("Data/spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

def filter_data(df, label_filter):
    if label_filter == "all":
        return df
    return df[df['label'] == label_filter]

def generate_chart(df, chart_type):
    count_df = df['label'].value_counts().reset_index()
    count_df.columns = ['Label', 'Count']

    if chart_type == "bar":
        fig = px.bar(
            count_df, x='Label', y='Count', color='Label', text='Count',
            title=f"{chart_type.capitalize()} Chart: Message Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
    elif chart_type == "pie":
        fig = px.pie(
            count_df, names='Label', values='Count',
            title="Pie Chart: Message Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
    else:
        return None

    fig.update_traces(textposition='inside')
    return fig.to_html(full_html=False)
