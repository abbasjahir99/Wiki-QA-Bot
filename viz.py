import json
from datetime import datetime
import pandas as pd

conversation_log_file = 'conversation_logs.json'

def log_conversation(user_query, response, predicted_topic, query_type, success):
    # Load existing logs
    try:
        with open(conversation_log_file, 'r') as file:
            logs = json.load(file)
    except FileNotFoundError:
        logs = []

    # Append new log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_query": user_query,
        "response": response,
        "predicted_topic": predicted_topic,
        "query_type": query_type,
        "success": success
    }
    logs.append(log_entry)

    # Save updated logs
    with open(conversation_log_file, 'w') as file:
        json.dump(logs, file, indent=4)

# Example usage:
# log_conversation("What is the capital of France?", "The capital of France is Paris.", "Geography", "wiki_qa", True)


def load_conversation_logs():
    # Load the conversation logs
    try:
        with open('conversation_logs.json', 'r') as file:
            logs = json.load(file)
    except FileNotFoundError:
        logs = []

    # Create DataFrame
    df = pd.DataFrame(logs)
    return df

# Load the data
df_logs = load_conversation_logs()
print(df_logs.head())


import matplotlib.pyplot as plt

def plot_conversations_over_time(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.resample('D').size().plot(kind='line', title='Total Number of Conversations Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Conversations')
    plt.show()

plot_conversations_over_time(df_logs)

def plot_queries_by_topic(df):
    topic_counts = df['predicted_topic'].value_counts()
    topic_counts.plot(kind='bar', title='Distribution of Queries by Topic', color='skyblue')
    plt.xlabel('Topics')
    plt.ylabel('Number of Queries')
    plt.xticks(rotation=45)
    plt.show()

plot_queries_by_topic(df_logs)



def plot_success_rate_by_topic(df):
    success_rate = df.groupby('predicted_topic')['success'].mean()
    success_rate.plot(kind='bar', title='Response Success Rate by Topic', color='green')
    plt.xlabel('Topics')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.show()

plot_success_rate_by_topic(df_logs)



def plot_average_response_length(df):
    df['response_length'] = df['response'].apply(lambda x: len(x.split()))
    avg_length = df.groupby('predicted_topic')['response_length'].mean()
    avg_length.plot(kind='bar', title='Average Response Length by Topic', color='orange')
    plt.xlabel('Topics')
    plt.ylabel('Average Response Length (words)')
    plt.xticks(rotation=45)
    plt.show()

plot_average_response_length(df_logs)



def plot_errors_by_topic(df):
    error_counts = df[df['success'] == False]['predicted_topic'].value_counts()
    error_counts.plot(kind='bar', title='Error Occurrence by Topic', color='red')
    plt.xlabel('Topics')
    plt.ylabel('Number of Errors')
    plt.xticks(rotation=45)
    plt.show()

plot_errors_by_topic(df_logs)



app = dash.Dash(__name__)

# Load and prepare data
df_logs = load_conversation_logs()

# Create plots
fig_queries_by_topic = px.bar(df_logs['predicted_topic'].value_counts(), title="Distribution of Queries by Topic")
fig_success_rate_by_topic = px.bar(df_logs.groupby('predicted_topic')['success'].mean(), title="Success Rate by Topic")

# Layout of the dashboard
app.layout = html.Div(children=[
    html.H1(children='Wiki Q/A Bot Analytics Dashboard'),

    html.Div(children='''Distribution of Queries by Topic:'''),
    dcc.Graph(
        id='queries-by-topic',
        figure=fig_queries_by_topic
    ),

    html.Div(children='''Success Rate by Topic:'''),
    dcc.Graph(
        id='success-rate-by-topic',
        figure=fig_success_rate_by_topic
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)


