from dash import dcc, html
import plotly.express as px
import json
from datetime import datetime
import pandas as pd

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

