from flask import Flask, request, render_template, jsonify, send_from_directory
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from groq import Groq
from datetime import datetime
import torch
import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

app = Flask(__name__)

# Load the transformer models for question answering and summarization
try:
    qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    topic_classifier = pipeline("text-classification", model="fine-tuned-topic-classifier")
except Exception as e:
    print(f"Error initializing models: {e}")
    qa_model, summarizer, topic_classifier = None, None, None

# Dash App for Visualization
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/analytics/')

# Initialize the Groq API-based query classifier
class QueryClassifier:
    def __init__(self):
        self.client = Groq(api_key="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" or os.environ.get("GROQ_API_KEY"))
        self.model = "llama3-8b-8192"

    def classify_query(self, query: str) -> str:
        prompt = (
            "Task: Classify if the following query is 'chit-chat' or 'wiki_qa'.\n"
            "\n"
            "Chit-chat queries are conversational, personal, or casual, like "
            "'How are you?' or 'What's your favorite color?'\n"
            "\n"
            "Wiki QA queries seek factual information or knowledge, like "
            "'What's the capital of France?' or 'How do plants grow?'\n"
            "\n"
            f"Query: {query}\n"
            "\n"
            "Reply with only one word - either 'chit-chat' or 'wiki_qa'"
        )
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip().lower()
            if 'chit' in result:
                return 'chit-chat'
            elif 'wiki' in result:
                return 'wiki_qa'
            else:
                return 'unknown'
        except Exception as e:
            print(f"Error classifying query: {e}")
            return 'unknown'

classifier = QueryClassifier()

class Chatbot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.max_length = 100
        self.temperature = 0.85
        self.top_k = 50
        self.top_p = 0.95

    def generate_response(self, user_input):
        input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
        output_ids = self.model.generate(
            input_ids,
            max_length=self.max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=True,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            early_stopping=False
        )
        response = self.tokenizer.decode(
            output_ids[:, input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        return response.strip()

chit_chat_bot = Chatbot()

label_map = {
    "LABEL_0": "Economy",
    "LABEL_1": "Education",
    "LABEL_2": "Entertainment",
    "LABEL_3": "Environment",
    "LABEL_4": "Food",
    "LABEL_5": "Health",
    "LABEL_6": "Politics",
    "LABEL_7": "Sports",
    "LABEL_8": "Technology",
    "LABEL_9": "Travel"
}

def predict_topic(user_query):
    try:
        prediction = topic_classifier(user_query)
        predicted_label = prediction[0]['label']
        return label_map.get(predicted_label, "Unknown")
    except Exception as e:
        print(f"Error predicting topic: {e}")
        return "Unknown"

def query_solr(user_query, topic=None):
    try:
        solr_url = 'http://35.208.156.208:8983/solr/IRF24P3/select'
        params = {
            'q': user_query,
            'qf': 'title^10 summary^5 topic^15',
            'defType': 'edismax',
            'fl': 'title,summary,url,score',
            'rows': 10,
            'wt': 'json'
        }
        if topic:
            params['fq'] = f'topic:"{topic}"'
        response = requests.get(solr_url, params=params)
        if response.status_code == 200:
            return response.json().get('response', {}).get('docs', [])
        else:
            print(f"Error querying Solr: Status Code {response.status_code}")
            return []
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return []

def log_conversation(user_query, response, predicted_topic, query_type, success):
    try:
        with open('conversation_logs.json', 'r') as file:
            logs = json.load(file)
    except FileNotFoundError:
        logs = []

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_query": user_query,
        "response": response,
        "predicted_topic": predicted_topic,
        "query_type": query_type,
        "success": success
    }

    logs.append(log_entry)

    with open('conversation_logs.json', 'w') as file:
        json.dump(logs, file, indent=4)

def load_logs():
    try:
        with open('conversation_logs.json', 'r') as file:
            logs = json.load(file)
        return pd.DataFrame(logs)
    except (FileNotFoundError, json.JSONDecodeError):
        return pd.DataFrame(columns=['timestamp', 'user_query', 'response', 'predicted_topic', 'query_type', 'success'])

def prepare_visualization_data():
    df_logs = load_logs()
    if df_logs.empty:
        return None

    df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
    topics_count = df_logs['predicted_topic'].value_counts().reset_index()
    topics_count.columns = ['Topic', 'Count']

    query_type_count = df_logs['query_type'].value_counts().reset_index()
    query_type_count.columns = ['Query Type', 'Count']

    return df_logs, topics_count, query_type_count

df_logs, topics_count, query_type_count = prepare_visualization_data()

dash_app.layout = html.Div([
    html.H1("Wiki Q/A Bot Analytics", style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(
            id='topics-chart',
            figure=px.bar(
                topics_count, x='Topic', y='Count', title="Queries by Topic",
                labels={'Topic': 'Topic', 'Count': 'Number of Queries'}
            )
        ),
        dcc.Graph(
            id='query-type-chart',
            figure=px.pie(
                query_type_count, names='Query Type', values='Count', title="Query Type Distribution"
            )
        )
    ], style={'display': 'flex', 'justifyContent': 'space-around'}),
])

@app.route('/analytics', methods=['GET'])
def analytics():
    return dash_app.index()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.form['user_query']
    query_type = classifier.classify_query(user_query)
    response = ""
    predicted_topic = "None"
    success = True

    if query_type == 'chit-chat':
        response = chit_chat_bot.generate_response(user_query)
    elif query_type == 'wiki_qa':
        predicted_topic = predict_topic(user_query)
        documents = query_solr(user_query, topic=predicted_topic)
        if not documents:
            response = "Sorry, I couldn't find any relevant information."
            success = False
        else:
            response = generate_detailed_answer(user_query, documents)
    else:
        response = "I'm not sure what you're asking. Could you try rephrasing your query?"
        success = False

    log_conversation(user_query, response, predicted_topic, query_type, success)
    return jsonify({'response': response, 'predicted_topic': predicted_topic, 'query_type': query_type})

def generate_detailed_answer(user_query, documents):
    if not documents:
        return "Sorry, I couldn't find any relevant information."
    combined_text = " ".join([doc['summary'] for doc in documents])
    try:
        qa_response = qa_model(question=user_query, context=combined_text)
        answer = qa_response['answer']
        if summarizer:
            summarized_response = summarizer(answer + " " + combined_text, max_length=300, min_length=150, do_sample=False)[0]['summary_text']
            return summarized_response
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error while generating the answer."

if __name__ == '__main__':
    app.run(debug=True)
