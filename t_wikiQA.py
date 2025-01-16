from transformers import pipeline
import json
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
import os
import os
from transformers import pipeline
import json
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

# Load the Groq API for query classification
from groq import Groq

# Flask imports
from flask import Flask, request, jsonify, render_template

# Load the question answering and summarization models
try:
    qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
except Exception as e:
    print(f"Error initializing models: {e}")
    qa_model = None
    summarizer = None

# Load the fine-tuned transformer model for topic classification
topic_classifier = pipeline("text-classification", model="fine-tuned-topic-classifier")

# Label to topic mapping dictionary
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

# Function to predict the topic using the fine-tuned transformer model
def predict_topic_transformer(user_query):
    try:
        prediction = topic_classifier(user_query)
        predicted_label = prediction[0]['label']
        # Map the label to the topic using the label_map dictionary
        predicted_topic = label_map.get(predicted_label, "Unknown")
        return predicted_topic
    except Exception as e:
        print(f"Error predicting topic using transformer model: {e}")
        return None

# Function to query Solr for relevant documents
def query_solr(user_query, topics=None):
    try:
        solr_query_url = 'http://35.208.156.208:8983/solr/IRF24P3/select'
        params = {
            'q': f'{user_query}',  # Use the user query as it is
            'qf': 'title^10 summary^5 topic^15',  # Boost topic and title more to improve relevance
            'defType': 'edismax',  # Use Extended DisMax query parser for better text handling
            'fl': 'title,summary,url,score',
            'rows': 20,  # Retrieve top 20 documents for broader context
            'wt': 'json'
        }

        # Apply topic filter if multiple topics are provided to narrow down the results
        if topics:
            topic_filter = " OR ".join([f'topic:"{topic}"' for topic in topics])
            params['fq'] = topic_filter

        response = requests.get(solr_query_url, params=params)
        if response.status_code == 200:
            docs = response.json().get('response', {}).get('docs', [])
            if not docs:
                print("No documents were found for this query.")
            return docs
        else:
            print(f"Error querying Solr: Status Code {response.status_code}, Response: {response.text}")
            return []
    except requests.RequestException as e:
        print(f"Request error while querying Solr: {e}")
        return []

# Function to select the most relevant documents before passing to QA
def extract_relevant_documents(user_query, documents):
    if not documents:
        return []

    summaries = [doc['summary'] for doc in documents]
    vectorizer = TfidfVectorizer()
    summary_vectors = vectorizer.fit_transform(summaries)
    query_vector = vectorizer.transform([user_query])

    # Calculate cosine similarity between user query and document summaries
    similarities = cosine_similarity(query_vector, summary_vectors).flatten()
    top_indices = np.argsort(similarities)[-5:]  # Get top 5 most similar summaries

    # Extract relevant documents
    relevant_docs = [documents[idx] for idx in top_indices if similarities[idx] > 0.15]  # Set a similarity threshold
    return relevant_docs

# Function to generate a detailed answer using both extractive and abstractive methods
def generate_detailed_answer(user_query, documents):
    if not documents:
        return "Sorry, I couldn't find any relevant information to answer your question."

    # Combine the selected summaries into one text to use as context
    combined_text = " ".join([doc['summary'] for doc in documents])

    # Use the QA model to find the answer within the combined text
    try:
        if qa_model is None:
            return "Question-answering feature is currently unavailable. Here are some details from the retrieved documents."

        qa_response = qa_model(question=user_query, context=combined_text)
        qa_answer = qa_response['answer']

        # Combine QA response with the retrieved summaries for additional context
        full_context = qa_answer + " " + combined_text

        # Use the summarizer to generate a more comprehensive response
        if summarizer is not None:
            summarized_response = summarizer(
                full_context, max_length=300, min_length=150, do_sample=False, clean_up_tokenization_spaces=True
            )[0]['summary_text']
            return summarized_response
        else:
            return qa_answer + " (Note: Summarization model not available to provide more details.)"

    except Exception as e:
        print(f"Error during QA or summarization: {e}")
        return "There was an error generating a detailed answer. Here are some details from the retrieved documents instead."

# Save and load conversation state to/from a JSON file
def save_conversation_state(conversation_state, filename='conversation_history.json'):
    with open(filename, 'w') as f:
        json.dump(conversation_state, f, indent=4)

def load_conversation_state(filename='conversation_history.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            # Convert topics_discussed back to a set
            data['topics_discussed'] = set(data['topics_discussed'])
            return data
    else:
        return {'topics_discussed': set(), 'history': defaultdict(list)}

class QueryClassifier:
    def __init__(self):
        """Initialize the classifier with the Groq API."""
        self.client = Groq(api_key="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" or os.environ.get("GROQ_API_KEY"))
        self.model = "llama3-8b-8192"

    def classify_query(self, query: str) -> str:
        """Classify if the query is 'chit-chat' or 'wiki_qa'."""
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

# Instantiate the classifier
classifier = QueryClassifier()

# Full Q/A bot function combining all steps with enhanced exception handling
def wiki_qa_chat():
    print("Welcome to the Wiki Q/A Bot! Type your questions below or type 'exit' to end the chat.")
    
    # Initialize conversation state
    conversation_state = load_conversation_state()

    while True:
        try:
            # Step 1: Get user input
            user_query = input("You: ").strip()

            if user_query.lower() in ['exit', 'quit']:
                print("Wiki Q/A Bot: Goodbye! Feel free to come back if you have more questions.")
                # Save conversation state before exiting
                save_conversation_state(conversation_state)
                break

            # Step 2: Classify the user query as either 'chit-chat' or 'wiki_qa'
            query_type = classifier.classify_query(user_query)
            if query_type == 'chit-chat':
                # Handle chit-chat queries (For simplicity, just provide a generic response)
                chit_chat_response = "I'm just a bot, but I'm here to chat with you! 😊 How can I assist you today?"
                print(f"Wiki Q/A Bot: {chit_chat_response}")
                continue
            elif query_type == 'wiki_qa':
                # Proceed with Wiki Q/A workflow
                # Predict the topic of the user's query
                predicted_topic = predict_topic_transformer(user_query)
                if predicted_topic is None:
                    response = "Could you please provide more details or specify the topic?"
                    print(f"Wiki Q/A Bot: {response}")
                    continue

                # Step 3: Check if the user is revisiting an earlier topic
                if predicted_topic in conversation_state['topics_discussed']:
                    # If the topic was already discussed, provide a reminder about previous conversation
                    print(f"Wiki Q/A Bot: It looks like we're revisiting the topic of '{predicted_topic}'. Here's what we discussed earlier:")
                    for entry in conversation_state['history'][predicted_topic]:
                        print(f"- {entry}")

                # Update conversation context
                conversation_state['topics_discussed'].add(predicted_topic)
                topics_to_retrieve = list(conversation_state['topics_discussed'])

                print(f"Predicted Topic for the query: {predicted_topic}")

                # Step 4: Query Solr to retrieve relevant documents considering multiple topics
                documents = query_solr(user_query, topics=topics_to_retrieve)
                if not documents:
                    response = "Could you try rephrasing your query or asking about something else?"
                    print(f"Wiki Q/A Bot: {response}")
                    continue

                # Step 5: Extract the most relevant documents
                relevant_docs = extract_relevant_documents(user_query, documents)
                if not relevant_docs:
                    response = "Could you provide more details?"
                    print(f"Wiki Q/A Bot: {response}")
                    continue

                # Step 6: Generate a detailed answer using the hybrid approach
                detailed_answer = generate_detailed_answer(user_query, relevant_docs)
                print(f"Wiki Q/A Bot: {detailed_answer}")

                # Step 7: Store the conversation history
                conversation_state['history'][predicted_topic].append(f"User: {user_query} -> Bot: {detailed_answer}")

            else:
                # Handle unknown cases
                print("Wiki Q/A Bot: I'm not sure what you're asking. Could you try rephrasing it?")

        except Exception as e:
            print(f"Wiki Q/A Bot: An unexpected error occurred. Please try again later. (Error: {e})")

# Run the Wiki Q/A chat bot
wiki_qa_chat()
