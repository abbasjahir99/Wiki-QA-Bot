import os
from groq import Groq

class QueryClassifier:
    def __init__(self):
        """Initialize the classifier with the Groq API."""
        self.client = Groq(api_key="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" or os.environ.get("GROQ_API_KEY"))
        self.model = "llama3-8b-8192"
    
    def classify_query(self, query: str) -> str:
        """
        Classify whether a query is conversational or knowledge-seeking.
        Returns 'chit-chat' or 'wiki_qa'.
        """
        # This prompt maintains a good balance between clarity and simplicity
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

# Example usage
if __name__ == "__main__":
    classifier = QueryClassifier()
    
    test_queries = [
        "How are you today?",
        "What is the capital of France?",
        "Do you like pizza?",
        "Who invented the telephone?"
    ]
    
    for query in test_queries:
        result = classifier.classify_query(query)
        print(f"Query: {query}")
        print(f"Type: {result}\n")