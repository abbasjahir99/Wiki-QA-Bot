import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

class Chatbot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """
        Initialize the neural chatbot using a pre-trained language model.
        We use DialoGPT as it's specifically trained on conversation data,
        making it suitable for natural dialogue generation.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Generation parameters that affect response diversity and quality
        self.max_length = 100
        self.temperature = 0.85  # Higher temperature for more creative responses
        self.top_k = 50  # Limit vocabulary to top K tokens
        self.top_p = 0.95  # Nucleus sampling parameter
        
        # Track conversation state
        self.is_chatting = True

    def generate_response(self, user_input):
        """
        Generate a response using neural language modeling.
        This method implements several techniques to ensure responses are:
        1. Natural and non-rule-based
        2. Contextually appropriate
        3. Diverse across multiple interactions
        """
        # Encode user input for the model
        input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
        
        # Generate response using advanced sampling techniques
        output_ids = self.model.generate(
            input_ids,
            max_length=self.max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=True,  # Enable sampling for non-deterministic outputs
            num_return_sequences=1,
            no_repeat_ngram_size=3,  # Prevent repetitive phrases
            early_stopping=False
        )
        
        # Decode the response, excluding the input prompt
        response = self.tokenizer.decode(
            output_ids[:, input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        
        return response.strip()

    def check_end_chat(self, user_input):
        """
        Check if user intends to end the chat using simple keyword matching.
        This is the only rule-based component and is separate from the core chat functionality.
        """
        end_phrases = {'goodbye', 'bye', 'quit', 'exit'}
        return any(phrase in user_input.lower() for phrase in end_phrases)

    def chat(self):
        """
        Main chat loop that maintains conversation until user decides to end it.
        """
        print("Hello! I'm a chatbot. Feel free to chat with me! (Say 'goodbye' to end)")
        
        while self.is_chatting:
            user_input = input("You: ").strip()
            
            if not user_input:
                print("Bot: I didn't catch that. Could you say something?")
                continue
                
            if self.check_end_chat(user_input):
                print("Bot: Goodbye! It was nice chatting with you!")
                self.is_chatting = False
                continue
            
            try:
                # Generate non-rule-based response using neural LM
                response = self.generate_response(user_input)
                print(f"Bot: {response}")
                
            except Exception as e:
                print(f"Bot: I'm having trouble processing that. Could you rephrase?")
                print(f"Debug info: {str(e)}")

def main():
    """
    Initialize and run the chatbot
    """
    chatbot = Chatbot()
    chatbot.chat()

if __name__ == "__main__":
    main()