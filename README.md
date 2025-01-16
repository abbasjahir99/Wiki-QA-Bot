## **Wiki Q/A Bot**

### **Project Overview**
The Wiki Q/A Bot is a conversational system designed to interact with users in two modes:
- **Chit-chat**: Handles casual and conversational queries using a neural language model.
- **Wiki Q/A**: Retrieves and summarizes factual information from indexed Wikipedia documents.

The system incorporates various components, including a fine-tuned topic classifier, Solr-based document retrieval, summarization, visualization, and analytics.

---

### **Features**
- Classifies user queries as **chit-chat** or **Wiki Q/A**.
- Predicts topics from a pre-defined set using a fine-tuned transformer model.
- Retrieves relevant documents from Solr for Wiki Q/A queries.
- Summarizes retrieved information using a neural summarization model.
- Logs and visualizes query statistics for analytics.

---

### **Technologies Used**
- **Python**: Core programming language.
- **Flask**: Web framework for the bot's frontend and backend.
- **Dash**: Visualization and analytics.
- **Transformers**: Pre-trained models for question answering, summarization, and topic classification.
- **Solr**: Document indexing and retrieval.
- **Groq API**: Query classification as chit-chat or Wiki Q/A.
- **HTML&CSS**: Frontend design and layout.

---

### **Installation**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/abbasjahir99/Wiki-QA-Bot.git
   cd Wiki-QA-Bot
   ```

2. **Set up a virtual environment:**
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # For Linux/Mac
   myenv\Scripts\activate    # For Windows
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the Solr server:**
   ```bash
   cd /path/to/solr
   bin/solr start
   ```

5. **Run the Flask app:**
   ```bash
   python app.py
   ```

6. **Access the app:**
   Open your browser and navigate to `http://localhost:5000`.

---

### **How to Use**
1. **Ask a Question**:
   - Enter your question in the input field.
   - The bot will classify it as either "chit-chat" or "Wiki Q/A."

2. **Predicted Topic**:
   - If the query is Wiki Q/A, the bot will display the predicted topic beside the chat.

3. **Analytics**:
   - Navigate to `http://localhost:5000/analytics` to view visualizations of the bot's interactions.

---

### **Future Enhancements**
- Add support for multi-language Q/A.
- Improve summarization by exploring larger transformer models.
- Enhance the chit-chat model for more context-aware conversations.

---

### **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
