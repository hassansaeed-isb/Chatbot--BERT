from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

def load_saved_model(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForQuestionAnswering.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def get_relevant_paragraphs(question, paragraphs, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    question_vec = vectorizer.fit_transform([question])
    paragraphs_vec = vectorizer.transform(paragraphs)
    similarities = cosine_similarity(question_vec, paragraphs_vec).flatten()
    relevant_indices = np.argsort(similarities)[-top_n:]
    return [paragraphs[i] for i in relevant_indices]

model_path = 'model'
tokenizer, model, device = load_saved_model(model_path)

# Load the paragraphs (can be replaced with more dynamic loading if necessary)
file_path = 'textdata.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    paragraphs = file.readlines()

@app.route('/')
def home():
    return render_template('GUI.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    relevant_paragraphs = get_relevant_paragraphs(question, paragraphs, top_n=5)
    answer = " ".join(relevant_paragraphs)  # You may want to integrate the actual model prediction here if needed
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)