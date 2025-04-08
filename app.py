import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for headless environments
import matplotlib.pyplot as plt
import io
import base64

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax

app = Flask(__name__)
CORS(app)  # Enable CORS so your React app can call this API

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

def analyze_sentiment(text):
    """
    Returns a tuple (rating, probability) for the given text.
    """
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    scores = softmax(output.logits.numpy()).flatten()
    max_index = scores.argmax()
    max_prob = scores[max_index]

    if max_index == 4 and max_prob > 0.4:
        rating = 5
    else:
        rating = round(max_index + 1)
    return max(1, min(5, rating)), float(max_prob)

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Expects JSON with:
       { "reviews": ["review text 1", "review text 2", ...] }
       
    For each review, computes sentiment analysis and aggregates the rating distribution.
    Generates a pie chart of the distribution and returns:
       - List of review analysis results,
       - Base64 encoded pie chart image.
    """
    data = request.get_json()
    reviews = data.get("reviews", [])
    
    results = []
    rating_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for review in reviews:
        rating, prob = analyze_sentiment(review)
        results.append({
            "review": review,
            "analysis": {
                "label": f"{rating} stars",
                "score": prob
            }
        })
        rating_distribution[rating] += 1

    # Generate the pie chart
    labels = list(rating_distribution.keys())
    sizes = list(rating_distribution.values())
    
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    plt.title("Reviews Distribution")
    
    # Save the plot to a memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    pie_chart_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()  # Close the figure to free memory

    return jsonify({
        "reviews": results,
        "pie_chart_base64": pie_chart_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
