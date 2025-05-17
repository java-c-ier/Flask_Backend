import io
import base64
from collections import Counter

import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax

app = Flask(__name__)
CORS(app)

# Load pretrained BERT sentiment model
tokenizer = BertTokenizer.from_pretrained(
    'nlptown/bert-base-multilingual-uncased-sentiment'
)
model = BertForSequenceClassification.from_pretrained(
    'nlptown/bert-base-multilingual-uncased-sentiment'
)

def analyze_sentiment(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    scores = softmax(output.logits.numpy()).flatten()
    max_idx = int(scores.argmax())
    max_prob = float(scores[max_idx])

    # map index to rating 1–5
    rating = max_idx + 1
    rating = max(1, min(5, rating))
    return rating, max_prob

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Expects JSON:
      {
        "reviews": {
          "Cleanliness_and_Hygiene": [...],
          "Doctor_and_Staff_Behaviour": [...],
          "Quality_of_Care": [...],
          "Wait_Times_and_Efficiency": [...]
        }
      }

    Returns:
      {
        "Cleanliness_and_Hygiene": {
          "reviews": [...],
          "average_rating": x.x,
          "pie_chart_base64": "...",
          "bar_chart_base64": "..."
        },
        "Doctor_and_Staff_Behaviour": { ... },
        "Quality_of_Care": { ... },
        "Wait_Times_and_Efficiency": { ... }
      }
    """
    data = request.get_json()
    categories = data.get("reviews", {})

    output = {}
    for aspect, reviews_list in categories.items():
        # sentiment analysis per aspect
        results = []
        dist = Counter()
        for text in reviews_list:
            rating, prob = analyze_sentiment(text)
            results.append({
                "review": text,
                "analysis": {
                    "label": f"{rating} stars",
                    "score": prob
                }
            })
            dist[rating] += 1

        # compute average rating
        total = sum(dist.values())
        avg = (sum(r * c for r, c in dist.items()) / total) if total else 0.0

        # ensure 1–5 keys
        for i in range(1, 6):
            dist.setdefault(i, 0)

        labels = list(dist.keys())
        counts = [dist[i] for i in labels]

        # pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(counts, labels=labels, autopct='%1.1f%%')
        # plt.title(f"{aspect.replace('_', ' ')} Distribution")
        buf1 = io.BytesIO()
        plt.savefig(buf1, format="png")
        buf1.seek(0)
        pie_b64 = base64.b64encode(buf1.read()).decode('utf-8')
        plt.close()

        # bar chart
        plt.figure(figsize=(6, 4))
        ax = plt.gca()
        bars = ax.bar(labels, counts)
        ax.set_xlabel("Star Rating")
        ax.set_ylabel("Number of Reviews")
        # ax.set_title(f"{aspect.replace('_', ' ')} Distribution", pad=40)  
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        max_count = max(counts) if counts else 0
        ax.set_yticks(range(0, max_count + 1))
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.1,
                str(int(h)),
                ha='center',
                va='bottom'
            )
        buf2 = io.BytesIO()
        plt.savefig(buf2, format="png", bbox_inches='tight')
        buf2.seek(0)
        bar_b64 = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close()

        output[aspect] = {
            "reviews": results,
            "average_rating": round(avg, 2),
            "pie_chart_base64": pie_b64,
            "bar_chart_base64": bar_b64
        }

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
