from flask import Flask, request, render_template
from transformers import pipeline
from transformers import BertTokenizerFast, BertForSequenceClassification


app = Flask(__name__)

model = BertForSequenceClassification.from_pretrained("./reddit_sentiment_model")
tokenizer = BertTokenizerFast.from_pretrained("./reddit_sentiment_model")


sentiment_pipeline = pipeline("text-classification",model=model, tokenizer=tokenizer)

label_map = {"LABEL_0":"Negative", "LABEL_1":"Neutral", "LABEL_2": "Positive"}

@app.route('/', methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        comment = request.form.get("comment")
        comment = [comment]
        
        sentiment = sentiment_pipeline(comment)
        
        label = sentiment[0]['label']
        label = label_map[label]
        result = f"Sentiment : {sentiment}"
    return render_template("index.html", result=result ,label = label)

if __name__ == '__main__':
    app.run(debug=True)

