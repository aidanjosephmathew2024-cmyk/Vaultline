from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# First, let's SEE the label mapping the model actually has
print("Label mapping:", model.config.id2label)

sample_headlines = [
    "Gold prices surge as investors flee to safe havens",
    "Commercial real estate market faces mounting distress",
    "Bond yields hold steady amid mixed economic data"
]

for headline in sample_headlines:
    inputs = tokenizer(headline, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    scores = {model.config.id2label[i]: round(probs[0][i].item(), 3) for i in range(len(probs[0]))}
    print(f"\n{headline}")
    print(f"  {scores}")