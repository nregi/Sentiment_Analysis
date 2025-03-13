import gradio as gr
from transformers import pipeline

model = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = model(text)[0]
    return f"{result['label']} (Score: {result['score']:.2f})"

iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter text here..."),
    outputs=gr.Textbox(label="Sentiment Output"),
    title="Sentiment Analysis API",
    description="Enter a sentence, and the model will predict if it's POSITIVE or NEGATIVE."
)

iface.launch()
