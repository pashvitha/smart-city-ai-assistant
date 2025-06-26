import os
from transformers import pipeline
import gradio as gr

# from getpass import getpass

# Prompt the user to enter the Hugging Face token securely
# os.environ['HF_TOKEN'] = getpass('🔑 Enter your Hugging Face token: ')
# hf_token = os.environ.get('HF_TOKEN')

hf_token = os.environ.get('HF_TOKEN')

# Load models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", token=hf_token)
qa_model = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad", token=hf_token)
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", token=hf_token)

translator_models = {
    "English to Hindi": pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi"),
    "Hindi to English": pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en"),
    "English to French": pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr"),
    "French to English": pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en"),
}

def summarize_text(text):
    summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
    return summary[0]['summary_text']

def answer_question(context, question):
    result = qa_model(question=question, context=context)
    return result['answer']

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return f"{result[0]['label']} (score: {result[0]['score']:.2f})"

def translate_text(text, direction):
    translator = translator_models[direction]
    output = translator(text)
    return output[0]['translation_text']

with gr.Blocks() as app:
    gr.Markdown("## 🌆 Sustainable Smart City AI Assistant")

    with gr.Tab("Summarizer"):
        inp = gr.Textbox(label="Enter Text to Summarize")
        out = gr.Textbox(label="Summary")
        btn = gr.Button("Summarize")
        btn.click(summarize_text, inputs=inp, outputs=out)

    with gr.Tab("Question Answering"):
        context = gr.Textbox(label="Context Paragraph")
        question = gr.Textbox(label="Question")
        answer = gr.Textbox(label="Answer")
        qa_btn = gr.Button("Get Answer")
        qa_btn.click(answer_question, inputs=[context, question], outputs=answer)

    with gr.Tab("Sentiment Analysis"):
        sent_input = gr.Textbox(label="Enter Text")
        sent_output = gr.Textbox(label="Sentiment")
        sent_btn = gr.Button("Analyze")
        sent_btn.click(analyze_sentiment, inputs=sent_input, outputs=sent_output)

    with gr.Tab("Translation"):
        trans_input = gr.Textbox(label="Enter Text")
        trans_direction = gr.Dropdown(choices=list(translator_models.keys()), label="Select Language Direction")
        trans_output = gr.Textbox(label="Translated Text")
        trans_btn = gr.Button("Translate")
        trans_btn.click(translate_text, inputs=[trans_input, trans_direction], outputs=trans_output)

app.launch(share=True)
