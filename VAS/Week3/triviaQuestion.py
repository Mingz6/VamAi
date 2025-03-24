# Use Copilot to generate a simple trivia question game using Gradio
# Prompt: create a simple proof of concept that gives me trivia questions and allows me to answer in gradio 5.10

import gradio as gr

# Sample trivia questions
trivia_questions = {
    "What is the capital of France?": "Paris",
    "What is 2 + 2?": "4",
    "Who wrote 'To Kill a Mockingbird'?": "Harper Lee"
}

def ask_question(question):
    return question

def check_answer(question, answer):
    correct_answer = trivia_questions.get(question)
    if correct_answer.lower() == answer.lower():
        return "Correct!"
    else:
        return f"Incorrect. The correct answer is {correct_answer}."

with gr.Blocks() as demo:
    question = gr.Dropdown(choices=list(trivia_questions.keys()), label="Select a trivia que∑∑∑stion")
    answer = gr.Textbox(label="Your Answer")
    result = gr.Textbox(label="Result", interactive=False)
    
    question.change(fn=ask_question, inputs=question, outputs=result)
    answer.submit(fn=check_answer, inputs=[question, answer], outputs=result)

demo.launch()
