# pip install gradio==5.10
import gradio as gr

def greet(name):
    return f"Hello, {name}!"

iface = gr.Interface(fn=greet, inputs="text", outputs="text", title="Hello World App", description="Enter your name to get a greeting.")

iface.launch()