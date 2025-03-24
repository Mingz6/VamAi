import gradio as gr
import json
import os
from together import Together

# Get the current directory
current_dir = os.path.dirname(os.path.abspath("__file__"))

# Navigate to the config.json in the parent directory
config_path = os.path.join(os.path.dirname(current_dir), "config.json")

# Load the configuration
with open(config_path, 'r') as f:
    config_text = f.read()
    # Remove any extra quotes if the JSON is stored as a string
    if config_text.startswith('"') and config_text.endswith('"'):
        config_text = config_text[1:-1].replace('\\"', '"')
    config = json.loads(config_text)

# TODO: 1. Customize your API key or use environment variables for security
# Now you can access your configuration
together_ai_token = config.get("together_ai_token")
model_name = config.get("model")

print(f"Loaded token: {together_ai_token[:5]}... for model: {model_name}")

# Initialize the Together client with your API token
client = Together(api_key=together_ai_token)

# Load prompt template from file - now in the same directory
prompt_template_path = os.path.join(current_dir, "prompttemplate.json")
with open(prompt_template_path, 'r') as f:
    prompt_template_data = json.load(f)
    prompt_template = prompt_template_data.get("prompt_template", "")

def load_dataset():
    # Updated path to look for DatasetGitHub.json in the same directory
    json_path = os.path.join(current_dir, "DatasetGitHub.json")
    try:
        with open(json_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {json_path}")
        return {"input": [], "expected_output": []}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}")
        return {"input": [], "expected_output": []}

# Load the dataset
dataset = load_dataset()

def prompt_llm(prompt):
    model = model_name
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def show_data_point(index):
    input_data = dataset["input"][index]
    quiz_data = dataset["expected_output"][index]

    # Update header with current index + 1
    header_text = f"## Example {index + 1}"

    # Format the input display with markdown and symbols, now with a box
    input_text = """
<div style="border: 2px solid rgba(150, 150, 150, 0.4); border-radius: 8px; padding: 15px; background-color: rgba(100, 220, 200, 0.15); color: var(--body-text-color);">

### 📖 Topic: {content}

🏷️ **Category**: {category}

🔗 **Source**: {source}

</div>
""".format(
        **input_data
    )

    # Format the output in markdown with styled box
    output_text = """
<div style="border: 2px solid #ddd; border-radius: 8px; padding: 15px; background-color: #ddfff5;">

### 📝 Quiz Questions

"""
    for i, quiz in enumerate(quiz_data, 1):
        output_text += f"#### Q{i}: {quiz['question']}\n\n"
        for j, option in enumerate(quiz["options"]):
            output_text += f"{chr(97 + j)}) {option}\n\n"
        output_text += f"\n✅ **Correct Answer**: {chr(96 + quiz['correct'] + 1)}\n"
        if "justification" in quiz:
            output_text += f"\n💡 **Explanation**: {quiz['justification']}\n"
        output_text += "\n---\n\n"

    output_text += "</div>"

    return header_text, input_text, output_text


def evaluate_output(llm_output, index):
    expected = dataset["expected_output"][index]

    # TODO: 5. Customize the evaluation prompt to match your specific evaluation criteria
    evaluation_prompt = f"""As an expert evaluator, compare the generated quiz questions with the expected output and rate them on three criteria (scale 1-10):

Generated Output:
{llm_output}

Expected Output:
{json.dumps(expected, indent=2)}

Rate on these criteria:
1. Content Accuracy: How well do the questions test relevant knowledge?
2. Question Quality: Are questions clear, well-structured, and unambiguous?
3. Educational Value: How effective are the explanations and learning value?

Provide ratings and brief justifications in this format:
Content Accuracy: [X/10]
[Brief justification]

Question Quality: [X/10]
[Brief justification]

Educational Value: [X/10]
[Brief justification]

Overall Average: [X/10]
"""

    response = prompt_llm(evaluation_prompt)
    return response


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
    # 📚 Omniscient Prompt XRay with LLM Evaluation
    Explore quiz questions generated for different topics.
    """
    )
    header = gr.Markdown("## Example 1")  # We'll update this dynamically
    with gr.Row():
        index_slider = gr.Slider(
            minimum=0,
            maximum=len(dataset["input"]) - 1,
            step=1,
            value=0,
            label="Data Point Index",
            container=False,
        )

    with gr.Row():
        # Input section with prompt
        with gr.Column(scale=1):
            gr.Markdown("## Input")
            with gr.Row():
                input_text = gr.Markdown(
                    label="Input",
                    value=show_data_point(0)[1],
                )
        with gr.Column():
            run_prompt_btn = gr.Button("Run Prompt")

            # TODO: 2. Modify this prompt template to match your specific use case
            # Use the loaded prompt template instead of hardcoded text
            prompt_llm_text = gr.TextArea(
                label="Prompt Template",
                value=prompt_template,
                lines=10,
                interactive=True,
                container=True,
                show_label=True,
                elem_classes="custom-textarea",
            )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Expected Output")
            output_text = gr.Markdown(
                label="Expected Output",
                value=show_data_point(0)[2],
            )

        # Output section with LLM output and expected output
        with gr.Column(scale=1):
            gr.Markdown("## LLM Output")
            llm_output = gr.TextArea(
                label="LLM Output",
                value="LLM output will appear here...",
                lines=10,
            )

            with gr.Row():
                approve_btn = gr.Button("Approve and Save")
                evaluate_btn = gr.Button("Evaluate Output")
                save_status = gr.Markdown("")
            evaluation_output = gr.TextArea(
                label="Evaluation",
                value="Evaluation will appear here...",
                lines=10,
                interactive=False,
            )

    # Add function to save approved output
    def save_approved_output(index, prompt_template, llm_output):
        input_data = dataset["input"][index]
        output_data = {
            "input": input_data,
            "prompt": prompt_template,
            "llm_output": llm_output,
        }

        # Save to JSON file with timestamp
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"approved_outputs_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(output_data, f, indent=4)

        return f"✅ Saved! File: `{filename}`"

    # Connect the approve button
    approve_btn.click(
        save_approved_output,
        inputs=[index_slider, prompt_llm_text, llm_output],
        outputs=[save_status],
    )

    index_slider.change(
        show_data_point,
        inputs=[index_slider],
        outputs=[header, input_text, output_text],
    )

    def generate_llm_response(index, prompt_template):
        input_data = dataset["input"][index]
        prompt = prompt_template.format(**input_data)
        response = prompt_llm(prompt)
        return response

    # Connect the run prompt button to only generate response
    run_prompt_btn.click(
        generate_llm_response,
        inputs=[index_slider, prompt_llm_text],
        outputs=[llm_output],
    )

    # Keep the separate evaluate button functionality
    evaluate_btn.click(
        evaluate_output,
        inputs=[llm_output, index_slider],
        outputs=[evaluation_output],
    )

if __name__ == "__main__":
    demo.launch(debug=True)
