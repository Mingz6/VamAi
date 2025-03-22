import gradio as gr
import json
from together import Together
import os


# TODO 1: Replace with your Together API key (https://www.together.ai/)
# Get the current notebook directory
notebook_dir = os.path.dirname(os.path.abspath("__file__"))

# Navigate to the config.json in the parent directory
config_path = os.path.join(os.path.dirname(notebook_dir), "config.json")

# Load the configuration
with open(config_path, 'r') as f:
    config_text = f.read()
    # Remove any extra quotes if the JSON is stored as a string
    if config_text.startswith('"') and config_text.endswith('"'):
        config_text = config_text[1:-1].replace('\\"', '"')
    config = json.loads(config_text)

# Now you can access your configuration
together_ai_token = config.get("together_ai_token")
model_name = config.get("model")

print(f"Loaded token: {together_ai_token[:5]}... for model: {model_name}")

client = Together(api_key=together_ai_token)  # Fixed to use together_ai_token instead of your_api_key

# Sample dataset with input texts and corresponding quiz questions
dataset = {
    "input": [
        {
            "content": "Quantum Computing Basics",
            "category": "text",
            "source": "educational_material",
        },
        {
            "content": "Machine Learning Fundamentals",
            "category": "text",
            "source": "educational_material",
        },
        {
            "content": "Climate Change Science",
            "category": "wikipedia",
            "source": "https://en.wikipedia.org/wiki/Climate_change",
        },
        {
            "content": "human_biology_101.pdf",
            "category": "pdf",
            "source": "course_materials",
        },
        {
            "content": "World War II Overview",
            "category": "wikipedia",
            "source": "https://en.wikipedia.org/wiki/World_War_II",
        },
    ],
    "expected_output": [
        [
            {
                "question": "What is a qubit?",
                "options": [
                    "A classical computer bit",
                    "A quantum bit that can be in superposition",
                    "A measurement unit",
                    "A quantum programming language",
                ],
                "correct": 1,
                "justification": "A qubit is a quantum bit that can exist in superposition, meaning it can be in multiple states simultaneously, unlike classical bits which can only be 0 or 1.",
            },
            # ... other quantum computing questions
        ],
        [
            {
                "question": "What is a key component of machine learning?",
                "options": [
                    "Manual programming only",
                    "Learning from data and patterns",
                    "Hardware manufacturing",
                    "Network cables",
                ],
                "correct": 1,
                "justification": "Machine learning fundamentally relies on algorithms that can learn from and make predictions based on data patterns, rather than being explicitly programmed with fixed rules.",
            },
            # ... other machine learning questions
        ],
        [
            {
                "question": "What is the greenhouse effect?",
                "options": [
                    "Plant growth in greenhouses",
                    "Atmospheric heat trapping",
                    "Solar panel technology",
                    "Wind patterns",
                ],
                "correct": 1,
                "justification": "The greenhouse effect is a natural process where certain gases in Earth's atmosphere trap heat from the sun, similar to how a greenhouse works, keeping the planet warm enough to sustain life.",
            },
            # ... other climate change questions
        ],
        [
            {
                "question": "What is the basic unit of life?",
                "options": [
                    "Atom",
                    "Cell",
                    "Molecule",
                    "Tissue",
                ],
                "correct": 1,
                "justification": "The cell is considered the basic unit of life because it is the smallest structure capable of performing all the functions necessary for life, including metabolism, growth, and reproduction.",
            },
            # ... other biology questions
        ],
        [
            {
                "question": "When did World War II begin?",
                "options": [
                    "1935",
                    "1939",
                    "1941",
                    "1945",
                ],
                "correct": 1,
                "justification": "World War II officially began on September 1, 1939, when Nazi Germany invaded Poland, leading Britain and France to declare war on Germany two days later.",
            },
            # ... other WWII questions
        ],
    ],
}


def prompt_llm(prompt):
    # TODO 2: You can experiment with different models here (see here https://api.together.ai/models)
    model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
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

    # Format the input display with markdown and symbols, with a dark-theme compatible box
    input_text = """
<div style="border: 2px solid var(--border-color-primary, #555); border-radius: 8px; padding: 15px; background-color: var(--background-fill-secondary, rgba(0, 100, 100, 0.1)); color: var(--body-text-color, inherit);">

### 📖 Topic: {content}

🏷️ **Category**: {category}

🔗 **Source**: {source}

</div>
""".format(
        **input_data
    )

    # Format the output in markdown with dark-theme compatible box
    output_text = """
<div style="border: 2px solid var(--border-color-primary, #555); border-radius: 8px; padding: 15px; background-color: var(--background-fill-secondary, rgba(0, 100, 100, 0.1)); color: var(--body-text-color, inherit);">

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


# Create Gradio interface with custom CSS
custom_css = """
:root {
    --body-text-color: var(--body-text-color, #f0f0f0);
    --border-color-primary: var(--border-color-primary, #555);
    --background-fill-primary: var(--background-fill-primary, #333);
    --background-fill-secondary: var(--background-fill-secondary, rgba(0, 100, 100, 0.1));
}

.custom-textarea textarea {
    color: var(--body-text-color) !important;
    background-color: var(--background-fill-primary) !important;
}

/* Ensure text is visible in both light and dark themes */
.markdown-text {
    color: var(--body-text-color) !important;
}
"""

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown(
        """
    # 📚 Omniscient Prompt XRay
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
                    elem_classes="markdown-text",
                )
        with gr.Column():
            run_prompt_btn = gr.Button("Run Prompt")

            prompt_llm_text = gr.TextArea(
                label="Prompt Template",
                # TODO 3: Modify the prompt template below for your specific AI task
                value="""You are an expert quiz generator, skilled at creating engaging and educational multiple-choice questions.

# Task Description
Generate 3 multiple-choice questions about the given topic with 4
options each. Include explanations for the correct answers.

# Output Format
- Q1: Question 1
- A1: Option 1
- A2: Option 2
- A3: Option 3
- A4: Option 4
- Correct Answer: 1
- Explanation: Explanation for the correct answer

# Input Content
{content}

# Requirements
1. Each question must have exactly 4 options
2. Include clear explanations for correct answers
3. Ensure questions test understanding, not just memorization
4. Use clear, concise language
5. Make sure all options are plausible""",
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
                elem_classes="markdown-text",
            )

        # Output section with LLM output and expected output
        with gr.Column(scale=1):
            gr.Markdown("## LLM Output")
            llm_output = gr.TextArea(
                label="LLM Output",
                value="LLM output will appear here...",
                lines=10,
                elem_classes="custom-textarea",
            )
            with gr.Row():
                approve_btn = gr.Button("Approve and Save")
                save_status = gr.Markdown("")

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
        prompt = prompt_template.format(content=input_data["content"])
        response = prompt_llm(prompt)
        return response

    # Connect the run prompt button
    run_prompt_btn.click(
        generate_llm_response,
        inputs=[index_slider, prompt_llm_text],
        outputs=[llm_output],
    )

if __name__ == "__main__":
    demo.launch()
