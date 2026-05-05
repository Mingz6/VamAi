import gradio as gr
import json
import os

# Load dataset from JSON file
def load_dataset():
    json_path = os.path.join(os.path.dirname(__file__), 'Dataset.json')
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

def show_data_point(index):
    # Simplified input display template
    input_template = """
    <div style="border: 2px solid rgba(150, 150, 150, 0.4); border-radius: 8px; padding: 15px; background-color: rgba(100, 220, 200, 0.15); color: var(--body-text-color);">

    ### 📖 Topic: {content}


    🏷️ **Category**: {category}

    🔗 **Source**: {source}
    </div>
    """

    input_data = dataset["input"][index]
    quiz_data = dataset["expected_output"][index]

    header_text = f"## Example {index + 1}"
    input_text = input_template.format(**input_data)

    # Format quiz questions with cleaner structure
    output_parts = [
        '<div style="border: 2px solid rgba(150, 150, 150, 0.4); border-radius: 8px; padding: 15px; background-color: rgba(100, 220, 200, 0.15); color: var(--body-text-color);">',
        "\n\n### 📝 Quiz Questions\n\n",
    ]

    for i, quiz in enumerate(quiz_data, 1):
        output_parts.extend(
            [
                f"#### Q{i}: {quiz['question']}\n\n",
                *[
                    f"{chr(97 + j)}) {option}\n\n\n"
                    for j, option in enumerate(quiz["options"])
                ],
                f"\n✅ **Correct Answer**: {chr(96 + quiz['correct'] + 1)}\n",
                f"\n💡 **Explanation**: {quiz.get('justification', 'No explanation provided.')}\n",
                "\n---\n\n",
            ]
        )

    output_parts.append("</div>")
    output_text = "".join(output_parts)

    return header_text, input_text, output_text


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
    # 📚 Omniscient Dataset Explorer
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

        with gr.Column(scale=1):
            gr.Markdown("## Input")
            input_text = gr.Markdown(
                label="Input",
                value=show_data_point(0)[1],  # Initialize with first example
            )
        with gr.Column(scale=2):
            gr.Markdown("## Expected Output")
            output_text = gr.Markdown(  # Changed from Textbox to Markdown
                label="Expected Output",
                value=show_data_point(0)[2],  # Initialize with first example
            )

    index_slider.change(
        show_data_point,
        inputs=[index_slider],
        outputs=[header, input_text, output_text],  # Add header to outputs
    )

if __name__ == "__main__":
    demo.launch()
