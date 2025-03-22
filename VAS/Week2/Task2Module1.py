# import libraries
import importlib
import subprocess
import sys

# First check for required packages and install if missing
required_packages = ['requests', 'Pillow', 'gradio', 'together']
missing_packages = []

for package in required_packages:
    try:
        importlib.import_module(package.lower())
    except ImportError:
        missing_packages.append(package)

# Install missing packages
if missing_packages:
    print(f"Installing missing packages: {', '.join(missing_packages)}")
    for package in missing_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

# Now import all required packages
import requests
from PIL import Image
import gradio as gr
from together import Together
import json

# Repository setup and configuration examples
dataset = {
    "input": [
        {
            "content": "Python Web Application Repository",
            "category": "repository",
            "source": "github.com/user/python-web-app",
        },
        {
            "content": "Data Processing Pipeline",
            "category": "repository",
            "source": "github.com/user/data-pipeline",
        },
        {
            "content": "Machine Learning Model Deployment",
            "category": "repository",
            "source": "gitlab.com/organization/ml-deployment",
        },
        {
            "content": "DevOps Infrastructure as Code",
            "category": "repository",
            "source": "github.com/team/infrastructure",
        },
        {
            "content": "Mobile App Backend API",
            "category": "repository",
            "source": "bitbucket.org/company/mobile-backend",
        },
    ],
    "expected_output": [
        [
            {
                "question": "What is the recommended way to set up the Python web application repository?",
                "options": [
                    "Clone the repository and run npm install",
                    "Clone the repository, create a virtual environment, and run pip install -r requirements.txt",
                    "Download the ZIP file and extract it to your web server",
                    "Fork the repository and modify the configuration files only",
                ],
                "correct": 1,
                "justification": "Python web applications typically require a virtual environment to isolate dependencies, and the requirements.txt file contains all the necessary packages that should be installed using pip.",
            },
            {
                "question": "Which configuration file would you modify to change the database connection settings?",
                "options": [
                    "package.json",
                    "app.py",
                    "config.yml or .env file",
                    "README.md",
                ],
                "correct": 2,
                "justification": "Database connection settings are typically stored in configuration files like config.yml, config.json, or environment files (.env) to separate configuration from code and allow for different environments.",
            },
            {
                "question": "What command would likely start the development server for this web application?",
                "options": [
                    "npm start",
                    "python app.py or flask run",
                    "java -jar app.jar",
                    "docker-compose up",
                ],
                "correct": 1,
                "justification": "Python web applications built with frameworks like Flask or Django typically use commands like 'python app.py', 'flask run', or 'python manage.py runserver' to start the development server.",
            },
        ],
        [
            {
                "question": "What type of dependencies would a data processing pipeline likely require?",
                "options": [
                    "Front-end libraries like React and Bootstrap",
                    "Database systems like MySQL and PostgreSQL only",
                    "Data manipulation libraries like Pandas, NumPy, and possibly Spark",
                    "Mobile development frameworks",
                ],
                "correct": 2,
                "justification": "Data processing pipelines typically rely on libraries specialized for data manipulation and analysis, such as Pandas for data frames, NumPy for numerical operations, and potentially Spark for large-scale data processing.",
            },
            {
                "question": "How would you configure the data pipeline to use a different data source?",
                "options": [
                    "Modify HTML templates",
                    "Update the source parameters in configuration files",
                    "Reinstall the entire application",
                    "Change the programming language",
                ],
                "correct": 1,
                "justification": "Data pipelines are typically configurable through configuration files where source parameters (like file paths, database connections, API endpoints) can be modified without changing the code itself.",
            },
            {
                "question": "What is a common way to schedule recurring data pipeline runs?",
                "options": [
                    "Manual execution only",
                    "Using cron jobs, Airflow, or similar scheduling tools",
                    "Through a mobile application",
                    "Only when the system boots up",
                ],
                "correct": 1,
                "justification": "Data pipelines often need to run on schedules to process data regularly. Tools like cron (for simple scheduling), Apache Airflow, or cloud schedulers are commonly used to automate and schedule pipeline execution.",
            },
        ],
        [
            {
                "question": "What environment components are typically needed for ML model deployment?",
                "options": [
                    "Only a web server",
                    "A GPU-enabled server with appropriate ML libraries and serving infrastructure",
                    "Just a database server",
                    "A content delivery network (CDN)",
                ],
                "correct": 1,
                "justification": "ML model deployment often requires specialized hardware like GPUs for inference, ML libraries (TensorFlow, PyTorch, etc.), and serving infrastructure (TensorFlow Serving, Flask API, etc.) to make predictions available to applications.",
            },
            {
                "question": "How would you update a deployed machine learning model?",
                "options": [
                    "Manually edit the model files",
                    "Replace the model artifact and update version references in configuration",
                    "Reinstall the operating system",
                    "Always train a new model from scratch",
                ],
                "correct": 1,
                "justification": "Updating a deployed ML model typically involves replacing the model artifact (file) with a new version and updating configuration references to point to the new model, often with version control to enable rollbacks if needed.",
            },
            {
                "question": "What configuration might be needed for scaling ML model inference?",
                "options": [
                    "Database indexes",
                    "Load balancer settings, instance count, and resource allocation",
                    "Email server settings",
                    "Social media integration",
                ],
                "correct": 1,
                "justification": "Scaling ML inference requires infrastructure configurations like load balancer settings to distribute requests, adjusting the number of serving instances, and allocating appropriate resources (memory, CPU/GPU) for handling prediction traffic.",
            },
        ],
        [
            {
                "question": "What tool is commonly used for Infrastructure as Code in cloud environments?",
                "options": [
                    "Microsoft Word",
                    "Terraform, AWS CloudFormation, or Pulumi",
                    "Adobe Photoshop",
                    "Social media platforms",
                ],
                "correct": 1,
                "justification": "Infrastructure as Code (IaC) relies on specialized tools like Terraform, AWS CloudFormation, Azure Resource Manager templates, or Pulumi that allow defining infrastructure in code files that can be version-controlled and automatically deployed.",
            },
            {
                "question": "How would you modify network security settings in an Infrastructure as Code repository?",
                "options": [
                    "Edit the infrastructure code files to update security group or firewall rules",
                    "Send an email to the network team",
                    "Physically access the server room",
                    "Post a request on social media",
                ],
                "correct": 0,
                "justification": "In an IaC approach, network security settings like security groups, firewall rules, and network ACLs are defined in code files. To modify these settings, you would edit the appropriate code files (e.g., Terraform .tf files) and deploy the changes following the established workflow.",
            },
            {
                "question": "What is a best practice for managing secrets in Infrastructure as Code?",
                "options": [
                    "Hardcode them in the infrastructure files",
                    "Share them via email",
                    "Use a secret management service like HashiCorp Vault or cloud-native secret managers",
                    "Write them in a notebook",
                ],
                "correct": 2,
                "justification": "Best practices for secret management in IaC include using specialized secret management services like HashiCorp Vault, AWS Secrets Manager, or Azure Key Vault, rather than including sensitive information directly in the infrastructure code.",
            },
        ],
        [
            {
                "question": "What database system might a mobile app backend API typically use?",
                "options": [
                    "Spreadsheet files",
                    "Relational (PostgreSQL, MySQL) or NoSQL (MongoDB, DynamoDB) databases",
                    "Printed paper records",
                    "Local storage only",
                ],
                "correct": 1,
                "justification": "Mobile app backends typically use databases to store user data, application state, and content. These can be relational databases like PostgreSQL or MySQL for structured data with relationships, or NoSQL databases like MongoDB or DynamoDB for more flexible data models.",
            },
            {
                "question": "How would you configure authentication for a mobile app backend API?",
                "options": [
                    "No authentication is needed",
                    "Configure OAuth, JWT, API keys, or other authentication mechanisms in the API configuration",
                    "Only allow access from specific IP addresses",
                    "Change the API URL regularly",
                ],
                "correct": 1,
                "justification": "Mobile app backend APIs require secure authentication. This is typically configured through mechanisms like OAuth for third-party authentication, JWT (JSON Web Tokens) for stateless authentication, or API keys for service-to-service communication.",
            },
            {
                "question": "What is a common way to handle API versioning in a mobile backend repository?",
                "options": [
                    "Create new folders/URLs with version numbers (e.g., /api/v1/, /api/v2/)",
                    "Change the server hostname for each version",
                    "Require users to uninstall and reinstall the app",
                    "Send paper notices to all users",
                ],
                "correct": 0,
                "justification": "API versioning is commonly handled by incorporating version numbers in the URL path (e.g., /api/v1/resources), using request headers, or query parameters. This allows maintaining backward compatibility while evolving the API.",
            },
        ],
    ],
}


def show_data_point(index):
    # Simplified input display template with theme-compatible styling
    input_template = """
    <div style="border: 2px solid var(--border-color-primary); border-radius: 8px; padding: 15px; background-color: var(--background-fill-primary); color: var(--body-text-color);">

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
        '<div style="border: 2px solid var(--border-color-primary); border-radius: 8px; padding: 15px; background-color: var(--background-fill-primary); color: var(--body-text-color);">',
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
with gr.Blocks(theme=gr.themes.Default()) as demo:
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
