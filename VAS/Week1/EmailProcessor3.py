# RAG to find the Best Matching Policy
# - This code finds the most relevant medical policy based on a user's question by comparing the meaning of their query with stored policies using AI-powered text similarity.
# - It processes example questions, retrieves the best-matching policy, and displays its content along with a similarity score to show how well the policy matches the question.
from ApiKey import API_KEY, HGToken
# import libraries
import requests
from PIL import Image

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

try:
    import gradio as gr
    from together import Together
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    # Note: The pip install commands below won't work in a regular Python script
    # You'll need to install these packages before running this script
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "-q", "together"])
    subprocess.check_call(["pip", "install", "-q", "gradio"])
    subprocess.check_call(["pip", "install", "-q", "sentence-transformers"])
    import gradio as gr
    from together import Together
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

# Get Client
client = Together(api_key=API_KEY)

def prompt_llm(prompt, show_cost=False):
    # This function allows us to prompt an LLM via the Together API

    # model
    model = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

    # Calculate the number of tokens
    tokens = len(prompt.split())

    # Calculate and print estimated cost for each model
    if show_cost:
        print(f"\nNumber of tokens: {tokens}")
        cost = (0.1 / 1_000_000) * tokens
        print(f"Estimated cost for {model}: ${cost:.10f}\n")

    # Make the API call
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def gen_image(prompt, width=256, height=256):
    # This function allows us to generate images from a prompt
    response = client.images.generate(
        prompt=prompt,
        model="stabilityai/stable-diffusion-xl-base-1.0",  # Using a supported model
        steps=30,
        n=1,
    )
    image_url = response.data[0].url
    image_filename = "image.png"

    # Download the image using requests instead of wget
    response = requests.get(image_url)
    with open(image_filename, "wb") as f:
        f.write(response.content)
    img = Image.open(image_filename)
    img = img.resize((height, width))

    return img

class EmailResponseRetriever:
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=HGToken)
        # Sample email responses - in production, this would come from a database
        self.examples = {
            "late_delivery": """
                ORIGINAL EMAIL:
                I haven't received my order yet and it's been 2 weeks. This is unacceptable.

                MY RESPONSE:
                I sincerely apologize for the delay in your order. I understand your frustration.
                I've checked with our shipping department and your package is currently in transit.
                I'll expedite this and send you the updated tracking information within the next hour.
                Please let me know if you need anything else.
            """,
            "refund_request": """
                ORIGINAL EMAIL:
                The product I received is damaged. I want my money back immediately.

                MY RESPONSE:
                I'm very sorry to hear about the damaged product. I completely understand your concern.
                I've initiated an immediate refund which will be processed within 2-3 business days.
                Would you like a return shipping label to send the damaged item back to us?
            """,
            "product_inquiry": """
                ORIGINAL EMAIL:
                Does this come in different sizes? And what colors are available?

                MY RESPONSE:
                Thank you for your interest! Yes, this product comes in S, M, L, and XL.
                We currently have it available in navy blue, forest green, and charcoal gray.
                I can provide detailed measurements for any specific size you're interested in.
            """,
            "technical_support": """
                ORIGINAL EMAIL:
                The software keeps crashing when I try to export my project.

                MY RESPONSE:
                I understand how frustrating technical issues can be. Let's resolve this together.
                First, please try clearing your cache and restarting the application.
                If that doesn't work, could you send me your error log? You can find it at Settings > Help > Export Log.
            """,
        }
        # Pre-compute embeddings for examples
        self.example_embeddings = {
            k: self.encoder.encode(v) for k, v in self.examples.items()
        }

    def get_relevant_example(self, query, top_k=2):
        query_embedding = self.encoder.encode(query)
        similarities = {
            k: cosine_similarity([query_embedding], [emb])[0][0]
            for k, emb in self.example_embeddings.items()
        }
        sorted_examples = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        relevant_examples = []
        for example_name, score in sorted_examples[:top_k]:
            if score > 0.3:  # Similarity threshold
                relevant_examples.append(self.examples[example_name])

        return (
            "\n\n".join(relevant_examples)
            if relevant_examples
            else "No relevant example found."
        )

class PolicyRetriever:
    def __init__(self):
        self.encoder = SentenceTransformer(
            "all-MiniLM-L6-v2", use_auth_token=HGToken
        )
        # Sample medical policies - in production, this would come from a database
        self.policies = {
            "privacy": """
                Patient Privacy Policy:
                - All patient information is confidential and protected under HIPAA
                - Access to medical records requires patient consent
                - Data sharing with third parties strictly regulated
            """,
            "appointments": """
                Appointment Policy:
                - 24-hour notice required for cancellations
                - Telehealth options available for eligible consultations
                - Emergency cases prioritized based on severity
            """,
            "insurance": """
                Insurance Policy:
                - We accept major insurance providers
                - Pre-authorization required for specific procedures
                - Co-pay due at time of service
            """,
            "medication": """
                Medication Policy:
                - Prescription refills require 48-hour notice
                - Controlled substances have strict monitoring protocols
                - Generic alternatives offered when available
            """,
        }
        # Pre-compute embeddings for policies
        self.policy_embeddings = {
            k: self.encoder.encode(v) for k, v in self.policies.items()
        }

    def get_relevant_policy(self, query, top_k=2):
        query_embedding = self.encoder.encode(query)
        similarities = {
            k: cosine_similarity([query_embedding], [emb])[0][0]
            for k, emb in self.policy_embeddings.items()
        }
        sorted_policies = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        relevant_policies = []
        for policy_name, score in sorted_policies[:top_k]:
            if score > 0.3:  # Similarity threshold
                relevant_policies.append(self.policies[policy_name])

        return (
            "\n\n".join(relevant_policies)
            if relevant_policies
            else "No relevant policy found."
        )


# Test functionality for PolicyRetriever
if __name__ == "__main__":
    # Example usage
    retriever = PolicyRetriever()

    # Test queries with expected policy matches
    test_queries = [
        "I need to cancel my appointment tomorrow morning",
        "Do you share my medical information with other doctors?",
        "When do I need to pay my insurance copay?",
        "How can I get my prescription refilled?",
    ]

    for query in test_queries:
        # Get similarity scores for all policies
        query_embedding = retriever.encoder.encode(query)
        similarities = {
            k: cosine_similarity([query_embedding], [emb])[0][0]
            for k, emb in retriever.policy_embeddings.items()
        }

        # Get the most relevant policy and its score
        best_match = max(similarities.items(), key=lambda x: x[1])
        policy_name, score = best_match

        print(f"\nQuery: {query}")
        print(f"Best matching policy: {policy_name}")
        print(f"Similarity score: {score:.3f}")
        print(f"Policy content:\n{retriever.policies[policy_name]}")
        print("-" * 80)

class EmailAgent:
    def __init__(self, role, client, policy_retriever=None, example_retriever=None):
        self.role = role
        self.client = client
        self.policy_retriever = policy_retriever
        self.example_retriever = example_retriever

    def process(self, content):
        # Get relevant policies if policy_retriever is available
        relevant_policies = ""
        if self.policy_retriever and self.role in ["analyzer", "drafter"]:
            relevant_policies = self.policy_retriever.get_relevant_policy(content)
        
        # Get relevant examples if example_retriever is available and role is drafter
        relevant_examples = ""
        if self.example_retriever and self.role == "drafter":
            relevant_examples = self.example_retriever.get_relevant_example(content)
        
        prompts = {
            # Analyzer prompt - extracts key information from email
            "analyzer": """SYSTEM: You are an expert email analyzer with years of experience in professional communication. Your role is to break down emails into their key components and provide clear, actionable insights.

            As an email analyzer, examine this email content and extract:
            1. Main topics and key points
            2. Urgency level
            3. Required actions
            4. Tone of the message

            INSTRUCTIONS:
            • Focus on extracting factual information without interpretation
            • Identify any deadlines or time-sensitive elements
            • Categorize the email priority (high/medium/low)
            • Show output only - no explanations or additional commentary

            Email: {content}

            Relevant policies:
            {policies}

            Provide a structured analysis.""",
            # Drafter prompt - creates email response based on analysis
            "drafter": """SYSTEM: You are a professional email response specialist with extensive experience in business communication. Your role is to craft clear, effective, and appropriate email responses based on provided analysis.

            As an email response drafter, using this analysis: {content}
            
            Relevant policies:
            {policies}
            
            Similar email examples for reference:
            {examples}

            Create a professional email response that:
            1. Addresses all key points
            2. Matches the appropriate tone
            3. Includes clear next steps
            4. References relevant policies when applicable

            INSTRUCTIONS:
            • Maintain consistent professional tone throughout response
            • Include specific details from the analysis
            • End with clear actionable next steps
            • Show output only - provide just the email response

            Write the complete response.""",
            # Reviewer prompt - evaluates the draft response
            "reviewer": """SYSTEM: You are a senior email quality assurance specialist with a keen eye for detail and professional standards. Your role is to ensure all email responses meet the highest standards of business communication.

            As an email quality reviewer, evaluate this draft response: {content}
            Check for:
            1. Professionalism and appropriateness
            2. Completeness (all points addressed)
            3. Clarity and tone
            4. Potential improvements

            INSTRUCTIONS:
            • Verify all original questions/requests are addressed
            • Check for appropriate formality and politeness
            • Ensure response is concise and well-structured
            • Show output only - return APPROVED or NEEDS_REVISION with brief feedback

            Return either APPROVED or NEEDS_REVISION with specific feedback.""",
        }

        return prompt_llm(prompts[self.role].format(
            content=content, 
            policies=relevant_policies,
            examples=relevant_examples
        ))


def process_email(email_content):
    # Create policy retriever and example retriever
    policy_retriever = PolicyRetriever()
    example_retriever = EmailResponseRetriever()
    
    # Get relevant policies and examples
    relevant_policies = policy_retriever.get_relevant_policy(email_content)
    relevant_examples = example_retriever.get_relevant_example(email_content)
    
    # Create agents
    analyzer = EmailAgent("analyzer", client, policy_retriever, example_retriever)
    drafter = EmailAgent("drafter", client, policy_retriever, example_retriever)

    # Process email
    analysis = analyzer.process(email_content)
    draft = drafter.process(analysis)

    return analysis, draft, relevant_policies, relevant_examples


# Example emails
example_emails = [
    """Dear Team,
I hope this email finds you well. We need to reschedule tomorrow's project meeting due to a conflict. Could we move it to Friday at 2 PM instead?
Best regards,
John""",
    """Subject: Urgent: Server Downtime
The production server is currently experiencing issues. We need immediate assistance to resolve this. Please respond ASAP.
-Sarah from DevOps""",
    """Hi Marketing Team,
Just wanted to follow up on the Q4 report. When can we expect the first draft for review?
Thanks,
Mike""",
]


class EmailDemo:
    def __init__(self):
        self.current_index = 0

    def get_current_email(self):
        return example_emails[self.current_index]

    def next_email(self, _):
        self.current_index = (self.current_index + 1) % len(example_emails)
        return example_emails[self.current_index]


def main():
    print("LLM Ready!")
    
    demo_state = EmailDemo()

    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# 📧 Email Processing System")
        gr.Markdown("View example emails and get AI-powered analysis and responses.")

        with gr.Row():
            email_input = gr.Textbox(
                value=demo_state.get_current_email(), lines=5, label="📝 Email Content"
            )
            next_button = gr.Button("⏭️ Next Example Email")

        process_button = gr.Button("🔄 Process Email")

        with gr.Row():
            analysis_output = gr.Textbox(
                lines=8, label="📊 Analysis", show_copy_button=True
            )
            draft_output = gr.Textbox(
                lines=8, label="✉️ Draft Response", show_copy_button=True
            )
            
        with gr.Row():
            policy_output = gr.Textbox(
                lines=6, label="📋 Relevant Policies", show_copy_button=True
            )
            example_output = gr.Textbox(
                lines=6, label="📝 Similar Email Examples", show_copy_button=True
            )

        # Set up event handlers
        next_button.click(
            demo_state.next_email, inputs=[email_input], outputs=[email_input]
        )

        process_button.click(
            process_email, 
            inputs=[email_input], 
            outputs=[analysis_output, draft_output, policy_output, example_output]
        )

    demo.launch()


if __name__ == "__main__":
    main()
