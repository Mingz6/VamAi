# AI Email Responder with Past Examples
# - This program retrieves and adapts past email responses to draft new replies while ensuring compliance with healthcare policies.
# - It analyzes the email, finds similar past responses, drafts a reply in a friendly tone, and reviews it for accuracy and compliance.
from ApiKey import API_KEY, HGToken
# import libraries
import requests
from PIL import Image
import numpy as np  # Added numpy import
import sys  # Added sys import

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
            "medical_records": """
                ORIGINAL EMAIL:
                Hey there, I was hoping to get my medical records. What do I need to do?

                MY RESPONSE:
                Hi! Happy to help you get those records. We just need a few quick things:
                1. Your signed OK (we'll send you the form)
                2. A quick form to fill out
                3. Your ID
                Just upload everything to our secure portal and we'll take care of the rest! Let me know if you need help.
            """,
            "insurance_verification": """
                ORIGINAL EMAIL:
                Quick question - do you guys take Aetna insurance?

                MY RESPONSE:
                Hey there! Yes, we work with Aetna and most other major insurance companies.
                Could you shoot me your:
                - Member ID
                - Group number
                I'll double-check everything and get back to you super quick (usually within a day).
                Sound good?
            """,
            "appointment_scheduling": """
                ORIGINAL EMAIL:
                Something came up and I need to move my appointment. Help!

                MY RESPONSE:
                Hey! No worries at all - life happens! 😊
                I've got a couple of spots open:
                - Tuesday @ 2pm
                - Wednesday morning at 10am
                Just let me know what works better for you and I'll get it switched right away!
            """,
            "medication_refill": """
                ORIGINAL EMAIL:
                Running low on my meds - need a refill asap!

                MY RESPONSE:
                Hey! Thanks for the heads up about your meds. I'm on it!
                Here's what's happening next:
                1. We'll review your refill request today
                2. Give your pharmacy a call
                3. They should have it ready in 1-2 days
                Need it sooner? Just let me know!
            """,
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
            """
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

    # Add the alias method for compatibility
    def get_relevant_response(self, query, top_k=2):
        return self.get_relevant_example(query, top_k)

class PolicyRetriever:
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=HGToken)
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

class EmailAgent:
    def __init__(self, role, client, policy_retriever=None, example_retriever=None):
        self.role = role
        self.client = client
        self.policy_retriever = policy_retriever or PolicyRetriever()
        self.example_retriever = example_retriever or EmailResponseRetriever()

        # Move prompts to be an instance variable
        self.prompts = {
            "analyzer": """SYSTEM: You are an expert email analyzer for a medical company.
            Your role is to break down emails into key components and provide clear, actionable insights.

            INSTRUCTIONS:
            • Extract main topics and key points from the email
            • Determine urgency level (Low, Medium, High)
            • List all required actions in bullet points
            • Analyze tone of the message (formal, informal, urgent, etc.)
            • Identify relevant company policies that apply
            • Highlight any compliance concerns
            • Limit response to 50 words maximum
            • Show response only without additional commentary

            CONTEXT (Company Policies):
            {policies}

            SIMILAR EMAILS:
            {examples}

            Email: {content}""",
            "drafter": """SYSTEM: You are a professional email response specialist for a medical company.
            Draft responses that align with our policies and maintain HIPAA compliance.

            INSTRUCTIONS:
            • Address all key points from the original email
            • Ensure response aligns with provided company policies
            • Verify HIPAA compliance in all content
            • Include clear next steps and action items
            • Maintain professional and empathetic tone
            • Add necessary disclaimers where applicable
            • Limit response to 50 words maximum
            • Show response only without additional commentary

            CONTEXT (Relevant Policies):
            {policies}

            SIMILAR RESPONSES:
            {examples}

            Based on this analysis: {content}""",
            "reviewer": """SYSTEM: You are a senior email quality assurance specialist for a medical company.
            Ensure responses meet healthcare communication standards and comply with policies.

            INSTRUCTIONS:
            • Verify compliance with all relevant policies
            • Check for HIPAA violations
            • Assess professional tone and clarity
            • Review completeness of response
            • Evaluate appropriate handling of sensitive information
            • Confirm all action items are clearly stated
            • Limit response to 50 words maximum
            • Show response only without additional commentary

            CONTEXT (Relevant Policies):
            {policies}

            Evaluate this draft response: {content}""",
            "sentiment": """SYSTEM: You are an expert in analyzing email sentiment and emotional context in
            healthcare communications.

            INSTRUCTIONS:
            • Analyze overall sentiment (positive, negative, neutral)
            • Identify emotional undertones
            • Detect urgency or stress indicators
            • Assess patient/sender satisfaction level
            • Flag any concerning language
            • Recommend tone adjustments if needed
            • Limit response to 50 words maximum
            • Show response only without additional commentary

            Email: {content}""",
            "policy_justifier": """SYSTEM: You are a policy expert. In 2 lines, explain why the following policies are relevant
            to this email content. Be specific and concise.

            Email content: {content}
            Selected policies: {policies}""",
            "example_justifier": """SYSTEM: You are an example matching expert. In 2 lines, explain why the following example responses are relevant
            to this email content. Be specific and concise.

            Email content: {content}
            Selected examples: {examples}""",
            "casual_drafter": """SYSTEM: You are a professional email response specialist for a medical company.
            Draft responses that align with our past successful responses while maintaining a friendly, conversational tone.

            INSTRUCTIONS:
            • Address all key points from the original email
            • Use a friendly, conversational tone like in our examples
            • Ensure HIPAA compliance in all content
            • Include clear next steps and action items
            • Maintain professional yet approachable tone
            • Add necessary disclaimers where applicable
            • Limit response to 50 words maximum
            • Show response only without additional commentary

            SIMILAR PAST RESPONSES:
            {examples}

            Based on this analysis: {content}""",
        }
    
    def process(self, content):
        # Get relevant policies for the email content
        relevant_policies = self.policy_retriever.get_relevant_policy(content)
        # Get relevant examples for the email content
        relevant_examples = ""
        if self.example_retriever:
            relevant_examples = self.example_retriever.get_relevant_example(content)
        
        # Add examples to the prompt if available
        if self.role in ["analyzer", "drafter", "casual_drafter", "example_justifier"]:
            return prompt_llm(
                self.prompts[self.role].format(
                    content=content, 
                    policies=relevant_policies if self.role != "casual_drafter" else "",
                    examples=relevant_examples
                )
            )
        else:
            return prompt_llm(
                self.prompts[self.role].format(
                    content=content, 
                    policies=relevant_policies
                )
            )

class EmailProcessingSystem:
    def __init__(self, client):
        # Create shared retrievers for all agents
        self.policy_retriever = PolicyRetriever()
        self.example_retriever = EmailResponseRetriever()
        
        # Initialize agents with both retrievers
        self.analyzer = EmailAgent("analyzer", client, self.policy_retriever, self.example_retriever)
        self.drafter = EmailAgent("drafter", client, self.policy_retriever, self.example_retriever)
        self.casual_drafter = EmailAgent("casual_drafter", client, self.policy_retriever, self.example_retriever)
        self.reviewer = EmailAgent("reviewer", client, self.policy_retriever, self.example_retriever)
        self.policy_justifier = EmailAgent("policy_justifier", client, self.policy_retriever, self.example_retriever)
        self.example_justifier = EmailAgent("example_justifier", client, self.policy_retriever, self.example_retriever)

    def process_email(self, email_content, use_casual_tone=False):
        max_attempts = 3
        attempt = 1

        while attempt <= max_attempts:
            print(f"\nProcessing email - Attempt {attempt}")

            # Step 1: Analyze email
            print("\nAnalyzing email content...")
            analysis = self.analyzer.process(email_content)

            # Step 2: Analyze sentiment
            print("\nAnalyzing sentiment...")
            sentiment = prompt_llm(
                self.analyzer.prompts["sentiment"].format(content=email_content)
            )

            # Step 3: Draft response - choose between casual or policy-based
            print(f"\nDrafting response based on analysis... {'(Casual tone)' if use_casual_tone else '(Policy-based)'}")
            if use_casual_tone:
                draft = self.casual_drafter.process(analysis)
            else:
                draft = self.drafter.process(analysis)

            # Get relevant policies and examples for display
            relevant_policies = self.policy_retriever.get_relevant_policy(email_content)
            relevant_examples = self.example_retriever.get_relevant_example(email_content)

            # Add policy justification
            policy_justification = self.policy_justifier.process(
                f"Email: {email_content}\nPolicies: {relevant_policies}"
            )
            
            # Add example justification
            example_justification = self.example_justifier.process(
                f"Email: {email_content}\nExamples: {relevant_examples}"
            )

            # Display formatted output
            print("\n" + "=" * 50)
            print("ORIGINAL EMAIL:\n")
            print(email_content)
            print("\n" + "=" * 50)
            print("DRAFT RESPONSE:\n")
            print(draft)
            
            if use_casual_tone:
                print("\n" + "=" * 50)
                print("EXAMPLE USED:\n")
                print(relevant_examples)
                print("\nEXAMPLE JUSTIFICATION:\n")
                print(example_justification)
            else:
                print("\n" + "=" * 50)
                print("POLICY USED:\n")
                print(relevant_policies)
                print("\nPOLICY JUSTIFICATION:\n")
                print(policy_justification)
                print("\n" + "=" * 50)
                print("SIMILAR EMAIL EXAMPLES:\n")
                print(relevant_examples)
                
            print("\n" + "=" * 50)

            # Ask user for feedback on the draft
            print("\nAre you satisfied with this draft? (y/n)")
            user_feedback = input().lower()

            if user_feedback != "y":
                print("\nMoving to next attempt...")
                attempt += 1
                continue

            # Step 4: Review response
            print("\nReviewing draft response...")
            review = self.reviewer.process(draft)
            print("\nReview completed. Feedback:")
            print(review)

            if "APPROVED" in review:
                return {
                    "status": "success",
                    "analysis": analysis,
                    "final_draft": draft,
                    "review": review,
                    "relevant_policies": relevant_policies,
                    "relevant_examples": relevant_examples,
                    "policy_justification": policy_justification,
                    "example_justification": example_justification,
                    "sentiment": sentiment
                }
            else:
                print(f"\nRevision needed. Feedback: {review}")
                attempt += 1

        return {"status": "failed", "message": "Maximum revision attempts reached"}

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

# Add the function to process emails with the new system
def process_emails(emails_list, use_casual_tone=False):
    email_system = EmailProcessingSystem(client)
    results = {}

    for email in emails_list:
        print(f"\nProcessing email: {email}")
        print("\nWould you like to process this email? (y/n)")

        user_input = input().lower()
        if user_input != "y":
            print("Skipping this email...")
            continue
            
        # Allow user to toggle tone for each email
        if not use_casual_tone:
            print("\nWould you like to use a more casual tone for this response? (y/n)")
            tone_choice = input().lower()
            current_tone = tone_choice == "y"
        else:
            current_tone = True

        result = email_system.process_email(email, use_casual_tone=current_tone)
        results[email] = result

    return results

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

# Add medical context examples
medical_example_emails = [
    """Subject: Patient Data Access Request
    Hello, I'm a referring physician and need access to my patient's recent test results.
    What's the procedure for requesting these records? Thanks.""",
    """Subject: Insurance Coverage Question
    Hi, I'm scheduled for a procedure next week and wanted to confirm if my insurance
    is accepted at your facility. I have BlueCross BlueShield. Best regards.""",
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
    
    # Test EmailResponseRetriever if specified
    if len(sys.argv) > 1 and sys.argv[1] == "--test-retriever":
        # Example usage
        print("\nTesting EmailResponseRetriever...")
        retriever = EmailResponseRetriever()

        # Test queries with expected example matches
        test_queries = [
            "My package hasn't arrived yet",
            "I got a broken item in the mail",
            "What sizes do you have?",
            "The app keeps crashing",
        ]

        for query in test_queries:
            # Get similarity scores for all examples
            query_embedding = retriever.encoder.encode(query)
            similarities = {
                k: cosine_similarity([query_embedding], [emb])[0][0]
                for k, emb in retriever.example_embeddings.items()
            }

            # Get the most relevant example and its score
            best_match = max(similarities.items(), key=lambda x: x[1])
            example_name, score = best_match

            print(f"\nQuery: {query}")
            print("-" * 80)
            print(f"Best matching example: {example_name}")
            print(f"Similarity score: {score:.3f}")
            print(f"Example content:\n{retriever.examples[example_name]}")
            print("-" * 80)
        return
    
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
    print("\n\nWelcome to the Medical Company Email Processing System!\n")
    # Choose which interface to use - uncomment one of these
    main()  # Launch the Gradio interface
    # results = process_emails(medical_example_emails)  # Use command line interface
