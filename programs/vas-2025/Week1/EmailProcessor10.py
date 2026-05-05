# Convert to flask
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ApiKey import API_KEY, HGToken
from together import Together
from flask import Flask, render_template, request, jsonify, session
import json

# Function to interact with LLM using Together API
def prompt_llm(prompt, client=None):
    if client is None:
        return "Error: Together client not initialized"
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling Together API: {str(e)}"

class EmailResponseRetriever:
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=HGToken)
        # Sample email response examples with more casual tone
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
        }
        # Pre-compute embeddings for examples
        self.example_embeddings = {
            k: self.encoder.encode(v) for k, v in self.examples.items()
        }

    def get_relevant_response(self, query, top_k=2):
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
    def __init__(self, role, client):
        self.role = role
        self.client = client
        self.response_retriever = EmailResponseRetriever()
        self.policy_retriever = PolicyRetriever()

        self.prompts = {
            "analyzer": """SYSTEM: You are an expert email analyzer for a medical company.
            Your role is to break down emails into key components and provide clear, actionable insights.

            INSTRUCTIONS:
            • Extract main topics and key points from the email
            • Determine urgency level (Low, Medium, High)
            • List all required actions in bullet points
            • Analyze tone of the message (formal, informal, urgent, etc.)
            • Consider similar past responses
            • Highlight any compliance concerns
            • Limit response to 50 words maximum
            • Show response only without additional commentary

            SIMILAR PAST RESPONSES:
            {examples}

            RELEVANT POLICIES:
            {policies}

            Email: {content}""",
            "drafter": """SYSTEM: You are a professional email response specialist for a medical company.
            Draft responses that align with our past successful responses while maintaining HIPAA compliance.

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

            RELEVANT POLICIES:
            {policies}

            Based on this analysis: {content}""",
            "reviewer": """SYSTEM: You are a senior email quality assurance specialist for a medical company.
            Ensure responses meet healthcare communication standards and match our friendly tone.

            INSTRUCTIONS:
            • Verify alignment with example responses
            • Check for HIPAA violations
            • Assess professional yet friendly tone
            • Review completeness of response
            • Evaluate appropriate handling of sensitive information
            • Confirm all action items are clearly stated
            • Limit response to 50 words maximum
            • Show response only without additional commentary

            SIMILAR PAST RESPONSES:
            {examples}

            RELEVANT POLICIES:
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
            "example_justifier": """SYSTEM: You are an example matching expert. In 2 lines, explain why the following example responses are relevant
            to this email content. Be specific and concise.

            Email content: {content}
            Selected examples: {examples}""",
            "policy_justifier": """SYSTEM: You are a policy expert. In 2 lines, explain why the following policies are relevant
            to this email content. Be specific and concise.

            Email content: {content}
            Selected policies: {policies}""",
        }

    def process(self, content):
        # Get relevant example responses for the email content
        relevant_examples = self.response_retriever.get_relevant_response(content)
        relevant_policies = self.policy_retriever.get_relevant_policy(content)
        return prompt_llm(
            self.prompts[self.role].format(content=content, examples=relevant_examples, policies=relevant_policies),
            self.client
        )


class EmailProcessingSystem:
    def __init__(self, client):
        self.analyzer = EmailAgent("analyzer", client)
        self.drafter = EmailAgent("drafter", client)
        self.reviewer = EmailAgent("reviewer", client)
        self.example_justifier = EmailAgent("example_justifier", client)
        self.policy_justifier = EmailAgent("policy_justifier", client)

    def process_email(self, email_content):
        # Step 1: Analyze email content
        print("\nAnalyzing email content...")
        analysis = self.analyzer.process(email_content)

        # Step 2: Analyze sentiment
        sentiment = prompt_llm(
            self.analyzer.prompts["sentiment"].format(content=email_content),
            self.analyzer.client
        )

        # Step 3: Draft response
        print("\nDrafting response based on analysis...")
        draft = self.drafter.process(analysis)

        # Get relevant policies and example responses for display
        relevant_policies = self.analyzer.policy_retriever.get_relevant_policy(email_content)
        relevant_examples = self.analyzer.response_retriever.get_relevant_response(email_content)

        # Add policy justification
        policy_justification = self.policy_justifier.process(
            f"Email: {email_content}\nPolicies: {relevant_policies}"
        )

        # Add example justification
        example_justification = self.example_justifier.process(
            f"Email: {email_content}\nExamples: {relevant_examples}"
        )

        # Step 4: Review response
        review = self.reviewer.process(draft)

        return {
            "status": "success",
            "analysis": analysis,
            "final_draft": draft,
            "review": review,
            "policies": relevant_policies,
            "examples": relevant_examples,
            "policy_justification": policy_justification,
            "example_justification": example_justification,
            "sentiment": sentiment
        }


# Sample emails for testing
sample_emails = [
    """Hi, I need to get my medical records from last month's visit.
    Can you help me with the process? Thanks!""",
    """Hello, I'm trying to schedule an appointment for next week.
    I have Aetna insurance and wanted to confirm you accept it before booking.""",
    """My appointment is tomorrow at 2pm but something urgent came up at work.
    Is there any way I could reschedule?""",
    """Running low on my blood pressure medication.
    Need to get it refilled before next week. What's the process?""",
    """Hi there, just moved to the area and looking for a new primary care doctor.
    Are you accepting new patients with United Healthcare?""",
]


# Flask application
app = Flask(__name__)
app.secret_key = 'medical_email_system_secret_key'  # For session management


@app.route('/')
def index():
    return render_template('index.html', emails=sample_emails)


@app.route('/process', methods=['POST'])
def process():
    email_content = request.form.get('email')
    
    try:
        # Initialize Together client
        client = Together(api_key=API_KEY)
        
        email_system = EmailProcessingSystem(client)
        result = email_system.process_email(email_content)
        
        # Store stats in session
        if 'approved_count' not in session:
            session['approved_count'] = 0
            session['disapproved_count'] = 0
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing email: {str(e)}"
        })


@app.route('/approve', methods=['POST'])
def approve():
    session['approved_count'] = session.get('approved_count', 0) + 1
    return jsonify({
        "status": "success", 
        "approved_count": session['approved_count'],
        "disapproved_count": session['disapproved_count']
    })


@app.route('/disapprove', methods=['POST'])
def disapprove():
    session['disapproved_count'] = session.get('disapproved_count', 0) + 1
    return jsonify({
        "status": "success", 
        "approved_count": session['approved_count'],
        "disapproved_count": session['disapproved_count']
    })


@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "approved_count": session.get('approved_count', 0),
        "disapproved_count": session.get('disapproved_count', 0)
    })


if __name__ == "__main__":
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Error launching application: {str(e)}")
        print("Check that API_KEY is properly set in ApiKey.py")
