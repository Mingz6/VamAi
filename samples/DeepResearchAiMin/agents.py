# suppress warnings
import warnings
import os

warnings.filterwarnings("ignore")
from together import Together

# Get Client
your_api_key = "your_key"
client = Together(api_key=your_api_key)


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


class SummarizerAgent:
    def __init__(self):
        self.client = Together(api_key=your_api_key)

    def process(self, content):
        prompt = """SYSTEM: You are an expert text summarizer. 
        Your task is to condense the provided text into a clear, informative summary of exactly 4 lines.

        INSTRUCTIONS:
        • Identify and include only the most important information
        • Maintain the core meaning of the original text
        • Ensure the summary is exactly 4 lines long
        • Use concise, clear language
        • Show output only - provide just the summary
        • Do not include any other text or comments, show only the summary
        * do not say "Here is a 4-line summary: " show the summary directly and nothing else
        
        Text to summarize: {content}
        
        Provide a 4-line summary:"""

        return prompt_llm(prompt.format(content=content))


class InsightAgent:
    def __init__(self):
        self.client = Together(api_key=your_api_key)

    def process(self, summary_files):
        # Read all summary files
        all_summaries = ""
        for file_path in summary_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    all_summaries += content + "\n\n"
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")

        return self._generate_insights(all_summaries)

    def process_text(self, summaries):
        # Process a list of summary texts directly
        all_summaries = "\n\n".join(summaries)
        return self._generate_insights(all_summaries)

    def _generate_insights(self, all_summaries):
        prompt = """SYSTEM: You are an expert analyst who can identify key insights from multiple summaries.

        INSTRUCTIONS:
        • Review the provided summaries about a topic
        • Identify 3 key insights that represent the most important takeaways
        • Format your response as exactly 3 bullet points
        • Each bullet point must be a single sentence
        • Be concise, clear, and informative
        • Do not include any introductory or concluding text
        • Show only the 3 bullet points, nothing else
        
        Summaries to analyze:
        {summaries}
        
        Provide exactly 3 bullet point insights:"""

        return prompt_llm(prompt.format(summaries=all_summaries))


class RecommenderAgent:
    def __init__(self):
        self.client = Together(api_key=your_api_key)

    def process(self, insights, summaries, user_goal, persona=""):
        prompt = """SYSTEM: You are an expert business consultant who provides actionable recommendations.

        INSTRUCTIONS:
        • Review the provided insights and summaries about a topic
        • Consider the user's specific goal: {user_goal}
        • Consider the user's persona: {persona}
        • Recommend exactly 2 specific, actionable steps the user can take to achieve their goal
        • Each recommendation should be practical, specific, and directly related to the goal
        • Format your response as exactly 2 bullet points
        • Each bullet point should be 1-2 sentences
        • Be concise, clear, and actionable
        • Do not include any introductory or concluding text
        • Show only the 2 bullet point recommendations, nothing else
        
        Insights:
        {insights}
        
        Additional context from summaries:
        {summaries}
        
        User's goal: {user_goal}
        User's persona: {persona}
        
        Provide exactly 2 actionable recommendations:"""

        return prompt_llm(
            prompt.format(
                insights=insights,
                summaries="\n\n".join(summaries),
                user_goal=user_goal,
                persona=persona if persona else "General user",
            )
        )

    def suggest_next_query(self, insights, summaries, user_goal, persona=""):
        """Generate a suggested next search query based on insights, summaries, and user goal."""
        prompt = f"""
        Based on the following insights and summaries about a topic, and considering the user's goal and persona,
        suggest ONE specific search query the user should try next to further their research.
        
        INSIGHTS:
        {insights}
        
        SUMMARIES:
        {summaries}
        
        USER'S GOAL:
        {user_goal}
        
        USER'S PERSONA:
        {persona if persona else "General user"}
        
        Suggest a specific, focused search query (5-10 words) that would help the user find additional 
        information to achieve their goal. The query should be different from what they've likely 
        already searched, and should help fill gaps in their current knowledge.
        
        NEXT SEARCH QUERY:
        """

        return prompt_llm(prompt)
