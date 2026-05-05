# MONEYCLAW


- **`src/workflows/`** — workflows that are combinations of components
- **`src/components/`** — tools and data connectors like access_gmail and access_calendar

## Style
- Be concise. Short answers, no filler.


## Code Generation Instructions

- When Writing Code always be concise and use the minimal number of code and libraries
- Check & Update requirements.txt with the new libraries needed for the new code (if any)
- Make sure to keep things modular and not to repeat code
- use templates for the html and have a base template
- if you see repeated code, consolidate them please
- make sure to put things in variables and all so you dont repeat the names
- make sure you have one llm-based class used for anything that needs llm (should be in src/llm.py)
- src/prompts is where all prompts should live, and they are in text file
- Always end with saying "🎯 All Done Amigo" after generating code
