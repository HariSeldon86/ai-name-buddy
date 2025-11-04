from typing import List
from rich import print

from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from database import (
    check_abbreviation_exists,
    check_keyword_exists,
)
from vectorstore import get_or_create_vectorstore


from config import Config


@tool
def keyword_lookup_tool(keyword: str) -> str:
    """Checks if a keyword already exists in the database. Returns a definitive answer."""
    if check_keyword_exists(keyword):
        return f"The keyword '{keyword}' already exists in the database. Please ask the user for a different one."
    return f"The keyword '{keyword}' is new."


@tool
def abbreviation_lookup_tool(abbreviation: str) -> str:
    """Checks if an abbreviation already exists in the database. Returns a definitive answer."""
    if check_abbreviation_exists(abbreviation):
        return f"The abbreviation '{abbreviation}' already exists. You must generate a new, unique abbreviation."
    return f"The abbreviation '{abbreviation}' is unique."


@tool
def name_suggestion_tool(keyword: str) -> str:
    """Generates a suggested abbreviation and description for a new keyword based on similar examples in the vector store.
    If the keyword includes a note about an already-taken abbreviation (e.g., 'Matching (avoid: Mtch)'), 
    it will generate an alternative."""
    vectorstore = get_or_create_vectorstore()
    retriever = vectorstore.as_retriever()
    llm = ChatOllama(model=Config.OLLAMA_LLM_MODEL)

    # Check if there's an "avoid" instruction in the keyword
    avoid_note = ""
    if "(avoid:" in keyword.lower():
        avoid_note = "\n\nIMPORTANT: Some abbreviations are already taken and must NOT be used. Generate a DIFFERENT abbreviation."
    
    template = """You are an AI assistant that helps create standardized abbreviations and descriptions for technical keywords.
    Use the following retrieved context to generate a new abbreviation and a brief description for the user's keyword.
    Ensure the description is concise (1 sentence) and clearly explains the meaning of the keyword.
    Ensure the abbreviation is consistent in style and format with the provided examples.{avoid_note}

    CRITICAL GENERAL RULES:
    1. Both Keywords and Abbreviations must have uppercase initial. No more uppercase letters are permitted. For example, "ESC" or "esc" are not correct; "Esc" must be defined instead.

    CRITICAL ABBREVIATION PREFERENCE (study the context examples carefully):
    1. Word ending with "-ing" will have "-g" at the end of the abbreviation
       Examples: "Closing" → "Clsg"
    2. Word ending with "-ed" will have "-d" at the end of the abbreviation
       Examples: "Estimated" → "Estimd"
    3. Word ending with "-ion" will have "-n" at the end of the abbreviation
       Examples: "Estimation" → "Estimn"
    4. Word ending with "-tor" or "-er" will have "-r" at the end of the abbreviation
       Examples: "Estimator" → "Estimr"
    5. Remove vowels from the middle of words to shorten them
    6. Preserve consonants that maintain recognizability
    7. Study the retrieved context examples for patterns related to the keyword

    Context: {{context}}

    User's Keyword: {{keyword}}

    Please provide the suggested abbreviation, description, and explanation in the following format:
    Abbreviation: [Your suggested abbreviation - must be UNIQUE and DIFFERENT from any mentioned as already taken]
    Description: [Your suggested description]
    Explanation: [Explain why you chose this abbreviation, referencing the similar examples from the context that influenced your decision. Mention specific patterns or conventions you followed, especially regarding suffix handling.]
    """.format(avoid_note=avoid_note)
    
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "keyword": RunnablePassthrough()}
        | prompt
        | llm
    )
    response = rag_chain.invoke(keyword)
    return response.content


tools: List[tool] = [
        keyword_lookup_tool,
        abbreviation_lookup_tool,
        name_suggestion_tool,
    ]

llm = ChatOllama(model=Config.OLLAMA_LLM_MODEL)

# Agent Prompt
template = """You are an expert AI assistant for creating standardized technical names.
Your goal is to help the user create a unique and contextually relevant abbreviation and description for a new keyword.

You have access to the following tools:
{tools}

To solve the user's request, you must use the following thought process:
1.  First, check if the user's keyword already exists using the `keyword_lookup_tool`.
2.  If the keyword exists, inform the user and stop.
3.  If the keyword is new, use the `name_suggestion_tool` to generate a candidate abbreviation and description.
4.  Take the abbreviation suggested by the `name_suggestion_tool` and use the `abbreviation_lookup_tool` to check if it is unique.
5.  If the abbreviation is NOT unique, you MUST call the `name_suggestion_tool` again. CRITICALLY IMPORTANT: In the keyword parameter, you must provide the original keyword followed by "(avoid: [abbreviation])" where [abbreviation] is the one that was already taken. For example: "Matching (avoid: Mtch)". This tells the tool to generate a DIFFERENT abbreviation.
6.  Repeat steps 4 and 5 until you have a unique abbreviation. Keep track of ALL abbreviations to avoid in the format: "keyword (avoid: abbr1, abbr2, abbr3)"
7.  Once you have confirmed the keyword is new and the suggested abbreviation is unique, provide the final abbreviation, description, and explanation to the user.

Use the following format for your response:

Thought: Do I need to use a tool? Yes
Action: The action to take, should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: Do I need to use a tool? No
Final Answer: [The final, validated abbreviation and description with explanation in a json format with fields "abbreviation", "description", and "explanation"]

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_template(template)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)