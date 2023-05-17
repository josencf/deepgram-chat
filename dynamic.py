import os

os.environ["SERPAPI_API_KEY"] = "3dc644f6681aa286a660d1fd7d5568aad165f608c4c58f97ed872e27ebe28c10"
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

#Initialize the chatbot
llm = OpenAI(temperature=0.9)

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")
