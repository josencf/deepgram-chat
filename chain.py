from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

#Initialize the chatbot
llm = OpenAI(temperature=0.9)

# Structure the prompt
# input_variables can be a list of as many strings as we want
prompt = PromptTemplate(
    input_variables=["entity", "product"],
    template="What is a good name for a {entity} that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run({"entity": 'company', "product":'colorful socks'}))
print(chain.run({"entity": 'CEO', "product":'colorful socks'}))
print(chain.run({"entity": 'company', "product":'chocolate pizza'}))
print(chain.run({"entity": 'CEO', "product":'chocolate pizza'}))

