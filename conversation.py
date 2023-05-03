from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0.99)
conversation = ConversationChain(llm=llm, verbose=True)

words = ""
while words != "EXIT":
    words = input("What would you like to say? ")
    output = conversation.predict(input=words)
    print(output)
    print()