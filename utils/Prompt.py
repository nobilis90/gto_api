from langchain.prompts import PromptTemplate
from langchain.prompts import (
                                SystemMessagePromptTemplate,
                                HumanMessagePromptTemplate,
                                ChatPromptTemplate,
                                MessagesPlaceholder
)


########### QUERY LLM ###################

######### Endpoint : /query/prompt_template

query_prompt = """
"Tell me a {adjective} joke about {content}."
"""



############### Summarization Prompt  ############################

############### MAP REDUCE  
map_prompt =  """
Write a summary of this chunk of text that includes the main points and any important details.
{text}
"""



combine_prompt = """
Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
```{text}```
BULLET POINT SUMMARY:
"""


############### STUFF  

prompt = """
Write a summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
```{text}```
BULLET POINT SUMMARY:
"""





############### Chat Prompt  ############################

############### simple conversation 

simple_conversation_prompt =  """Answer the question as truthfully as possible. You can use the conversation history to have more context.

Current conversation:
{history}

Human: {input}

AI Assistant:
"""


############### simple conversation with context

scc_system_msg_template = """You are a helpful assistant. Answer the question as truthfully as possible using the provided context. 
    If the answer is not contained within the text below, say 'I don't know'. 
    If the query is unclear or ambiguous, ask for clarification. Always be polite and concise.
    
    Historical conversation : {history}
    
    """

scc_human_msg_template = "{input}"

