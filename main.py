import os
import uuid
import logging
from utils.GenAIPlatformLLM import GenAIPlatformLLM
from utils.Auth import load_tokens, decrypt_token, encrypt_token
from utils.Auth import add_user_token as add_user
from utils.Prompt import query_prompt, map_prompt, combine_prompt, prompt, simple_conversation_prompt, scc_system_msg_template, scc_human_msg_template
from langchain.prompts import (
                                SystemMessagePromptTemplate,
                                HumanMessagePromptTemplate,
                                ChatPromptTemplate,
                                MessagesPlaceholder
)
from langchain.schema import AIMessage, HumanMessage
from langchain.chains.summarize import load_summarize_chain
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.prompt import PromptTemplate
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader 
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from elasticsearch import Elasticsearch
#from transformers import GPT2TokenizerFast
import pandas as pd
#import tiktoken
from typing import List
#import requests
import urllib3 

### Chat LLM
from langchain.chains import ConversationChain 
from langchain.memory import ConversationBufferMemory

# Disable InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

###################### Logger ######################


# Setup logger
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('log/api.log')
    console_handler = logging.StreamHandler()
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(file_formatter)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logging = setup_logger()



###################### environment variable ######################


load_dotenv()


try:
    os.environ["BR_URL"] = os.getenv("BR_URL")
    #os.environ["BR_KEY"] = os.getenv("BR_KEY")
except Exception as err:
    logging.error("Environment variables is missing BR_URL")
    raise SystemExit()


app = FastAPI()
script_dir = os.path.dirname(os.path.abspath(__file__))
logging.info("Application started")


api_key_header = APIKeyHeader(name="X-Api-Key")

# in-memory storage for session memory
conversation_memories: Dict[str, ConversationBufferMemory] = {}


###################### Common ######################


async def verify_api_key(api_key: str = Depends(api_key_header)):
    
    tokens = load_tokens("auth/tokens.json")

    decrypted_token = decrypt_token(api_key)

    for user_email, roles in tokens.items():
        for original_token in roles.values():
            if original_token == decrypted_token["decrypted_token"]:
                return {"token": api_key, "mail": user_email ,"llm_token": os.getenv("BR_KEY") }
                
    raise HTTPException(status_code=401, detail="Invalid API key")

    



def split_text(text):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
        separators = ["\n", "\r\n"]
    )

    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs


def get_memory(session_id: Optional[str] = None):
    # Check if session_id is provided, if not, create a new one
    if not session_id:
        session_id = str(uuid.uuid4())
        memory = ConversationBufferMemory()
        conversation_memories[session_id] = memory
    else:
        memory = conversation_memories.get(session_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Session not found")

    return memory, session_id

###################### Routes ######################


@app.get("/", description="Welcome endpoint, provides information about the API and available endpoints.")
async def root():
    return JSONResponse(content={"message": "Welcome to the GTO API. Use /docs to see the available endpoints."})


#class GenAIModelRequest(BaseModel):
#    model_id: str
#    br_api_key: str
#    br_url: str
#    model_kwargs: Dict

# class GenAIModelResponse(BaseModel):
#     api_key: str
#     url: str
#     provider: str
#     model_id: str
#     model_kwargs: Dict


@app.post("/genai_model", description="Retrieve and instantiate a GenAI model based on the provided model ID, API key, and URL.")
async def get_genai_model(
    model_id: str, 
    br_api_key: str,
    br_url: str,
    model_kwargs:Dict
 ) : 
    try:
        provider = GenAIPlatformLLM.get_provider_by_model_id(model_id)

        llm = GenAIPlatformLLM(
            api_key=br_api_key,
            url=br_url,
            provider=provider,
            model_id=model_id,
            model_kwargs=model_kwargs
        )

        return llm
        # return {
        #     "api_key": llm.api_key,
        #     "url": llm.url,
        #     "provider": llm.provider,
        #     "model_id": llm.model_id,
        #     "model_kwargs": llm.model_kwargs,
        # }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


########### ADMIN ###################

# @app.post("/admin/add_user_token", tags=["Admin"], description="Add a new token for a user.")
# def add_user_token( user_email: str, user_type: str, api_key: str = Depends(verify_api_key)):

#     decrypted_token = decrypt_token(api_key["token"])

#     if decrypted_token["role"] == "admin":
#         logging.info(f"Admin token ({api_key['mail']}) used to add a new token for {user_email} ({user_type})")
    
#         try:
#             new_token = add_user( user_email, user_type)
#             logging.info(f"Successfull new token for {user_email} ({user_type})")
#             return JSONResponse(content={"message": f"New token for {user_email} ({user_type}) successfully created" , "new_token" : new_token })
#         except ValueError as e:
#             raise HTTPException(status_code=500, detail=str(e))
#     else:
#         logging.error(f"Unauthorized access by {api_key['mail']}")
#         raise HTTPException(status_code=401, detail="Unauthorized access")



########### QUERY LLM ###################

@app.post("/query/simple" , tags=["Query"], description="Process a simple user query using the specified model and return the generated answer.")
async def get_simple_answer(model_id:str, user_query:str, api_key: str = Depends(verify_api_key), api_endpoint= os.getenv("BR_URL") , model_kwargs:Dict = {"temperature": 0}):
    try:
        
        logging.info(f"{api_key['mail']} - Processing a simple query with model {model_id} and user query: {user_query}")

        llm = await get_genai_model(model_id, api_key["llm_token"], api_endpoint, model_kwargs)

        prompt = PromptTemplate.from_template("{question}")

        chain = prompt | llm
        answer = chain.invoke(user_query)

        logging.info(f"Answer generated successfully")

        return JSONResponse(content={"answer": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/query/prompt_template" , tags=["Query"], description="This endpoint processes a user query using a specified prompt template and returns the generated answer.")
async def get_answer_with_prompt(model_id:str, user_query:dict = {"adjective": "italien", "content":"france"}, prompt_template:str = query_prompt ,  api_key: str = Depends(verify_api_key), api_endpoint= os.getenv("BR_URL") , model_kwargs:Dict = {"temperature": 0}):
    try:
        logging.info(f"{api_key['mail']} - Processing prompt template query with model {model_id} ")

        llm = await get_genai_model(model_id, api_key["llm_token"], api_endpoint, model_kwargs)

        prompt = ChatPromptTemplate.from_template(prompt_template)

        chain = prompt | llm
        answer = chain.invoke(user_query)

        logging.info(f"Answer generated successfully")
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


########### SUMMARIZATION WITH LLM ###################


@app.post("/summarization/map_reduce" ,  tags=["Summarization"], description="Summarize a given text using the map-reduce method with the specified model and prompts.")
async def summarization_map_reduce(model_id:str, text:str, api_key: str = Depends(verify_api_key), api_endpoint= os.getenv("BR_URL") , model_kwargs:Dict = {"temperature": 0}, map_prompt:str = map_prompt, combine_prompt:str = combine_prompt, ) -> JSONResponse:
    try:

        logging.info(f"{api_key['mail']} - Processing a map-reduce summarization with model {model_id} ")

        docs =  split_text(text)

        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])        
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

        llm = await get_genai_model(model_id, api_key["llm_token"], api_endpoint, model_kwargs)

        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=combine_prompt_template,
            verbose=False
        )

        answer = summary_chain.invoke(docs)

        logging.info(answer)
        logging.info(f"Answer generated successfully")
        # JSONResponse(content={"answer": answer})
        return answer
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/summarization/stuff",  tags=["Summarization"], description="Summarize a given text using the 'stuff' method with the specified model and prompt.")
async def summarization_stuff(model_id:str, text:str, api_key: str = Depends(verify_api_key), api_endpoint= os.getenv("BR_URL") , model_kwargs:Dict = {"temperature": 0}, prompt:str = prompt) -> JSONResponse:
    try:
        logging.info(f"{api_key['mail']} - Processing a 'stuff' summarization with model {model_id} ")
        docs =  split_text(text)
        logging.info(text)
        prompt_stuff = PromptTemplate.from_template(prompt)   

        llm = await get_genai_model(model_id, api_key["llm_token"], api_endpoint, model_kwargs)

        summary_chain = load_summarize_chain(llm=llm,
                                            chain_type="stuff",
                                            prompt=prompt_stuff,
                                            verbose=False
                                            )
        answer = summary_chain.invoke({"input_documents" : docs})

        return answer
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





########### Chat LLM ###################




class ConversationResponse(BaseModel):
    response: str
    session_id: str


@app.post("/chat/simple_conversation", response_model=ConversationResponse , tags=["Chat"], description="Engage in a simple conversation with the model, maintaining session memory.")
async def simple_conversation(model_id:str, user_query:str , api_key: str = Depends(verify_api_key), api_endpoint= os.getenv("BR_URL") , model_kwargs:Dict = {"temperature": 0}, prompt:str = simple_conversation_prompt, session_id: Optional[str] = None ):
    try:
        global conversation_memories

        llm = await get_genai_model(model_id, api_key["llm_token"], api_endpoint, model_kwargs)

        memory , session_id = get_memory(session_id)

        PROMPT = PromptTemplate(input_variables=["history", "input"], template=prompt)

        conversation = ConversationChain(memory=memory, prompt=PROMPT, llm=llm, verbose=False)
        
        answer = conversation.predict(input=user_query)
    
        #return {"response": reponse, "session_id": session_id}
        return JSONResponse(content={"response": answer, "session_id": session_id})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/chat/conversation_with_context" , tags=["Chat"], description="Engage in a conversation with the model, providing additional context and maintaining session memory.")
async def conversation_with_context(model_id:str, 
                                    user_query:str ,
                                    rag_context:str ,
                                    api_key: str = Depends(verify_api_key),  
                                    api_endpoint= os.getenv("BR_URL"),
                                    model_kwargs:Dict = {"temperature": 0},
                                    system_msg_template:str = scc_system_msg_template,
                                    human_msg_template:str = scc_human_msg_template ,
                                    session_id: Optional[str] = None 
                                    ):
    try:

    
        global conversation_memories
        llm = await get_genai_model(model_id, api_key["llm_token"], api_endpoint, model_kwargs)

        memory , session_id = get_memory(session_id)


        # Ensure history is correctly formatted as a list of messages
        if not memory.chat_memory.messages:
            memory.chat_memory.add_user_message("I'm Nicolas, how are you?")
            memory.chat_memory.add_ai_message("Hello Nicolas! I'm an AI assistant, How can I assist you today?")
            

        system_msg_template = SystemMessagePromptTemplate.from_template(scc_system_msg_template)
        human_msg_template = HumanMessagePromptTemplate.from_template(scc_human_msg_template)

        PROMPT = ChatPromptTemplate.from_messages([
            system_msg_template,
            human_msg_template,
            SystemMessagePromptTemplate.from_template(template="Provide any additional information that might help clarify the user's query if needed."),
        ])

        history = memory.chat_memory.messages
        conversation = ConversationChain(memory=memory, prompt=PROMPT, llm=llm, verbose=False)
        
        answer = conversation.predict( input=f"\n\nContext:\n{rag_context}\n\nQuery:\n{user_query}")
    
        #return {"response": reponse, "session_id": session_id}
        return JSONResponse(content={"response": answer, "session_id": session_id})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




########### ELASTIC QUERY  ###################




@app.post("/elastic/query_without_scroll", tags=["Elasticsearch"], description="Query Elasticsearch without using the scroll feature, returning a limited set of results.")
async def elk_query_without_scroll(query:str, index:str , es_host:str, es_port:int, es_scheme:str, user:str, password:str):

    # construct the full host URL
    es_full_host = f"{es_scheme}://{es_host}:{es_port}"

    # Elasticsearch query
    query = query

    try:
        # Connect to Elasticsearch with basic authentication and SSL context
        es = Elasticsearch(hosts=es_full_host , basic_auth=(user, password), verify_certs=False)
        logging.info(f"Querying Elasticsearch at {es_full_host} ")

        # iPerform a search query
        answer = es.search(index=index, body=query)
        answer = answer['hits']['hits']
        logging.info(f"Data retrieved successfully")

        #return answer
        return JSONResponse(content={"answer": answer})

    except Exception as e:
        logging.error("Error while querying Elasticsearch", exc_info=True)
        return "Error: " + str(e)




@app.post("/elastic/query_with_scroll", tags=["Elasticsearch"], description="Query Elasticsearch using the scroll feature, allowing retrieval of large sets of results.")
async def elk_query_with_scroll(query:str, index:str, es_host:str, es_port:int, es_scheme:str, user:str, password:str):
    try:

        # construct the full host URL
        es_full_host = f"{es_scheme}://{es_host}:{es_port}"
        logging.info(f"Querying Elasticsearch at {es_full_host} ")

        # Elasticsearch query
        query = query

        # Scroll parameter (keep the search context alive for this duration)
        scroll = '5m'  

        # Connect to Elasticsearch with basic authentication and SSL context
        es = Elasticsearch(hosts=es_full_host , basic_auth=(user, password), verify_certs=False)
        logging.info(f"connexion established: {es}")

        # Perform a search query
        response = es.search(index=index, body=query, scroll=scroll)
        logging.info(f"Search retrieved successfully")

        sid = response['_scroll_id']
        scroll_size = response['hits']['total']['value']
        total_hits = scroll_size
        logging.info(f"Scroll size: {scroll_size}")
        
        # Initialize a list to keep track of the results
        results = []

        # Start scrolling
        while scroll_size > 0:
            # Before fetching the next page, add the current batch to our results list
            results += response['hits']['hits']
            
            # Fetch the next page of results
            response = es.scroll(scroll_id=sid, scroll=scroll)
            
            # Update the scroll ID and size
            sid = response['_scroll_id']
            scroll_size = len(response['hits']['hits'])
            

        #return results
        return JSONResponse(content={"answer": results, "total_hits": total_hits})

    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/elastic/elk_answer_to_dataframe", tags=["Elasticsearch"], description="Convert the result of an Elasticsearch query into a pandas DataFrame and return it.")
async def elk_answer_to_dataframe(query_result:List[Dict]) :
    # List to store the dictionaries of each hit
    data = []

    for item in query_result:
        
        # Initialize an empty dictionary for each hit
        hit_data = {}
        
        # Extract the fields from the hit
        fields = item.get('fields', {})

        # Add the fields to the hit_data dictionary, excluding '_index', '_id', and 'sort'
        for key, value in fields.items():
            hit_data[key] = value[0] if isinstance(value, list) and len(value) == 1 else value
        # Append the hit_data dictionary to the data list
        data.append(hit_data)

    df = pd.DataFrame(data)

    return JSONResponse(content={"dataframe": df.to_dict()})




@app.post("/text/token_count", tags=["text"], description="Count the number of tokens in a given text.")
def count_tokens_gpt(text:str, encoding_name:str = "cl100k_base") :

    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))

    return JSONResponse(content={"token_count": num_tokens})



#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="10.199.154.4", port=8002)