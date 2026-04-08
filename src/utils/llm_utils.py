from src.utils.generic_Utils import CONFIG
import os
import threading
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils.db_utils import insert_llm_costing_evaluation, get_llm_pricing
from src.utils.generic_Utils import CONFIG
from langchain_openai import ChatOpenAI
import sys
from google.oauth2 import service_account
from langchain_google_vertexai import ChatVertexAI
from src.utils.log_utils import insert_logs


if os.path.exists(CONFIG["GOOGLE_SERVICE_ACCOUNT_JSON"]) and CONFIG["REGION_CONSTRAINT_LLM"]==True:
    credentials = service_account.Credentials.from_service_account_file(
        CONFIG["GOOGLE_SERVICE_ACCOUNT_JSON"] )
    print("Google service account credentials loaded for region constraint.")

provider = CONFIG["LLM_PROVIDER"].lower()



class LLMInitializer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LLMInitializer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Separate cached instances for each provider
        self._google_llm = None
        self._openai_llm = None
        self._lock_instances = threading.Lock()

    def get_llm(self, modelVariant=None):
        """Return a cached LLM instance per provider."""
        global provider  # assuming provider is set somewhere
        with self._lock_instances:
            if provider == "gemini":
                if self._google_llm is None:
                    if CONFIG["REGION_CONSTRAINT_LLM"] == False:
                        print("Region constraint not applied")
                        model_name = modelVariant if modelVariant else CONFIG["GEMINI_MODEL"]
                        print(f"Using Gemini LLM with model variant: {model_name}")
                        self._google_llm = ChatGoogleGenerativeAI(
                            model=model_name,
                            timeout=120,
                            max_retries=2,
                            temperature=0,
                            google_api_key=os.getenv("GEMINI_API_KEY")
                        )
                    else:  # Region constraint applied
                        print("Region constraint applied")
                        self._google_llm = ChatVertexAI(
                            model="gemini-2.5-flash",
                            location=CONFIG["TERRITORY"],
                            credentials=credentials,
                            temperature=0
                        )
                return self._google_llm

            elif provider == "openai":
                if self._openai_llm is None:
                    print("Using OpenAI LLM")
                    self._openai_llm = ChatOpenAI(
                        model=CONFIG["OPENAI_MODEL"],
                        api_key=os.getenv("OPEN_AI_API_KEY")
                    )
                return self._openai_llm

            else:
                raise ValueError(f"Unsupported provider: {provider}")


        
get_llm = LLMInitializer().get_llm

# def get_llm(modelVariant=None):
#     if provider == "gemini" and CONFIG["REGION_CONSTRAINT_LLM"]==False:
#         print("Region constraint not applied")
#         if modelVariant:
#             print(f"Using Gemini LLM with model variant: {modelVariant}")
#             googleLlm = ChatGoogleGenerativeAI(model=modelVariant,timeout = 120, max_retries=2, temperature=0, google_api_key=os.getenv("GEMINI_API_KEY"))
#             return googleLlm
#         else:
#             googleLlm = ChatGoogleGenerativeAI(model=CONFIG["GEMINI_MODEL"],timeout = 120, max_retries=2, temperature=0, google_api_key=os.getenv("GEMINI_API_KEY"))
#         return googleLlm
#     elif provider == "gemini" and CONFIG["REGION_CONSTRAINT_LLM"]==True:
#         print("Region constraint applied")
#         googleLlm = ChatVertexAI(
#         model="gemini-2.5-flash",
#         location=CONFIG["TERRITORY"],
#         credentials=credentials,
#         temperature=0)
#         return googleLlm

            
#     elif provider == "openai":
#         print("Using OpenAI LLM")
#         openAiLlm = ChatOpenAI(model=CONFIG["OPENAI_MODEL"], api_key=os.getenv("OPEN_AI_API_KEY"))
#         return openAiLlm

llm = get_llm()





def invoke_with_costing_evalution(prompt, batchId=None, fileId=None, promptId=None, **kwargs):
    """
    Invokes the LLM and logs costing with batchId, fileId, and promptId.
    """
    error = {}
    result = {}
    try:
        try :
            result=get_llm().invoke(prompt, **kwargs)
        except Exception as e:
            print(f"Error invoking LLM: {e}")
            error["errorMessage"] = "Error invoking LLM: " + str(e)
            return error

        insert_logs(message=f"LLM invocation successful. {result}", logType='information', feature="", bulkId=batchId, fileId=fileId)

        # Check if result has 'usage_metadata' attribute
        if hasattr(result, 'usage_metadata') and result.usage_metadata:
            if isinstance(result.usage_metadata, dict) and result.usage_metadata['total_tokens']:
                print("input_tokens :",result.usage_metadata["input_tokens"],"output_tokens :",result.usage_metadata["output_tokens"], "total_tokens:", result.usage_metadata['total_tokens'])
                
                pricing_info=get_llm_pricing(modelVariant=CONFIG["GEMINI_MODEL"])
                print("pricing_info type : ",type(pricing_info))
                total_cost = 0
                if pricing_info:
                    ratePerMillionInputTokens = pricing_info.get("ratePerMillionInputTokens")
                    ratePerMillionOutputTokens = pricing_info.get("ratePerMillionOutputTokens")
                    input_cost = (result.usage_metadata["input_tokens"] / 1_000_000) * ratePerMillionInputTokens
                    output_cost = (result.usage_metadata["output_tokens"] / 1_000_000) * ratePerMillionOutputTokens
                    total_cost = input_cost + output_cost
                    print(f"Cost for input tokens: {input_cost}, Cost for output tokens: {output_cost}, Total cost: {total_cost}")

                insert_llm_costing_evaluation(batchId=batchId, 
                                            fileId=fileId,
                                            promptId=promptId, 
                                            inputTokens=result.usage_metadata["input_tokens"], 
                                            outputTokens=result.usage_metadata["output_tokens"], 
                                            totalTokens=result.usage_metadata['total_tokens'],
                                            totalCostInUSD=total_cost)
        else:
            print("total_tokens not found in result")
        return result
    except Exception as e:
        print(f"Error invoke_with_costing_evalution : {e}")
        error["errorMessage"] = str(e)
        return error
    
def invoke_answer_with_costing_evaluation(prompt, batchId=None, fileId=None, promptId=None, **kwargs):
    """
    Invokes the answer LLM using gemini-2.5-flash (not lite) at temperature=0.
    Uses streaming to keep the connection alive for large prompts, preventing
    504/DeadlineExceeded errors on long-running generation. Chunks are collected
    and reassembled into a single AIMessage-compatible object before returning,
    so callers see no difference from a regular invoke() response.
    """
    error = {}
    try:
        answer_model = CONFIG.get("GEMINI_ANSWER_MODEL", "gemini-2.5-flash")
        if CONFIG["REGION_CONSTRAINT_LLM"]:
            answer_llm = ChatVertexAI(
                model=answer_model,
                location=CONFIG["TERRITORY"],
                credentials=credentials,
                temperature=0,
            )
        else:
            answer_llm = ChatGoogleGenerativeAI(
                model=answer_model,
                timeout=480,
                max_retries=3,
                temperature=0,
                google_api_key=os.getenv("GEMINI_API_KEY"),
            )

        # Stream chunks and collect — keeps the HTTP/2 connection alive throughout
        # generation so long responses don't hit the idle timeout that causes 504s.
        chunks = list(answer_llm.stream(prompt, **kwargs))
        if not chunks:
            raise ValueError("LLM stream returned no chunks")

        # Merge all chunks into a single response object
        result = chunks[0]
        for chunk in chunks[1:]:
            result = result + chunk

    except Exception as e:
        print(f"Error invoking answer LLM: {e}")
        error["errorMessage"] = "Error invoking answer LLM: " + str(e)
        return error

    insert_logs(message=f"Answer LLM invocation successful.", logType='information', feature="", bulkId=batchId, fileId=fileId)

    if hasattr(result, 'usage_metadata') and result.usage_metadata:
        if isinstance(result.usage_metadata, dict) and result.usage_metadata.get('total_tokens'):
            pricing_info = get_llm_pricing(modelVariant=answer_model)
            total_cost = 0
            if pricing_info:
                input_cost = (result.usage_metadata["input_tokens"] / 1_000_000) * pricing_info.get("ratePerMillionInputTokens", 0)
                output_cost = (result.usage_metadata["output_tokens"] / 1_000_000) * pricing_info.get("ratePerMillionOutputTokens", 0)
                total_cost = input_cost + output_cost
            insert_llm_costing_evaluation(
                batchId=batchId, fileId=fileId, promptId=promptId,
                inputTokens=result.usage_metadata["input_tokens"],
                outputTokens=result.usage_metadata["output_tokens"],
                totalTokens=result.usage_metadata["total_tokens"],
                totalCostInUSD=total_cost,
            )
    return result


def invoke_dora_compliance_with_costing_evalution(prompt, batchId=None, fileId=None, promptId=None, **kwargs):
    """
    Invokes the LLM and logs costing with batchId, fileId, and promptId.
    """
    error = {}
    result = {}
    try:
        try :
            modelVariant= CONFIG["GEMINI_REASONING_MODEL"]
            result=get_llm(modelVariant=modelVariant).invoke(prompt, **kwargs)
        except Exception as e:
            print(f"Error invoking LLM: {e}")
            error["errorMessage"] = "Error invoking LLM: " + str(e)
            return error
                    
        # Check if result has 'usage_metadata' attribute
        if hasattr(result, 'usage_metadata') and result.usage_metadata:
            if isinstance(result.usage_metadata, dict) and result.usage_metadata['total_tokens']:
                print("input_tokens :",result.usage_metadata["input_tokens"],"output_tokens :",result.usage_metadata["output_tokens"], "total_tokens:", result.usage_metadata['total_tokens'])
                
                pricing_info=get_llm_pricing(modelVariant=CONFIG["GEMINI_MODEL"])
                print("pricing_info type : ",type(pricing_info))
                total_cost=0
                if pricing_info:
                    ratePerMillionInputTokens = pricing_info.get("ratePerMillionInputTokens")
                    ratePerMillionOutputTokens = pricing_info.get("ratePerMillionOutputTokens")
                    input_cost = (result.usage_metadata["input_tokens"] / 1_000_000) * ratePerMillionInputTokens
                    output_cost = (result.usage_metadata["output_tokens"] / 1_000_000) * ratePerMillionOutputTokens
                    total_cost = input_cost + output_cost
                    print(f"Cost for input tokens: {input_cost}, Cost for output tokens: {output_cost}, Total cost: {total_cost}")

                insert_llm_costing_evaluation(batchId=batchId, 
                                            fileId=fileId,
                                            promptId=promptId, 
                                            inputTokens=result.usage_metadata["input_tokens"], 
                                            outputTokens=result.usage_metadata["output_tokens"], 
                                            totalTokens=result.usage_metadata['total_tokens'],
                                            totalCostInUSD=total_cost)
        else:
            print("total_tokens not found in result")
        return result
    except Exception as e:
        print(f"Error invoke_with_costing_evalution : {e}")
        error["errorMessage"] = str(e)
        return error