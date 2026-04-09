from typing import Optional, Dict, Any
from src.utils.connection_utils import db , client
from src.utils.generic_Utils import generate_formatted_uuid
from bson import ObjectId
import ssl, pymongo, os
from pymongo.errors import PyMongoError
from pymongo.errors import ConnectionFailure, OperationFailure
from src.utils.log_utils import insert_logs
import datetime
from src.utils.generic_Utils import CONFIG
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

def _is_transient_mongo_error(exc: BaseException) -> bool:
    """Returns True for SSL/network errors that are safe to retry on MongoDB."""
    if isinstance(exc, (ConnectionFailure, ssl.SSLError)):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in (
        "sslv3", "bad record mac", "connection reset", "broken pipe",
        "timed out", "temporarily unavailable", "network error",
    ))

_mongo_retry = retry(
    retry=retry_if_exception(_is_transient_mongo_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)

# MongoDB collection names
CLASSIFICATION_COLLECTION = "classification_details"
CONTENT_COLLECTION = "contract_content_holder"
AGENT_ACTIVITY_COLLECTION = "agent_activity"

MONGODB_URI = os.environ.get("MONGODB_URI")
DB_NAME = os.environ.get('MONGODB_DB_NAME')
connection_string = os.environ.get('AZURE_BLOB_STORAGE_CONNECTION_STRING')

# def setup_database():
#     """Checks the MongoDB connection and logs the status."""
#     insert_logs(message="-- setup_database started --", logType='information')
#     try:
#         db.command('ping')
#         insert_logs(message="MongoDB connection successful. Collections will be created on first use.", logType='information')
#         return True
#     except ConnectionFailure as e:
#         insert_logs(message=f"CRITICAL: MongoDB connection failed: {e}", logType='critical')
#         return False
#     finally:
#         insert_logs(message="-- setup_database finished --", logType='information')

class MongoDBExecutor:
    def __init__(self):
        self.connected = True
        self.client = client
        self.db = db
        # try:
        #     self.client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        #     self.db = self.client[DB_NAME]
        #     self.client.admin.command('ping')
        #     self.connected = True
        #     insert_logs(message="MongoDBExecutor initialized successfully.", logType='information')
        # except ConnectionFailure as e:
        #     insert_logs(message=f"CRITICAL: MongoDBExecutor failed to connect on initialization: {e}", logType='critical')
   
        
    
    def execute_query(self, query_data: Dict) -> Dict[str, Any]:
        # self.db = self.connect()
        if self.db is None:
            insert_logs(message="Query execution skipped: MongoDBExecutor is not connected.", logType='error')
            return {"success": False, "error": "MongoDB not connected", "results": [], "count": 0}
        
        collection_name = query_data.get("collection")
        pipeline = query_data.get("pipeline", [])
        insert_logs(message=f"-- Executing query on collection '{collection_name}' --", logType='information')

        try:
            if not collection_name:
                raise ValueError("No collection specified in query_data.")
            
            collection = self.db[collection_name]
            cursor = collection.aggregate(pipeline)
            results = [self._convert_objectid(doc) for doc in cursor]
            
            insert_logs(message=f"Query executed successfully, found {len(results)} record(s) in '{collection_name}'.", logType='information')
            return {"success": True, "collection": collection_name, "results": results, "count": len(results), "pipeline": pipeline}
        except OperationFailure as e:
            insert_logs(message=f"DB_OPERATION_ERROR: Query failed on '{collection_name}': {e}", logType='error')
            return {"success": False, "error": str(e), "results": [], "count": 0}
        except Exception as e:
            insert_logs(message=f"An unexpected error occurred during query execution on '{collection_name}': {e}", logType='error')
            return {"success": False, "error": str(e), "results": [], "count": 0}

    def _convert_objectid(self, obj):
        if isinstance(obj, ObjectId): return str(obj)
        elif isinstance(obj, dict): return {key: self._convert_objectid(value) for key, value in obj.items()}
        elif isinstance(obj, list): return [self._convert_objectid(item) for item in obj]
        return obj

MongoDBExecutorInstance = MongoDBExecutor()
def insert_agent_activity(timestamp, contract_name, agent, outcome, db_path=None):
    insert_logs(message=f"-- Inserting agent activity for '{agent}' --", logType='information')
    try:
        doc = {
            "Timestamp": str(timestamp), "Contract_Name": contract_name,
            "Agent": agent, "Outcome": outcome
        }
        db[AGENT_ACTIVITY_COLLECTION].insert_one(doc)
        insert_logs(message="Agent activity record inserted successfully.", logType='information')
        return True, None
    except OperationFailure as e:
        error_msg = f"DB_OPERATION_ERROR: Failed to insert agent activity. Error: {e}"
        insert_logs(message=error_msg, logType='error')
        return False, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during agent activity insertion: {e}"
        insert_logs(message=error_msg, logType='error')
        return False, error_msg
        
def read_prompt(promptId: str) -> any:
    """Reads a prompt from the MongoDB collection 'prompts' based on prompt Id."""
    insert_logs(message=f"-- Reading prompt with ID: {promptId} --", logType='information')
    try:
        collection = db['prompts']
        promptInfo = collection.find_one({"promptId": promptId})
        if promptInfo:
            insert_logs(message=f"Successfully found prompt '{promptInfo.get('promptName', promptId)}'.", logType='information')
            return promptInfo
        else:
            insert_logs(message=f"DB_WARNING: Prompt with ID '{promptId}' not found in the 'prompts' collection.", logType='error')
            return None
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Could not read prompt '{promptId}'. Error: {e}", logType='error')
        return None
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred while reading prompt '{promptId}': {e}", logType='error')
        return None

def updateCollection(collectionName, keyAttributeName, keyAttributeValue, updateAttributes):
    """Update a document in a MongoDB collection based on a key attribute."""
    insert_logs(message=f"-- Updating collection '{collectionName}' where '{keyAttributeName}' is '{keyAttributeValue}' --", logType='information')
    try:
        result = db[collectionName].update_one(
            {keyAttributeName: keyAttributeValue},
            {"$set": updateAttributes}
        )
        if result.matched_count == 0:
            insert_logs(message=f"DB_WARNING: No document found in '{collectionName}' matching the query. Nothing updated.", logType='error')
            return False
        elif result.modified_count == 0:
            insert_logs(message=f"Document found in '{collectionName}' but no changes were needed.", logType='information')
            return False
        elif result.modified_count > 0:
            insert_logs(message=f"Successfully updated {result.modified_count} document(s) in '{collectionName}'.", logType='information')
            return True
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to update '{collectionName}'. Error: {e}", logType='error')
        return False
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred while updating '{collectionName}': {e}", logType='error')
        return False

def upsertCollection(collectionName, keyAttributeName, keyAttributeValue, updateAttributes):
    """Update a document (or insert if not exists) in a MongoDB collection."""
    insert_logs(message=f"-- Upserting collection '{collectionName}' where '{keyAttributeName}' is '{keyAttributeValue}' --", logType='information')
    try:
        result = db[collectionName].update_one(
            {keyAttributeName: keyAttributeValue},
            {"$set": updateAttributes},
            upsert=True
        )
        if result.upserted_id:
            insert_logs(message=f"No document found. Inserted a new document with ID '{result.upserted_id}' into '{collectionName}'.", logType='information')
            return False
        elif result.modified_count > 0:
            insert_logs(message=f"Found and updated {result.modified_count} document(s) in '{collectionName}'.", logType='information')
            return True
        else:
            insert_logs(message=f"Document found in '{collectionName}' but no changes were needed.", logType='information')
            return False
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to upsert '{collectionName}'. Error: {e}", logType='error')
        return False
    except Exception as e:
        print(e)
        insert_logs(message=f"An unexpected error occurred while upserting '{collectionName}': {e}", logType='error')
        return False

def insert_json_string_as_document(jsonOutput, collectionName):
    """Inserts a dictionary or a list of dictionaries into a specified MongoDB collection."""
    insert_logs(message=f"-- Inserting document(s) into '{collectionName}' --", logType='information')
    try:
        if isinstance(jsonOutput, list):
            if not jsonOutput:
                insert_logs(message="Input list is empty, no documents to insert.", logType='information')
            db[collectionName].insert_many(jsonOutput)
            insert_logs(message=f"Successfully inserted {len(jsonOutput)} documents into '{collectionName}'.", logType='information')
        elif isinstance(jsonOutput, dict):
            db[collectionName].insert_one(jsonOutput)
            insert_logs(message=f"Successfully inserted 1 document into '{collectionName}'.", logType='information')
        else:
            raise TypeError("Input data must be a dictionary or a list of dictionaries.")
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to insert document(s) into '{collectionName}'. Error: {e}", logType='error')
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred while inserting into '{collectionName}': {e}", logType='error')

def insert_obligation(name, description, metadata) -> str:
    """Insert a new obligation document into the obligations collection."""
    insert_logs(message=f"-- Inserting obligation '{name}' --", logType='information')
    try:
        obligationExtractionId = generate_formatted_uuid()
        obligation = {
            "obligationExtractionId": obligationExtractionId,
            "name": name, "description": description, "metadata": metadata
        }
        db["obligationextractions"].insert_one(obligation)
        insert_logs(message=f"Successfully inserted obligation '{name}' with ID '{obligationExtractionId}'.", logType='information')
        return obligationExtractionId
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to insert obligation '{name}'. Error: {e}", logType='error')
        return ""
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred while inserting obligation '{name}': {e}", logType='error')
        return ""

def insert_collection(collectionName: str, data: dict) -> bool:
    """Insert a single document into a specified collection."""
    insert_logs(message=f"-- Inserting document into '{collectionName}' --", logType='information')
    try:
        db[collectionName].insert_one(data)
        insert_logs(message=f"Successfully inserted document into '{collectionName}'.", logType='information')
        return True
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to insert document into '{collectionName}'. Error: {e}", logType='error')
        return False
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred while inserting into '{collectionName}': {e}", logType='error')
        return False

def insert_llm_costing_evaluation(batchId:str, fileId:str, promptId:str, inputTokens:int, outputTokens:int, totalTokens:int, totalCostInUSD:float) -> bool:
    """Insert LLM costing evaluation data into the 'costevalutionforllm' collection."""
    insert_logs(message=f"-- Inserting LLM cost evaluation for fileId '{fileId}' --", logType='information')
    try:
        data = {
            "batchId": batchId, "fileId": fileId, "promptId": promptId,
            "inputTokens": inputTokens, "outputTokens": outputTokens,
            "totalTokens": totalTokens, "totalCostInUSD": totalCostInUSD
        }
        db["costevalutionforllm"].insert_one(data)
        return True
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to insert LLM costing. Error: {e}", logType='error')
        return False
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred while inserting LLM costing: {e}", logType='error')
        return False

def get_llm_pricing(modelVariant: str) -> Optional[Dict[str, Any]]:
    """Fetch LLM pricing details for a specific model variant."""
    insert_logs(message=f"-- Fetching LLM pricing for model '{modelVariant}' --", logType='information')
    try:
        pricing_info = db['llmpricing'].find_one({"modelVariant": modelVariant})
        if not pricing_info:
            insert_logs(message=f"DB_WARNING: No pricing info found for model '{modelVariant}'.", logType='error')
        return pricing_info
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to fetch LLM pricing. Error: {e}", logType='error')
        return None
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred while fetching LLM pricing: {e}", logType='error')
        return None

def get_costing_per_fileId(batchId: str, fileId: str) -> Optional[tuple]:
    """Fetch and aggregate LLM costing details for a specific batch and file ID."""
    insert_logs(message=f"-- Calculating total cost for fileId '{fileId}' in batch '{batchId}' --", logType='information')
    try:
        consumption_info_list = list(db['costevalutionforllm'].find({"batchId": batchId, "fileId": fileId}))
        if not consumption_info_list:
            insert_logs(message=f"No costing data found for fileId '{fileId}'.", logType='information')
            return 0.0, 0.0, 0.0

        total_cost_per_fileId = sum(float(cl.get("totalCostInUSD", 0)) for cl in consumption_info_list)
        total_input_tokens_per_fileId = sum(float(cl.get("inputTokens", 0)) for cl in consumption_info_list)
        total_output_tokens_per_fileId = sum(float(cl.get("outputTokens", 0)) for cl in consumption_info_list)
        
        return total_cost_per_fileId, total_input_tokens_per_fileId, total_output_tokens_per_fileId
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to fetch costing data. Error: {e}", logType='error')
        return None, None, None
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred while fetching costing data: {e}", logType='error')
        return None, None, None

def get_collection_details_by_key(collectionName: str, key:str, value: str) -> any:
    """Reads a document from a MongoDB collection based on a specific key-value pair."""
    insert_logs(message=f"-- Fetching from '{collectionName}' where '{key}' is '{value}' --", logType='information')
    try:
        document = db[collectionName].find_one({key: value})
        if not document:
            insert_logs(message=f"DB_WARNING: No document found in '{collectionName}' for the given key-value pair.", logType='error')
            return {"error": f"Document with {key}='{value}' not found."}
        return document
    
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to fetch document. Error: {e}", logType='error')
        return {"error": f"DB_OPERATION_ERROR: Failed to fetch document. Error: {e}"}
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred while fetching document: {e}", logType='error')
        return {"error": f"An unexpected error occurred while fetching document: {e}"}

def get_promptids_by_documentType(documentType: str) -> list:
    insert_logs(message=f"-- Getting prompt IDs for document type '{documentType}' --", logType='information')
    try:
        document = get_collection_details_by_key("documenttypes", "documentType", documentType)
        if not document:
            insert_logs(message=f"Document type '{documentType}' not found. Cannot retrieve prompt mappings.", logType='error')
            return []
        
        documentId = document.get("documentId")
        if not documentId:
            insert_logs(message=f"Document type '{documentType}' exists but has no 'documentId'.", logType='error')
            return []

        mapping_document = db['documentmappings'].find_one({"documentId": documentId})
        if not mapping_document:
            insert_logs(message=f"No prompt mapping found for documentId '{documentId}'.", logType='information')
            return []
            
        return mapping_document.get("promptIds", [])
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred in get_promptids_by_documentType: {e}", logType='error')
        return []

def get_Attribute_Clause_promptids_by_documentType(documentType: str) -> list:
    """Fetches prompt IDs for a document type, filtered for 'Attribute' or 'Clause' types."""
    try:
        insert_logs(message=f"-- Getting Attribute/Clause prompts for document type '{documentType}' --", logType='information')
        prompt_ids = get_promptids_by_documentType(documentType)
        if not prompt_ids:
            return []

        documents = db['prompts'].find({
            "promptId": {"$in": prompt_ids},
            "promptType": {"$in": ["Attribute", "Clause"]}
        })
        return [doc["promptId"] for doc in documents if "promptId" in doc]
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to filter prompts. Error: {e}", logType='error')
        return []
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred in get_Attribute_Clause_promptids: {e}", logType='error')
        return []

def get_obligation_promptids_by_documentType(documentType: str) -> list:
    """Fetches prompt IDs for a document type, filtered for 'Obligation' type."""
    try:
        insert_logs(message=f"-- Getting Obligation prompts for document type '{documentType}' --", logType='information')
        prompt_ids = get_promptids_by_documentType(documentType)
        if not prompt_ids:
            return []

        documents = db['prompts'].find({
            "promptId": {"$in": prompt_ids},
            "promptType": "Obligation"
        })
        return [doc["promptId"] for doc in documents if "promptId" in doc]
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to filter obligation prompts. Error: {e}", logType='error')
        return []
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred in get_obligation_promptids: {e}", logType='error')
        return []

def get_id_from_prompt_name(promptName: str) -> Optional[str]:
    try:
        insert_logs(message=f"-- Getting prompt ID for prompt name '{promptName}' --", logType='information')
        document = db["prompts"].find_one({"promptName": promptName})
        if document:
            return document.get("promptId")
        else:
            insert_logs(message=f"DB_WARNING: No prompt found with name '{promptName}'.", logType='error')
            return None
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to get prompt ID for name '{promptName}'. Error: {e}", logType='error')
        return None
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred getting prompt ID for name '{promptName}': {e}", logType='error')
        return None

def insert_totalTimeTaken(batchId, fileId=None, totalTimeTakenByFile=None, totalTimeTakenByBatch=None):
    insert_logs(message=f"-- Inserting total time taken for batch '{batchId}' --", logType='information')
    try:
        if fileId is not None and totalTimeTakenByFile is not None:
            db["batches"].update_one(
                {"batchId": batchId, "files.fileId": fileId},
                {"$set": {"files.$.totalTimeTakenByFile": f"{totalTimeTakenByFile} secs"}}
            )
            insert_logs(message=f"Updated totalTimeTakenByFile for fileId {fileId}.", logType='information')
        
        if totalTimeTakenByBatch is not None:
            db["batches"].update_one(
                {"batchId": batchId},
                {"$set": {"totalTimeTakenByBatch": f"{totalTimeTakenByBatch:.2f} secs"}}
            )
            insert_logs(message=f"Updated totalTimeTakenByBatch for batchId {batchId}.", logType='information')
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to insert total time taken. Error: {e}", logType='error')
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred while inserting total time taken: {e}", logType='error')

def get_documentTypes():
    insert_logs(message="-- Fetching all document types --", logType='information')
    try:
        documenttype_collection = db["documenttypes"]
        return [doc.get("documentType") for doc in documenttype_collection.find() if doc.get("documentType") != "Classification"]
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to fetch document types. Error: {e}", logType='error')
        return []
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred while fetching document types: {e}", logType='error')
        return []
    

def update_and_insert_agent_messages(messageId, agentMessages_to_add):
    """
    Updates an existing document by appending messages to the agentMessage array,
    or inserts a new document if one with the messageId does not exist.
    """
    insert_logs(message=f"-- Upserting agent messages for messageId: '{messageId}' --", logType='information')
    
    # Ensure the message to add is always a list
    if not isinstance(agentMessages_to_add, list):
        agentMessages_to_add = [agentMessages_to_add]

    # Add latestTimestamp in UTC to each message
    for msg in agentMessages_to_add:
        msg['latestTimestamp'] = datetime.datetime.utcnow()

    try:
        @_mongo_retry
        def _do_upsert():
            return db["agentmessages"].update_one(
                {"messageId": messageId},
                {
                    "$push": {"agentMessage": {"$each": agentMessages_to_add}},
                    "$setOnInsert": {"messageId": messageId},
                },
                upsert=True,
            )

        _do_upsert()
        insert_logs(message="Agent message record upserted successfully in agentmessages collection.", logType='information')
        return True, None
    except OperationFailure as e:
        error_msg = f"DB_OPERATION_ERROR: Failed to upsert agent message in agentmessages collection. Error: {e}"
        insert_logs(message=error_msg, logType='error')
        return False, error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred during agent message upsertion in agentmessages collection: {e}"
        insert_logs(message=error_msg, logType='error')
        return False, error_msg

def updateConversation(collectionName, keyAttributeName, keyAttributeValue, updatePayload, array_filters=None):
    """
    Updates a document in a MongoDB collection using a flexible update payload.

    Args:
        collectionName (str): The name of the collection.
        keyAttributeName (str): The name of the key attribute for the query (e.g., '_id').
        keyAttributeValue (any): The value of the key attribute.
        updatePayload (dict): A dictionary representing the update operations (e.g., {'$set': {'field': 'value'}, '$push': {'array': 'item'}}).
    """
    insert_logs(message=f"-- Updating collection '{collectionName}' where '{keyAttributeName}' is '{keyAttributeValue}' --", logType='information')
    try:
        @_mongo_retry
        def _do_update():
            return db[collectionName].update_one(
                {keyAttributeName: keyAttributeValue},
                updatePayload,
                array_filters=array_filters,
            )

        result = _do_update()
        if result.matched_count == 0:
            insert_logs(message=f"DB_WARNING: No document found in '{collectionName}' matching the query. Nothing updated.", logType='error')
            return False
        elif result.modified_count == 0:
            insert_logs(message=f"Document found in '{collectionName}' but no changes were needed.", logType='information')
            return False
        elif result.modified_count > 0:
            insert_logs(message=f"Successfully updated {result.modified_count} document(s) in '{collectionName}'.", logType='information')
            return True
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to update '{collectionName}'. Error: {e}", logType='error')
        return False
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred while updating '{collectionName}': {e}", logType='error')
        return False

def get_user_input(messageId):
    """
    Retrieves the content of a user's message from the database.

    Args:
        messageId: The ID of the message to retrieve.

    Returns:
        The message content as a string, or None if an error occurs
        or the message is not found.
    """
    try:
        messagesCollection = db["messages"]
        userMessageDoc = messagesCollection.find_one({"messageId": messageId, "role": "user"})
        
        # Check if the document was found
        if not userMessageDoc:
            print(f"⚠️ No user message found for messageId={messageId}")
            return None
        
        # Safely get the 'content' field
        user_input = userMessageDoc.get("content")
        if isinstance(user_input, list) and len(user_input)>0:
            user_input=user_input[0].get("content")

        if user_input is None:
            print(f"⚠️ 'content' field is missing in the document for messageId={messageId}")
            return None
            
        return user_input

    except PyMongoError as e:
        # This will catch any database-related errors (e.g., connection issues)
        print(f"❌ An error occurred while accessing the database: {e}")
        return None
        
    except Exception as e:
        # A general catch-all for any other unexpected errors
        print(f"❌ An unexpected error occurred: {e}")
        return None
    
def get_fileInfo_from_messages(conversationId, messageId):
    """
    Retrieves structured file information from a MongoDB 'messages' collection
    based on the given conversationId and messageId.

    Returns:
        list[dict] | None: List of dicts with keys 'fileId', 'fileName', 'blobName', 
                           or None if no attachments found.
    """
    try:
        messages_record = db["messages"].find_one(
            {"conversationId": conversationId, "messageId": messageId}
        )

        if not messages_record:
            print(f"No message found for conversationId={conversationId}, messageId={messageId}")
            return None

        attachments = messages_record.get("attachments", [])
        if not isinstance(attachments, list) or not attachments:
            print("No valid 'attachments' field found in message record.")
            return None

        # Build structured file info list
        fileInfo = [
            {
                "fileId": attachment.get("fileId"),
                "fileName": attachment.get("fileName"),
                "blobName": attachment.get("blobName")
            }
            for attachment in attachments
        ]
        return fileInfo

    except Exception as e:
        print(f"Error in get_fileInfo_from_messages: {e}")
        return None

def get_fileInfo_from_conversation(conversationId):
    """
    Retrieves structured file information for a conversation.

    Primary path: messages.attachments (legacy / main-app pipeline).
    Fallback path: filePages collection (CI pipeline — files uploaded via
        POST /ci/upload are stored there directly, not in message attachments).

    Returns:
        list[dict] | None: List of dicts with keys 'fileId', 'fileName', 'blobName',
                           or None if no files found by either path.
    """
    try:
        # --- Primary: messages.attachments ---
        messages_record = db["messages"].find({"conversationId": conversationId})
        all_attachments = []
        for message in messages_record:
            attachments = message.get("attachments", [])
            if isinstance(attachments, list):
                all_attachments.extend(attachments)

        if all_attachments:
            return [
                {
                    "fileId": a.get("fileId"),
                    "fileName": a.get("fileName"),
                    "blobName": a.get("blobName", ""),
                }
                for a in all_attachments
            ]

        # --- Fallback: filePages (CI upload pipeline) ---
        fp_records = list(db["filePages"].find(
            {"conversationId": conversationId},
            {"fileId": 1, "fileName": 1, "localPath": 1},
        ))
        if fp_records:
            return [
                {
                    "fileId": r.get("fileId"),
                    "fileName": r.get("fileName", ""),
                    "blobName": r.get("localPath", ""),
                }
                for r in fp_records
                if r.get("fileId")
            ]

        print(f"No valid attachments found for conversationId={conversationId}")
        return None

    except Exception as e:
        print(f"Error in get_fileInfo_from_messages: {e}")
        return None

def get_aiMessageId(conversationId):
    try:
        messages_collection = db["messages"]
        message_record = messages_collection.find_one({"conversationId": conversationId, "role": "ai"})
        if message_record:
            return message_record.get("messageId")
        else:
            return None  # No AI message found
    except Exception as e:
        print(f"Error retrieving AI messageId: {e}")
        return None

def get_personaBasedChatPrompt():
    try:
        promptId = get_id_from_prompt_name("PERSONA_BASED_QUERY_REWRITER")
        if not promptId:
            raise ValueError("Prompt ID not found for PERSONA_BASED_QUERY_REWRITER.")

        persona_based_prompt = read_prompt(promptId=promptId)
        return persona_based_prompt["promptText"] if persona_based_prompt else None

    except Exception as e:
        print(f"Error retrieving persona-based chat prompt: {e}")
        return None

# Get the matching collection rows from the collection based on the query provided
def get_collection_rows(collectionName: str, query: Dict) -> list:
    try:
        collection = db[collectionName]
        matching_rows = collection.find(query)
        return list(matching_rows)
    except Exception as e:
        print(f"Error retrieving matching rows: {e}")
        return []

def get_workbench_promptInfo_by_batchId(batchId: str, fileId: str, fileStatus: str) -> Optional[Dict[str, Any]]:
    """Fetch workbench prompt information for a specific batch and file ID."""
    insert_logs(message=f"-- Fetching workbench prompt info for batchId '{batchId}' and fileId '{fileId}' --", logType='information')
    try:

        # Fetch the workbench info based on batchId
        # Fetch the status of fileId from files collection if needed
        file_info = db['files'].find_one({"fileId": fileId})
        if not file_info:
            insert_logs(message=f"DB_WARNING: No file info found for fileId '{fileId}'.", logType='error')
            return None

        if(fileStatus == "queued" or fileStatus == "Queued"):
            insert_logs(message=f"FileId '{fileId}' is still in 'Queued' status. fetch all workbench prompts irrespective of isProcessed flag", logType='information')
        else:
            insert_logs(message=f"FileId '{fileId}' is in '{fileStatus}' status. Fetching only unprocessed workbench prompts.", logType='information')

        workbench_info = db['tabularworkbenches'].find_one({"batchId": batchId})
        if not workbench_info:
            insert_logs(message=f"DB_WARNING: No workbench prompt info found for batchId '{batchId}' and fileId '{fileId}'.", logType='error')
            return None

        # If the file is not queued, filter extractionColumns to only include unprocessed prompts
        # extractionColumns is json array of objects with label, prompt, isProcessed flag
        # So we filter based on isProcessed flag and return only those prompts which are not processed
        # keep label and prompt in the dictionary

        if fileStatus.lower() != "queued" and workbench_info and "extractionColumns" in workbench_info:
            unprocessed_columns = [
                col for col in workbench_info["extractionColumns"]
                if not col.get("isProcessed", False)
            ]
        else:
            unprocessed_columns = workbench_info.get("extractionColumns", []) if workbench_info else []

        # extract label and prompt from unprocessed_columns and store as dictionary 
        promptsInfo = [
            {"promptName": col["label"], "promptText": col["prompt"]}
            for col in unprocessed_columns
        ]

        return promptsInfo
    except OperationFailure as e:
        insert_logs(message=f"DB_OPERATION_ERROR: Failed to fetch workbench prompt info. Error: {e}", logType='error')
        return None
    except Exception as e:
        insert_logs(message=f"An unexpected error occurred while fetching workbench prompt info: {e}", logType='error')
        return None

def update_is_processed_flag(collection_name, batch_id, field, value):
    """
    Connects to MongoDB and updates a field in the extractionColumns array
    for documents matching the given batchId.
    """
    try:
        collection = db[collection_name]
        
        # 2. Define the Query (Filter)
        # Find documents where the 'batchId' matches the target.
        query = {
            "batchId": batch_id
        }

        # 3. Define the Update Operation
        # The $set operator combined with the positional operator $[] updates ALL 
        # elements in the 'extractionColumns' array to the new value.
        update_operation = {
            "$set": {
                f"extractionColumns.$[].{field}": value
            }
        }
        
        # 4. Execute the Update
        # By default, update_many updates all documents matching the query.
        result = collection.update_many(query, update_operation)
        
        # 5. Output Result
        if result.modified_count > 0:
            insert_logs(message=f"Successfully updated {result.modified_count} document(s).", logType='information')
            insert_logs(message=f"Batch ID '{batch_id}' updated: 'extractionColumns.*.{field}' is now set to {value}.", logType='information')
        elif result.matched_count > 0 and result.modified_count == 0:
             insert_logs(message=f"Matched {result.matched_count} document(s), but 0 modifications were made.", logType='warning')
             insert_logs(message=f"This likely means the field 'extractionColumns.*.{field}' was already set to {value}.", logType='warning')
        else:
            insert_logs(message=f"No documents found with batchId: '{batch_id}' in the collection.", logType='warning')

    except Exception as e:
        insert_logs(message=f"An error occurred while updating isProcessed flag: {e}", logType='error')

def upsert_documentextractions(collection_name: str, extractionData: dict):
    """
    Performs an 'Upsert' (Update or Insert) operation on the collection.

    :param collection_name: The name of the collection (e.g., "documentextractions").
    :param extractionData: The dictionary containing the data to insert/update.
    """
    if not all(key in extractionData for key in ['fileId', 'name']):
        insert_logs(message="Error: extractionData must contain 'fileId' and 'name' for upsert.", logType='error')
        return

    # 1. Define the unique filter (query)
    # This determines if a record already exists.
    query = {
        "fileId": extractionData["fileId"],
        "name": extractionData["name"]
    }

    # 2. Define the update operation
    # Use $set to update only the fields present in extractionData.
    update_operation = {
        "$set": extractionData
    }

    # 3. Perform the update with upsert=True
    # If a document matching the query is found, it's updated with the $set data.
    # If no document is found, a new document is inserted using the combination
    # of the 'query' and the '$set' data. (In this case, the $set data *is* the full document).
    try:

        result = db[collection_name].update_one(
            query,
            update_operation,
            upsert=True  # <-- This is the key to the upsert operation
        )

        if result.upserted_id:
            insert_logs(message=f"SUCCESS: Inserted new record into '{collection_name}' (ID: {result.upserted_id})", logType='information')
        else:
            insert_logs(message=f"SUCCESS: Updated existing record in '{collection_name}' (Matched: 1)", logType='information')

    except Exception as e:
        insert_logs(message=f"ERROR: Failed to perform upsert on '{collection_name}': {e}", logType='error')
