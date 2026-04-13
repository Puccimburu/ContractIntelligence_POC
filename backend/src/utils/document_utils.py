import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from src.utils.pdf_utils import is_scanned_pdf, scanned_pdf_to_text, convert_docx_to_pdf
from src.utils.log_utils import insert_logs
from langchain.text_splitter import RecursiveCharacterTextSplitter



def loadDocumentText(document_path : str):

    file_extension = os.path.splitext(document_path)[1].lower()
    loaded_documents: List[Document] = []
    document_text = ""

    if file_extension == '.pdf':
        if is_scanned_pdf(document_path, 0.05):
            print(f"Using scanned PDF loader for {document_path}")
            document_text = scanned_pdf_to_text(document_path, pages_to_load=None)
        else:
            print(f"Using PyPDFLoader for {document_path}")
            loader = PyPDFLoader(document_path)
            loaded_documents = loader.load()

    elif file_extension == '.docx':
        print(f"Using UnstructuredWordDocumentLoader for {document_path}")
        if not document_path:
            raise ValueError("No file path provided to UnstructuredWordDocumentLoader.")
        loader = UnstructuredWordDocumentLoader(document_path)
        loaded_documents = loader.load()
    elif file_extension == '.txt':
        print(f"Using standard text loader for {document_path}")
        with open(document_path, 'r', encoding='utf-8') as f:
            loaded_documents = [Document(page_content=f.read())]
    else:
        print('Extension not supported')
        return ''

    if loaded_documents:
        document_text = "\n".join([doc.page_content for doc in loaded_documents])

    return document_text



def loadDocumentTextPageWise(document_path: str, bulkId: str = "N/A") -> str:
    """
    Loads text content from a document based on its file extension.
    Handles different loader types and scanned PDFs.
    """
    file_extension = os.path.splitext(document_path)[1].lower()
    loaded_documents: List[Document] = []
    document_text = ""
    pageWiseText = {}

    insert_logs(message=f"Attempting to load document: {document_path}", logType='information', bulkId=bulkId)

    try:
        if file_extension == '.pdf':
            if is_scanned_pdf(document_path, 0.05):
                insert_logs(message=f"Using scanned PDF loader for {document_path}", logType='information', bulkId=bulkId)
                document_text, pageWiseText = scanned_pdf_to_text(document_path, pages_to_load=None)
            else:
                insert_logs(message=f"Using PyPDFLoader for {document_path}", logType='information', bulkId=bulkId)
                loader = PyPDFLoader(document_path)
                loaded_documents = loader.load()

        elif file_extension == '.docx':
            # --- MODIFIED BLOCK ---
            # Load DOCX content directly without converting to PDF
            insert_logs(message=f"Using UnstructuredWordDocumentLoader for {document_path}", logType='information', bulkId=bulkId)
            if not document_path:
                raise ValueError("No file path provided to UnstructuredWordDocumentLoader.")
            
            # Use the loader mentioned in the log
            loader = UnstructuredWordDocumentLoader(document_path)
            loaded_documents = loader.load()

            # Split the document into chunks (simulating pages)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # Approximate page size in characters
                chunk_overlap=200,  # Overlap to maintain context
                length_function=len,
            )
            loaded_documents = text_splitter.split_documents(loaded_documents)

            # Convert to PDF for consistent page-wise extraction
            fileName = os.path.basename(document_path)
            file_base_name = fileName.split(".docx")[0]
            pdf_file_name = f"{file_base_name}.pdf"
            converted_pdf_path = os.path.join(os.path.dirname(document_path), pdf_file_name)
            
            try:
                insert_logs(message=f"Converting DOCX to PDF: {pdf_file_name}", logType='information', bulkId=bulkId)
                convert_docx_to_pdf(input_path=document_path, output_path=converted_pdf_path)
                insert_logs(message="Successfully converted DOCX to PDF.", logType='information', bulkId=bulkId)
            except Exception as conversion_error:
                insert_logs(message=f"Could not convert DOCX to PDF. Continuing with main text. Error: {conversion_error}", logType='error', bulkId=bulkId)


        elif file_extension == '.txt':
            insert_logs(message=f"Using standard text loader for {document_path}", logType='information', bulkId=bulkId)
            with open(document_path, 'r', encoding='utf-8') as f:
                loaded_documents = [Document(page_content=f.read())]
                
        else:
            insert_logs(message=f"Extension not supported for file: {document_path}", logType='error', bulkId=bulkId)
            return ''

        if loaded_documents:
            document_text = "\n".join([doc.page_content for doc in loaded_documents])
            pageWiseText = {i+1: doc.page_content for i, doc in enumerate(loaded_documents)}

        return document_text, pageWiseText

    except Exception as e:
        error_msg = f"Error loading document {document_path}. Details: {e.__class__.__name__}: {e}"
        insert_logs(message=error_msg, logType='error', bulkId=bulkId)
        return '', {}

