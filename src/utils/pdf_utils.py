import pytesseract
import os, sys
import subprocess
import fitz
from pdf2image import convert_from_path
from google.cloud import vision
from dotenv import load_dotenv
from docx2pdf import convert
from src.utils.generic_Utils import CONFIG
from src.utils.log_utils import insert_logs
from pypdf import PdfReader



def get_poppler_path():
    """
    Determines the correct Poppler binary path from configuration.

    It checks for a valid Windows path first, then a Linux path from the 
    global CONFIG dictionary. It logs the outcome and returns the found path.

    Returns:
        str or None: The path to the Poppler binary directory if found, otherwise None.
    """
    # Initialize path variable
    poppler_bin_path = None
    
    # Check for a valid Windows path in the configuration
    if "POPPLER_BIN_PATH_FOR_WINDOWS" in CONFIG and os.path.exists(CONFIG["POPPLER_BIN_PATH_FOR_WINDOWS"]):
        poppler_bin_path = CONFIG["POPPLER_BIN_PATH_FOR_WINDOWS"]
    
    # If no Windows path, check for a valid Linux path
    elif "POPPLER_BIN_PATH_FOR_LINUX" in CONFIG and os.path.exists(CONFIG["POPPLER_BIN_PATH_FOR_LINUX"]):
        poppler_bin_path = CONFIG["POPPLER_BIN_PATH_FOR_LINUX"]
    
    # Log the result before returning
    if poppler_bin_path:
        insert_logs(message=f"Using Poppler bin path: {poppler_bin_path}", logType='information')
    else:
        insert_logs(message="Poppler path not found in configuration. Will rely on system PATH.", logType='information')  
    return poppler_bin_path


def convert_pdf_to_images(pdf_path, poppler_path=None, max_pages=None):
    """Converts each page of a PDF into an image with robust error handling."""
    insert_logs(message=f"-- convert_pdf_to_images started for: {pdf_path} --", logType='information')
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at '{pdf_path}'")
            
        images = convert_from_path(pdf_path, poppler_path=poppler_path, last_page=max_pages)
        insert_logs(message=f"Successfully converted {len(images)} page(s) to images.", logType='information')
        return images
    except FileNotFoundError as e:
        insert_logs(message=f"CRITICAL: {e}", logType='critical')
        return []
    except Exception as e:
        # This is critical as the OCR process depends on these images.
        insert_logs(message=f"CRITICAL: Failed to convert PDF to images. Poppler might be missing or misconfigured. Error: {e}", logType='critical')
        return []
    finally:
        insert_logs(message=f"-- convert_pdf_to_images finished for: {pdf_path} --", logType='information')

def ocr_images_to_text(image_list):
    """Performs OCR on a list of images and concatenates the extracted text."""
    insert_logs(message=f"-- ocr_images_to_text started for {len(image_list)} image(s) --", logType='information')
    full_text = ""
    page_text_dict = {}

    try:
        if "TESSERACT_CMD_FOR_WINDOWS" in CONFIG and os.path.exists(CONFIG["TESSERACT_CMD_FOR_WINDOWS"]):
            pytesseract.pytesseract.tesseract_cmd = CONFIG["TESSERACT_CMD_FOR_WINDOWS"]
    except Exception as e:
        insert_logs(message=f"Error configuring Tesseract path: {e}", logType='error')

    try:
        for i, img in enumerate(image_list):
            page_num = i + 1
            
            try:
                page_text = pytesseract.image_to_string(img, lang='eng')
                page_text_dict[f"Page-{page_num}"] = page_text
                full_text += page_text + f"\n--- End of Page {page_num} ---\n\n"
                insert_logs(message=f"OCR completed successfully for page {page_num}.", logType='information')
            except Exception as e:
                # Log as an error, but continue to process other pages.
                insert_logs(message=f"OCR_ERROR: Failed to perform OCR on page {page_num}. Tesseract may be misconfigured. Error: {e}", logType='error')
    except Exception as e:
        # This catches errors with the loop itself, which would be critical.
        insert_logs(message=f"CRITICAL: An unexpected error occurred during the image OCR loop. Error: {e}", logType='critical')
    finally:
        insert_logs(message="-- ocr_images_to_text finished --", logType='information')
    return full_text, page_text_dict

def ocr_pdf_with_vision_paginated(file_path: str, mime_type: str):
    """Performs OCR on a local PDF file using Google Vision API with detailed error handling."""
    insert_logs(message=f"-- ocr_pdf_with_vision_paginated started for: {file_path} --", logType='information')
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at '{file_path}'")

        full_text = ""
        page_text_dict = {}

        client = vision.ImageAnnotatorClient()
        with open(file_path, "rb") as f:
            content = f.read()

        input_config = vision.InputConfig(content=content, mime_type=mime_type)
        features = [vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)]
        request = vision.AnnotateFileRequest(input_config=input_config, features=features)
        
        insert_logs(message="Sending request to Google Vision API...", logType='information')
        response = client.batch_annotate_files(requests=[request])
        file_response = response.responses[0]
        
        for i, image_response in enumerate(file_response.responses):
            page_number = i + 1
            page_text = image_response.full_text_annotation.text
            page_text_dict[f"Page-{page_number}"] = page_text
            full_text += page_text + f"\n--- End of Page {page_number} ---\n\n"
            insert_logs(message=f"Google Vision OCR completed for page {page_number}.", logType='information')
            
        insert_logs(message="Google Vision OCR process finished successfully.", logType='information')
        return full_text, page_text_dict
        
    except FileNotFoundError as e:
        insert_logs(message=f"CRITICAL: {e}", logType='critical')
        return "", {}
    except Exception as e:
        insert_logs(message=f"CRITICAL: An unexpected error occurred in ocr_pdf_with_vision_paginated. Error: {e}", logType='critical')
        return "", {}
    finally:
        insert_logs(message=f"-- ocr_pdf_with_vision_paginated finished for: {file_path} --", logType='information')

def scanned_pdf_to_text(pdf_path, pages_to_load=None):
    """Converts a scanned PDF file to plain text using an appropriate OCR method."""
    insert_logs(message=f"-- scanned_pdf_to_text started for: {pdf_path} --", logType='information')
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at '{pdf_path}'")
        environment = os.getenv("APP_ENV", "development") # Default to development if not set

        if environment.lower() == "production":
            insert_logs(message="Production environment detected. Using Google Vision OCR.", logType='information')
            MIME_TYPE = "application/pdf"
            return ocr_pdf_with_vision_paginated(pdf_path, MIME_TYPE)
        else:
            insert_logs(message="Development environment detected. Using local Tesseract OCR.", logType='information')
            poppler_bin_path = get_poppler_path()
            images = convert_pdf_to_images(pdf_path, poppler_bin_path, max_pages=pages_to_load)
            if not images:
                insert_logs(message="No images were generated from PDF. Cannot perform OCR.", logType='error')
                return "", {}
            return ocr_images_to_text(images)
            
    except Exception as e:
        # Catches the FileNotFoundError and any other unexpected error.
        insert_logs(message=f"CRITICAL: Failed to process scanned PDF. Error: {e}", logType='critical')
        return "", {}
    finally:
        insert_logs(message=f"-- scanned_pdf_to_text finished for: {pdf_path} --", logType='information')

def is_scanned_pdf(pdf_path, text_threshold=0.05):
    """Checks if a PDF is likely scanned by analyzing its text content area."""
    insert_logs(message=f"-- is_scanned_pdf check started for: {pdf_path} --", logType='information')
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at '{pdf_path}'")
            
        doc = fitz.open(pdf_path)
        total_page_area = 0.0
        total_text_area = 0.0

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            total_page_area += abs(page.rect)  # Area of the entire page

            text_blocks = page.get_text("blocks")
            for b in text_blocks:
                r = fitz.Rect(b[:4])  # Rectangle of the text block
                total_text_area += abs(r)
        
        doc.close()

        if total_page_area == 0:
            insert_logs(message="PDF has zero page area; assuming it is scanned or empty.", logType='information')
            return True

        text_percentage = total_text_area / total_page_area
        is_scanned = text_percentage < text_threshold
        insert_logs(message=f"Text coverage: {text_percentage:.2%}. Is scanned: {is_scanned}", logType='information')
        return is_scanned
        
    except Exception as e:
        # If any error occurs opening or processing the PDF, it's safer to assume it's scanned and needs OCR.
        insert_logs(message=f"An error occurred in is_scanned_pdf. Defaulting to True (scanned). Error: {e}", logType='error')
        return True
    finally:
        insert_logs(message=f"-- is_scanned_pdf check finished for: {pdf_path} --", logType='information')

def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of a text-based PDF."""
    insert_logs(message=f"-- extract_text_from_pdf started for: {pdf_path} --", logType='information')
    page_text_dict = {}
    doc = None
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at '{pdf_path}'")
        
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            page_text = page.get_text("text")
            # Normalize whitespace for consistency
            single_line_text = ' '.join(page_text.split())
            page_text_dict[f"Page-{i+1}"] = single_line_text
        
        if not page_text_dict:
            insert_logs(message="Could not extract any text from the document.", logType='error')
        else:
            insert_logs(message=f"Successfully extracted text from {len(page_text_dict)} page(s).", logType='information')

        return page_text_dict
        
    except Exception as e:
        # Any failure to open or read the PDF is critical for this function's purpose.
        insert_logs(message=f"CRITICAL: Failed to open or extract text from PDF. It might be corrupted or image-based. Error: {e}", logType='critical')
        return {}
    finally:
        if doc:
            doc.close()
        insert_logs(message=f"-- extract_text_from_pdf finished for: {pdf_path} --", logType='information')

def convert_docx_to_pdf(input_path: str, output_path: str = None):
    """Converts a DOCX file to PDF with robust error handling."""
    insert_logs(message=f"-- convert_docx_to_pdf started for: {input_path} --", logType='information')

    if sys.platform == "win32" :
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input DOCX file not found at '{input_path}'")
            
            insert_logs(message=f"Converting '{input_path}' to PDF at output path '{output_path}'...", logType='information')
            convert(input_path, output_path)
            insert_logs(message="DOCX to PDF conversion complete.", logType='information')
        except Exception as e:
            # A failure here is critical as the function's sole purpose is this conversion.
            insert_logs(message=f"CRITICAL: An error occurred during DOCX to PDF conversion. Check if Word is installed (on Windows) or LibreOffice (on Linux). Error: {e}", logType='critical')
            # Re-raise to let the caller handle the failure.
            raise
        finally:
            insert_logs(message=f"-- convert_docx_to_pdf finished for: {input_path} --", logType='information')
    else:
        pdf_file_path = convert_docx_to_pdf_linux(input_path)
        if not pdf_file_path:
            insert_logs(message="CRITICAL: DOCX to PDF conversion failed on Linux.", logType='critical')
            return None
        insert_logs(message="DOCX to PDF conversion complete.", logType='information')
        insert_logs(message=f"-- convert_docx_to_pdf finished for: {input_path} --", logType='information')
        return pdf_file_path

def convert_docx_to_pdf_linux(input_file, output_path=None):
    """
    Converts a .docx file to .pdf on Linux using LibreOffice.

    :param input_file: The path to the input .docx file.
    :param output_path: The directory to save the PDF. 
                        Defaults to the same directory as the input file.
    """
    if not os.path.exists(input_file):
        insert_logs(message=f"Converting '{input_file}' to PDF at output path '{output_path}'...", logType='error')
        return False

    if output_path is None:
        output_path = os.path.dirname(os.path.abspath(input_file))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    insert_logs(message=f"Converting {input_file} to PDF in {output_path}...", logType='information')


    try:

        # The command to run
        command = [
            'libreoffice',
            '--headless',        # Run without a GUI
            '--convert-to',      # Specify conversion
            'pdf',               # Convert *to* PDF
            '--outdir',          # Specify the output directory
            output_path,
            input_file           # The input file
        ]

        # Run the command
        result = subprocess.run(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            timeout=30,  # Add a timeout
            check=True   # Raise an error if the command fails
        )
        
        # Output filename
        pdf_name = os.path.splitext(os.path.basename(input_file))[0] + '.pdf'
        pdf_file_path = os.path.join(output_path, pdf_name)
        
        if os.path.exists(pdf_file_path):
            insert_logs(message=f"Successfully converted to {pdf_file_path}", logType='information')
            return pdf_file_path
        else:
            insert_logs(message="Conversion ran, but output PDF not found.", logType='error')
            insert_logs(message=f"STDOUT: {result.stdout.decode('utf-8')}", logType='error')
            insert_logs(message=f"STDERR: {result.stderr.decode('utf-8')}", logType='error')
            return False

    except FileNotFoundError:
        insert_logs(message="Error: 'libreoffice' command not found.", logType='error')
        insert_logs(message="Please install LibreOffice on your Linux VM.", logType='error')
        return False
    except subprocess.CalledProcessError as e:
        insert_logs(message=f"Conversion failed with error: {e.stderr.decode('utf-8')}", logType='error')
        return False
    except subprocess.TimeoutExpired:
        insert_logs(message="Error: Conversion timed out.", logType='error')
        return False