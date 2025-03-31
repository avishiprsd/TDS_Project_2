from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import openai
import os
import tempfile
import zipfile
import pandas as pd
import shutil
import json
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env")

app = FastAPI()

# Configure OpenAI
api_key = os.getenv("OPENAI_API_KEY")


def extract_zip(file_path: str, extract_to: str):
    """Extract a zip file to a directory."""
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def read_csv_content(csv_path: str):
    """Read a CSV file and return its content as a string."""
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV columns: {df.columns.tolist()}")

        return df.to_string(index=False)  # Return CSV content as a string
    except Exception as e:
        return f"Error reading CSV: {str(e)}"


def read_xlsx_content(xlsx_path: str):
    """Read an Excel file and return its content as a string."""
    try:
        df = pd.read_excel(xlsx_path)
        print(df)
        return df.to_string(index=False)
    except Exception as e:
        return f"Error reading XLSX: {str(e)}"


def read_txt_content(txt_path: str):
    """Read a text file and return its content."""
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception as e:
        return f"Error reading TXT: {str(e)}"


def read_json_content(json_path: str):
    """Read a JSON file and return its content."""
    try:
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error reading JSON: {str(e)}"


def read_pdf_content(pdf_path: str):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PdfReader(pdf_path)
        text_content = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        return text_content.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"


def read_docx_content(docx_path: str):
    """Extract text from a DOCX file."""
    try:
        doc = Document(docx_path)
        text_content = "\n".join([p.text for p in doc.paragraphs])
        return text_content.strip()
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

def process_question_with_llm(question: str, context: str = ""):
    """Use OpenAI's API to answer the question."""
    try:
        prompt = f"""
        You are an expert teaching assistant for IIT Madras's Online Degree in Data Science program.
        Your task is to help students with their graded assignments by providing accurate answers.

        Context: {context}

        Question: {question}

        Provide only the exact answer that should be entered in the assignment, with no additional explanation or formatting.
        """

        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful teaching assistant that provides concise, accurate answers to data science assignment questions."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error processing question with LLM: {str(e)}"


@app.post("/api/")
async def answer_question(
    question: str = Form(...), files: List[UploadFile] = File(None)
    
):
    print("Request received!", flush=True)

    try:
        print(f"Received question: {question}")
        print(f"Received {len(files)} file(s).") 
        answers = []
        file_processed = False

        # Process files if uploaded
        if files:
            print(f"Received {len(files)} file(s).")
            for file in files:
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = os.path.join(temp_dir, file.filename)
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)

                    # Process file types
                    if file.filename.endswith(".zip"):
                        extract_dir = os.path.join(temp_dir, "extracted")
                        os.makedirs(extract_dir, exist_ok=True)
                        extract_zip(file_path, extract_dir)

                        for root, _, file_names in os.walk(extract_dir):
                            for f in file_names:
                                ext = f.split(".")[-1].lower()
                                full_path = os.path.join(root, f)
                                answer = process_file_based_on_type(full_path, ext, question)
                                if answer:
                                    answers.append(answer)
                                    file_processed = True

                    else:
                        ext = file.filename.split(".")[-1].lower()
                        answer = process_file_based_on_type(file_path, ext, question)
                        if answer:
                            answers.append(answer)
                            file_processed = True

        # Use LLM if no answers found
        if not answers:
            llm_answer = process_question_with_llm(question)
            answers.append(llm_answer)

        return JSONResponse(content={"answers": answers})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def process_file_based_on_type(file_path: str, ext: str, question: str):
    """Process different file types and pass content as context to LLM."""
    print(ext)
    if ext == "csv":
        context = read_csv_content(file_path)
    elif ext == "xlsx":
        context = read_xlsx_content(file_path)
    elif ext == "txt":
        context = read_txt_content(file_path)
    elif ext == "json":
        context = read_json_content(file_path)
    elif ext == "pdf":
        context = read_pdf_content(file_path)
    elif ext == "docx":
        context = read_docx_content(file_path)
    else:
        context = None

    # Pass content to LLM if available
    if context and not context.startswith("Error"):
        return process_question_with_llm(question, context)

    return None



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
