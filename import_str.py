import streamlit as st
import fitz
# from langchain.document_loaders import PyPDFLoader
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter
import spacy
from langchain.embeddings import HuggingFaceEmbeddings
import time
# import os
# from uuid import uuid4
from pathlib import Path
import re
from langchain.vectorstores import FAISS
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
def save_path(template_file):
    save_folder = r'C:\Users\seq_amal\Strmlitapp\data'
    save_path = Path(save_folder, template_file.name)
    with open(save_path, mode='wb') as w:
        w.write(template_file.getvalue())

    if save_path.exists():
        st.success(f'File {template_file.name} is successfully saved!')
    return save_path

# Function definitions
def get_form_structure(file_path):
  """
  Extracts form structure from a PDF document.

  Args:
    file_path: Path to the PDF file.

  Returns:
    List of dictionaries containing information about each form field, including:
      name: Field name.
      bbox: Bounding box coordinates (left, bottom, right, top).
      type: Field type (e.g., "text", "checkbox").

  Raises:
    IOError: If file cannot be opened.
  """

  with open(file_path, "rb") as f:
    pdf_bytes = f.read()

  # Convert file content to PdfReader object
  reader = PdfReader(BytesIO(pdf_bytes))

  form_fields = []
  for page_num in range(len(reader.pages)):
    # Get page dictionary
    page = reader.pages[page_num]

    # Check if page dictionary has "/AcroForm" field
    if "/AcroForm" not in page:
      continue

    # Extract form fields for current page
    page_fields = page["/AcroForm"]
    for field_name in page_fields.keys():
      # Get field dictionary
      field = page_fields[field_name]

      # Extract relevant information
      bbox = field.get("Rect", [0, 0, 0, 0])
      field_type = field.get("/T", "unknown")

      form_fields.append({
        "name": field_name,
        "bbox": bbox,
        "type": field_type,
      })

  return form_fields

def extract_data(file_path):
    #spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    with open(str(file_path), "r") as f:
        text = f.read().strip()
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({"text": ent.text, "label": ent.label_})
    return entities

def match_data_to_fields(form_fields, extracted_data):
    matched_data = {}
    for field in form_fields:
        for entity in extracted_data:
            if field["name"].lower() in entity["text"].lower():
                matched_data[field["name"]] = entity["text"]
                break
    return matched_data
qAPI_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
qheaders = {"Authorization": "Bearer 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"}
def queryQ(payload):
	response = requests.post(qAPI_URL, headers=qheaders, json=payload)
	return response.json()

def fill_form(pdf_path, data):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        )
        texts = [" "]
        index = FAISS.from_texts(texts,embeddings)
        # Initialize Faiss index and Hugging Face pipeline
        
        for document in data:
            text_splitter = RecursiveCharacterTextSplitter(
               chunk_size=100,
               chunk_overlap=50,
               )
            documents = text_splitter.split_text(document["text"])
            index.add_texts(documents)
        
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)

        # Iterate through pages and lines
        for page in doc:
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4]  # Get block text

                # Identify empty form line using regex
                if re.match(r"(.*):\s*$", text):
                    field_label = re.match(r"(.*):", text).group(1)  # Extract field label
                    relevant_documents = index.similarity_search_with_relevance_scores(field_label)
                    answer = []                   
                    for document, relevance_score in relevant_documents:
                        if relevance_score > 0.4:
                            print(relevance_score,document.page_content)
                            answer.append(document.page_content)
                    answer = ' '.join(answer)
                    # Semantic search using Faiss
                    query = f"What is the value for {field_label}?"
                    output = queryQ({
        "inputs": {
            "question": query,
            "context": answer
        },
    })
                    time.sleep(20)
                    # Fill empty space with retrieved information
                    filled_text = text.replace(":", ": " + output)
                    page.insert_text((block[0], block[1], block[2], block[3]), filled_text)

        # Save filled PDF
        doc.save("Filled_up.pdf")
        doc.close()
# Streamlit app
mystyle = '''
Hello nice to meet you
'''
st.title("Form Filler 🪄")
st.markdown(mystyle)

# Upload documents section
st.subheader("Upload Documents ")
template_file = st.file_uploader("Form Template (.pdf)", type=["pdf"], accept_multiple_files=False)
document_files = st.file_uploader("Relevant Documents (.txt)", type=["txt"], accept_multiple_files=True)

# Processing and results section
if template_file and document_files:
    st.markdown("**The file is sucessfully Uploaded.**")
    template=str(save_path(template_file))
    # doc=save_path(document_files)
    progress_bar = st.empty()
    st.write("Analyzing documents...")
    with st.spinner("Processing documents..."):
        start_time = time.time()
        form_fields = get_form_structure(template)
        extracted_data = []
        for document in document_files:
            doc=save_path(document)
            print(doc)
            extracted_data.extend(extract_data(doc))
        matched_data = match_data_to_fields(form_fields, extracted_data)
        progress_bar.progress(100)
        end_time = time.time()
        st.write(f"Documents processed in {round(end_time - start_time, 2)} seconds.")

    # Display extracted entities
    st.subheader("Extracted Entities ")
    st.table(extracted_data)
    # Missing data and manual input
    missing_fields = [field for field in form_fields if field["name"] not in matched_data]
    if missing_fields:
        st.write("**Missing data:**")
        for field in missing_fields:
            new_data = st.text_input(f"{field['name']}:")
            if new_data:
                matched_data[field["name"]] = new_data

    # Generate and download filled form
    if st.button("Generate Filled Form 🪄"):
        fill_form(template, extracted_data)
        st.success("Filled form generated! Download it here:")
        st.download_button("filled_form.pdf", "filled_form.pdf")

else:
    st.info(
        "Please upload both the form template and relevant documents to proceed. We accept PDFs for templates and TXT files for documents."
    )

#