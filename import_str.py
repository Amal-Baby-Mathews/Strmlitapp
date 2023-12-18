import streamlit as st
import fitz
# from langchain.document_loaders import PyPDFLoader
from io import BytesIO
from PyPDF2 import PdfReader
import spacy
from langchain.embeddings import HuggingFaceEmbeddings
import time
import os
# from uuid import uuid4
from pathlib import Path
import re
from langchain.vectorstores import FAISS
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from spacy.matcher import Matcher
from dotenv import load_dotenv

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
    matcher = Matcher(nlp.vocab)
    pattern = [
    {'ORTH': {'REGEX': r'^([A-Za-z ]+)(:([^\n]+))$'}},
  ]

    matcher.add("CustomLabel", [pattern])
    with open(str(file_path), "r") as f:
        text = f.read().strip()

        doc = nlp(text)
        #print(doc)
        entities = []
        # for ent in doc.ents:
        #     entities.append({"text": str(ent), "label": ent.label_})
        for ent in doc:
            if ent.dep_ == "appos":  # Check for appositional modifier dependency
                entities.append({"text": f"{ent.head.text}: {ent}", "label": "InformationPair"})

    
    return entities

def match_data_to_fields(form_fields, extracted_data):
    matched_data = {}
    for field in form_fields:
        for entity in extracted_data:
            if field["name"].lower() in entity["text"].lower():
                matched_data[field["name"]] = entity["text"]
                break
    return matched_data
load_dotenv() #create .env file with the corresponding passwords
# qAPI_URL = str(os.environ["qAPI_URL"])
# qheaders = str(os.environ["qheaders"])
# print(qAPI_URL)
qAPI_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
qheaders = {"Authorization": "Bearer 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"}

def queryQ(payload):
	response = requests.post(qAPI_URL, headers=qheaders, json=payload)
	return response.json()

def fill_form(pdf_path, doc):
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
        
        with open(str(doc), "r") as f:
            text = f.read().strip()

            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=40,
            chunk_overlap=5,
            )
        documents = text_splitter.split_text(text)
        index.add_texts(documents)
        
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)

        # Iterate through pages and lines
        for page in doc:
            
            
            for word in page.get_text("words"):
                
                word_text=word[4]
                x0, y0, x1, y1 = word[0:4]

                # Identify empty form line using regex
                if re.match(r"(.*):\s*$", word_text):
                    #print(word_text)
                    field_label = word_text # Extract field label
                    relevant_documents = index.similarity_search_with_relevance_scores(field_label)
                    
                    answer = []                   
                    for document, relevance_score in relevant_documents:
                        
                        
                        if relevance_score > 0.2 and relevance_score < 0.9:
                            
                            print("relevance!!",relevance_score,document.page_content, "|||",field_label)
                            answer.append(document.page_content)
                        #else:
                            #print("no relevance yet!!",relevance_score,document.page_content,"|||",field_label)
                    if answer !=[]:
                        answer = ' '.join(answer)
                    
                    # Semantic search using Faiss
                    if answer !=[] and answer is not None:
                        
                        query = f"What is the value for {field_label}?"
                        output = queryQ({
            "inputs": {
                "question": query,
                "context": answer
            },
        })
                        # Fill empty space with retrieved information
                        retries = 3  # Define the number of retries

                        for attempt in range(retries):
                            try:
                                
                                out = output["answer"]
                                break  # Success, break out of the loop
                            except KeyError:
                                print(output)
                                print(f"KeyError encountered on attempt {attempt + 1}/{retries}. Retrying in 20 seconds...")
                                time.sleep(20)  # Wait for 20 seconds before retrying
                        filled_text = len(word_text)*" " + out
                        # avg_char_width = fitz.utils.get_avg_char_width(page, word_text)
                        # text_width = fitz.Rect(0, 0, len(word_text) * avg_char_width, 0).width
                        # filled_text_width = fitz.Rect(0, 0, len(filled_text) * fitz.utils.get_avg_char_width(page, filled_text), 0).width
                        # adjustment = (text_width - filled_text_width) / 2
                        point = (x0 + 30 ,y1-5)
                        # page.insert_text(point, filled_text)
                        # Calculate the position to insert filled_text to the right of word_text
                        filled_text_x0 = x0  # Start the filled text immediately after word_text
                        filled_text_y0 = y0  # Align filled text with the same baseline as word_text

                        # Insert filled_text to the right of word_text in the PDF
                        page.insert_text(point, filled_text)
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
    st.write("Analyzing documents")
    with st.spinner("Processing documents..."):
        start_time = time.time()
        form_fields = get_form_structure(template)
        extracted_data = []
        for document in document_files:
            doc=save_path(document)
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
        # fill_form(template, extracted_data)
        fill_form(template, doc)
        st.success("Filled form generated! Download it here:")
        st.download_button("filled_form.pdf", "filled_form.pdf")

else:
    st.info(
        "Please upload both the form template and relevant documents to proceed. We accept PDFs for templates and TXT files for documents."
    )

#
