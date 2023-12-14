import streamlit as st
# from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyMuPDFLoader as PyMuPDF
import spacy
import time
import os
from uuid import uuid4
from pathlib import Path

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
    with PyMuPDF.open(file_path) as pdf:
        form_fields = []
        for page in pdf:
            for field in page.form_fields:
                form_fields.append({
                    "name": field.name,
                    "bbox": field.bbox,
                    "type": field.type,
                })
    return form_fields

def extract_data(file_path):
    nlp = spacy.load("en_core_web_lg")
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

def fill_form(template_path, data):
    with PyMuPDF.open(template_path, "rb") as pdf:
        writer = PyMuPDF.PdfWriter()
        for page in pdf:
            new_page = writer.addPage()
            new_page.copyContentsFrom(page)
            for field_name, field_value in data.items():
                for field in page.form_fields:
                    if field.name == field_name:
                        field.setValue(field_value)
                        break
        writer.save("filled_form.pdf")

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
    template=save_path(template_file)
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
    st.subheader("Extracted Entities ", style="font-size: 18px; color: #333")
    st.table(extracted_data, header_style="font-weight: bold; color: #333", cell_style="text-align: left")

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
        fill_form(template, matched_data)
        st.success("Filled form generated! Download it here:")
        st.download_button("filled_form.pdf", "filled_form.pdf")

else:
    st.info(
        "Please upload both the form template and relevant documents to proceed. We accept PDFs for templates and TXT files for documents."
    )

#