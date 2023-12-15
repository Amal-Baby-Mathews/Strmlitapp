import streamlit as st
# from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFLoader
from io import BytesIO
from PyPDF2 import PdfReader, PdfWriter
import spacy
import time
# import os
# from uuid import uuid4
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

def fill_form(template_path, data):
  """
  Fills a PDF form with provided data and saves the filled document.

  Args:
    template_path: Path to the PDF form template.
    data: Dictionary mapping form field names to their corresponding values.

  Returns:
    None. Saves the filled PDF document as "filled_form.pdf".

  Raises:
    KeyError: If a data key doesn't match any field names in the form.
  """

  reader = PdfReader(template_path)
  writer = PdfWriter()

  for page_num in range(len(reader.pages)):
    # Get page object
    page = reader.pages[page_num]

    # Create a new page and copy content from original page
    new_page = writer.add_page(page)
    new_page.merge_page(page)

    # Loop through data and update matching form fields
    for field_name, field_value in data.items():
      found = False
      for field in page["/AcroForm"].keys():
        if field == field_name:
          # Set field value if found
          page["/AcroForm"][field].update({"V": field_value})
          found = True
          break

      if not found:
        raise KeyError(f"Field name '{field_name}' not found in the form.")

  # Save the filled form as "filled_form.pdf"
  writer.write("filled_form.pdf")

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
        fill_form(template, matched_data)
        st.success("Filled form generated! Download it here:")
        st.download_button("filled_form.pdf", "filled_form.pdf")

else:
    st.info(
        "Please upload both the form template and relevant documents to proceed. We accept PDFs for templates and TXT files for documents."
    )

#