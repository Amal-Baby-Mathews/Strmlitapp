import streamlit as st
from langchain.document_loaders import PyPDFLoader
import spacy
import time
import os
from uuid import uuid4
def save_path(uploaded_file, target_folder="data"):
  """
  Saves an uploaded file to the program directory and returns its absolute path.

  Args:
      uploaded_file: An `UploadedFile` object containing the uploaded data.

  Returns:
      The absolute path of the saved file.
  """

  # Get the filename and extension
  filename, extension = os.path.splitext(uploaded_file.name)

  # Generate a unique filename to avoid collision
  unique_filename = f"{filename}_{uuid4()}{extension}"
  file_path =os.path.join(os.path.dirname(__file__), target_folder)

  # Create the "data" folder if it doesn't exist
  if not os.path.exists(file_path):
    os.makedirs(file_path)


  # Write the uploaded file data to the new file
  with open(file_path, "wb") as f:
    f.write(uploaded_file.getvalue())

  # Return the absolute path of the saved file
  return os.path.abspath(file_path)
# Function definitions
def get_form_structure(template_path):
    
    file_path = save_path(template_path)
    reader = PyPDFLoader(file_path) 
    form_fields = []
    for page in reader.pages:
        for field in page.form_fields:
            form_fields.append({
                "name": field.name,
                "bbox": field.bbox,
                "type": field.type,
            })
    return form_fields

def extract_data(document_path):
    file_path=save_path(document_path)
    nlp = spacy.load("en_core_web_lg")
    with open(file_path, "r") as f:
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
    with PyPDFLoader(template_path) as reader, ReportLab.PdfWriter() as writer:
        for page in reader.pages:
            writer.addPage(page)
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
    progress_bar = st.empty()
    st.write("Analyzing documents...")
    with st.spinner("Processing documents..."):
        start_time = time.time()
        form_fields = get_form_structure(template_file)
        extracted_data = []
        for document in document_files:
            extracted_data.extend(extract_data(document))
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
        fill_form(template_file, matched_data)
        st.success("Filled form generated! Download it here:")
        st.download_button("filled_form.pdf", "filled_form.pdf")

else:
    st.info(
        "Please upload both the form template and relevant documents to proceed. We accept PDFs for templates and TXT files for documents."
    )

#