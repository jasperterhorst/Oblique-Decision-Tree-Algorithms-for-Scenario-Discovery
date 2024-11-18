import os
from docx import Document

# Define the relative path (two levels up) and file name
file_name = "Assignment.docx"
file_path = os.path.abspath(os.path.join(os.getcwd(), "../", file_name))

# Check if the file exists
if os.path.exists(file_path):
    # Load the Word document
    doc = Document(file_path)

    # Extract text from the Word document
    content = [para.text for para in doc.paragraphs if para.text.strip()]  # Filter out empty paragraphs

    # Print the content
    for paragraph in content:
        print(paragraph)
else:
    print(f"The file '{file_name}' was not found at: {file_path}")
