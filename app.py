import streamlit as st
import requests
import base64
import fitz  # PyMuPDF
import os

def call_groq_api(prompt: str, model: str, max_tokens: int, image_data: str = None) -> str:
    """
    Call the Groq API with the given prompt, model, and max_tokens.
    
    Adjust the API endpoint, headers, and payload as required by the Groq API documentation.
    """
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    if image_data:
        # For image analysis using llama-3.2-90b-vision-preview
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt + "\n"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data
                            }
                        }
                    ]
                }
            ],
            "model": model,
            "max_tokens": max_tokens
        }
    else:
        # For text-based queries using other models
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "model": model,
            "max_tokens": max_tokens
        }
    
    # Retrieve the API key from Streamlit secrets
    api_key = st.secrets["GROQ_API_KEY"]
    if not api_key:
        return "API key is not set in the secrets.toml file."
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No answer provided")
        else:
            error_message = response.json().get("error", {}).get("message", "Unknown error")
            return f"Error {response.status_code}: {error_message}"
    except Exception as e:
        return f"Request failed: {e}"

def extract_text_and_images_from_pdf(pdf_path):
    """
    Extract text and images from a PDF file.
    """
    text_content = []
    image_data = []
    
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        
        # Extract text from the page
        text_content.append(page.get_text())
        
        # Extract images from the page
        image_list = page.get_images(full=True)
        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_data.append(base64.b64encode(image_bytes).decode('utf-8'))
    
    return '\n'.join(text_content), image_data

def main():
    st.title("MUJ Quiz: Get all your answers here!")
    st.markdown("Enter your question to get answers from different models using preloaded PDF files.")
    
    # User input for text prompt
    prompt = st.text_input("Enter your prompt:")
    
    # Set max_tokens to a fixed value
    max_tokens = 5000
    
    if st.button("Submit"):
        if not prompt:
            st.warning("Please enter a prompt!")
            return
        
        # Directory containing PDF files
        pdf_directory = "pdf_files"
        
        # Initialize combined text and image data
        combined_text = ""
        combined_images = []
        
        # Process each PDF file in the directory
        for filename in os.listdir(pdf_directory):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_directory, filename)
                text_content, image_data = extract_text_and_images_from_pdf(pdf_path)
                combined_text += text_content + "\n"
                combined_images.extend(image_data)
        
        # Combine prompt with extracted text content
        full_prompt = f"{prompt}\n{combined_text}"
        
        # Define models for text-based queries
        text_models = ["qwen-2.5-32b", "deepseek-r1-distill-llama-70b", "gemma2-9b-it"]
        answers = {}
        
        with st.spinner("Fetching answers..."):
            if combined_images:
                # Use llama-3.2-90b-vision-preview for image analysis
                for img in combined_images:
                    answer = call_groq_api(full_prompt, "llama-3.2-90b-vision-preview", max_tokens, img)
                    answers["llama-3.2-90b-vision-preview"] = answer
            else:
                # Use other models for text-based queries
                for model in text_models:
                    answer = call_groq_api(full_prompt, model, max_tokens)
                    answers[model] = answer
        
        st.success("Answers fetched successfully!")
        st.markdown("### Answers")
        for model, answer in answers.items():
            with st.expander(f"Model: {model.upper()}"):
                st.write(answer)

if __name__ == "__main__":
    main()
