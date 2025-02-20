import streamlit as st
import requests
import base64

def call_groq_api(prompt: str, model: str, max_tokens: int, image_data: str = None) -> str:
    """
    Call the Groq API with the given prompt, model, and max_tokens.
    
    Adjust the API endpoint, headers, and payload as required by the Groq API documentation.
    """
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    if image_data:
        messages[0]["content"] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data}}
        ]
    
    payload = {
        "messages": messages,
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

def main():
    st.title("MUJ Quiz: Get all your answers here!")
    st.markdown("Enter your question or upload an image to get answers from different models.")
    
    # User input for text prompt
    prompt = st.text_input("Enter your prompt:")
    
    # Image upload field
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    # Set max_tokens to a fixed value
    max_tokens = 5000
    
    if st.button("Submit"):
        if not prompt and not uploaded_file:
            st.warning("Please enter a prompt or upload an image!")
            return
        
        image_data = None
        if uploaded_file:
            # Encode the uploaded image to base64
            image_data = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
        
        models = ["qwen-2.5-32b", "deepseek-r1-distill-llama-70b", "gemma2-9b-it"]
        answers = {}
        
        with st.spinner("Fetching answers..."):
            for model in models:
                answer = call_groq_api(prompt, model, max_tokens, image_data)
                answers[model] = answer
        
        st.success("Answers fetched successfully!")
        st.markdown("### Answers")
        for model, answer in answers.items():
            with st.expander(f"Model: {model.upper()}"):
                st.write(answer)

if __name__ == "__main__":
    main()
