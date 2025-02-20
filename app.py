import streamlit as st
import requests
import base64

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
            image_data = f"data:image/jpeg;base64,{image_data}"
        
        # Define models for text-based queries
        text_models = ["qwen-2.5-32b", "deepseek-r1-distill-llama-70b", "gemma2-9b-it"]
        answers = {}
        
        with st.spinner("Fetching answers..."):
            if image_data:
                # Use llama-3.2-90b-vision-preview for image analysis
                answer = call_groq_api(prompt, "llama-3.2-90b-vision-preview", max_tokens, image_data)
                answers["llama-3.2-90b-vision-preview"] = answer
            else:
                # Use other models for text-based queries
                for model in text_models:
                    answer = call_groq_api(prompt, model, max_tokens)
                    answers[model] = answer
        
        st.success("Answers fetched successfully!")
        st.markdown("### Answers")
        for model, answer in answers.items():
            with st.expander(f"Model: {model.upper()}"):
                st.write(answer)

if __name__ == "__main__":
    main()
