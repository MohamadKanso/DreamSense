# DreamSense

DreamSense is a free AI web app for creative dream interpretation.

- Type your dream
- Answer a few short questions
- Get a poetic or prose analysis
- Optionally, generate an image or listen to your result
- Download your poem, image, or audio

## How to use

1. Install requirements:  
   `pip install -r requirements.txt`

2. Add your Hugging Face token to a `.env` file:  
   `HF_TOKEN=your_hf_token_here`

3. Run the app:  
   `streamlit run streamlit_app.py`

## Features

- Dream analysis with AI (Zephyr-7B)
- Follow-up questions for clarity
- Poetic interpretation
- Optional image and audio output
- Download buttons for everything

## Hosting

Works on Streamlit Cloud.  
Set your Hugging Face token as a secret (`HF_TOKEN`).

## Notes

- Some requests may take 10–30 seconds.
- Melody feature is coming soon.
- Please use UK English for comments and docs.

---