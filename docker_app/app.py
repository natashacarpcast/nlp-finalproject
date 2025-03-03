import streamlit as st
import pickle
import torch
import bertopic

# Check if CUDA is available and map the model to CPU
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
        
        # Check if model contains torch tensors and move them to CPU
        if isinstance(model, torch.nn.Module):
            model = model.to('cpu')
        return model

# Load the model explicitly on CPU
model = load_model('BERTopic_model_19_22.pkl')

# Use the model in your Streamlit app
st.title('BERTopic Model Demo')
st.write('Loaded BERTopic model:', model)