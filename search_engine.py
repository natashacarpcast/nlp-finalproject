###version 3

# run docker start recursing_mclean 
import streamlit as st
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import openai
import numpy as np
import time
openai_key = "xx" # enter your openai key here

# Set Streamlit to wide mode
st.set_page_config(layout="wide")

# Sidebar Configuration
st.sidebar.header("Configuration")

# Server URL Selection
server_options = [
    "http://localhost:6333",
    "http://127.0.0.1:6333",
    "http://192.168.1.100:6333",
]
server_url = st.sidebar.selectbox("Select Qdrant Server URL", server_options, index=0)
qdrant_host = server_url

# Collection Name Input
collection_name = st.sidebar.text_input("Collection Name", "life_pro_tips")

# Number of Results Slider
results_limit = st.sidebar.number_input("Number of Results", min_value=5, max_value=100, value=10, step=5)

# st.markdown(
#     """
#     <style>
#     .stRadio, .stTextInput, .stButton { font-size: 26px !important; font-weight: bold; }
#     .stDataFrame { font-size: 20px !important; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


st.title("Life Pro Tips Search", anchor=False)
# Search Type Selection
search_type = st.radio("Search Type", ["Keyword Search (BM25)", "Semantic Search"])

# Model selection (only for Semantic Search)
model_options = {
    "all-MiniLM-L6-v2": "Sentence-Transformer: all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2": "Sentence-Transformer: all-MiniLM-L12-v2",
    "all-mpnet-base-v2": "Sentence-Transformer: all-mpnet-base-v2",
}
selected_model = st.sidebar.selectbox("Select Embedding Model(Semantic Search Only)", list(model_options.keys()), disabled=(search_type == "Keyword Search (BM25)"))

# Load Model (for Semantic Search)
@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

model = load_embedding_model(selected_model) if search_type == "Semantic Search" else None

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_submissions.csv")
    df["year"] = pd.to_datetime(df["created"]).dt.year
    df["month"] = pd.to_datetime(df["created"]).dt.month
    df["date"] = df["year"].astype(str) + "/" + df["month"].astype(str).str.zfill(2)
    return df

data = load_data()

# OpenAI API Key
openai.api_key = openai_key  #  API key

def keyword_search(query, top_k):
    """Perform keyword-based search using BM25"""
    query = query.strip().lower()  
    corpus = data["cleaned_title"].astype(str).str.lower().tolist()
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    
    data["bm25_score"] = scores
    results = data.nlargest(top_k, "bm25_score")[["author", "cleaned_title", "date"]]
    results = results.rename(columns={"cleaned_title": "tips"})

    # if results["tips"].str.startswith('ltp').any():
    #     #remove ltp from the content
    #     results["tips"] = results["tips"].str.replace("ltp", "")
    results["tips"] = results["tips"].str.replace(r'^\s*lpt\s+', '', regex=True)
    return results

def semantic_search(query, top_k):
    """Perform semantic search using Qdrant"""
    query_vector = model.encode(query).tolist()
    
    qdrant_client = QdrantClient(qdrant_host)
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    df_results = pd.DataFrame([{
        "author": res.payload["author"],
        "tips": res.payload["cleaned_title"],
        # "date": f"{res.payload['year']}/{str(res.payload['month']).zfill(2)}"
        "date": res.payload["date"]
    } for res in results])

    # if df_results["tips"].str.startswith('ltp').any():
    #     #remove ltp from the content
    #     df_results["tips"] = df_results["tips"].str.replace("ltp", "")

    df_results["tips"] = df_results["tips"].str.replace(r'^\s*lpt\s+', '', regex=True)
    return df_results

def summarize_results(results):
    """Summarizes search results using OpenAI"""
    if results.empty:
        return "No relevant tips found."
    
    combined_text = " ".join(results["tips"].tolist())

    try:
        messages = [   
        {
            "role": "system",
            "content": (
                "You are a web app developer creating a search engine for life-pro tips. "
                'The tips come from a subreddit called "LifeProTips," and your tone should be friendly, engaging, and conversational. '
                "Format the response as if you're chatting with a friend, making it easy to digest. "
                "Your task is to summarize the given tips into a **single, well-structured paragraph**, so users can quickly grasp the key insights. "
                "Always Use relevant and friendly emojis at the **beginning of sentences** to make it engaging, but keep them minimal for readability. "
            ),
        },
        {
            "role": "user",
            "content": (
                f"Summarize the following tips into **one cohesive paragraph in only 2-3 sentences**, avoiding disjointed sentences. "
                'Start with: **When talking about [search query], most Redditors suggest that:..** and ensure the summary flows naturally. '
                "Ensure the summary flows naturally, connecting ideas smoothly ( 'Most Redditors suggest... Others recommend... Additionally...'). "
                f"Here are the tips: {combined_text}"
            ),  
        }]

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        summary = response.choices[0].message.content
        summary = summary.replace("suggest that", "suggest that\n")
        return summary
    except Exception as e:
        return f"Failed to generate summary: {e}"
    
    

# Search Input
query = st.text_input("Enter search query to look for life-pro tips:")

if st.button("Search"):
    if query:
        start_time = time.time()
        
        if search_type == "Keyword Search (BM25)":
            results = keyword_search(query, results_limit)
        else:
            results = semantic_search(query, results_limit)

        execution_time = time.time() - start_time
        st.info(f"Search completed in {execution_time:.2f} seconds")

        if not results.empty:

            st.write("Here are the relevant tips to your query:")
            st.dataframe(
                results.style.hide(axis='index').set_properties(**{'text-align': 'left'}),
                use_container_width=True,
                hide_index=True  
            )
            st.subheader("Summary of Key Tips 💡")
            st.write(summarize_results(results))
        else:
            st.write("No results found.")
    else:
        st.warning("Please enter a search term.")








###version 2
# import streamlit as st
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient, models
# import openai
# import numpy as np
# import time

# # Set Streamlit to wide mode
# st.set_page_config(layout="wide")

# # Sidebar Configuration
# st.sidebar.header("Configuration")

# # Server URL Selection
# server_options = [
#     "http://localhost:6333",
#     "http://127.0.0.1:6333",
# ]
# server_url = st.sidebar.selectbox("Select Qdrant Server URL", server_options, index=0)
# qdrant_host = server_url

# # Collection Name Input
# collection_name = st.sidebar.text_input("Collection Name", "life_pro_tips")

# # Number of Results Slider
# results_limit = st.sidebar.number_input("Number of results", min_value=5, max_value=100, value=10, step=5)

# # Search Type Selection
# search_type = st.radio("Search Type", ["Keyword Search", "Semantic Search"])

# # Model selection (only for Semantic Search)
# model_options = {
#     "all-MiniLM-L6-v2": "Sentence-Transformer: all-MiniLM-L6-v2",
#     "all-mpnet-base-v2": "Sentence-Transformer: all-mpnet-base-v2",
# }
# selected_model = st.sidebar.selectbox("Select Embedding Model", list(model_options.keys()), disabled=(search_type == "Keyword Search"))

# # Load Model
# @st.cache_resource
# def load_embedding_model(model_name):
#     return SentenceTransformer(model_name)

# model = load_embedding_model(selected_model) if search_type == "Semantic Search" else None

# # Load Data
# @st.cache_data
# def load_data():
#     return pd.read_csv("data/cleaned_submissions.csv")

# data = load_data()

# # OpenAI API Key
# openai.api_key = "your-openai-api-key"  # Replace with actual API key

# def keyword_search(query, top_k):
#     """Perform keyword-based search"""
#     mask = data['title+bodytext'].str.contains(query, case=False, na=False)
#     results = data[mask].sort_values('score', ascending=False).head(top_k)
#     return results

# def semantic_search(query, top_k):
#     """Perform semantic search using Qdrant"""
#     query_vector = model.encode(query).tolist()
    
#     qdrant_client = QdrantClient(qdrant_host)
#     results = qdrant_client.search(
#         collection_name=collection_name,
#         query_vector=query_vector,
#         limit=top_k
#     )

#     return pd.DataFrame([{
#         "Title": res.payload["title"],
#         "Text": res.payload["bodytext"],
#         "Score": res.payload["score"]
#     } for res in results])

# def summarize_results(results):
#     """Summarizes search results using OpenAI"""
#     combined_text = " ".join(results["Text"].tolist())

#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": "Summarize these life-pro tips into 2-3 sentences."},
#             {"role": "user", "content": combined_text}
#         ]
#     )
#     return response["choices"][0]["message"]["content"]

# # Search Input
# query = st.text_input("Enter search query:")

# if st.button("Search"):
#     if query:
#         start_time = time.time()
        
#         results = keyword_search(query, results_limit) if search_type == "Keyword Search" else semantic_search(query, results_limit)
        
#         execution_time = time.time() - start_time
#         st.info(f"Search completed in {execution_time:.2f} seconds")

#         if not results.empty:
#             st.dataframe(results, use_container_width=True)
#             st.subheader("Summary of Key Tips")
#             st.write(summarize_results(results))
#         else:
#             st.write("No results found.")
#     else:
#         st.warning("Please enter a search term.")





###version 1
# import streamlit as st
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient, models
# import numpy as np
# import openai

# # Load data
# data = pd.read_csv("data/cleaned_submissions.csv")

# # Initialize sentence transformer
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Initialize Qdrant client
# qdrant_client = QdrantClient("http://localhost:6333")
# collection_name = "life_pro_tips"

# def keyword_search(query, top_k=10):
#     """Searches for tips using simple keyword matching"""
#     mask = data['title+bodytext'].str.contains(query, case=False, na=False)
#     results = data[mask].sort_values('score', ascending=False).head(top_k)
#     return results

# def semantic_search(query, top_k=10):
#     """Searches for tips using semantic search in Qdrant"""
#     query_vector = model.encode(query).tolist()
    
#     results = qdrant_client.search(
#         collection_name=collection_name,
#         query_vector=query_vector,
#         limit=top_k
#     )

#     # Extract result details
#     result_data = []
#     for result in results:
#         result_data.append({
#             "title": result.payload["title"],
#             "body": result.payload["body"],
#             "score": result.payload["score"]
#         })
    
#     return pd.DataFrame(result_data)

# def summarize_results(results):
#     """Uses OpenAI API to summarize results"""
#     
#     combined_text = " ".join(results["body"].tolist())

#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": "Summarize the following tips into 2-3 sentences:"},
#             {"role": "user", "content": combined_text}
#         ]
#     )

#     return response["choices"][0]["message"]["content"]

# def search_page():
#     st.title("Search Life Pro Tips")

#     search_type = st.radio("Search Type", ["Keyword Search", "Semantic Search"])
#     query = st.text_input("Enter your search query")

#     if query:
#         if search_type == "Keyword Search":
#             results = keyword_search(query)
#         else:
#             results = semantic_search(query)

#         st.subheader(f"Found {len(results)} matching tips")
        
#         # Display results
#         for _, row in results.iterrows():
#             st.markdown(f"### {row['title']}")
#             st.markdown(f"**Score:** {row['score']}")
#             st.markdown(f"*{row['body']}*")
#             st.markdown("---")
        
#         # Summarize results
#         summary = summarize_results(results)
#         st.subheader("Summary of Key Tips")
#         st.write(summary)

# if __name__ == "__main__":
#     search_page()
