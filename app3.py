import streamlit as st
import pandas as pd
from PIL import Image
from bertopic import BERTopic
import plotly.express as px
import pickle
import plotly.graph_objects as go
import numpy as np
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from top2vec import Top2Vec

st.set_page_config(layout="wide")

# Functions to load the saved models on CPU
@st.cache_resource
def load_model(model_choice):
    #return model depending on user choice
    if model_choice == "BERTopic":
        return BERTopic.load("my_model")
    elif model_choice == "Top2Vec":
        return Top2Vec.load("model_top2vec")

#Function for loading embeddings, for BERTopic
embeddings1 = np.load("embeddings.npy")

#Function for loading the data. Reduced data for BERTopic (cleaned_posts1)
#and complete data for Top2Vec (cleaned_submissions)
@st.cache_data
def load_data(model_choice):
    if model_choice == "BERTopic":
        return pd.read_csv("cleaned_posts1.csv")
    elif model_choice == "Top2Vec":
        return pd.read_csv("cleaned_submissions.csv")

##Functions for creating word clouds
def create_wordcloud(model_choice, model1, topic_num):
    if model_choice == "BERTopic":
        words = model1.get_topic(topic_num)
        word_freq = {word: freq for word, freq in words}
        wc = WordCloud(background_color="white", max_words=1000).generate_from_frequencies(word_freq)
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(5, 8))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')  # Remove axis
        return fig
    if model_choice == "Top2Vec":
        fig = model1.generate_topic_wordcloud(topic_num=topic_num, 
                                        background_color='white', reduced=False)
        return fig


# Apply custom CSS for aesthetics and icons
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
    body {background-color: #f5f7fa;}
    .sidebar .sidebar-content {background-color: #2c3e50;}
    h1, h2, h3, h4, h5, h6 {color: #34495e;}
    .stButton>button {background-color: #3498db; color: white; border-radius: 5px;}
    .stButton>button:hover {background-color: #2980b9;}
    .stSelectbox, .stSlider {background-color: #ecf0f1; border-radius: 5px;}
    .topic-section {border: 1px solid #3498db; border-radius: 5px; margin-bottom: 20px; padding: 10px; background-color: #ecf0f1;}
    .topic-header {color: #3498db; font-size: 20px; font-weight: bold; display: flex; align-items: center;}
    .top-words {font-style: italic; color: #34495e;}
    .explore-button {text-align: right;}
    .icon {margin-right: 5px; color: #3498db;}
    </style>
""", unsafe_allow_html=True)

# Sidebar with icons for navigation
st.sidebar.title("Life Pro Tips Explorer: Discover, Learn, Improve")
st.sidebar.markdown("""
    <i class="fas fa-home icon"></i> Home  
    <i class="fas fa-lightbulb icon"></i> Explore Topics  
    <i class="fas fa-search icon"></i> Search Tips  
""", unsafe_allow_html=True)
option = st.sidebar.selectbox("Choose an option", ["Home", "Explore Topics", "Search Tips"])

# Home Page
if option == "Home":
    st.title("**Life Pro Tips** 📝")
    st.markdown("<h3 style='color: #3498db;'> Master the art of living, one tip at a time 🎯</h3>", unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([2, 1])  # Adjust column widths as needed

    # Left column: Tip of the Day
    with col1:
        st.image("bulb1.png")

    # Right column: Image
    with col2:
        st.markdown("<h3 style='color: #3498db;'><i class='fas fa-lightbulb icon'></i> Tip of the Day</h3>", unsafe_allow_html=True)
        st.write("💡 *Placeholder for a random tip from the dataset*")
        


# Explore Topics
elif option == "Explore Topics":
    st.header("Explore Topics 📊")
    
    # Sidebar options for model and topics
    model_choice = st.sidebar.selectbox("Choose a Model", ["BERTopic", "Top2Vec"])
    num_topics = st.sidebar.slider("Number of Topics", 1, 20, 10)
    
    # Use session state to preserve the "view topics" mode
    if "view_topics" not in st.session_state:
        st.session_state["view_topics"] = False
    if st.sidebar.button("View Topics 🔍"):
        st.session_state["view_topics"] = True
    
    # LOAD MODEL AND DATA
    model = load_model(model_choice)
    df = load_data(model_choice)

    #VISUALIZE MODEL AND INTERTOPIC DISTANCE MAP VISUALIZATION (BERTOPIC)
    if model_choice == "BERTopic":
        posts = df['cleaned_posts'].tolist()
        model.update_topics(posts, top_n_words=100)
        #timestamps for dynmaic topic modeling
        df['created'] = pd.to_datetime(df['created'])
        dates = df['created']
        timestamps = dates.tolist()
        # topics over time
        topics_over_time = model.topics_over_time(posts, timestamps, nr_bins=24)

        # Render topics visualization (documents and intertopic map) regardless
        st.markdown("<h3 style='color: #3498db;'><i class='fas fa-chart-pie icon'></i> Documents & Topic Visualization</h3>", unsafe_allow_html=True)
        fig2 = model.visualize_documents(posts, embeddings=embeddings1, sample=0.1)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<h3 style='color: #3498db;'><i class='fas fa-chart-pie icon'></i> Intertopic Distance Map Visualization</h3>", unsafe_allow_html=True)
        fig = model.visualize_topics()
        st.plotly_chart(fig, use_container_width=True)

    
    # Only display the topics details if View Topics mode is active
    if st.session_state.get("view_topics"):
        st.markdown("<h3 style='color: #3498db;'><i class='fas fa-list-alt icon'></i> Topic Details</h3>", unsafe_allow_html=True)
        
        if model_choice == "BERTopic":
            for topic_num in range(num_topics):
                # Render topic summary (placeholder)
                topic_words = model.get_topic(topic_num)
                words = ", ".join([word for word, _ in topic_words[:11]])
                st.markdown(f"""
                <div class="topic-section">
                    <h4 class="topic-header">
                        <i class="fas fa-hashtag icon"></i> TOPIC {topic_num + 1}
                    </h4>
                    <p class="top-words"><i class="fas fa-tags icon"></i> Top Words: {words}</p>
                </div>
                """, unsafe_allow_html=True)
            
                # Define unique keys for session state and button
                topic_toggle_key = f"explore_topic_{topic_num}"
                topic_button_key = f"explore_topic_button_{topic_num}"
            
                if topic_toggle_key not in st.session_state:
                    st.session_state[topic_toggle_key] = False
            
                # Button to toggle topic details
                if st.button(f"Explore Topic {topic_num + 1} 🚀", key=topic_button_key):
                    st.session_state[topic_toggle_key] = True
            
                # If the topic detail flag is set, render its detailed section in a full-width container with an expander.
                if st.session_state.get(topic_toggle_key):
                    with st.container():
                        with st.expander(f"Topic {topic_num + 1} Details", expanded=True):
                            # Dynamic topic modeling plot (full width)
                            fig3 = model.visualize_topics_over_time(topics_over_time, topics=[topic_num])
                            st.plotly_chart(fig3, use_container_width=True)

                            # Create two columns for the bar chart and semicircle plot
                            col1, col2 = st.columns(2)
                            with col1:
                                # WordCloud Placeholder (full width)
                                st.markdown("<h4 style='color: #3498db;'><i class='fas fa-cloud icon'></i> Word Cloud </h4>", unsafe_allow_html=True)
                                fig_wc = create_wordcloud(model_choice, model, topic_num)
                                st.pyplot(fig_wc)
                            with col2:
                                # Top 5 Documents Placeholder (full width)
                                st.markdown("<h4 style='color: #3498db;'><i class='fas fa-file-alt icon'></i> Top 5 Relevant Posts </h4>", unsafe_allow_html=True)
                                for i in range(1, 6):
                                    st.write(f"📄 **Post {i}**: Placeholder text for relevant document.")
    
                            # Create two columns for the bar chart and semicircle plot
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("<h4 style='color: #3498db;'><i class='fas fa-chart-bar icon'></i> Top 10 Words Contribution </h4>", unsafe_allow_html=True)
                                fig4 = model.visualize_barchart(n_words=10, topics=[topic_num])
                                st.plotly_chart(fig4, use_container_width=True)
                            with col2:
                                st.markdown("<h4 style='color: #3498db;'><i class='fas fa-chart-pie icon'></i> Sentiment Distribution </h4>", unsafe_allow_html=True)
                                st.image("https://via.placeholder.com/300x300?text=Sentiment+Plot+Placeholder", use_column_width=True)
        
        elif model_choice == "Top2Vec":

            #PREPARE FOR FUNCTIONALITY

            #GET WORDS OF TOPICS
            #The function get_topics get words for all 
            #of the topics on a nested array. 
            topic_words, word_scores, topic_nums = model.get_topics(num_topics)
            #Convert it to list of lists where each list will
            #be a topic, and each topic will have a list of their words
            topic_words_list = topic_words.tolist()

            #SET LABELS OF TOPICS
            labels = {
                0: "Hygiene and Cleaning",
                1: "Savings",
                2: "Telecommunications",
                3: "Eating Behaviour",
                4: "Reddit",
                5: "Sleep",
                6: "Traffic and Road Safety",
                7: "Restaurant and Food Ordering",
                8: "Hair care and shaving",
                9: "Asking and offering help",
                10: "Reading and writing",
                11: "Communication and Conversations",
                12: "Pets",
                13: "Customer service",
                14: "Money management",
                15: "Self-Improvement and Well-being",
                16: "Friendships and Social Relationships",
                17: "Debating and Argumenting",
                18: "Career and Education",
                19: "Email and Spam Management"}

            #ASSIGN DOCUMENTS TO TOPICS
            docs_ids = df["id"].to_list()
            # Classify documents by topic
            topic_numbers, _, _, _ = model.get_documents_topics(doc_ids=docs_ids)
            #Create column with topic numbers
            df["topic"] = topic_numbers

            #CHECK FREQUENCY THROUGH THE YEARS
            #Calculate number of posts per topic per year
            topics_df = df.groupby(['topic', 'year']).size().reset_index(name='topic_count')
            #Plot frequency
            def plot_top2vec_topic_frequency(topic):
                topic_data = df[df['topic'] == topic]
                plt.plot(topic_data['year'], topic_data['topic_count'], label=topic)
                plt.xlabel('Year')
                plt.ylabel('Frequency of Topic')
                plt.title('Frequency of "{}" Topic Over Time'.format(labels[topic]))
                plt.show()

            #Show words for each topic
            for index, topic in enumerate(topic_words_list):
                words = ", ".join([word for word in topic[:11]])
                st.markdown(f"""
                <div class="topic-section">
                    <h4 class="topic-header">
                        <i class="fas fa-hashtag icon"></i> TOPIC {index + 1}
                    </h4>
                    <p class="top-words"><i class="fas fa-tags icon"></i> Top Words: {words}</p>
                </div>
                """, unsafe_allow_html=True)

            # Define unique keys for session state and button
                topic_toggle_key = f"explore_topic_{index}"
                topic_button_key = f"explore_topic_button_{index}"
            
                if topic_toggle_key not in st.session_state:
                    st.session_state[topic_toggle_key] = False
            
                # Button to toggle topic details
                if st.button(f"Explore Topic {index + 1} 🚀", key=topic_button_key):
                    st.session_state[topic_toggle_key] = True

                # If the topic detail flag is set, render its detailed section in a full-width container with an expander.
                if st.session_state.get(topic_toggle_key):
                    with st.container():
                        with st.expander(f"Topic {index + 1} Details", expanded=True):
                            plot_top2vec_topic_frequency(index)
                            
                            
                            















# Search Tips
elif option == "Search Tips":
    st.header("Search Tips")
    query = st.text_input("Enter a keyword to search for tips:")