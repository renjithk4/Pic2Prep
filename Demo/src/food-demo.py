import streamlit as st
from PIL import Image
import json
from main import load_qa_pairs, get_response

# Load the JSON file
qa_pairs = load_qa_pairs('../assets/qa_pairs.json')

st.title("Pic2Prep")

# Inject custom CSS for button styling
st.markdown("""
    <style>
    .stButton button {
        background-color: #C8C2D0;
        color: black;
        font-size: 16px;
        border-radius: 8px;
        padding: 8px 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for messages if not already done
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar for image upload
st.sidebar.title("Food Image Upload")
image_file = st.sidebar.file_uploader("Upload a food image", type=['jpg', 'png'])

# Show the uploaded image on the sidebar if any
if image_file:
    image = Image.open(image_file)
    st.sidebar.image(image, caption="Uploaded Food Image", use_column_width=True)

# Add some default questions as buttons
default_questions = ["What are the ingredients present in this image?", "What are the cooking actions?", "Give me step-by-step instructions"]
selected_question = None

st.write("**Questions you can try:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button(default_questions[0]):
        selected_question = default_questions[0]
with col2:
    if st.button(default_questions[1]):
        selected_question = default_questions[1]
with col3:
    if st.button(default_questions[2]):
        selected_question = default_questions[2]

# User input for the chatbot
user_question = st.chat_input("How may I assist you?")

# Handle default question button click
if selected_question:
    user_question = selected_question

# Handle user input
if user_question:
    # Append user question to the message history
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Get response based on the question
    response = get_response(user_question, qa_pairs)
    
    # Append the assistant's response to the message history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear the input field after submission
    user_question = ""  # This line clears the input field

# Display chat messages after updating
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Message when no image is uploaded
if not image_file:
    st.sidebar.write("Hey! I’m Pic2Prep—your interactive food guide! Snap a pic of your dish, and I’ll break down the ingredients, recipe, and cooking steps. Let’s turn your food photos into tasty answers!")
