import os
import streamlit as st
from PIL import Image
from transformers import pipeline
from PyPDF2 import PdfReader
from st_on_hover_tabs import on_hover_tabs

os.environ['CURL_CA_BUNDLE'] = ''

st.set_page_config(
    page_title="DocuSense",
    layout="centered",
    page_icon="Resources/icon.jpg",
    initial_sidebar_state="auto",
)

logo_image = Image.open('Resources/logo.jpg')
bot_image = Image.open('Resources/bot.jpg')

st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home','About','Contact Me'],iconName=['home','info','contact_support'], default_choice=0) 

if tabs == 'Home':
	col1, col2 = st.columns(2)
	with col1:
    		st.image(logo_image, use_column_width=True)
	with col2:
		st.image(bot_image, use_column_width=True)

	if "messages" not in st.session_state:
    		st.session_state.messages = []

	for message in st.session_state.messages:
    		with st.chat_message(message["role"]):
        		st.markdown(message["content"])

	@st.cache_data(show_spinner=True)
	def instantiate_pipe():
    		model_name = "timpal0l/mdeberta-v3-base-squad2"
    		print("Loading model: ", model_name)
    		pipe = pipeline('question-answering', model=model_name, tokenizer=model_name)
    		return pipe

	pipe = instantiate_pipe()
	files_dir = 'Documents'
	files = os.listdir(files_dir)

	context = ""
	for file_name in files:
	    uploaded_file = open(os.path.join(files_dir, file_name), 'rb')

	    if file_name.split(".")[-1].lower() == "pdf":
	        reader = PdfReader(uploaded_file)
	        for page in reader.pages:
	            context += page.extract_text().strip() + "\n"
	    elif file_name.split(".")[-1].lower() == "txt":
	        for line in uploaded_file:
	            context += line.decode("utf-8").strip() + "\n"
	    context += "\n"

	if text_input := st.chat_input("What is up?"):
	    st.session_state.messages.append({"role": "user", "content": text_input, "avatar":"🧑🏻"})
	    with st.chat_message(name="user", avatar="🧑🏻"):
	        st.markdown(text_input)

	if text_input:
	    with st.spinner(f"Working..."):
	        qa_input = {
	            'question': text_input,
	            'context': context
	        }
	        pipe_response = pipe(qa_input)
	    with st.chat_message("assistant", avatar=bot_image):
	        response = st.write(pipe_response['answer'])
	    st.session_state.messages.append({"role": "assistant", "content": pipe_response['answer']})

	st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)

elif tabs == 'About':
    st.markdown('''
    This app is an GenAI powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [🤗 Hugging Face](https://huggingface.co/timpal0l/mdeberta-v3-base-squad2) mdeberta-v3-base-squad2 model

    ''')
    st.write('Made with ❤️  by Prashant')

elif tabs == 'Contact Me':
    st.title('Contact Me')

