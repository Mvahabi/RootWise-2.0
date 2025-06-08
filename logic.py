import os
import gradio as gr
import shutil
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, get_response_synthesizer
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from llama_index.core import Settings
from openai import OpenAI
import time
import subprocess
import uuid
from pdf2image import convert_from_path
import requests

# Initialize global variables
query_engine = None
user_rag_last_modified = 0
user_rag_file = None  

global nvidia_embed_model 
nvidia_embed_model = NVIDIAEmbedding(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=os.getenv("NVIDIA_API_KEY")
    )

client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY"),
  base_url="https://integrate.api.nvidia.com/v1"
)

rag_data = []
rag_store = './system_data'  # redundant? (this is specified in main(), feels like should be one variable)

#
# Initialize rag database:
# - load all documents in ./system_data into the faiss db, catch errors
# - create the faiss vector store
# - initialize "sentence splitter" to handle large documents, apply this as a transformation for the vector store
#

def initialize_rag(file_path):
    global query_engine, rag_store

    if not os.path.exists(rag_store):
        return "Error: system_data directory not found."

    try:
        documents = []
        for fname in os.listdir(rag_store):
            full_path = os.path.join(rag_store, fname)

            if not (fname.endswith(".txt") or fname.endswith(".pdf")):
                print(f"Skipping unsupported file: {fname}")
                continue

            try:
                print(f"Loading: {full_path}")
                reader = SimpleDirectoryReader(
                    input_files=[full_path],
                    file_extractor={".pdf": PDFReader()}
                )
                docs = reader.load_data()
                documents.extend(docs)

            except Exception as file_err:
                print(f"Skipping {full_path} due to error: {file_err}")

        if not documents:
            return "Error: No .txt files found in system_data."

        # Create Faiss vector store
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(1536))

        splitter = SentenceSplitter(chunk_size=400, chunk_overlap=50)

        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[splitter],
            vector_store=vector_store,
            embed_model=nvidia_embed_model
        )
        query_engine = index.as_query_engine()

        return "Query engine initialized successfully."

    except FileNotFoundError as fnf_error:
        return f"FileNotFoundError: {str(fnf_error)}"

    except Exception as e:
        print(f"Failed to initialize query engine. Exception: {str(e)}")
        return f"Failed to initialize query engine. Exception: {str(e)}"

#
# Handling userRAG
# Three functions
#

def user_rag_updated(file_path):
    global user_rag_last_modified
    current_time = os.path.getmtime(file_path)
    if current_time != user_rag_last_modified:
        user_rag_last_modified = current_time
        return True
    return False


def user_rag(file_path):
    global query_engine
    if not os.path.exists(file_path):
        return
    if query_engine is None or user_rag_updated(file_path):
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(1536))
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            embed_model=nvidia_embed_model
        )
        query_engine = index.as_query_engine()

def set_user_name(name):
    global user_rag_file
    if not name:
        return "Please enter a valid name."
    os.makedirs(rag_store, exist_ok=True)
    user_rag_file = os.path.join(rag_store, f"{name}RAG.txt")
    if not os.path.exists(user_rag_file):
        with open(user_rag_file, 'w') as f:
            f.write(f"{name}'s RAG session initialized.\n")
    return f"File {name}RAG.txt ready."


def append_to_user_rag(entry):
    global user_rag_file
    if not user_rag_file:
        return "Please enter your name first."
    with open(user_rag_file, 'a') as f:
        f.write(f"{entry}\n")
    try:
        user_rag(user_rag_file)
        return "Entry added and index updated."
    except Exception as e:
        return f"Entry added, but update failed: {str(e)}"

#
# Plug in vision transformer:
# - using subprocess, run vis-transformer.py on the uploaded image (modular)
# - catch the output vegetable detections
# - call add_to_rag() so that the vegetables detected are added the the user's 'ingredients'
#

def detect_vegetables(image_path):
    try:
        result = subprocess.run(
            ["python3", "vis-transformer.py", image_path],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.splitlines()
        vegs = ""
        for line in lines:
            if line.startswith("Identified Vegetables:"):
                items = line.split(":", 1)[1].strip(" []\n")
                vegs = [v.strip("' ") for v in items.split(',') if v.strip()]

        if vegs:
            for veg in vegs:
                add_to_rag(season='', ingredients=veg.strip(" [']\n"), restrictions='')                    # could use some add_to_rag() improvement
        return vegs if vegs else "No vegetables detected."
    
    except subprocess.CalledProcessError as e:
        return [f"Error: {e.stderr.strip()}"]
    
#
# Handle image input for vegetable detection:
# - handles error inputs
# - calls detect_vegetables()
#

def handle_image_upload(file_obj):
    if not hasattr(file_obj, "name"):
        return ["Invalid file"]
    os.makedirs("images", exist_ok=True)
    temp_path = f"images/{uuid.uuid4().hex}.jpg"
    shutil.copyfile(file_obj.name, temp_path)
    vegs = detect_vegetables(temp_path)
    os.remove(temp_path)  

    output = ""
    for veg in vegs:
        output += f"{veg}, "
    return output

#
# Document uploading:
# - handles errors
# - adds uploaded files to the RAG database
#

from llama_index.readers.file import PDFReader

def load_documents(file_objs):
    global query_engine, rag_store

    try:
        # Normalize input to a list
        if not file_objs:
            return "No files selected."
        if not isinstance(file_objs, list):
            file_objs = [file_objs]

        documents = []

        for file_obj in file_objs:
            if not hasattr(file_obj, "name"):
                return f"Uploaded object has no name: {file_obj}"

            file_name = os.path.basename(file_obj.name)

            if file_name == "/" or file_name.strip() == "":
                return f"Invalid file name received: {file_name}"

            if not (file_name.endswith(".txt") or file_name.endswith(".pdf")):
                print(f"Skipping unsupported file: {file_name}")
                continue

            dest_path = os.path.join(rag_store, file_name)
            os.makedirs(rag_store, exist_ok=True)

            shutil.copyfile(file_obj.name, dest_path)
            print(f"Copied file to: {dest_path}")
            print(f"Processing file: {file_name} (ext: {os.path.splitext(file_name)[1]})")

            try:
                reader = SimpleDirectoryReader(
                    input_files=[dest_path],
                    file_extractor={
                        ".pdf": PDFReader(),
                        ".txt": None
                    }
                )
                docs = reader.load_data()
                print(f"Loaded {len(docs)} documents from {file_name}")
                documents.extend(docs)

            except Exception as e:
                print(f"Error loading {file_name}: {e}")

        if not documents:
            return "No valid documents were uploaded."

        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(1536))
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            embed_model=nvidia_embed_model
        )
        query_engine = index.as_query_engine()

        return "Documents loaded successfully!"

    except Exception as e:
        return f"Error loading documents: {str(e)}"

#
# Handles adding ingredients, season information, and dietary restriction inputs
# - defines the file path and makes a file in the system_data directory if ther isn't one already
# - writes entries out to the file
# - links it to the faiss vectorstore db
#

def add_to_rag(season, ingredients, restrictions):
    global rag_data, query_engine

    base_dir = 'system_data'
    os.makedirs(base_dir, exist_ok=True)

    season_path = os.path.join(base_dir, 'given_season.txt')
    ingredients_path = os.path.join(base_dir, 'given_ingredients.txt')
    restrictions_path = os.path.join(base_dir, 'given_restrictions.txt')

    for path in [season_path, ingredients_path, restrictions_path]:
        open(path, 'a').close()

    # Write entries to separate files
    if season:
        with open(season_path, 'w') as f:
            f.write(f"Season: {season}\n")

    if ingredients:
        with open(ingredients_path, 'a') as f:
            f.write(f"Ingredients: {ingredients}\n")

    if restrictions:
        with open(restrictions_path, 'a') as f:
            f.write(f"Dietary Restrictions: {restrictions}\n")

    # Load documents from all three files
    documents = []
    for path in [season_path, ingredients_path, restrictions_path]:
        documents.extend(SimpleDirectoryReader(input_files=[path]).load_data())

    if not documents:
        return "No new data here."

    try:
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(1536))
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=vector_store,
            embed_model=nvidia_embed_model
        )
        query_engine = index.as_query_engine()

        return "Input added to RAG database!"
    except Exception as e:
        return f"Error updating RAG: {str(e)}"

#
# Handles conversation with the LLM
# - unpacks and references specific files in system_data
# - 
#

def call_nvidia_chat(messages, model="meta/llama3-70b-instruct"):
    url = f"https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('NGC_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} - {response.json()}")
    return response.json()["choices"][0]["message"]["content"].strip()


def stream_response(message, history):
    global query_engine

    if query_engine is None:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please load documents first."}
        ]
        return

    try:
        rag_path = f"./system_data/{user_rag_file}.txt"
        if os.path.exists(rag_path):
            with open(rag_path, "r") as f:
                rag_contents = f.read().strip()
        else:
            rag_contents = ""

        rag_excerpt = ' '.join(rag_contents.split()[:120])

        ingredient_path = f"./system_data/given_ingredients.txt"
        if os.path.exists(ingredient_path):
            with open(ingredient_path, "r") as f:
                ingredients = f.read().strip()
        else:
            ingredients = ""

        season_path = f"./system_data/given_season.txt"
        if os.path.exists(season_path):
            with open(season_path, "r") as f:
                season = f.read().strip()
        else:
            season = ""

        allergy_path = f"./system_data/given_restrictions.txt"
        if os.path.exists(allergy_path):
            with open(allergy_path, "r") as f:
                allergies = f.read().strip()
        else:
            allergies = ""

        prompt = (
            "You are a warm, sustainability-minded farmers market assistant who supports eco-conscious food choices with kindness, patience, and wisdom.\n\n"
            "**IMPORTANT:** If the user message is a simple greeting (e.g., 'hi', 'hello', 'hey'), just respond with a short, friendly reply like 'Hi there!' and wait for them to continue. Do not offer suggestions or ask follow-up questions yet.\n\n"
            "If the user gives more context, then:\n"
            "- Offer helpful ideas from ./system_data and their knowledge base (prioritize this excerpt):\n"
            f"{rag_excerpt}\n\n"
            f"- Focus on sustainable cooking, seasonal produce (current season: {season}), zero-waste practices, and food-as-medicine wisdom\n"
            "- Share only 2–3 relevant suggestions at a time, using clear bullet points if needed\n"
            "- Ask at most one kind, helpful follow-up question — only if the user provides enough info\n\n"
            f"NEVER suggest recipes with ingredients the user is allergic to: {allergies}\n"
            f"PRIORITIZE ingredients the user has: {ingredients}\n"
            "DO NOT overwhelm the user. Be clear, kind, and wait for more info if needed.\n"
            "Your main responsibilities are:\n"
            "- Gently offer information from ./system_data and the user's knowledge base to share practical and heartfelt wisdom\n"
            "- Recommend ideas based on sustainable cooking, seasonal produce (current season: {season}), food preservation, and zero-waste practices\n"
            "- Explain how actions like reducing food waste or preserving herbs can save money, support health, or build community\n"
            "- Prompt the user to share details through soft, specific, *single* questions — only when necessary, and never in a rushed way\n\n"
        )

        # Only include the latest user/assistant message for context
        truncated_history = ""
        if history:
            last_messages = history[-2:]  # Only last exchange
            for m in last_messages:
                truncated_history += f"{m['role'].capitalize()}: {m['content'][:300]}\n"  # Clip each to 300 chars max

        rag_retrieval = query_engine.query(message)

        # Construct final prompt
        full_prompt = (
            prompt
            + f"Here is relevant information from the system_data documents:\n{rag_retrieval}\n\n"
            + "\nRecent context:\n"
            + truncated_history
            + f"User: {message[:300]}\n"
            + "Now continue the conversation in character."
        )
        
        response = call_nvidia_chat([
            {"role": "system", "content": prompt},
            {"role": "user", "content": full_prompt}
        ])

        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": str(response)}
        ]

    except Exception as e:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Error processing query: {str(e)}"}
        ]

#
# This function only checks if the files in the system are the right type
# depreciated I think?
# #

def list_system_data_files():
    try:
        files = os.listdir(rag_store)
        return [f for f in files if f.endswith((".txt", ".pdf"))]
    except Exception as e:
        return [f"Error: {e}"]

#
# 
#

def read_selected_file(filename):
    if not filename:
        return gr.update(value="No file selected."), gr.update(visible=False)
    full_path = os.path.join(rag_store, filename)
    if not os.path.exists(full_path):
        return gr.update(value="File not found."), gr.update(visible=False)

    if filename.endswith(".txt"):
        with open(full_path, "r") as f:
            return gr.update(value=f.read()), gr.update(visible=False)

    elif filename.endswith(".pdf"):
        try:
            images = convert_from_path(full_path, dpi=100)
            os.makedirs("temp_renders", exist_ok=True)
            image_paths = []

            for i, img in enumerate(images):
                img_path = f"temp_renders/{uuid.uuid4().hex}_page_{i}.png"
                img.save(img_path, "PNG")
                image_paths.append(img_path)

            # Return placeholder text and show the first image (could be made into carousel later)
            return gr.update(value="PDF rendered below:"), gr.update(value=image_paths[0], visible=True)

        except Exception as e:
            return gr.update(value=f"Error rendering PDF: {str(e)}"), gr.update(visible=False)

    else:
        return gr.update(value="Unsupported file type."), gr.update(visible=False)

#
# This just shows the "about us" page
#   

def show_pdf():
    file_path = "./about_us.pdf"
    return gr.update(visible=True), gr.update(visible=True)

#
# this puts away the about us page
#

def hide_pdf():
    return gr.update(visible=False), gr.update(visible=False)