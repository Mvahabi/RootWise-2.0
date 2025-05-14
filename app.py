import os
import gradio as gr
import shutil
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, get_response_synthesizer
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.embeddings.nvidia import NVIDIAEmbedding
# from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from llama_index.core import Settings
from openai import OpenAI
import time
import subprocess
import uuid

# # Initialize global variables
query_engine = None


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

        # Pass it explicitly
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


def detect_vegetables(image_path):
    try:
        result = subprocess.run(
            ["python3", "vis-transformer.py", image_path],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.splitlines()
        vegs = []
        for line in lines:
            if line.startswith("Identified Vegetables:"):
                items = line.split(":", 1)[1].strip(" []\n")
                vegs = [v.strip("' ") for v in items.split(',') if v.strip()]
        add_to_rag(season='', ingredients=str(vegs), restrictions='')                    # could use some add_to_rag() improvement
        return vegs if vegs else ["No vegetables detected."]
    except subprocess.CalledProcessError as e:
        return [f"Error: {e.stderr.strip()}"]

def handle_image_upload(file_obj):
    if not hasattr(file_obj, "name"):
        return ["Invalid file"]
    os.makedirs("images", exist_ok=True)
    temp_path = f"images/{uuid.uuid4().hex}.jpg"
    shutil.copyfile(file_obj.name, temp_path)
    vegs = detect_vegetables(temp_path)
    os.remove(temp_path)  
    return vegs


def get_files_from_input(file_objs):
    if not file_objs:
        return []
    if isinstance(file_objs, str):
        return [file_objs]
    if isinstance(file_objs, list):
        return [f.name if hasattr(f, 'name') else f for f in file_objs]
    return []

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

            try:
                reader = SimpleDirectoryReader(
                    input_files=[dest_path],
                    file_extractor={".pdf": PDFReader()}
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
    

def add_to_rag(season, ingredients, restrictions):
    global rag_data, query_engine
    file_path = 'system_data/user_rag.txt'

    new_entry = {
        "season": season,
        "ingredients": ingredients.split(','),
        "restrictions": restrictions.split(',')
    }
    rag_data.append(new_entry)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as file:
        file.write(
        f"Season: {new_entry['season']}, Ingredients: {', '.join(new_entry['ingredients'])}, Dietary Restrictions: {', '.join(new_entry['restrictions'])}\n"
    )

    documents = []
    documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())

    if not documents:
        return f"No new data here." 

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
        return f"Error updating rag: {str(e)}"

def embed_query(text):
    return nvidia_embed_model.embed([text])[0]

def chat(message, history):
    global query_engine
    embedding = embed_query(message)

    if query_engine is None:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "I don't have any documents to reference, but I'll try my best."}
        ]

    try:
        response = query_engine.query(message)
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": str(response)}
        ]
    except Exception as e:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"An error occurred: {str(e)}"}
        ]

def stream_response(message, history):
    global query_engine

    if query_engine is None:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Please load documents first."}
        ]
        return

    try:
        response = query_engine.query(
            "You are a friendly and sustainability-minded farmers market assistant. "
            "Your job is to *prompt the user* to make environmentally conscious food choices by asking them helpful, specific questions."
            "\n\nYour overall goals are:\n"
            "- Recommend recipes based on seasonal ingredients, user preferences, and allergies\n"
            "- Offer food scrap tips: composting, freezing leftovers, zero-waste cooking (e.g., carrot top pesto)\n"
            "- Share food donation resources (e.g., Food Not Bombs drop-offs)\n"
            "- Give food storage tips (e.g., storing greens with paper towels)\n"
            "- Briefly mention food-as-medicine properties (e.g., turmeric for inflammation)\n\n"

            "RULES:\n"
            "- NEVER suggest recipes that include ingredients the user is allergic to\n"
            "- DO NOT overwhelm the user with information all at once\n"
            "- DO NOT answer unless you have asked the user a question first\n"
            "- Begin each turn by asking a short, kind, useful question to understand what the user has or needs\n"
            "- Use concise, personable bullet points when explaining things\n"
            "- Be warm, respectful, and never contradict the user\n"
            "- Always prioritize sustainability\n\n"

            "EXAMPLE FIRST PROMPTS:\n"
            "- 'What ingredients do you have on hand today?'\n"
            "- 'Are you cooking for anyone with dietary restrictions?'\n"
            "- 'Do you have any leftover veggies or scraps you'd like to use?'\n\n"

            "Now respond based on this user message:\n"
            f"{message}"
        )
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": str(response)}
        ]


    except Exception as e:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Error processing query: {str(e)}"}
        ]


def show_pdf():
    file_path = "./about_us.pdf"
    return gr.update(visible=True), gr.update(visible=True)

def hide_pdf():
    return gr.update(visible=False), gr.update(visible=False)

with gr.Blocks(css=".gradio-container {background-color: #8A9A5B;} h1 {text-align: center; font-family: 'Georgia', cursive, sans-serif;}") as demo:

    with gr.Row():
        gr.Image( type="filepath", value="./frontpage.png", visible=True)

    with gr.Row():
        with gr.Column():
            veg_image = gr.File(label="Upload Vegetable Photo", file_types=["image"])
            detect_button = gr.Button("Detect Vegetables")
            detected_output = gr.Textbox(label="Detected Vegetables")
            detect_button.click(handle_image_upload, inputs=[veg_image], outputs=[detected_output])

        with gr.Column():
            chatbot = gr.Chatbot(type='messages')
            msg = gr.Textbox(label="Enter your question", interactive=True)
            clear = gr.Button("Clear")
            
            msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot], queue=True)
            msg.submit(lambda: "", outputs=[msg])
            clear.click(lambda: None, None, chatbot, queue=False)

    with gr.Column():
        file_upload = gr.File(label="Upload Documents", file_types=[".txt", ".pdf"])
        load_button = gr.Button("Load Documents")
        load_button.click(load_documents, inputs=[file_upload], outputs=gr.Textbox(label="Status"))

    with gr.Column():
            season_input = gr.Textbox(label="Season")
            ingredients_input = gr.Textbox(label="Ingredients (comma-separated)")
            restrictions_input = gr.Textbox(label="Dietary Restrictions (comma-separated)")

            add_rag_button = gr.Button("Add to RAG Database")
            add_rag_button.click(add_to_rag, inputs=[season_input, ingredients_input, restrictions_input], outputs=gr.Textbox(label="RAG Status"))

    with gr.Column():
        open_pdf_button = gr.Button("About Us")
        pdf_viewer = gr.Image(label="PDF Viewer", type="filepath", value="./about_us.png", visible=False)
        close_pdf_button = gr.Button("Close PDF", visible=False)

        open_pdf_button.click(
            show_pdf,
            inputs=[],
            outputs=[pdf_viewer, close_pdf_button],
            queue=False
        )

        close_pdf_button.click(
            hide_pdf,
            inputs=[],
            outputs=[pdf_viewer, close_pdf_button],
            queue=False
        )

if __name__ == "__main__":
  result = initialize_rag('./system_data')
  print(f"RAG init result: {result}")

  demo.queue()
#   demo.launch(share=True)
  demo.launch(server_name="0.0.0.0", server_port=7860)
  print("herelo")