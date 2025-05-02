import os
import gradio as gr
import shutil
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, get_response_synthesizer
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.embeddings.nvidia import NVIDIAEmbedding
# from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from openai import OpenAI
import time

####

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
rag_store = './system_data'

def initialize_rag(file_path):
    global query_engine, rag_store

    if not os.path.exists(rag_store):
        return "Error: system_data directory not found."

    try:
        documents = []
        for fname in os.listdir(rag_store):
            if fname.endswith(".txt"):
                full_path = os.path.join(rag_store, fname)
                documents.extend(SimpleDirectoryReader(input_files=[full_path]).load_data())

        if not documents:
            return "Error: No .txt files found in system_data."

        # Create Faiss vector store
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(1536))

        # Pass it explicitly
        index = VectorStoreIndex.from_documents(
            documents,
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
        if not file_objs:
            return "No files selected."
        
        file_paths = get_files_from_input(file_objs)
        documents = []
        print(f" \n\n file paths: {file_paths} \n\n")
        for file_path in file_paths:
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())
            file_name = os.path.basename(file_path)
            destination_f = f"{os.path.dirname(rag_store)}/{rag_store}/{file_name}"
            if file_name.endswith(".txt"):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                shutil.copyfile(file_path, destination_f)

        if not documents:
            return f"No documents found in the selected files."

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
            "You are a helpful, friendly farmers market employee that is responding to the user: \n" + message +
            " \n Your eventual goal is to offer the user the following: \n" +
            " \n - **Recipes**: Cook time. Community notes and tips. \n" +
            " \n DO NOT RECOMMEND RECIPES THAT INVOLVE FOODS THAT THE USER IS ALLERGIC TO!! \n\n" +
            " \n - **Food Scrap Recommendations**: Composting tips. Freezing methods for future use (e.g., stir-fry mix, chopped fruit). Zero-waste cooking ideas (e.g., carrot top pesto).\n" +
            " \n But you have these additional features: \n" +
            " \n - Food donation resources, including Food Not Bombs drop-off locations. \n" +
            " \n Food lifespan and storage tips (e.g., using paper towels to extend the life of greens).\n" +
            " Food as Medicine insights: Health benefits (e.g., turmeric for inflammation). Spiritual properties (e.g., calming effects of certain herbs).\n" +
            " \n\n Be sure to remember to be brief, that the user is always right, and sustainability is extremely important."
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
            season_input = gr.Textbox(label="Season")
            ingredients_input = gr.Textbox(label="Ingredients (comma-separated)")
            restrictions_input = gr.Textbox(label="Dietary Restrictions (comma-separated)")

            add_rag_button = gr.Button("Add to RAG Database")
            add_rag_button.click(add_to_rag, inputs=[season_input, ingredients_input, restrictions_input], outputs=gr.Textbox(label="RAG Status"))

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
  initialize_rag('./system_data')
  demo.queue()
  demo.launch(share=True)
  print("herelo")