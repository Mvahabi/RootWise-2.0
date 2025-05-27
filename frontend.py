import gradio as gr
from logic import (
    set_user_name,
    append_to_user_rag,
    handle_image_upload,
    stream_response,
    load_documents,
    add_to_rag,
    list_system_data_files,
    read_selected_file,
    show_pdf,
    hide_pdf
)

with gr.Blocks(css="""
.gradio-container {background-color: #8A9A5B;}
h1 {text-align: center; font-family: 'Georgia', cursive, sans-serif;}
.section {padding: 1rem; margin: 1rem; border-radius: 12px; background: rgba(255, 255, 255, 0.85);}
""") as demo:

    with gr.Row():
        gr.Image(type="filepath", value="./frontpage.png", visible=True)

    with gr.Row():
        with gr.Column(elem_classes="section"):
            gr.Markdown("## Idea Journal")
            gr.Markdown("Heard something tasty? Thought of a clever zero-waste trick? **This is your personal notepad.**")

            user_name = gr.Textbox(label="What's your name?", placeholder="e.g. Lily, FoodWizard42...")
            name_submit = gr.Button("Start My Idea Journal")

            user_entry = gr.Textbox(
                label="New thought, recipe, tip, or idea?",
                lines=4,
                placeholder="Write anything — e.g. 'carrot top pesto sounds good', 'swap rice for barley', 'ask grandma about fermentation'"
            )
            entry_submit = gr.Button("Save to My Notepad")

            name_submit.click(set_user_name, inputs=[user_name], outputs=[user_name])
            entry_submit.click(append_to_user_rag, inputs=[user_entry], outputs=[user_entry])

        with gr.Column(elem_classes="section"):
            gr.Markdown("## Upload & Detect Vegetables")
            veg_image = gr.File(label="Upload an Image", file_types=["image"])
            detect_button = gr.Button("Detect Vegetables")
            detected_output = gr.Textbox(interactive=False, show_label=False)
            detect_button.click(handle_image_upload, inputs=[veg_image], outputs=[detected_output])


    with gr.Column(elem_classes="section"):
        gr.Markdown("## Chat with the Assistant")
        chatbot = gr.Chatbot(type='messages')
        msg = gr.Textbox(label="Ask a Question", placeholder="give me recipe")
        clear = gr.Button("Clear Chat")
        msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot], queue=True)
        msg.submit(lambda: "", outputs=[msg])
        clear.click(lambda: None, None, chatbot, queue=False)

    with gr.Row():
        with gr.Column(elem_classes="section"):
            gr.Markdown("## Data Tools")

            gr.Markdown("### Load Custom Documents")
            file_upload = gr.File(label="Upload Documents", file_types=[".txt", ".pdf"])
            load_button = gr.Button("Load Documents")
            load_button.click(load_documents, inputs=[file_upload], outputs=gr.Textbox(label="Status"))

            gr.Markdown("### Add to Your RAG Dataset")
            ingredients_input = gr.Textbox(label="Ingredients (comma-separated)")
            add_ingredients_button = gr.Button("Submit Ingredients")
            add_ingredients_button.click(lambda s: add_to_rag("", s, ""), inputs=[ingredients_input], outputs=[ingredients_input])

            season_input = gr.Textbox(label="Season")
            add_season_button = gr.Button("Submit Season")
            add_season_button.click(lambda s: add_to_rag(s, "", ""), inputs=[season_input], outputs=[season_input])

            restrictions_input = gr.Textbox(label="Dietary Restrictions (comma-separated)")
            add_restrictions_button = gr.Button("Submit Restrictions")
            add_restrictions_button.click(lambda r: add_to_rag("", "", r), inputs=[restrictions_input], outputs=[restrictions_input])

            gr.Markdown("### System File Viewer")
            refresh_button = gr.Button("Refresh File List")
            file_list = gr.Dropdown(choices=[], label="Available Files")
            file_contents = gr.Textbox(label="File Preview", interactive=False)
            file_preview = gr.Image(label="PDF Snapshot", visible=False)

            def refresh_files():
                files = list_system_data_files()
                return gr.update(choices=files, value=None)

            refresh_button.click(refresh_files, outputs=[file_list])
            file_list.change(read_selected_file, inputs=[file_list], outputs=[file_contents, file_preview])

            gr.Markdown("### About This Project")
            open_pdf_button = gr.Button("Show About Page")
            pdf_viewer = gr.Image(type="filepath", value="./about_us.png", visible=False)
            close_pdf_button = gr.Button("Close", visible=False)
            open_pdf_button.click(show_pdf, outputs=[pdf_viewer, close_pdf_button], queue=False)
            close_pdf_button.click(hide_pdf, outputs=[pdf_viewer, close_pdf_button], queue=False)
