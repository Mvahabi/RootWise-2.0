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

with gr.Blocks(css=".gradio-container {background-color: #8A9A5B;} h1 {text-align: center; font-family: 'Georgia', cursive, sans-serif;}") as demo:

    with gr.Row():
        gr.Image(type="filepath", value="./frontpage.png", visible=True)

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "Heard something tasty? Thought of a clever zero-waste trick? "
                "**This is your personal notepad.** Type whatever you'd like to remember or explore later!"
            )

            user_name = gr.Textbox(label="What's your name?", placeholder="e.g. Lily, FoodWizard42...")
            name_submit = gr.Button("Start My Idea Journal")
            name_status = gr.Textbox(interactive=False, show_label=False)

            user_entry = gr.Textbox(
                label="New thought, recipe, tip, or idea?",
                lines=4,
                placeholder="Write anything — e.g. 'carrot top pesto sounds good', 'swap rice for barley', 'ask grandma about fermentation'"
            )
            entry_submit = gr.Button("Save to My Notepad")
            entry_status = gr.Textbox(interactive=False, show_label=False)

            name_submit.click(set_user_name, inputs=[user_name], outputs=[name_status])
            entry_submit.click(append_to_user_rag, inputs=[user_entry], outputs=[entry_status])

        with gr.Column():
            gr.Markdown("### Step 3: Upload and Detect Vegetables")
            veg_image = gr.File(label="Vegetable Image", file_types=["image"])
            detect_button = gr.Button("Detect")
            detected_output = gr.Textbox(label="", interactive=False, show_label=False)
            detect_button.click(handle_image_upload, inputs=[veg_image], outputs=[detected_output])

        with gr.Column():
            gr.Markdown("### Step 4: Chat with the Assistant")
            chatbot = gr.Chatbot(type='messages')
            msg = gr.Textbox(label="Ask a Question")
            clear = gr.Button("Clear Chat")

            msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot], queue=True)
            msg.submit(lambda: "", outputs=[msg])
            clear.click(lambda: None, None, chatbot, queue=False)

    with gr.Accordion("Upload and Load Documents", open=False):
        file_upload = gr.File(label="Upload .txt or .pdf", file_types=[".txt", ".pdf"])
        load_button = gr.Button("Load")
        load_button.click(load_documents, inputs=[file_upload], outputs=gr.Textbox(label="Load Status"))

    with gr.Accordion("Add Ingredients to RAG", open=False):
        ingredients_input = gr.Textbox(label="Ingredients (comma-separated)")
        add_ingredients_button = gr.Button("Submit Ingredients")
        add_ingredients_button.click(
            lambda s: add_to_rag("", s, ""),  # only ingredients filled
            inputs=[ingredients_input],
            outputs=gr.Textbox(label="Status")
        )

    with gr.Accordion("Add Seasonal Info to RAG", open=False):
        season_input = gr.Textbox(label="Season")
        add_season_button = gr.Button("Submit Season")
        add_season_button.click(
            lambda s: add_to_rag(s, "", ""),  # only season filled
            inputs=[season_input],
            outputs=gr.Textbox(label="Status")
        )

    with gr.Accordion("Add Allergies/Dietary Restrictions", open=False):
        restrictions_input = gr.Textbox(label="Dietary Restrictions (comma-separated)")
        add_restrictions_button = gr.Button("Submit Restrictions")
        add_restrictions_button.click(
            lambda r: add_to_rag("", "", r),  # only restrictions filled
            inputs=[restrictions_input],
            outputs=gr.Textbox(label="Status")
        )

    with gr.Accordion("Explore System Files", open=False):
        refresh_button = gr.Button("Refresh File List")
        file_list = gr.Dropdown(choices=[], label="Available Files")
        file_contents = gr.Textbox(label="File Preview", interactive=False)
        file_preview = gr.Image(label="PDF Snapshot", visible=False)

        def refresh_files():
            files = list_system_data_files()
            return gr.update(choices=files, value=None)

        refresh_button.click(refresh_files, outputs=[file_list])
        file_list.change(read_selected_file, inputs=[file_list], outputs=[file_contents, file_preview])

    with gr.Accordion("About This Project", open=False):
        open_pdf_button = gr.Button("Show About Page")
        pdf_viewer = gr.Image(type="filepath", value="./about_us.png", visible=False, label="")
        close_pdf_button = gr.Button("Close", visible=False)

        open_pdf_button.click(show_pdf, outputs=[pdf_viewer, close_pdf_button], queue=False)
        close_pdf_button.click(hide_pdf, outputs=[pdf_viewer, close_pdf_button], queue=False)
