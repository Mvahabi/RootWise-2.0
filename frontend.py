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
.gradio-container {
    background-color: #F2F6EA;
    font-family: 'Georgia', serif;
    color: #2D3A2E;
}

h1, h2, h3, .gr-markdown h2 {
    text-align: center;
    font-family: 'Georgia', serif;
    color: #3B4A3E;
}

.section {
    padding: 1.25rem;
    margin: 1rem;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.95);
    box-shadow: 0px 1px 4px rgba(0, 0, 0, 0.05);
}

.message {
    padding: 0.6rem 1rem !important;
    margin: 0.4rem 0 !important;
    background-color: #F9FAF5 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    color: #2D3A2E !important;
    border: none !important;
}

.message.user {
    background-color: #E6F0D7 !important;
               
}
""") as demo:

    with gr.Row():
        gr.Image(type="filepath", value="./frontpage.png", visible=True)

    with gr.Row():
        with gr.Column(elem_classes="section"):
            gr.Markdown("## Personal Notepad")
            gr.Markdown("Heard something tasty or thought of a clever zero-waste trick? This is your personal notepad (where dreams take root).")

            user_name = gr.Textbox(
                label="What's your name?",
                placeholder="🌱 Pick a name — like Kevin or PickleQueen... This will be your key to this notebook"
            )
            name_submit = gr.Button("Start Notepad")

            user_entry = gr.Textbox(
                label="New thought, recipe, tip, or idea?",
                lines=4,
                placeholder="E.g. 'I want to try purple yam bread', 'we always freeze carrot tops', 'ask mom about our stew herbs'"
            )
            entry_submit = gr.Button("Save to Database")

            name_submit.click(set_user_name, inputs=[user_name], outputs=[user_name])
            entry_submit.click(append_to_user_rag, inputs=[user_entry], outputs=[user_entry])

        with gr.Column(elem_classes="section"):
            # State variables
            season_state = gr.State("")
            restrictions_state = gr.State("")

            # Section header and instructions
            gr.Markdown("## Upload Vegetables")
            gr.Markdown(
                "Snap a photo of your backyard harvest or an overwhelmingly beautiful farmer's market stand. "
                "We'll identify the ingredients and add them to your personalized knowledge base."
            )

            # Upload and detection UI
            veg_image = gr.File(label="📷 Upload Image", file_types=["image"])
            detect_button = gr.Button("🔍 Detect Vegetables")
            detected_output = gr.Textbox(interactive=False, show_label=False)

            # Post-detection options
            add_button = gr.Button("➕ Add to Database", visible=False)
            confirmation_msg = gr.Textbox(interactive=False, visible=False, show_label=False)

            # Detection logic + conditional UI update
            def handle_and_show(img):
                result = handle_image_upload(img)
                show_add_button = gr.update(visible=bool(result))
                hide_confirmation = gr.update(visible=False)
                return result, show_add_button, hide_confirmation

            detect_button.click(
                fn=handle_and_show,
                inputs=[veg_image],
                outputs=[detected_output, add_button, confirmation_msg]
            )

            # Add-to-database logic and UI reset
            def add_and_reset(season, ingredients, restrictions):
                if "," in ingredients:
                    ingredients = ingredients.split(",")
                    for veg in ingredients:
                        clean_veg = veg.strip(" [']\n")
                        add_to_rag(season, clean_veg, restrictions)
                else:
                    clean_veg = ingredients.strip(" [']\n")

                hide_add_button = gr.update(visible=False)
                return "Input added to RAG database!", hide_add_button

            add_button.click(
                fn=add_and_reset,
                inputs=[season_state, detected_output, restrictions_state],
                outputs=[gr.Textbox(label="Status"), add_button]
            )



    with gr.Column(elem_classes="section"):
        gr.Markdown("## Chat")
        gr.Markdown("Ask for a zero-waste lunch, a gut-friendly dinner idea, or ways to preserve the okra from your neighbor. RootWise chats are powered by data on functional medicine and whatever else you want it to remember.")
        chatbot = gr.Chatbot(type='messages')
        msg = gr.Textbox(
            label="Ask a Question",
            placeholder="e.g. What can I make with squash peels and miso?"
        )
        clear = gr.Button("🧹 Clear Chat")
        msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot], queue=True)
        msg.submit(lambda: "", outputs=[msg])
        clear.click(lambda: None, None, chatbot, queue=False)

    with gr.Row():
        gr.Markdown("## 📚 Data Tools 📚")

    with gr.Row():
        with gr.Column(elem_classes="section"):
            gr.Markdown("### 📄 Load Custom Documents")
            gr.Markdown("Bring your own wisdom! Upload PDFs or text files — think: ancestral recipe books, clinic notes, seed saving guides.")
            file_upload = gr.File(label="Upload Documents", file_types=['.txt', '.pdf'], file_count="multiple")
            load_button = gr.Button("📥 Load Documents")
            load_button.click(load_documents, inputs=[file_upload], outputs=gr.Textbox(label="Status"))

        with gr.Column(elem_classes="section"):
            gr.Markdown("### 🌾 Add to Your RAG Dataset")
            gr.Markdown("Tailor RootWise to your context: share what’s in season, what ingredients you love, and what you can’t eat.")
            ingredients_input = gr.Textbox(
                label="Ingredients (comma-separated)",
                placeholder="e.g. lentils, daikon, lemon zest"
            )
            add_ingredients_button = gr.Button("➕ Add Ingredients")
            add_ingredients_button.click(lambda s: add_to_rag("", s, ""), inputs=[ingredients_input], outputs=[ingredients_input])

            season_input = gr.Textbox(
                label="Season",
                placeholder="e.g. early summer, monsoon, winter"
            )
            add_season_button = gr.Button("📅 Add Season")
            add_season_button.click(lambda s: add_to_rag(s, "", ""), inputs=[season_input], outputs=[season_input])

            restrictions_input = gr.Textbox(
                label="Dietary Restrictions (comma-separated)",
                placeholder="e.g. gluten-free, low FODMAP, nut allergy"
            )
            add_restrictions_button = gr.Button("🚫 Add Restrictions")
            add_restrictions_button.click(lambda r: add_to_rag("", "", r), inputs=[restrictions_input], outputs=[restrictions_input])

        with gr.Column(elem_classes="section"):
            gr.Markdown("### 📂 System File Viewer")
            gr.Markdown("Peek inside RootWise’s brain. Here’s where your knowledge and documents live — transparent, traceable, and open.")
            refresh_button = gr.Button("🔄 Refresh File List")
            file_list = gr.Dropdown(choices=[], label="Available Files")
            file_contents = gr.Textbox(label="File Preview", interactive=False)
            file_preview = gr.Image(label="PDF Snapshot", visible=False)

            def refresh_files():
                files = list_system_data_files()
                return gr.update(choices=files, value=None)

            refresh_button.click(refresh_files, outputs=[file_list])
            file_list.change(read_selected_file, inputs=[file_list], outputs=[file_contents, file_preview])

        with gr.Column(elem_classes="section"):
            gr.Markdown("### 🌍 About This Project 🌍 ")
            open_pdf_button = gr.Button("Show About Page")
            pdf_viewer = gr.Image(type="filepath", value="./about_us.png", visible=False)
            close_pdf_button = gr.Button("Close", visible=False)
            open_pdf_button.click(show_pdf, outputs=[pdf_viewer, close_pdf_button], queue=False)
            close_pdf_button.click(hide_pdf, outputs=[pdf_viewer, close_pdf_button], queue=False)
