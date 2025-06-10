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
            gr.Markdown("## ✏️ Personal Notepad")
            gr.Markdown("Every food system has stories. This is your personal scratchpad — where dreams of fermented miso, grandparent recipes, or zero-waste ideas take root.")

            user_name = gr.Textbox(
                label="What's your name?",
                placeholder="🌱 Pick a name — like Lily, FarmerJay, or PickleQueen..."
            )
            name_submit = gr.Button("Start My Idea Journal")

            user_entry = gr.Textbox(
                label="New thought, recipe, tip, or idea?",
                lines=4,
                placeholder="E.g. 'I want to try purple yam bread', 'we always freeze carrot tops', 'ask mom about our stew herbs'"
            )
            entry_submit = gr.Button("Save to My Notepad")

            name_submit.click(set_user_name, inputs=[user_name], outputs=[user_name])
            entry_submit.click(append_to_user_rag, inputs=[user_entry], outputs=[user_entry])

        with gr.Column(elem_classes="section"):
            gr.Markdown("## 🥬 Upload & Detect Vegetables")
            gr.Markdown("Upload a photo of your fridge finds, backyard harvest, or farmers market bounty. We'll ID the ingredients and add them to your RAG knowledge base — automatically.")
            veg_image = gr.File(label="📷 Upload Image", file_types=["image"])
            detect_button = gr.Button("🔍 Detect Vegetables")
            detected_output = gr.Textbox(interactive=False, show_label=False)
            detect_button.click(handle_image_upload, inputs=[veg_image], outputs=[detected_output])

    with gr.Column(elem_classes="section"):
        gr.Markdown("## 💬 Chat with the Assistant")
        gr.Markdown("Ask for a zero-waste lunch, a gut-friendly dinner idea, or ways to preserve the okra from your neighbor. RootWise chats are powered by dynamic prompts and locally informed advice.")
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
        gr.Markdown("## 📚 Data Tools")

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
            gr.Markdown("### 🌍 About This Project")
            gr.Markdown("Why does RootWise exist? Because tech can be local, loving, and low-waste. Learn how this system connects AI with land, food, and care.")
            open_pdf_button = gr.Button("📖 Show About Page")
            pdf_viewer = gr.Image(type="filepath", value="./about_us.png", visible=False)
            close_pdf_button = gr.Button("❌ Close", visible=False)
            open_pdf_button.click(show_pdf, outputs=[pdf_viewer, close_pdf_button], queue=False)
            close_pdf_button.click(hide_pdf, outputs=[pdf_viewer, close_pdf_button], queue=False)
