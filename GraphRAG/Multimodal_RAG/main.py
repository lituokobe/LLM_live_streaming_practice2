import os

import gradio as gr

from Live_Streaming_practice2.GraphRAG.Multimodal_RAG.utils.common_utils import delete_directory_if_non_empty, \
    get_filename, get_sorted_md_files
from Live_Streaming_practice2.GraphRAG.Multimodal_RAG.utils.log_utils import log
from Live_Streaming_practice2.GraphRAG.graph_rag_demo.dots_ocr.parser import do_parse

base_md_dir = r'./output'

class ProcessorAPP:
    def __init__(self):
        self.pdf_path = None
        self.md_dir = None
        self.md_dir = None
        self.md_files = None
        self.file_contents = {}

    def upload_pdf(self, pdf_file):
        """Handle PDF file upload"""
        log.info(f"Uploading PDF file: {pdf_file}")
        self.pdf_path = pdf_file if pdf_file else None
        if self.pdf_path:
            return [
                f"PDF uploaded: {os.path.basename(self.pdf_path)}",  # status
                gr.Button(interactive=True)
            ]
        else:
            return [
                "Upload failed, please re-upload PDF file",  # status
                gr.Button(interactive=False)
            ]

    def parse_pdf(self):
        """Parse PDF file and generate MD files"""
        md_files_dir = os.path.join(base_md_dir, get_filename(self.pdf_path, False))
        delete_directory_if_non_empty(md_files_dir)
        do_parse(input_path=self.pdf_path, num_thread=32, no_fitz_preprocess=True)
        if os.path.isdir(md_files_dir):
            self.md_dir = md_files_dir
            log.info(f"PDF parsed, generated {len(os.listdir(md_files_dir))} MD files")
            self.md_files = get_sorted_md_files(md_files_dir)
            log.info(f"PDF parsed, MD file list: {self.md_files}")
            # show the content of the MD files in GUI, read all MD files
            for f in self.md_files:
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        self.file_contents[f] = file.read()
                except Exception as e:
                    print(f"Read file {f} error: {e}")
                    self.file_contents[f] = f"Read file content error: {e}"
            file_names = [os.path.basename(f) for f in self.md_files]
            return [
                f"Parsing completed, generated {len(self.md_files)} MD files",  # status
                gr.Dropdown(choices=file_names, value = None, label="MD file list", interactive=True),  # file_dropdown
                gr.Button(interactive=False),  # parse_btn
                gr.update(interactive=True)  # save_btn - 使用 gr.update
            ]
        else:
            return [
                f"Parsing failure",  # status
                gr.Dropdown(interactive=False),  # file_dropdown
                gr.Button(interactive=True),  # parse_btn
                gr.update(interactive=False)  # save_btn - 使用 gr.update
            ]

    def select_md_file(self, selected_file):
        """Select a MD file and display its content"""
        log.info(f"Selecting MD file: {selected_file}")
        if selected_file:
            show_file = None
            for f in self.md_files:
                if os.path.basename(f) == selected_file:
                    show_file = f
                    break
            if show_file and show_file in self.file_contents:
                return self.file_contents[show_file]
            else:
                return "No file selected"
        else:
            return "Failure to load the file."

    def create_interface(self):
        """Create GUI of multimodal knowledge database"""
        with gr.Blocks() as app:
            gr.Markdown("## Parsing PDF and Create Knowledge Base")

            with gr.Row():
                pdf_upload = gr.File(label="Upload PDF file")
                parse_btn = gr.Button("ParsPDF", variant="primary", interactive=False) # by default non-interactor

            status = gr.Textbox(label="Status", value="Waiting...", interactive=False)

            with gr.Row():
                # MD file list
                file_dropdown = gr.Dropdown(choices=[], label="MD file list", interactive=False)
                # MD file content
                content = gr.Textbox(label="MD file content", lines=20, interactive=False)

            save_btn = gr.Button("Save to Database", variant="secondary", interactive=False)

            # 绑定按钮点击事件
            pdf_upload.change(
                fn=self.upload_pdf,
                inputs=pdf_upload,
                outputs=[status, parse_btn]
            )

            parse_btn.click(
                fn=self.parse_pdf,
                inputs=[],
                outputs=[status, file_dropdown, parse_btn, save_btn]
            )

            file_dropdown.change(
                fn=self.select_md_file,
                inputs=file_dropdown,
                outputs=content
            )

        return app



if __name__ == '__main__':
    app = ProcessorAPP()
    interface = app.create_interface()
    interface.launch()