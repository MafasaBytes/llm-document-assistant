import gradio as gr
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from retriever.QA_chain import retriever_qa

logger = logging.getLogger(__name__)


def _safe_qa(file, query):
    """
    Thin wrapper around retriever_qa that guarantees a user-friendly
    string is always returned — even if something goes completely wrong.
    """
    try:
        return retriever_qa(file, query)
    except Exception as exc:
        logger.exception("Unhandled error surfaced to the UI.")
        return f"Something went wrong: {exc}"


def launch_gradio_app():
    # Professional medical theme with refined colors
    theme = gr.themes.Soft(
        primary_hue="blue",           
        secondary_hue="slate",
        neutral_hue="gray",
        font=[gr.themes.GoogleFont('Inter'), gr.themes.GoogleFont('Open Sans'), 'ui-sans-serif', 'system-ui'],
        font_mono=[gr.themes.GoogleFont('JetBrains Mono'), 'ui-monospace', 'Consolas', 'monospace'],
    ).set(
        # Light theme colors
        body_background_fill="#f8fafc",
        block_background_fill="#ffffff",
        block_border_color="#e2e8f0",
        button_primary_background_fill="#3b82f6",
        button_primary_background_fill_hover="#2563eb",
        button_secondary_background_fill="#f1f5f9",
        # Dark theme colors
        body_background_fill_dark="#0f172a",
        block_background_fill_dark="#1e293b",
        block_border_color_dark="#334155",
        button_primary_background_fill_dark="#3b82f6",
        button_primary_background_fill_hover_dark="#2563eb",
        input_background_fill_dark="#334155",
    )

    with gr.Blocks(
        title="HealthDoc AI Assistant",
        theme=theme,
        css="""
        .gradio-container {
            max-width: 1100px !important;
            margin: auto !important;
            padding: 1.5rem !important;
        }
        h1 {
            color: #1e40af !important;
            font-weight: 600 !important;
            margin-bottom: 0.5rem !important;
        }
        .header-text {
            color: #475569 !important;
            font-size: 1.05rem !important;
            line-height: 1.5 !important;
            margin-bottom: 1.5rem !important;
        }
        .upload-container {
            background: white !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 0.6rem !important;
            padding: 1.2rem !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        }
        .query-box {
            border-radius: 0.6rem !important;
            border: 1px solid #cbd5e1 !important;
        }
        .ask-button {
            height: 3.2rem !important;
            font-size: 1.05rem !important;
            font-weight: 500 !important;
            border-radius: 0.6rem !important;
            transition: all 0.2s !important;
            margin-top: 0.5rem !important;
        }
        .output-box {
            min-height: 180px !important;
            background: #f8fafc !important;
            border-radius: 0.6rem !important;
            font-size: 1.02rem !important;
            line-height: 1.6 !important;
            padding: 1.1rem !important;
            border: 1px solid #e2e8f0 !important;
        }
        .examples-container {
            margin-top: 1.5rem !important;
        }
        .examples-label {
            color: #64748b !important;
            font-weight: 500 !important;
        }
        .disclaimer {
            font-size: 0.85rem !important;
            color: #64748b !important;
            margin-top: 1.5rem !important;
            line-height: 1.5 !important;
        }
        """
    ) as app:
        gr.Markdown("""
        # 🏥 HealthDoc AI Assistant
        """)

        gr.Markdown("""
        Upload a healthcare document (PDF) and ask questions about its content.
        The assistant extracts and reasons over the information in the document.
        """, elem_classes="header-text")

        with gr.Group(elem_classes="upload-container"):
            with gr.Row():
                with gr.Column(scale=1, min_width=340):
                    pdf_input = gr.File(
                        label="Upload Healthcare Document (PDF)",
                        file_types=[".pdf"],
                        type="filepath",
                        elem_id="pdf-upload"
                    )

                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g. What is the recommended dosage? Are there any contraindications?",
                        lines=3,
                        max_lines=5,
                        elem_classes="query-box"
                    )

                    submit_btn = gr.Button(
                        "Ask HealthDoc AI",
                        variant="primary",
                        elem_classes="ask-button"
                    )

                with gr.Column(scale=2):
                    output = gr.Textbox(
                        label="AI Response",
                        interactive=False,
                        lines=12,
                        elem_classes="output-box"
                    )

                    # Fixed: Removed elem_classes from Examples
                    gr.Examples(
                        examples=[
                            ["What is the name of the applicant?"],
                            ["What is the recommended dosage?"],
                            ["Are there any contraindications?"],
                            ["What are the clinical trial results?"],
                            ["What are the most common side effects?"]
                        ],
                        inputs=query_input,
                        label="Example Questions"
                    )

        # Add medical disclaimer
        gr.Markdown("""
        **Disclaimer:** This tool is for research purposes only. Always consult with a healthcare professional for medical advice.
        """, elem_classes="disclaimer")

        # Connect the function
        submit_btn.click(
            fn=_safe_qa,
            inputs=[pdf_input, query_input],
            outputs=output,
            api_name="healthdoc_qa"
        )

    return app.launch(
        server_name="127.0.0.1",
        server_port=7860,
    )

if __name__ == "__main__":
    launch_gradio_app()
