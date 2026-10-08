"""Gradio interface for saved MVTec PatchCore models."""
import argparse
from pathlib import Path
import gradio as gr
from mvtec_patchcore import Inspector

def build_app(run_dir):
    inspector = Inspector(run_dir)
    categories = inspector.categories()
    if not categories:
        raise RuntimeError(
            f"No trained category models found in {Path(run_dir).resolve() / 'models'}. "
            "Train models with the notebook or download a trained run folder. "
            "Pass its path using --run-dir. See README.md."
        )
    def inspect(image, category):
        try:
            return inspector.inspect(image, category)
        except Exception as error:
            raise gr.Error(str(error)) from error
    with gr.Blocks(title="Visual Product Inspection") as demo:
        gr.Markdown("# Visual Product Inspection\nSelect the product category, upload a photo, and inspect the highlighted regions.")
        with gr.Row():
            category = gr.Dropdown(categories, value=categories[0], label="Product category")
            image = gr.Image(type="pil", label="Upload product image", sources=["upload"])
        button = gr.Button("Inspect image", variant="primary")
        with gr.Row():
            original = gr.Image(label="Resized input")
            heatmap = gr.Image(label="Anomaly heatmap")
            overlay = gr.Image(label="Highlighted regions")
        result = gr.JSON(label="Inspection result")
        gr.Markdown("Scores are not defect probabilities. Use images similar to the training data.")
        button.click(inspect, [image, category], [original, heatmap, overlay, result], concurrency_limit=1)
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Folder containing models/*.pt")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app(args.run_dir).queue().launch(server_name="127.0.0.1", server_port=args.port, share=False)
