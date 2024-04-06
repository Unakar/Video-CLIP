'''
The backend of the whole sysytem. Mainly build with gradio.
Waiting for User Feedback.
'''
import gradio as gr
from backend_config import *
import retrieval.retrieval_pipeline as retrieval_pipeline


class System_Backend:
    def __init__(self,topk=10,retrieval=retrieval_pipeline):
        self.topk = topk
        self.retrieval = retrieval
        self.page = self.build_page()

    def build_page(self):
        with gr.Blocks(theme=gr.themes.Default(spacing_size=gr.themes.sizes.spacing_lg, radius_size=gr.themes.sizes.radius_lg,   text_size=gr.themes.sizes.text_lg)) as demo:
            gr.HTML(HTML)
            gr.Markdown("## CLIP-base Model")
            with gr.Row():
                with gr.Column():
                    inp = gr.Textbox(placeholder="Your Query Prompt Here")
                    btn = gr.Button(value="Search")
                    ex = [["natural wonders of the world"],["yoga routines for morning energy"], 
                        ["baking chocolate cake"],["birds fly in the sky"]]
                    gr.Examples(examples=ex,inputs=[inp])
                with gr.Column():
                    out = [gr.HTML() for _ in range(5)]
            btn.click(self.retrieval.retrieval, inputs=inp, outputs=out)
            gr.Markdown("the end")
        demo.launch(share=True)

if __name__ == "__main__":
    System_Backend()