import gradio as gr
import plotly.graph_objects as go
from evaluation.compare import run_comparison


def build_comparison_chart(simple_scores, agentic_scores):
    categories = ["Accuracy", "Completeness", "Relevance"]
    simple_values = [simple_scores["accuracy"],
                     simple_scores["completeness"], simple_scores["relevance"]]
    agentic_values = [agentic_scores["accuracy"],
                      agentic_scores["completeness"], agentic_scores["relevance"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Simple RAG", x=categories, y=simple_values))
    fig.add_trace(go.Bar(name="Agentic RAG", x=categories, y=agentic_values))
    fig.update_layout(barmode="group", yaxis_range=[
                      0, 5], title="Simple RAG vs Agentic RAG")
    return fig


def run_and_plot(num_tests):
    simple_scores, agentic_scores = run_comparison(int(num_tests))
    fig = build_comparison_chart(simple_scores, agentic_scores)
    summary = (
        f"**Simple RAG** — Accuracy: {simple_scores['accuracy']:.2f}, "
        f"Completeness: {simple_scores['completeness']:.2f}, Relevance: {simple_scores['relevance']:.2f}\n\n"
        f"**Agentic RAG** — Accuracy: {agentic_scores['accuracy']:.2f}, "
        f"Completeness: {agentic_scores['completeness']:.2f}, Relevance: {agentic_scores['relevance']:.2f}"
    )
    return fig, summary


def main():
    with gr.Blocks(title="RAG Evaluation Comparison") as ui:
        gr.Markdown("# 📊 Simple RAG vs Agentic RAG — Evaluation Comparison")

        with gr.Row():
            num_tests = gr.Slider(
                minimum=1, maximum=20, value=10, step=1, label="Number of tests to run")
            run_button = gr.Button("Run Comparison", variant="primary")

        plot_output = gr.Plot(label="Score Comparison")
        summary_output = gr.Markdown()

        run_button.click(run_and_plot, inputs=[num_tests], outputs=[
                         plot_output, summary_output])

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
