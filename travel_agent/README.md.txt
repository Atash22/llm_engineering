# ✈️ TravelBuddy — Multimodal AI Travel Agent

A Gradio chatbot that plans trips using OpenAI tool-calling:
live weather (Open-Meteo), live currency rates (Frankfurter),
a stateful itinerary, gpt-image-1 destination art, TTS voice
replies, and one-command PDF export.

## Run it
1. `pip install openai gradio requests python-dotenv pillow fpdf2`
2. Create a `.env` file with `OPENAI_API_KEY=sk-...`
3. Open `travel_agent.ipynb` and run all cells.