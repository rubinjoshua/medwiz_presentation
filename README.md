# Medwiz Presentation

Offline, terminal-native demo for Medwiz executives showing two AI capabilities:

1. **Sig translation** – translate messy free-text prescription instructions ("sigs") into precise structured JSON.
2. **Dose validation** – validate that the sig’s dosage/frequency is plausible for the given drug using a local medical knowledge base.

Everything runs **fully offline** during the presentation:

- Local LLM via Ollama + LangChain (no calls to external APIs once models are pulled).
- Local vector DB (Chroma) for both sig examples and medical knowledge.
- Right-hand terminal pane shows evolving tables (inputs and outputs with ✅/❌).
- Left-hand pane shows the pipeline in action: retrievals, prompts, LLM outputs, and CSV updates.

## Quickstart (high level)

1. Install Python 3.11+ and `pip`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install [Ollama](https://ollama.com/) and pull a small local model (good on an M2 with 16GB RAM), e.g.:
   ```bash
   ollama pull llama3.2
   ```
4. Generate demo data and indexes (while online):
   ```bash
   python scripts/setup_data.py
   python scripts/build_indexes.py
   ```
5. For the live pitch (wifi off):
   - Right pane:
     ```bash
     python run_display.py
     ```
   - Left pane:
     ```bash
     python main_pipeline.py
     ```

The left side walks through translating each sig and validating it, while the right side shows the table filling up with ✅ (OK) and ❌ (needs attention).
