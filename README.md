# Freedom QJ League Analytics

This folder contains everything needed to deploy the Streamlit app publicly.

## Included files
- pp.py – Streamlit entry point (RAG-based stat resolver).
- equirements.txt – Python dependencies.
- 	eam_stats_with_season_ru_with_alt.csv – dataset used by the app.
- stat_aliases.json – stat alias mapping file.
- logo.png – logo displayed in the app header.

## Quick start
1. Ensure Python 3.10+ is installed.
2. Create and activate a virtual environment if desired.
3. Install dependencies:
   `ash
   pip install -r requirements.txt
   `
4. Run Streamlit locally:
   `ash
   streamlit run app.py
   `

## Deploying to Streamlit Community Cloud
1. Push these files to a public GitHub repository.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app, pointing to pp.py on the main branch.
4. In *App settings ➜ Secrets*, add your OPENAI_API_KEY if you plan to use the AI assistant.
5. Click *Deploy* and share the generated link.

## Environment variables
- OPENAI_API_KEY *(optional)* – required for the AI assistant tab to work. Without it, the rest of the UI still functions.
