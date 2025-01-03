# Sentence Rephrasing with Masked Phrase

## Setup Instructions
- Note: All code should be run from root directory!
### 1. Create a Virtual Environment

#### Using Conda:
```bash
conda create --name rephrase_env python=3.8
conda activate rephrase_env
```

#### Using Python `venv`:
```bash
python3 -m venv rephrase_env
# On Windows:
rephrase_env\Scripts\activate
# On macOS/Linux:
source rephrase_env/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Obtain the `.env` File and Data Folder

Download the `.env` file (with your OpenAI API key) and the `data` folder (containing the CSV file) from the project’s Google Drive. Place them in the project root.

### 4. Set Up `.env` File on Your Own

Create the `.env` file in the root directory, add your OpenAI API key:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the Script
```bash
python create_paraphrase.py
```

The output will be saved in `data` folder.

### 6. Running opensource LLM

Download [Ollama](https://ollama.com/download) and then deploy it in your local machine using the following script before running our opensource script
```bash
ollama serve 
```
For further documentation refer to [Ollama Docs](https://github.com/ollama/ollama)