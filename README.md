# KA App Analyzer

Application for analyzing customer dialogue sessions using LLM.

## Description

This application connects to a message database and uses the ChatOpenAI model to analyze customer dialogues. It classifies the topics of customer requests for an energy sales company and generates detailed analysis reports.

## Features

- Retrieves dialogue sessions from a database
- Formats dialogue texts for analysis
- Uses an LLM to analyze and classify customer dialogues
- Generates detailed reports on conversation topics, sentiment, and key points
- Exports results to Excel for easy review

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file with the following configuration:

```
# LLM Configuration
API_KEY=ollama
BASE_URL=http://your-llm-server-url/v1
MODEL_NAME=T-pro
TEMPERATURE=0.1
MAX_COMPLETION_TOKENS=1024

# Database Configuration
DB_PATH=messages.db
```

## Usage

Simply run the script:

```bash
python testing_ka.py
```

The script will:
1. Connect to your database
2. Retrieve all available dialogue sessions
3. Process each session using the LLM
4. Save analysis results to an Excel file