import sqlite3
import openai
import datetime
import random
import requests
from typing import List, Dict, Any
from tabulate import tabulate
import os
import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_llm(user_text, max_tokens=None, temperature=None, top_p=0.5):
    # Get parameters from environment variables, with fallbacks to function parameters
    api_key = os.getenv('API_KEY', 'ollama')
    base_url = os.getenv('BASE_URL', 'BASE_URL')
    model_name = os.getenv('MODEL_NAME', 'T-pro')
    env_temp = float(os.getenv('TEMPERATURE', '0.1'))
    env_max_tokens = int(os.getenv('MAX_COMPLETION_TOKENS', '1024'))
    
    # Use function parameters if provided, otherwise use env vars
    max_tokens = max_tokens if max_tokens is not None else env_max_tokens
    temperature = temperature if temperature is not None else env_temp
    
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        max_completion_tokens=max_tokens       
    )
    
    prompt = f"""
    Задача: Классифицировать тематику обращений клиентов энергосбытовой компании на основе представленных диалогов. 
    Необходимо создать набор категорий для автоматической обработки будущих запросов:\n{user_text}"""
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Ошибка: {str(e)}"

def get_messages_by_session_id(session_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve all messages with the same session_id from the database.
    
    Args:
        session_id: The unique identifier for the session
        
    Returns:
        List of message dictionaries ordered by timestamp
    """
    db_path = os.getenv('DB_PATH', 'messages.db')
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  
        cursor = conn.cursor()
        
        query = """
            SELECT id, session_id, content, timestamp, sender
            FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """
        
        cursor.execute(query, (session_id,))
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            result.append({
                "id": row["id"],
                "session_id": row["session_id"],
                "content": row["content"],
                "timestamp": row["timestamp"],
                "sender": row["sender"]
            })
            
        conn.close()
        return result
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []

def list_all_sessions() -> List[str]:
    """
    List all unique session IDs in the database.
    
    Returns:
        List of unique session IDs
    """
    db_path = os.getenv('DB_PATH', 'messages.db')
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT session_id FROM messages")
        sessions = cursor.fetchall()
        
        conn.close()
        return [session[0] for session in sessions]
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []

def create_dialogue_from_messages(messages: List[Dict[str, Any]]) -> str:
    """
    Combine messages into a single dialogue string.
    
    Args:
        messages: List of message dictionaries ordered by timestamp
        
    Returns:
        A formatted dialogue string
    """
    dialogue = ""
    
    for msg in messages:
        # Format: [Timestamp] [Sender] Message content
        timestamp = msg['timestamp']
        if isinstance(timestamp, str):
            # Try to parse the timestamp if it's a string
            try:
                dt = datetime.datetime.fromisoformat(timestamp)
                timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        
        dialogue += f"[{timestamp}] [{msg['sender']}]: {msg['content']}\n\n"
    
    return dialogue.strip()

def analyze_dialogue(dialogue: str) -> Dict[str, Any]:
    """
    Send the dialogue to Hyperbolic API for analysis and classification.
    
    Args:
        dialogue: The formatted dialogue string
        
    Returns:
        Dictionary containing analysis results with tags
    """
    try:
        prompt = f"""Analyze the following dialogue and provide:
1. A brief summary
2. Main topics discussed
3. Sentiment analysis
4. Key action items (if any)
5. Classification tags (e.g., #support, #sales, #technical, #complaint)

Dialogue:
{dialogue}"""

        analysis = get_llm(prompt, max_tokens=1000, temperature=0.3)
        
        return {
            "analysis": analysis,
            "model_used": os.getenv('MODEL_NAME', 'T-pro'),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"API error: {e}")
        return {
            "error": str(e),
            "mock_analysis": "This is a mock analysis since API is not configured. The dialogue appears to be about customer support."
        }

def process_session(session_id: str) -> Dict[str, Any]:
    """
    Main function to process a session: retrieve messages, create dialogue, and analyze.
    
    Args:
        session_id: The unique identifier for the session
        
    Returns:
        Dictionary with the dialogue and analysis results
    """
    # Get messages for the session
    messages = get_messages_by_session_id(session_id)
    
    if not messages:
        return {"error": "No messages found for this session ID"}
    
    dialogue = create_dialogue_from_messages(messages)
    
    analysis = analyze_dialogue(dialogue)
    
    return {
        "session_id": session_id,
        "message_count": len(messages),
        "dialogue": dialogue,
        "analysis": analysis
    }

def save_analysis_to_file(results: List[Dict[str, Any]], output_file: str = "analysis_results.txt"):
    """
    Save analysis results to a file in table format.
    
    Args:
        results: List of analysis results
        output_file: Path to the output file
    """
    table_data = []
    for result in results:
        analysis = result.get("analysis", {})
        if isinstance(analysis, dict):
            analysis_text = analysis.get("analysis", "No analysis available")
        else:
            analysis_text = analysis
            
        table_data.append([
            result["session_id"],
            result["message_count"],
            analysis_text,
            analysis.get("model_used", "N/A"),
            analysis.get("timestamp", "N/A")
        ])
    
    headers = ["Session ID", "Message Count", "Analysis", "Model Used", "Timestamp"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Dialogue Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(table)
    
    print(f"\nAnalysis results saved to {output_file}")

def save_analysis_to_excel(results: List[Dict[str, Any]], output_file: str = "analysis_results.xlsx"):
    """
    Save analysis results to an Excel file.
    
    Args:
        results: List of analysis results
        output_file: Path to the output Excel file
    """
    # Prepare data for the DataFrame
    data = []
    for result in results:
        analysis = result.get("analysis", {})
        if isinstance(analysis, dict):
            analysis_text = analysis.get("analysis", "No analysis available")
        else:
            analysis_text = analysis
            
        data.append({
            "Session ID": result["session_id"],
            "Message Count": result["message_count"],
            "Analysis": analysis_text,
            "Model Used": analysis.get("model_used", "N/A"),
            "Timestamp": analysis.get("timestamp", "N/A")
        })
    
    df = pd.DataFrame(data)
    
    # Save to Excel
    df.to_excel(output_file, index=False, sheet_name='Analysis Results')
    
    # Auto-adjust column widths
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, index=False, sheet_name='Analysis Results')
        worksheet = writer.sheets['Analysis Results']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).apply(len).max(),
                len(str(col))
            )
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 100)
    
    print(f"\nAnalysis results saved to {output_file}")

if __name__ == "__main__":
    # List all sessions
    sessions = list_all_sessions()
    print(f"Available sessions: {sessions}")
    
    # Process each session and collect results
    results = []
    for session_id in sessions:
        print(f"\n{'='*50}")
        print(f"Processing session: {session_id}")
        print(f"{'='*50}")
        
        result = process_session(session_id)
        results.append(result)
        
        print("\nDialogue:")
        print(result["dialogue"])
        
        print("\nAnalysis:")
        if "error" in result.get("analysis", {}):
            print("Mock analysis (API not configured):")
            print(result["analysis"].get("mock_analysis", "No analysis available"))
        else:
            print(result["analysis"].get("analysis", "No analysis available"))
        
        print(f"{'='*50}\n")
    
    save_analysis_to_excel(results)