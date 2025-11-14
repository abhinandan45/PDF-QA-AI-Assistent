from flask import Flask, render_template, request, jsonify, session
from rag import RAGIndex
import requests
import os
from werkzeug.utils import secure_filename
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  

# ⚙️ Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔑 OpenRouter API configuration
API_KEY = "sk-or-v1-cf0e013bbeceb65c0266ced0b95dc811341563315245599222ec18e7d99b5ac"
MODEL = "deepseek/deepseek-chat"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://openrouter.ai/",
    "X-Title": "Abhinandan-RAG-App"
}

# Store RAG instances per session
rag_instances = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_rag_instance(session_id):
    """Get or create RAG instance for session"""
    return rag_instances.get(session_id)

def set_rag_instance(session_id, rag_instance):
    """Set RAG instance for session"""
    rag_instances[session_id] = rag_instance

# 🚀 Function to call DeepSeek API
def ask_api(prompt):
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers strictly based on the provided context. If the answer cannot be found in the context, say 'I cannot find this information in the document.'"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 600,
        "temperature": 0.4
    }

    try:
        resp = requests.post(ENDPOINT, headers=headers, json=data, timeout=30)
        if resp.status_code != 200:
            logger.error(f"❌ API Error: {resp.status_code} - {resp.text}")
            return "⚠️ API Error. Please check your key or try again."
        ans = resp.json()
        return ans["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "⚠️ Request timeout. Please try again."
    except Exception as e:
        logger.error(f"❌ API Exception: {str(e)}")
        return f"⚠️ Exception: {str(e)}"

@app.route("/")
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template("index2.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_id = session['session_id']
    
    if 'pdf' not in request.files:
        return jsonify({"success": False, "message": "No file selected"})
    
    file = request.files['pdf']
    
    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected"})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
            file.save(filepath)
            
            logger.info(f"📤 File uploaded: {filename}")
            
            # Initialize RAG with uploaded PDF
            rag = RAGIndex(filepath)
            set_rag_instance(session_id, rag)
            
            # Store file info in session
            session['uploaded_file'] = {
                'name': filename,
                'size': os.path.getsize(filepath),
                'path': filepath
            }
            
            # Get document info for debugging
            doc_info = rag.get_document_info()
            logger.info(f"📊 Document info: {doc_info}")
            
            return jsonify({
                "success": True, 
                "message": "File uploaded successfully!",
                "file": {
                    "name": filename,
                    "size": session['uploaded_file']['size']
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Error processing file: {str(e)}")
            return jsonify({"success": False, "message": f"Error processing file: {str(e)}"})
    
    return jsonify({"success": False, "message": "Invalid file type. Please upload a PDF."})

@app.route("/ask", methods=["POST"])
def ask():
    if 'session_id' not in session:
        return jsonify({"response": "Please upload a PDF file first."})
    
    session_id = session['session_id']
    rag = get_rag_instance(session_id)
    
    if not rag:
        return jsonify({"response": "Please upload a PDF file first."})
    
    query = request.json.get("message", "").strip()
    
    if not query:
        return jsonify({"response": "Please enter a question."})
    
    try:
        logger.info(f"❓ Query: {query}")
        
        docs = rag.retrieve(query)
        context = "\n".join(docs)
        
        logger.info(f"📚 Retrieved {len(docs)} context chunks")
        for i, doc in enumerate(docs):
            logger.info(f"Context {i+1}: {doc[:100]}...")

        final_prompt = f"""
Based on the following context from a PDF document, please answer the user's question. 
If the information is not available in the context, please say "I cannot find this information in the document."

Context from PDF:
{context}

User Question: {query}

Please provide a helpful answer based only on the context above:
"""

        logger.info(f"📝 Sending prompt to API...")
        response = ask_api(final_prompt)
        logger.info(f"🤖 API Response: {response}")
        
        return jsonify({"response": response})
    
    except Exception as e:
        logger.error(f"❌ Error in /ask: {str(e)}")
        return jsonify({"response": f"Error processing your request: {str(e)}"})

@app.route("/files", methods=["GET"])
def get_files():
    if 'session_id' not in session or 'uploaded_file' not in session:
        return jsonify({"files": []})
    
    return jsonify({
        "files": [session['uploaded_file']]
    })

@app.route("/clear", methods=["POST"])
def clear_files():
    if 'session_id' in session:
        session_id = session['session_id']
        # Remove RAG instance
        if session_id in rag_instances:
            rag_instances.pop(session_id)
        
        # Remove uploaded file
        if 'uploaded_file' in session:
            filepath = session['uploaded_file']['path']
            if os.path.exists(filepath):
                os.remove(filepath)
            session.pop('uploaded_file')
    
    return jsonify({"success": True, "message": "Files cleared successfully"})

@app.route("/suggestions", methods=["GET"])
def get_suggestions():
    suggestions = [
        "Summarize the document",
        "What are the key points?"
    ]
    return jsonify({"suggestions": suggestions})

@app.route("/debug", methods=["GET"])
def debug_info():
    """Debug endpoint to check RAG status"""
    if 'session_id' not in session:
        return jsonify({"error": "No session"})
    
    session_id = session['session_id']
    rag = get_rag_instance(session_id)
    
    if not rag:
        return jsonify({"error": "No RAG instance"})
    
    doc_info = rag.get_document_info()
    return jsonify({
        "session_id": session_id,
        "document_info": doc_info
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
