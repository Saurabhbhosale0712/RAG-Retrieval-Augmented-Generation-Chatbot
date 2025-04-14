Blog: https://medium.com/@saurabhbhosale0002/building-an-intelligent-feedback-driven-chatbot-with-phi-2-and-sentence-transformers-1c03d5bf52a0


#Intelligent Chatbot with RAG-Based Contextual Understanding**  
 
----

📌 Project Title: Context-Aware Question Answering Chatbot using Embeddings & Generative Language Model

🌟 Overview:
This project is an intelligent chatbot that answers user questions based on context retrieved from a knowledge base using embeddings and generates responses using a transformer-based language model (Phi-2). The chatbot allows interaction and collects user feedback to evaluate and improve the quality of answers.

---

🔁 Complete Workflow Breakdown:

1. 🔧 Setup & Model Initialization

   ✅ Code:
   ```python
   from sentence_transformers import SentenceTransformer
   from transformers import AutoTokenizer, AutoModelForCausalLM
   import torch
   ```

   ✅ Purpose:
   Import the required libraries to load:
   - SentenceTransformer for encoding queries into embeddings.
   - Transformers (from HuggingFace) to load the Phi-2 language model.

   ✅ Why:
   These models enable semantic understanding (via embeddings) and natural language generation (via Phi-2).

---

2. 🤖 Load Pretrained Models

   ✅ Code:
   ```python
   embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
   tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
   phi_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16, device_map="auto")
   ```

   ✅ Purpose:
   - all-MiniLM-L6-v2: Encodes sentences into vector representations for semantic search.
   - Phi-2: A powerful language model for generating human-like answers.

   ✅ Why:
   - MiniLM is lightweight, fast, and accurate for vector search.
   - Phi-2 generates fluent answers based on retrieved knowledge.

---

3. 🧠 Start Interactive QA Loop

   ✅ Code:
   ```python
   while True:
       query = input("Ask your question (or type 'byy'/'stop' to exit): ").strip().lower()
       if query in ["byy", "stop"]:
           print("Thanks for using the chatbot! 👋")
           break
   ```

   ✅ Purpose:
   - Accepts user input.
   - Allows repeated question-answer interaction until the user exits.

   ✅ Why:
   Enables a continuous chat experience for the user.

---

4. 📈 Semantic Embedding of the Query

   ✅ Code:
   ```python
   query_embedding = embedding_model.encode(query).tolist()
   results = collection.query(query_embeddings=[query_embedding], n_results=3)
   retrieved_chunks = [res["text"] for res in results["metadatas"][0]]
   context = "\n".join(retrieved_chunks)
   ```

   ✅ Purpose:
   - Encodes the query into a semantic vector.
   - Searches a vector database (e.g., ChromaDB, FAISS, Pinecone) to retrieve top-matching documents or text chunks.

   ✅ Why:
   Retrieves the most relevant information from a knowledge base to provide accurate context for answering the query.

---

5. ✍️ Generate the Answer using Phi-2 Model

   ✅ Code:
   ```python
   prompt = f"""Use the following context to answer the question below:

   Context:
   {context}

   Question:
   {query}

   Answer:"""
   ```

   ✅ Purpose:
   - Builds a structured prompt with context + question to guide the language model.
   - Tokenizes the prompt and feeds it into Phi-2.

   ✅ Why:
   Phi-2 needs a clear prompt with relevant background to generate focused answers.

---

6. 🗣️ Model Inference & Output

   ✅ Code:
   ```python
   inputs = tokenizer(prompt, return_tensors="pt").to(phi_model.device)
   outputs = phi_model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
   response = tokenizer.decode(outputs[0], skip_special_tokens=True)
   answer = response.split("Answer:")[-1].strip()
   print("\n🤖 Chatbot Answer:", answer)
   ```

   ✅ Purpose:
   - Performs model inference.
   - Extracts only the answer portion of the generated output.
   - Displays it to the user.

   ✅ Why:
   Ensures clean and human-readable output.

---

7. 📝 Collect Feedback

   ✅ Code:
   ```python
   feedback = input("💬 Was this answer helpful? (yes/no): ").strip().lower()
   feedback_log.append({
       "question": query,
       "answer": answer,
       "feedback": feedback
   })
   ```

   ✅ Purpose:
   - Collects user feedback on the chatbot's response.
   - Stores it in a list for later evaluation or analysis.

   ✅ Why:
   Useful for improving model performance, training future models, or tuning response generation.

---

🧠 Technologies & Concepts Used:

| Component                      | Purpose                                               |
|--------------------------------|--------------------------------------------------------|
| SentenceTransformer            | Convert user query into vector form for semantic search |
| Vector Store (e.g., Chroma/FAISS) | Store document embeddings and retrieve relevant content |
| Transformers (Phi-2)           | Generate human-like responses using context-aware prompts |
| Prompt Engineering             | Create structured inputs for accurate generation    |
| Feedback Mechanism             | Capture user satisfaction to improve chatbot behavior |

---

📦 Potential Use Cases:

- Customer Support Bots
- Personalized Knowledge Assistants
- Document Q&A Systems (PDFs, websites)
- Research Summarization Assistants
- Contextual Help in Education or Software

---

📌 Next Steps (Optional Enhancements):

- Save feedback_log to CSV or JSON
- Add PDF/URL ingestion to feed the vector DB (already included in earlier code)
- Use LangChain or Haystack for modular pipeline building
- Build a frontend with Streamlit or Gradio
- Deploy the chatbot as a web or mobile app

---


