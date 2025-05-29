
## Great question — if you're asked in an interview about challenges while building a RAG-based chatbot, and you want to explain the issue you faced (i.e., *old embeddings staying in the vector DB*), here's a **professional and concise way** to present it

---

### ✅ **Sample Answer:**

> While building a RAG-based chatbot using PDF and URL inputs, one of the key challenges I faced was managing the vector database effectively — specifically, **handling stale or outdated embeddings**.
>
> For example, if I first uploaded a Machine Learning PDF, it got embedded and stored correctly in the vector database. However, when I later uploaded a medical-related document or a URL, the chatbot still retrieved responses based on the earlier Machine Learning content.
>
> The issue was that the **previous embeddings were not cleared**, so the system mixed context from different sources, leading to **irrelevant or misleading answers**.
>
> I solved this by explicitly clearing the existing data in the vector store before storing new embeddings using `collection.delete(where={})`. This ensured that each session worked only with the latest uploaded content, improving response accuracy.

---


## That’s a smart point to bring up in an interview — **resource constraints** are a real-world challenge, and showing how you handled them demonstrates both **practical thinking** and **adaptability**.

---

### ✅ **Sample Answer (Resource Constraint & Model Choice):**

> While working on a Retrieval-Augmented Generation (RAG) chatbot project, I faced **hardware limitations** on my personal machine — I had 8GB RAM and no high-end GPU.
>
> This made it challenging to run large-scale LLMs like GPT-style models locally. To overcome this, I opted for **Phi-2**, a lightweight and efficient LLM by Microsoft that's optimized for resource-constrained environments. It worked well on my setup, especially when paired with techniques like mixed precision (`torch_dtype=torch.float16`) and `device_map="auto"` for efficient memory usage.
>
> This choice allowed me to **prototype and test locally** without relying on expensive cloud resources, while still maintaining acceptable response quality.

---
