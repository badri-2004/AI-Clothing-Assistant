# 👕 AI Clothing Store Assistant

This project is an intelligent fashion shopping assistant powered by [CrewAI](https://docs.crewai.com), [Gemini Pro / Flash](https://ai.google.dev/gemini-api/docs/overview), [ChromaDB](https://www.trychroma.com/), and [Streamlit](https://streamlit.io/). Users can ask fashion-related queries, upload outfit images, or get company FAQs answered — all from a friendly assistant interface.

---

## ✨ Features

- 💬 Natural language query support (e.g., "Show me similar shirts in blue")
- 🧠 **Query routing** via CrewAI (decides between company FAQs or product recommendations)
- 🖼️ **Image-based product understanding** using Gemini 2.0 Flash
- 🔍 **Text-based retrieval** using `all-mpnet-base-v2` and ChromaDB vector search
- ✅ Sanity-checked results (gender and clothing-type match only)
- 🛍️ Final product suggestions are presented in clean conversational format

---

Working Deployed link : [Deployed on streamlit](https://ai-clothing-assistant-imb4hcwifbmswfcfr6cvwf.streamlit.app/)

## 🧠 CrewAI Agents Overview

### 📌 Crew 1: `ChatRAGCrew` (Router)
Handles:
- Company FAQs
- Returns / shipping / location info
- Friendly chat
- Delegates fashion queries to ecommerce crew

Uses Gemini 1.5 via TXTSearchTool for PDF/TXT-based retrieval.

### 🛍️ Crew 2: `EcommerceCrew`
Handles:
- Fashion product discovery
- Combines Gemini Vision + vector search + filtering

Flow:
1. **GeminiClothingDescriptorTool**: Transcribes image to detailed description using Gemini 2.0 Flash
2. **Query Analyzer**: Rewrites query
3. **Fashion Expert**: Suggests appropriate item type
4. **RAG Query Expert**: Compresses for vector search
5. **Retriever + Verifier**: Fetches and sanity-checks based on gender/type
6. **Presenter**: Formats results conversationally

---

## 🖼️ Image Processing (NO Embeddings!)

We **do not use DINO or CLIP** embeddings.  
Instead, we **transcribe the image into a fashion description** using:

```python
model="gemini-2.0-flash"
```

via `GeminiClothingDescriptorTool`. It extracts:
- Garment type
- Pattern, fit, color, texture
- Gender orientation
- Season/usage

---

## 🧪 Example Usage

```python
# chat_RAG_combined.py
from your_module import main, EcommerceSearchCrew

result = main(
  image_path="images/green_tshirt.jpg",
  text_query="I want this in black",
  crew_=EcommerceSearchCrew(),  # Your ecommerce crew logic
  top_k=5
)
print(result)
```

Or use the **Streamlit UI**:

```bash
streamlit run streamlit_app.py
```

You can:
- Upload an image (e.g., of a shirt)
- Type: "Show me something like this but for women"
- See refined suggestions

---

## 🛠️ Setup Instructions

1. Clone the repo

```bash
git clone https://github.com/your_username/ai-clothing-store-assistant.git
cd ai-clothing-store-assistant
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Setup `.env` file

```env
GEMINI_API_KEY=your_api_key_here
```

4. Build the vector store

```bash
python vectorstore_maker.py
```

5. Launch the app

```bash
streamlit run streamlit_app.py
```

---

## 🔒 Notes

- This project uses only **Gemini APIs** — no OpenAI, CLIP, or DINO models.
- Retrieval is **text-only**, enriched with Gemini’s image-to-text descriptions.

---

## 📌 Sample Query Ideas

| Upload Image | Text Query                 | Result                                           |
|--------------|----------------------------|--------------------------------------------------|
| Red T-shirt  | "This in blue"             | Returns blue t-shirts                           |
| None         | "Suggest Formal Attire"    | Returns Formal attire                           |
| None         | "Beige summer shorts"      | Returns Beige Summer Shorts                     |
| Skirt        | "I want pants like this"   | Skips image and focuses on pants (text-only)    |
| Hoodie       | "Something similar"        | Uses both image + text for combo search         |

---

## 🤝 Credits

- 🧠 CrewAI: Agent-based control flow
- 🔎 ChromaDB: Fast vector store
- 🌈 Gemini Vision: Rich fashion understanding
- 🖼️ Streamlit: UI interface

---

## 📷 Screenshots
![Main Screen](https://github.com/badri-2004/AI-Clothing-Assistant/blob/main/Screenshot%202025-06-10%20184934.png)
![Example Usage](https://github.com/badri-2004/AI-Clothing-Assistant/blob/main/screen-capture2-ezgif.com-optimize.gif)

