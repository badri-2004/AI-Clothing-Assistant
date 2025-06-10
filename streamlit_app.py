import streamlit as st
import os
import warnings
import torch
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
# MUST be the first Streamlit call
st.set_page_config(
    page_title="Clothing Store AI Assistant",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Imports ---
import sys
from pathlib import Path
import tempfile
import json
from dataclasses import dataclass
from typing import Literal, Optional, List, Dict
torch.classes.__path__ = []
# Fix the import path issue
project_root = Path(__file__).parent.resolve()
src_path = project_root / "clothing_store_assistant" / "src" / "clothing_store_assistant"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Now import your CrewAI class
try:
    from crews.ecommerce_crew.crew import EcommerceSearchCrew

    CREW_AVAILABLE = True
except ImportError as e:
    CREW_AVAILABLE = False
    import_error_message = f"CrewAI import error: {e}"


@dataclass
class Message:
    origin: Literal["human", "ai"]
    content: Dict


def load_css():
    st.markdown("""
    <style>
    .chat-row { display: flex; margin: 1rem 0; gap: 1rem; align-items: flex-start; }
    .row-reverse { flex-direction: row-reverse; }
    .chat-bubble { padding: 1rem; border-radius: 1rem; max-width: 70%; word-wrap: break-word; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .human-bubble { background-color: #f0f2f6; margin-left: auto; border: 1px solid #e1e4ea; }
    .ai-bubble { background-color: #e3f2fd; border: 1px solid #bbdefb; }
    .product-card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; background: white; margin: 0.5rem 0; }
    .product-card:hover { transform: translateY(-3px); box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
    .product-name { font-weight: bold; color: #1a73e8; margin-bottom: 0.5rem; font-size: 1.1em; }
    .product-id { color: #666; font-size: 0.9em; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()


def process_crew_inputs(text_query: str, uploaded_image):
    """Process inputs using the chat RAG crew routing system"""

    # Handle image upload
    image_path = ""
    if uploaded_image is not None:
        try:
            image_path = os.path.join(st.session_state.temp_dir, "uploaded_image.jpg")
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getvalue())
        except Exception as e:
            st.error(f"Error saving uploaded image: {e}")
            image_path = ""

    try:
        from crews.ecommerce_crew.chat_rag_crew import main
        crew_ = EcommerceSearchCrew()
        print(image_path)
        result = str(main(image_path, text_query,crew_,10))
        return process_main_result(result)

    except ImportError as e:
        return {
            "message": "System error: Unable to load the chat RAG crew module. Please check your setup.",
            "products": [],
            "source": "import_error"
        }
    except Exception as e:
        return {
            "message": f"An error occurred while processing your request: {str(e)}",
            "products": [],
            "source": "processing_error"
        }


def process_main_result(result):
    """Process the result from main() function and standardize the response format"""
    try:
        # Result is always a string - either helpful message or JSON
        if isinstance(result, str):
            # Check if it's a delegation response
            if "delegate_to_ecommerce" in result.lower():
                return {
                    "message": "I'm having trouble processing your request. Please try rephrasing your question.",
                    "products": [],
                    "source": "routing_error"
                }

            # Try to parse as JSON (ecommerce response)
            if result.strip().startswith('{'):
                try:
                    parsed_result = json.loads(result)

                    message = parsed_result.get("message", "")
                    products = parsed_result.get("products", [])

                    # Handle nested JSON in message field (yesterday's pattern)
                    if isinstance(message, str) and message.strip().startswith('{'):
                        try:
                            nested_data = json.loads(message)
                            return {
                                "message": nested_data.get("message", message),
                                "products": nested_data.get("products", products),
                                "source": "ecommerce"
                            }
                        except json.JSONDecodeError:
                            # If nested parsing fails, use the original parsed data
                            return {
                                "message": message,
                                "products": products,
                                "source": "ecommerce"
                            }
                    else:
                        return {
                            "message": message,
                            "products": products,
                            "source": "ecommerce"
                        }

                except json.JSONDecodeError:
                    # Not valid JSON, treat as plain FAQ answer
                    return {
                        "message": result,
                        "products": [],
                        "source": "faq"
                    }
            else:
                # Plain string FAQ answer
                return {
                    "message": result,
                    "products": [],
                    "source": "faq"
                }

        # Fallback for non-string results (shouldn't happen based on your description)
        else:
            return {
                "message": str(result),
                "products": [],
                "source": "faq"
            }

    except Exception as e:
        return {
            "message": "I encountered an error while processing your request. Please try again.",
            "products": [],
            "source": "processing_error"
        }


def display_ai_response(response_json):
    """Display AI response with separated message and product images"""

    source = response_json.get("source", "unknown")

    # Handle FAQ responses (from PDF RAG)
    if source == "faq":
        st.markdown("### 📋 Deeplearners Fashion - Company Information")
        st.info("Response from our Knowledge Base")

        message = response_json.get("message", "")
        if message:
            clean_message = message.replace('\\n', '\n').replace('\\"', '"')
            st.markdown(clean_message)
        else:
            st.warning("No response message received.")
        return

    # Handle ecommerce/product responses or use existing logic
    message_text = ""
    products = []

    if isinstance(response_json, dict):
        raw_message = response_json.get("message", "")

        if isinstance(raw_message, str) and raw_message.startswith('{"'):
            try:
                nested_data = json.loads(raw_message)
                message_text = nested_data.get("message", "")
                products = nested_data.get("products", [])
            except json.JSONDecodeError:
                message_text = raw_message
                products = []
        else:
            message_text = raw_message
            products = response_json.get("products", [])

    # Display the message separately first
    if message_text:
        if source == "ecommerce":
            st.markdown("### 🛍️ Product Recommendations")
        else:
            st.markdown("### 💬 AI Response")

        clean_message = message_text.replace('\\n', '\n').replace('\\"', '"')
        st.markdown(clean_message)
        st.divider()

    # Display products in a grid layout
    if products and len(products) > 0:
        st.markdown("### 🛍️ Product Recommendations")
        st.markdown(f"*Found {len(products)} matching items*")

        for i in range(0, len(products), 3):
            cols = st.columns(3)

            for j in range(3):
                if i + j < len(products):
                    product = products[i + j]

                    with cols[j]:
                        product_name = product.get('product_name', 'Unnamed Product')
                        product_id = product.get('product_id', 'N/A')
                        product_link = product.get('link', '')

                        if product_link:
                            try:
                                st.image(
                                    product_link,
                                    caption=product_name,
                                    use_container_width=True
                                )

                                st.markdown(f"**{product_name}**")
                                st.caption(f"Product ID: {product_id}")

                                st.link_button(
                                    "🔍 View Full Size",
                                    product_link,
                                    use_container_width=True
                                )

                            except Exception as e:
                                st.error(f"Could not load image: {product_name}")
                                st.code(f"Image URL: {product_link}")
                        else:
                            st.warning(f"No image available for {product_name}")

                        st.markdown("---")
    else:
        if message_text and source == "ecommerce":
            st.info("No products found in this response.")


def on_click_callback():
    human_prompt = st.session_state.human_prompt
    uploaded_image = st.session_state.image_input
    if not human_prompt.strip() and not uploaded_image:
        return

    user_content = {"text": human_prompt}
    if uploaded_image:
        user_content["image"] = "uploaded"
    st.session_state.history.append(Message("human", user_content))

    with st.spinner("🤖 AI is processing your request..."):
        ai_response = process_crew_inputs(human_prompt, uploaded_image)
        st.session_state.history.append(Message("ai", ai_response))


def main():
    load_css()
    initialize_session_state()

    st.title("👗 Deeplearners Fashion Assistant")
    st.markdown("*Your AI-powered fashion companion for style advice and company information*")

    st.divider()

    with st.sidebar:
        st.header("🛠️ How to Use")
        st.markdown("### 💡 Ask about:")
        st.markdown("- **Company policies**: Returns, shipping, payments")
        st.markdown("- **Style advice**: Product recommendations")
        st.markdown("- **Upload images**: Find similar items")
        st.markdown("- **Company info**: Location, contact details")

        st.divider()
        st.markdown("### 📞 Need help?")
        st.markdown("Email: support@deeplearnersfashion.com")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.markdown("## 💬 Fashion Consultation")
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.history:
            if msg.origin == "human":
                with st.chat_message("human", avatar="👤"):
                    st.write(msg.content.get('text', ''))
                    if msg.content.get("image"):
                        st.success("📸 Image uploaded with this message")
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    display_ai_response(msg.content)

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text_area("Ask a question or describe what you're looking for...",
                         key="human_prompt", height=100,
                         placeholder="e.g., 'What is your return policy?' or 'Show me summer dresses'")
        with col2:
            st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="image_input")
            st.form_submit_button("Send 🚀", on_click=on_click_callback, use_container_width=True)
            if st.session_state.get("image_input"):
                st.image(st.session_state["image_input"], caption="Uploaded Image", width=150)

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #666; padding: 1rem;'>"
                "Powered by CrewAI 🤖 | Built with Streamlit ⚡ | Deeplearners Fashion 👗"
                "</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
