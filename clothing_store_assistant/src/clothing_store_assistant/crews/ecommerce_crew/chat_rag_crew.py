# File: chat_RAG_combined.py
from crewai import Agent, Crew, Task, Process, LLM
from crewai_tools import TXTSearchTool
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer

load_dotenv()

# Setup Gemini LLM for PDF RAG tool
# PDF RAG Tool using Gemini
pdf_tool = TXTSearchTool(
    config=dict(
        llm=dict(
            provider="google",
            config=dict(
                model="gemini-1.5-pro-latest",
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.7,
            ),
        ),
        embedder=dict(
            provider="google",
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                title="Deeplearners Fashion Knowledge Base",
            ),
        ),
    ),
    txt="C:/Users/badri/PycharmProjects/PythonProject4/clothing_store_assistant/src/clothing_store_assistant/crews/ecommerce_crew/company_faqs.txt",
)

class ChatRAGCrew:
    def __init__(self):
        self.router_agent = Agent(
            role="Smart Customer Service Router",
            goal=(
                "Serve as the primary customer service interface for Deeplearners Fashion. "
                "Answer FAQs from the company document, engage in friendly chitchat, and only delegate "
                "to the ecommerce agent when users explicitly ask about fashion products, dresses, "
                "style suggestions, or upload an image."
            ),
            backstory=(
                "You are a helpful, friendly, and intelligent customer support representative at Deeplearners Fashion. "
                "You're trained to understand when a customer just wants to chat, get help with their order, or "
                "learn about company policies. You can access the Comprehensive guide TXT document to answer company-related queries. "
                "Only when the user clearly asks for specific products, dresses, outfits, or uploads an image, "
                "you delegate to the specialized fashion product agent. "
                "Always pass user queries to the TXT search tool as plain strings, not objects or dictionaries."
            ),
            tools=[pdf_tool],
            verbose=True,
            allow_delegation=False,  # Prevent auto-delegation; we control this manually
        )

        self.routing_task = Task(
            description=(
                "Analyze the user's query: '{text_query}'\n\n"

                "→ If the query is about returns, shipping, account management, product quality, "
                "or general company policies or any information about the company like ceo,location,contact etc, **rephrase it** into a clearer and more formal question, "
                "then search the Comprehensive guide TXT using the rephrased query string.\n\n"

                "→ If the user uploaded an image of a **damaged or defective item** (like a torn dress) "
                "and is complaining about product quality, respond with an apology and assistance. "
                "**Do not delegate to ecommerce**.\n\n"

                "→ If the query is about product recommendations, style tips, or discovering similar items "
                "and involves an uploaded image or fashion intent, respond with exactly: `'delegate_to_ecommerce'`.\n\n"

                "→ If the user initiates friendly conversation or small talk (e.g., 'hello', 'how are you?'), "
                "respond warmly and naturally.\n\n"

                "**Always pass the rephrased query string**  when using the TXT search tool and not any other type like dict. Simple plain string.\n"
                "YOU MIGHT NOT ALWAYS GET THE ANSWER IN FIRST TRY TRY FOR 4-5 Times WITH DIFFERENT REPHRASED QUERIES BEFORE GIVING UP, SEARCH IN DIFFERENT SECTIONS AND SOMEHOW GET AN ANSWER"
            ),
            expected_output=(
                "Either a helpful natural language answer using the rephrased query, or exactly: 'delegate_to_ecommerce'."
            ),
            agent=self.router_agent,
        )

        self.crew = Crew(
            agents=[self.router_agent],
            tasks=[self.routing_task],
            process=Process.sequential,
            verbose=True,
        )

    def kickoff(self, inputs: dict):
        return self.crew.kickoff(inputs=inputs)


def main(image_path: str, text_query: str,crew_, top_k: int = 3,):
    print(image_path)
    print(type(image_path))
    routing_response = str(ChatRAGCrew().kickoff({"text_query": text_query}))

    if "delegate_to_ecommerce" in routing_response.lower():
        return crew_.crew().kickoff({
            "image_path": image_path,
            "text_query": text_query,
            "top_k": top_k
        })
    else:
        return routing_response