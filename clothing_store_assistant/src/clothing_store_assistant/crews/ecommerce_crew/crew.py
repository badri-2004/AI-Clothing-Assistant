from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from  clothing_store_assistant.src.clothing_store_assistant.tools.text_query_tool import TextQueryTool
from clothing_store_assistant.src.clothing_store_assistant.tools.vision_tool import GeminiClothingDescriptorTool
from dotenv import load_dotenv


load_dotenv()
@CrewBase
class EcommerceSearchCrew:
    """E-commerce multimodal search crew."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        # Initialize tools
        self.text_tool = TextQueryTool()

    @agent
    def Query_Analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['Query_Analyzer'],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def Fashion_Expert(self) -> Agent:
        return Agent(
            config=self.agents_config['Fashion_Expert'],
            tools=[GeminiClothingDescriptorTool()],
            verbose=True,
            allow_delegation=False,

        )

    @agent
    def RAG_Query_Expert(self) -> Agent:
        return Agent(
            config=self.agents_config['RAG_Query_Expert'],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def RAG_Query_Retriever(self) -> Agent:
        return Agent(
            config=self.agents_config['RAG_Query_Retriever'],
            tools=[self.text_tool],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def Verifier(self) -> Agent:
        return Agent(
            config=self.agents_config['Verifier'],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def results_presenter(self) -> Agent:
        return Agent(
            config=self.agents_config['results_presenter'],
            verbose=True,
            allow_delegation=False,
        )

    @task
    def analyze_query_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_query_task'],
            agent=self.Query_Analyzer()
        )

    @task
    def Fashion_Suggestion_Task(self) -> Task:
        return Task(
            config=self.tasks_config['Fashion_Suggestion_Task'],
            agent=self.Fashion_Expert(),
        )

    @task
    def RAG_Query_Generation(self) -> Task:
        return Task(
            config=self.tasks_config['RAG_Query_Generation'],
            agent=self.RAG_Query_Expert(),
        )

    @task
    def RAG_Query_Retrieval(self) -> Task:
        return Task(
            config=self.tasks_config['RAG_Query_Retrieval'],
            agent=self.RAG_Query_Retriever(),
        )

    @task
    def Verification_Task(self) -> Task:
        return Task(
            config=self.tasks_config['Verification_Task'],
            agent=self.Verifier(),
        )

    @task
    def present_results_task(self) -> Task:
        return Task(
            config=self.tasks_config['present_results_task'],
            agent=self.results_presenter(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.Query_Analyzer(),
                self.Fashion_Expert(),
                self.RAG_Query_Expert(),
                self.RAG_Query_Retriever(),
                self.Verifier(),
                self.results_presenter()
            ],
            tasks=[
                self.analyze_query_task(),
                self.Fashion_Suggestion_Task(),
                self.RAG_Query_Generation(),
                self.RAG_Query_Retrieval(),
                self.Verification_Task(),
                self.present_results_task()
            ],
            process=Process.sequential,
            verbose=True
        )

