from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from doctor_patient_dialogue.tools.dialogue_parser import DialogueParserTool
from doctor_patient_dialogue.tools.subjective_tool import SubjectiveExtractorTool
from doctor_patient_dialogue.tools.objective_tool import ObjectiveExtractorTool
from doctor_patient_dialogue.tools.history_subjective_tool import HistorySubjectiveExtractorTool





@CrewBase
class DoctorPatientDialogue():
    """Doctor–Patient Dialogue Parsing Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def dialogue_parser_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['dialogue_parser_agent'],
            tools=[DialogueParserTool()],
            verbose=True,
            allow_delegation=False,
            max_iter=1  # 🔑 THIS IS KEY
        )

    @task
    def parse_dialogue_task(self) -> Task:
        return Task(
            config=self.tasks_config['parse_dialogue_task']
        )
    
    @agent
    def subjective_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["subjective_agent"],
            tools=[SubjectiveExtractorTool()],
            verbose=True,
            max_iter=1,
            allow_delegation=False
        )

    @task
    def subjective_task(self) -> Task:
        return Task(
            config=self.tasks_config["subjective_task"]
        )
    
    @agent
    def history_subjective_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["history_subjective_agent"],
            tools=[HistorySubjectiveExtractorTool()],
            verbose=True,
            max_iter=1,
            allow_delegation=False
        )
    
    @task
    def history_subjective_task(self) -> Task:
        return Task(
            config=self.tasks_config["history_subjective_task"]
        )


    
    @agent
    def objective_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["objective_agent"],
            tools=[ObjectiveExtractorTool()],
            verbose=True,
            max_iter=1,
            allow_delegation=False
        )
    
    @task
    def objective_task(self) -> Task:
        return Task(
            config=self.tasks_config["objective_task"]
        )
    
    @agent
    def assessment_plan_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["assessment_plan_agent"],
            memory=False,
            max_iter=2,
            verbose=True
        )

    @task
    def assessment_plan_task(self) -> Task:
        return Task(
            config=self.tasks_config["assessment_plan_task"]
        )



    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            max_iterations=1,
        )
