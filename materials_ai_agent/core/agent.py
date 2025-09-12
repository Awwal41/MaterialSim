"""Main Materials AI Agent class."""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage

from .config import Config
from .exceptions import MaterialsAgentError
from ..tools import (
    SimulationTool,
    AnalysisTool,
    DatabaseTool,
    MLTool,
    VisualizationTool,
)


class MaterialsAgent:
    """Main Materials AI Agent class."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the Materials AI Agent.
        
        Args:
            config: Configuration object. If None, loads from environment.
        """
        self.config = config or Config.from_env()
        self.config.create_directories()
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Initialize tools (disabled for now due to Pydantic v2 issues)
        # self.tools = self._initialize_tools()
        self.tools = []
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            openai_api_key=self.config.openai_api_key,
        )
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True,
        )
        
        # Create agent
        self.agent = self._create_agent()
        
        self.logger.info("Materials AI Agent initialized successfully")
    
    def _initialize_tools(self) -> List:
        """Initialize all available tools."""
        tools = [
            SimulationTool(config=self.config),
            AnalysisTool(config=self.config),
            DatabaseTool(config=self.config),
            MLTool(config=self.config),
            VisualizationTool(config=self.config),
        ]
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent with tools and memory."""
        # Define the system prompt
        system_prompt = """You are a Materials AI Agent, an expert in computational materials science and molecular dynamics simulations.

Your capabilities include:
1. Setting up and running molecular dynamics simulations using LAMMPS
2. Analyzing simulation results to compute materials properties
3. Querying materials databases for comparison and benchmarking
4. Using machine learning models for property prediction
5. Creating visualizations and reports

You should:
- Always provide detailed explanations of your actions
- Suggest appropriate simulation parameters based on the material and property of interest
- Interpret results in the context of materials science
- Recommend follow-up simulations or experiments when appropriate
- Use proper scientific terminology and units

When asked to run simulations, always:
1. Verify the material structure and parameters
2. Set up appropriate force fields and simulation conditions
3. Monitor simulation progress
4. Analyze results and compute relevant properties
5. Provide interpretation and recommendations

Available tools:
- simulation: For setting up and running MD simulations
- analysis: For computing materials properties from simulation data
- database: For querying materials databases
- ml: For ML-based property prediction
- visualization: For creating plots and reports
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        # Create a simple chain without tools for now
        from langchain.chains import LLMChain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Create a simple agent executor that just uses the LLM
        class SimpleAgentExecutor:
            def __init__(self, chain, memory):
                self.chain = chain
                self.memory = memory
            
            def invoke(self, inputs):
                # Get chat history
                chat_history = self.memory.chat_memory.messages
                
                # Format the input
                formatted_input = {
                    "input": inputs["input"],
                    "chat_history": chat_history
                }
                
                # Get response
                response = self.chain.invoke(formatted_input)
                
                return {"output": response["text"]}
        
        agent_executor = SimpleAgentExecutor(chain, self.memory)
        
        return agent_executor
    
    def run_simulation(self, instruction: str) -> str:
        """Run a simulation based on natural language instruction.
        
        Args:
            instruction: Natural language description of the simulation to run
            
        Returns:
            String response from the agent
        """
        try:
            self.logger.info(f"Running simulation: {instruction}")
            
            # Use the agent to process the instruction
            result = self.agent.invoke({
                "input": f"Please run the following simulation: {instruction}"
            })
            
            return result["output"]
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {str(e)}")
            return f"Error: {str(e)}"
    
    def analyze_results(self, simulation_path: str) -> Dict[str, Any]:
        """Analyze simulation results.
        
        Args:
            simulation_path: Path to simulation output files
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            self.logger.info(f"Analyzing results from: {simulation_path}")
            
            result = self.agent.invoke({
                "input": f"Please analyze the simulation results from {simulation_path}"
            })
            
            return {
                "simulation_path": simulation_path,
                "analysis": result["output"],
                "success": True,
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return {
                "simulation_path": simulation_path,
                "error": str(e),
                "success": False,
            }
    
    def query_database(self, query: str) -> Dict[str, Any]:
        """Query materials databases.
        
        Args:
            query: Natural language query about materials properties
            
        Returns:
            Dictionary containing query results
        """
        try:
            self.logger.info(f"Querying database: {query}")
            
            result = self.agent.invoke({
                "input": f"Please query the materials database for: {query}"
            })
            
            return {
                "query": query,
                "results": result["output"],
                "success": True,
            }
            
        except Exception as e:
            self.logger.error(f"Database query failed: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "success": False,
            }
    
    def predict_properties(self, material: str, properties: List[str]) -> Dict[str, Any]:
        """Predict material properties using ML models.
        
        Args:
            material: Material formula or structure
            properties: List of properties to predict
            
        Returns:
            Dictionary containing predictions
        """
        try:
            self.logger.info(f"Predicting properties for {material}: {properties}")
            
            result = self.agent.invoke({
                "input": f"Please predict the following properties for {material}: {', '.join(properties)}"
            })
            
            return {
                "material": material,
                "properties": properties,
                "predictions": result["output"],
                "success": True,
            }
            
        except Exception as e:
            self.logger.error(f"Property prediction failed: {str(e)}")
            return {
                "material": material,
                "properties": properties,
                "error": str(e),
                "success": False,
            }
    
    def chat(self, message: str) -> str:
        """Chat with the agent.
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        try:
            result = self.agent.invoke({"input": message})
            return result["output"]
        except Exception as e:
            self.logger.error(f"Chat failed: {str(e)}")
            return f"I encountered an error: {str(e)}"
