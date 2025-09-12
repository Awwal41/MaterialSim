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
from ..tools.simulation import create_simulation_tools


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
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
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
        return []
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent with tools and memory."""
       
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

When asked to run simulations, you MUST use the available tools to actually execute the simulation, not just provide instructions.

Available tools:
- simulation: For setting up and running MD simulations
- analysis: For computing materials properties from simulation data
- database: For querying materials databases
- ml: For ML-based property prediction
- visualization: For creating plots and reports

IMPORTANT: When a user asks you to run a simulation, you MUST use the simulation tool to actually execute it, not just provide instructions or code.
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        
        try:
            agent = create_openai_tools_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True
            )
            
            return agent_executor
            
        except Exception as e:
            self.logger.warning(f"Failed to create agent with tools: {e}. Falling back to simple LLM.")
            
            
            from langchain.chains import LLMChain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
           
            class SimpleAgentExecutor:
                def __init__(self, chain, memory):
                    self.chain = chain
                    self.memory = memory
                
                def invoke(self, inputs):
                   
                    chat_history = self.memory.chat_memory.messages
                    
                    
                    formatted_input = {
                        "input": inputs["input"],
                        "chat_history": chat_history
                    }
                    
                    
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
            
            material, temperature, force_field, n_steps = self._parse_simulation_instruction(instruction)
            
            from ..simple_simulation import run_simple_simulation
            
         
            result = run_simple_simulation(
                material=material,
                temperature=temperature,
                n_steps=n_steps,
                force_field=force_field
            )
            
            if result["success"]:
                return f"✅ {result['message']}\n\nSimulation completed successfully!\nDirectory: {result['simulation_directory']}\nOutput files: {result['output_files']}"
            else:
                return f"❌ Simulation failed: {result['error']}"
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {str(e)}")
            return f"Error: {str(e)}"
    
    def _parse_simulation_instruction(self, instruction: str) -> tuple:
        """Parse simulation instruction to extract parameters.
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            Tuple of (material, temperature, force_field, n_steps)
        """
        from .materials_database import MaterialsDatabase
        
        instruction_lower = instruction.lower()
        materials_db = MaterialsDatabase()
        
        
        material = None
        for formula, props in materials_db.get_all_materials().items():
            if (formula.lower() in instruction_lower or 
                props.description.lower() in instruction_lower or
                any(alias in instruction_lower for alias in [formula.lower(), props.formula.lower()])):
                material = formula
                break
        
        # Fallback to simple keyword matching
        if not material:
            if "silicon" in instruction_lower or "si" in instruction_lower:
                material = "Si"
            elif "aluminum" in instruction_lower or "al" in instruction_lower:
                material = "Al"
            elif "copper" in instruction_lower or "cu" in instruction_lower:
                material = "Cu"
            elif "iron" in instruction_lower or "fe" in instruction_lower:
                material = "Fe"
            elif "water" in instruction_lower or "h2o" in instruction_lower:
                material = "H2O"
            else:
                material = "Si"  # Default fallback
        
        # Extract temperature
        temperature = self.config.default_temperature
        import re
        temp_match = re.search(r'(\d+)\s*k', instruction_lower)
        if temp_match:
            temperature = float(temp_match.group(1))
            # Ensure temperature is within limits
            temperature = max(self.config.min_temperature, min(temperature, self.config.max_temperature))
        
        # Extract force field
        force_field = self.config.default_force_field
        for ff in self.config.available_force_fields:
            if ff.lower() in instruction_lower:
                force_field = ff
                break
        
        # Extract number of steps
        n_steps = self.config.default_n_steps
        steps_match = re.search(r'(\d+)\s*steps?', instruction_lower)
        if steps_match:
            n_steps = int(steps_match.group(1))
            # Ensure n_steps is within limits
            n_steps = max(self.config.min_n_steps, min(n_steps, self.config.max_n_steps))
        
        return material, temperature, force_field, n_steps
    
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
