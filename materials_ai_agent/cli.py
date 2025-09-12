"""Command-line interface for Materials AI Agent."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .core.agent import MaterialsAgent
from .core.config import Config
from .core.exceptions import MaterialsAgentError


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Materials AI Agent - Autonomous computational materials science agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  materials-agent run "Simulate silicon thermal conductivity at 300K"
  materials-agent analyze ./simulations/silicon_300K/
  materials-agent predict "Al2O3" --properties "elastic_modulus,thermal_conductivity"
  materials-agent query "silicon band gap"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run simulation command
    run_parser = subparsers.add_parser('run', help='Run a simulation')
    run_parser.add_argument('instruction', help='Natural language simulation instruction')
    run_parser.add_argument('--output-dir', '-o', help='Output directory for simulation')
    run_parser.add_argument('--config', '-c', help='Path to configuration file')
    
    # Analyze results command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze simulation results')
    analyze_parser.add_argument('simulation_path', help='Path to simulation directory')
    analyze_parser.add_argument('--properties', nargs='+', help='Properties to compute')
    analyze_parser.add_argument('--config', '-c', help='Path to configuration file')
    
    # Predict properties command
    predict_parser = subparsers.add_parser('predict', help='Predict material properties')
    predict_parser.add_argument('material', help='Material formula')
    predict_parser.add_argument('--properties', nargs='+', required=True, 
                               help='Properties to predict')
    predict_parser.add_argument('--config', '-c', help='Path to configuration file')
    
    # Query database command
    query_parser = subparsers.add_parser('query', help='Query materials database')
    query_parser.add_argument('query', help='Database query')
    query_parser.add_argument('--database', choices=['mp', 'nomad'], default='mp',
                             help='Database to query')
    query_parser.add_argument('--config', '-c', help='Path to configuration file')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    interactive_parser.add_argument('--config', '-c', help='Path to configuration file')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    args = parser.parse_args()
    
    if args.command == 'version':
        print(f"Materials AI Agent v{__import__('materials_ai_agent').__version__}")
        return
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Load configuration
        config = Config.from_env()
        if args.config:
            # Load from file if specified
            config = Config.from_file(args.config)
        
        # Initialize agent
        agent = MaterialsAgent(config)
        
        # Execute command
        if args.command == 'run':
            result = agent.run_simulation(args.instruction)
            print_result(result)
            
        elif args.command == 'analyze':
            properties = args.properties or ['rdf', 'msd', 'elastic']
            result = agent.analyze_results(args.simulation_path)
            print_result(result)
            
        elif args.command == 'predict':
            result = agent.predict_properties(args.material, args.properties)
            print_result(result)
            
        elif args.command == 'query':
            result = agent.query_database(args.query)
            print_result(result)
            
        elif args.command == 'interactive':
            run_interactive_mode(agent)
            
    except MaterialsAgentError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def print_result(result: dict):
    """Print formatted result."""
    if result.get('success', False):
        print("✓ Success!")
        if 'result' in result:
            print(result['result'])
        elif 'analysis' in result:
            print(result['analysis'])
        elif 'predictions' in result:
            print(result['predictions'])
        elif 'results' in result:
            print(result['results'])
    else:
        print("✗ Failed!")
        if 'error' in result:
            print(f"Error: {result['error']}")


def run_interactive_mode(agent: MaterialsAgent):
    """Run interactive mode."""
    print("Materials AI Agent - Interactive Mode")
    print("Type 'help' for commands, 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'help':
                print_help()
            elif user_input.startswith('run '):
                instruction = user_input[4:].strip()
                result = agent.run_simulation(instruction)
                print_result(result)
            elif user_input.startswith('analyze '):
                path = user_input[8:].strip()
                result = agent.analyze_results(path)
                print_result(result)
            elif user_input.startswith('predict '):
                parts = user_input[8:].strip().split()
                if len(parts) >= 2:
                    material = parts[0]
                    properties = parts[1:]
                    result = agent.predict_properties(material, properties)
                    print_result(result)
                else:
                    print("Usage: predict <material> <property1> <property2> ...")
            elif user_input.startswith('query '):
                query = user_input[6:].strip()
                result = agent.query_database(query)
                print_result(result)
            else:
                # General chat
                response = agent.chat(user_input)
                print(response)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def print_help():
    """Print help information."""
    help_text = """
Available commands:
  run <instruction>     - Run a simulation (e.g., "run simulate silicon at 300K")
  analyze <path>        - Analyze simulation results
  predict <material> <properties> - Predict material properties
  query <query>         - Query materials database
  help                  - Show this help
  quit/exit/q          - Exit interactive mode

Examples:
  run simulate the thermal conductivity of silicon at 300 K using Tersoff potential
  analyze ./simulations/silicon_300K/
  predict Al2O3 elastic_modulus thermal_conductivity
  query silicon band gap
    """
    print(help_text)


if __name__ == '__main__':
    main()
