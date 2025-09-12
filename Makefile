# Makefile for Materials AI Agent

.PHONY: help install test clean build docs examples gui test-gui install-gui

help:
	@echo "Materials AI Agent - Available Commands"
	@echo "========================================"
	@echo "install    - Install dependencies and package"
	@echo "install-gui - Install GUI dependencies"
	@echo "test       - Run test suite"
	@echo "test-gui   - Test GUI components"
	@echo "gui        - Launch web GUI"
	@echo "gui-v2     - Launch enhanced conversational GUI"
	@echo "gui-integrated - Launch fully integrated GUI"
	@echo "build      - Build the package"
	@echo "docs       - Generate documentation"
	@echo "examples   - Run example scripts"
	@echo "clean      - Clean build artifacts"
	@echo "lint       - Run linting checks"
	@echo "format     - Format code with black"

install:
	@echo "Installing Materials AI Agent..."
	pip install -r requirements.txt
	pip install -e .

install-gui:
	@echo "Installing GUI dependencies..."
	pip install -r requirements_gui.txt

test:
	@echo "Running tests..."
	python -m pytest tests/ -v

test-gui:
	@echo "Testing GUI components..."
	python test_gui.py

gui:
	@echo "Launching web GUI..."
	python launch_gui.py

gui-v2:
	@echo "Launching enhanced conversational GUI..."
	python launch_gui_v2.py

gui-integrated:
	@echo "Launching fully integrated GUI..."
	python launch_integrated_gui.py

build:
	@echo "Building package..."
	python build.py

docs:
	@echo "Generating documentation..."
	@if [ -d "docs/source" ]; then \
		cd docs && make html; \
	else \
		echo "Documentation already generated in docs/"; \
	fi

examples:
	@echo "Running examples..."
	python examples/basic_simulation.py
	python examples/ml_training_example.py

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name ".pytest_cache" -delete
	rm -rf .coverage
	rm -rf htmlcov/

lint:
	@echo "Running linting checks..."
	flake8 materials_ai_agent/ tests/ examples/
	black --check materials_ai_agent/ tests/ examples/

format:
	@echo "Formatting code..."
	black materials_ai_agent/ tests/ examples/

setup-dev:
	@echo "Setting up development environment..."
	python build.py
	make install
	make test

ci:
	@echo "Running CI pipeline..."
	make lint
	make test
	make examples

all: clean install test build docs examples
