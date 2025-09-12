from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="materialsim-ai-agent",
    version="0.1.0",
    author="MaterialSim AI Agent Team",
    author_email="contact@materialsim-ai-agent.com",
    description="An autonomous LLM agent for computational materials science and molecular dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/materialsim-ai-agent/materialsim-ai-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "cupy-cuda11x>=12.0.0",
            "cudf>=23.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "materials-agent=materials_ai_agent.cli:main",
        ],
    },
)
