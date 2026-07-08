from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="materialsim-ai-agent",
    version="0.1.0",
    author="MaterialSim AI Agent Team",
    author_email="awwalola@umich.edu",
    description="An autonomous LLM agent for computational materials science and molecular dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Awwal41/MaterialSim",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "gui": ["voice_stt/frontend/*"],
    },
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
        # Materials databases (Materials Project) + structure interop.
        "database": [
            "pymatgen>=2023.8.10",
            "mp-api>=0.33.0",
        ],
        # Molecules/polymers from SMILES (RDKit).
        "structures": [
            "rdkit>=2023.9.1",
        ],
        # Universal machine-learned interatomic potentials (need PyTorch).
        "mlip": [
            "torch>=2.1.0",
            "mace-torch>=0.3.6",
            "chgnet>=0.3.0",
            "matgl>=1.1.0",
        ],
        # OpenMM engine + all-atom bonded force fields (OPLS/GAFF/OpenFF).
        # NOTE: OpenMM installs most reliably via conda-forge.
        "openmm": [
            "openmm>=8.1.0",
            "openmmforcefields>=0.12.0",
            "openff-toolkit>=0.15.0",
        ],
        # Interactive 3D structure/trajectory viewer for the GUI workspace.
        "viewer": [
            "py3Dmol>=2.0.4",
            "stmol>=0.0.9",
        ],
        # Voice / JARVIS GUI extras.
        "voice": [
            "edge-tts>=6.1.0",
            "SpeechRecognition>=3.10.0",
            "pydub>=0.25.1",
            "ddgs>=9.0.0",
        ],
        # Property-prediction ML + extra plotting/storage.
        "ml": [
            "scikit-learn>=1.3.2",
            "seaborn>=0.12.2",
            "plotly>=5.17.0",
            "h5py>=3.9.0",
        ],
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
        # Everything pip-installable at once (LAMMPS binary still separate).
        "all": [
            "pymatgen>=2023.8.10",
            "mp-api>=0.33.0",
            "rdkit>=2023.9.1",
            "torch>=2.1.0",
            "mace-torch>=0.3.6",
            "chgnet>=0.3.0",
            "matgl>=1.1.0",
            "openmm>=8.1.0",
            "openmmforcefields>=0.12.0",
            "openff-toolkit>=0.15.0",
            "scikit-learn>=1.3.2",
            "seaborn>=0.12.2",
            "plotly>=5.17.0",
            "h5py>=3.9.0",
            "py3Dmol>=2.0.4",
            "stmol>=0.0.9",
            "edge-tts>=6.1.0",
            "SpeechRecognition>=3.10.0",
            "pydub>=0.25.1",
            "ddgs>=9.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "materials-agent=materials_ai_agent.cli:main",
        ],
    },
)
