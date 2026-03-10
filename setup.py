from setuptools import setup, find_packages

setup(
    name="areal",
    version="0.1.0",
    description="AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning",
    author="AReaL Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.38.0",
        "accelerate>=0.27.0",
        "wandb>=0.16.0",
        "sglang>=0.1.13",
        "vllm>=0.3.3",
        "ray>=2.9.0",
        "Pillow",
        "numpy",
        "scipy",
        "typer",
        "pydantic",
        "megatron-core>=0.6.0",  # Note: Often needs manual install or specific index
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
        ],
    },
)
