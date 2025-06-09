# Cart-Pole with QLoRA Optimisation

## Overview

This project fine-tunes a Large Language Model (LLM) using Proximal Policy Optimization (PPO) to learn a policy for the Cart-Pole environment from OpenAI Gym. Leveraging Unsloth optimizations, QLoRA, and additional techniques such as experience replay and memory-efficient training (via bitsandbytes and Triton), the model learns to balance the pole by selecting optimal actions based on a natural language–informed state prompt.

## Objective

The main objective of this project is to:

- **Train an LLM to play Cart-Pole:** Use reinforcement learning (PPO) to adapt a pre-trained language model so that it can predict the best actions (i.e., push left or right) to keep the pole balanced.
- **Utilize natural language prompts:** Frame environment states as detailed prompts to guide the LLM in understanding the task.
- **Leverage modern optimization techniques:** Incorporate tools such as Unsloth, bitsandbytes, and Triton for optimized fine-tuning and efficient deployment on GPUs.

## Functional Components

- **Training Script (`training_with_ppo.py`):**  
  Implements the training loop where episodes are executed, experiences are gathered, and the PPO algorithm updates the LLM’s policy based on the collected data. It also includes optional experience replay for combining PPO with DQN updates.

- **LLM RL Agent (`llm_rl_improved_agent.py`):**  
  Defines the RL agent that uses an LLM for decision making. It contains methods for:

  - Formatting state data into natural language prompts.
  - Forward propagation (action generation) and memory management.
  - Saving and loading the fine-tuned model.

- **CUDA Installer (`cuda_installer.py`):**  
  Provides a utility to install CUDA-enabled versions of PyTorch if CUDA is not already available.

- **Supporting Optimizations:**
  - **Unsloth:** Enhances the baseline LLM to work more efficiently in RL settings.
  - **bitsandbytes & Triton:** Provide memory-efficient operations and quantization support (GPU-centric) for faster inference and training.

## Technologies Used

- **Python 3.11.8:** The primary programming language.
- **PyTorch:** Deep learning framework used for model training and optimization.
- **Gymnasium (OpenAI Gym):** Provides the Cart-Pole environment.
- **Unsloth:** Adds performance optimizations and advanced fine-tuning strategies.
- **BitsAndBytes & Triton:** Handle memory-efficient quantized operations and GPU-specific optimizations.
- **UV Package Manager:** Used for creating and managing the virtual environment.
- **Proximal Policy Optimization (PPO):** The reinforcement learning algorithm used to train the LLM agent.

## Installation Steps

1. **Install UV Package Manager**

   - Download the UV standalone package and place it in your system’s PATH.
   - Verify the installation by running:
     ```bash
     uv --version
     ```

2. **Create and Activate the Virtual Environment**

   - Open your project directory and create a Python 3.11.8 virtual environment:
     ```bash
     uv venv --python 3.11.8 .venv
     ```
   - Activate the virtual environment:
     ```bash
     .\.venv\Scripts\activate
     ```

3. **Install Project Dependencies**

   - Install all packages listed in the `requirements.lock` file:
     ```bash
     uv pip install -r requirements.lock
     ```
   - If you encounter issues with **triton** and **bitsandbytes**, install them manually:
     ```bash
     uv pip install bitsandbytes==0.43.0 triton
     ```

4. **Run the Project**

   - Start training by running:
     ```bash
     python training_with_ppo.py --episodes 200 --batch-size 64
     ```

5. **Troubleshooting**
   - If issues arise, try deleting the virtual environment and repeating steps 2–3:
     ```bash
     rm -r .\.venv
     uv venv --python 3.11.8 .venv
     .\.venv\Scripts\activate
     uv pip install -r requirements.lock
     ```

## Additional Notes

- **CUDA & GPU Support:**  
  Ensure you have the latest NVIDIA drivers and CUDA Toolkit installed (CUDA 11.8 recommended) to fully utilize GPU acceleration. Note that certain libraries (e.g., Triton) may have limited support on Windows—consider using WSL2 or a Linux environment for best performance.

- **Prompt Engineering:**  
  The LLM state prompt (formatted in `LLMRLImprovedAgent._format_state_prompt`) includes detailed instructions to contextualize the environment state. This assists the LLM in generating an action decision rather than a text response.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or further assistance, please contact [Your Contact Information] or open an issue in the repository.
