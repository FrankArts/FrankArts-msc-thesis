# Code files for my Msc Thesis

This repository contains the code files that I used to complete my Msc Thesis, titled "Using large language model embeddings to enable processing of natural language task instructions in a reinforcement learning agent".

The Task Instructions and Embeddings folder contains all the files used to generate task instructions, convert them into embeddings using the OpenAI API, and checking the similarity of the generated embeddings.

The Robogym Files folder contains all the files related to getting the robogym environment to work. 
To make this part work, you need to 
1. download the robogym library from https://github.com/openai/robogym
2. place the environment files in the robogym>envs>rearrange folder
3. place the goal generator files in the robogym>envs>rearrange>goals folder.

Finally, the PPO training folder contains the scripts used to train a PPO agent using the task embeddings and the robogym environment.
