import openai
import tiktoken
import pickle
import numpy as np
import time

# Enter `openai.api_key = <API-KEY>` to enable openai API

#Load the task instruction sentences
with open('C:/Users/Frank/Desktop/Tilburg University/Thesis/Task Instructions/task_instructions.pickle', 'rb') as handle:
    instructions = pickle.load(handle)

# Check the number of tokens required to get embeddings, to calculate total cost for using the API:
# instructions_tokens = [len(tiktoken.get_encoding("cl100k_base").encode(instruction)) for instruction in instructions]
# print(sum(instructions_tokens)) # 3200; 512 sentences of 5-8 tokens each.
# Total cost was about euro 0.0018

#Generate the embeddings using the openai library, store everything in a numpy array
embeddings = []
request_count = 0
for instruction in instructions:
    print("Processing instruction {}...".format(instruction))
    #The script returns an error when too many requests per minute are sent to the API.
    #Maximum requests per minute = 60, so set to 50 to avoid any errors
    if request_count >= 50:
        #If request_count >= 50, we need to wait a bit to avoid errors due to exceeding the rate limit
        print("Sleeping for 61 seconds to avoid exceeding the rate limit...")
        time.sleep(61)
        request_count = 0
    embeddings += [openai.Embedding.create(input = instruction, model = "text-embedding-ada-002")['data'][0]['embedding']]
    request_count += 1

embeddings = np.array(embeddings)

#Store the numpy array containing the embeddings
with open('C:/Users/Frank/Desktop/Tilburg University/Thesis/Task Instructions/task_embeddings.pickle', 'wb') as handle:
    pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
np.save('C:/Users/Frank/Desktop/Tilburg University/Thesis/Task Instructions/task_embeddings.npy', embeddings)

np.savetxt('C:/Users/Frank/Desktop/Tilburg University/Thesis/Task Instructions/task_embeddings.csv', embeddings, delimiter = ";")
    
print(embeddings.shape)
print(embeddings[0])

with open('C:/Users/Frank/Desktop/Tilburg University/Thesis/Task Instructions/task_instructions_green_yellow.pickle', 'rb') as handle:
    yg_instructions = pickle.load(handle)
    
#Generate the embeddings using the openai library, store everything in a numpy array
# This time for the green and yellow task instructions.
embeddings_yg = []
request_count = 0
for instruction in yg_instructions:
    print("Processing instruction {}...".format(instruction))
    #The script returns an error when too many requests per minute are sent to the API.
    #Maximum requests per minute = 60, so set to 50 to avoid any errors
    if request_count >= 50:
        #If request_count >= 50, we need to wait a bit to avoid errors due to exceeding the rate limit
        print("Sleeping for 61 seconds to avoid exceeding the rate limit...")
        time.sleep(61)
        request_count = 0
    embeddings_yg += [openai.Embedding.create(input = instruction, model = "text-embedding-ada-002")['data'][0]['embedding']]
    request_count += 1
    
embeddings = np.array(embeddings_yg)

# #Store the numpy array containing the embeddings
with open('C:/Users/Frank/Desktop/Tilburg University/Thesis/Task Instructions/task_embeddings_yg.pickle', 'wb') as handle:
    pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
np.save('C:/Users/Frank/Desktop/Tilburg University/Thesis/Task Instructions/task_embeddings_yg.npy', embeddings)

np.savetxt('C:/Users/Frank/Desktop/Tilburg University/Thesis/Task Instructions/task_embeddings_yg.csv', embeddings, delimiter = ";")
    
print(embeddings.shape)
print(embeddings[0])