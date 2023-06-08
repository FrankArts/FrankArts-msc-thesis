import pickle
import numpy as np

with open('C:/Users/Frank/Desktop/Tilburg University/Thesis/Task Instructions/reach_task_ids.pickle', 'rb') as f:
    task_ids = pickle.load(f)
    
with open('C:/Users/Frank/Desktop/Tilburg University/Thesis/Task Instructions/reach_task_embeddings.pickle', 'rb') as f:
    task_embeddings = pickle.load(f)
    
with open('C:/Users/Frank/Desktop/Tilburg University/Thesis/Task Instructions/reach_task_ids_green.pickle', 'rb') as f:
    task_ids_green = pickle.load(f)
    
with open('C:/Users/Frank/Desktop/Tilburg University/Thesis/Task Instructions/reach_task_embeddings_green.pickle', 'rb') as f:
    task_embeddings_green = pickle.load(f)
    
    
def calc_cosine_similarity_within(array_of_emb):
    # Returns the average cosine similarity of all embeddings in array_of_emb,
    # excluding cosine similarity of any one embedding with itself
    cosine_list = []
    for i in range(len(array_of_emb)):
        for j in range(len(array_of_emb)):
            if i != j:
                cosine = np.dot(array_of_emb[i,:], array_of_emb[j,:])
                cosine_list.append(cosine)
    return cosine_list

def calc_cosine_similarity_between(corresponding_tasks, other_tasks):
    # Returns the average cosine similarity of each embedding in corresponding_tasks to all embeddings in other_tasks
    cosines_all_tasks = []
    for i in range(len(corresponding_tasks)):
        cosines_for_this_task = []
        for j in range(len(other_tasks)):
            cosine = np.dot(corresponding_tasks[i,:], other_tasks[j,:])
            cosines_for_this_task.append(cosine)
        cosines_all_tasks.append(cosines_for_this_task)
    return cosines_all_tasks
    
print("Original sentences:")
for task in set(task_ids):
    task_idx = [i for i in range(len(task_ids)) if task_ids[i]==task]
    other_task_idx = [i for i in range(len(task_ids)) if task_ids[i] != task]
    corresponding_tasks = task_embeddings[task_idx, :] #[task_embeddings[i] for i in task_idx]
    other_tasks = task_embeddings[other_task_idx, :] #[task_embeddings[i] for i in other_task_idx]
    print("Task {}".format(task))
    print("Mean cosine similarity within: {}".format(np.mean(calc_cosine_similarity_within(corresponding_tasks))))
    print("Std cosine similarity within: {}".format(np.std(calc_cosine_similarity_within(corresponding_tasks))))
    print("Mean cosine similarity between: {}".format(np.mean(calc_cosine_similarity_between(corresponding_tasks, other_tasks))))
    print("Std cosine similarity between: {}".format(np.std(calc_cosine_similarity_between(corresponding_tasks, other_tasks))))
print()
print("Green sentences:")
for task in set(task_ids_green):
    task_idx = [i for i in range(len(task_ids_green)) if task_ids_green[i]==task]
    other_task_idx = [i for i in range(len(task_ids_green)) if task_ids_green[i] != task]
    corresponding_tasks = task_embeddings_green[task_idx, :] #[task_embeddings[i] for i in task_idx]
    other_tasks = task_embeddings_green[other_task_idx, :] #[task_embeddings[i] for i in other_task_idx]
    print("Task {}".format(task))
    print("Mean cosine similarity within: {}".format(np.mean(calc_cosine_similarity_within(corresponding_tasks))))
    print("Std cosine similarity within: {}".format(np.std(calc_cosine_similarity_within(corresponding_tasks))))
    #print("Mean cosine similarity between: {}".format(np.mean(calc_cosine_similarity_between(corresponding_tasks, other_tasks))))
    #print("Std cosine similarity between: {}".format(np.std(calc_cosine_similarity_between(corresponding_tasks, other_tasks))))
    print("Compared to red/blue instructions:")
    print("Mean cosine similarity between: {}".format(np.mean(calc_cosine_similarity_between(corresponding_tasks, task_embeddings))))
    print("Std cosine similarity between: {}".format(np.std(calc_cosine_similarity_between(corresponding_tasks, task_embeddings))))
    
# Task instruction identifiers: 
    # 2, [Reach, red, sphere]
    # 4, [Reach, red, cube]
    # 6, [Reach, blue, sphere]
    # 8, [Reach, blue, cube]
# For green instructions:
    # 2, [Reach, green, sphere]
    # 4, [Reach, green, cube]