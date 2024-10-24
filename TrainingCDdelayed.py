
from itertools import product
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.stats import ks_2samp
from fastdtw import fastdtw
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from sklearn.metrics import confusion_matrix
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
import json
import numpy as np
import os
import torch
from sklearn.neighbors import KernelDensity
from Models import ddpm_schedules
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import mean_squared_error
import argparse
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CyclicLR

from ModelsCD import Model_mlp_diff,  Model_Cond_Diffusion, ObservationEmbedder, SpeakingTurnDescriptorEmbedder, ChunkDescriptorEmbedder

import wandb


print(torch.cuda.is_available())
# Create the parser
parser = argparse.ArgumentParser(description='Train and/or Evaluate the Diffusion Model')

# Add arguments
parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
parser.add_argument('--train', action='store_true', help='Run training')
parser.add_argument('--evaluate', action='store_true',help='Run evaluation')
parser.add_argument('--gpu', action='store_true',help='Run evaluation')
parser.add_argument('--expo', action='store_true', help='Run training')
parser.add_argument('--cycle', action='store_true', help='Run training')
parser.add_argument('--evaluation_param', type=int, default=10, help='Integer parameter for evaluation (default: 0)')

# Parse arguments
args = parser.parse_args()

# Determine whether to run training and/or evaluation
run_training = args.train or (not args.train and not args.evaluate)
run_evaluation = args.evaluate or (not args.train and not args.evaluate)

# Load the configuration file
config_path = args.config
config_basename = os.path.splitext(os.path.basename(config_path))[0]
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Set the model path with reference to the config name
model_filename = f'saved_model_{config_basename}.pth'
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_filename)

os.environ["WANDB_MODE"] = "offline" #Server only (or wandb offline command just before wandb online to reactivate)
# wandb sync --sync-all : command pour synchroniser les meta données sur le site
#rm -r wandb (remove les meta données un fois le train fini)

DATASET_PATH = "dataset"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


SAVE_DATA_DIR = config.get("save_data_dir", "output")
EXTRA_DIFFUSION_STEPS = config.get("extra_diffusion_steps", [0, 2, 4, 8, 16, 32])
GUIDE_WEIGHTS = config.get("guide_weights", [0.0, 4.0, 8.0])

n_hidden = 512

# Set training parameters from config or defaults
n_epoch = config.get("num_epochs", 1)
lrate = config.get("learning_rate", 1e-4)
base_lr =config.get("base_lr", 0.00001)
max_lr =config.get("max_lr", 0.001)
gamma = config.get("gamma", 0.1)
#sampling_rate = config.get("sampling_rate", 25)
batch_size = config.get("batch_size", 32)
n_T = config.get("num_T", 50)
net_type = config.get("net_type", "transformer")
num_event_types = config.get("num_event_types", 12)
event_embedding_dim = config.get("event_embedding_dim", 64)
#continuous_embedding_dim = config.get("continuous_embedding_dim", 3)
embed_output_dim = config.get("embed_output_dim", 128)
drop_prob = config.get("drob_prob", 0.1)
guide_w = config.get("guide_w", 3)

num_facial_types = 7
facial_embed_dim = 32
cnn_output_dim = 512  # Output dimension after passing through CNN layers
lstm_hidden_dim = 256

EXPERIMENTS = [
    {
        "exp_name": "diffusion",
        "model_type": "diffusion",
        "drop_prob": drop_prob,
    },
]


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Diffusion Model 2 training",

    # track hyperparameters and run metadata
    config={
        "learning_rate": lrate,
        "architecture": config_path,
        "dataset": "AnnoMI",
        "epochs": n_epoch,
    }
)


def evaluate_expression_match(y_pred, y_target, mi_behaviors, sequence_length, expression, mi_act):
    """
    Evaluate the amount of a specific facial expression in the target and predicted sequences
    when the MI behavior vector contains a specific dialogue act, and compare the match counts.

    Args:
    - y_pred: Predicted sequences (batch_size, seq_length)
    - y_target: Target sequences (batch_size, seq_length)
    - mi_behaviors: MI behavior vectors for each sequence in the batch (batch_size, mi_vector_length)
    - sequence_length: The length of each sequence
    - expression: The facial expression to count (default is 1)
    - mi_act: The MI dialogue act to filter by (default is 5)
    """
    total_expression_target = 0
    total_expression_pred = 0
    total_frames_filtered = 0

    for pred_seq, target_seq, mi_vector in zip(y_pred, y_target, mi_behaviors):
        if mi_act in mi_vector:
            total_frames_filtered += sequence_length
            pred_count = np.sum(np.array(pred_seq) == expression)
            target_count = np.sum(np.array(target_seq) == expression)
            total_expression_pred += pred_count
            total_expression_target += target_count

    return total_expression_target, total_expression_pred, total_frames_filtered


def count_zero_sequences(dataloader):
    count = 0
    for x_batch, y_batch, _, _ in dataloader:
        print("the observation sequence is : ")
        print(x_batch[0])
        print(" the action sequence is : ")
        print(y_batch[0])
        # # Check if each sequence contains only 0 and sum the True values
        # contains_only_zeros = (x_batch == 0).all(dim=1)
        # count += contains_only_zeros.sum().item()
    return count

def is_valid_chunk(chunk):
    if not chunk[0]:  # Check if the chunk[0] is an empty list
        return False
    for event in chunk[0]:
        if event[0] is float or event[1] is None or event[2] is None:
            return False
    return True

# Function to create and log a scatter plot
def log_scatter_plot(x, y, title, x_label, y_label, key):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

    # Log using wandb
    wandb.log({key: wandb.Image(plt)})
    plt.close()

def transform_action_to_sequence(events, sequence_length):
    # This remains the same as the transform_to_sequence function
    sequence = [0] * sequence_length
    for event in events:

        event_type, start_time, duration = event
        start_sample = int(start_time * sequence_length)
        end_sample = int(start_sample + (duration * sequence_length))
        for i in range(start_sample, min(end_sample, sequence_length)):
            sequence[i] = event_type
    return sequence

def transform_obs_to_sequence(events, sequence_length):
    facial_expression_events = [21, 27, 31]  # Define facial expression event types
    sequence = [0] * sequence_length
    mi_behaviors = []  # To store MI behaviors
    for event in events:
        event_type, start_time, duration = event
        if event_type not in facial_expression_events and event_type == round(event_type):
            mi_behaviors.append(event_type)
        else:
            start_sample = int(start_time * sequence_length)
            end_sample = int(start_sample + (duration * sequence_length))
            for i in range(start_sample, min(end_sample, sequence_length)):
                if event_type == round(event_type):
                    sequence[i] = event_type
                else:
                    sequence[i] = 0
    return sequence, mi_behaviors

def load_data_from_folder(folder_path):
    all_data = []  # This list will hold all our chunks (both observations and actions) from all files.
    total_lines = 0
    # Iterate over each file in the directory
    for file in os.listdir(folder_path):
        # If the file ends with '.txt', we process it
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), 'r') as f:
                # Read the lines and filter out any empty ones.
                lines = f.readlines()
                total_lines += len(lines)
                non_empty_lines = [line.strip() for line in lines if line.strip() != ""]

                # Transform the non-empty line strings into actual list of tuples.
                chunks = [eval(line) for line in non_empty_lines]


                # Extract observation, action and chunk descriptor
                observation_chunks = [chunk[:-1] for chunk in chunks[::2]]  # get all tuples except the last one
                action_chunks = chunks[1::2]  # extract every second element starting from 1
                chunk_descriptors = [chunk[-1] for chunk in chunks[::2]]

                # Replace None values by -1 in chunk descriptor
                for i in range(len(chunk_descriptors)):
                    event = list(chunk_descriptors[i])
                    if event[2] is None:
                        event[2] = -1
                    if event[1] is None:
                        event[1] = -1
                    chunk_descriptors[i] = tuple(event)

                # Extend the all_data list with the observation, action, and chunk descriptor
                all_data.extend(list(zip(observation_chunks, action_chunks, chunk_descriptors)))
    return all_data  # Return the master list containing chunks from all files

def rare_event_criteria(observation, action):
    """
    Define the criteria to identify a rare event in a sequence.
    :param observation: Observation part of the sequence.
    :param action: Target action part of the sequence.
    :return: A string indicating the type of rare event ('observation', 'action', 'both', or None).
    """
    # Define the rare event type for observation and action
    rare_event_types_observation = [11, 13, 4, 6, 5, 27, 31]
    rare_event_types_action = [26, 30]  # Example rare event types for action

    rare_observation = any(event[0] in rare_event_types_observation for event in observation)
    rare_action = any(event[0] in rare_event_types_action for event in action)

    if rare_observation and rare_action:
        return 'both'
    elif rare_observation:
        return 'observation'
    elif rare_action:
        return 'action'
    else:
        return None


def oversample_sequences(data, rare_event_criteria, oversampling_factor_obs=3, oversampling_factor_act=2, oversampling_factor_both=3):
    """
    Oversample sequences that contain rare events, with specific factors for observation, action, and both.
    :param data: List of sequences (each sequence is a tuple of observation, action, chunk_descriptor).
    :param rare_event_criteria: Function to determine the type of rare event in a sequence.
    :param oversampling_factor_obs: Factor by which to oversample rare sequences in observations.
    :param oversampling_factor_act: Factor by which to oversample rare sequences in actions.
    :param oversampling_factor_both: Factor by which to oversample sequences with rare events in both.
    :return: List of sequences with oversampled rare events.
    """
    oversampled_data = []
    for sequence in data:
        observation, action, chunk_descriptor = sequence
        rare_event_type = rare_event_criteria(observation, action)

        if rare_event_type == 'observation':
            oversampled_data.extend([sequence] * oversampling_factor_obs)
        elif rare_event_type == 'action':
            oversampled_data.extend([sequence] * oversampling_factor_act)
        elif rare_event_type == 'both':
            oversampled_data.extend([sequence] * oversampling_factor_both)
        else:
            oversampled_data.append(sequence)
    return oversampled_data

def preprocess_data(data):
    filtered_data = [chunk for chunk in data if is_valid_chunk(chunk)]

    # Compute the average duration of speaking turns
    total_duration = sum([max(event[2] for event in chunk[0]) for chunk in filtered_data])
    average_duration = total_duration / len(filtered_data)
    sequence_length = int(average_duration * 25)  # Assuming 25  SAMPLING RATE IS HERE TO ADJUST

    data = filtered_data

    for chunk in data:
        if not chunk[0]:  # Skip if the observation vector is empty
            continue

        valid_start_times1 = [event[1] for event in chunk[0] if isinstance(event[1], float) and event[1] > 0]
        valid_start_times2 = [event[1] for event in chunk[1] if isinstance(event[1], float) and event[1] > 0]

        if not valid_start_times1 and valid_start_times2:# Skip if no valid start times
            continue
        min_start_time = min(min(valid_start_times1), min(valid_start_times2))

        # Calculate speaking turn duration as the end of the last event minus min_start_time
        valid_end_times1 = [(event[1] + event[2]) for event in chunk[0] if isinstance(event[1], float) and event[1] > 0]
        valid_end_times2 = [(event[1] + event[2]) for event in chunk[1] if isinstance(event[1], float) and event[1] > 0]
        max_end_time = max(max(valid_end_times1), max(valid_end_times2)) if valid_end_times1 and valid_end_times2 else 0
        speaking_turn_duration = max_end_time - min_start_time

        if speaking_turn_duration <= 0: # Skip turns with non-positive duration
            continue

        # Normalize start times and durations within each chunk
        for vector in [0, 1]:  # 0 for observation, 1 for action
            for i, event in enumerate(chunk[vector]):
                event_type, start_time, duration = event
                if start_time == 0.0:
                    continue
                if start_time<min_start_time:
                    start_time = min_start_time
                # Standardize the starting times relative to the speaking turn's start
                normalized_start_time = (start_time - min_start_time)


                # Normalize start times and durations against the speaking turn duration
                normalized_start_time = normalized_start_time / speaking_turn_duration
                normalized_duration = duration / speaking_turn_duration


                # Update the event with normalized values
                chunk[vector][i] = (event_type, round(normalized_start_time, 3), round(normalized_duration, 3))

    return data, sequence_length

# Assuming the functions 'load_data_from_folder' and 'preprocess_data' are defined as in your provided code.

class MyCustomDataset(Dataset):
    def __init__(self, folder_path, train_or_test="train", indices_path=None, train_prop=0.90, oversample_rare_events=False, Delay=17):

        # Define new mappings for facial expressions and MI behaviors
        facial_expression_mapping = {0: 0, 16: 1, 26: 2, 30: 3, 21: 4, 27: 5, 31: 6}
        mi_behavior_mapping = {39: 1, 38: 2, 40: 3, 41: 4, 3: 5, 4: 6, 5: 7, 6: 8, 8: 9, 11: 10, 13: 11, 12: 12}

        # Load and preprocess data
        raw_data = load_data_from_folder(folder_path)
        self.raw_data_length = len(raw_data)
        processed_data, sequence_length = preprocess_data(raw_data)
        self.sequence_length = sequence_length
        self.Delay = Delay

        # Oversample sequences with rare events
        if oversample_rare_events:
            processed_data = oversample_sequences(processed_data, rare_event_criteria)

        # If indices_path is provided, use it to load the indices for the split
        if indices_path and os.path.exists(indices_path):
            with open(indices_path, 'r') as f:
                indices = json.load(f)
            train_indices, test_indices = indices['train'], indices['test']
        else:
            # Otherwise, create the split and save the indices
            shuffled_indices = np.random.permutation(len(processed_data))
            train_indices = shuffled_indices[:int(len(processed_data) * train_prop)]
            val_indices = shuffled_indices[int(len(processed_data) * train_prop):]
            if not indices_path:
                indices_path = 'data_indices.json'
            with open(indices_path, 'w') as f:
                json.dump({'train': train_indices.tolist(), 'test': val_indices.tolist()}, f)

        # Use the indices to create the data split
        if train_or_test == "train":
            self.data = [processed_data[i] for i in train_indices]
        elif train_or_test == "test":
            self.data = [processed_data[i] for i in test_indices]
        else:
            raise ValueError("split should be either 'train' or 'validation'")

        # # Transform data into previous action integration (Tour/tour)
        # self.transformed_data = []
        # last_action = None  # Initialize last action as None
        # for observation, action, chunk_descriptor in self.data:
        #     for i in range(len(action)):
        #         if i == 0 and last_action is not None:
        #             prev_action = last_action
        #         else:
        #             prev_action = action[i - 1] if i > 0 else [0] * len(
        #                 action[0])  # Use zero vector if no previous action
        #
        #         x = observation + [prev_action]
        #         y = action[i]
        #         self.transformed_data.append((x, y, chunk_descriptor))
        #
        #     last_action = action[-1]  # Save the last action of the current sequence


        # Transform data into sequences
        self.transformed_data = []
        for observation, action, chunk_descriptor in self.data:
            # Transform both observation and action to sequences
            x, z = transform_obs_to_sequence(observation, sequence_length)
            y = transform_action_to_sequence(action, sequence_length)

            # Reassign event values based on the new mappings
            x = [facial_expression_mapping.get(item, item) for item in x]
            y = [facial_expression_mapping.get(item, item) for item in y]
            z = [mi_behavior_mapping.get(item, item) for item in z]  # Assuming z is a list of MI behavior events

            # Delay y by N frames (prepend N zeros)
            x = [0] * self.Delay + x
            # Pad x with N zeros at the end
            y = y + [0] * self.Delay

            self.sequence_length = sequence_length + self.Delay

            self.transformed_data.append((x, y, z, chunk_descriptor))


    def __len__(self):
        # Return the number of transformed data points
        return len(self.transformed_data)

    def get_seq_len(self):
        # Return the number of transformed data points
        return self.sequence_length

    def __getitem__(self, idx):
        # Retrieve the transformed data point at the given index
        x, y, z, chunk_descriptor = self.transformed_data[idx]

        # Convert lists into tensors for PyTorch
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        z_tensor = torch.tensor(z, dtype=torch.float32)
        chunk_descriptor_tensor = torch.tensor(chunk_descriptor, dtype=torch.float32)

        return x_tensor, y_tensor, z_tensor, chunk_descriptor_tensor

    def collate_fn(batch):
        # Unzip the batch into separate lists
        x_data, y_data, z_data, chunk_descriptors = zip(*batch)

        # # Separate observations and previous actions
        # observations = [x[:-1] for x in x_data]  # All elements except the last one
        # prev_actions = [x[-1] for x in x_data]  # Only the last element

        # Pad the observation sequences
        observations_padded = torch.stack([x.clone().detach() for x in x_data])
        #
        y_padded = torch.stack([y.clone().detach() for y in y_data])
        z_padded = torch.nn.utils.rnn.pad_sequence([z.clone().detach() for z in z_data],
                                                              batch_first=True, padding_value=0)

        # Convert chunk_descriptors to tensor
        chunk_descriptors_tensor = torch.stack([cd.clone().detach() for cd in chunk_descriptors])


        x_tensors = observations_padded
        z_tensors = z_padded
        y_tensors = y_padded

        return x_tensors, y_tensors, z_tensors, chunk_descriptors_tensor


def training(experiment, n_epoch, lrate, device, n_hidden, batch_size, n_T, net_type, EXTRA_DIFFUSION_STEPS, guide_w):
    # Unpack experiment settings
    exp_name = experiment["exp_name"]
    model_type = experiment["model_type"]
    drop_prob = experiment["drop_prob"]

    torch.autograd.set_detect_anomaly(True)

    # get datasets set up
    tf = transforms.Compose([])

    if args.gpu:
        #Dataset for gpu
        folder_path = '~/Observaton_Context_Tuples'
        expanded_folder_path = os.path.expanduser(folder_path)
        folder_path = expanded_folder_path
    else:
        # Update the dataset path here (dataset for local run)
        folder_path = 'C:/Users/NEZIH YOUNSI/Desktop/Hcapriori_input/Observaton_Context_Tuples'

    # Use MyCustomDataset instead of ClawCustomDataset
    torch_data_train = MyCustomDataset(folder_path, train_or_test="train", train_prop=0.90, oversample_rare_events=True, Delay=17)
    test_dataset = MyCustomDataset(folder_path, train_or_test="test", train_prop=0.90, indices_path = 'data_indices.json',oversample_rare_events=True, Delay=17 )
    dataload_train = DataLoader(torch_data_train, batch_size=batch_size, shuffle=True, collate_fn=MyCustomDataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=MyCustomDataset.collate_fn)

    sequence_length = torch_data_train.get_seq_len()

    # Calculate the total number of batches
    total_batches = len(dataload_train)
    print(f"Total number of batches: {total_batches}")
    #total_samples = total_batches * batch_size
    #zero_sequences_count = count_zero_sequences(dataload_train)
    # print("Total of empty actions")
    # print(zero_sequences_count)
    # print("sur un total de ")
    # print(total_samples)


    #OPTION 1
    observation_embedder = ObservationEmbedder(num_facial_types, facial_embed_dim, cnn_output_dim, lstm_hidden_dim, sequence_length)
    #OPTION 2
    #observation_embedder = ObservationEmbedder(num_facial_types, facial_embed_dim, lstm_hidden_dim, sequence_length)
    #OPTION3
    #observation_embedder = ObservationEmbedder(num_facial_types, facial_embed_dim, num_heads = 8, num_layers=2, sequence_length = 137, transformer_input_dim = 256)


    mi_embedder = SpeakingTurnDescriptorEmbedder(num_event_types, event_embedding_dim, embed_output_dim)
    chunk_embedder = ChunkDescriptorEmbedder(continious_embedding_dim=16 ,valence_embedding_dim=8,  output_dim=64)
   # Determine the shape of input and output tensors
    sample_observation, sample_action, sample_z, _ = torch_data_train[0]
    input_shape = sample_observation.shape
    output_dim = sample_action.shape[0]

    x_dim = input_shape
    #torch_data_train.image_all.shape[1:]
    y_dim = output_dim
    #torch_data_train.action_all.shape[1]
    t_dim = 1
    # create model


    if model_type == "diffusion":
        #ici qu'on appel le model_mlp_diff fusionné a mon embedding model
        nn_model = Model_mlp_diff(
            observation_embedder, mi_embedder, chunk_embedder, sequence_length, net_type="transformer").to(device)
        model = Model_Cond_Diffusion(
            nn_model,
            observation_embedder,
            mi_embedder,
            chunk_embedder,
            betas=(1e-4, 0.02),
            n_T=n_T,
            device=device,
            x_dim=x_dim,
            y_dim=y_dim,
            drop_prob=drop_prob,
            guide_w = guide_w,
        )
    else:
        raise NotImplementedError
    if run_training:
        # Count the number of trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

        model.to(device)
        optim = torch.optim.Adam(model.parameters(), lr=lrate)

        if args.expo:
            scheduler = ExponentialLR(optim, gamma=gamma)
        elif args.cycle:
            scheduler = CyclicLR(optim, base_lr, max_lr,
                                 step_size_up=5 * len(dataload_train),  # 5 times the number of batches in one epoch
                                 mode='triangular',  # Other modes include 'triangular2', 'exp_range'
                                 cycle_momentum=False)  # If True, momentum is cycled inversely to learning rate
        else:
            scheduler = StepLR(optim, step_size=10, gamma=gamma)

        for ep in tqdm(range(n_epoch), desc="Epoch"):
            results_ep = [ep]
            model.train()

            # lrate decay
            optim.param_groups[0]["lr"] = lrate * ((np.cos((ep / n_epoch) * np.pi) + 1) / 2)

            # train loop
            pbar = tqdm(dataload_train)
            loss_ep, n_batch = 0, 0
            for x_batch, y_batch, z_batch, chunk_descriptor in pbar:
                #need to concat the chunk descriptor after the first test and see its impact
                x_batch = x_batch.type(torch.FloatTensor).to(device) #obs
                y_batch = y_batch.type(torch.FloatTensor).to(device) #targets
                z_batch = z_batch.type(torch.FloatTensor).to(device)
                chunk_descriptor = chunk_descriptor.type(torch.FloatTensor).to(device)

                loss = model.loss_on_batch(x_batch, y_batch, z_batch, chunk_descriptor)
                optim.zero_grad()
                loss.backward()
                loss_ep += loss.detach().item()
                n_batch += 1
                pbar.set_description(f"train loss: {loss_ep / n_batch:.4f}")
                wandb.log({"loss": loss})
                optim.step()
            scheduler.step()
            results_ep.append(loss_ep / n_batch)

        torch.save(model.state_dict(), model_path)

    if run_evaluation:
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        model.eval()

# EVALUATION OF NOISE ESTIMATION
        noise_estimator = model.nn_model
        loss_mse = nn.MSELoss()
        total_validation_loss = 0.0

        for x_batch, y_batch, z_batch, chunk_descriptor in test_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            z_batch = z_batch.to(device)
            chunk_descriptor = chunk_descriptor.to(device)

            # Sample t uniformly for each data point in the batch
            t_noise = torch.randint(1, model.n_T + 1, (y_batch.shape[0], 1)).to(device)

            # Randomly sample some noise, noise ~ N(0, 1)
            noise = torch.randn_like(y_batch).to(device)

            context_mask = torch.bernoulli(torch.zeros(x_batch.shape[0]) + drop_prob).to(device)

            # Add noise to clean target actions
            y_noised = model.sqrtab[t_noise] * y_batch + model.sqrtmab[t_noise] * noise

            with torch.no_grad():
                # Use the model to estimate the noise
                estimated_noise = noise_estimator(y_noised, x_batch, z_batch, chunk_descriptor, t_noise.float() / model.n_T, context_mask)

            # Calculate the loss between the true noise and the estimated noise
            validation_loss = loss_mse(noise, estimated_noise)
            total_validation_loss += validation_loss.item()

        # Compute the average validation loss
        average_validation_loss = total_validation_loss / len(test_dataloader)
        


# DIRECT KDE CASE
        extra_diffusion_steps = 0
        guide_weight = guide_w
        kde_samples = args.evaluation_param
        total_batches = len(test_dataloader)
        # Initialize variables to calculate the overall metrics
        total_accuracy = 0
        total_edit_distance = 0
        total_sequences = 0
         # Initialize counters
        correct_activations = 0
        correct_classless_activations = 0
        correct_activations_for_class_1 = 0
        correct_activations_for_class_2 = 0
        correct_activations_for_class_3 = 0
        total_activations_ground_truth_1 =0
        total_activations_ground_truth_2 = 0
        total_activations_ground_truth_3 = 0
        total_activations_ground_truth = 0
        correct_non_activations = 0
        total_non_activations_ground_truth = 0
        overall_total_expression_target1 = 0
        overall_total_expression_pred1 = 0
        overall_total_frames_filtered1 = 0
        overall_total_expression_target2 = 0
        overall_total_expression_pred2 = 0
        overall_total_frames_filtered2 = 0
        overall_total_expression_target3 = 0
        overall_total_expression_pred3 = 0
        overall_total_frames_filtered3 = 0
        overall_total_expression_target4 = 0
        overall_total_expression_pred4 = 0
        overall_total_frames_filtered4 = 0
        overall_total_expression_target5 = 0
        overall_total_expression_pred5 = 0
        overall_total_frames_filtered5 = 0
        overall_total_expression_target6 = 0
        overall_total_expression_pred6 = 0
        overall_total_frames_filtered6 = 0


        all_preds = []
        all_targets = []
        # Initialize lists to store KS statistic and p-value
        ks_statistics = []
        p_values = []
        # Initialize list to store DTW distances
        dtw_distances = []
        dtw_distances_obs_target = []
        dtw_distances_obs_pred = []

        print(f"Total number of test batches: {total_batches}")

        for x_batch, y_batch, z_batch, chunk_descriptor in test_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            z_batch = z_batch.to(device)
            chunk_descriptor = chunk_descriptor.to(device)



            ########## Obs filtering begin  #######
            # Initialize container for filtered indices
            valid_indices = []

            # Check if the data point in x_batch is only composed of 0 and filter them
            for idx, x in enumerate(x_batch):
                if not torch.all(x == 0):
                    valid_indices.append(idx)

            # If all x_batch data points are composed of zeros, continue to the next batch
            if not valid_indices:
                continue

            # Select only the valid data points for processing
            x_batch = x_batch[valid_indices]
            y_batch = y_batch[valid_indices]
            z_batch = z_batch[valid_indices]
            chunk_descriptor = chunk_descriptor [valid_indices]

            ########## Obs filtering end #######

            # Generate multiple predictions for KDE
            all_predictions = []
            all_traces = []
            for _ in range(kde_samples):  # Number of predictions to generate for KDE (Find the best number to fit KDE and best predicitons)
                with torch.no_grad():
                    
                    model.guide_w = guide_weight
                    y_pred_= model.sample(x_batch,z_batch, chunk_descriptor).detach().cpu().numpy()
                    #y_pred_, y_pred_trace_ = model.sample(x_batch, return_y_trace=True)
                    all_predictions.append(y_pred_)
                    #all_traces.append(y_pred_trace_)

            # Apply KDE for each data point and store best predictions
            best_predictions = np.zeros_like(y_batch.cpu().numpy())
            #best_traces = []
            for i, idx in enumerate(valid_indices):
                single_pred_samples = np.array([pred[i] for pred in all_predictions])
                #single_trace_samples = np.array([trace[i] for trace in all_traces])
                kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(single_pred_samples)
                log_density = kde.score_samples(single_pred_samples)
                best_idx = np.argmax(log_density)
                best_predictions[i] = single_pred_samples[best_idx]
                 # Apply modifications to each element in the sequence
                best_predictions[i] = np.round(best_predictions[i])
                best_predictions[i][best_predictions[i] == 4] = 3  # Replace 4 with 3
                best_predictions[i][best_predictions[i] < 0] = 0  # Replace negative values with 0
                
                print("la target :")
                print(y_batch[i])
                print("la sequence obs :")
                print(x_batch[i])
                print("le Mi behaviors :")
                print(z_batch[i])
                print("le chunk_descriptor")
                print(chunk_descriptor[i])
                print("la prediction :")
                print(np.round(best_predictions[i]))

            ######### Evaluation Metrics 1  per MI behaviors / AU #####
            # Accumulate metrics for each batch (1,1)
            batch_total_expression_target1, batch_total_expression_pred1, batch_total_frames_filtered1 = \
                evaluate_expression_match(best_predictions, y_batch.cpu().numpy(), z_batch.cpu().numpy(),sequence_length, 1, 1)

            # Update overall metrics
            overall_total_expression_target1 += batch_total_expression_target1
            overall_total_expression_pred1 += batch_total_expression_pred1
            overall_total_frames_filtered1 += batch_total_frames_filtered1

            # Accumulate metrics for each batch (1,2)
            batch_total_expression_target2, batch_total_expression_pred2, batch_total_frames_filtered2 = \
                evaluate_expression_match(best_predictions, y_batch.cpu().numpy(), z_batch.cpu().numpy(),
                                          sequence_length, 2, 1)

            # Update overall metrics
            overall_total_expression_target2 += batch_total_expression_target2
            overall_total_expression_pred2 += batch_total_expression_pred2
            overall_total_frames_filtered2 += batch_total_frames_filtered2

            # Accumulate metrics for each batch (1,3)
            batch_total_expression_target3, batch_total_expression_pred3, batch_total_frames_filtered3 = \
                evaluate_expression_match(best_predictions, y_batch.cpu().numpy(), z_batch.cpu().numpy(),
                                          sequence_length, 3, 1)

            # Update overall metrics
            overall_total_expression_target3 += batch_total_expression_target3
            overall_total_expression_pred3 += batch_total_expression_pred3
            overall_total_frames_filtered3 += batch_total_frames_filtered3

            # Accumulate metrics for each batch (1,4) ==  (9 , 1)
            batch_total_expression_target4, batch_total_expression_pred4, batch_total_frames_filtered4 = \
                evaluate_expression_match(best_predictions, y_batch.cpu().numpy(), z_batch.cpu().numpy(),
                                          sequence_length, 1, 9)

            # Update overall metrics
            overall_total_expression_target4 += batch_total_expression_target4
            overall_total_expression_pred4 += batch_total_expression_pred4
            overall_total_frames_filtered4 += batch_total_frames_filtered4

            # Accumulate metrics for each batch (1,5) ==  (9 , 2)
            batch_total_expression_target5, batch_total_expression_pred5, batch_total_frames_filtered5 = \
                evaluate_expression_match(best_predictions, y_batch.cpu().numpy(), z_batch.cpu().numpy(),
                                          sequence_length, 2, 9)

            # Update overall metrics
            overall_total_expression_target5 += batch_total_expression_target5
            overall_total_expression_pred5 += batch_total_expression_pred5
            overall_total_frames_filtered5 += batch_total_frames_filtered5

            # Accumulate metrics for each batch (1,6) ==  (9 , 3) 9 == question
            batch_total_expression_target6, batch_total_expression_pred6, batch_total_frames_filtered6 = \
                evaluate_expression_match(best_predictions, y_batch.cpu().numpy(), z_batch.cpu().numpy(),
                                          sequence_length, 3, 9)

            # Update overall metrics
            overall_total_expression_target6 += batch_total_expression_target6
            overall_total_expression_pred6 += batch_total_expression_pred6
            overall_total_frames_filtered6 += batch_total_frames_filtered6

            #######  Evaluation metrics 2 ##########

            # Convert the tensors to lists of integers for edit distance computation
            y_pred_list = best_predictions.tolist()
            y_target_list = y_batch.cpu().numpy().tolist()
            x_target_list = x_batch.cpu().numpy().tolist()

            # Extend the lists across all batches for F1 computation
            all_preds.extend(y_pred_list)
            all_targets.extend(y_target_list)


            # Compute metrics for each sequence in the batch
            for pred, target, target_obs in zip(y_pred_list, y_target_list, x_target_list):

                Au_mapping = {0: 0, 4: 1, 5: 2, 6: 3}

                # Frame-wise accuracy
                correct_predictions = np.sum(np.array(pred) == np.array(target))
                accuracy_per_sequence = correct_predictions / len(target)
                total_accuracy += accuracy_per_sequence


                # Edit distance
                edit_distance = levenshtein_distance(pred, target)
                total_edit_distance += edit_distance

                # Perform KS test
                ks_statistic, p_value = ks_2samp(pred, target)
                ks_statistics.append(ks_statistic)
                p_values.append(p_value)

                target_obs = [Au_mapping.get(item, item) for item in target_obs]

                # Compute DTW distance (pred_ traget)
                distance, _ = fastdtw(pred, target)
                dtw_distances.append(distance)

                # Compute DTW distance (obs traget)
                distance, _ = fastdtw(target_obs, target)
                dtw_distances_obs_target.append(distance)

                # Compute DTW distance (obs pred)
                distance, _ = fastdtw(target_obs, pred)
                dtw_distances_obs_pred.append(distance)

            # Update the total number of sequences processed
            total_sequences += y_batch.shape[0]

            # Iterate over each pair of predicted and target sequences (AHR NHR)
            for pred, target in zip(y_pred_list, y_target_list):
                # Convert sequences to arrays for easier element-wise comparison
                pred_array = np.array(pred)
                target_array = np.array(target)
                
                # Count activations in ground truth and correct predictions
                is_active_pred = pred_array > 0  # Assuming AU > 0 indicates activation
                is_active_target = target_array > 0
                is_active_1 = target_array == 1
                is_active_2 = target_array == 2
                is_active_3 = target_array == 3

                correct_activations += np.sum((pred_array == target_array) & is_active_target)
                correct_classless_activations += np.sum(is_active_pred & is_active_target)
                # Correct activations specifically for class 1
                correct_activations_for_class_1 += np.sum((pred_array == target_array) & (target_array == 1))
                # Correct activations specifically for class 2
                correct_activations_for_class_2 += np.sum((pred_array == target_array) & (target_array == 2))
                # Correct activations specifically for class 3
                correct_activations_for_class_3 += np.sum((pred_array == target_array) & (target_array == 3))

                total_activations_ground_truth += np.sum(is_active_target)
                total_activations_ground_truth_1 += np.sum(is_active_1)
                total_activations_ground_truth_2 += np.sum(is_active_2)
                total_activations_ground_truth_3 += np.sum(is_active_3)
                
                # Count non-activations in ground truth and correct predictions
                correct_non_activations += np.sum((pred_array == target_array) & ~is_active_target)
                total_non_activations_ground_truth += np.sum(~is_active_target)

        print(f'Average Validation Loss for Noise Estimation: {average_validation_loss}')

        # Calculate average KS statistic and p-value
        average_ks_statistic = np.mean(ks_statistics)
        average_p_value = np.mean(p_values)
        print(f'Average KS Statistic: {average_ks_statistic}')
        print(f'Average P-value: {average_p_value}')

        # Calculate average DTW distance
        average_dtw_distance = np.mean(dtw_distances)
        print(f'Average DTW Distance pred_target: {average_dtw_distance}')

        # Calculate average DTW distance
        average_dtw_distance = np.mean(dtw_distances_obs_target)
        print(f'Average DTW Distance obs_target: {average_dtw_distance}')

        # Calculate average DTW distance
        average_dtw_distance = np.mean(dtw_distances_obs_pred)
        print(f'Average DTW Distance obs_pred: {average_dtw_distance}')

        # Compute the average metrics over all batches
        average_accuracy = total_accuracy / total_sequences
        average_edit_distance = total_edit_distance / total_sequences
        average_edit_distance = average_edit_distance / y_batch.shape[1]
        print(f'Average frame-wise accuracy over the validation set: {average_accuracy:.2f}')
        print(f'Average Levenshtein distance over the batch: {average_edit_distance:.2f}')
        # Calculate AHR and NHR
        ahr = correct_activations / total_activations_ground_truth if total_activations_ground_truth > 0 else 0
        achr = correct_classless_activations / total_activations_ground_truth if total_activations_ground_truth > 0 else 0
        nhr = correct_non_activations / total_non_activations_ground_truth if total_non_activations_ground_truth > 0 else 0
        ahr_mouthup = correct_activations_for_class_1 / total_activations_ground_truth_1 if total_activations_ground_truth_1 > 0 else 0
        ahr_nosewrinkle = correct_activations_for_class_2 / total_activations_ground_truth_2 if total_activations_ground_truth_2 > 0 else 0
        ahr_mouthdown = correct_activations_for_class_3 / total_activations_ground_truth_3 if total_activations_ground_truth_3 > 0 else 0

        print(f'AHR: {ahr:.4f}')
        print(f'ACHR: {achr:.4f}')
        print(f'NHR: {nhr:.4f}')
        print(f'AHR MOUTH UP: {ahr_mouthup:.4f}')
        print(f'AHR MOUTH DOWN: {ahr_mouthdown:.4f}')
        print(f'AHR NOSE WRINKLE: {ahr_nosewrinkle:.4f}')

        # Compute overall metrics after the loop
        if overall_total_frames_filtered1 > 0:
            overall_target_rate = overall_total_expression_target1 / overall_total_frames_filtered1
            overall_pred_rate = overall_total_expression_pred1 / overall_total_frames_filtered1
            print(f"Overall target expression rate Therapist/ Mouthup: {overall_target_rate:.2f}")
            print(f"Overall predicted expression rate Therapist/ Mouthup: {overall_pred_rate:.2f}")
        else:
            print("No sequences with specified MI act found in the dataset.")

        # Compute overall metrics after the loop
        if overall_total_frames_filtered2 > 0:
            overall_target_rate = overall_total_expression_target2 / overall_total_frames_filtered2
            overall_pred_rate = overall_total_expression_pred2 / overall_total_frames_filtered2
            print(f"Overall target expression rate Therapist/ NW: {overall_target_rate:.2f}")
            print(f"Overall predicted expression rate Therapist/ NW: {overall_pred_rate:.2f}")
        else:
            print("No sequences with specified MI act found in the dataset.")

        # Compute overall metrics after the loop
        if overall_total_frames_filtered3 > 0:
            overall_target_rate = overall_total_expression_target3 / overall_total_frames_filtered3
            overall_pred_rate = overall_total_expression_pred3 / overall_total_frames_filtered3
            print(f"Overall target expression rate Therapist/ Mouthdown: {overall_target_rate:.2f}")
            print(f"Overall predicted expression rate Therapist/ Mouthdown: {overall_pred_rate:.2f}")
        else:
            print("No sequences with specified MI act found in the dataset.")

        # Compute overall metrics after the loop
        if overall_total_frames_filtered4 > 0:
            overall_target_rate = overall_total_expression_target4 / overall_total_frames_filtered4
            overall_pred_rate = overall_total_expression_pred4 / overall_total_frames_filtered4
            print(f"Overall target expression rate Question/ Mouthup: {overall_target_rate:.2f}")
            print(f"Overall predicted expression rate Question/ Mouthup: {overall_pred_rate:.2f}")
        else:
            print("No sequences with specified MI act found in the dataset.")

        # Compute overall metrics after the loop
        if overall_total_frames_filtered5 > 0:
            overall_target_rate = overall_total_expression_target5 / overall_total_frames_filtered5
            overall_pred_rate = overall_total_expression_pred5 / overall_total_frames_filtered5
            print(f"Overall target expression rate Question/ NW: {overall_target_rate:.2f}")
            print(f"Overall predicted expression rate Question/ NW: {overall_pred_rate:.2f}")
        else:
            print("No sequences with specified MI act found in the dataset.")

        # Compute overall metrics after the loop
        if overall_total_frames_filtered6 > 0:
            overall_target_rate = overall_total_expression_target6 / overall_total_frames_filtered6
            overall_pred_rate = overall_total_expression_pred6 / overall_total_frames_filtered6
            print(f"Overall target expression rate Question/ Mouthdown: {overall_target_rate:.2f}")
            print(f"Overall predicted expression rate Question/ Mouthdown: {overall_pred_rate:.2f}")
        else:
            print("No sequences with specified MI act found in the dataset.")





if __name__ == "__main__":
    os.makedirs(SAVE_DATA_DIR, exist_ok=True)
    for experiment in EXPERIMENTS:
        training(experiment, n_epoch, lrate, device, n_hidden, batch_size, n_T, net_type, EXTRA_DIFFUSION_STEPS, guide_w)
