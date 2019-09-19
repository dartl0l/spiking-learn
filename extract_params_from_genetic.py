import re
import os
import sys
# sys.path.append('./src/')
import json
from spiking_network_learning_alghorithm import traits
# import numpy as np


def get_parameters_of_simulation(input_folder, traits):
    settings = json.load(open(input_folder + '/settings.json', 'r'))
    traits_set = (traits.data_traits,
                  traits.network_traits,
                  traits.neuron_traits,
                  traits.synapse_traits)
    settings_set = (settings['data'],
                    settings['network'],
                    settings['model']['neuron_out'], 
                    settings['model']['syn_dict_stdp'])

    parameters = {}

    for trait, setting in zip(traits_set, settings_set):
        for current_trait in trait:
            parameters[current_trait] = setting[current_trait] / trait[current_trait]['details']['max']
    return parameters


def get_accuracy_of_simulation(input_folder):
    acc = 0.0
    with open(input_folder + '/acc.txt', 'r') as acc_file:
        for line in acc_file:
            acc_str = re.search("^(Accuracy test mean: )(\d.*)", line)
            if acc_str:
                # print(acc_str.group(2))
                acc = float(acc_str.group(2))
    return acc


def collect_parameters(input_directory, good_acc_threshold=0.9):
    good_genomes = []
    good_generations = []

    r_genome = re.compile(r'genome*')
    r_generation = re.compile(r'generation*')

    genome_folder_names = [x for x in os.listdir(input_directory) if r_genome.search(x)]
    generation_folder_names = ['output/' + x for x in os.listdir(input_directory + '/output') if r_generation.search(x)]

    for genome_folder in genome_folder_names:
        try:
            parameters = get_parameters_of_simulation(input_directory + genome_folder, traits)
            acc = get_accuracy_of_simulation(input_directory + genome_folder)
        except FileNotFoundError:
            print('no files in folder')
        if acc > good_acc_threshold:
            good_genomes.append((parameters, acc))

    for generation_folder in generation_folder_names:
        try:
            parameters = get_parameters_of_simulation(input_directory + generation_folder, traits)
            acc = get_accuracy_of_simulation(input_directory + generation_folder)
        except FileNotFoundError:
            print('no files in folder')
        if acc > good_acc_threshold:
            good_generations.append((parameters, acc))

    # print(good_genomes)
    # print(good_generations)

    return good_genomes, good_generations


if __name__ == '__main__':
    input_dir = sys.argv[1]
    # output_dir = sys.argv[2]

    good_genomes, good_generations = collect_parameters(input_dir) # , output_dir)