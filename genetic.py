# coding: utf-8

import sys
from os import system, chdir, getcwd, mkdir, path
from shutil import copytree

import numpy
import json # to read parameters from file
import MultiNEAT as neat
import mpi4py.futures as fut
from mpi4py import MPI 
from spiking_network_learning_alghorithm import traits
from spiking_network_learning_alghorithm.solver import solve_task



def prepare_genomes(genome):
    current_trait_values = genome.GetGenomeTraits()

    print(traits.data_traits)
    print(traits.neuron_traits)
    print(traits.network_traits)
    print(traits.synapse_traits)

    data_trait_values = {trait_name: trait_value
                         for trait_name, trait_value in current_trait_values.items()
                            if trait_name in traits.data_traits}
    network_trait_values = {trait_name: trait_value
                            for trait_name, trait_value in current_trait_values.items()
                                if trait_name in traits.network_traits}
    neuron_trait_values = {trait_name: trait_value
                           for trait_name, trait_value in current_trait_values.items()
                                if trait_name in traits.neuron_traits}
    synapse_trait_values = {trait_name: trait_value
                            for trait_name, trait_value in current_trait_values.items()
                                if trait_name in traits.synapse_traits}
    
    settings = json.load(open('settings.json', 'r'))
    for trait in data_trait_values:
        settings['data'][trait] = data_trait_values[trait]

    for trait in network_trait_values:
        settings['network'][trait] = network_trait_values[trait]

    for trait in neuron_trait_values:
        settings['model']['neuron_out'][trait] = neuron_trait_values[trait]

    for trait in synapse_trait_values:
        settings['model']['syn_dict_stdp'][trait] = synapse_trait_values[trait]

    try:
        mkdir('genome' + str(genome.GetID()))
    except FileExistsError:
        print('folder genome' + str(genome.GetID()) + 'exists')

    chdir('genome' + str(genome.GetID()))
    json.dump(settings, open('settings.json', 'w'), indent=4)
    chdir('..') # out of genomeID


def evaluate(genome):
    directory_name = getcwd() + '/genome' + str(genome.GetID()) + '/'
    print("Start sim in " + str(directory_name))
    fitness, res = solve_task(directory_name)
    print("Stop sim in " + str(directory_name))
    return fitness


def evaluate_futures(genome):
    directory_name = getcwd() + '/genome' + str(genome.GetID()) + '/'
    print("Start sim in " + str(directory_name))
    os.system("cd " + directory_name + " ;"
              "python genetic_solver.py " + directory_name + "; ")
    with open('fitness.txt') as fitness_file:
        fitness = fitness_file.readline()
    os.chdir('..')
    print("Stop sim in " + str(directory_name))
    return fitness


def main(use_futures, redirect_out=True):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # mode = MPI.MODE_CREATE | MPI.MODE_WRONLY

    if redirect_out:
        sys.stdout = open('out.txt', 'w')
        sys.stderr = open('err.txt', 'w')
    
    print("Prepare traits and genomes")

    neat_params = neat.Parameters()
    system("grep -v '//' < input/neat_parameters.txt | grep . > input/neat_parameters.filtered.txt")
    neat_params.Load('input/neat_parameters.filtered.txt')
    system("rm input/neat_parameters.filtered.txt")

    # system("grep -v '//' < input/global_parameters.json | grep . > input/global_parameters.filtered.json")
    # network_parameters = json.load(open('input/global_parameters.filtered.json', 'r'))
    # system("rm input/global_parameters.filtered.json")
    # mode = MPI.MODE_RDONLY
    network_parameters = json.load(open('input/global_parameters.json', 'r'))
    # network_parameters = json.load(open(comm, 'input/global_parameters.json', mode))

    for trait_name, trait_value in traits.network_traits.items():
        neat_params.SetGenomeTraitParameters(trait_name, trait_value)
    for trait_name, trait_value in traits.neuron_traits.items():
        # change to SetNeuronTraitParameters to let the neuron parameters mutate individually for each neuron
        neat_params.SetGenomeTraitParameters(trait_name, trait_value)
    for trait_name, trait_value in traits.synapse_traits.items():
        # change to SetLinkTraitParameters to let the synapse parameters mutate individually for each synapse
        neat_params.SetGenomeTraitParameters(trait_name, trait_value)

    genome = neat.Genome(
        0, # Some genome ID, I don't know what it means.
        network_parameters['inputs_number'],
        2, # ignored for seed_type == 0, specifies number of hidden units if seed_type == 1
        network_parameters['outputs_number'],
        False, #fs_neat. If == 1, a minimalistic perceptron is created: each output is connected to a random input and the bias.
        neat.ActivationFunction.UNSIGNED_SIGMOID, # output neurons activation function
        neat.ActivationFunction.UNSIGNED_SIGMOID, # hidden neurons activation function
        0, # seedtype
        neat_params, # global parameters object returned by neat.Parameters()
        0 # number of hidden layers
    )

    population = neat.Population(
        genome,
        neat_params,
        True, # whether to randomize the population
        0.5, # how much to randomize
        0 # the RNG seed
    )


    # fh = MPI.File.Open(comm, "datafile", mode) 
    # line1 = str(comm.rank)*(comm.rank+1) + '\n' 
    # line2 = chr(ord('a')+comm.rank)*(comm.rank+1) + '\n' 
    # fh.Write_ordered(line1) 
    # fh.Write_ordered(line2) 
    # fh.Close() 
    print("Start solving generations")
    outfile = open('output/fitness.txt', 'w')
    # mode = MPI.MODE_CREATE | MPI.MODE_WRONLY
    # outfile = MPI.File.Open(comm, 'output/fitness.txt', mode) 
    # with open('output/fitness.txt', 'w') as outfile:
    for generation_number in range(network_parameters['generations']):
        print("Generation " + str(generation_number) + " started")
        genome_list = neat.GetGenomeList(population)
        fitnesses_list = []

        # map(prepare_genomes, genome_list)
        for genome in genome_list:
            prepare_genomes(genome)

        if use_futures:
            executor = fut.MPIPoolExecutor()
            # fitnesses_list = executor.map(evaluate_futures, genome_list)
            for fitness in executor.map(evaluate_futures, genome_list):
                fitnesses_list.append(fitness)
        else:
            # fitnesses_list = map(evaluate, genome_list)
            for genome in genome_list:
                fitnesses_list.append(evaluate(genome))

        neat.ZipFitness(genome_list, fitnesses_list)

        population.GetBestGenome().Save('output/best_genome.txt')
        # mode = MPI.MODE_APPEND
        # genomefile = MPI.File.Open(comm, 'output/best_genome.txt', mode) 
        # genomefile.Write_ordered('\n' + str(population.GetBestGenome().GetNeuronTraits()) + 
        #                          '\n' + str(population.GetBestGenome().GetGenomeTraits()))
        # genomefile.Close()
        genomefile = open('output/best_genome.txt', 'a')
        genomefile.write('\n' + str(population.GetBestGenome().GetNeuronTraits()) + 
                         '\n' + str(population.GetBestGenome().GetGenomeTraits()))
        genomefile.close()
        # copytree('genome' + str(population.GetBestGenome().GetID()), 
        #          'output/generation' + str(generation_number) + '_best_genome')
        try:
            copytree('genome' + str(population.GetBestGenome().GetID()), 
                     'output/generation' + str(generation_number) + '_best_genome')
        except FileExistsError:
            print('folder generation' + str(generation_number) + '_best_genome exists')

        # outfile.Write_ordered(str(generation_number) + '\t' + str(max(fitnesses_list)) + '\n')
        outfile.write(str(generation_number) + '\t' + str(max(fitnesses_list)) + '\n')
        outfile.flush()
        # sys.stderr.write(
        #     '\rGeneration ' + str(generation_number)
        #     + ': fitness = ' + str(population.GetBestGenome().GetFitness())
        # )

        # advance to the next generation
        print("Generation " + str(generation_number) + \
              ": fitness = " + str(population.GetBestGenome().GetFitness()))
        print("Generation " + str(generation_number) + " finished")
        population.Epoch()
    # outfile.Close()
    outfile.close()


if __name__ == '__main__':
    use_futures = True
    main(use_futures)


