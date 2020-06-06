# coding: utf-8

import sys
import pickle
from shutil import copytree
from os import system, chdir, getcwd, mkdir

import json  # to read parameters from file
import MultiNEAT as neat
import mpi4py.futures as fut
from mpi4py import MPI

from spiking_network_learning_algorithm import traits
from spiking_network_learning_algorithm.solver import solve_task


def prepare_genomes(genome):
    current_trait_values = genome.GetGenomeTraits()

    print('==================')
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
    print(getcwd())
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
        print('folder genome ' + str(genome.GetID()) + ' exists')

    chdir('genome' + str(genome.GetID()))
    json.dump(settings, open('settings.json', 'w'), indent=4)
    # chdir('../..')  # out of genomeID
    chdir('..')  # out of genomeID


def evaluate(genome):
    directory_name = getcwd() + '/genome' + str(genome.GetID()) + '/'
    print("Start sim in " + str(directory_name))
    fitness, res = solve_task(directory_name)
    print("Stop sim in " + str(directory_name))
    return fitness


def evaluate_futures(genome):
    directory_name = getcwd() + '/genome' + str(genome.GetID()) + '/'
    print("Start sim in " + str(directory_name))
    system("cd " + directory_name + " ;"
           "python genetic_solver.py " + directory_name + "; ")
    with open('fitness.txt') as fitness_file:
        fitness = fitness_file.readline()
    chdir('../..')
    print("Stop sim in " + str(directory_name))
    return fitness


def main(use_futures, continue_genetic=False, redirect_out=True):
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()

    # mode = MPI.MODE_CREATE | MPI.MODE_WRONLY

    if redirect_out:
        sys.stdout = open('out.txt', 'w')
        sys.stderr = open('err.txt', 'w')

    network_parameters = json.load(open('input/global_parameters.json', 'r'))

    if continue_genetic:
        population = pickle.load(open('output/last_population.pkl', 'rb'))
    else:
        population = create_population(network_parameters)

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
        fitness_list = []

        # map(prepare_genomes, genome_list)
        for genome in genome_list:
            prepare_genomes(genome)

        if use_futures:
            executor = fut.MPIPoolExecutor()
            for fitness in executor.map(evaluate_futures, genome_list):
                fitness_list.append(fitness)
        else:
            for genome in genome_list:
                fitness_list.append(evaluate(genome))

        neat.ZipFitness(genome_list, fitness_list)

        population.GetBestGenome().Save('output/best_genome.txt')
        # mode = MPI.MODE_APPEND
        # genome_file = MPI.File.Open(comm, 'output/best_genome.txt', mode)
        # genome_file.Write_ordered('\n' + str(population.GetBestGenome().GetNeuronTraits()) +
        #                          '\n' + str(population.GetBestGenome().GetGenomeTraits()))
        # genome_file.Close()
        genome_file = open('output/best_genome.txt', 'a')
        genome_file.write('\n' + str(population.GetBestGenome().GetNeuronTraits()) +
                          '\n' + str(population.GetBestGenome().GetGenomeTraits()))
        genome_file.close()
        # copytree('genome' + str(population.GetBestGenome().GetID()), 
        #          'output/generation' + str(generation_number) + '_best_genome')
        try:
            copytree('genome' + str(population.GetBestGenome().GetID()), 
                     'output/generation' + str(generation_number) + '_best_genome')
        except FileExistsError:
            print('folder generation' + str(generation_number) + '_best_genome exists')

        # outfile.Write_ordered(str(generation_number) + '\t' + str(max(fitness_list)) + '\n')
        outfile.write(str(generation_number) + '\t' + str(max(fitness_list)) + '\n')
        outfile.flush()
        # sys.stderr.write(
        #     '\rGeneration ' + str(generation_number)
        #     + ': fitness = ' + str(population.GetBestGenome().GetFitness())
        # )

        # advance to the next generation
        print("Generation " + str(generation_number) +
              ": fitness = " + str(population.GetBestGenome().GetFitness()))
        print("Generation " + str(generation_number) + " finished")
        population.Epoch()
        pickle.dump(population, open('output/last_population.pkl', 'wb'))
    # outfile.Close()
    outfile.close()


def create_population(network_parameters):
    print("Prepare traits and genomes")
    neat_params = neat.Parameters()
    system("grep -v '//' < input/neat_parameters.txt | grep . > input/neat_parameters.filtered.txt")
    neat_params.Load('input/neat_parameters.filtered.txt')
    system("rm input/neat_parameters.filtered.txt")
    # system("grep -v '//' < input/global_parameters.json | grep . > input/global_parameters.filtered.json")
    # network_parameters = json.load(open('input/global_parameters.filtered.json', 'r'))
    # system("rm input/global_parameters.filtered.json")
    # mode = MPI.MODE_RDONLY
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
        0,  # Some genome ID, I don't know what it means.
        network_parameters['inputs_number'],
        2,  # ignored for seed_type == 0, specifies number of hidden units if seed_type == 1
        network_parameters['outputs_number'],
        False,  # fs_neat. If == 1, a minimalistic perceptron is created: each output is connected
        # to a random input and the bias.
        neat.ActivationFunction.UNSIGNED_SIGMOID,  # output neurons activation function
        neat.ActivationFunction.UNSIGNED_SIGMOID,  # hidden neurons activation function
        0,  # seedtype
        neat_params,  # global parameters object returned by neat.Parameters()
        0,  # number of hidden layers
        0
    )
    population = neat.Population(
        genome,
        neat_params,
        True,  # whether to randomize the population
        0.5,  # how much to randomize
        0  # the RNG seed
    )
    return population


if __name__ == '__main__':
    use_fut = False
    continue_population = False
    main(use_fut)

