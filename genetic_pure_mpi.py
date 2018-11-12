import sys
from os import system, chdir, getcwd, mkdir
from shutil import copytree

import numpy
import json # to read parameters from file
import mpi4py
import MultiNEAT as neat
import traits
from iris_for_genetic import solve_task


def prepare_genomes(genome):
    current_trait_values = genome.GetGenomeTraits()
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
    # settings['n_coding_neurons'] = network_trait_values['n_coding_neurons']
    # settings['epochs'] = network_trait_values['epochs']
    # settings['h_time'] = network_trait_values['h_time']
    settings['noise_freq'] = network_trait_values['noise_freq']
    
    settings['neuron_out']['tau_minus'] = neuron_trait_values['tau_minus']
    settings['neuron_out']['C_m'] = neuron_trait_values['C_m']
    settings['neuron_out']['tau_m'] = neuron_trait_values['tau_m']
    settings['neuron_out']['t_ref'] = neuron_trait_values['t_ref']

    settings['syn_dict_stdp']['alpha'] = synapse_trait_values['alpha']
    settings['syn_dict_stdp']['lambda'] = synapse_trait_values['lambda']
    settings['syn_dict_stdp']['tau_plus'] = synapse_trait_values['tau_plus']

    mkdir('genome' + str(genome.GetID()))
    chdir('genome' + str(genome.GetID()))
    json.dump(settings, open('settings.json', 'w'), indent=4)
    chdir('..') # out of genomeID


def evaluate(genome):
    directory_name = getcwd() + '/genome' + str(genome.GetID()) + '/'
    print("Start sim in " + str(directory_name))
    fitness = solve_task(directory_name)
    print("Stop sim in " + str(directory_name))
    return fitness


# def get_evaluation_result(genome):
#     print('Evalutaion genome' + str(genome.GetID()))
#     chdir('genome' + str(genome.GetID()))
#     print(getcwd())
#     with open('fitness.txt', 'r') as fit_file:
#         fitness = float(fit_file.readline())
#     # fitness = float(numpy.loadtxt('fitness.txt'))
#     #desired_weights = numpy.loadtxt('../desired_weights/final_weights.txt')
#     #actual_weights = numpy.loadtxt('final_weights.txt')
#     print(fitness)
#     chdir('..') # out of genomeID
#     return fitness

#parallel_executor = concurrent.futures.ProcessPoolExecutor(max_workers=15)
# Change to MPI as below to run on a cluster.
#parallel_executor = mpi4py.futures.MPIPoolExecutor(max_workers=4)

# outfile = open('output/fitness.txt', 'w')

def main(workers=1):
    sys.stdout = open('out.txt', 'w')
    sys.stderr = open('err.txt', 'w')
    print("Prepare traits and genomes")

    neat_params = neat.Parameters()
    system("grep -v '//' < input/neat_parameters.txt | grep . > input/neat_parameters.filtered.txt")
    neat_params.Load('input/neat_parameters.txt')
    system("rm input/neat_parameters.filtered.txt")

    system("grep -v '//' < input/global_parameters.json | grep . > input/global_parameters.filtered.json")
    network_parameters = json.load(open('input/global_parameters.json', 'r'))
    system("rm input/global_parameters.filtered.json")

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
    print("Start solving generations with " + str(workers) + " workers")

    with open('output/fitness.txt', 'w') as outfile:
        for generation_number in range(2):
            print("Generation " + str(generation_number) + " started")
            genome_list = neat.GetGenomeList(population)
            for genome in genome_list:
                prepare_genomes(genome)
            # executor = fut.MPIPoolExecutor(max_workers=workers)
            # result = executor.map(evaluate, genome_list)
            # fitnesses_list = map(get_evaluation_result, genome_list)
            print(result)
            print("Waiting for result")
            fitnesses_list = list(result)
            print("Result ready")
            print(fitnesses_list)

            neat.ZipFitness(genome_list, fitnesses_list)
            if generation_number % network_parameters['result_watching_step'] == 0:
                population.GetBestGenome().Save('output/best_genome.txt')
                with open('output/best_genome.txt', 'a') as genomefile:
                    genomefile.write(
                        '\n' + str(population.GetBestGenome().GetNeuronTraits())
                        + '\n' + str(population.GetBestGenome().GetGenomeTraits())
                    )
                    genomefile.close()
                copytree('genome' + str(population.GetBestGenome().GetID()), 
                         'output/generation' + str(generation_number) + '_best_genome')

            outfile.write(
                str(generation_number)
                + '\t' + str(max(fitnesses_list))
                + '\n'
            )
            outfile.flush()
            sys.stderr.write(
                '\rGeneration ' + str(generation_number)
                + ': fitness = ' + str(population.GetBestGenome().GetFitness())
            )

            # advance to the next generation
            population.Epoch()
            print("Generation " + str(generation_number) + 
                  ": fitness = " + str(population.GetBestGenome().GetFitness()))
            print("Generation " + str(generation_number) + "finished")


if __name__ == '__main__':
    comm = mpi4py.MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        if len(sys.argv) > 1:
            main(int(sys.argv[1]))
        else:
            main()


