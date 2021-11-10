import argparse

from datetime import datetime
from keras.models import model_from_json, load_model, save_model

from utils import load_MNIST, load_CIFAR
from utils import filter_val_set, get_trainable_layers
from utils import generate_adversarial, filter_correct_classifications
from coverages.idc import ImportanceDrivenCoverage
from coverages.neuron_cov import NeuronCoverage
from coverages.tkn import DeepGaugeLayerLevelCoverage
from coverages.kmn import DeepGaugePercentCoverage
from coverages.ss import SSCover
from coverages.sa import SurpriseAdequacy
from coverages.doubt import Doubt
import numpy as np


__version__ = 0.9


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

def by_indices(outs, indices):
    return [[outs[i][0][indices]] for i in range(len(outs))]


def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """

    # define the program description
    text = 'Coverage Analyzer for DNNs'

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # add new command-line arguments
    parser.add_argument("-V", "--version",  help="show program version",
                        action="version", version="DeepFault %f" % __version__)
    parser.add_argument("-M", "--model", help="Path to the model to be loaded.\
                        The specified model will be used.")#, required=True)
                        # choices=['lenet1','lenet4', 'lenet5'], required=True)
    parser.add_argument("-DS", "--dataset", help="The dataset to be used (mnist\
                        or cifar10).", choices=["mnist","cifar10"])#, required=True)
    parser.add_argument("-A", "--approach", help="the approach to be employed \
                        to measure coverage", choices=['idc','nc','kmnc',
                        'nbc','snac','tknc','ssc', 'lsa', 'dsa', 'doubt', 'random'])
    parser.add_argument("-C", "--class", help="the selected class", type=int)
    parser.add_argument("-Q", "--quantize", help="quantization granularity for \
                        combinatorial other_coverage_metrics.", type= int)
    parser.add_argument("-L", "--layer", help="the subject layer's index for \
                        combinatorial cov. NOTE THAT ONLY TRAINABLE LAYERS CAN \
                        BE SELECTED", type= int)
    parser.add_argument("-KS", "--k_sections", help="number of sections used in \
                        k multisection other_coverage_metrics", type=int)
    parser.add_argument("-KN", "--k_neurons", help="number of neurons used in \
                        top k neuron other_coverage_metrics", type=int)
    parser.add_argument("-RN", "--rel_neurons", help="number of neurons considered\
                        as relevant in combinatorial other_coverage_metrics", type=int)
    parser.add_argument("-AT", "--act_threshold", help="a threshold value used\
                        to consider if a neuron is activated or not.", type=float)
    parser.add_argument("-R", "--repeat", help="index of the repeating. (for\
                        the cases where you need to run the same experiments \
                        multiple times)", type=int)
    parser.add_argument("-LOG", "--logfile", help="path to log file")
    parser.add_argument("-ADV", "--advtype", help="path to log file")


    # parse command-line arguments


    # parse command-line arguments
    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    args = parse_arguments()
    model_path     = args['model'] if args['model'] else 'neural_networks/LeNet5'
    dataset        = args['dataset'] if args['dataset'] else 'mnist'
    approach       = args['approach'] if args['approach'] else 'idc'
    num_rel_neurons= args['rel_neurons'] if args['rel_neurons'] else 6 # was 2
    act_threshold  = args['act_threshold'] if args['act_threshold'] else 0
    top_k          = args['k_neurons'] if args['k_neurons'] else 3
    k_sect         = args['k_sections'] if args['k_sections'] else 10 # in article - 1000
    selected_class = args['class'] if not args['class']==None else -1 #ALL CLASSES
    repeat         = args['repeat'] if args['repeat'] else 1
    logfile_name   = args['logfile'] if args['logfile'] else 'result.log'
    quantization_granularity = args['quantize'] if args['quantize'] else 3
    adv_type      = args['advtype'] if args['advtype'] else 'fgsm'

    logfile = open(logfile_name, 'a')

    ####################
    # 0) Load data
    if dataset == 'mnist':
            X_train, Y_train, X_test, Y_test = load_MNIST(channel_first=False)
            img_rows, img_cols = 28, 28
    else:
        X_train, Y_train, X_test, Y_test = load_CIFAR()
        img_rows, img_cols = 32, 32

    if not selected_class == -1:
        X_train, Y_train = filter_val_set(selected_class, X_train, Y_train) #Get training input for selected_class
        X_test, Y_test = filter_val_set(selected_class, X_test, Y_test) #Get testing input for selected_class
#     number = 10000
#     X_test, Y_test = X_test[:number], Y_test[:number]



    ####################
    # 1) Setup the model
    model_name = model_path.split('/')[-1]

    try:
        json_file = open(model_path + '.json', 'r') #Read Keras model parameters (stored in JSON file)
        file_content = json_file.read()
        json_file.close()

        model = model_from_json(file_content)
        model.load_weights(model_path + '.h5')

        # Compile the model before using
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    except:
        model = load_model(model_path + '.h5')

    # 2) Load necessary information
    trainable_layers = get_trainable_layers(model)
    non_trainable_layers = list(set(range(len(model.layers))) - set(trainable_layers))
    print('Trainable layers: ' + str(trainable_layers))
    print('Non trainable layers: ' + str(non_trainable_layers))

    experiment_folder = 'experiments'

    #Investigate the penultimate layer
    subject_layer = args['layer'] if not args['layer'] == None else -1
    subject_layer = trainable_layers[subject_layer]

    skip_layers = [0] #SKIP LAYERS FOR NC, KMNC, NBC etc.
    for idx, lyr in enumerate(model.layers):
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)

    print("Skipping layers:", skip_layers)

    ####################
    # 3) Analyze Coverages
    if approach == 'nc': # just set treshhold for each neurons and if activation neuron for test imput higher, than test input add

        nc = NeuronCoverage(model, threshold=.75, skip_layers = skip_layers) #SKIP ONLY INPUT AND FLATTEN LAYERS
        coverage, _, _, _, _, used_inp_index = nc.test(X_test[3*139:4*139]) # used_inp_index is the index which cover all neurons that was covered by full datasets
        print("Your test set's coverage is: ", coverage)
        _, orig_acc =  model.evaluate(X_test[3*139:4*139], Y_test[3*139:4*139])
        print("Accuracy: ",orig_acc)

#         nc.set_measure_state(nc.get_measure_state())
#         #print(used_inp_index)
#         #print(X_test[used_inp_index].shape)
#         _, orig_acc =  model.evaluate(X_test[used_inp_index], Y_test[used_inp_index])
#         print("Accuracy: ",orig_acc)
#         coverage, _, _, _, _, used_inp_index = nc.test(X_test[used_inp_index])
#         print("Your test set's coverage is: ", coverage)

    elif approach == 'idc': # define main neurons and than set multiplication of activation func of neuron
        #idc approach
        #print("\nRunning IDC for %d relevant neurons" % (num_rel_neurons))

        #X_train_corr, Y_train_corr, _, _, = filter_correct_classifications(model,
                                                                          # X_train,
                                                                          # Y_train)

        #idc = ImportanceDrivenCoverage(model, model_name, num_rel_neurons, selected_class,
                                       #subject_layer, X_train_corr, Y_train_corr)
        idc_idxs = []

#         coverage, covered_combinations, max_comb, idc_idxs = idc.test(X_test)
#         print("Analysed %d test inputs" % len(Y_train_corr))
#         print("IDC test set coverage: %.2f%% " % (coverage))
#         print("Covered combinations: ", len(covered_combinations))
#         print("Total combinations: ",   max_comb)
#         _, orig_acc =  model.evaluate(X_test, Y_test)
#         print("Accuracy: ",orig_acc)

        #idc.set_measure_state(covered_combinations)
        
        
        #nc approach
        nc = NeuronCoverage(model, threshold=.75, skip_layers = skip_layers) #SKIP ONLY INPUT AND FLATTEN LAYERS
        coverage, _, _, _, _, nc_idxs = nc.test(X_test) # nc_inxs is the index which cover all neurons that was covered by full datasets
        print("Your test set's coverage is: ", coverage)
#         _, orig_acc =  model.evaluate(X_test[3*139:4*139], Y_test[3*139:4*139])
#         print("Accuracy: ",orig_acc)
        res = {
            'model_name': model_name,
            'num_section': k_sect,
        }
        # kmnc, nbc, snac
        dg = DeepGaugePercentCoverage(model, k_sect, X_train, None, skip_layers)
        score = dg.test(X_test)
        kmn_idxs = score[-1]
        #nbc and snac
        low_used_inps = score[-3] 
        strong_used_inps = score[-2]
        
        
        #tknc approach
        dg = DeepGaugeLayerLevelCoverage(model, top_k, skip_layers=skip_layers)
        orig_coverage, _, _, _, _, orig_incrs, tknc_idxs = dg.test(X_test)
        print(orig_incrs)
        #print(used_inps)
        
        #lsa approach 
        upper_bound = 2000

        layer_names = [model.layers[-3].name]

        sa = SurpriseAdequacy(0, model, X_train, layer_names, upper_bound, dataset)

        lsa_idxs = []
#         coverage_lsa, lsa_idxs = sa.test(X_test, dataset, "lsa")
        #print(coverage_sa, idxs)
        
        coverage_dsa, dsa_idxs = sa.test(X_test, dataset, "dsa")
        #print(coverage_dsa, dsa_idxs)
        
        
        # put all together
        new_idxs = [idc_idxs, nc_idxs, kmn_idxs, low_used_inps, strong_used_inps, tknc_idxs, lsa_idxs, dsa_idxs]
        
        results_union = set().union(*new_idxs)
        print(len(results_union))
        #coverage, covered_combinations, max_comb, idc_idxs = idc.test(X_test[new_idxs])
#         print("IDC test set coverage: %.2f%% " % (coverage))
#         coverage, _, _, _, _, nc_inxs = nc.test(X_test[new_idxs])
#         print("Your test set's coverage is: ", coverage)
#         _, orig_acc =  model.evaluate(X_test, Y_test)
#         print("Accuracy full: ",orig_acc)
        _, orig_acc =  model.evaluate(X_test[list(results_union)], Y_test[list(results_union)])
        print("Accuracy full: ",orig_acc)
        

    elif approach == 'kmnc' or approach == 'nbc' or approach == 'snac': #https://arxiv.org/pdf/1803.07519.pdf 3.1
#sector devided on k_sect sections (during training define low and high boandary for each neurons) and count coverage all of that sections on every layer
        res = {
            'model_name': model_name,
            'num_section': k_sect,
        }

        dg = DeepGaugePercentCoverage(model, k_sect, X_train, None, skip_layers)
        score = dg.test(X_test)
        print(score[0], score[-1])

    elif approach == 'tknc': #https://arxiv.org/pdf/1803.07519.pdf 3.2
        # firstly count top k neurons in each layer and than count coverage of each neurons in top_k in each layer (see the code)

        dg = DeepGaugeLayerLevelCoverage(model, top_k, skip_layers=skip_layers)
        orig_coverage, _, _, _, _, orig_incrs, used_inps = dg.test(X_test)
        print(orig_incrs)
        print(used_inps)
        
        #_, orig_acc =  model.evaluate(X_test[:1000], Y_test[:1000])

    elif approach == 'ssc':

        ss = SSCover(model, skip_layers=non_trainable_layers)
        score = ss.test(X_test)

        print("Your test set's coverage is: ", score)

    elif approach == 'lsa' or approach == 'dsa':
        upper_bound = 2000

        layer_names = [model.layers[-3].name]

        #for lyr in model.layers:
        #    layer_names.append(lyr.name)
        print("HERE")
        sa = SurpriseAdequacy(0, model, X_train, layer_names, upper_bound, dataset)
        print("HERE")
        coverage_sa, idxs = sa.test(X_test, dataset, approach)
        print(coverage_sa, idxs)
    elif approach == 'doubt':
        doubt = Doubt(model, 128)
        inxs = doubt.test_the_most_doubt_and_not(X_test)
        _, orig_acc =  model.evaluate(X_test[list(inxs)], Y_test[list(inxs)])
        print("Accuracy full: ",orig_acc)
        
    elif approach == 'random':
        inxs = np.random.randint(X_test.shape[0], size=712)
        _, orig_acc =  model.evaluate(X_test[list(inxs)], Y_test[list(inxs)])
        print("Accuracy random: ", orig_acc)
        print(Y_test.shape)
        
        _, orig_acc =  model.evaluate(X_test, Y_test)
        print("Accuracy full: ",orig_acc)

    logfile.close()

