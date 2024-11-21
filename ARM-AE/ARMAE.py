
import click
import pandas as pd
from arm_ae.BenchmarkManager import *
from arm_ae.ARMAE import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np




@click.command()
@click.option(
    '--input-path', '-ip', 
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default='../data/nursery.csv', 
    help='the path of the input data',
    required = True
)

@click.option(
    '--armae-results-path', '-arp', 
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default='../Results/NN/', 
    help='the path of the results of ARM-AE algorithms',
    required = False
)


@click.option(
    '--nb_epoch', '-ne', 
    type=int,
    default=2, 
    help='the number of epoch of training  for the ARM-AE',
    required = False
)

@click.option(
    '--batch_size', '-bs', 
    type=int,
    default=128, 
    help='the batch_size for the  training of ARM-AE',
    required = False
)

@click.option(
    '--learning_rate', '-lr', 
    type=float,
    default=10e-3, 
    help='the learning rate for the  training of ARM-AE',
    required = False
)

@click.option(
    '--likeness', '-lk', 
    type=float,
    default=0.5, 
    help='the proportion of similar items in rule with the same consequents',
    required = False
)

@click.option(
    '--number_of_rules', '-nbor', 
    type=int,
    default=2, 
    help='The number of rule per consequent',
    required = False
)

@click.option(
    '--nb_antecedents', '-nba', 
    type=int,
    default=2, 
    help='The maximum number of antecedents',
    required = False
)

@click.option(
    '--is-Loaded-Model', '-ilm', 
    is_flag=True,
    help='do you want to train the ARM-AE model again',
    required = False
)

@click.option(
    '--model-path', '-mp', 
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default='models/', 
    help='the path to save the models',
    required = False
)

@click.option(
    '--encoder-path', '-ep', 
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help='if you want to use a pretrained model those are the path to update',
    required = False
)

@click.option(
    '--decoder-path', '-dp', 
    type=click.Path(exists=True, file_okay=True, dir_okay=False), 
    help='if you want to use a pretrained model those are the path to update',
    required = False,
)





def cli(input_path,armae_results_path,
        nb_epoch,batch_size,
        learning_rate,likeness,number_of_rules,
        nb_antecedents,
        is_loaded_model,model_path,encoder_path,
        decoder_path):
    
        datasetName = input_path.split('/')[-1].split('.')[0]
        
        data = pd.read_csv(input_path,index_col=0,dtype=float,header=0)
        
        dataSize = len(data.loc[0])
    
        NN = ARMAE(dataSize,maxEpoch=nb_epoch,batchSize=batch_size,learningRate=learning_rate, likeness=likeness)

        dataLoader = NN.dataPretraitement(data)
        if not is_loaded_model:
            NN.train(dataLoader,model_path)
        else:
            NN.load(encoder_path,decoder_path)


        NN.generateRules(data, numberOfRules=number_of_rules, nbAntecedent=nb_antecedents,
                                                        path=armae_results_path + datasetName +'.csv')
    
    


if __name__ == '__main__':
    cli()



