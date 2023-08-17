import numpy as np

import random

import torch
import torch.nn as nn
import torch.optim as optim


class DaBrain( nn.Module ):
    
    def __init__(self, seed):
        super( DaBrain, self ).__init__()
        torch.manual_seed(seed)
        return
    
    def setArch( self, inputSize, numStateEnsembles, stateHiddenLayerSize, stateEnsembleOutputSize, featureHiddenLayerSize, numStateFeatures, numOutputEnsembles, outputEnsembleHiddenLayerSize, outputEnsembleFeatures, learningRate, epsilon = 0.01 ):
        self.learningRate = learningRate
        self.epsilonGreed = epsilon

        self.stateEnsembles = []
        self.stateFeatureEnsembles = []
        self.outputEnsembles = []

        self.numStateEnsembles = numStateEnsembles
        self.numOutputEnsembles = numOutputEnsembles
        
        self.thalamusSpaceSize = numStateFeatures * numStateEnsembles
        totalInputSize = inputSize + self.thalamusSpaceSize

        # Initialize the state ensembles
        for i in range( 0, numStateEnsembles ):
            inputLayer = nn.Linear( totalInputSize, stateHiddenLayerSize )
            nl = nn.Softplus()
            outputLayer = nn.Linear( stateHiddenLayerSize, stateEnsembleOutputSize )

            stateModel = nn.Sequential( inputLayer, nl, outputLayer )

            stateOpt = optim.SGD( stateModel.parameters(), lr=learningRate )

            self.stateEnsembles.append( (stateModel, stateOpt) )
            pass

        # Initialize the feature projection ensembles
        for i in range( 0, numStateEnsembles ):
            inputLayer = nn.Linear( stateEnsembleOutputSize, featureHiddenLayerSize )
            nl = nn.ReLU()
            outputLayer = nn.Linear( featureHiddenLayerSize, numStateFeatures )

            featureModel = nn.Sequential( inputLayer, nl, outputLayer )

            featureOpt = optim.SGD( featureModel.parameters(), lr=learningRate )

            self.stateFeatureEnsembles.append( (featureModel, featureOpt) )
            pass

        # Initialize the output ensembles

        for i in range( 0, numOutputEnsembles ):

            # + 1 to input the action as well as the state ensemble output size.
            inputLayer = nn.Linear( stateEnsembleOutputSize + 1, outputEnsembleHiddenLayerSize )
            nl = nn.ReLU()
            outputLayer = nn.Linear( outputEnsembleHiddenLayerSize, outputEnsembleFeatures )

            outputModel = nn.Sequential( inputLayer, nl, outputLayer )

            outputOpt = optim.SGD( outputModel.parameters(), lr=learningRate )

            self.outputEnsembles.append( (outputModel, outputOpt) )
            pass

        return

    def resetThalamus( self ):
        self.savedThalamusFeatures = np.zeros( self.thalamusSpaceSize )
        return


    def forward( self, inputs, action, context, numIterations ):
        if numIterations < 1:
            print( f"num iterations was < 1 in DaBrain.compute {0/0}" )
            exit()
            pass
        
        initializeThalamusFeatures = self.savedThalamusFeatures

        initializeStateOutputs = [None] * self.numStateEnsembles

        thalamusFeatures = initializeThalamusFeatures
        stateValues = initializeStateOutputs

        for iteration in range( 0, numIterations ):
            
            leInput = torch.cat( ( torch.tensor( thalamusFeatures, dtype=torch.float ), inputs ), dim=-1 )
            
            # Feed through state computation
            for stateIndex in range( 0, self.numStateEnsembles ):
                leSubnet = self.stateEnsembles[stateIndex][0]
                
                stateValues[stateIndex] = leSubnet( leInput )
                pass

            # Feed through feature computation
            computedFeatures = [None]*self.numStateEnsembles
            for featureIndex in range( 0, self.numStateEnsembles ):
                leSubnet = self.stateFeatureEnsembles[featureIndex][0]

                featuresAsTensor = leSubnet( stateValues[featureIndex] )
                computedFeatures[featureIndex] = featuresAsTensor.detach().numpy()
                pass

            # Collect features
            thalamusFeatures = np.concatenate( computedFeatures )
            pass

        self.savedThalamusFeatures = thalamusFeatures

        # Compute output from associated state:

        leOutputEnsemble = self.outputEnsembles[context][0]
        leData = stateValues[context]

        leInput = torch.cat( ( torch.tensor( [action], dtype=torch.float ), leData ), dim=-1 )

        reward = leOutputEnsemble( leInput ).item()

        return reward

    def pickAction( self, inputs, numActions, context, numIterations, exploration=True ):

        bestReward = -1
        bestAction = -1

        if exploration and ( np.random.uniform() < self.epsilonGreed ):
            bestAction = np.random.randint( 0, high=numActions )
            bestReward = self.forward( inputs, bestAction, context, numIterations )
            pass
        else:
            for action in range( 0, numActions ):
                reward = self.forward( inputs, action, context, numIterations )
                if reward > bestReward:
                    bestAction = action
                    bestReward = reward
                    pass
                pass
            pass
        
        return bestAction, bestReward

    def updateWeights( self, reward, numIterations, context ):
        
        leOutputEnsemblePair = self.outputEnsembles[context]
        leOutputEnsemble = leOutputEnsemblePair[0]
        leOutputOptimizer = leOutputEnsemblePair[1]

        # Zero all grads:

        leOutputOptimizer.zero_grad()

        for stateIter in range( 0, self.numStateEnsembles ):
            leOpt = self.stateEnsembles[stateIter][1]
            leOpt.zero_grad()
            pass

        for featIter in range( 0, self.numStateEnsembles ):
            leOpt = self.stateFeatureEnsembles[featIter][1]
            leOpt.zero_grad()
            pass

        # Propagate the loss:
        lastStateEnsembleOpt = self.stateEnsembles[context][1]
        lastStateEnsembleOpt.step()

        for iteration in range( 1, numIterations ):
            
            for featureEnsemblePair in self.stateFeatureEnsembles:
                leOpt = featureEnsemblePair[1]
                leOpt.step()
                pass

            for stateEnsemblePair in self.stateEnsembles:
                leOpt = stateEnsemblePair[1]
                leOpt.step()
                pass

            pass

        return

    pass
