import numpy as np

import random

import torch
import torch.nn as nn
import torch.optim as optim


class AgentQCG(nn.Module):

    def __init__(self, seed):
        super( AgentQCG, self ).__init__()
        torch.manual_seed(seed)
        return

    def setArch( self, numContexts, layerSizes, learningRate, activation=nn.ReLU(), replayBufferSize=0, dropoutRate=0.0, epsilonExploration = 0.01 ):
        if replayBufferSize > 0:
            self.replayBuffer = []
            self.maxReplaySize = replayBufferSize
            pass
        else:
            self.maxReplaySize = 0
            pass

        self.models = []
        self.optimizers = []

        for i in range( 0, numContexts ):

            layers = []
            numLayers = len( layerSizes )
            for layer in range( 0, numLayers-2 ):
                layers.append( nn.Linear( layerSizes[layer], layerSizes[layer+1] ) )
                layers.append( activation )
                layers.append( nn.Dropout( dropoutRate ) )
                pass

            self.learningRate = learningRate

            layers.append( nn.Linear( layerSizes[-2], layerSizes[-1] ) )
        
            self.models.append( nn.Sequential( *layers ) )

            self.optimizers.append( optim.SGD( self.models[i].parameters(), lr=self.learningRate ) )

            pass

        self.epsilonGreed = epsilonExploration

        return

    # For dropout:
    def resetDropout( self ):
        for model in self.models:
            for layer in model.children():
                if isinstance( layer, nn.Dropout ):
                    layer.train()
                    pass 
                pass
            pass
        return

    def forward( self, x, context ):
        return self.models[context]( x )

    # For experience buffer:
    def pushExperience( self, obs, action, reward, obsN, context ):

        if self.maxReplaySize == 0:
            return

        if len( self.replayBuffer ) >= self.maxReplaySize:
            self.replayBuffer.pop(0)
            pass
        
        self.replayBuffer.append( ( obs, action, reward, obsN, context ) )
        
        return
    
    def sampleExperience( self, times ):
        currentLength = len( self.replayBuffer )
        if currentLength == 0:
            return None
        result = [ self.replayBuffer[random.randint(0, currentLength-1)] for t in range( 0, times ) ]
        return result

    def pickAction( self, observation, numActions, context, exploration=True ):

        bestReward = -1
        bestAction = -1

        if exploration and ( np.random.uniform() < self.epsilonGreed ):
            bestAction = np.random.randint( 0, high=numActions )
            leInput = torch.cat( ( torch.tensor( [bestAction], dtype=torch.float ), observation ), dim=-1 )
            bestReward = self.forward( leInput, context ).item()
            pass

        else:
            for i in range( 0, numActions ):
                
                leInput = torch.cat( ( observation, torch.tensor( [i], dtype=torch.float ) ), dim=-1 )
                projectedReward = self.forward( leInput, context ).item()
                
                if projectedReward > bestReward:
                    bestAction = i
                    bestReward = projectedReward
                    pass
                pass
            pass
        return bestAction, bestReward

    def updateWeights( self, reward, context ):
        self.optimizers[context].zero_grad()
        loss = torch.tensor( -reward, requires_grad=True )
        loss.backward()
        self.optimizers[context].step()
        return

    def replayExperience( self, times ):
        if self.maxReplaySize == 0:
            return

        for i in range( 0, times ):
            experiences = self.sampleExperience( 1 )
            experience = experiences[0]
            observation = experience[0] 
            action = experience[1]
            acReward = experience[2]
            context = experience[4]
            leInput = torch.cat( ( torch.tensor( [action], dtype=torch.float ), observation ), dim=-1 )
            projectedReward = self.forward( leInput, context )
            self.updateWeights( (acReward - projectedReward)**2, context )
            pass
        return
    pass





