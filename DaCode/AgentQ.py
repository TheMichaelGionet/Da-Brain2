import numpy as np

import random

import torch
import torch.nn as nn
import torch.optim as optim


class AgentQ(nn.Module):

    def __init__(self, seed):
        super( AgentQ, self ).__init__()
        torch.manual_seed(seed)
        return

    def setArch( self, layerSizes, learningRate, activation=nn.ReLU(), replayBufferSize=0, dropoutRate=0.0, epsilonExploration = 0.01 ):
        if replayBufferSize > 0:
            self.replayBuffer = []
            self.maxReplaySize = replayBufferSize
            pass
        else:
            self.maxReplaySize = 0
            pass

        self.layers = []
        numLayers = len( layerSizes )
        for layer in range( 0, numLayers-2 ):
            self.layers.append( nn.Linear( layerSizes[layer], layerSizes[layer+1] ) )
            self.layers.append( activation )
            self.layers.append( nn.Dropout( dropoutRate ) )
            pass

        self.learningRate = learningRate

        self.layers.append( nn.Linear( layerSizes[-2], layerSizes[-1] ) )

        #print( "Building neural network with structure:" )
        #for layer in self.layers:
        #    print( f"{layer}" )
        #    pass
        
        self.model = nn.Sequential( *self.layers )

        self.optimizer = optim.SGD( self.model.parameters(), lr=self.learningRate )

        self.epsilonGreed = epsilonExploration

        return

    # For dropout:
    def resetDropout( self ):
        for layer in self.model.children():
            if isinstance( layer, nn.Dropout ):
                layer.train()
                pass 
            pass
        return

    def forward( self, x ):
        return self.model( x )

    # For experience buffer:
    def pushExperience( self, obs, action, reward, obsN ):

        if self.maxReplaySize == 0:
            return

        if len( self.replayBuffer ) >= self.maxReplaySize:
            self.replayBuffer.pop(0)
            pass
        
        self.replayBuffer.append( ( obs, action, reward, obsN ) )
        
        return
    
    def sampleExperience( self, times ):
        currentLength = len( self.replayBuffer )
        if currentLength == 0:
            return None
        result = [ self.replayBuffer[random.randint(0, currentLength-1)] for t in range( 0, times ) ]
        return result

    def pickAction( self, observation, numActions, exploration=True ):

        bestReward = -1
        bestAction = -1

        if exploration and ( np.random.uniform() < self.epsilonGreed ):
            bestAction = np.random.randint( 0, high=numActions )
            leInput = torch.cat( ( torch.tensor( [bestAction], dtype=torch.float ), observation ), dim=-1 )
            bestReward = self.forward( leInput ).item()
            pass

        else:
            for i in range( 0, numActions ):
                
                leInput = torch.cat( ( observation, torch.tensor( [i], dtype=torch.float ) ), dim=-1 )
                projectedReward = self.forward( leInput ).item()

                print( f"projectedReward = {projectedReward}" )
                
                if projectedReward > bestReward:
                    bestAction = i
                    bestReward = projectedReward
                    pass
                pass
            pass
        return bestAction, bestReward

    def updateWeights( self, reward ):

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print( f"Layer:{name}, Shape: {param.shape}" )
                print( param.data )
                pass
            pass

        self.optimizer.zero_grad()
        loss = torch.tensor( -reward, requires_grad=True )
        print( f"loss = {loss}" )
        loss.backward()
        self.optimizer.step()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print( f"Layer:{name}, Shape: {param.shape}" )
                print( param.data )
                pass
            pass

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
            leInput = torch.cat( ( torch.tensor( [action], dtype=torch.float ), observation ), dim=-1 )
            projectedReward = self.forward( leInput )
            self.updateWeights( (acReward - projectedReward)**2 )
            pass
        return
    pass





