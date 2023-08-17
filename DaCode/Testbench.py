import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image

import time
import random

import torch
import torch.nn as nn
import torch.optim as optim

import gym

# 
# Setup environment
#

np.random.seed(69420)

numEpisodes = 500

numSteps = 60*60


environmentNames = [ 'MountainCar-v0', 'CartPole-v1', 'Acrobot-v1', 'LunarLander-v2' ]
environments = []

largestInputSize = 0
largestActionSize = 0

maxEnvironments = 4

for envIndex in range( 0, maxEnvironments ):
    envN = environmentNames[envIndex]
    env = gym.make( envN )
    observationSpace = env.observation_space
    print( f"observation space = {observationSpace}" )
    actionSpace = env.action_space
    
    inputSize = observationSpace.shape[0]
    actionSize = actionSpace.n

    if inputSize > largestInputSize:
        largestInputSize = inputSize
        pass

    if actionSize > largestActionSize:
        largestActionSize = actionSize
        pass

    environments.append( ( env, inputSize, actionSize ) )
    pass

print( f"largest input size = {largestInputSize}" )


#env = gym.make( 'MountainCar-v0' )
#env = gym.make( 'CartPole-v1' )
#env = gym.make( 'Acrobot-v1' )

# These work 
#env = gym.make( "BeamRider-v4" )
#env = gym.make( "VideoPinball-v4" )

# These are not being imported for some reason.
#env = gym.make( "Tetris-v4" )
#env = gym.make( "Pacman-v4" )
#env = gym.make( "BasicMath-v4" )
#env = gym.make( "VideoCheckers-v4" )
#env = gym.make( "VideoChess-v4" )
#env = gym.make( "DonkeyKong-v0" )


#
# Setup agent:
#


OptionAgentQ = 0
OptionAgentQContext = 1
OptionDaBrain = 2

useOption = OptionAgentQ

# Agent that uses Q function.

if useOption == OptionAgentQ:
    import AgentQ
    agent = AgentQ.AgentQ( 101 )
    agent.setArch( layerSizes=[largestInputSize + 1, 64, 80, 48, 1], learningRate=0.15, activation=nn.ReLU(), replayBufferSize=2**6, dropoutRate=0.0, epsilonExploration = 0.15 )
    replaySampleBatch = 0
    pass

# Agent that uses Q function and multiplexer based context gating.
elif useOption == OptionAgentQContext:
    import AgentQContext
    agent = AgentQContext.AgentQCG( 101 )
    agent.setArch( numContexts=len(environments), layerSizes=[largestInputSize + 1, 16, 20, 12, 1], learningRate=0.15, activation=nn.ReLU(), replayBufferSize=2**6, dropoutRate=0.00, epsilonExploration = 0.15 )
    replaySampleBatch = 0
    pass

elif useOption == OptionDaBrain:
    import DaBrain
    agent = DaBrain.DaBrain( 101 )
    agent.setArch( inputSize=largestInputSize, numStateEnsembles=8, stateHiddenLayerSize=64, stateEnsembleOutputSize=20, featureHiddenLayerSize=20, numStateFeatures=8, numOutputEnsembles=len( environments ), outputEnsembleHiddenLayerSize=64, outputEnsembleFeatures=1, learningRate=0.05, epsilon=0.15 )
    numIterationsForFeedback = 4
    pass

#
# Actual stuff:
#

#for name, param in agent.model.named_parameters():
#    if param.requires_grad:
#        print( f"Layer:{name}, Shape: {param.shape}" )
#        print( param.data )
#        pass
#    pass

enableProgressiveTests = False

rewardDecay = 0.9

for context in range( len( environments ) ):

    print( f"context = {context}" )

    for episode in range( numEpisodes+1 ):

        if episode % 100 == 0:
            print( f"Episode = {episode}" )
            pass

        #context = random.randint( 0, len(environments) - 1 )
    
        gameThisTime = environments[ context ]

        env = gameThisTime[0]
        #print( f"playing {env}" )
        numInputs = gameThisTime[1]
        numActions = gameThisTime[2]

        rawobs = env.reset()

        decayingReward = 0

        padAmount = largestInputSize - numInputs
        obsButNp = np.pad( rawobs, (0, largestInputSize - numInputs ) )
        obs = torch.tensor( obsButNp, dtype=torch.float )

        if useOption == OptionDaBrain:
            agent.resetThalamus()
            pass

        for step in range( numSteps ):

            if useOption == OptionAgentQ:
                action, expectedReward = agent.pickAction( obs, numActions )
                pass
            elif useOption == OptionAgentQContext:
                action, expectedReward = agent.pickAction( obs, numActions, context )
                pass
            elif useOption == OptionDaBrain:
                action, expectedReward = agent.pickAction( obs, numActions, context, numIterationsForFeedback )
                pass
    
            rawNewObs, reward, done, info = env.step( action )

            newobs = torch.tensor( np.pad( rawNewObs, (0, largestInputSize - numInputs) ), dtype=torch.float )

            # Using a better reward function since most of the ones provided kind of suck for training.
            encouragingReward = 0

            # MountainCar
            if context == 0:
                encouragingReward = 100 * ( rawNewObs[0] - 0.6 )
                pass
            # CartPole:
            elif context == 1:
                poleAngle = newobs[2]
                poleAngVel = newobs[3]
                encouragingReward = 0.2/(1 + poleAngle**2) + 1/( 1 + poleAngVel**2 )
                pass
            # Acrobot:
            elif context == 2:
                cosTheta1 = newobs[0]
                sinTheta1 = newobs[1]
                cosTheta2 = newobs[2]
                sinTheta2 = newobs[3]
                encouragingReward = -cosTheta1 + cosTheta1*cosTheta2 - sinTheta1*sinTheta2
                pass
            # LunarLander:
            elif context == 3:
                encouragingReward = reward
                pass

            #decayingReward = rewardDecay*decayingReward + encouragingReward
            decayingReward = max( decayingReward, encouragingReward )

            rewardDifference = ( decayingReward - expectedReward )**2

            if useOption == OptionAgentQ:
                agent.pushExperience( obs, action, decayingReward, newobs )
                agent.updateWeights( rewardDifference )
                if step % 16 == 0:
                    agent.replayExperience( replaySampleBatch )
                    pass
                pass
            elif useOption == OptionAgentQContext:
                agent.pushExperience( obs, action, decayingReward, newobs, context )
                agent.updateWeights( rewardDifference, context )
                if step % 16 == 0:
                    agent.replayExperience( replaySampleBatch )
                    pass
                pass
            elif useOption == OptionDaBrain:
                agent.updateWeights( rewardDifference, numIterationsForFeedback, context )
                pass
    
            if done:
                break

            obs = newobs
    
            pass
    
        if ( useOption == OptionAgentQ ) or ( useOption == OptionAgentQContext ):
            agent.resetDropout()
            pass

        if enableProgressiveTests and ( episode == 500 ):
            print( "Demonstrating all progress across games:" )

            for context in range( 0, len( environments ) ):
                game = environments[context]
                env = game[0]
                numInputs = game[1]
                numActions = game[2]
                rawobs = env.reset()

                if useOption == OptionDaBrain:
                    agent.resetThalamus()
                    pass
            
                obs = torch.tensor( np.pad( rawobs, (0, largestInputSize - numInputs) ), dtype=torch.float )

                for step in range( numSteps ):

                    startTime = time.time()

                    if useOption == OptionAgentQ:
                        action, expectedReward = agent.pickAction( obs, numActions, exploration=False )
                        pass
                    elif useOption == OptionAgentQContext:
                        action, expectedReward = agent.pickAction( obs, numActions, context, exploration=False )
                        pass
                    elif useOption == OptionDaBrain:
                        action, expectedReward = agent.pickAction( obs, numActions, context, numIterationsForFeedback, exploration=False )
                        pass

                    rawNewObs, reward, done, info = env.step( action )

                    newobs = torch.tensor( np.pad( rawNewObs, (0, largestInputSize - numInputs) ), dtype=torch.float )

                    env.render( mode="rgb" )
                    endTime = time.time()

                    sleepTime = max( 1/60 - (endTime - startTime), 0 )

                    time.sleep( sleepTime )

                    if done:
                        env.close()
                        break

                    obs = newobs
                    pass
                pass
            pass

        if ( useOption == OptionAgentQ ) or ( useOption == OptionAgentQContext ):
            agent.replayExperience( 128 )
            pass
            
        pass

# Testing post training:

    cumulativeRewards = [ 0 ] * len( environments )

    for test in range( 100 ):

        for context in range( 0, len( environments ) ):
            game = environments[context]
            env = game[0]
            numInputs = game[1]
            numActions = game[2]
            rawobs = env.reset()

            if useOption == OptionDaBrain:
                agent.resetThalamus()
                pass
            
            obs = torch.tensor( np.pad( rawobs, (0, largestInputSize - numInputs) ), dtype=torch.float )

            for step in range( numSteps ):

                #startTime = time.time()

                if useOption == OptionAgentQ:
                    action, expectedReward = agent.pickAction( obs, numActions, exploration=False )
                    pass
                elif useOption == OptionAgentQContext:
                    action, expectedReward = agent.pickAction( obs, numActions, context, exploration=False )
                    pass
                elif useOption == OptionDaBrain:
                    action, expectedReward = agent.pickAction( obs, numActions, context, numIterationsForFeedback, exploration=False )
                    pass

                rawNewObs, reward, done, info = env.step( action )

                # Using a better reward function since most of the ones provided kind of suck for training.
                encouragingReward = 0

                # MountainCar
                if context == 0:
                    encouragingReward = 100 * ( rawNewObs[0] - 0.6 )
                    pass
                # CartPole:
                elif context == 1:
                    poleAngle = newobs[2]
                    poleAngVel = newobs[3]
                    encouragingReward = 0.2/(1 + poleAngle**2) + 1/( 1 + poleAngVel**2 )
                    pass
                # Acrobot:
                elif context == 2:
                    cosTheta1 = newobs[0]
                    sinTheta1 = newobs[1]
                    cosTheta2 = newobs[2]
                    sinTheta2 = newobs[3]
                    encouragingReward = -cosTheta1 + cosTheta1*cosTheta2 - sinTheta1*sinTheta2
                    pass
                # LunarLander:
                elif context == 3:
                    encouragingReward = reward
                    pass

                cumulativeRewards[context] += encouragingReward

                newobs = torch.tensor( np.pad( rawNewObs, (0, largestInputSize - numInputs) ), dtype=torch.float )

                #env.render( mode="rgb" )
                #endTime = time.time()

                #sleepTime = max( (endTime - startTime) - 1/30, 0 )

                #time.sleep( sleepTime )

                if done:
                    env.close()
                    break
                pass

            obs = newobs
            pass
        pass

    averageRewards = [ reward/100 for reward in cumulativeRewards ]

    print( f"Average Rewards per game = {averageRewards}" )
    pass


for game in environments:
    env = game[0]
    env.close()
    pass

print( f"done" )

