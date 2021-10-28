#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
import collections
import sys

import graphics
import numpy as np
import robot


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model


# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


# -----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    h=len(all_possible_hidden_states)
    m=np.zeros(h)
    n=np.zeros(h)
    # TODO: Compute the forward messages
    for i in range(num_time_steps-1):
        m=np.zeros(h)
        forward_messages[i+1]=robot.Distribution()
        if observations[i] == None:
            for j in range(h):        
                for k in forward_messages[i]: 
                    if all_possible_hidden_states[j] in transition_model(k):
                        m[j] += transition_model(k)[all_possible_hidden_states[j]]*forward_messages[i][k]
                if m[j]!=0:
                    forward_messages[i+1][all_possible_hidden_states[j]]=m[j]
        else:
            for j in range(h):        
                for k in forward_messages[i]: 
                    if all_possible_hidden_states[j] in transition_model(k):
                        if observations[i] in observation_model(k):
                            m[j] += observation_model(k)[observations[i]]*transition_model(k)[all_possible_hidden_states[j]]*forward_messages[i][k]      
                if m[j]!=0:
                    forward_messages[i+1][all_possible_hidden_states[j]]=m[j]
        forward_messages[i+1].renormalize()    
    
    backward_messages = [None] * num_time_steps
    backward_messages[0]=robot.Distribution()
    for j in all_possible_hidden_states:
        backward_messages[0][j] = 1
    backward_messages[0].renormalize()
    # TODO: Compute the backward messages
    for i in range(num_time_steps-1):
        n=np.zeros(h)
        backward_messages[i+1]=robot.Distribution()
        if observations[num_time_steps-1-i] == None:
            for j in range(h):
                for k in backward_messages[i]:        
                    if k in transition_model(all_possible_hidden_states[j]) :
                            n[j]+= transition_model(all_possible_hidden_states[j])[k]*backward_messages[i][k]
                if n[j]!=0:
                    backward_messages[i+1][all_possible_hidden_states[j]]=n[j]
        else:
            for j in range(h):
                for k in backward_messages[i]:        
                    if k in transition_model(all_possible_hidden_states[j]) :
                        if observations[num_time_steps-1-i] in observation_model(k):                        
                            n[j]+= observation_model(k)[observations[num_time_steps-1-i]]*transition_model(all_possible_hidden_states[j])[k]*backward_messages[i][k]
                if n[j]!=0:
                    backward_messages[i+1][all_possible_hidden_states[j]]=n[j]
        backward_messages[i+1].renormalize()
    marginals = [None] * num_time_steps # remove this
    # TODO: Compute the marginals    
    for i in range(num_time_steps): 
        marginals[i]=robot.Distribution()
        if observations[i] == None:
            for j in forward_messages[i] :
                if j in backward_messages[num_time_steps-1-i]:
                    marginals[i][j]= forward_messages[i][j]*backward_messages[num_time_steps-1-i][j]             
        else:    
            for j in forward_messages[i] :
                if j in backward_messages[num_time_steps-1-i]:
                    if observations[i] in observation_model(j):
                        marginals[i][j]= forward_messages[i][j]*observation_model(j)[observations[i]]*backward_messages[num_time_steps-1-i][j]      
        marginals[i].renormalize()
    return marginals


def Viterbi(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of estimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #

    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this
    messages = [None] * num_time_steps
    messages[0] = prior_distribution
    traces = [None] * (num_time_steps-1)
    middle={}
    middlemin=+np.inf
    minstate={}
    for i in range(num_time_steps-1):
        messages[i+1]=robot.Distribution()
        traces[i] = {}       
        if observations[i] == None:
            for j in all_possible_hidden_states: 
                middle={}
                middlemin=+np.inf
                minstate={}    
                for k in messages[i]: 
                    if j in transition_model(k):
                        middle[k] = -careful_log(transition_model(k)[j]) + messages[i][k]
                if middle != {} :
                    for k in middle:
                        if middlemin>=middle[k]:
                            middlemin=middle[k]
                            minstate=k
                    messages[i+1][j] = middlemin
                    traces[i][j] = minstate                
        else:
            for j in all_possible_hidden_states: 
                middle={}
                middlemin=+np.inf
                minstate={} 
                for k in messages[i]: 
                    if j in transition_model(k):
                        if observations[i] in observation_model(k):
                            middle[k] = -careful_log(transition_model(k)[j])-careful_log(observation_model(k)[observations[i]])+messages[i][k]
                if middle != {} :
                     for k in middle:
                        if middlemin>=middle[k]:
                            middlemin=middle[k]
                            minstate=k
                     messages[i+1][j] = middlemin
                     traces[i][j] = minstate          
    middle={}
    middlemin=+np.inf
    minstate={}
    if observations[num_time_steps-1] == None:
        for k in messages[num_time_steps-1]:
            if middlemin>=messages[num_time_steps-1][k]:
                middlemin=messages[num_time_steps-1][k]
                minstate=k 
        estimated_hidden_states[num_time_steps-1] = minstate
    else:
        for k in messages[num_time_steps-1]:
            if observations[num_time_steps-1] in observation_model(k):
                middle[k]= -careful_log(observation_model(k)[observations[num_time_steps-1]])+messages[num_time_steps-1][k]
        for k in middle:
            if middlemin>=middle[k]:
                middlemin=middle[k]
                minstate=k 
        estimated_hidden_states[num_time_steps-1] = minstate        
    for i in range(num_time_steps-1)[::-1]:
        estimated_hidden_states[i]=traces[i][estimated_hidden_states[i+1]]
        
    return estimated_hidden_states


def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of estimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #


    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps
    estimated_hidden_states2 = [None] * num_time_steps
    messages1 = [None] * num_time_steps
    messages1[0] = prior_distribution
    messages2 = [None] * num_time_steps
    messages2[0] = prior_distribution
    traces1 = [None] * (num_time_steps-1)
    traces2 = [None] * (num_time_steps-1)
    middle={}
    m={}
    result={}
    r=0
    middle={}
    middlemin=+np.inf
    middlemax=-np.inf
    minstate={}
    key=1
    est={}
    # TODO: Compute the forward messages
    for i in range(num_time_steps-1):
        messages1[i+1]=robot.Distribution()
        messages2[i+1]=robot.Distribution()
        traces1[i] = {}
        traces2[i] = {}
        if observations[i] == None:
            for j in all_possible_hidden_states: 
                middle={}
                middlemin=+np.inf
                minstate={}
                for k in messages1[i]: 
                    if j in transition_model(k):
                        middle[k] = -careful_log(transition_model(k)[j]) + messages1[i][k]                     
                if middle != {} :
                    for k in middle:
                        if middlemin>=middle[k]:
                            middlemin=middle[k]
                            minstate=k
                    messages1[i+1][j] = middlemin
                    traces1[i][j] = minstate                              
                    middle[minstate]=+np.inf
                    middlemin=+np.inf
                    minstate={}
                    for k in middle:
                        if middlemin>=middle[k]:
                            middlemin=middle[k]
                            minstate=k
                    if middlemin!=+np.inf:
                        messages2[i+1][j] = middlemin
                        traces2[i][j] = minstate
        else:
            for j in all_possible_hidden_states: 
                middle={}
                middlemin=+np.inf
                minstate={}
                for k in messages1[i]: 
                    if j in transition_model(k):
                        if observations[i] in observation_model(k):
                            middle[k] = -careful_log(transition_model(k)[j])-careful_log(observation_model(k)[observations[i]]) + messages1[i][k]
                if middle != {} :
                    for k in middle:
                        if middlemin>=middle[k]:
                            middlemin=middle[k]
                            minstate=k
                    messages1[i+1][j] = middlemin
                    traces1[i][j] = minstate                              
                    middle[minstate]=+np.inf
                    middlemin=+np.inf
                    minstate={}
                    for k in middle:
                        if middlemin>=middle[k]:
                            middlemin=middle[k]
                            minstate=k
                    if middlemin!=+np.inf:
                        messages2[i+1][j] = middlemin
                        traces2[i][j] = minstate
    middle={}
    middlemin=+np.inf
    minstate={}
    if observations[num_time_steps-1] == None:
        for k in messages1[num_time_steps-1]:
            if middlemin>=messages1[num_time_steps-1][k]:
                middlemin=messages1[num_time_steps-1][k]
                minstate=k 
        estimated_hidden_states[num_time_steps-1] = minstate        
        messages1[minstate]=+np.inf
        middlemin=+np.inf
        minstate={}
        for k in messages1[num_time_steps-1]:
            if middlemin>=messages1[num_time_steps-1][k]:
                middlemin=messages1[num_time_steps-1][k]
                minstate=k 
        if middlemin!=+np.inf:
            estimated_hidden_states2[num_time_steps-1] = minstate
    else:
        for k in messages1[num_time_steps-1]:
            if observations[num_time_steps-1] in observation_model(k):
                middle[k] = -careful_log(observation_model(k)[observations[num_time_steps-1]]) + messages1[num_time_steps-1][k]
        for k in middle:
            if middlemin>=middle[k]:
                middlemin=middle[k]
                minstate=k 
        estimated_hidden_states[num_time_steps-1] = minstate
        middle[minstate]=+np.inf
        middlemin=+np.inf
        minstate={}
        for k in middle:
            if middlemin>=middle[k]:
                middlemin=middle[k]
                minstate=k        
        if middlemin!=+np.inf:
            estimated_hidden_states2[num_time_steps-1] = minstate
    if estimated_hidden_states2[num_time_steps-1]!=None:
        for i in range(num_time_steps-1)[::-1]:
            if estimated_hidden_states2[i+1] in traces1[i]:
                estimated_hidden_states2[i]=traces1[i][estimated_hidden_states2[i+1]]
            else:
                estimated_hidden_states2={}
                break
    else:
        estimated_hidden_states2={}
    m[num_time_steps-1] = estimated_hidden_states2
    est = estimated_hidden_states[num_time_steps-1]
    for j in range(num_time_steps-1):
        m[j]={}
        key=1
        estimated_hidden_states = [None] * num_time_steps
        estimated_hidden_states[num_time_steps-1]=est
        for i in range(num_time_steps-1)[::-1]:
            if i!=j: 
                if traces1[i]!=None:
                    if estimated_hidden_states[i+1] in traces1[i]:
                        estimated_hidden_states[i]=traces1[i][estimated_hidden_states[i+1]]
                    else:
                        key=0
                        break
                else:
                    key=0
                    break
            else:
                if traces2[i]!=None:
                    if estimated_hidden_states[i+1] in traces2[i]:
                        estimated_hidden_states[i]=traces2[i][estimated_hidden_states[i+1]]
                    else:
                        key=0
                        break
                else:
                    key=0
                    break
        if key==1:
            m[j] = estimated_hidden_states
    for i in range(num_time_steps):
        if m[i]!={}:
            middle=1
            for j in range(num_time_steps-1):
                if observations[j] != None:
                    middle *= observation_model(m[i][j])[observations[j]]*transition_model(m[i][j])[m[i][j+1]]
                else:
                    middle *= transition_model(m[i][j])[m[i][j+1]]
            if observations[num_time_steps-1] == None: 
                result[i] = prior_distribution[m[i][0]]*middle
            else:
                result[i] = prior_distribution[m[i][0]]*middle*observation_model(m[i][num_time_steps-1])[observations[num_time_steps-1]]
    for k in result:
        if middlemax<=result[k]:
            middlemax=result[k]
            r=k 
    estimated_hidden_states = m[r]    
    return estimated_hidden_states


# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 2
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
           print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
