import numpy as np 
from scipy import linalg

def mdp_computePR(P,R):


    # mdp_computePR  Computes the reward for the system in one state 
    #                chosing an action
    # Arguments --------------------------------------------------------------
    # Let S = number of states, A = number of actions
    #   P(SxSxA)  = transition matrix 
    #              P could be an array with 3 dimensions or 
    #              a cell array (1xA), each cell containing a matrix (SxS) possibly sparse
    #   R(SxSxA) or (SxA) = reward matrix
    #              R could be an array with 3 dimensions (SxSxA) or 
    #              a cell array (1xA), each cell containing a sparse matrix (SxS) or
    #              a 2D array(SxA) possibly sparse  
    # Evaluation -------------------------------------------------------------
    #   PR(SxA)   = reward matrix

    # MDPtoolbox: Markov Decision Processes Toolbox
    # Copyright (C) 2009  INRA
    # Redistribution and use in source and binary forms, with or without modification, 
    # are permitted provided that the following conditions are met:
    #    * Redistributions of source code must retain the above copyright notice, 
    #      this list of conditions and the following disclaimer.
    #    * Redistributions in binary form must reproduce the above copyright notice, 
    #      this list of conditions and the following disclaimer in the documentation 
    #      and/or other materials provided with the distribution.
    #    * Neither the name of the <ORGANIZATION> nor the names of its contributors 
    #      may be used to endorse or promote products derived from this software 
    #      without specific prior written permission.
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
    # ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
    # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
    # IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
    # INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
    # BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
    # DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
    # LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
    # OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
    # OF THE POSSIBILITY OF SUCH DAMAGE.


    PR = R
    return PR


def mdp_computePpolicyPRpolicy(P, R, policy):
    A = P.shape[2]

    Ppolicy = np.zeros([policy.shape[0], P.shape[1]])
    PRpolicy = np.zeros(policy.shape[0])

    for a in range(A):
        ind = policy == a # the rows that use action a
        if ind.sum() != 0:
            Ppolicy[ind, :] = P[ind,:,a]
            PR = R
            PRpolicy[ind] = PR[ind, a]
        
    return [Ppolicy, PRpolicy]

def mdp_eval_policy_matrix(P, R, discount, policy):

    S = P.shape[0]
    A = P.shape[2]

    [Ppolicy, PRpolicy] = mdp_computePpolicyPRpolicy(P, R, policy)
 
    # V = PR + gPV  => (I-gP)V = PR  => V = inv(I-gP)*PR
    Vpolicy = linalg.solve((np.identity(S) - discount*Ppolicy), PRpolicy)
    return Vpolicy

def mdp_bellman_operator_with_Q(P, PR, discount, Vprev):
    S = P.shape[0]
    A = PR.shape[1]

    Q = np.zeros([PR.shape[0], PR.shape[1]])

    for a in range(A):
        Q[:, a] = PR[:, a] + discount * np.dot(P[:, :, a], Vprev)
    V = Q.max(axis=1)
    policy = Q.argmax(axis=1)
    return [V, Q, policy]


def mdp_policy_iteration_with_Q(P, R, discount, policy0):

    S = P.shape[0]
    A = P.shape[2]

        
    PR = mdp_computePR(P,R)

    # initialization of optional arguments
    eval_type = 0
    max_iter = 1000
    iter = 0
    policy = policy0
    is_done = False
    while not is_done:
        iter += 1
        V = mdp_eval_policy_matrix(P,PR,discount,policy)

        [_, Q, policy_next] = mdp_bellman_operator_with_Q(P, PR, discount, V)
        
        n_different = sum(policy_next != policy)
        if all(policy_next == policy) or iter == max_iter or (iter > 20 and n_different <=5) or iter > 50:
            is_done = True
        else:
            policy = policy_next
        
    return [V, policy, iter, Q]
