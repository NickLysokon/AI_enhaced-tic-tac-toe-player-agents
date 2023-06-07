import random
import time
import math

import numpy as np

import Gobblet_Gobblers_Env as gge

not_on_board = np.array([-1, -1])


# agent_id is which player I am, 0 - for the first player , 1 - if second player
def dumb_heuristic1(state, agent_id):
    is_final = gge.is_final_state(state)
    # this means it is not a final state
    if is_final is None:
        return 0
    # this means it's a tie
    if is_final is 0:
        return -1
    # now convert to our numbers the win
    winner = int(is_final) - 1
    # now winner is 0 if first player won and 1 if second player won
    # and remember that agent_id is 0 if we are first player  and 1 if we are second player won
    if winner == agent_id:
        # if we won
        return 1
    else:
        # if other player won
        return -1


# checks if a pawn is under another pawn
def is_hidden(state, agent_id, pawn):
    pawn_location = gge.find_curr_location(state, pawn, agent_id)
    for key, value in state.player1_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    for key, value in state.player2_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    return False


# count the numbers of pawns that i have that aren't hidden
def dumb_heuristic2(state, agent_id):
    sum_pawns = 0
    if agent_id == 0:
        for key, value in state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1
    if agent_id == 1:
        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1

    return sum_pawns


def smart_heuristic(state, agent_id):

    triples = 0
    doubles = 0
    ones = 0

    if agent_id == 0:

        for key, value in state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                ones += 1

        for key1, value1 in state.player1_pawns.items():
            for key2, value2 in state.player1_pawns.items():

             if (not np.array_equal(value1[0], not_on_board) and not is_hidden(state, agent_id, key1)
                    and not np.array_equal(value2[0], not_on_board) and not is_hidden(state, agent_id, key2)
                    and (value2[0] == value1[0] + 1).all()):
                doubles += 1

             if (not np.array_equal(value1[0], not_on_board) and not is_hidden(state, agent_id, key1)
                    and not np.array_equal(value2[0], not_on_board) and not is_hidden(state, agent_id, key2)
                    and (value2[0] == value1[0] + 4).all()):
                doubles += 1


             if (not np.array_equal(value1[0], not_on_board) and not is_hidden(state, agent_id, key1)
                    and not np.array_equal(value2[0], not_on_board) and not is_hidden(state, agent_id, key2)
                    and (value2[0] == value1[0] + 2).all()):
                doubles += 1



        for key1, value1 in state.player1_pawns.items():
            for key2, value2 in state.player1_pawns.items():
                for key3,value3 in state.player1_pawns.items():

                  if(not np.array_equal(value1[0], not_on_board) and not is_hidden(state, agent_id, key1)
                    and not np.array_equal(value2[0] , not_on_board) and not is_hidden(state, agent_id, key2)
                    and not np.array_equal(value3[0], not_on_board) and not is_hidden(state, agent_id, key3)
                    and (value2[0] == value1[0] - 1).all() and (value3[0] == value1[0] + 1).all()):
                     triples += 1



                  if (not np.array_equal(value1[0], not_on_board) and not is_hidden(state, agent_id, key1)
                    and not np.array_equal(value2[0], not_on_board) and not is_hidden(state, agent_id, key2)
                    and not np.array_equal(value3[0], not_on_board) and not is_hidden(state, agent_id, key3)
                    and (value2[0] == value1[0] + 4).all() and (value3[0] == value1[0] + 8).all()):
                     triples += 1

                  if (not np.array_equal(value1[0], not_on_board) and not is_hidden(state, agent_id, key1)
                    and not np.array_equal(value2[0], not_on_board) and not is_hidden(state, agent_id, key2)
                    and not np.array_equal(value3[0], not_on_board) and not is_hidden(state, agent_id, key3)
                    and (value2[0] == value1[0] + 2).all() and (value3[0] == value1[0] + 4).all()):
                      triples += 1



    if agent_id == 1:

        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                ones += 1


        for key1, value1 in state.player2_pawns.items():
            for key2, value2 in state.player2_pawns.items():



             if (not np.array_equal(value1[0], not_on_board) and not is_hidden(state, agent_id, key1)
                    and not np.array_equal(value2[0], not_on_board) and not is_hidden(state, agent_id, key2)
                    and (value2[0] == value1[0] + 1).all()):
                doubles += 1

             if (not np.array_equal(value1[0], not_on_board) and not is_hidden(state, agent_id, key1)
                    and not np.array_equal(value2[0], not_on_board) and not is_hidden(state, agent_id, key2)
                    and (value2[0] == value1[0] + 4).all()):
                doubles += 1


             if (not np.array_equal(value1[0], not_on_board) and not is_hidden(state, agent_id, key1)
                    and not np.array_equal(value2[0], not_on_board) and not is_hidden(state, agent_id, key2)
                    and (value2[0] == value1[0] + 2).all()):
                doubles += 1



        for key1, value1 in state.player2_pawns.items():
            for key2, value2 in state.player2_pawns.items():
                for key3,value3 in state.player2_pawns.items():

                  if(not np.array_equal(value1[0], not_on_board) and not is_hidden(state, agent_id, key1)
                    and not np.array_equal(value2[0] , not_on_board) and not is_hidden(state, agent_id, key2)
                    and not np.array_equal(value3[0], not_on_board) and not is_hidden(state, agent_id, key3)
                    and (value2[0] == value1[0] - 1).all() and (value3[0] == value1[0] + 1).all()):
                     triples += 1



                  if (not np.array_equal(value1[0], not_on_board) and not is_hidden(state, agent_id, key1)
                    and not np.array_equal(value2[0], not_on_board) and not is_hidden(state, agent_id, key2)
                    and not np.array_equal(value3[0], not_on_board) and not is_hidden(state, agent_id, key3)
                    and (value2[0] == value1[0] + 4).all() and (value3[0] == value1[0] + 8).all()):
                     triples += 1

                  if (not np.array_equal(value1[0], not_on_board) and not is_hidden(state, agent_id, key1)
                    and not np.array_equal(value2[0], not_on_board) and not is_hidden(state, agent_id, key2)
                    and not np.array_equal(value3[0], not_on_board) and not is_hidden(state, agent_id, key3)
                    and (value2[0] == value1[0] + 2).all() and (value3[0] == value1[0] + 4).all()):
                      triples += 1



    sum =  ones + doubles*10 + triples*100

    return sum

# IMPLEMENTED FOR YOU - NO NEED TO CHANGE
def human_agent(curr_state, agent_id, time_limit):
    print("insert action")
    pawn = str(input("insert pawn: "))
    if pawn.__len__() != 2:
        print("invalid input")
        return None
    location = str(input("insert location: "))
    if location.__len__() != 1:
        print("invalid input")
        return None
    return pawn, location


# agent_id is which agent you are - first player or second player
def random_agent(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    rnd = random.randint(0, neighbor_list.__len__() - 1)
    return neighbor_list[rnd][0]


# TODO - instead of action to return check how to raise not_implemented
def greedy(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = 0
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = dumb_heuristic2(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


# TODO - add your code here
def greedy_improved(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = 0
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = smart_heuristic(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


def min_max(curr_state, agent_id, d, turn):
    neighbor_list = curr_state.get_neighbors()
    if gge.is_final_state(curr_state) or d == 0:
        return smart_heuristic(curr_state, agent_id)

    if turn == agent_id:
        cur_max = -float('inf')
        for neighbor in neighbor_list:
            v = min_max(neighbor[1], agent_id, d - 1, 1-turn)
            cur_max = max(v, cur_max)
        return  cur_max

    if turn != agent_id:
        cur_min = float('inf')
        for neighbor in neighbor_list:
            v = min_max(neighbor[1], agent_id, d - 1, 1-turn)
            cur_min = min(v, cur_min)

        return cur_min

def rb_heuristic_min_max(curr_state, agent_id, time_limit):
    d = 0
    max_neighbor = None
    turn = agent_id
    max_heuristic =0

    start_time = time.time()  # remember when we started
    while (time.time() - start_time) < time_limit and d <= 2 :

        neighbor_list = curr_state.get_neighbors()

        for neighbor in neighbor_list:
             if (time.time() - start_time ) > time_limit or d >= 2:
                return max_neighbor[0]

             curr_heuristic = min_max(neighbor[1], agent_id, d, turn)
             if curr_heuristic >= max_heuristic:
                max_heuristic = curr_heuristic
                max_neighbor = neighbor
        d += 1


def alpha_beta_aux(curr_state,agent_id,d,turn,alpha,beta):
    if gge.is_final_state(curr_state) or d==0:
        value= smart_heuristic(curr_state,agent_id)
        return value
    neighbor_list = curr_state.get_neighbors()

    if turn==agent_id:
        curr_max = float('-inf')
        for neighbor in neighbor_list:
            v = alpha_beta_aux(neighbor[1],agent_id,d-1,turn,alpha,beta)
            curr_max = max(curr_max, v)
            alpha = max(alpha, curr_max)
            if curr_max >= beta: return float('inf')
        return curr_max
    else:
        curr_min = float('inf')
        for neighbor in neighbor_list:
            v = alpha_beta_aux(neighbor[1],agent_id,d-1,turn,alpha,beta)
            curr_min = min(curr_min, v)
            beta = min(beta, curr_min)
            if curr_min <= alpha: return float('-inf')
        return curr_min


def alpha_beta(curr_state, agent_id, time_limit):
    alpha = float('-inf')
    beta = float('inf')
    d = 0
    turn = agent_id
    max_neighbor = None
    max_heuristic = 0
    start_time = time.time()
    while (time.time()-start_time) < time_limit and d<=2:
        neighbor_list = curr_state.get_neighbors()
        for neighbor in neighbor_list:
             if (time.time() - start_time) > time_limit or d>=2:
                return max_neighbor[0]
             curr_heuristic = alpha_beta_aux(curr_state, agent_id, d, turn,alpha,beta)
             if curr_heuristic >= max_heuristic:
                max_heuristic = curr_heuristic
                max_neighbor = neighbor
        d += 1

def get_move_importance(curr_state,neighbor):
    if is_important_move(curr_state,neighbor): return 2
    return 1

def size_cmp(size1,size2):
    if "S" in size1 and "S" in size2: return False
    if "B" in size1 and "B" in size2: return False
    if "M" in size1 and "M" in size2: return False
    if "S" in size1 and ("M" in size2 or "B" in size2 ): return True
    if "M" in size1 and "B" in size2: return True
    return False


def is_important_move(curr_state,neighbor):
    if "S1" in neighbor[0][0] or "S2" in neighbor[0][0]: return True
    location = neighbor[0][1]
    expected_pawn = neighbor[0][0]
    #check if another pawn located in this location
    pawns = ["B1", "B2", "M1", "M2", "S1", "S2"]
    for pawn in pawns:
        curr_location = gge.find_curr_location(curr_state,pawn,0)
        if curr_location==location and size_cmp(pawn,expected_pawn)==True:
            return True
    return False


def get_move_possibility(curr_state,neighbor_list):
    x= 0
    for neighbor in neighbor_list:
        if is_important_move(curr_state,neighbor): x = x+2
        else: x = x+1
    return 1/x


def expectimax_aux(curr_state, agent_id,d,turn):
    if gge.is_final_state(curr_state) or d==0:
        value = smart_heuristic(curr_state,agent_id)
        return value

    neighbor_list = curr_state.get_neighbors()
    if turn==agent_id:
        curr_max = float('-inf')
        for neighbor in neighbor_list:
            v = expectimax_aux(neighbor[1],agent_id,d-1,turn)
            curr_max = max(v,curr_max)
        return curr_max

    else:
        v = 0
        move_possibility = get_move_possibility(curr_state,neighbor_list)
        for neighbor in neighbor_list:
            p = move_possibility * get_move_importance(curr_state,neighbor)
            temp = expectimax_aux(neighbor[1],agent_id,d-1,turn)
            v += p *temp
        return v


def expectimax(curr_state, agent_id, time_limit):
    d = 0
    max_neighbor = None
    turn = agent_id
    max_heuristic = 0

    start_time = time.time()  # remember when we started
    while (time.time() - start_time) < time_limit and d <= 2:

        neighbor_list = curr_state.get_neighbors()

        for neighbor in neighbor_list:
            if (time.time() - start_time) > time_limit or d >= 2:
                return max_neighbor[0]

            curr_heuristic = expectimax_aux(neighbor[1], agent_id, d, turn)
            if curr_heuristic >= max_heuristic:
                max_heuristic = curr_heuristic
                max_neighbor = neighbor
        d += 1

# these is the BONUS - not mandatory
def super_agent(curr_state, agent_id, time_limit):
    return alpha_beta(curr_state,agent_id,time_limit)

