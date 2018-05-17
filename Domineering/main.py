"""
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" #see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import numpy
model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
model.compile(optimizer=keras.optimizers.sgd(), loss=keras.losses.mse)

X = [
[1],
[2]
]

Y = [
[1],
[3]
]

tb_callback = keras.callbacks.TensorBoard()

model.fit(numpy.array(X), numpy.array(Y), callbacks=[tb_callback], epochs=2000)

print(model.predict(numpy.array(X)))"""

import numpy as np
from random import randint

sizeBoard = 9
board = np.zeros((sizeBoard, sizeBoard))
rdmValue = np.zeros((sizeBoard, sizeBoard))
moveHorizontal = []
moveVertical = []
hashTable = {}


"""xor(bool(a), bool(b))"""



class State:
    """ Correspond a un noeud de l'arbre """
    """ param : coup joué pour cet état (donc cet hashcode) / total res / Nombre de passage / a qui le tour de jouer """
    def __init__(self, move):
        self.total = 0
        self.passage = 1
        self.move = move
        self.mean = 0
        self.turn = 0
    

def setRandomValue(board):
    for x in range(0, sizeBoard - 1):
        for y in range(0, sizeBoard - 1):
            rdmValue[x, y] = randint(0 , 9000000000000000000)

def setHash(state):
    hash = 0
    for x in range(0, sizeBoard):
        for y in range(0, sizeBoard):
            if state[x, y] != 0:
                piece = state[x, y]
                hash ^= int(rdmValue[x, y])
    return hash

"""
def addToHashTable(state):
    hash = findHash(state)
        TODO : hashTable[Hash] = (mean, sumPlayouts, tabPlayouts[])
    hashTable[hash] = (mean, sumPlayouts, [])
"""

def clearBoard(state):
    state = np.zeros((sizeBoard, sizeBoard))

def playout(state):
    print("Playout")
    hWin = 0

    """ 0 : Vertical   1 : Hortizontal """
    playerTurn = 0
    res = 0
    legalMoves(state)
    while len(moveHorizontal) > 0 and len(moveVertical) > 0:
        pos = 0
        legalMoves(state)

        move = []
        if playerTurn == 0:
            pos = randint(0, len(moveVertical) - 1)
            move = moveVertical[pos]
        elif playerTurn == 1:
            pos = randint(0, len(moveHorizontal) - 1)
            move = moveHorizontal[pos]

        state = play(state, move)
        """"""
        if playerTurn == 0:
            playerTurn = 1
        elif playerTurn == 1:
            playerTurn = 0

    """"""
    if len(moveVertical) == 0:
        hWin = True
    elif len(moveHorizontal) == 0:
        hWin = False

    print(state)
    return hWin

def legalMoves(state):
    print("legalMove")

    moveHorizontal.clear()
    moveVertical.clear()

    for x in range(0, sizeBoard):
        adjacent = False
        adjacentCase = ()
        for y in range(0, sizeBoard):
            if state[x, y] == 0 and adjacent == False:
                adjacent = True
                adjacentCase = (x, y)
            elif state[x, y] == 1 or state[x, y] == 2:
                adjacent = False
            elif adjacent == True and state[x, y] == 0:
                """print("Horizontal player can play")"""
                moveHorizontal.append([adjacentCase, (x, y)])
                adjacentCase = (x, y)

    for y in range(0, sizeBoard):
        adjacent = False
        adjacentCase = ()
        for x in range(0, sizeBoard):
            if state[x, y] == 0 and adjacent == False:
                adjacent = True
                adjacentCase = (x, y)
            elif state[x, y] == 1 or state[x, y] == 2:
                adjacent = False
            elif adjacent == True and state[x, y] == 0:
                """print("Vertical player can play")"""
                moveVertical.append([adjacentCase, (x, y)])
                adjacentCase = (x, y)

    """
    print("MoveHorizontal size :")
    print(len(moveHorizontal))
    print("MoveVertical size :")
    print(len(moveVertical))
    """

def play(state, move):
    """print("play")"""
    itsOkay = False
    itsVert = False
    cpt = 0
    oldX = 0
    oldY = 0

    """print(state)"""
    for (x, y) in move:
        if cpt == 1:
            if x == oldX:
                itsVert = False
            elif y == oldY:
                itsVert = True
        cpt += 1

    for (x, y) in move:
        if itsVert == True:
            state[x, y] = 1
        elif itsVert == False:
            state[x, y] = 2

    hash = setHash(state)
    newStateToHash = State(move)
    hashTable[hash] = newStateToHash
    return state



def seen(state):
    print("seen")
    hash = 0
    for x in range(0, sizeBoard):
        for y in range(0, sizeBoard):
            if state[x, y] != 0:
                piece = state[x, y]
                hash ^= int(rdmValue[x, y])
    return hashTable[hash]

"""
def uct(state):
	if terminal(state):
		return score(state)
	moves = legalMoves(state)
	t = seen(state)
	if t!= None:
	    t.sumPlayout -> nbPassageNoeudCourrant
	    t.playouts[i] -> nbPassageNoeurFilsAIndexI
	    
	    for 
		    best = argmax(t.mean[i] + c * sqrt(log(t.sumPlayouts) / t.playouts[i]))
		state = play(state, moves[best])
		res = UCT(state)
		update t with res
	else:
		t = new entry in transposition table
		res = playout(state)
		update t with res
	return res
"""

""" ----------------------- MAIN -----------------------"""

setRandomValue(board)
winner = playout(board)
print(hashTable)

"""
cpt = 0
for i in range(0, 1000):
    clearBoard(board)
    winner = playout(board)
    if winner == True:
        print("Horizontal player win")
        cpt += 1
    else:
        print("Vertical player win")
    i = i + 1

print(cpt)
meanH = cpt / 1000
print("Mean win by horizontal player on 1000 playout :")
print(meanH)
"""
"""
print("MoveHorizontal")
print(moveHorizontal)
print("MoveVertical")
print(moveVertical)
"""
