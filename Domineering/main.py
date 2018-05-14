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

moveHorizontal = []
moveVertical = []

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

    print(state)
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

    return state

def seen(state):
    print("seen")


""" ----------------------- MAIN -----------------------"""

winner = playout(board)

if winner == True:
    print("Horizontal player win")
else:
    print("Vertical player win")


"""
print("MoveHorizontal")
print(moveHorizontal)
print("MoveVertical")
print(moveVertical)
"""