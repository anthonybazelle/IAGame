import numpy as np
from random import randint


"""-------------------- GLOBAL VALUE ----------------------"""
sizeBoard = 8
rdmValue = np.zeros((sizeBoard, sizeBoard))
moveHorizontal = []
moveVertical = []
hashTable = {}
bug = 0

class State:
	""" Correspond a un noeud de l'arbre """
	""" param : coup joué pour cet état (donc cet hashcode) / total res / Nombre de passage / a qui le tour de jouer """
	def __init__(self, move):
		self.total = 0
		self.passage = 1
		self.move = move
		self.mean = 0
		self.turn = 0


def setRandomValue():
	for x in range(0, sizeBoard):
		for y in range(0, sizeBoard):
			rdmValue[x, y] = randint(0 , 9000000000000000000)

def getHash(state):
	hash = 0
	for x in range(0, sizeBoard):
		for y in range(0, sizeBoard):
			if state[x, y] != 0:
				piece = state[x, y]
				hash ^= int(rdmValue[x, y])
	return hash

def clearBoard(state):
	state = np.zeros((sizeBoard, sizeBoard))

def playout(boardParam):
	print("Playout")
	hWin = 0

	""" 0 : Vertical   1 : Hortizontal """
	playerTurn = 0
	res = 0
	legalMoves(boardParam)
	while len(moveHorizontal) > 0 and len(moveVertical) > 0:
		pos = 0
		legalMoves(boardParam)

		if len(moveHorizontal) == 0 or len(moveVertical) == 0:
			break
		move = []
		player = 0
		if playerTurn == 0:
			"""print("Size moveVertical:")
			print(len(moveVertical))"""
			pos = randint(0, len(moveVertical))
			"""print("Pos :")
			print(pos)"""
			move = moveVertical[pos - 1]
			player = 0
		elif playerTurn == 1:
			"""print("Size moveHorizontal:")
			print(len(moveHorizontal))"""
			pos = randint(0, len(moveHorizontal))
			"""print("Pos :")
			print(pos)"""
			move = moveHorizontal[pos - 1]
			player = 1

		play(boardParam, move, player)
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


	print(boardParam)
	return hWin

def legalMoves(state):
	"""print("legalMove")"""

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
	print("Size moveVert :")
	print(len(moveVertical))
	print("Size moveHoriz :")
	print(len(moveHorizontal))"""

def play(boardParam, move, player):
	"""print("play")"""

	for (x, y) in move:
		if player == 0:
			boardParam[x, y] = 1
		elif player == 1:
			boardParam[x, y] = 2

	hash = getHash(boardParam)
	newStateToHash = State(move)
	hashTable[hash] = newStateToHash

	return boardParam

def seen(state):
	print("seen")
	global bug
	hash = 0
	for x in range(0, sizeBoard - 1):
		for y in range(0, sizeBoard - 1):
			if state[x, y] != 0:
				piece = state[x, y]
				hash ^= int(rdmValue[x, y])
	if hash not in hashTable.keys():
		bug += 1
		return
	return hashTable[hash]

def selectBestMoveIndex(hashCurrentNode, currentNode):
	moves = []
	if currentNode.turn == 0:
		moves = moveVertical
	elif currentNode.turn == 1:
		moves = moveHorizontal

	maxValue = 0
	indice = 0

	"""argmax(t.mean[i] + c * sqrt(log(t.sumPlayouts) / t.playouts[i]))"""
	for i in range(0, len(moves) - 1):
		viergeBoard = np.zeros((sizeBoard, sizeBoard))
		stateMove = play(viergeBoard, moves[i])
		hashMove = getHash(stateMove)
		hashChild = hashMove ^ hashCurrentNode
		child = seen(hashChild)

		if child == None:
			print("NO CHILD IN SELECT BEST MOVE INDEX !")
			child = State(stateMove)
			hashTable[hashChild] = child

		mean = child.total / child.passage
		c = sqrt(2)
		res = mean + c * sqrt(log(currentNode.passage) / child.passage)
		if res > maxValue:
			maxValue = res
			indice = i

	return indice


def uct(boardParam, move):
	"""if terminal(state):
		return score(state)"""
	legalMoves(boardParam)
	t = seen(boardParam)
	hashState = getHash(boardParam)

	if t != None:
		if t.turn == 0:
			moves = moveVertical
		elif t.turn == 1:
			moves = moveHorizontal
		best = selectBestMoveIndex(hashState, t)
		boardParam = play(boardParam, moves[best])
		res = UCT(boardParam, best)
		if res == 0:
			res = 1
			t.total += res
		else:
			res = 0
	else:
		newHash = getHash(boardParam)
		res = playout(boardParam)
		newObj = State(move)
		hashTable[newHash] = newObj
	return res

""" ----------------------- MAIN -----------------------"""

setRandomValue()

for i in range(0, 1000):
	board = np.zeros((sizeBoard, sizeBoard))
	winner = playout(board)

print(len(hashTable))

for i in range(0, 100):
	board = np.zeros((sizeBoard, sizeBoard))
	if seen(board) not in hashTable.keys():
		newState = State(None)
		hashTable[seen(board)] = newState
	uct(board, selectBestMoveIndex(seen(board), hashTable[seen(board)]))

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
