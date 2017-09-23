
import math
import random



class LinearReg:

	def __init__(self, size):
		self.size = size
		self.w = [0 for i in xrange(size)]
		self.stepSize = 0.1

	def getLoss(self, y, x):
		return y - jdotj(self.w, x)

	def getGradient(self, y, x):
		tmp = [0 for i in xrange(len(x[0]))]
		for j in xrange(len(tmp)):
			tmpj = 0
			for i, y_i in enumerate(y):
				x_i = x[i]
				x_j_i = x[i][j]
				tmpj += self.getLoss(y_i, x_i) * x_j_i
			tmp[j] += self.stepSize * tmpj / len(y)
			#print "%d\t%f" % (j, tmp[j])
		return tmp
			
	def updateWeight(self, deltaW):
		self.w = jaddj(deltaW, self.w)

def getAbs(l):
	res = 0
	for i in l:
		res += i*i
	t = math.sqrt(res)
	#print l, t
	return t

class GD:

	def __init__(self, loss, x, y, round):
		self.stop = 0.0001
		self.round = round
		self.realRound = 1
		self.loss = loss
		self.x = x
		self.y = y

	def runARound(self):
		deltaG = self.loss.getGradient(self.y, self.x)
		self.loss.updateWeight(deltaG)
		return deltaG

	def learn(self):
		deltaG = self.runARound()
		while self.realRound < self.round and self.stop < getAbs(deltaG):
			deltaG = self.runARound()
			self.realRound += 1
		if self.realRound < self.round:
			print "Reach the gap: %f" % getAbs(deltaG)
		else:
			print "Run off round."


def genRandomData(featureSize, sampleSize):
	y = []
	x = []
	w = []
	for i in xrange(featureSize):
		w.append(random.random())
	for i in xrange(sampleSize):
		tmp = []
		for j in xrange(featureSize):
			tmp.append(random.random())
		x.append(tmp)
		y.append(jdotj(tmp, w))
	return y,x,w

def test():
	featureSize = 2
	sampleSize = 500
	y, x, w = genRandomData(featureSize, sampleSize)
	lg = LinearReg(featureSize)
	gd = GD(lg, x, y, sampleSize)
	gd.learn()
	#print gd.loss.w, gd.realRound
	allLoss = 0
	for i, x_i in enumerate(x):
		loss = lg.getLoss(y[i], x_i)
		allLoss += loss
	print "true model is %s" % (str(w))
	print "my model is   %s" % (lg.w)
	print "avgLoss: %f" % (allLoss/sampleSize)

if __name__ == '__main__':
	test()
