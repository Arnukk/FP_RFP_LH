import numpy
import math
from scipy import stats


def RandomizedFictitiousPlay(A, Epsilon):
    n = len(A[0])
    m = len(A)

    X = numpy.matrix(numpy.zeros((m, 1), dtype=int))
    Y = numpy.matrix(numpy.zeros((n, 1), dtype=int))
    X[0] = 1
    Y[0] = 1

    numpy.random.shuffle(X)
    numpy.random.shuffle(Y)

    t = int(round(6*math.log(2*n*m)/pow(Epsilon, 2)))

    for i in range(t):

        Ax = numpy.array(numpy.transpose(A) * X).tolist()
        #print Ax
        Ay = numpy.array(A * Y).tolist()
        #print Ay
        values = Ay
        probabilities = []
        for item in Ay:
            probabilities.append(pow(math.e, Epsilon*item[0]/2))
        while True:
            try:
                theprobabilities = []
                temp = sum(probabilities)
                theprobabilities[:] = [x / temp for x in probabilities]
                distrib = stats.rv_discrete(values=(values, theprobabilities))
                xchoice = Ay.index(distrib.rvs(size=1)[0])
                break
            except:
                pass

        values = Ax
        probabilities = []
        for item in Ax:
            probabilities.append(pow(math.e, -Epsilon*item[0]/2))
        while True:
            try:
                theprobabilities = []
                temp = sum(probabilities)
                theprobabilities[:] = [x / temp for x in probabilities]
                distrib = stats.rv_discrete(values=(values, theprobabilities))
                ychoice = Ax.index(distrib.rvs(size=1)[0])
                break
            except:
                pass

        #print xchoice
        X[xchoice] += 1
        #print X
        #print ychoice
        Y[ychoice] += 1
        #print Y
    return X/float(t+1), Y/float(t+1)


def FictitiousPlay(A, t):

    n = len(A[0])
    m = len(A)

    X = numpy.matrix(numpy.zeros((m, 1), dtype=int))
    Y = numpy.matrix(numpy.zeros((n, 1), dtype=int))
    X[0] = 1
    Y[0] = 1

    numpy.random.shuffle(X)
    numpy.random.shuffle(Y)

    for i in range(t):
        Ax = numpy.array(numpy.transpose(A) * X).tolist()

        Ay = numpy.array(A * Y).tolist()

        xchoice = Ax.index(min(Ax))
        ychoice = Ay.index(max(Ay))
        #print xchoice
        X[ychoice] += 1
        #print X
        #print ychoice
        Y[xchoice] += 1
        #print Y

    return X/float(t+1), Y/float(t+1)


#The payoff Matrix
A = numpy.identity(5, dtype=int)
#A = numpy.array([[1, 0, 2, -2], [-1, 1, -1, 0]])

print FictitiousPlay(A, 10000)
print RandomizedFictitiousPlay(A, 0.1)


#r = 1

#while r >= 0.3:
#    temp1, temp = RandomizedFictitiousPlay(A, 0.1)
#    Ax = numpy.array(numpy.transpose(A) * temp1).tolist()
#    Ay = numpy.array(A * temp).tolist()
#    r = abs(max(Ay)[0] - min(Ax)[0])
#    print r

#print temp1, temp

#while temp
#print RandomizedFictitiousPlay(A, 0.1)
#print FictitiousPlay(A, 1700)