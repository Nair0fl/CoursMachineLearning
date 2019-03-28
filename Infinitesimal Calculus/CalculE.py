import numpy
logE=1
precision=10
i=1
lastValue=9
while(True):
    if(round(numpy.log(lastValue),precision)<logE):
        lastValue=lastValue+(1/i)
    if(round(numpy.log(lastValue),precision)>logE):
        lastValue=lastValue-(1/i)
    if(round(numpy.log(lastValue),precision)==logE):
        break;
    i+=1
print(lastValue)