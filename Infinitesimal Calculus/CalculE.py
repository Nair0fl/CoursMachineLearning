import numpy
#Valeur du log de e
logE=1
#Précision au n decimal
precision=10
#nombre de tour
i=1
#Valeur de a
a=5
while(True):
     #Quand le ln(a) est inférieur à ln(e)
    if(round(numpy.log(a),precision)<logE):
        a=a+(1/i)
    #Quand le ln(a) est superieur à ln(e)
    if(round(numpy.log(a),precision)>logE):
        a=a-(1/i)
    #Seulement quand le résulat arrondi à la précision vaut ln(e)
    if(round(numpy.log(a),precision)==logE):
        break;
    i+=1
print("Valeur de e : "+a)