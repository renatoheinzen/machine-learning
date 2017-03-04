
import pandas as pd
import numpy  as np
from sklearn.naive_bayes      import MultinomialNB
from sklearn.ensemble         import AdaBoostClassifier
from sklearn.multiclass       import OneVsRestClassifier
from sklearn.multiclass       import OneVsOneClassifier
from sklearn.svm              import LinearSVC
from collections              import Counter
from sklearn.cross_validation import cross_val_score

def fit_and_predict(nome, 
	                modelo, 
	                treino_dados, 
	                treino_marcacoes):
    
    k = 10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv = k)
    taxa_acerto = np.mean(scores)

    msg = "Taxa de acerto: {0}: {1}".format(nome,taxa_acerto)
    print msg
    return taxa_acerto


data_frame = pd.read_csv('situacao_cliente.csv')

data_frame_x = data_frame[['recencia',
                           'frequencia',
                           'semanas_de_inscricao']]
data_frame_y = data_frame['situacao']

dummies_x = pd.get_dummies(data_frame_x)
dummies_y = data_frame_y

x = dummies_x.values
y = dummies_y.values

porcentagem_treino = 0.8

tamanho_treino = int(porcentagem_treino * len(y))
#tamanho_validacao = len(y) - tamanho_treino

treino_dados = x[0:tamanho_treino]
treino_marcacoes = y[0:tamanho_treino]

#fim_treino = (tamanho_treino + tamanho_teste)

validacao_dados = x[tamanho_treino:]
validacao_marcacoes = y[tamanho_treino:]

resultados = {}

modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB",
	                                    modeloMultinomial,
	                                    treino_dados,
	                                    treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

modeloAdaBoost = AdaBoostClassifier()
resultadoAdaboost = fit_and_predict("AdaBoost", 
	                                modeloAdaBoost,
	                                treino_dados,
	                                treino_marcacoes)
resultados[resultadoAdaboost] = modeloAdaBoost

modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0))
resultadoOneVsRest = fit_and_predict("OneVsRest", 
	                                  modeloOneVsRest,
	                                  treino_dados,
						              treino_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0))
resultadoOneVsOne = fit_and_predict("OneVsOne", 
	                                 modeloOneVsOne,
	                                 treino_dados,
	                                 treino_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne

print resultados

maximo   = max(resultados)
vencedor = resultados[maximo]
vencedor.fit(treino_dados,treino_marcacoes)

resultado = vencedor.predict(validacao_dados)
acertos   = (resultado == validacao_marcacoes)
total_acertos   = sum(acertos)
total_elementos = len(validacao_marcacoes)
taxa_acerto     = 100.0 * total_acertos / total_elementos

msg = "Taxa de acerto do vencedor entre os algoritmos: {0}".format(taxa_acerto)
print msg

# a eficacia de um algoritmo burro
acerto_base = max(Counter(validacao_marcacoes).itervalues())
taxa_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print ("Taxa de acerto base: %f" % taxa_acerto_base)
print ("Total de testes: %d" % len(validacao_dados))

