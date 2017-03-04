
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from collections import Counter

def fit_and_predict(nome,modelo, treino_dados, teste_dados,
	                treino_marcacoes, teste_marcacoes):
	modelo.fit(treino_dados,treino_marcacoes)

	resultado = modelo.predict(teste_dados)

	acertos = (resultado == teste_marcacoes)

	total_acertos = sum(acertos)
	total_elementos = len(teste_dados)
	taxa_acerto = 100.0 * total_acertos / total_elementos

	msg = "Taxa de acerto: {0}: {1}".format(nome,taxa_acerto)
	print msg
	return taxa_acerto


data_frame = pd.read_csv('busca.csv')

data_frame_x = data_frame[['home','busca','logado']]
data_frame_y = data_frame['comprou']

dummies_x = pd.get_dummies(data_frame_x)
dummies_y = data_frame_y

x = dummies_x.values
y = dummies_y.values

porcentagem_treino = 0.8
porcentagem_teste  = 0.1

tamanho_treino = int(porcentagem_treino * len(y))
tamanho_teste = int(porcentagem_teste * len(y))
tamanho_validacao = len(y) - tamanho_treino - tamanho_teste

treino_dados = x[0:tamanho_treino]
treino_marcacoes = y[0:tamanho_treino]

fim_treino = (tamanho_treino + tamanho_teste)

teste_dados = x[tamanho_treino:fim_treino]
teste_marcacoes = y[tamanho_treino:fim_treino]

validacao_dados = x[-tamanho_validacao:]
validacao_marcacoes = y[-tamanho_validacao:]

modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB",modeloMultinomial,treino_dados, teste_dados,
	            treino_marcacoes, teste_marcacoes)

modeloAdaBoost = AdaBoostClassifier()
resultadoAdaboost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost,treino_dados, teste_dados,
	            treino_marcacoes, teste_marcacoes)

if resultadoAdaboost > resultadoMultinomial:
	vencedor = modeloAdaBoost
else:
	vencedor = modeloMultinomial

resultado = vencedor.predict(validacao_dados)
acertos = (resultado == validacao_marcacoes)
total_acertos = sum(acertos)
total_elementos = len(validacao_marcacoes)
taxa_acerto = 100.0 * total_acertos / total_elementos

msg = "Taxa de acerto do vencedor entre os dois algoritmos: {0}".format(taxa_acerto)
print msg

# a eficacia de um algoritmo burro
acerto_base = max(Counter(validacao_marcacoes).itervalues())
taxa_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print ("Taxa de acerto base: %f" % taxa_acerto_base)

print ("Total de testes: %d" % len(validacao_dados))