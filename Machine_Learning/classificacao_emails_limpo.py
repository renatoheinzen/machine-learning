#!-*- coding: utf8 -*-
import pandas as pd
import numpy  as np
from sklearn.naive_bayes      import MultinomialNB
from sklearn.ensemble         import AdaBoostClassifier
from sklearn.multiclass       import OneVsRestClassifier
from sklearn.multiclass       import OneVsOneClassifier
from sklearn.svm              import LinearSVC
from collections              import Counter
from sklearn.cross_validation import cross_val_score
import nltk

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


def vetorizar_texto(texto, tradutor):

	vetor = [0] * len(tradutor)

	for palavra in texto:
		if len(palavra) > 0 :
			raiz = stemmer.stem(palavra) 
			if raiz in tradutor:
				posicao = tradutor[raiz]
				vetor[posicao] +=1
	return vetor

#nltk.download("punkt")

classificacoes = pd.read_csv('email.csv', encoding = 'utf-8')
textos_puros = classificacoes['email']
frases = textos_puros.str.lower()
textos_quebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]

#nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words("portuguese")
Dicionario = set()

#nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()

for lista in textos_quebrados:
	validas = [stemmer.stem(palavra) 
	            for palavra in lista 
	            if palavra not in stopwords and len(palavra) > 2]
	Dicionario.update(validas)

total_palavras = len(Dicionario)

tuplas =  (zip(Dicionario,xrange(total_palavras)))
tradutor = {palavra:indice for palavra, indice in tuplas}

vetores_texto = [vetorizar_texto(texto,tradutor) for texto in textos_quebrados]
marcas = classificacoes['classificacao']

x = np.array(vetores_texto)
y = np.array(marcas.tolist())

porcentagem_treino = 0.8

tamanho_treino = int(porcentagem_treino * len(y))
tamanho_validacao = len(y) - tamanho_treino

treino_dados = x[0:tamanho_treino]
treino_marcacoes = y[0:tamanho_treino]

validacao_dados = x[-tamanho_treino:]
validacao_marcacoes = y[-tamanho_treino:]

resultados = {}

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






































