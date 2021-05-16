# -*- coding: utf-8 -*-
"""
Created on Sat May 15 20:40:28 2021

@author: leona
"""

import pandas as pd

from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from tqdm import tqdm # verifica o progresso de uma tarefa
from statistics import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


df = pd.read_csv('conjunto_de_treinamento.csv')

df = df.drop(['estado_onde_nasceu',  #informacao desnecessaria
                    'id_solicitante',
                    'grau_instrucao',  # apenas possui valor 0
                    'estado_onde_trabalha',
                    'codigo_area_telefone_trabalho',
                    'codigo_area_telefone_residencial',
                    'possui_telefone_celular',  # apenas possui valor "N"
                    'local_onde_trabalha',  #local_onde_reside igual a local_onde_trabalha
                    'estado_onde_reside',
                    'grau_instrucao_companheiro',
                    'profissao_companheiro',

                   ],axis=1)


df_teste = pd.read_csv('conjunto_de_teste.csv')

df_teste = df_teste.drop(['estado_onde_nasceu',  #informacao desnecessaria
#                     'id_solicitante',
                    'grau_instrucao',  # apenas possui valor 0
                    'estado_onde_trabalha',
                    'codigo_area_telefone_trabalho',
                    'codigo_area_telefone_residencial',
                    'possui_telefone_celular',  # apenas possui valor "N"
                    'local_onde_trabalha',  #local_onde_reside igual a local_onde_trabalha
                    'estado_onde_reside',

#                     'possui_email',
                    'grau_instrucao_companheiro',
                    'profissao_companheiro',
#                     'local_onde_reside',
#                     'forma_envio_solicitacao',
#                     'sexo'

                   ],axis=1)



df = pd.get_dummies(df,columns=[
                                'forma_envio_solicitacao',
                                'sexo'
                               ])

binarizador = LabelBinarizer()
for v in ['possui_telefone_residencial',
        'vinculo_formal_com_empresa','possui_telefone_trabalho']:
    df[v] = binarizador.fit_transform(df[v])

df = df.dropna()

# df['profissao'] = df['profissao'].fillna(df['profissao'].mode()[0])
# =============================================================================
# df['profissao'] = df['profissao'].fillna(df['profissao'].mode()[0])
# df['ocupacao'] = df['ocupacao'].fillna(df['ocupacao'].mode()[0])
# df['meses_na_residencia'] = df['meses_na_residencia'].fillna(df['meses_na_residencia'].mode()[0])
# =============================================================================


df_teste = pd.get_dummies(df_teste,columns=[
                                'forma_envio_solicitacao',
                                'sexo'
                               ])

binarizador = LabelBinarizer()
for v in ['possui_telefone_residencial',
        'vinculo_formal_com_empresa','possui_telefone_trabalho']:
    df_teste[v] = binarizador.fit_transform(df_teste[v])


df_teste['profissao'] = df_teste['profissao'].fillna(df_teste['profissao'].mode()[0])
df_teste['ocupacao'] = df_teste['ocupacao'].fillna(df_teste['ocupacao'].mode()[0])
df_teste['meses_na_residencia'] = df_teste['meses_na_residencia'].fillna(df_teste['meses_na_residencia'].mode()[0])

atributos_selecionados = [
'sexo_ ',
'sexo_M',
'sexo_F',
'sexo_N',
'possui_email',
'local_onde_reside',
'tipo_endereco',
'idade',
'estado_civil',
'qtde_dependentes',
'dia_vencimento',
'possui_telefone_residencial',
'meses_na_residencia',
'renda_mensal_regular',
'renda_extra',
'possui_cartao_visa',
'possui_cartao_mastercard',
'possui_outros_cartoes',
'qtde_contas_bancarias',
'qtde_contas_bancarias_especiais',
'valor_patrimonio_pessoal',
'possui_carro',
'vinculo_formal_com_empresa',
'meses_no_trabalho',
'profissao',
'ocupacao',
'forma_envio_solicitacao_correio',
'forma_envio_solicitacao_internet',
'forma_envio_solicitacao_presencial',

'inadimplente'
]

dados_selecionados = df[atributos_selecionados]

dados_embaralhados = dados_selecionados.sample(frac=1,random_state=12345)

x = dados_embaralhados.loc[:,dados_embaralhados.columns!='inadimplente'].values
y = dados_embaralhados.loc[:,dados_embaralhados.columns=='inadimplente'].values

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x,
    y.ravel(),
    test_size=0.01,
    shuffle=True,
    random_state=777
    )

ajustador_de_escala = MinMaxScaler()
ajustador_de_escala.fit(x_treino)

x_treino = ajustador_de_escala.transform(x_treino)
x_teste  = ajustador_de_escala.transform(x_teste)

# Usando RandomForest

clfRF = RandomForestClassifier(
    max_depth=4,
#                              random_state=0,
                             criterion='entropy',
#                              max_features='log2',
                             min_samples_split=2,
                             min_samples_leaf=1,
                             min_weight_fraction_leaf=0.0)
# clf.fit(x_treino, y_treino)

q = 10
scores = cross_val_score(
        clfRF,
        x_treino,
        y_treino.ravel(),
        cv=q
        )

print (
#         f'k = {k}',
#     f"numero de estimadore: {k}",
        f'scores:{scores}',
        f'acurácia média = {round(100*mean(scores),2)} %'
        )


clfGB = GradientBoostingClassifier(
#     loss='exponential',
#     n_estimators=245,
    learning_rate=1.0,
     max_depth=1,
#     random_state=0,
# max_leaf_nodes=10
)

q = 10
scores = cross_val_score(
        clfGB,
        x,
        y.ravel(),
        cv=q
        )

print (
#         f'k = {k}',
        f'scores:{scores}',
        f'acurácia média = {round(100*mean(scores),2)} %'
        )


clfRF.fit(x_treino,y_treino.ravel())

clfGB.fit(x_treino,y_treino.ravel())

atributos_selecionados = [
'sexo_ ',
'sexo_M',
'sexo_F',
'sexo_N',
'possui_email',
'local_onde_reside',
'tipo_endereco',
'idade',
'estado_civil',
'qtde_dependentes',
'dia_vencimento',
'possui_telefone_residencial',
'meses_na_residencia',
'renda_mensal_regular',
'renda_extra',
'possui_cartao_visa',
'possui_cartao_mastercard',
'possui_outros_cartoes',
'qtde_contas_bancarias',
'qtde_contas_bancarias_especiais',
'valor_patrimonio_pessoal',
'possui_carro',
'vinculo_formal_com_empresa',
'meses_no_trabalho',
'profissao',
'ocupacao',
'forma_envio_solicitacao_correio',
'forma_envio_solicitacao_internet',
'forma_envio_solicitacao_presencial',
]

testando_modelo = df_teste[atributos_selecionados]


resultadoRF = clfRF.predict(testando_modelo)

resultadoGB = clfGB.predict(testando_modelo)

dict_resultadoRF = {'id_solicitante': df_teste['id_solicitante'], 'inadimplente': resultadoRF.tolist()}
df_resultado = pd.DataFrame(data=dict_resultadoRF)
df_resultado.to_csv (r'previsao_aprovacao_credito_ramdom.csv', index = False, header=True)

dict_resultadoGB = {'id_solicitante': df_teste['id_solicitante'], 'inadimplente': resultadoGB.tolist()}
df_resultado = pd.DataFrame(data=dict_resultadoGB)
df_resultado.to_csv (r'previsao_aprovacao_credito_gradient.csv', index = False, header=True)

