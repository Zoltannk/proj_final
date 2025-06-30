from fastapi import FastAPI

app = FastAPI()


# Função para diagnóstico dos modelos
@app.get("/items/{modelo}")
def AVALIA_MOD(modelo, y_prob, y_obs, titulo, base):

    pred_Y_test = {}
    pred_Y_data = pd.DataFrame(modelo.predict_proba(y_prob)[:,1], columns=["y_prob"])
    pred_Y_data["y_hat"] = modelo.predict(y_prob)
    obs_Y_data = y_obs.reset_index()
    pred_data = pd.concat([obs_Y_data,
                         pred_Y_data], axis = 1)

    pred_data["erro2"]     = (pred_data.Target - pred_data.y_prob)**2

    print("--------------------------------------------")
    print("----------------",base,"--------------------")
    print("\n")
    KS = ks_2samp(pred_data.loc[pred_data.Target == 0, "y_prob"], pred_data.loc[pred_data.Target == 1, "y_prob"])
    KS = round(KS[0],2)*100

    ASE = round(np.mean(pred_data.erro2),2)

    AUC = roc_auc_score(pred_data.Target,pred_data.y_prob)
    GINI = 2*AUC - 1
    AUC = round(AUC, 2)*100

    # Curva Roc
    fpr, tpr, thresholds  = roc_curve(pred_data.Target,pred_data.y_prob)

    trace1 = go.Scatter(x = fpr, y = tpr,
                        mode = "lines",
                        line = dict(color = "#b30000", width = 2),
                        name = 'AUC: %0.2f' % AUC)
    trace2 = go.Scatter(x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color = "navy", width=2, dash = "dash"),
                        showlegend=False)
    layout = go.Layout(xaxis=dict(title = "FP"),
                       yaxis=dict(title = "TP"))
    plot_roc = go.Figure(data=[trace1, trace2], layout=layout)
    plot_roc.update_layout(title_text = "Curva ROC "+titulo)

    # Matriz de confusão
    cm = confusion_matrix(pred_data.Target, pred_data.y_hat)
    conf_m = cm/np.sum(cm)*100
    TP = round(conf_m[0,0],2)
    FP = round(conf_m[0,1],2)
    FN = round(conf_m[1,0],2)
    TN = round(conf_m[1,1],2)

    print("Matriz de confusão default",
         "\n","|","TP", TP,"|","FP",FP,"|","\n","|","FN", FN,"|","TN", TN,"|")
    print("\n")

    CONCORD = round(TP + TN ,2)
    PRECS   = TP/(FP+TP)
    RECALL  = TP/(FN+TP)

    LogLoss = log_loss(pred_data.Target, pred_data.y_hat)

    d = {"Metricas": ["Acurácia","Precisão","Recall","LogLoss","AUC","KS","Gini","ASE"],
         "Valores": [CONCORD,PRECS,RECALL,LogLoss,AUC, KS,GINI,ASE]}

    # Base de medidas de qualidade dos ajuste
    METRICAS = pd.DataFrame(data=d)

    # Base para gráfico de ordenação
    pred_data = pred_data.sort_values(by=["y_prob"])

    pred_data["FX_yprob"] = pd.qcut(pred_data.y_prob, 10)
    pred_data["Default"]  = np.mean(pred_data.Target)

    ordena = pred_data.groupby("FX_yprob")[["Target","Default"]].mean().reset_index()
    ordena["FX_yprob2"] = lab = np.arange(10)
    ordena["LIFT"] = ordena.Target/ordena.Default

#--------------------------------------------------------
    lab1 = titulo[7:]

    if lab1 == "Desbalanc":
        col = "#008ae6"
    else:
        col = "#8a8a5c"
#----------------------------------------------------------
    # Gráfico de ordenação
    fig_ord = data=go.Scatter(x = ordena.FX_yprob2, y = ordena.LIFT, name = "Lift "+titulo, marker=dict(color = col))
#    fig_ord.update_layout(title_text = "Ordenação: Lift "+titulo)

    print("Acurácia:", CONCORD,"|","Precisão:", round(PRECS,2),"|","Recall:", round(RECALL,2))
    print("\n")
    print("KS:", KS,"|", "AUC:",AUC,"|","GINI:",round(GINI,2), "|","LogLoss:",round(LogLoss,2), "|","ASE:", ASE)
    print("\n")
    return(pred_data,ordena,METRICAS,fig_ord,plot_roc,titulo,lab1,AUC)



# Função para avaliar importância das variáveis/features
@app.post("/items/{modelo}")
def ML_IMPORTANCIA(modelo, feat):
    importances = modelo.feature_importances_
    indices = np.argsort(importances)
    indices = pd.Series(indices)
    features = pd.Series(feat)

    print("    Results: Importância das Variáveis")
    print("============================================")
    tt = pd.Series(importances[indices],
                 index=features[indices]).sort_values(ascending=False)
    print(tt)
    print("============================================")