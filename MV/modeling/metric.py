from sklearn.metrics import accuracy_score, auc, average_precision_score, classification_report, precision_recall_curve, roc_auc_score,f1_score

def AUPRC(truth, preds):
    auprc = []
    if len(preds) == 0:
        return 0.0
    for i in range(len(truth)):
        true = [int(t) for t in truth[i]]
        pred = preds[i]
        precision, recall, _ = precision_recall_curve(true, pred)
        auprc.append(auc(recall, precision))
    return auprc

def EditDist(a,b):
    m,n = len(a)+1,len(b)+1


    d = [[0]*n for i in range(m)]

    d[0][0]=0
    for i in range(1,m):
        d[i][0] = d[i-1][0] + 1

    for j in range(1,n):
        d[0][j] = d[0][j-1]+1
    
    temp = 0

    for i in range(1,m):
        for j in range(1,n):
            if a[i-1]==b[j-1]:
                temp = 0
            else:
                temp = 1
            
            d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+temp)

    return d[m-1][n-1]