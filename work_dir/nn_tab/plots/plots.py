import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn import metrics
import numpy as np
import pandas as pd
from ..logger import get_logger


def auc_plot(y_preds, y_test_data, responsescore_prob, exp_name):
    logger = get_logger()
    logger.info("Generating AUC plots and computing classification metrics")
    
    y_hat = [1 if y_preds[i]>=0.5 else 0 for i in range(len(y_preds))]
    logger.info(f"Classification report:\n{classification_report(y_test_data, y_hat)}")

    f1_score_class_1 = f1_score(y_test_data, y_hat, pos_label=1)
    logger.info(f'F1 score class 1: {f1_score_class_1}')

    f1_score_class_0 = f1_score(y_test_data, y_hat, pos_label=0)
    logger.info(f'F1 score class 0: {f1_score_class_0}')

    aucpr = average_precision_score(y_test_data, y_preds)
    logger.info(f"AUCPR: {aucpr}")
    
    
    y_hat_rs = [0 if el<0.5 else 1 for el in responsescore_prob]
    
    f1_score_class_1_rs = f1_score(y_test_data, y_hat_rs, pos_label=1)
    logger.info(f'F1 score class 1 (response score): {f1_score_class_1_rs}')

    f1_score_class_0_rs = f1_score(y_test_data, y_hat_rs, pos_label=0)
    logger.info(f'F1 score class 0 (response score): {f1_score_class_0_rs}')

    aucpr_rs = average_precision_score(y_test_data, responsescore_prob)
    logger.info(f"AUCPR (response score): {aucpr_rs}")
    
    fpr_new, tpr_new, _ = metrics.roc_curve(y_test_data,y_preds )
    auc_new = metrics.roc_auc_score(y_test_data,y_preds)
    
    
    fpr_old, tpr_old, _ = metrics.roc_curve(y_test_data,responsescore_prob )
    auc_old = metrics.roc_auc_score(y_test_data,responsescore_prob)
    
    plt.figure(figsize=(10,6))
    plt.plot(fpr_old,tpr_old,label="Responsescore, auc="+str(np.round(auc_old, decimals=4)))

    plt.plot(fpr_new,tpr_new,label="Neural_Net, auc="+str(np.round(auc_new, decimals=4)))

    plt.legend(loc=4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    
    # Create directory inside current working dir
    plot_dir = os.path.join(os.getcwd(), "plots", exp_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save plot
    plot_path = os.path.join(plot_dir, "auc_plots.jpg")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    logger.info(f"AUC plot saved at: {plot_path}")

    logger.info(f"AUC ROC neural net: {auc_new}")
    
    
## Function to get the important business KPI's
def plot_info(y_test_data, responsescore_prob, y_preds, name, test_data, exp_name):
    threshold = [0.01*i for i in range(1,101)]
    df = {'fraud_sw':y_test_data, 'responsescore_divided': responsescore_prob, 'pred_prob':y_preds, 'amt_usd':test_data['amt_usd']}
    data = pd.DataFrame(df)
    total_amt = data['amt_usd'].sum()
    resp3_data =[]


    ## KPI for responsescore
    for thresh in threshold:

        
        tp = data[(data['fraud_sw']==1) & (data['responsescore_divided']>=thresh)].shape[0]
        fp = data[(data['fraud_sw']==0) & (data['responsescore_divided']>=thresh)].shape[0]

        tn = data[(data['fraud_sw']==0) & (data['responsescore_divided']<thresh)].shape[0]
        fn = data[(data['fraud_sw']==1) & (data['responsescore_divided']<thresh)].shape[0]


        ttnr = tn/ (tn+ fp)

        tdr = tp/ (tp+ fn)
        
        if tp!= 0:
            tfpr = fp/tp
        else:
            tfpr=0
            
        # tfpr = fp / (fp + tn)

        fraud_bps = tp / data.shape[0]*10000
        amt_bps = (data[(data[ 'fraud_sw']==1) & (data['responsescore_divided']>=thresh)]['amt_usd']).sum()/total_amt*10000
        
        resp3_data.append([thresh,tp,fp,tn,fn,ttnr,tfpr,tdr, fraud_bps,amt_bps])
    
    df_resp3_data = pd.DataFrame(resp3_data)
    df_resp3_data.columns = ["threshold", "tp", "fp", "tn", "fn", "ttnr", "tfpr", "tdr", "fraud_bps","amt_bps"]

    # Create directory inside current working dir
    kpi_dir = os.path.join(os.getcwd(), "KPI", exp_name)
    os.makedirs(kpi_dir, exist_ok=True)
    
    # Save CSV
    kpi_rs3_path = os.path.join(kpi_dir, "kpi_rs3.csv")
    df_resp3_data.to_csv(kpi_rs3_path)

    ## KPI's for our model
    pred_data =[]
    for thresh in threshold:
  
        tp = data[(data['fraud_sw']==1) & (data['pred_prob']>=thresh)].shape[0]
        fp = data[(data['fraud_sw']==0) & (data['pred_prob']>=thresh)].shape[0]

        tn = data[(data['fraud_sw']==0) & (data['pred_prob']<thresh)].shape[0]
        fn = data[(data['fraud_sw']==1) & (data['pred_prob']<thresh)].shape[0]


        ttnr = tn/ (tn+ fp)

        tdr = tp/ (tp+ fn)

        if tp!= 0:
            tfpr = fp/tp
        else:
            tfpr=0
        
      

        fraud_bps = tp / data.shape[0]*10000
        amt_bps = (data[(data[ 'fraud_sw']==1) & (data['pred_prob']>=thresh)]['amt_usd']).sum()/total_amt*10000
        pred_data.append([thresh,tp,fp,tn,fn,ttnr,tfpr,tdr, fraud_bps,amt_bps])

    df_pred_data = pd.DataFrame(pred_data)
    df_pred_data.columns = ["threshold", "tp", "fp", "tn", "fn", "ttnr", "tfpr", "tdr", "fraud_bps","amt_bps"]

    # Save CSV
    kpi_model_path = os.path.join(kpi_dir, "kpi_model.csv")
    df_pred_data.to_csv(kpi_model_path)

    return df_resp3_data, df_pred_data


def ttnr_tdr(df_resp3, df_pred, name, exp_name):
    plt.figure(figsize=(10,6))
    plt.plot(df_pred['ttnr'], df_pred['tdr'],label='neural_net', color='blue')
    plt.plot(df_resp3['ttnr'], df_resp3['tdr'],label='Response_Score', color='red')

    plt.xlabel('TTNR')
    plt.ylabel('TTDR')
    plt.title('TTNR vs. TTDR')

    # Add a legend
    plt.legend()
    
    # Create directory inside current working dir
    plot_dir = os.path.join(os.getcwd(), "plots", exp_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save plot
    plot_path = os.path.join(plot_dir, f"{name}_ttnr_vs_tdr.jpg")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    
    
def ttnr_tfpr(df_resp3, df_pred, name, exp_name):
    plt.figure(figsize=(10,6))
    plt.plot(df_pred['ttnr'], df_pred['tfpr'], label='neural_net', color='blue')
    plt.plot(df_resp3['ttnr'], df_resp3['tfpr'], label='Response_Score', color='red')

    plt.xlabel('TTNR')
    plt.ylabel('TFPR')  # Replace 'TDR' with 'FPR'
    plt.title('TTNR vs. TFPR')  # Update the title accordingly

    # Add a legend
    plt.legend()
    
    # Create directory inside current working dir
    plot_dir = os.path.join(os.getcwd(), "plots", exp_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save plot
    plot_path = os.path.join(plot_dir, f"{name}_TTNR_vs_TFPR.jpg")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()



def ttnr_fraud_bps(df_resp3, df_pred, name, exp_name):
    plt.figure(figsize=(10,6))
    plt.plot(df_pred['ttnr'], df_pred['fraud_bps'], label='neural_net', color='blue')
    plt.plot(df_resp3['ttnr'], df_resp3['fraud_bps'], label='Response_Score', color='red')


    plt.xlabel('TTNR')
    plt.ylabel('Basis Points')
    plt.title('TTNR vs. Basis Points')

    # Add a legend
    plt.legend()
    
    # Create directory inside current working dir
    plot_dir = os.path.join(os.getcwd(), "plots", exp_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save plot
    plot_path = os.path.join(plot_dir, f"{name}_TTNR_vs_Basis_Points.jpg")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()




## Get the plots for business KPI's
def tnxs_plots(y_preds, y_test_data, responsescore_prob, name, test_data, exp_name):
    df_resp3 , df_pred = plot_info(y_test_data, responsescore_prob, y_preds, name, test_data, exp_name)
    df_resp3.sort_values(by='threshold', inplace= True)
    df_pred.sort_values(by='threshold', inplace= True)

    ttnr_tdr(df_resp3, df_pred, name, exp_name)
    ttnr_tfpr(df_resp3, df_pred, name, exp_name)
    ttnr_fraud_bps(df_resp3, df_pred, name, exp_name)




# function to plot the loss curves
def plot_loss_curve(exp_name, model_name, train_losses, val_losses, title='Loss Curve'):
    logger = get_logger()
    logger.info("Generating loss curve plot for training and validation losses")
    
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Training_loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation_loss')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Create directory inside current working dir with nested structure
    plot_dir = os.path.join(os.getcwd(), "plots", exp_name, model_name)
    os.makedirs(plot_dir, exist_ok=True)

    # Save plot
    plot_path = os.path.join(plot_dir, f"{model_name}_loss.jpg")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Loss curve saved at: {plot_path}")
    logger.info("Training completed successfully!")