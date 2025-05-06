import numpy as np
from random import choices
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import os

def distance_eucli(p, x):
    return np.linalg.norm(p - x)

def distance_matrix_n(embedding_tab, comparaison_size=15, distance_metric=distance_eucli, normalisation = False):
    true_positive_matrix = []
    dmatrix = []
    for i, embedding in enumerate(embedding_tab):
        true_positive_matrix.append(np.zeros(comparaison_size + 1))
        true_positive_matrix[i][0] = 1

        # Selection des negatifs
        negatifs_id = choices(range(0, len(embedding_tab)), k=comparaison_size)
        while i in negatifs_id:
            negatifs_id = choices(range(0, len(embedding_tab)), k=comparaison_size)

        # Calcul des distances
        local_dmatrix = np.zeros(comparaison_size + 1)
        local_dmatrix[0] = distance_metric(embedding[0], embedding[1])
        for j in range(len(negatifs_id)):
            local_dmatrix[j+1] = distance_metric(embedding[0], embedding_tab[negatifs_id[j]][1])

        if normalisation:
            local_dmatrix /= np.max(local_dmatrix)
        local_dmatrix = np.where(local_dmatrix <= 0.000001, 0.000001, local_dmatrix) 
        dmatrix.append(local_dmatrix)

    true_positive_matrix = np.array(true_positive_matrix)
    dmatrix = np.array(dmatrix)

    return true_positive_matrix, dmatrix

def calculer_matrice_distances(vecteurs, distanceFunction):
    """
    Calcule la matrice des distances euclidiennes entre une liste de vecteurs.

    :param vecteurs: Liste de vecteurs (listes ou tableaux NumPy).
    :return: Matrice des distances euclidiennes entre les vecteurs.
    """
    n = len(vecteurs)  # Nombre de vecteurs
    matrice_distances = np.zeros((n, n))  # Initialiser la matrice des distances à zéro

    for i in tqdm(range(n)):
        for j in range(n):  # La matrice n est pas symetrique
            # Calculer la distance euclidienne entre les vecteurs i et j
            distance = distanceFunction(torch.squeeze(vecteurs[i][0]), torch.squeeze(vecteurs[j][1]))
            matrice_distances[i, j] = distance
            # i = anchor, j = postive

    return matrice_distances

def get_cmc_curve(distance_matrix, labels=None):
    echantillon_max = 20

    num_queries, num_candidates = distance_matrix.shape
    if labels is not None:
        matches = labels
    else:
        matches = np.diag(np.ones(len(distance_matrix)))

    cmc_scores = np.zeros(num_candidates)
    all_AP = []
    for i in range(num_queries):
        # Trier les candidats pour la requête i par score de similarité décroissant (ou distance croissante)
        sorted_indices = np.argsort(distance_matrix[i])
        sorted_matches = matches[i][sorted_indices]

        # Trouver l'indice de la première correspondance correcte
        correct_match_index = np.where(sorted_matches == 1)[0][0]
        
        # Incrémenter les scores CMC pour tous les rangs à partir de ce point
        cmc_scores[correct_match_index:] += 1

        local_cmc = np.zeros(num_candidates)
        local_cmc[correct_match_index:] += 1
        if not echantillon_max:
            echantillon_max = local_cmc.sum()
        tmp_cmc = local_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc)[:echantillon_max]
        average_precision = tmp_cmc.sum() / echantillon_max
        all_AP.append(average_precision)

    # Calculer les taux de réussite cumulatifs
    cmc_curve = cmc_scores / num_queries
    mAP = np.mean(all_AP)

    return cmc_curve, mAP

def metric_EXPLAIN(distance_matrix):
    
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    AccuracyR1Linear = ((col_ind == row_ind).sum()/len(col_ind))

    sommeDistanceBest = distance_matrix[row_ind, col_ind].sum()
    sommeDistanceTrue = distance_matrix[row_ind, row_ind].sum()

    return AccuracyR1Linear*100, sommeDistanceBest, sommeDistanceTrue


def plot_and_save(tab_resnet, folder):

    eucliTorch = lambda a, b : (a - b).pow(2).sum(-1)
    cosine_distance = lambda v1, v2: 1 - torch.dot(v1, v2) / (v1.norm() * v2.norm())

    matrice_distances = calculer_matrice_distances(tab_resnet, distanceFunction=eucliTorch)

    #CMC Curve
    cmc, mAP = get_cmc_curve(matrice_distances)

    print(f"mAP : {mAP:.3f}, CMC-1 : {cmc[0]:.3f}, CMC-5 : {cmc[5]:.3f}, CMC-10 : {cmc[10]:.3f}, CMC-20 : {cmc[20]:.3f},")
    print(f"{mAP:.3f}, {cmc[0]:.3f}, {cmc[5]:.3f}, {cmc[10]:.3f}, {cmc[20]:.3f}")
    # Tracer la courbe CMC
    plt.plot(range(1, len(matrice_distances) + 1), cmc, color='blue', label=f'CMC (Rank-1 = {cmc[0]:.3f})')
    plt.xlabel('Rang')
    plt.ylabel('Taux de réussite cumulatif')
    plt.title('Courbe CMC')
    plt.legend(loc='lower right')

    # Sauvegardez la figure
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(os.path.join(folder, 'CMC_all.png'))  # Sauvegarde en format PNG

    #Precision Recall Curve
    # print("matrice shape : ", matrice_distances.shape)
    # print("matrice shape : ", np.reshape(matrice_distances, -1).shape)
    # print("diag shape : ",np.diag(np.ones(len(matrice_distances))).shape)
    # print(np.reshape(np.diag(np.ones(len(matrice_distances))), -1).shape)
    ypred = [a.mean() / a for a in matrice_distances]
    precision, recall, thresholds = precision_recall_curve(y_true=np.reshape(np.diag(np.ones(len(matrice_distances))), -1), y_score=np.reshape(ypred, -1), pos_label=1)
    AG_precision = average_precision_score(y_true=np.reshape(np.diag(np.ones(len(matrice_distances))), -1), y_score=np.reshape(ypred, -1))

    # create precision recall curve
    plt.clf()
    plt.plot(recall, precision, color='purple', label=f'Courbe Rappel-Precision (AG = {AG_precision:.3f})')
    # add axis labels to plot
    plt.title(f'Precision-Recall Curve k={len(matrice_distances)}')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc='lower right', shadow=True, fontsize='small')

    plt.savefig(os.path.join(folder, 'PrecisionRecall_all.png'))  # Sauvegarde en format PNG


    ##-----------------------------------

    comparaison_size = [20, 30, 50, 100, len(matrice_distances)]

    # Initialiser la figure et la grille de subplots
    fig, axes = plt.subplots(nrows=3, ncols=len(comparaison_size), figsize=(15, 15))

    # Itérer sur la grille pour tracer les courbes
    for i, ax in enumerate(axes.flat):
        # Calculer l'indice de la courbe dans le tableau 'donnees'
        size = comparaison_size[i%len(comparaison_size)]
        whatToPrint = i//len(comparaison_size)

        # Calcul distance matrix
        d_mat = distance_matrix_n(tab_resnet, comparaison_size=size, distance_metric=eucliTorch, normalisation = False)
        ypred = [a.mean() / a for a in d_mat[1]]

        if whatToPrint == 0:
            cmc, mAP = get_cmc_curve(d_mat[1], d_mat[0])

            ax.plot(range(1, len(d_mat[1][0]) + 1), cmc, color='blue', label=f'CMC (Rank-1 = {cmc[0]:.3f})')
            ax.set_xlabel('Rang')
            ax.set_ylabel('Taux de réussite cumulatif')
            ax.set_title(f'Courbe CMC k={size}')
            ax.legend(loc='lower right', shadow=True, fontsize='small')

        if whatToPrint == 1:
            precision, recall, thresholds = precision_recall_curve(y_true=np.reshape(d_mat[0], -1), y_score=np.reshape(ypred, -1), pos_label=1)
            rand_precision, rand_recall, rand_thresholds = precision_recall_curve(y_true=np.reshape(d_mat[0], -1), y_score=np.random.random(len(np.reshape(d_mat[0], -1))), pos_label=1)
            AG_precision = average_precision_score(y_true=np.reshape(d_mat[0], -1), y_score=np.reshape(ypred, -1))

            # create precision recall curve
            ax.plot(recall, precision, color='purple', label=f'Courbe Rappel-Precision (AG = {AG_precision:.3f})')
            ax.plot(rand_recall[:-1], rand_precision[:-1], color='navy', lw=2, linestyle='--', label='Aléatoire')
            # add axis labels to plot
            ax.set_title(f'Precision-Recall Curve k={size}')
            ax.set_ylabel('Precision')
            ax.set_xlabel('Recall')
            ax.legend(loc='lower right', shadow=True, fontsize='small')

        if whatToPrint == 2:
            fpr, tpr, seuils = roc_curve(y_true=np.reshape(d_mat[0], -1), y_score=np.reshape(ypred, -1))
            roc_auc = roc_auc_score(y_true=np.reshape(d_mat[0], -1), y_score=np.reshape(ypred, -1))

            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire')
            ax.set_xlabel('Taux de faux positifs')
            ax.set_ylabel('Taux de vrais positifs')
            ax.set_title(f'Courbe ROC k={size}')
            ax.legend(loc='lower right', shadow=True, fontsize='small')

            # Metric EXPLAIN
            AccuracyR1Linear, sommeDistanceBest, sommeDistanceTrue = metric_EXPLAIN(matrice_distances[:size, :size])
            textToPrint = f"Accucacy EXPLAIN k={size} : {AccuracyR1Linear}%,\n Somme distance : {sommeDistanceBest},\n Somme GroundTruth : {sommeDistanceTrue}"
            ax.text(0.5,-0.2, textToPrint, size=8, ha="center", transform=ax.transAxes)

    # Ajuster l'espacement si nécessaire
    plt.tight_layout()

    # Afficher la figure
    plt.savefig(os.path.join(folder, 'Multi_metrics.png'))