# Pipeline de Traitement des Données Spikeball / Roundnet

## 1. Récupération des datasets

J'ai sélectionné les trois seuls datasets pertinents disponibles en ligne, tous hébergés sur Roboflow. Ces datasets contiennent des annotations pour les objets suivants :
- Spikeball
- Net
- (pour le dataset "Roundnet Project" uniquement) Person with Ball

### Liens vers les datasets Roboflow :
- [Dataset 1 : Roundnet AI](https://universe.roboflow.com/damsito/roundnet-ai-11dx2/dataset/2)
- [Dataset 2 : Roundnet Project](https://universe.roboflow.com/damsito/roundnet-project-yjqis/dataset/2)
- [Dataset 3 : Spikeball RoundnetAI](https://universe.roboflow.com/damsito/spikeball-roundnetai-2cstd/dataset/2)

## 2. Téléchargement et format

Chaque dataset a été téléchargé au format YOLO, en sélectionnant toutes les images dans le set de train afin de faciliter leur regroupement et d'éviter les doublons ou séparations inutiles entre train/valid/test.

## 3. Regroupement des données

Toutes les images et leurs labels ont été fusionnés dans un seul dossier de travail, ce qui permet d'avoir un dataset global et cohérent pour l'entraînement des modèles.

## 4. Résumé des labels disponibles

- **Spikeball** : présent dans tous les datasets
- **Net** : présent dans tous les datasets
- **Person with Ball** : uniquement dans "Roundnet Project"

## 5. Objectif

L'objectif de cette pipeline est de constituer un dataset riche et varié pour la détection d'objets liés au Spikeball/Roundnet, en maximisant la diversité des sources et la cohérence des annotations.
