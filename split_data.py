
import os
import pandas as pd
import random
import shutil
import json
from datetime import datetime

def count_objects_in_folder(folder_path):
    ball_count = 0
    net_count = 0
    for label_file in os.listdir(folder_path):
        if not label_file.endswith('.txt'):
            continue
        with open(os.path.join(folder_path, label_file), 'r') as f:
            for line in f:
                class_id = line.strip().split()[0]
                if class_id == '1':  # Spikeball
                    ball_count += 1
                elif class_id == '0':  # Net
                    net_count += 1
    return ball_count, net_count

def prepare_dataset(base_path):
    video_folders = [
        folder for folder in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, folder)) and folder != 'labels'
    ]

    stats = []
    for folder in video_folders:
        label_path = os.path.join(base_path, folder, 'labels')
        if not os.path.exists(label_path):
            continue
        ball, net = count_objects_in_folder(label_path)
        stats.append({'video': folder, 'Spikeball': ball, 'Net': net})

    df = pd.DataFrame(stats)
    total_ball = df['Spikeball'].sum()
    total_net = df['Net'].sum()

    target = {
        'train': (0.8 * total_ball, 0.8 * total_net),
        'val': (0.1 * total_ball, 0.1 * total_net),
        'test': (0.1 * total_ball, 0.1 * total_net),
    }

    return df, target

def run_random_split_optimization(df, target, n_iterations, train_range, val_range, split_file="./DATASET/best_split.json"):
    """
    Méthode alternative plus simple et robuste utilisant des splits aléatoires optimisés.
    Sauvegarde automatiquement le meilleur split trouvé dans un fichier JSON.
    
    Args:
        df: DataFrame avec colonnes 'video', 'Spikeball', 'Net'
        target: dictionnaire avec les cibles pour chaque split
        n_iterations: nombre d'essais aléatoires
        train_range: tuple (min, max) pour le nombre de vidéos en train
        val_range: tuple (min, max) pour le nombre de vidéos en validation
        split_file: chemin vers le fichier JSON de sauvegarde
    
    Returns:
        (best_assignment, min_error)
    """
    # Charger le meilleur split existant
    best_error_so_far = float('inf')
    best_assignment_so_far = None
    improvements_found = 0

    if os.path.exists(split_file):
        try:
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            best_error_so_far = split_data['error']
            best_assignment_so_far = split_data['assignment']
            print(f"   📁 Found previous best split with error: {best_error_so_far:.2f}")
        except:
            print(f"   ⚠️  Error reading existing split file, starting fresh")
    else:
        print(f"   📄 No previous split found, starting fresh")

    # Fonction interne pour sauvegarder les améliorations
    def save_on_improvement(assignment, error, iteration_info):
        nonlocal best_error_so_far, best_assignment_so_far, improvements_found
        
        # Vérifier si c'est effectivement une amélioration
        if error < best_error_so_far:
            improvements_found += 1
            old_best = best_error_so_far
            best_error_so_far = error
            best_assignment_so_far = assignment
            
            # Créer les données de sauvegarde
            stats_for_save = summarize_distribution(df, assignment, target)
            split_data = {
                "error": float(error),
                "timestamp": datetime.now().isoformat(),
                "assignment": assignment,
                "iteration_found": iteration_info,
                "statistics": {
                    "train": {
                        "videos": stats_for_save['train']['videos'],
                        "video_count": int(stats_for_save['train']['video_count']),
                        "ball_count": int(stats_for_save['train']['ball_count']),
                        "net_count": int(stats_for_save['train']['net_count']),
                        "ball_percentage": float(stats_for_save['train']['ball_percentage']),
                        "net_percentage": float(stats_for_save['train']['net_percentage'])
                    },
                    "val": {
                        "videos": stats_for_save['val']['videos'],
                        "video_count": int(stats_for_save['val']['video_count']),
                        "ball_count": int(stats_for_save['val']['ball_count']),
                        "net_count": int(stats_for_save['val']['net_count']),
                        "ball_percentage": float(stats_for_save['val']['ball_percentage']),
                        "net_percentage": float(stats_for_save['val']['net_percentage'])
                    },
                    "test": {
                        "videos": stats_for_save['test']['videos'],
                        "video_count": int(stats_for_save['test']['video_count']),
                        "ball_count": int(stats_for_save['test']['ball_count']),
                        "net_count": int(stats_for_save['test']['net_count']),
                        "ball_percentage": float(stats_for_save['test']['ball_percentage']),
                        "net_percentage": float(stats_for_save['test']['net_percentage'])
                    }
                }
            }
            
            # Sauvegarder immédiatement
            os.makedirs(os.path.dirname(split_file), exist_ok=True)
            with open(split_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            
            improvement = old_best - error
            print(f"🎉 NEW RECORD at evaluation {iteration_info:,}!")
            print(f"   📈 Improved from {old_best:.2f} to {error:.2f} (gain: {improvement:.2f})")
            print(f"   💾 SAVED immediately to {split_file}")

    # Calculer le total de labels par vidéo
    df_work = df.copy()
    df_work['total_labels'] = df_work['Spikeball'] + df_work['Net']
    
    # Cibles totales en labels (non pas en ratios d'objets séparés)
    target_train_total = target['train'][0] + target['train'][1]  # balls + nets
    target_val_total = target['val'][0] + target['val'][1]
    target_test_total = target['test'][0] + target['test'][1]
    
    best_split = None
    min_error = float('inf')
    
    print(f"🔄 Optimisation par méthode aléatoire...")
    print(f"Cibles: Train={target_train_total:.0f}, Val={target_val_total:.0f}, Test={target_test_total:.0f} labels")
    print(f"Ranges utilisés: Train={train_range}, Val={val_range}")
    
    # Calculer le nombre total de combinaisons possibles
    train_combinations = train_range[1] - train_range[0] + 1
    val_combinations = val_range[1] - val_range[0] + 1
    total_combinations_per_iteration = train_combinations * val_combinations
    total_combinations = n_iterations * total_combinations_per_iteration
    
    print(f"Espace de recherche: {total_combinations:,} combinaisons totales")
    print(f"({n_iterations:,} iterations × {total_combinations_per_iteration} combinaisons/iteration)")
    
    successful_evaluations = 0
    
    for iteration in range(n_iterations):
        # Mélanger les vidéos aléatoirement
        shuffled = df_work.sample(frac=1, random_state=random.randint(0, 100000))
        
        # Essayer différents cutoffs
        for train_count in range(train_range[0], train_range[1] + 1):
            for val_count in range(val_range[0], val_range[1] + 1):
                # Vérifier que ça rentre dans le total
                test_count = len(df_work) - train_count - val_count
                if test_count < 1:  # Au moins 1 vidéo pour test
                    continue
                
                successful_evaluations += 1
                
                # Créer les splits
                train_set = shuffled.iloc[:train_count]
                val_set = shuffled.iloc[train_count:train_count + val_count]
                test_set = shuffled.iloc[train_count + val_count:]
                
                # Calculer les totaux pour chaque split
                train_balls = train_set['Spikeball'].sum()
                train_nets = train_set['Net'].sum()
                val_balls = val_set['Spikeball'].sum()
                val_nets = val_set['Net'].sum()
                test_balls = test_set['Spikeball'].sum()
                test_nets = test_set['Net'].sum()
                
                # Calculer l'erreur (même formule qu'Optuna)
                error = 0
                error += (train_balls - target['train'][0]) ** 2
                error += (train_nets - target['train'][1]) ** 2
                error += (val_balls - target['val'][0]) ** 2
                error += (val_nets - target['val'][1]) ** 2
                error += (test_balls - target['test'][0]) ** 2
                error += (test_nets - target['test'][1]) ** 2
                
                if error < min_error:
                    min_error = error
                    best_split = {
                        'train': train_set['video'].tolist(),
                        'val': val_set['video'].tolist(),
                        'test': test_set['video'].tolist(),
                        'error': error,
                        'stats': {
                            'train': {'balls': train_balls, 'nets': train_nets, 'count': train_count},
                            'val': {'balls': val_balls, 'nets': val_nets, 'count': val_count},
                            'test': {'balls': test_balls, 'nets': test_nets, 'count': test_count}
                        }
                    }
                    
                    # Convertir au format attendu et sauvegarder si amélioration
                    best_assignment = {}
                    for split_name in ['train', 'val', 'test']:
                        for video in best_split[split_name]:
                            best_assignment[video] = split_name
                    save_on_improvement(best_assignment, error, iteration * total_combinations_per_iteration + successful_evaluations)
        
        # Affichage du progrès
        if (iteration + 1) % 2000 == 0:
            efficiency = (successful_evaluations / ((iteration + 1) * total_combinations_per_iteration)) * 100
            print(f"  Iteration {iteration + 1}/{n_iterations}, meilleure erreur: {min_error:.0f}")
            print(f"    Efficacité: {efficiency:.1f}% ({successful_evaluations:,} évaluations valides)")
    
    # Convertir au format attendu par le reste du code
    best_assignment = {}
    for split_name in ['train', 'val', 'test']:
        for video in best_split[split_name]:
            best_assignment[video] = split_name
    
    print(f"✅ Optimisation terminée. Meilleure erreur: {min_error:.0f}")
    print(f"📈 Statistiques: {successful_evaluations:,} évaluations valides sur {total_combinations:,} possibles")
    print(f"🎯 Améliorations trouvées: {improvements_found}")
    
    # Utiliser le meilleur split global s'il est meilleur que celui de cette session
    if best_assignment_so_far is not None and best_error_so_far < min_error:
        print(f"🏆 Using best global split (error: {best_error_so_far:.2f}) instead of session best (error: {min_error:.2f})")
        final_assignment = best_assignment_so_far
        final_error = best_error_so_far
    else:
        print(f"🏆 Using session best split (error: {min_error:.2f})")
        final_assignment = best_assignment
        final_error = min_error
    
    # Affichage détaillé de la meilleure répartition
    print(f"\n🎯 MEILLEURE RÉPARTITION TROUVÉE:")
    print("=" * 50)
    
    total_balls = sum([best_split['stats'][split]['balls'] for split in ['train', 'val', 'test']])
    total_nets = sum([best_split['stats'][split]['nets'] for split in ['train', 'val', 'test']])
    total_videos = sum([best_split['stats'][split]['count'] for split in ['train', 'val', 'test']])
    
    for split_name in ['train', 'val', 'test']:
        stats = best_split['stats'][split_name]
        target_balls = target[split_name][0]
        target_nets = target[split_name][1]
        
        ball_pct = (stats['balls'] / total_balls) * 100 if total_balls > 0 else 0
        net_pct = (stats['nets'] / total_nets) * 100 if total_nets > 0 else 0
        video_pct = (stats['count'] / total_videos) * 100 if total_videos > 0 else 0
        
        ball_error = abs(stats['balls'] - target_balls)
        net_error = abs(stats['nets'] - target_nets)
        
        print(f"{split_name.upper()} SET ({stats['count']} vidéos, {video_pct:.1f}% du total):")
        print(f"  Balls: {stats['balls']}/{target_balls:.0f} ({ball_pct:.1f}% du total, erreur: {ball_error:.0f})")
        print(f"  Nets:  {stats['nets']}/{target_nets:.0f} ({net_pct:.1f}% du total, erreur: {net_error:.0f})")
        print(f"  Vidéos: {', '.join(best_split[split_name][:3])}{'...' if len(best_split[split_name]) > 3 else ''}")
        print()
    
    print(f"🎯 Erreur globale d'optimisation: {final_error:.2f}")
    
    return final_assignment, final_error

def summarize_distribution(df, assignment, target):
    summary = {'train': [], 'val': [], 'test': []}
    for vid, split in assignment.items():
        summary[split].append(vid)

    final_stats = {}
    total_ball = df['Spikeball'].sum()
    total_net = df['Net'].sum()

    for split in ['train', 'val', 'test']:
        if len(summary[split]) == 0:
            continue  # skip empty splits
        split_df = df[df['video'].isin(summary[split])]
        ball_count = split_df['Spikeball'].sum()
        net_count = split_df['Net'].sum()
        ball_pct = (ball_count / total_ball) * 100 if total_ball > 0 else 0
        net_pct = (net_count / total_net) * 100 if total_net > 0 else 0

        final_stats[split] = {
            'videos': summary[split],
            'video_count': len(summary[split]),
            'ball_count': ball_count,
            'net_count': net_count,
            'ball_percentage': ball_pct,
            'net_percentage': net_pct
        }

    return final_stats

def compute_video_count_range_for_split(df, target_ratio, object_totals, object_keys):
    """
    Calcule le nombre de vidéos minimum et maximum pour atteindre les ratios donnés
    pour chaque split (ex: 0.8 pour train, 0.1 pour val/test).

    Pour le minimum : 
    - Pour chaque objet (net, spikeball), trier les vidéos par cet objet décroissant
    - Trouver combien de vidéos il faut pour dépasser la proportion
    - Prendre ce nombre - 1
    - Prendre le MAX entre net et spikeball

    Pour le maximum : on prend le MIN entre net et spikeball (le premier qui dépasse trop)

    Args:
        df: DataFrame avec colonnes 'Spikeball', 'Net' (par vidéo)
        target_ratio: float (ex: 0.8 pour train)
        object_totals: dict avec les totaux globaux, ex: {'Spikeball': 4465, 'Net': 1256}
        object_keys: noms des colonnes

    Returns:
        (min_video_count, max_video_count)
    """
    
    min_videos_per_obj = {}
    
    # Calcul du minimum pour chaque objet séparément
    for obj in object_keys:
        # Trier par cet objet spécifique décroissant
        df_sorted_obj = df.copy().sort_values(by=obj, ascending=False).reset_index(drop=True)
        
        cumulative = 0
        target_value = target_ratio * object_totals[obj]
        
        print(f"  Calcul MIN pour {obj} (cible: {target_value:.0f}):")
        
        for i in range(len(df_sorted_obj)):
            cumulative += df_sorted_obj.loc[i, obj]
            
            # Debug pour les premières vidéos et près de l'objectif
            if i < 3 or cumulative >= target_value * 0.9:
                print(f"    Vidéo {i+1}: +{df_sorted_obj.loc[i, obj]} → cumul = {cumulative}/{target_value:.0f} ({cumulative/target_value*100:.1f}%)")
            
            # Quand on dépasse la cible
            if cumulative >= target_value:
                min_videos_per_obj[obj] = max(1, i)  # i vidéos suffisent, donc min = i (mais au moins 1)
                print(f"    ✅ {obj} dépasse à la vidéo {i+1}, donc min = {i} vidéos")
                break
        else:
            # Si on n'atteint jamais la cible, on prend toutes les vidéos
            min_videos_per_obj[obj] = len(df_sorted_obj)
            print(f"    ⚠️  {obj} n'atteint jamais la cible, min = {len(df_sorted_obj)} vidéos")
    
    # Le minimum global est le MAX entre les objets (le plus contraignant)
    min_videos = max(min_videos_per_obj.values())
    print(f"  Debug MIN - Par objet: {min_videos_per_obj}, min global: {min_videos} (MAX)")

    # Calcul du maximum : pour chaque objet, trier par croissant (vidéos les plus pauvres d'abord)
    # et voir combien il faut pour atteindre la proportion, puis prendre le MIN entre les objets
    max_videos_per_obj = {}
    
    for obj in object_keys:
        # Trier par cet objet spécifique CROISSANT (du moins au plus de labels)
        df_sorted_obj = df.copy().sort_values(by=obj, ascending=True).reset_index(drop=True)
        
        cumulative = 0
        target_value = target_ratio * object_totals[obj]
        
        print(f"  Calcul MAX pour {obj} (cible: {target_value:.0f}, tri croissant):")
        
        for i in range(len(df_sorted_obj)):
            cumulative += df_sorted_obj.loc[i, obj]
            
            # Debug pour les premières vidéos et près de l'objectif
            if i < 3 or cumulative >= target_value * 0.9:
                print(f"    Vidéo {i+1}: +{df_sorted_obj.loc[i, obj]} → cumul = {cumulative}/{target_value:.0f} ({cumulative/target_value*100:.1f}%)")
            
            # Quand on dépasse la cible
            if cumulative >= target_value:
                max_videos_per_obj[obj] = max(1, i)  # i vidéos suffisent, donc max = i (mais au moins 1)
                print(f"    ✅ {obj} dépasse à la vidéo {i+1}, donc max = {i} vidéos")
                break
        else:
            # Si on n'atteint jamais la cible, on prend toutes les vidéos
            max_videos_per_obj[obj] = len(df_sorted_obj)
            print(f"    ⚠️  {obj} n'atteint jamais la cible, max = {len(df_sorted_obj)} vidéos")
    
    # Le maximum global est le MIN entre les objets (le plus contraignant pour les vidéos pauvres)
    max_videos = min(max_videos_per_obj.values())
    print(f"  Debug MAX - Par objet: {max_videos_per_obj}, max global: {max_videos} (MIN)")

    # S'assurer que max >= min (cohérence)
    if max_videos < min_videos:
        print(f"  Attention: max_videos ({max_videos}) < min_videos ({min_videos}), ajustement à min_videos")
        max_videos = min_videos

    return min_videos, max_videos

def create_split_folders(assignment, source_base_path, output_base_path):
    """
    Create train/val/test folders and copy images/labels according to the assignment
    
    Args:
        assignment: dict mapping video names to splits ('train', 'val', 'test')
        source_base_path: path to dataset_by_video folder
        output_base_path: path where to create the split folders
    
    Returns:
        dict with count of copied files per split
    """
    
    # Create base output directory
    os.makedirs(output_base_path, exist_ok=True)
    
    # Create split directories
    for split in ['train', 'val', 'test']:
        split_images_dir = os.path.join(output_base_path, split, 'images')
        split_labels_dir = os.path.join(output_base_path, split, 'labels')
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)
    
    # Copy files according to assignment
    copied_files = {'train': 0, 'val': 0, 'test': 0}
    
    for video, split in assignment.items():
        video_path = os.path.join(source_base_path, video)
        
        if not os.path.exists(video_path):
            print(f"⚠️  Video folder not found: {video}")
            continue
            
        # Copy images
        images_source = os.path.join(video_path, 'images')
        labels_source = os.path.join(video_path, 'labels')
        
        if os.path.exists(images_source):
            for img_file in os.listdir(images_source):
                if img_file.endswith(('.jpg', '.png', '.jpeg')):
                    src = os.path.join(images_source, img_file)
                    dst = os.path.join(output_base_path, split, 'images', img_file)
                    shutil.copy2(src, dst)
                    copied_files[split] += 1
        
        # Copy labels  
        if os.path.exists(labels_source):
            for lbl_file in os.listdir(labels_source):
                if lbl_file.endswith('.txt'):
                    src = os.path.join(labels_source, lbl_file)
                    dst = os.path.join(output_base_path, split, 'labels', lbl_file)
                    shutil.copy2(src, dst)
    
    return copied_files
