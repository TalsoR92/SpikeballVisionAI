import os
import shutil
import re
from collections import defaultdict
import yaml

def delete_txt_files_in_subfolders(root_path):
    """
    Supprime tous les fichiers .txt dans tous les dossiers contenus dans le répertoire donné.
    
    Args:
        root_path (str): Chemin du répertoire contenant les dossiers à scanner.
    """
    if not os.path.exists(root_path):
        print(f"❌ Le répertoire '{root_path}' n'existe pas.")
        return
    
    deleted_count = 0
    
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path, folder_name)
        
        # Vérifie que c'est bien un dossier
        if not os.path.isdir(folder_path):
            continue
            
        print(f"📁 Scan du dossier : {folder_name}")
        
        # Parcourt tous les fichiers dans ce dossier
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                try:
                    os.remove(file_path)
                    print(f"🗑️ Supprimé : {filename}")
                    deleted_count += 1
                except OSError as e:
                    print(f"❌ Erreur lors de la suppression de {filename}: {e}")
    
    print(f"\n✅ Suppression terminée. {deleted_count} fichiers .txt supprimés au total.")


def gather_images_and_labels(root_path, dest_folder):
    """
    Gather all images and labels from the 'train' subfolders of each dataset in the given root path
    into a new destination folder.

    Args:
        root_path (str): Path containing the datasets to scan.
        dest_folder (str): Path to the folder where all images and labels will be copied.
    """
    # Create images and labels subfolders in the destination folder
    images_dest = os.path.join(dest_folder, 'images')
    labels_dest = os.path.join(dest_folder, 'labels')
    os.makedirs(images_dest, exist_ok=True)
    os.makedirs(labels_dest, exist_ok=True)

    for dataset_folder in os.listdir(root_path):
        dataset_path = os.path.join(root_path, dataset_folder)
        if not os.path.isdir(dataset_path):
            continue
        train_path = os.path.join(dataset_path, 'train')
        if not os.path.exists(train_path):
            continue
        for subdir in ['images', 'labels']:
            subdir_path = os.path.join(train_path, subdir)
            if not os.path.exists(subdir_path):
                continue
            # Choose the right destination subfolder
            dest_subfolder = images_dest if subdir == 'images' else labels_dest
            for filename in os.listdir(subdir_path):
                src = os.path.join(subdir_path, filename)
                dst = os.path.join(dest_subfolder, filename)
                # To avoid filename collisions, prefix with the dataset folder name
                if os.path.exists(dst):
                    dst = os.path.join(dest_subfolder, f"{dataset_folder}_{filename}")
                shutil.copy2(src, dst)


def get_base_name(filename):
    # Extrait le nom avant ".rf"
    if '.rf.' in filename:
        return filename.split('.rf.')[0]
    return os.path.splitext(filename)[0]

def merge_datasets(dataset_dirs, output_dir, priority_dataset="Roundnet Project"):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    candidates = defaultdict(list)

    # Indexe toutes les images
    for dataset in dataset_dirs:
        image_dir = os.path.join(dataset, "train/images")
        for fname in os.listdir(image_dir):
            if fname.endswith((".jpg", ".png")):
                base = get_base_name(fname)
                candidates[base].append((dataset, fname))

    print(f"🧩 {len(candidates)} groupes d’images uniques détectés (via nom avant '.rf')")

    kept = 0
    for base, versions in candidates.items():
        chosen = None

        # On privilégie l'image venant du dataset "Roundnet Project"
        for dataset, fname in versions:
            if priority_dataset in dataset:
                chosen = (dataset, fname)
                break

        if not chosen:
            chosen = versions[0]  # sinon on garde la première

        dataset, fname = chosen
        img_src = os.path.join(dataset, "train/images", fname)
        lbl_src = os.path.join(dataset, "train/labels", os.path.splitext(fname)[0] + ".txt")

        # Nom final (on garde le même)
        img_dst = os.path.join(output_dir, "images", fname)
        lbl_dst = os.path.join(output_dir, "labels", os.path.splitext(fname)[0] + ".txt")

        shutil.copy(img_src, img_dst)
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_dst)
        else:
            print(f"⚠️ Pas de label trouvé pour {fname}")

        kept += 1

    print(f"\n✅ Fusion terminée. {kept} images retenues et copiées dans {output_dir}")


def sort_images_by_video(source_dir, root_dir=None):
    """
    Trie les images et labels par vidéo en utilisant le préfixe jusqu'à '_f_'.
    Crée des dossiers séparés pour chaque vidéo avec des sous-dossiers 'images' et 'labels'.
    Les fichiers originaux sont copiés (préservés dans le dossier source).

    Args:
        source_dir (str): Dossier source contenant les images et labels à classer.
        root_dir (str, optional): Dossier racine où créer les nouveaux dossiers. 
                                 Si None, utilise le répertoire courant.
    """
    if root_dir is None:
        root_dir = os.getcwd()
    
    if not os.path.exists(source_dir):
        print(f"❌ Le dossier source '{source_dir}' n'existe pas.")
        return
    
    # Expression régulière pour extraire le préfixe jusqu'à "_f_"
    pattern = re.compile(r'^(.*?_f_)')
    
    copied_images = 0
    copied_labels = 0
    
    print(f"📁 Tri des fichiers depuis '{source_dir}' vers '{root_dir}'...")
    
    # Parcours des fichiers dans le dossier source
    for filename in os.listdir(source_dir):
        src_path = os.path.join(source_dir, filename)
        
        # Ignore les dossiers
        if os.path.isdir(src_path):
            continue
            
        match = pattern.match(filename)
        if not match:
            print(f"⚠️ Pas de préfixe '_f_' trouvé pour : {filename}")
            continue
            
        prefix = match.group(1).rstrip("_")  # exemple : "IMG_1588_f"
        
        # Images
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            dest_dir = os.path.join(root_dir, prefix, "images")
            os.makedirs(dest_dir, exist_ok=True)
            dst_path = os.path.join(dest_dir, filename)
            shutil.copy(src_path, dst_path)
            print(f"✅ Image copiée : {filename} → {prefix}/images/")
            copied_images += 1
            
        # Labels (supposés être .txt)
        elif filename.lower().endswith('.txt'):
            dest_dir = os.path.join(root_dir, prefix, "labels")
            os.makedirs(dest_dir, exist_ok=True)
            dst_path = os.path.join(dest_dir, filename)
            shutil.copy(src_path, dst_path)
            print(f"✅ Label copié : {filename} → {prefix}/labels/")
            copied_labels += 1
    
    print(f"\n🎉 Classement terminé ! {copied_images} images et {copied_labels} labels copiés.")


def clean_labels_by_class_ids_silent(dataset_dir, valid_class_ids):
    """
    Version silencieuse de clean_labels_by_class_ids pour vérifications internes.
    
    Args:
        dataset_dir (str): Chemin vers le dossier du dataset
        valid_class_ids (list): Liste des IDs de classes valides
    
    Returns:
        dict: Statistiques de nettoyage
    """
    train_dir = os.path.join(dataset_dir, "train")
    labels_dir = os.path.join(train_dir, "labels")
    
    if not os.path.exists(labels_dir):
        return None
    
    valid_ids_str = [str(id) for id in valid_class_ids]
    
    stats = {
        'files_processed': 0,
        'files_modified': 0,
        'lines_removed': 0,
        'lines_kept': 0,
        'invalid_ids_found': set()
    }
    
    for file_name in os.listdir(labels_dir):
        if not file_name.endswith('.txt'):
            continue
            
        label_path = os.path.join(labels_dir, file_name)
        new_lines = []
        file_modified = False
        
        stats['files_processed'] += 1
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # Ligne vide
                    new_lines.append(line)
                    continue
                    
                parts = line.split()
                if parts:
                    class_id = parts[0]
                    if class_id in valid_ids_str:
                        new_lines.append(line)
                        stats['lines_kept'] += 1
                    else:
                        file_modified = True
                        stats['lines_removed'] += 1
                        stats['invalid_ids_found'].add(class_id)
        
        # Réécrire le fichier si modifié
        if file_modified:
            with open(label_path, 'w') as f:
                for line in new_lines:
                    f.write(line + '\n')
            stats['files_modified'] += 1
    
    return stats


def clean_labels_by_class_ids(dataset_dir, valid_class_ids):
    """
    Supprime toutes les lignes de labels qui n'ont pas un ID de classe valide.
    Utile quand les fichiers .txt contiennent des IDs obsolètes.
    
    Args:
        dataset_dir (str): Chemin vers le dossier du dataset (contenant train/images et train/labels)
        valid_class_ids (list): Liste des IDs de classes valides (ex: [0, 1])
    
    Returns:
        dict: Statistiques de nettoyage
    """
    train_dir = os.path.join(dataset_dir, "train")
    labels_dir = os.path.join(train_dir, "labels")
    
    if not os.path.exists(labels_dir):
        print(f"❌ Le dossier labels '{labels_dir}' n'existe pas.")
        return None
    
    valid_ids_str = [str(id) for id in valid_class_ids]
    
    stats = {
        'files_processed': 0,
        'files_modified': 0,
        'lines_removed': 0,
        'lines_kept': 0,
        'invalid_ids_found': set()
    }
    
    print(f"🔧 NETTOYAGE DES LABELS PAR ID DE CLASSE")
    print(f"📂 Dataset: {dataset_dir}")
    print(f"✅ IDs valides: {valid_class_ids}")
    print("=" * 50)
    
    for file_name in os.listdir(labels_dir):
        if not file_name.endswith('.txt'):
            continue
            
        label_path = os.path.join(labels_dir, file_name)
        new_lines = []
        file_modified = False
        lines_removed_this_file = 0
        
        stats['files_processed'] += 1
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # Ligne vide
                    new_lines.append(line)
                    continue
                    
                parts = line.split()
                if parts:
                    class_id = parts[0]
                    if class_id in valid_ids_str:
                        # ID valide, on garde la ligne
                        new_lines.append(line)
                        stats['lines_kept'] += 1
                    else:
                        # ID invalide, on supprime la ligne
                        file_modified = True
                        lines_removed_this_file += 1
                        stats['lines_removed'] += 1
                        stats['invalid_ids_found'].add(class_id)
        
        # Réécrire le fichier si modifié
        if file_modified:
            with open(label_path, 'w') as f:
                for line in new_lines:
                    f.write(line + '\n')
            stats['files_modified'] += 1
            print(f"🗑️ {file_name}: {lines_removed_this_file} lignes supprimées")
    
    print(f"\n✅ NETTOYAGE TERMINÉ:")
    print(f"   📁 Fichiers traités: {stats['files_processed']}")
    print(f"   📝 Fichiers modifiés: {stats['files_modified']}")
    print(f"   🗑️ Lignes supprimées: {stats['lines_removed']}")
    print(f"   ✅ Lignes conservées: {stats['lines_kept']}")
    
    if stats['invalid_ids_found']:
        print(f"   ❌ IDs invalides trouvés et supprimés: {sorted(stats['invalid_ids_found'])}")
    
    return stats


def remap_yolo_labels_inplace(dataset_dir, new_class_order):
    """
    Modifie les labels YOLO en place dans un dataset existant et met à jour le fichier data.yaml.
    Lit l'ordre actuel des classes depuis data.yaml et les réorganise selon le nouvel ordre.
    
    Args:
        dataset_dir (str): Chemin vers le dossier du dataset (contenant train/images et train/labels)
        new_class_order (list): Liste des noms de classes dans le nouvel ordre voulu
    
    Raises:
        ValueError: Si le nombre de classes ne correspond pas ou si des classes sont manquantes
        FileNotFoundError: Si le fichier data.yaml n'existe pas
    """
    train_dir = os.path.join(dataset_dir, "train")
    labels_dir = os.path.join(train_dir, "labels")
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    
    if not os.path.exists(labels_dir):
        print(f"❌ Le dossier labels '{labels_dir}' n'existe pas.")
        return
        
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"❌ Le fichier data.yaml n'existe pas dans '{dataset_dir}'")
    
    # Lit l'ordre actuel des classes depuis data.yaml
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    if 'names' not in yaml_data:
        raise ValueError("❌ Le fichier data.yaml ne contient pas de section 'names'")
    
    old_names = yaml_data['names']
    
    # Vérifications
    # Vérifier que toutes les classes dans new_class_order existent dans old_names
    missing_classes = set(new_class_order) - set(old_names)
    if missing_classes:
        raise ValueError(f"❌ Classes introuvables dans le dataset actuel: {missing_classes}")
    
    # Classes qui seront supprimées (présentes dans old_names mais pas dans new_class_order)
    classes_to_remove = set(old_names) - set(new_class_order)
    
    if classes_to_remove:
        print(f"⚠️ ATTENTION: Les classes suivantes seront SUPPRIMÉES du dataset:")
        for cls in sorted(classes_to_remove):
            old_idx = old_names.index(cls)
            print(f"   🗑️ '{cls}' (ID: {old_idx}) sera supprimée")
        print(f"   📊 Total: {len(classes_to_remove)} classes seront supprimées")
    
    # Si l'ordre est déjà correct, on vérifie quand même les fichiers de labels
    # car il peut y avoir des incohérences entre data.yaml et les fichiers .txt
    if old_names == new_class_order:
        print("✅ L'ordre des classes dans data.yaml est déjà correct.")
        print("🔍 Vérification des fichiers pour d'éventuelles incohérences...")
        
        # Vérifier directement les fichiers - logique intégrée
        valid_ids_str = [str(i) for i in range(len(new_class_order))]
        total_removed = 0
        files_modified = 0
        
        for file_name in os.listdir(labels_dir):
            if not file_name.endswith('.txt'):
                continue
                
            label_path = os.path.join(labels_dir, file_name)
            new_lines = []
            file_modified = False
            
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Ligne vide
                        new_lines.append(line)
                        continue
                        
                    parts = line.split()
                    if parts:
                        class_id = parts[0]
                        if class_id in valid_ids_str:
                            new_lines.append(line)
                        else:
                            file_modified = True
                            total_removed += 1
            
            # Réécrire le fichier si modifié
            if file_modified:
                with open(label_path, 'w') as f:
                    for line in new_lines:
                        f.write(line + '\n')
                files_modified += 1
        
        if total_removed > 0:
            print(f"⚠️ Incohérence détectée ! {total_removed} lignes avec des IDs invalides supprimées de {files_modified} fichiers.")
            
            # Mettre à jour le data.yaml pour être sûr
            yaml_content = {
                "nc": len(new_class_order),
                "names": new_class_order
            }
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False)
            
            return f"✅ Incohérences corrigées dans : {dataset_dir}"
        else:
            print("✅ Aucune incohérence détectée, tout est conforme.")
            return f"✅ Aucune modification nécessaire : {dataset_dir}"
    
    print(f"🔄 Changement: {old_names} → {new_class_order}")
    if len(new_class_order) < len(old_names):
        print(f"📉 Réduction du nombre de classes: {len(old_names)} → {len(new_class_order)}")
    
    # Mapping des classes - seulement pour les classes conservées
    remap_dict = {}
    for i, name in enumerate(old_names):
        if name in new_class_order:
            # Cette classe sera conservée et remappée
            remap_dict[str(i)] = str(new_class_order.index(name))
        else:
            # Cette classe sera supprimée
            remap_dict[str(i)] = None  # Marquée pour suppression

    # Premier passage : identifier toutes les classes présentes dans les fichiers
    print("🔍 Analyse des fichiers labels pour détecter les classes présentes...")
    all_classes_found = set()
    
    for file_name in os.listdir(labels_dir):
        if not file_name.endswith('.txt'):
            continue

        label_path = os.path.join(labels_dir, file_name)
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    all_classes_found.add(parts[0])
    
    # Identifier les classes qui seront supprimées
    valid_ids = [str(i) for i in range(len(new_class_order))]
    classes_to_remove = all_classes_found - set(valid_ids)
    
    if classes_to_remove:
        print(f"🗑️ ATTENTION: Les IDs de classes suivants seront SUPPRIMÉS des fichiers:")
        for class_id in sorted(classes_to_remove):
            print(f"   ❌ ID '{class_id}' sera supprimé")
        print(f"   📊 Total: {len(classes_to_remove)} IDs de classes seront supprimés\n")
    else:
        print("✅ Tous les IDs de classes dans les fichiers sont valides\n")
    
    # Deuxième passage : traitement avec nettoyage complet
    print("🔧 Application du remapping et nettoyage des IDs invalides...")
    
    modified_count = 0
    
    # Traitement du remapping si nécessaire
    if old_names != new_class_order:
        for file_name in os.listdir(labels_dir):
            if not file_name.endswith('.txt'):
                continue

            label_path = os.path.join(labels_dir, file_name)
            new_lines = []
            file_modified = False

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        old_class = parts[0]
                        if old_class in remap_dict and remap_dict[old_class] is not None:
                            if parts[0] != remap_dict[old_class]:
                                file_modified = True
                            parts[0] = remap_dict[old_class]
                            new_lines.append(' '.join(parts))
                        # Les autres lignes sont ignorées (supprimées)
                    else:
                        # Ligne vide - on la conserve
                        new_lines.append(line.strip())

            # Réécrit le fichier seulement s'il y a eu des modifications
            if file_modified:
                with open(label_path, 'w') as f:
                    for line in new_lines:
                        f.write(line + '\n')
                modified_count += 1
                print(f"✅ Modifié : {file_name}")
    
    # Nettoyage final pour s'assurer qu'il ne reste aucun ID invalide
    valid_class_ids = list(range(len(new_class_order)))
    final_clean_stats = clean_labels_by_class_ids_silent(dataset_dir, valid_class_ids)
    
    if final_clean_stats and final_clean_stats['lines_removed'] > 0:
        print(f"🗑️ Nettoyage final: {final_clean_stats['lines_removed']} lignes supplémentaires supprimées")
        print(f"   📁 {final_clean_stats['files_modified']} fichiers modifiés lors du nettoyage final")
        if final_clean_stats['invalid_ids_found']:
            print(f"   ❌ IDs supprimés: {sorted(final_clean_stats['invalid_ids_found'])}")
    else:
        print("✅ Aucun nettoyage supplémentaire nécessaire")

    # Met à jour le fichier data.yaml
    yaml_content = {
        "nc": len(new_class_order),
        "names": new_class_order
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\n✅ Remapping terminé. {modified_count} fichiers de labels modifiés dans : {dataset_dir}")
    print(f"📄 Fichier data.yaml mis à jour avec {len(new_class_order)} classes")
    
    return f"✅ Dataset modifié en place : {dataset_dir}"


def remove_images_without_labels(dataset_path):
    """
    Supprime les images qui n'ont pas de fichiers de labels correspondants ou qui ont des labels vides dans un dataset YOLO.
    
    Args:
        dataset_path (str): Chemin vers le dataset (doit contenir les dossiers images/ et labels/)
    
    Returns:
        dict: Statistiques de nettoyage (images supprimées, conservées)
    """
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")
    
    # Vérifier que les dossiers existent
    if not os.path.exists(images_dir):
        print(f"❌ Dossier images introuvable: {images_dir}")
        return None
    
    if not os.path.exists(labels_dir):
        print(f"❌ Dossier labels introuvable: {labels_dir}")
        return None
    
    # Lister tous les fichiers images et labels
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
    
    # Créer un set des noms de base des labels valides (non vides)
    valid_label_basenames = set()
    empty_label_files = []
    
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)
        # Vérifier si le fichier label est vide ou ne contient que des espaces
        try:
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if content:  # Le fichier contient du contenu non vide
                    valid_label_basenames.add(os.path.splitext(label_file)[0])
                else:  # Le fichier est vide
                    empty_label_files.append(label_file)
        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture du label {label_file}: {e}")
            empty_label_files.append(label_file)
    
    # Identifier les images sans labels valides
    images_to_remove = []
    images_to_keep = []
    
    for image_file in image_files:
        image_basename = os.path.splitext(image_file)[0]
        if image_basename in valid_label_basenames:
            images_to_keep.append(image_file)
        else:
            images_to_remove.append(image_file)
    
    # Statistiques avant suppression
    stats = {
        'total_images': len(image_files),
        'total_labels': len(label_files),
        'empty_labels': len(empty_label_files),
        'valid_labels': len(valid_label_basenames),
        'images_with_valid_labels': len(images_to_keep),
        'images_without_valid_labels': len(images_to_remove),
        'images_removed': 0,
        'empty_labels_removed': 0,
        'removal_successful': True
    }
    
    print(f"🔍 ANALYSE DU DATASET: {dataset_path}")
    print(f"   📊 Total images: {stats['total_images']}")
    print(f"   📄 Total labels: {stats['total_labels']}")
    print(f"   ✅ Labels valides (non vides): {stats['valid_labels']}")
    print(f"   ❌ Labels vides: {stats['empty_labels']}")
    print(f"   ✅ Images avec labels valides: {stats['images_with_valid_labels']}")
    print(f"   ❌ Images sans labels valides: {stats['images_without_valid_labels']}")
    
    # Supprimer les fichiers de labels vides
    if empty_label_files:
        print(f"\n🗑️ Suppression de {len(empty_label_files)} fichiers de labels vides...")
        for label_file in empty_label_files:
            label_path = os.path.join(labels_dir, label_file)
            try:
                os.remove(label_path)
                stats['empty_labels_removed'] += 1
                print(f"   🗑️ Label vide supprimé: {label_file}")
            except OSError as e:
                print(f"   ❌ Erreur lors de la suppression du label {label_file}: {e}")
                stats['removal_successful'] = False
    
    # Supprimer les images sans labels valides
    if images_to_remove:
        print(f"\n🗑️ Suppression de {len(images_to_remove)} images sans labels valides...")
        
        for image_file in images_to_remove:
            image_path = os.path.join(images_dir, image_file)
            try:
                os.remove(image_path)
                stats['images_removed'] += 1
                print(f"   🗑️ Image supprimée: {image_file}")
            except OSError as e:
                print(f"   ❌ Erreur lors de la suppression de {image_file}: {e}")
                stats['removal_successful'] = False
        
        print(f"\n✅ Nettoyage terminé:")
        print(f"   🗑️ {stats['images_removed']}/{len(images_to_remove)} images supprimées")
        print(f"   🗑️ {stats['empty_labels_removed']}/{len(empty_label_files)} labels vides supprimés")
    else:
        if empty_label_files:
            print(f"\n✅ Nettoyage des labels vides terminé: {stats['empty_labels_removed']} fichiers supprimés")
        print(f"✅ Toutes les images restantes ont des labels valides!")
    
    return stats


def create_data_yaml(dataset_path, class_names, output_path=None):
    """
    Crée un fichier data.yaml pour YOLOv8/YOLO training.
    
    Args:
        dataset_path (str): Chemin vers le dataset final (contenant train/, val/, test/)
        output_path (str, optional): Chemin où créer le data.yaml. Si None, créé dans dataset_path
        class_names (list, optional): Liste des noms de classes. Si None, utilise les classes par défaut
    
    Returns:
        str: Chemin vers le fichier data.yaml créé
    """
    if not os.path.exists(dataset_path):
        print(f"❌ Le chemin du dataset '{dataset_path}' n'existe pas.")
        return None
    if not class_names or not isinstance(class_names, list):
        print("❌ La liste des noms de classes doit être fournie et être une liste non vide.")
        return None
    # Définir le chemin de sortie
    if output_path is None:
        yaml_path = os.path.join(dataset_path, 'data.yaml')
    else:
        yaml_path = output_path
    
    # Créer les chemins relatifs ou absolus
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    test_path = os.path.join(dataset_path, 'test')
    
    # Vérifier que les dossiers existent
    for split_name, split_path in [('train', train_path), ('val', val_path), ('test', test_path)]:
        if not os.path.exists(split_path):
            print(f"⚠️ Attention: Le dossier {split_name} n'existe pas: {split_path}")
    
    # Créer le contenu YAML
    yaml_content = {
        'path': os.path.abspath(dataset_path),  # Chemin absolu vers le dataset
        'train': 'train',  # Chemin relatif par rapport à 'path'
        'val': 'val',      # Chemin relatif par rapport à 'path'
        'test': 'test',    # Chemin relatif par rapport à 'path'
        'nc': len(class_names),  # Nombre de classes
        'names': class_names     # Noms des classes
    }
    
    # Écrire le fichier YAML
    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Fichier data.yaml créé avec succès: {yaml_path}")
        print(f"📊 Configuration:")
        print(f"   - Nombre de classes: {len(class_names)}")
        print(f"   - Classes: {', '.join(class_names)}")
        print(f"   - Chemin du dataset: {yaml_content['path']}")
        
        return yaml_path
        
    except Exception as e:
        print(f"❌ Erreur lors de la création du fichier data.yaml: {e}")
        return None

