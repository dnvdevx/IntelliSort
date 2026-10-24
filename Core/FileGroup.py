import re
from difflib import SequenceMatcher
from collections import defaultdict
from datetime import datetime, timedelta

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from file_node import FileNode, extract_text, SIZE_TINY, SIZE_SMALL, SIZE_MEDIUM

NAME_SIMILARITY_THRESHOLD = 0.80
COMBINED_ALPHA = 0.35
COMBINED_BETA = 0.65
assert abs(COMBINED_ALPHA + COMBINED_BETA - 1.0) < 1e-9, (
    f"COMBINED_ALPHA + COMBINED_BETA must equal 1.0, got {COMBINED_ALPHA + COMBINED_BETA}"
)


class FileGrouper:
    def __init__(self):
        self.group = dict()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._clip_model = None
        self._clip_processor = None

    def _get_model(self):
        if self.model is None:
            import os, logging
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            print(f"Loading text model on {self.device}...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        return self.model

    def _get_clip(self):
        if self._clip_model is None:
            import os, logging, warnings
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
            print(f"Loading image model on {self.device}...")
            from transformers import CLIPModel, CLIPProcessor
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return self._clip_model, self._clip_processor

    def _embed_image(self, path):
        from PIL import Image as PILImage
        model, processor = self._get_clip()
        img = PILImage.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            vision_outputs = model.vision_model(**inputs, output_attentions=False, output_hidden_states=False, return_dict=True)
            # last_hidden_state[:, 0] is the CLS token, same as pooler_output in older versions
            cls_token = vision_outputs.last_hidden_state[:, 0, :]
            features = model.visual_projection(cls_token)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return features.squeeze(0).cpu().numpy()

    def _is_image(self, node):
        return node.extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']

    def group_extensions(self, file_nodes):
        extension_groups = defaultdict(set)
        for node in file_nodes:
            ext = node.extension[1:] if node.extension.startswith('.') else node.extension
            ext = ext if ext else "no_extension"
            extension_groups[ext].add(node)
        self.group['extensions'] = dict(extension_groups)

    def group_by_size(self, file_nodes):
        size_groups = defaultdict(set)
        for node in file_nodes:
            size_groups[node.size_category()].add(node)
        self.group['size'] = dict(size_groups)

    def group_by_date(self, file_nodes):
        date_groups = defaultdict(set)
        now = datetime.now()
        for node in file_nodes:
            if node.last_modified is None:
                date_groups['unknown'].add(node)
            elif node.last_modified > now - timedelta(days=7):
                date_groups['last_week'].add(node)
            elif node.last_modified > now - timedelta(days=30):
                date_groups['last_month'].add(node)
            elif node.last_modified > now - timedelta(days=365):
                date_groups['last_year'].add(node)
            else:
                date_groups['more_than_a_year'].add(node)
        self.group['date'] = dict(date_groups)

    def clean(self, name):
        name = name.lower()
        name = re.sub(r'\d+', '', name)
        name = re.sub(r'[_\-\(\)\s]+', '', name)
        return name

    def similar(self, a, b):
        return SequenceMatcher(None, a, b).ratio()

    def content_similarity(self, node1, node2, embeddings=None, nodes=None):
        if embeddings is not None and nodes is not None:
            idx1 = nodes.index(node1)
            idx2 = nodes.index(node2)
            emb1 = embeddings[idx1].unsqueeze(0)
            emb2 = embeddings[idx2].unsqueeze(0)
            return torch.nn.functional.cosine_similarity(emb1, emb2).item()

        try:
            texts = [extract_text(node1.path), extract_text(node2.path)]
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception as e:
            print(f"Error calculating content similarity: {e}")
            return 0

    def combined_similarity(self, node1, node2, embeddings=None, nodes=None):
        name_sim = self.similar(self.clean(node1.name), self.clean(node2.name))
        cont_sim = self.content_similarity(node1, node2, embeddings=embeddings, nodes=nodes)
        return COMBINED_ALPHA * name_sim + COMBINED_BETA * cont_sim

    def group_by_name(self, file_nodes):
        groups = {}
        for node in file_nodes:
            cleaned_name = self.clean(node.name)
            matched_key = None
            for key in groups:
                if self.similar(cleaned_name, key) >= NAME_SIMILARITY_THRESHOLD:
                    matched_key = key
                    break
            if matched_key is None:
                groups[cleaned_name] = {node}
            else:
                groups[matched_key].add(node)
        self.group['name'] = groups

    def _build_embeddings(self, nodes):
        TARGET_DIM = 512

        image_nodes = [n for n in nodes if self._is_image(n)]
        text_nodes  = [n for n in nodes if not self._is_image(n)]

        text_embeddings_np = {}
        if text_nodes:
            texts = []
            for node in text_nodes:
                try:
                    text = extract_text(node.path)
                    texts.append(text if text.strip() else " ")
                except Exception as e:
                    print(f"Error extracting text from {node.filename}: {e}")
                    texts.append(" ")
            try:
                raw = self._get_model().encode(
                    texts,
                    convert_to_tensor=False,
                    device=self.device,
                    show_progress_bar=True,
                )
                for node, vec in zip(text_nodes, raw):
                    padded = np.zeros(TARGET_DIM, dtype=np.float32)
                    padded[:len(vec)] = vec
                    text_embeddings_np[node] = padded
            except Exception as e:
                print(f"Error building text embeddings: {e}")

        image_embeddings_np = {}
        for node in image_nodes:
            try:
                vec = self._embed_image(node.path)
                image_embeddings_np[node] = vec.astype(np.float32)
            except Exception as e:
                print(f"Could not embed image {node.filename}: {e}")

        ordered_nodes = []
        all_embeddings = []

        for node in text_nodes:
            ordered_nodes.append(node)
            all_embeddings.append(
                text_embeddings_np.get(node, np.zeros(TARGET_DIM, dtype=np.float32))
            )

        for node in image_nodes:
            ordered_nodes.append(node)
            all_embeddings.append(
                image_embeddings_np.get(node, np.zeros(TARGET_DIM, dtype=np.float32))
            )

        if not ordered_nodes:
            return None, nodes

        return np.array(all_embeddings, dtype=np.float32), ordered_nodes

    def group_by_combined(self, file_nodes):
        nodes = list(file_nodes)

        embeddings_np, ordered_nodes = self._build_embeddings(nodes)

        if embeddings_np is None or len(ordered_nodes) < 2:
            self.group['combined'] = {"ungrouped": set(nodes)}
            return

        all_blank = np.all(np.abs(embeddings_np) < 1e-6)
        if all_blank:
            print("Warning: no text or image content could be extracted. Grouping by name only.")
            self.group_by_name(file_nodes)
            self.group['combined'] = self.group.get('name', {"ungrouped": set(nodes)})
            return

        n_clusters = max(2, min(len(ordered_nodes) // 5, 500))
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(embeddings_np)
        except Exception as e:
            print(f"KMeans clustering failed: {e}")
            labels = [0] * len(ordered_nodes)

        raw_groups = defaultdict(list)
        for node, label in zip(ordered_nodes, labels):
            raw_groups[label].append(node)

        final_groups = {}
        for label, group_nodes in raw_groups.items():
            cluster_indices = [ordered_nodes.index(n) for n in group_nodes]
            cluster_embeddings = embeddings_np[cluster_indices]
            center = kmeans.cluster_centers_[label]
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            representative = ordered_nodes[closest_idx]
            base_name = self.clean(representative.name) or representative.filename
            group_name = base_name
            suffix = 1
            while group_name in final_groups:
                suffix += 1
                group_name = f"{base_name}_{suffix}"
            final_groups[group_name] = set(group_nodes)

        self.group['combined'] = final_groups

    def group_by_reference_file(self, ref_path, file_nodes, threshold=0.5):
        nodes = list(file_nodes)

        try:
            ref_node = FileNode(ref_path)
            if self._is_image(ref_node):
                try:
                    ref_embedding = self._embed_image(ref_path)
                    ref_is_image = True
                except Exception as e:
                    print(f"Could not embed reference image: {e}")
                    return []
            else:
                ref_text = extract_text(ref_path)
                if not ref_text.strip():
                    ref_text = ref_node.name.replace("-", " ").replace("_", " ")
                ref_is_image = False
        except Exception as e:
            print(f"Error reading reference file: {e}")
            return []

        results = []

        if self._is_image(ref_node):
            for node in nodes:
                if not self._is_image(node):
                    continue
                try:
                    candidate_emb = self._embed_image(node.path)
                    score = float(
                        np.dot(ref_embedding, candidate_emb) /
                        (np.linalg.norm(ref_embedding) * np.linalg.norm(candidate_emb) + 1e-8)
                    )
                    if score >= threshold:
                        results.append((score, node))
                except Exception as e:
                    print(f"Error processing {node.filename}: {e}")
        else:
            text_nodes = [n for n in nodes if not self._is_image(n)]
            if text_nodes:
                candidate_texts = []
                valid_nodes = []
                for node in text_nodes:
                    try:
                        text = extract_text(node.path)
                        candidate_texts.append(text if text.strip() else " ")
                        valid_nodes.append(node)
                    except Exception as e:
                        print(f"Error extracting text from {node.filename}: {e}")

                if candidate_texts:
                    try:
                        all_texts = [ref_text] + candidate_texts
                        embeddings = self._get_model().encode(
                            all_texts,
                            convert_to_tensor=True,
                            device=self.device,
                            show_progress_bar=False,
                        )
                        ref_emb = embeddings[0].unsqueeze(0)
                        for i, node in enumerate(valid_nodes):
                            cand_emb = embeddings[i + 1].unsqueeze(0)
                            score = torch.nn.functional.cosine_similarity(ref_emb, cand_emb).item()
                            if score >= threshold:
                                results.append((score, node))
                    except Exception as e:
                        print(f"Error during batch encoding: {e}")

        results.sort(key=lambda x: x[0], reverse=True)
        return [node for score, node in results]