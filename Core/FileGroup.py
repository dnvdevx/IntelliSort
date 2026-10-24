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

from file_node import FileNode, extract_text, SIZE_TINY, SIZE_SMALL, SIZE_MEDIUM, SUPPORTED_TEXT_EXTENSIONS

NAME_SIMILARITY_THRESHOLD = 0.80
COMBINED_ALPHA = 0.35
COMBINED_BETA = 0.65
assert abs(COMBINED_ALPHA + COMBINED_BETA - 1.0) < 1e-9, (f"COMBINED_ALPHA + COMBINED_BETA must equal 1.0, got {COMBINED_ALPHA + COMBINED_BETA}")
CHUNK_TOKENS = 250
MAX_CHUNKS_PER_FILE = 16
KMEANS_N_INIT = 100
CLIP_CHUNK_WORDS = 35
LINK_Z = 2.0

class FileGrouper:
    def __init__(self):
        self.group = dict()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._clip_model = None
        self._clip_processor = None
        self.unprocessed = []
        self.cross_modal = False
        self.link_modalities = False
        self._texts = {}
        self._clip_text_cache = {}

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
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
            pooled = vision_outputs.pooler_output
            features = model.visual_projection(pooled)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return features.squeeze(0).cpu().numpy()

    def _encode_texts_clip(self, texts):
        model, processor = self._get_clip()
        tok = processor.tokenizer
        out = []
        with torch.no_grad():
            for t in texts:
                words = t.split()
                chunks = [" ".join(words[i:i + CLIP_CHUNK_WORDS]) for i in range(0, len(words), CLIP_CHUNK_WORDS)]
                chunks = chunks[:MAX_CHUNKS_PER_FILE] or [""]
                enc = tok(chunks, padding=True, truncation=True, max_length=77, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                o = model.text_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], return_dict=True)
                f = model.text_projection(o.pooler_output)
                f = torch.nn.functional.normalize(f, p=2, dim=-1).mean(dim=0)
                f = torch.nn.functional.normalize(f, p=2, dim=-1)
                out.append(f.cpu().numpy())
        return np.array(out, dtype=np.float32)

    def _center(self, rows):
        m = np.array(rows, dtype=np.float32)
        m = m - m.mean(axis=0, keepdims=True)
        m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
        return list(m)

    def _clip_text_vectors(self, nodes):
        missing = [n for n in nodes if n not in self._clip_text_cache]
        if missing:
            vecs = self._encode_texts_clip([self._texts[n] for n in missing])
            for n, v in zip(missing, vecs):
                self._clip_text_cache[n] = v
        return np.array([self._clip_text_cache[n] for n in nodes], dtype=np.float32)

    def _link_clusters(self, t_lab, i_lab, t_clip, i_vec, z=LINK_Z):
        self.last_link_scores = []
        t_ids = sorted(set(t_lab.tolist()))
        i_ids = sorted(set(i_lab.tolist()))
        if not t_ids or not i_ids:
            return []
        cos = t_clip @ i_vec.T
        mu = cos.mean()
        sigma = cos.std()
        if sigma == 0:
            return []
        block = np.array([[cos[t_lab == a][:, i_lab == b].mean() for b in i_ids] for a in t_ids])
        score = (block - mu) / sigma
        links = []
        for r in range(len(t_ids)):
            c = int(np.argmax(block[r]))
            if int(np.argmax(block[:, c])) != r:
                continue
            self.last_link_scores.append((t_ids[r], i_ids[c], float(score[r, c])))
            if score[r, c] > z:
                links.append((t_ids[r], i_ids[c]))
        return links

    def _cluster_linked(self, embeddings_np, ordered_nodes, seed=42, z=LINK_Z):
        is_img = np.array([self._is_image(n) for n in ordered_nodes])
        t_idx = np.where(~is_img)[0]
        i_idx = np.where(is_img)[0]

        def run_kmeans(X):
            k = max(1, min(len(X) // 5, 500))
            if k < 2:
                return np.zeros(len(X), dtype=int)
            return KMeans(n_clusters=k, random_state=seed, n_init=KMEANS_N_INIT).fit_predict(X)

        t_lab = run_kmeans(embeddings_np[t_idx])
        i_lab = run_kmeans(embeddings_np[i_idx]) + t_lab.max() + 1
        labels = np.zeros(len(ordered_nodes), dtype=int)
        labels[t_idx] = t_lab
        labels[i_idx] = i_lab

        t_clip = self._clip_text_vectors([ordered_nodes[i] for i in t_idx])
        links = self._link_clusters(t_lab, i_lab, t_clip, embeddings_np[i_idx], z)
        for a, b in links:
            labels[labels == b] = a
        return labels, links

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

    def _encode_texts(self, texts):
        model = self._get_model()
        tokenizer = model.tokenizer
        window = max(8, min(CHUNK_TOKENS, getattr(model, "max_seq_length", 256) - 2))

        chunks, owner, weight = [], [], []
        for i, text in enumerate(texts):
            ids = tokenizer(text, add_special_tokens=False, truncation=False,
                            verbose=False)["input_ids"]
            if len(ids) <= window:
                chunks.append(text)
                owner.append(i)
                weight.append(max(len(ids), 1))
                continue
            for start in range(0, min(len(ids), window * MAX_CHUNKS_PER_FILE), window):
                piece = ids[start:start + window]
                chunks.append(tokenizer.decode(piece))
                owner.append(i)
                weight.append(len(piece))

        emb = np.asarray(model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device,
            show_progress_bar=True,
        ), dtype=np.float32)

        out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
        for vec, o, w in zip(emb, owner, weight):
            out[o] += w * vec
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    def _build_embeddings(self, nodes):

        TARGET_DIM = 512
        unprocessed = []
 
        image_nodes = [n for n in nodes if self._is_image(n)]
        text_nodes  = [n for n in nodes if not self._is_image(n)]
        use_clip_text = self.cross_modal and len(image_nodes) > 0
 
        text_embeddings_np = {}
        if text_nodes:
            usable, texts = [], []
            for node in text_nodes:
                if node.extension not in SUPPORTED_TEXT_EXTENSIONS:
                    unprocessed.append((node, f"unsupported file type ({node.extension or 'no extension'})"))
                    continue
                try:
                    text = extract_text(node.path)
                except Exception as e:
                    unprocessed.append((node, f"text extraction failed: {e}"))
                    continue
                if not text.strip():
                    unprocessed.append((node, "no extractable text"))
                    continue
                usable.append(node)
                texts.append(text)
                self._texts[node] = text
 
            if usable:
                try:
                    if use_clip_text:
                        raw = self._encode_texts_clip(texts)
                        for node, vec in zip(usable, raw):
                            text_embeddings_np[node] = vec
                    else:
                        raw = self._encode_texts(texts)
                        for node, vec in zip(usable, raw):
                            padded = np.zeros(TARGET_DIM, dtype=np.float32)
                            padded[:len(vec)] = vec
                            text_embeddings_np[node] = padded
                except Exception as e:
                    print(f"Error building text embeddings: {e}")
                    for node in usable:
                        unprocessed.append((node, f"text embedding failed: {e}"))
 
        image_embeddings_np = {}
        for node in image_nodes:
            try:
                vec = self._embed_image(node.path)
                image_embeddings_np[node] = vec.astype(np.float32)
            except Exception as e:
                print(f"Could not embed image {node.filename}: {e}")
                unprocessed.append((node, f"image embedding failed: {e}"))

        if use_clip_text and text_embeddings_np and image_embeddings_np:
            tk = list(text_embeddings_np.keys())
            for node, vec in zip(tk, self._center([text_embeddings_np[n] for n in tk])):
                text_embeddings_np[node] = vec
            ik = list(image_embeddings_np.keys())
            for node, vec in zip(ik, self._center([image_embeddings_np[n] for n in ik])):
                image_embeddings_np[node] = vec
 
        ordered_nodes = []
        all_embeddings = []
        for node in text_nodes:
            if node in text_embeddings_np:
                ordered_nodes.append(node)
                all_embeddings.append(text_embeddings_np[node])
        for node in image_nodes:
            if node in image_embeddings_np:
                ordered_nodes.append(node)
                all_embeddings.append(image_embeddings_np[node])
 
        if not ordered_nodes:
            return None, [], unprocessed
 
        return np.array(all_embeddings, dtype=np.float32), ordered_nodes, unprocessed
 
    def group_by_combined(self, file_nodes):
        nodes = sorted(file_nodes, key=lambda n: n.path)

        embeddings_np, ordered_nodes, unprocessed = self._build_embeddings(nodes)
        self.unprocessed = unprocessed

        if embeddings_np is None or len(ordered_nodes) < 2:
            self.group['combined'] = {"ungrouped": set(nodes)}
            return

        all_blank = np.all(np.abs(embeddings_np) < 1e-6)
        if all_blank:
            print("Warning: no text or image content could be extracted. Grouping by name only.")
            self.group_by_name(file_nodes)
            self.group['combined'] = self.group.get('name', {"ungrouped": set(nodes)})
            return

        has_text = any(not self._is_image(n) for n in ordered_nodes)
        has_img = any(self._is_image(n) for n in ordered_nodes)
        use_links = self.link_modalities and not self.cross_modal and has_text and has_img

        n_clusters = max(2, min(len(ordered_nodes) // 5, 500))
        kmeans = None
        try:
            if use_links:
                labels, links = self._cluster_linked(embeddings_np, ordered_nodes)
                print(f"Linked {len(links)} text group(s) with image group(s)")
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=KMEANS_N_INIT)
                labels = kmeans.fit_predict(embeddings_np)
        except Exception as e:
            print(f"Clustering failed: {e}")
            labels = [0] * len(ordered_nodes)
            kmeans = None

        raw_groups = defaultdict(list)
        for node, label in zip(ordered_nodes, labels):
            raw_groups[label].append(node)

        final_groups = {}
        for label, group_nodes in raw_groups.items():
            cluster_indices = [ordered_nodes.index(n) for n in group_nodes]
            cluster_embeddings = embeddings_np[cluster_indices]
            center = kmeans.cluster_centers_[label] if kmeans is not None else cluster_embeddings.mean(axis=0)
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
