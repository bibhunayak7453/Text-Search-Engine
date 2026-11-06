"""
TEXT SEARCH ENGINE
Author: Bibhu Ranjan Nayak (ID: B124032, Branch: CSE-A)
Repository: github.com/bibhunayak7453/Text-Search-Engine
"""

import re
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import defaultdict

try:
    from nltk.stem import PorterStemmer
    HAS_NLTK = True
    porter = PorterStemmer()
except ImportError:
    HAS_NLTK = False

# --- LINGUISTIC TOOLKIT ---

STOP = {"the", "and", "of", "to", "a", "in", "is", "that", "it", "with", "as", "for", "was", "on", "are", "by"}

def stem(word):
    """Stems a word using NLTK Porter Stemmer or a fallback."""
    word = word.lower()
    if HAS_NLTK:
        return porter.stem(word)
    # Simple fallback suffix stripping
    if word.endswith('ing') and len(word) > 4: return word[:-3]
    if word.endswith('ed') and len(word) > 3: return word[:-2]
    if word.endswith('s') and not word.endswith('ss') and len(word) > 2: return word[:-1]
    return word

def soundex(word):
    """Computes the classic American Soundex code."""
    if not word: return ""
    word = word.upper()
    mapping = {"B":"1","F":"1","P":"1","V":"1", "C":"2","G":"2","J":"2","K":"2","Q":"2","S":"2","X":"2","Z":"2",
               "D":"3","T":"3", "L":"4", "M":"5","N":"5", "R":"6"}
    res = [word[0]]
    for char in word[1:]:
        if char in mapping:
            code = mapping[char]
            if code != res[-1]: res.append(code)
    res = [c for c in res if c.isalpha() or c.isdigit()]
    res = "".join(res).replace(res[0], "", 1) if len(res) > 1 and res[1] == mapping.get(word[0]) else "".join(res)
    return (res[0] if res else word[0]) + ("".join(c for c in res[1:] if c.isdigit()) + "000")[:3]

def edit_distance(s1, s2):
    """Standard dynamic-programming Levenshtein distance."""
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    for i in range(len(s1) + 1): dp[i][0] = i
    for j in range(len(s2) + 1): dp[0][j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[len(s1)][len(s2)]

def kgrams(word, k=2):
    """Generates k-grams for a word (padded with $)."""
    padded = "$" + word + "$"
    return set(padded[i:i+k] for i in range(len(padded) - k + 1))

def tokenize(text):
    """Regex-based tokenizer yielding (token, position, start, end)."""
    for pos, match in enumerate(re.finditer(r"[A-Za-z0-9]+(?:'[a-z]+)?", text)):
        yield match.group(0), pos, match.start(), match.end()

# --- SEARCH ENGINE BACKEND ---

class SearchEngine:
    def __init__(self):
        self.doc_texts = []
        self.doc_lengths = []
        
        # Indexes
        self.pos_index = defaultdict(lambda: defaultdict(list)) # stem -> {doc_id -> [positions]}
        self.surface_map = {} # original -> stem
        self.kgram_index = defaultdict(set) # bigram -> {vocab_words}
        self.soundex_index = defaultdict(set) # soundex -> {vocab_words}
        self.vocab_freq = defaultdict(int) # original_word -> count

        self.notes = []

    def build(self, full_text):
        """Builds the four indexes in a single pass."""
        self.__init__()
        
        # Document Segmentation
        docs = re.split(r'\n\s*\n', full_text.strip())
        if len(docs) <= 1 and '\n' in full_text:
            docs = [d for d in full_text.split('\n') if d.strip()]
        if not docs: docs = [full_text]
        
        self.doc_texts = docs
        
        for doc_id, text in enumerate(docs):
            doc_len = 0
            for token, pos, start, end in tokenize(text):
                doc_len += 1
                token_lower = token.lower()
                self.vocab_freq[token_lower] += 1
                
                if token_lower in STOP:
                    continue
                
                stemmed = stem(token_lower)
                self.pos_index[stemmed][doc_id].append(pos)
                self.surface_map[token_lower] = stemmed
                
                for gram in kgrams(token_lower):
                    self.kgram_index[gram].add(token_lower)
                    
                self.soundex_index[soundex(token_lower)].add(token_lower)
            self.doc_lengths.append(doc_len)

    def search(self, query_str):
        """Parses and executes the query."""
        if not self.doc_texts: return [], []
        self.notes = []
        self.tokens = self._tokenize_query(query_str)
        self.current_tok = 0
        
        try:
            result_docs = self.p_or()
        except Exception:
            return [], [] # Parsing error fallback
            
        ranked = self._rank(result_docs, self._extract_stems(query_str))
        return ranked, self.notes

    def _tokenize_query(self, query_str):
        # Extract quoted phrases and normal tokens
        tokens = []
        for match in re.finditer(r'("[^"]+"|\(|\)|AND|OR|NOT|[^\s()]+)', query_str):
            tokens.append(match.group(0))
        return tokens

    # -- Recursive Descent Parser --
    def _match(self, *expected):
        if self.current_tok < len(self.tokens) and self.tokens[self.current_tok] in expected:
            self.current_tok += 1
            return True
        return False

    def p_or(self):
        docs = self.p_and()
        while self._match('OR'):
            docs = docs.union(self.p_and())
        return docs

    def p_and(self):
        docs = self.p_not()
        while self._match('AND') or (self.current_tok < len(self.tokens) and self.tokens[self.current_tok] not in ('OR', ')')):
            # Implicit AND for space-separated terms if explicit AND is missing
            docs = docs.intersection(self.p_not())
        return docs

    def p_not(self):
        if self._match('NOT'):
            all_docs = set(range(len(self.doc_texts)))
            return all_docs - self.p_atom()
        return self.p_atom()

    def p_atom(self):
        if self.current_tok >= len(self.tokens): return set()
        tok = self.tokens[self.current_tok]
        
        if tok == '(':
            self.current_tok += 1
            docs = self.p_or()
            self._match(')')
            return docs
            
        self.current_tok += 1
        
        if tok.startswith('"') and tok.endswith('"'):
            return self.phrase(tok[1:-1])
        return self.term(tok)

    # -- Resolvers --
    def term(self, word):
        word_lower = word.lower()
        if word_lower in STOP: return set(range(len(self.doc_texts))) # Ignore stop words logically
        
        if word_lower.startswith('soundex:'):
            target = word_lower.split(':')[1]
            code = soundex(target)
            matches = self.soundex_index.get(code, set())
            self.notes.append(f"Soundex {code} for '{target}': {', '.join(matches) if matches else 'None'}")
            return self._union_stems(matches)
            
        if '*' in word_lower or '?' in word_lower:
            pattern = re.compile(word_lower.replace('*', '.*').replace('?', '.') + '$')
            # K-gram filter optimization omitted for brevity, scanning vocab
            matches = [w for w in self.vocab_freq if pattern.match(w)]
            self.notes.append(f"Wildcard '{word_lower}': {', '.join(matches)}")
            return self._union_stems(matches)

        stemmed = stem(word_lower)
        if stemmed in self.pos_index:
            return set(self.pos_index[stemmed].keys())
            
        # Automatic Spell Correction
        best_candidate = self._spell_correct(word_lower)
        if best_candidate:
            self.notes.append(f"Spell-corrected '{word_lower}' -> '{best_candidate}'")
            return set(self.pos_index[stem(best_candidate)].keys())
            
        return set()

    def phrase(self, phrase_str):
        phrase_stems = [stem(w.lower()) for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", phrase_str.lower()) if w.lower() not in STOP]
        if not phrase_stems: return set()
        
        # Intersect docs
        docs = set(self.pos_index.get(phrase_stems[0], {}).keys())
        for s in phrase_stems[1:]:
            docs.intersection_update(self.pos_index.get(s, {}).keys())
            
        result = set()
        for doc in docs:
            # Check consecutive positions
            positions = [self.pos_index[s][doc] for s in phrase_stems]
            for p0 in positions[0]:
                if all((p0 + i) in positions[i] for i in range(1, len(positions))):
                    result.add(doc)
                    break
        return result

    # -- Helpers --
    def _union_stems(self, words):
        docs = set()
        for w in words: docs.update(self.pos_index.get(stem(w), {}).keys())
        return docs

    def _spell_correct(self, word):
        q_grams = kgrams(word)
        candidates = defaultdict(int)
        for gram in q_grams:
            for w in self.kgram_index.get(gram, []):
                candidates[w] += 1
                
        valid_cands = []
        for w, overlap in candidates.items():
            jaccard = overlap / (len(q_grams) + len(kgrams(w)) - overlap)
            if jaccard >= 0.2:
                ed = edit_distance(word, w)
                if ed <= 2:
                    valid_cands.append((ed, -self.vocab_freq[w], w))
                    
        if valid_cands:
            valid_cands.sort()
            return valid_cands[0][2]
        return None
        
    def _extract_stems(self, query_str):
        return [stem(w.lower()) for w in re.findall(r"[A-Za-z0-9]+", query_str) if w.lower() not in STOP and w.upper() not in ('AND', 'OR', 'NOT')]

    def _rank(self, docs, query_stems):
        N = len(self.doc_texts)
        scores = []
        for doc in docs:
            score = 0
            for s in query_stems:
                tf = len(self.pos_index.get(s, {}).get(doc, []))
                if tf > 0:
                    df = len(self.pos_index.get(s, {}))
                    tf_wt = 1 + math.log10(tf)
                    idf = math.log10(1 + (N / df))
                    score += tf_wt * idf
            scores.append((score, doc))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [(d, s) for s, d in scores][:50] # 50-result display limit


# --- GUI (TKINTER) ---

class App:
    def __init__(self, root):
        self.engine = SearchEngine()
        self.root = root
        self.root.title("Text Search Engine")
        self.root.geometry("800x600")
        
        style = ttk.Style()
        style.theme_use('clam')
        
        # Top Frame
        top_frame = ttk.Frame(root, padding=10)
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="Text Search Engine", font=("Helvetica", 18, "bold")).pack()
        ttk.Label(top_frame, text="Tokenization · Stemming · Inverted Index · Wildcard · Soundex · Spell Correction", foreground="gray").pack()
        
        ttk.Button(top_frame, text="Load .txt File", command=self.load_file).pack(pady=5)
        self.status_lbl = ttk.Label(top_frame, text="No file loaded yet", foreground="green")
        self.status_lbl.pack()
        
        # Box 1: Query Area
        box1 = ttk.LabelFrame(root, text="BOX 1 • QUERY", padding=10)
        box1.pack(fill=tk.X, padx=10, pady=5)
        
        search_frame = ttk.Frame(box1)
        search_frame.pack(fill=tk.X)
        
        self.query_var = tk.StringVar()
        self.entry = ttk.Entry(search_frame, textvariable=self.query_var, font=("Helvetica", 12))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,10))
        self.entry.bind("<Return>", lambda e: self.search())
        
        ttk.Button(search_frame, text="Search", command=self.search).pack(side=tk.LEFT, padx=2)
        ttk.Button(search_frame, text="Clear", command=self.clear).pack(side=tk.LEFT)
        
        chip_frame = ttk.Frame(box1)
        chip_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Label(chip_frame, text="Try: ").pack(side=tk.LEFT)
        chips = [
            ("Boolean", "football AND NOT tennis"),
            ("Phrase", '"data science"'),
            ("Wildcard", "basket*"),
            ("Soundex", "soundex:ronaldo"),
            ("Spell check", "tournemant")
        ]
        for name, ex in chips:
            btn = ttk.Button(chip_frame, text=name, command=lambda e=ex: self.query_var.set(e) or self.search())
            btn.pack(side=tk.LEFT, padx=2)
            
        # Box 2: Results Area
        box2 = ttk.LabelFrame(root, text="BOX 2 • RESULTS", padding=10)
        box2.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.text_area = tk.Text(box2, wrap=tk.WORD, font=("Consolas", 11))
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.text_area.tag_configure("highlight", background="yellow")
        self.text_area.tag_configure("note", foreground="green")
        self.text_area.tag_configure("header", font=("Consolas", 11, "bold"))
        
        scrollbar = ttk.Scrollbar(box2, command=self.text_area.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.config(yscrollcommand=scrollbar.set)
        self.text_area.insert(tk.END, "Load a text file, then type a query above.")
        self.text_area.config(state=tk.DISABLED)

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if not filepath: return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            self.engine.build(content)
            self.status_lbl.config(text=f"{filepath.split('/')[-1]} — {len(self.engine.doc_texts)} documents, {len(self.engine.vocab_freq)} unique terms indexed")
            self.clear()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def search(self):
        q = self.query_var.get().strip()
        if not q or not self.engine.doc_texts: return
        
        results, notes = self.engine.search(q)
        
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        
        self.text_area.insert(tk.END, f"Query:  {q}\n")
        for note in notes:
            self.text_area.insert(tk.END, f"• {note}\n", "note")
            
        self.text_area.insert(tk.END, f"\n{len(results)} document(s) found\n", "header")
        self.text_area.insert(tk.END, "-"*50 + "\n\n")
        
        # Stems to highlight
        query_stems = set(self.engine._extract_stems(q))
        
        for doc_id, score in results:
            self.text_area.insert(tk.END, f"Document {doc_id + 1}   ", "header")
            self.text_area.insert(tk.END, f"relevance {score:.3f}\n")
            
            raw_text = self.engine.doc_texts[doc_id]
            # Highlighting logic based on tokens and character offsets
            last_idx = 0
            for token, pos, start, end in tokenize(raw_text):
                self.text_area.insert(tk.END, raw_text[last_idx:start])
                t_lower = token.lower()
                is_match = False
                
                # Check wildcard/soundex resolved notes vs normal stems
                if stem(t_lower) in query_stems:
                    is_match = True
                else:
                    # Also highlight if it matched via wildcard/soundex resolution
                    for word in self.engine.vocab_freq:
                        if t_lower == word and stem(word) in self.engine._union_stems(self.engine.notes):
                            is_match = True
                            
                if is_match:
                    self.text_area.insert(tk.END, raw_text[start:end], "highlight")
                else:
                    self.text_area.insert(tk.END, raw_text[start:end])
                last_idx = end
            self.text_area.insert(tk.END, raw_text[last_idx:] + "\n\n")
            
        self.text_area.config(state=tk.DISABLED)

    def clear(self):
        self.query_var.set("")
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        self.text_area.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()