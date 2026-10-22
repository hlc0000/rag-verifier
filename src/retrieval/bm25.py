"""手写 BM25 关键词检索：字符 bigram 处理中文、整词处理英文，零依赖全确定。

不用 jieba 等分词器：分词器本身有不确定性（同词不同切法），且引入重依赖。
中文按字两两切（"保管费"→保管/管费），英文按单词，中英文混合文档通吃。
"""
import math
import re

TOKEN_RE = re.compile(r'[a-z0-9]+|[一-鿿]+')


def tokenize(text: str) -> list[str]:
    """英文/数字取整词，中文串取字符 bigram（单字保留单字）。"""
    out = []
    for m in TOKEN_RE.finditer(text.lower()):
        s = m.group()
        if s.isascii():
            out.append(s)
        elif len(s) == 1:
            out.append(s)
        else:
            out.extend(s[i:i + 2] for i in range(len(s) - 1))
    return out


class BM25:
    """标准 BM25（k1=1.5, b=0.75）。文档规模千级，暴力打分毫秒级。"""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(docs_tokens)
        self.tf = []           # 每文档的词频 dict
        self.dl = []           # 每文档长度
        df: dict[str, int] = {}
        for tokens in docs_tokens:
            freq: dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self.tf.append(freq)
            self.dl.append(len(tokens))
            for t in freq:
                df[t] = df.get(t, 0) + 1
        self.avdl = sum(self.dl) / max(self.n_docs, 1)
        # IDF = ln(1 + (N - n + 0.5) / (n + 0.5))，保证非负
        self.idf = {t: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
                    for t, n in df.items()}

    def score(self, query: str) -> list[float]:
        """对全部文档打分，返回与入库顺序一致的分数列表。"""
        q_tokens = set(tokenize(query))
        scores = []
        for i in range(self.n_docs):
            freq = self.tf[i]
            norm = self.k1 * (1 - self.b + self.b * self.dl[i] / self.avdl)
            s = 0.0
            for t in q_tokens:
                f = freq.get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + norm)
            scores.append(s)
        return scores
