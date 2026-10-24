"""LLM 答案生成：全系统仅两处用 LLM（语义验证 + 这里的最终生成），其余全确定。

OpenAI 兼容端点配置与语义验证层共用同一组环境变量（.env 自动加载）。
"""
import json
import ssl
import urllib.request
import urllib.error

# 复用语义验证层的配置与 .env 加载（导入即触发 _load_dotenv）
from rag_verifier.chunking.semantic_verifier import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, _ssl_context

ANSWER_PROMPT = """你是严谨的文档问答助手。下面是从文档库检出的证据块，每块标注了来源与质量验证标签。

【证据块】
{context}

【问题】
{question}

回答规则：
1. 只能依据证据块内容回答，禁止编造块中没有的信息
2. 标题、引用行、图表/文献标题也是证据的一部分——问题问的数字、名称、结论若出现在标题或引用行里（如 "speeds up ... ten-fold"），即为有效证据，直接使用并标注出处
3. 关键结论后用【块ID】标注出处
4. 标注"引用补全"的块，是为补全其他块的跨块引用而定向取出的定义内容，可正常使用
5. 标注"不完整证据"的块内容可能缺上下文，引用时需说明该不确定性
6. 证据不足时直接说"根据现有证据无法确定"，并说明缺什么
7. 用中文回答"""


def chat(prompt: str, max_tokens: int = 800) -> str:
    """调用 OpenAI 兼容 chat/completions 端点。"""
    if not LLM_API_KEY:
        raise RuntimeError("未配置 LLM_API_KEY / DEEPSEEK_API_KEY")
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        LLM_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120, context=_ssl_context()) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def answer(question: str, context: str) -> str:
    return chat(ANSWER_PROMPT.format(context=context, question=question))
