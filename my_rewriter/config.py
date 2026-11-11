import os
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

CACHE_PATH = 'cache'
CASE_RULES_PATH = 'stackoverflow-rewrite-rules-query-optimization.jsonl'

def init_llms(model_type: str = '', load_model=True) -> dict[str, str]:
    fast_llm = None
    if 'oss' in model_type:
        if load_model:
            # (compat) transformers 5.x는 RoPE 설정을 rope_parameters 로 옮겨서
            # Qwen2Config.rope_theta 속성이 사라졌다. gte-Qwen2 커스텀 코드는
            # config.rope_theta 를 직접 접근하므로 shim 으로 되살린다.
            try:
                from transformers.models.qwen2.configuration_qwen2 import Qwen2Config as _Qwen2Config
                if not hasattr(_Qwen2Config, "rope_theta"):
                    def _rope_theta(self):
                        rp = getattr(self, "rope_parameters", None)
                        if isinstance(rp, dict) and "rope_theta" in rp:
                            return rp["rope_theta"]
                        rs = getattr(self, "rope_scaling", None)
                        if isinstance(rs, dict) and "rope_theta" in rs:
                            return rs["rope_theta"]
                        return 1000000.0  # config.json 기본값
                    _Qwen2Config.rope_theta = property(_rope_theta)
            except Exception:
                pass
            Settings.embed_model = HuggingFaceEmbedding(
                model_name='Alibaba-NLP/gte-Qwen2-1.5B-instruct',
                max_length=131072,
                # transformers 5.2 네이티브 Qwen2 구현 사용. 커스텀 원격코드는
                # rope_theta / tokenization_qwen2_fast 등에서 5.x와 충돌하므로 끈다.
                # (gte-Qwen2는 표준 Qwen2 아키텍처라 네이티브 구현으로 동일하게 동작)
                trust_remote_code=False,
                device='cpu', # Ollama가 양쪽 GPU 모두 점유(각 ~8GB) → embedding은 CPU 사용
            )
        embed_dim = 1536
    else:
        if load_model:
            Settings.embed_model = OpenAIEmbedding(
                model="text-embedding-3-small"
            )
        embed_dim = 1536
    
    if 'oss' in model_type:
        if load_model:
            # (override) Slurm에서 GPU마다 별도 ollama serve를 다른 포트로 띄우기 위해
            # api_base / model 을 환경변수로 뺀다. 미설정 시 기존 기본값 유지(하위호환).
            Settings.llm = OpenAILike(
                model=os.environ.get("OLLAMA_MODEL", "gpt-oss:20b"),
                api_key="ollama",
                api_base=os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434/v1"),
                context_window=131072,
                is_chat_model=True,
                timeout=300,  # 20B 모델 복잡 쿼리 처리 시간 고려 (기본 60초 → 5분)
                # Local Ollama is on loopback.  Retrying a timed-out 5-minute
                # generation three more times only queues the same request
                # behind a stalled runner; job-level resume handles failures.
                max_retries=0,
            )
    elif 'deepseek' in model_type:
        if load_model:
            # (Add) DeepSeek 추론 모델 (thinking mode, high effort).
            # OpenAILike.reasoning_effort 생성자 인자는 llama_index 내부적으로
            # self.model in O1_MODELS 일 때만 요청 본문에 실리므로 (OpenAI o-series 전용),
            # deepseek-v4-pro 에는 적용되지 않는다. additional_kwargs 는 항상 무조건
            # 요청 kwargs 에 병합되므로 reasoning_effort/extra_body(thinking)를
            # additional_kwargs 로 직접 주입한다.
            #
            # DEEPSEEK_THINKING 환경변수로 thinking on/off 제어 가능(기본 enabled,
            # 기존 실험과 동일). 직접 실측 결과 thinking 끄면 호출당 ~100초 → ~4초로
            # 단축됨 — 속도가 급하면 실행 전 `export DEEPSEEK_THINKING=disabled`.
            _thinking_enabled = os.environ.get("DEEPSEEK_THINKING", "enabled") != "disabled"
            _additional_kwargs = {
                "extra_body": {"thinking": {"type": "enabled" if _thinking_enabled else "disabled"}},
            }
            if _thinking_enabled:
                _additional_kwargs["reasoning_effort"] = "high"
            Settings.llm = OpenAILike(
                model=os.environ.get("REASONING_MODEL", "deepseek-v4-pro"),
                api_key=os.environ["REASONING_MODEL_API_KEY"],
                api_base=os.environ.get("REASONING_MODEL_URL", "https://api.deepseek.com"),
                context_window=131072,
                is_chat_model=True,
                timeout=600 if _thinking_enabled else 120,
                additional_kwargs=_additional_kwargs,
            )
            # Lightweight pipeline decisions use V4-Flash with thinking
            # explicitly disabled.  Keep credentials/base configurable so a
            # separate compatible endpoint can be supplied without touching
            # the reasoning-model configuration.
            fast_llm = OpenAILike(
                model=os.environ.get('FAST_MODEL', 'deepseek-v4-flash'),
                api_key=os.environ.get(
                    'FAST_MODEL_API_KEY', os.environ['REASONING_MODEL_API_KEY']),
                api_base=os.environ.get(
                    'FAST_MODEL_URL',
                    os.environ.get('REASONING_MODEL_URL', 'https://api.deepseek.com')),
                context_window=131072,
                is_chat_model=True,
                timeout=int(os.environ.get('FAST_MODEL_TIMEOUT', '120')),
                additional_kwargs={
                    # [2026-09-16] thinking 을 환경변수로 뺐다. 기본값 'disabled' 는
                    # 기존과 동일하므로 변수를 설정하지 않으면 동작이 바뀌지 않는다.
                    # gated 경로의 규칙 선택/정렬/힌트 생성은 전부 fast_llm 을 쓰므로
                    # (distribution_guidance.py 의 LLM 호출 7곳), 여기를 못 바꾸면
                    # 추론 모델 ablation 자체가 불가능하다.
                    'extra_body': {'thinking': {
                        'type': os.environ.get('FAST_MODEL_THINKING', 'disabled')}},
                    **({'reasoning_effort':
                        os.environ.get('FAST_MODEL_REASONING_EFFORT', 'high')}
                       if os.environ.get('FAST_MODEL_THINKING') == 'enabled' else {}),
                },
            )
    elif 'gpt3' in model_type:
        if load_model:
            Settings.llm = OpenAI(
                model="gpt-3.5-turbo-0125"
            )
    else:
        if load_model:
            Settings.llm = OpenAI(
                model="gpt-4o"
            )
    if fast_llm is None and load_model:
        fast_llm = Settings.llm
    from my_rewriter.prompts import GEN_CASE_REWRITE_SYS_PROMPT, GEN_CASE_REWRITE_USER_PROMPT, SELECT_CASE_RULE_SYS_PROMPT, SELECT_CASE_RULE_USER_PROMPT, CLUSTER_REWRITE_SYS_PROMPT, CLUSTER_REWRITE_USER_PROMPT, SUMMARIZE_REWRITE_SYS_PROMPT, SUMMARIZE_REWRITE_USER_PROMPT, SELECT_RULES_SYS_PROMPT, SELECT_RULES_USER_PROMPT, ARRANGE_RULE_SETS_SYS_PROMPT, ARRANGE_RULE_SETS_USER_PROMPT, ARRANGE_RULES_SYS_PROMPT, ARRANGE_RULES_USER_PROMPT, REARRANGE_RULES_SYS_PROMPT, REARRANGE_RULES_USER_PROMPT, SELECT_ARRANGE_RULES_SYS_PROMPT, SELECT_ARRANGE_RULES_USER_PROMPT, RAG_SELECT_ARRANGE_RULES_SYS_PROMPT, RAG_SELECT_ARRANGE_RULES_USER_PROMPT
    # (Add) Distribution-Aware 모듈(Module 1) 전용 신규 프롬프트. baseline 키는 위에서 변경하지 않음.
    from my_rewriter.prompts import GEN_DISTRIBUTION_GUIDANCE_SYS_PROMPT, GEN_DISTRIBUTION_GUIDANCE_USER_PROMPT, DIST_SELECT_RULES_SYS_PROMPT, DIST_SELECT_RULES_USER_PROMPT, DIST_ARRANGE_RULE_SETS_SYS_PROMPT, DIST_ARRANGE_RULE_SETS_USER_PROMPT, DIST_ARRANGE_RULES_SYS_PROMPT, DIST_ARRANGE_RULES_USER_PROMPT, DIST_REARRANGE_RULES_SYS_PROMPT, DIST_REARRANGE_RULES_USER_PROMPT
    return {
        'GEN_CASE_REWRITE_SYS_PROMPT': GEN_CASE_REWRITE_SYS_PROMPT,
        'GEN_CASE_REWRITE_USER_PROMPT': GEN_CASE_REWRITE_USER_PROMPT,
        'SELECT_CASE_RULE_SYS_PROMPT': SELECT_CASE_RULE_SYS_PROMPT,
        'SELECT_CASE_RULE_USER_PROMPT': SELECT_CASE_RULE_USER_PROMPT,
        'CLUSTER_REWRITE_SYS_PROMPT': CLUSTER_REWRITE_SYS_PROMPT,
        'CLUSTER_REWRITE_USER_PROMPT': CLUSTER_REWRITE_USER_PROMPT,
        'SUMMARIZE_REWRITE_SYS_PROMPT': SUMMARIZE_REWRITE_SYS_PROMPT,
        'SUMMARIZE_REWRITE_USER_PROMPT': SUMMARIZE_REWRITE_USER_PROMPT,
        'SELECT_RULES_SYS_PROMPT': SELECT_RULES_SYS_PROMPT,
        'SELECT_RULES_USER_PROMPT': SELECT_RULES_USER_PROMPT,
        'ARRANGE_RULE_SETS_SYS_PROMPT': ARRANGE_RULE_SETS_SYS_PROMPT,
        'ARRANGE_RULE_SETS_USER_PROMPT': ARRANGE_RULE_SETS_USER_PROMPT,
        'ARRANGE_RULES_SYS_PROMPT': ARRANGE_RULES_SYS_PROMPT,
        'ARRANGE_RULES_USER_PROMPT': ARRANGE_RULES_USER_PROMPT,
        'REARRANGE_RULES_SYS_PROMPT': REARRANGE_RULES_SYS_PROMPT,
        'REARRANGE_RULES_USER_PROMPT': REARRANGE_RULES_USER_PROMPT,
        'SELECT_ARRANGE_RULES_SYS_PROMPT': SELECT_ARRANGE_RULES_SYS_PROMPT,
        'SELECT_ARRANGE_RULES_USER_PROMPT': SELECT_ARRANGE_RULES_USER_PROMPT,
        'RAG_SELECT_ARRANGE_RULES_SYS_PROMPT': RAG_SELECT_ARRANGE_RULES_SYS_PROMPT,
        'RAG_SELECT_ARRANGE_RULES_USER_PROMPT': RAG_SELECT_ARRANGE_RULES_USER_PROMPT,
        # (Add) Distribution-Aware 모듈(Module 1) 전용 신규 프롬프트
        'GEN_DISTRIBUTION_GUIDANCE_SYS_PROMPT': GEN_DISTRIBUTION_GUIDANCE_SYS_PROMPT,
        'GEN_DISTRIBUTION_GUIDANCE_USER_PROMPT': GEN_DISTRIBUTION_GUIDANCE_USER_PROMPT,
        'DIST_SELECT_RULES_SYS_PROMPT': DIST_SELECT_RULES_SYS_PROMPT,
        'DIST_SELECT_RULES_USER_PROMPT': DIST_SELECT_RULES_USER_PROMPT,
        'DIST_ARRANGE_RULE_SETS_SYS_PROMPT': DIST_ARRANGE_RULE_SETS_SYS_PROMPT,
        'DIST_ARRANGE_RULE_SETS_USER_PROMPT': DIST_ARRANGE_RULE_SETS_USER_PROMPT,
        'DIST_ARRANGE_RULES_SYS_PROMPT': DIST_ARRANGE_RULES_SYS_PROMPT,
        'DIST_ARRANGE_RULES_USER_PROMPT': DIST_ARRANGE_RULES_USER_PROMPT,
        'DIST_REARRANGE_RULES_SYS_PROMPT': DIST_REARRANGE_RULES_SYS_PROMPT,
        'DIST_REARRANGE_RULES_USER_PROMPT': DIST_REARRANGE_RULES_USER_PROMPT,
        'EMBED_DIM': embed_dim,
        'FAST_LLM': fast_llm,
        '_MAX_DOC_CHARS': 8000 if 'gpt3' in model_type else 10 ** 9, # (override) LLM이 gpt 3.5인 경우만 8000자 이상에서 truncate 수행
    }

def init_db_config(database: str) -> dict[str, str]:
    return {
        'host': os.environ.get('PG_HOST', 'localhost'),
        'port': int(os.environ.get('PG_PORT', '5432')),
        'user': os.environ.get('PG_USER', 'seulgit02'),
        'password': os.environ.get('PG_PASSWORD', ''),
        'dbname': database,
        'db': 'postgresql'
    }
