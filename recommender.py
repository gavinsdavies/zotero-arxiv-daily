from asyncio.log import logger
import numpy as np
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
from datetime import datetime
from keyword_filter import calculate_keyword_bonus

def rerank_paper(candidate:list[ArxivPaper],corpus:list[dict],model:str='avsolatorio/GIST-small-Embedding-v0') -> list[ArxivPaper]:
    encoder = SentenceTransformer(model)
    #sort corpus by date, from newest to oldest
    corpus = sorted(corpus,key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),reverse=True)
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    corpus_feature = encoder.encode([paper['data']['abstractNote'] for paper in corpus])
    candidate_feature = encoder.encode([paper.summary for paper in candidate])
    sim = encoder.similarity(candidate_feature,corpus_feature) # [n_candidate, n_corpus]
    scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]
    
    # Add keyword bonus to scores
    for s, c in zip(scores, candidate):
        keyword_bonus = calculate_keyword_bonus(c.title, c.summary)
        c.score = s.item() + keyword_bonus
        if keyword_bonus > 0:
            logger.debug(f"Added keyword bonus {keyword_bonus:.2f} to {c.title[:50]}")
    
    candidate = sorted(candidate,key=lambda x: x.score,reverse=True)
    return candidate
