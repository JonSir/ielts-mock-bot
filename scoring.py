import re
from collections import Counter

STOPWORDS = set("""
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its
itself they them their theirs themselves what which who whom this that these those am is are was were be been being have
has had having do does did doing a an the and but if or because as until while of at by for with about against between
into through during before after above below to from up down in out on off over under again further then once here there
when where why how all any both each few more most other some such no nor not only own same so than too very s t can
will just don don should now
""".split())

FILLERS = {"uh","um","erm","like","you know","i mean","sort of","kind of","well"}

def tokenize(text: str):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return words

def metrics_from_text_and_times(text: str, speech_ms: int, pauses_ms: int):
    words = tokenize(text)
    n_words = len(words)
    minutes = max(1e-6, speech_ms / 60000.0)
    wpm = n_words / minutes

    filler_count = 0
    t = " " + text.lower() + " "
    for f in FILLERS:
        filler_count += t.count(" " + f + " ")
    filler_ratio = filler_count / max(1, n_words)

    total = max(1e-6, speech_ms + pauses_ms)
    pause_rate = pauses_ms / total

    content_words = [w for w in words if w not in STOPWORDS]
    uniq = len(set(content_words))
    ttr = uniq / max(1, len(content_words))

    bigrams = list(zip(words, words[1:]))
    rep = 0.0
    if bigrams:
        counts = Counter(bigrams)
        repeated = sum(c for c in counts.values() if c > 1)
        rep = repeated / len(bigrams)

    sents = re.split(r"[.!?]+", text)
    sents = [s.strip() for s in sents if s.strip()]
    clause_markers = sum(len(re.findall(r"\b(when|while|because|although|though|which|that|who|where|if|unless|until|since|after|before)\b", s.lower())) for s in sents)
    clause_variety = clause_markers / max(1, len(sents))
    verbish = re.compile(r"\b(am|is|are|was|were|be|been|being|do|does|did|have|has|had|go|goes|went|gone|make|makes|made|say|says|said|think|thinks|thought|feel|feels|felt|want|wants|wanted|need|needs|needed|can|could|will|would|shall|should|may|might|must)\b")
    bad = sum(1 for s in sents if not verbish.search(s.lower()))
    grammar_err_rate = bad / max(1, len(sents))

    return {
        "wpm": wpm,
        "filler_ratio": min(1.0, filler_ratio),
        "pause_rate": min(1.0, pause_rate),
        "ttr": min(1.0, ttr),
        "repetition": min(1.0, rep),
        "grammar_err_rate": min(1.0, grammar_err_rate),
        "clause_variety": clause_variety
    }

def clamp(x, lo, hi): return max(lo, min(hi, x))

def bands_from_metrics(m, asr_conf=0.7, rate_stability=0.7):
    fluency = clamp(9 - (m['pause_rate']*3 + m['filler_ratio']*4), 1, 9)
    lexis = clamp(5 + (m['ttr']*6) - (m['repetition']*2), 1, 9)
    grammar = clamp(6 - m['grammar_err_rate']*8 + m['clause_variety']*2, 1, 9)
    pron = clamp(5 + (asr_conf*3) + (rate_stability*2), 1, 9)
    def to_half(x): return round(x*2)/2
    scores = {
        "fluency": to_half(fluency),
        "lexical_resource": to_half(lexis),
        "grammar_range_accuracy": to_half(grammar),
        "pronunciation": to_half(pron)
    }
    overall = to_half(sum(scores.values())/4)
    return scores | {"overall": overall}
