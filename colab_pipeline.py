"""
Extended Colab pipeline script for Powerball analysis with heavier search and hosted LLM critique (Hugging Face inference).
Run this in Colab. Configure HUGGINGFACE_API_TOKEN in the environment for hosted LLM steps.
"""

# Install guard: the notebook installs requirements; this script assumes deps present.
import os, sys, time, math, random, json, requests, re
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd
import numpy as np
from tqdm import tqdm

# Optional DEAP for GA
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except Exception:
    DEAP_AVAILABLE = False

# Configurable parameters (heavier-search defaults)
RULES = {"main_count":7, "main_min":1, "main_max":35, "powerball_min":1, "powerball_max":20}
DRAW_DATE_TARGET = os.environ.get('DRAW_DATE_TARGET', '2026-09-03')
MONTE_CARLO_POOL = int(os.environ.get('MONTE_CARLO_POOL', 20000))
GA_POPULATION = int(os.environ.get('GA_POPULATION', 300))
GA_GENERATIONS = int(os.environ.get('GA_GENERATIONS', 800))
ANNEAL_ITER = int(os.environ.get('ANNEAL_ITER', 20000))
REPEAT_RUNS = int(os.environ.get('REPEAT_RUNS', 3))

# Hosted LLM models to try (Hugging Face hosted inference)
HF_MODELS = [
    'mistralai/mistral-7b-instruct-v0.1',
    'meta-llama/Llama-2-7b-chat',
    # Fallbacks / smaller models
    'stabilityai/gpt-4o-mini',
]

HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_API_TOKEN')

# Helpers
def extract_numbers(s):
    return list(map(int, re.findall(r'\d+', s)))

# Fetch function with CSV fallback
def fetch_history_or_csv(csv_path='sample_history.csv'):
    # Try ozlotteries scraping first (best-effort). If fails, use provided CSV
    urls = [
        'https://www.ozlotteries.com/powerball/results',
        'https://www.ozlotteries.com/powerball/results/archives'
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=15, headers={'User-Agent':'Mozilla/5.0'})
            if r.status_code!=200: continue
            txt = r.text
            # quick check for "Powerball" in page
            if 'Powerball' in txt:
                # minimal scrape: look for groups of 8 numbers in page text
                nums = extract_numbers(txt)
                # attempt to chunk by 8
                draws = []
                for i in range(0, max(0,len(nums)-7), 8):
                    main = nums[i:i+7]
                    pb = nums[i+7]
                    if len(main)==7:
                        draws.append({'main': ' '.join(f"{n:02d}" for n in main), 'powerball': int(pb)})
                if draws:
                    df = pd.DataFrame(draws)
                    return df
        except Exception:
            continue
    # fallback to CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # normalize mains to list
        df['main'] = df['main'].apply(lambda s: [int(x) for x in re.findall(r'\d+', str(s))])
        df['powerball'] = df['powerball'].astype(int)
        return df
    raise RuntimeError(f"SOURCE ACCESS FAILURE: unable to fetch official draw history from https://www.ozlotteries.com/ and no CSV at {csv_path}")

# Feature builder
def build_features(df):
    history = df.reset_index(drop=True)
    total = len(history)
    main_counts = Counter()
    pb_counts = Counter()
    last_seen_main = {i:-1 for i in range(RULES['main_min'], RULES['main_max']+1)}
    last_seen_pb = {i:-1 for i in range(RULES['powerball_min'], RULES['powerball_max']+1)}
    pair_counts = Counter(); triple_counts = Counter()
    for idx,row in history.iterrows():
        mains = list(map(int,row['main']))
        pb = int(row['powerball'])
        for m in mains:
            main_counts[m]+=1; last_seen_main[m]=idx
        pb_counts[pb]+=1; last_seen_pb[pb]=idx
        for a,b in combinations(sorted(mains),2): pair_counts[(a,b)]+=1
        for a,b,c in combinations(sorted(mains),3): triple_counts[(a,b,c)]+=1
    freq_main = {n: main_counts[n]/max(1,total) for n in range(RULES['main_min'], RULES['main_max']+1)}
    freq_pb = {n: pb_counts[n]/max(1,total) for n in range(RULES['powerball_min'], RULES['powerball_max']+1)}
    recency_main = {n: (total-last_seen_main[n]) if last_seen_main[n]>=0 else total+1 for n in freq_main}
    recency_pb = {n: (total-last_seen_pb[n]) if last_seen_pb[n]>=0 else total+1 for n in freq_pb}
    return {
        'total_draws': total,
        'freq_main': freq_main, 'freq_pb': freq_pb,
        'recency_main': recency_main, 'recency_pb': recency_pb,
        'pair_counts': pair_counts, 'triple_counts': triple_counts
    }

# Scoring function (same as earlier, with normalization helpers)
def score_line(main_list, pb, features, weights=None):
    if weights is None:
        weights = {'freq':1.0,'recency':0.9,'pair':1.2,'balance':0.5,'sum':0.25}
    fm = features['freq_main']; rg = features['recency_main']; pc = features['pair_counts']
    freq_score = sum(fm[n] for n in main_list)
    max_gap = max(rg.values()) if rg else 1
    recency_score = sum(rg[n]/max_gap for n in main_list)
    pair_score = sum(pc[tuple(sorted((a,b)))] for a,b in combinations(main_list,2))
    evens = sum(1 for n in main_list if n%2==0); odds = len(main_list)-evens
    balance_score = 1 - abs(evens-odds)/len(main_list)
    s = sum(main_list); midpoint = (RULES['main_min']+RULES['main_max'])/2 * RULES['main_count']
    sum_score = 1 - abs(s-midpoint)/midpoint
    pb_freq = features['freq_pb'].get(pb,0); max_pb_gap = max(features['recency_pb'].values()) if features['recency_pb'] else 1
    pb_rec = features['recency_pb'].get(pb,0)/max_pb_gap
    pb_score = pb_freq*1.0 + pb_rec*0.6
    total = (weights['freq']*freq_score + weights['recency']*recency_score + weights['pair']*pair_score + weights['balance']*balance_score + weights['sum']*sum_score)
    return float(total + 0.4*pb_score)

# Generators (same families)
def gen_frequency(features, n, seed=0):
    rnd = random.Random(seed); weights = {n: features['freq_main'][n] + 0.3*(features['recency_main'][n]/max(1,max(features['recency_main'].values()))) for n in features['freq_main']}
    lines=set()
    while len(lines)<n:
        pop=list(weights.keys()); w=[max(1e-9,weights[p]) for p in pop]
        chosen = rnd.choices(pop,weights=w,k=RULES['main_count']*3)
        uniq=[]
        for c in chosen:
            if c not in uniq: uniq.append(c)
            if len(uniq)==RULES['main_count']: break
        if len(uniq)<RULES['main_count']: continue
        pb = rnd.choices(list(features['freq_pb'].keys()),weights=[features['freq_pb'][x] for x in features['freq_pb']],k=1)[0]
        lines.add((tuple(sorted(uniq)),int(pb)))
    return [(list(k[0]),k[1]) for k in lines]

def gen_overdue(features,n,seed=1):
    rnd=random.Random(seed); gaps=features['recency_main']; top=[n for n,_ in sorted(gaps.items(),key=lambda x:-x[1])[:24]]
    lines=set()
    while len(lines)<n:
        mains=sorted(rnd.sample(top,k=RULES['main_count']))
        pb=rnd.choice(sorted(features['recency_pb'].keys(),key=lambda x:-features['recency_pb'][x])[:6])
        lines.add((tuple(mains),int(pb)))
    return [(list(k[0]),k[1]) for k in lines]

def gen_pair(features,n,seed=2):
    rnd=random.Random(seed); pc=features['pair_counts']
    if not pc: return gen_frequency(features,n,seed)
    top_pairs=[p for p,_ in sorted(pc.items(),key=lambda x:-x[1])[:200]]
    lines=set()
    while len(lines)<n:
        pair=rnd.choice(top_pairs); mains=set(pair)
        pool=list(features['freq_main'].keys()); weights=[features['freq_main'][x] for x in pool]
        while len(mains)<RULES['main_count']:
            mains.add(rnd.choices(pool,weights=weights,k=1)[0])
        pb=rnd.choices(list(features['freq_pb'].keys()),weights=[features['freq_pb'][x] for x in features['freq_pb']],k=1)[0]
        lines.add((tuple(sorted(mains)),int(pb)))
    return [(list(k[0]),k[1]) for k in lines]

def gen_monte(features,n,seed=3):
    rnd=random.Random(seed); cand={}
    for _ in range(n*3):
        mains=sorted(rnd.sample(list(features['freq_main'].keys()),k=RULES['main_count']))
        pb=rnd.choice(list(features['freq_pb'].keys()))
        s=score_line(mains,pb,features)
        cand[(tuple(mains),pb)]=s
    best=sorted(cand.items(),key=lambda x:-x[1])[:n]
    return [(list(k[0]),k[1]) for k,v in best]

# Candidate pool builder
def build_pool(features,seeds=(10,11,12,13),mc_pool=MONTE_CARLO_POOL):
    pool={}
    gens=[lambda: gen_frequency(features,200,seeds[0]), lambda: gen_overdue(features,200,seeds[1]), lambda: gen_pair(features,300,seeds[2]), lambda: gen_monte(features,mc_pool,seeds[3])]
    for g in gens:
        for mains,pb in g():
            key=(tuple(mains),pb)
            if key not in pool:
                pool[key]=score_line(mains,pb,features)
    pool_list=[(list(k[0]),k[1],v) for k,v in pool.items()]
    pool_list.sort(key=lambda x:-x[2])
    return pool_list

# Diversity selection
def select_diverse(pool,k=8,overlap_penalty=1.6):
    selected=[]
    for mains,pb,score in pool:
        if len(selected)>=k: break
        overlap=sum(len(set(mains).intersection(s[0])) for s in selected)
        eff=score - overlap_penalty*overlap
        selected.append((mains,pb,score))
    # pad if needed
    i=0
    while len(selected)<k and i<len(pool):
        mains,pb,score=pool[i]
        if (tuple(mains),pb) not in [(tuple(x[0]),x[1]) for x in selected]: selected.append((mains,pb,score))
        i+=1
    scores=[s for _,_,s in selected]; min_s,max_s=min(scores),max(scores)
    games=[]
    for idx,(mains,pb,s) in enumerate(selected,1):
        conf = 50 if max_s==min_s else int(100*(s-min_s)/(max_s-min_s))
        games.append({'id':idx,'main':sorted(mains),'powerball':int(pb),'score':float(s),'confidence':conf})
    return games

# Validation
def validate(games):
    checks=[]; seen=set()
    for g in games:
        mains=g['main']; pb=g['powerball']
        if len(mains)!=RULES['main_count']: checks.append('wrong main count')
        if any(not (RULES['main_min']<=n<=RULES['main_max']) for n in mains): checks.append('main out of range')
        if len(set(mains))!=len(mains): checks.append('duplicate mains')
        if not (RULES['powerball_min']<=pb<=RULES['powerball_max']): checks.append('pb out of range')
        f=(tuple(mains),pb)
        if f in seen: checks.append('duplicate line')
        seen.add(f)
    return (len(checks)==0,checks)

# Reconcile two runs
def reconcile(A,B):
    combined={}
    for g in A+B:
        key=(tuple(g['main']),g['powerball']); combined.setdefault(key,{'count':0,'score':0,'g':g}); combined[key]['count']+=1; combined[key]['score']+=g.get('score',0)
    keys=sorted(combined.items(),key=lambda x:(-x[1]['count'],-x[1]['score']))
    selected=[v['g'] for k,v in keys][:8]
    # pad if needed
    if len(selected)<8:
        union = sorted([g for g in A+B], key=lambda x:-x.get('score',0))
        for g in union:
            if len(selected)>=8: break
            if (tuple(g['main']),g['powerball']) not in [(tuple(s['main']),s['powerball']) for s in selected]: selected.append(g)
    ok,checks=validate(selected)
    return selected, {'checks_passed':ok,'checks':checks}

# Hosted LLM critic/generator helpers (Hugging Face Inference API)
HF_API_URL = 'https://api-inference.huggingface.co/models/'

def hf_batch_score_candidates(candidates, model_name, token, batch_size=8):
    # candidates: list of (mains,pb) tuples; returns dict key->score (0-100)
    headers={'Authorization':f'Bearer {token}'}
    out={}
    for i in range(0,len(candidates),batch_size):
        batch=candidates[i:i+batch_size]
        prompts=[f"Rate this lottery line for Powerball Australia based on historical features and chance of jackpot: mains={','.join(map(str,m))} PB={pb}. Provide a numeric score 0-100 only." for m,pb in batch]
        # Use HF batch endpoint
        payload={'inputs':prompts}
        try:
            r=requests.post(HF_API_URL+model_name, headers=headers, json=payload, timeout=60)
            if r.status_code==200:
                results=r.json()
                # results may be string outputs; extract numeric
                for cand,resp in zip(batch, results):
                    text = resp.get('generated_text') if isinstance(resp,dict) else (resp[0] if isinstance(resp,list) else str(resp))
                    score_match = re.search(r'(\d{1,3})', text)
                    sc = int(score_match.group(1)) if score_match else 50
                    out[(tuple(cand[0]),cand[1])]=sc
            else:
                # fallback assign neutral 50
                for cand in batch: out[(tuple(cand[0]),cand[1])]=50
        except Exception:
            for cand in batch: out[(tuple(cand[0]),cand[1])]=50
    return out

# GA optimizer (optional, uses DEAP if available)
def run_ga_optimize(pool, features, pop_size=GA_POPULATION, gens=GA_GENERATIONS):
    if not DEAP_AVAILABLE:
        print('DEAP not available; skipping GA')
        return []
    # Represent individuals as indices into pool
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    pool_keys = [k for k in pool]
    npool = len(pool_keys)
    toolbox.register('idx', random.randrange, npool)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.idx, n=8)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    def eval_ind(ind):
        # unique constraint penalty
        lines = set()
        score=0
        for i in ind:
            mains,pb,sc = pool_keys[i]
            tup=(tuple(mains),pb)
            if tup in lines: score-=10
            else: score+=sc
            lines.add(tup)
        return (score,)
    toolbox.register('evaluate', eval_ind)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutUniformInt, low=0, up= max(0,npool-1), indpb=0.1)
    toolbox.register('select', tools.selTournament, tournsize=3)
    pop=toolbox.population(n=pop_size)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=gens, verbose=False)
    best=tools.selBest(pop, k=5)
    results=[]
    for b in best:
        lines=[]
        for i in b:
            mains,pb,sc = pool_keys[i]
            lines.append((mains,pb,sc))
        results.append(lines)
    return results

# Simulated annealing: simple hillclimb
def annealer_optimize(pool, features, iterations=ANNEAL_ITER):
    # pool is sorted list (mains,pb,score)
    rnd=random.Random(42)
    best = [pool[i] for i in range(min(8,len(pool)))]
    best_score = sum(x[2] for x in best)
    current = best[:]
    current_score = best_score
    for t in range(iterations):
        # propose swap: replace one entry with a random pool entry
        i = rnd.randrange(len(current))
        cand = list(current)
        replacement = pool[rnd.randrange(len(pool))]
        cand[i] = replacement
        cand_score = sum(x[2] for x in cand)
        if cand_score > current_score or rnd.random() < 0.001:
            current = cand; current_score = cand_score
            if current_score > best_score:
                best = current; best_score = current_score
    return [ {'main':sorted(x[0]), 'powerball':x[1], 'score':x[2]} for x in best ]

# Full pipeline run
def pipeline_run(seed=1234, use_hf=False, hf_models=HF_MODELS):
    df = fetch_history_or_csv(csv_path='sample_history.csv')
    features = build_features(df)
    pool = build_pool(features, seeds=(seed+1,seed+2,seed+3,seed+4), mc_pool=MONTE_CARLO_POOL)
    # optional HF critic scoring: rescore top N with hosted models
    if use_hf and HUGGINGFACE_TOKEN:
        top_candidates = [(p[0],p[1]) for p in pool[:500]]
        hf_scores = {}
        for model in hf_models:
            try:
                batch_scores = hf_batch_score_candidates(top_candidates, model, HUGGINGFACE_TOKEN, batch_size=8)
                for k,v in batch_scores.items(): hf_scores.setdefault(k,[]).append(v)
            except Exception:
                continue
        # average HF scores and blend into original score
        for i,(mains,pb,orig) in enumerate(pool[:500]):
            key=(tuple(mains),pb)
            if key in hf_scores:
                avg = sum(hf_scores[key])/len(hf_scores[key])
                # scale HF 0-100 to 0-1 and combine
                pool[i] = (mains,pb, orig + 0.02*avg)
    # selection
    selected = select_diverse(pool,k=8,overlap_penalty=1.6)
    ok,checks = validate(selected)
    if not ok:
        raise RuntimeError('Validation failed: ' + ','.join(checks))
    for g in selected:
        g['justification'] = f"score {g['score']:.2f}"
        g['method'] = 'ensemble'
    return {'features_summary':{'total_draws': features['total_draws']}, 'games':selected, 'features': features}

# Repeated-run orchestration and reconciliation
def execute_two_runs(seedA=202609, seedB=202610, use_hf=False):
    runA = pipeline_run(seedA, use_hf=use_hf)
    runB = pipeline_run(seedB, use_hf=use_hf)
    final_games, verification = reconcile(runA['games'], runB['games'])
    out = {
        'draw_date': DRAW_DATE_TARGET,
        'game_type':'Powerball (Australia)',
        'rules_confirmed': f"{RULES['main_count']} main from {RULES['main_min']}-{RULES['main_max']}; Powerball {RULES['powerball_min']}-{RULES['powerball_max']}",
        'sources': ['https://www.ozlotteries.com/ (scraped/fallback CSV)'],
        'passA': runA['features_summary'], 'passB': runB['features_summary'],
        'games': final_games,
        'verification': verification
    }
    human = [ f"Game {g['id']}: " + '-'.join(f"{n:02d}" for n in g['main']) + f" | PB {g['powerball']:02d} (conf {g.get('confidence',50)})" for g in final_games ]
    return out, human

if __name__=='__main__':
    use_hf = True if HUGGINGFACE_TOKEN else False
    try:
        out, human = execute_two_runs(seedA=random.randint(1,1<<30), seedB=random.randint(1,1<<30), use_hf=use_hf)
        print(json.dumps(out, indent=2))
        for line in human: print(line)
    except RuntimeError as e:
        print(str(e))
