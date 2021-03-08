import os, requests, time
import pickle as pkl
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from datetime import datetime


# URL and headers for pageview website
pv_url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/%s/daily/%s/%s"
pv_head = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36"}

# Setting important dates
stime = "2015070100"
etime = str(datetime.today().date()).replace("-", "") + "00"

def save_pagenames(pnames):
    with open("pagenames.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(pnames), )
        
def load_pagenames():
    try:
        with open("pagenames.txt", "r", encoding="utf-8") as f:
            return f.read().split("\n")
    except:
        return []

def collect_pages(site, keywords, limit=None):
    # Search for each keyword
    searches = [site.search(kw.lower(), where="text", namespaces=0) for kw in keywords]
    
    # Get the pages and flatten
    pages_raw = [list(s) for s in searches]
    pages = list(set([item for sublist in pages_raw for item in sublist]))
    ret = {}
    old_names = [] # load_pagenames()
    ignored = []
    
    if limit:
        pages = pages[:limit]
    
    # We filter out pages that do not have a WikiData item
    for p in tqdm(pages):
        try:
            if p.title() not in old_names:
                x = p.data_item().title()
        except:
            ignored.append(p.title())
            continue

        ret[x] = p
    
    new_pnames = list(set(old_names + [p.title() for p in ret]))
    save_pagenames(new_pnames)
    
    print("Pages ignored :", ignored)
    print(len(ret), "pages currently tracked.")
    
    return ret

def byteRep(size):
    names = ["", "K", "M", "G", "T", "P"]
    i = 0
    
    while size >= 1000 and i < len(names) - 1:
        i += 1
        size = int(size) / 1000
    
    return str(size) + " " + names[i] + "B"

def savePages(name, pages):
    fn = name + ".pkl"
    path = os.path.join("pickles", fn)
    
    with open(path, "wb") as f:
        pkl.dump(pages, f)
        
    total_size = os.path.getsize(path)
    part_size = total_size / len(pages)
    print("Pages saved to '" + fn + "', average space per page :", byteRep(part_size) + ", total :", byteRep(total_size))

def loadPages(name):
    fn = name + ".pkl"
    path = os.path.join("pickles", fn)
          
    with open(path, "rb") as f:
        pages = pkl.load(f)
        
    return pages

def batches(iterable, n=1):
    l = len(iterable)
    
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def getData(code, page, strings):
    t1 = time.time()
    rvs = list(page.revisions(reverse=False, content=True))
    t2 = time.time()
    mns = getMentions(rvs, code, strings)
    t3 = time.time()
    szs = getSizes(rvs, code)
    t4 = time.time()
    eds = getEdits(rvs, code)
    t5 = time.time()
    vws = getViews(page, code)
    t6 = time.time()
    print(code, "done")
    
    xxx = str(t2-t1) + "\n" + str(t3-t2) + "\n" + str(t4-t3) + "\n" + str(t5-t4) + "\n" + str(t6-t5)
    
    return (mns, szs, eds, vws, xxx)

def updateTimeSeries(pages, strings, rescan=False, flush=False, batch_size=2):
    res = tuple([] for _ in range(4))
    pool = mp.Pool(2)#mp.cpu_count())
    
    try:
        for batch in batches(list(pages.items()), batch_size):
            full_batch = [b + (strings,) for b in batch]
            sub_res = pool.starmap(getData, full_batch)   
            #sub_res = tuple([pd.concat(l, axis=1) for l in map(list, list(zip(*sub_res)))])
            #res = pd.concat([res, sub_res], axis=1)
            for _, _, _, _, xxx in sub_res:
                print(xxx)
                print("next")
    except Exception as e:
        pool.terminate()
            
    return res

def getCounts(revs, strings, idx):
    text = revs[idx]
    return sum([text.text.count(s) for s in strings])

def getOrUpdate(revs, strings, counts, idx, changes):
    if idx not in counts:
        temp = getCounts(revs, strings, idx)
        counts[idx] = temp
        
        # Do not consider the count if an earlier revision had more
        if not any([counts[k] > temp for k in counts.keys() if k < idx]):
            changes[temp] = min(changes.get(temp) or idx, idx) 
    
    return counts[idx]

def getMentions(revs_flip, strings, code):
    if not getCounts(revs_flip, strings, 0):
        return None
    
    # Reversing revisions
    revs = revs_flip[::-1]

    # Start with whole scope
    queue = [(0, len(revs) - 1)]
    
    # To avoid double checking revisions we store the counts here
    cnts = {}
    
    # And here we store the count-index pairs
    changes = {}

    while queue:
        # Process first element
        r0, r1 = queue[0]
        queue = queue[1:]

        # Only proceed if current scope covers multiple indices
        if r0 != r1:
            # Get counts for both indices
            v0 = getOrUpdate(revs, strings, cnts, r0, changes)
            v1 = getOrUpdate(revs, strings, cnts, r1, changes)

            # Only proceed if there is a change of count in the current scope
            if v0 != v1 and abs(r1 - r0) > 1:
                mid = (r0 + r1) // 2
                queue.extend([(r0, mid), (mid, r1)])

    changes = {revs[v]["timestamp"]: k for k, v in changes.items()}
    changes = {datetime.combine(k.date(), k.time()): v for k, v in changes.items()}
    
    # Here we simplify our data to a maximum of one point per month (we take the last one)
    changes = pd.Series(changes, name="Mentions").sort_index().groupby(pd.Grouper(freq="1M")).nth(-1)
    return changes

def getSizes(revs, code):
    df = pd.DataFrame([dict(r) for r in revs])
    df = df[["userid", "timestamp", "size"]]
    df = df.set_index("timestamp")
    
    # Get absolute size from relative size
    df["diff"] = (df['size'] - df['size'].shift(1)).abs()
    df["diff"] = df["diff"].fillna(df["size"])
    
    # Sample every month and shift by 1 day to get 1st of month
    se = df["size"].groupby(pd.Grouper(freq="1M")).nth(-1).resample("1M").pad()
    se.index = se.index.shift(1, freq="D")
    
    return se.rename(code)

def getEdits(revs, code):
    df = pd.DataFrame([dict(r) for r in revs])
    df = df.set_index("timestamp")
    df["size"] = 1

    se = df["size"].groupby(pd.Grouper(freq="1M")).count()
    se.index = se.index.shift(-1, freq="M").shift(1, freq="D")
    
    return se.rename(code)

# Need to make it wiki-independant
def getViews(page, code):
    req = requests.get(pv_url % (page.title(), stime, etime), headers=pv_head)
    se = pd.Series({datetime.strptime(str(item['timestamp'])[:-2], "%Y%m%d"): item['views'] for item in req.json()['items']}, name="Views")
    return se.rename(code)