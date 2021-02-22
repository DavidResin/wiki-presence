'''
Arrays to complete:
- Mentions over time in each page
- Page size over time in each page
- Edits over time
- Editor count over time

Later on we can memorize the editors and do stuff with this

Long terme : perform search and process simultaneously
'''

def getCounts(revs, strings, idx):
    page = revs[idx]
    return sum(pageCounts(page, strings))

def getOrUpdate(revs, strings, counts, idx, changes):
    if idx not in counts:
        temp = getCounts(revs, strings, idx)
        counts[idx] = temp
        
        # Do not consider the count if an earlier revision had more
        if not any([counts[k] > temp for k in counts.keys() if k < idx]):
            changes[temp] = min(changes.get(temp) or idx, idx) 
    
    return counts[idx]

def getMentions(page, strings):
    if not getCounts([page], strings, 0):
        return None
    
    cnt = page.revision_count()
    revs = list(page.revisions(reverse=True, content=True))

    # Start with whole scope
    queue = [(0, cnt - 1)]
    
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
    
    # Here we simplify our data to a maximum of one point per day (we take the last one)
    changes = pd.Series(changes, name="Mentions").sort_index().groupby(pd.Grouper(freq="1D")).nth(-1)
    return changes

def updateMentions(pagecodes, rescan=False):
    path = os.path.join("pickles", "en_mentions.pkl")
    
    try:
        prev = pd.read_pickle(path)
    except:
        prev = pd.DataFrame()
        
    dfs = [prev]
    
    if rescan:
        pagecodes += list(prev.columns)
    
    for code in tqdm(pagecodes):
        try:
            # Recover page from code
            p = pwb.Page(wiki_site, pwb.ItemPage(repo, code).sitelinks["enwiki"].ns_title())
        except:
            continue

        # Set limit timestamp (or None if no data yet)
        ts = prev[code].last_valid_index() if code in prev.columns else None
        
        # Get values after that timestamp
        df = getMentions(p, epfl_alts)
        
        if df is not None and len(df):
            df.name = code
            df = df.groupby(pd.Grouper(freq="1M")).nth(-1).resample("1M").pad()
            df.index = df.index.shift(1, freq="D")
            
            # Combine with old data if it exists
            if code in prev.columns:
                df = df.combine_first(prev[code])

            dfs.append(df)
    
    curr = pd.concat(dfs, axis=1)
    curr = curr.ffill(axis=0)
    curr.to_pickle(path)
    
    return curr