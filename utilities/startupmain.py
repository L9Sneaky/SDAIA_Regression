from VGChartz import get_games_data
import threading
import concurrent.futures as cf

def trycatch(lst, index):
#     print(lst)
    try:
        return lst[index+1]
    except Exception:
        return

x = [x for x in range(1, 7)]
run_ranges= [[y, trycatch(x, i)] for i, y in enumerate([x for x in range(1, 7)])]
run_ranges.pop()
with cf.ThreadPoolExecutor() as executor:
    results =[]
    for i, run_range in enumerate(run_ranges):
        results.append(executor.submit(get_games_data, run_range[0], run_range[1], i))
    for f in cf.as_completed(results):
        f.result()



df =results[0].result()

for i in results[1:]:
    df = pd.concat([df, i.result()]).reset_index(drop=True)




df.to_csv('F:/Books/T10/sdaia t5/MTA Project/SDAIA_Regression/data/vgchartz_games.csv')
