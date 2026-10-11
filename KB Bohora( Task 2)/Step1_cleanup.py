import openpyxl, csv

wb = openpyxl.load_workbook('sportsref_download.xltx', data_only=True)
ws = wb['Worksheet']
rows = list(ws.iter_rows(min_row=2, values_only=True))
data = rows[2:]

out = []
for r in data:
    rk, player, pos, squad, age, born, nineties, crdy, crdr, crd2, fls, fld, off, crs, intc, tklw, pkw, pkc = r
    if rk is None or player is None:
        continue
    if nineties is None or nineties == 0:
        continue
    fouls_per90 = fls / nineties
    out.append([player, pos, squad, age, nineties, fls, round(fouls_per90, 3)])

with open('wc2026_player_misc.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Player','Pos','Squad','Age','Nineties','Fls','Fouls_per90'])
    w.writerows(out)

print(f'Done. {len(out)} players saved.')
