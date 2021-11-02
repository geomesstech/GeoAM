import numpy as np
import math

# Neue Art von Exception
class Done(Exception): pass

# Liste der Fest- und Messpunkte (dict)
plist = {
    'A': {'name': 'A (fest)', 'h': 127.344, 'h0': None},
    'B': {'name': 'B (fest)', 'h': 141.659, 'h0': None},
    'C': {'name': 'C (fest)', 'h': 113.947, 'h0': None},
    'P1': {'name': 'P1', 'h': None, 'h0': None},
    'P2': {'name': 'P2', 'h': None, 'h0': None},
    'P3': {'name': 'P3', 'h': None, 'h0': None},
    'P4': {'name': 'P4', 'h': None, 'h0': None}
}

# Liste der Beobachtungen (array)
beob = [
    {'von': 'A', 'nach': 'P1', 'dh': 9.386, 's': 11.3},
    {'von': 'P2', 'nach': 'P1', 'dh': 2.147, 's': 5.0},
    {'von': 'P2', 'nach': 'P3', 'dh': 5.290, 's': 3.7},
    {'von': 'P3', 'nach': 'B', 'dh': 1.799, 's': 10.3},
    {'von': 'P4', 'nach': 'P3', 'dh': 1.894, 's': 6.1},
    {'von': 'P2', 'nach': 'P4', 'dh': 3.421, 's': 8.4},
    {'von': 'P1', 'nach': 'P4', 'dh': 1.262, 's': 3.9},
    {'von': 'C', 'nach': 'P4', 'dh': 24.034, 's': 15.9}
]


# Näherungswerte für Messpunkte
for key, obj in plist.items(): # key = P1, obj =  {'name': 'P1', 'h': None, 'h0': None}
    if obj['h'] is not None: # Festpunkt
        continue
    if obj['h0'] is not None: # Näherungswerte da??
        continue
    try:
        von_indizes = list(index for (index, d) in enumerate(beob) if d["von"] == key) # von_indizes = [6]
        for von in von_indizes:
            if plist[beob[von]['nach']]['h'] is not None:
                plist[key]['h0'] = plist[beob[von]['nach']]['h'] - beob[von]['dh']
                raise Done # Näherungswert wurde bestimmt
            elif plist[beob[von]['nach']]['h0'] is not None:
                plist[key]['h0'] = plist[beob[von]['nach']]['h0'] - beob[von]['dh']
                raise Done
        nach_indizes = list(index for (index, d) in enumerate(beob) if d["nach"] == key) # nach_indizes = [0, 1]
        for nach in nach_indizes:
            if plist[beob[nach]['von']]['h'] is not None:
                plist[key]['h0'] = plist[beob[nach]['von']]['h'] + beob[nach]['dh']
                raise Done
            elif plist[beob[nach]['von']]['h0'] is not None:
                plist[key]['h0'] = plist[beob[nach]['von']]['h0'] + beob[nach]['dh']
                raise Done
    except Done:
        continue

# print(plist.items())

# Liste der Unbekannten
ulist = list(key for key, obj in plist.items() if obj["h"] == None) # ulist = ["P1", "P2", "P3", "P4"]
# Anzahl der Unbekannten
u = len(ulist)
# Anzahl der Beobachtungen
n = len(beob)
# Designmatrix A
A = np.zeros((n, u))
for i, b in enumerate(beob):
    for j, ukey in enumerate(ulist):
        if ukey == b['von']:
            A[i, j] = -1
        if ukey == b['nach']:
            A[i, j] = 1

#print(A)

# Vektor der gekürzten Beobachtungen
ltilde = np.zeros(n)
for i, b in enumerate(beob):
    hvon, hnach = 0, 0
    if plist[b['von']]['h'] is not None:
        hvon = plist[b['von']]['h']
    elif plist[b['von']]['h0'] is not None:
        hvon = plist[b['von']]['h0']
    if plist[b['nach']]['h'] is not None:
        hnach = plist[b['nach']]['h']
    elif plist[b['nach']]['h0'] is not None:
        hnach = plist[b['nach']]['h0']

    ltilde[i] = b['dh'] - hnach + hvon

# print(ltilde)

# Gewichtsmatrix
s = list(b['s'] for b in beob) # Liste der Strecken
P = np.diag(10 / np.array(s))
# print(P)

# Normalgleichungsmatrix N = A' * P * A
N = A.T.dot(P).dot(A)

# Kofaktormatrix Q = N^(-1)
Q = np.linalg.inv(N)

# Rechte Seite y = A' * P * ltilde
y = A.T.dot(P).dot(ltilde)

# Zuschläge zur Näherungswerte dx = N^(-1) * y
dx = np.linalg.inv(N).dot(y)

# Vektor der Verbesserungen v = A * dx - ltilde
v = A.dot(dx) - ltilde

# mittlerer Gewichtseinheitsfehler sqrt(v'*P*v / (n -u))
m0 = math.sqrt(v.T.dot(P).dot(v) / (n - u))
#print(m0)

for i, ukey in enumerate(ulist):
    plist[ukey]['h'] = plist[ukey]['h0'] + dx[i] # h = h0 + dx
    # mittlerer Fehler der Unbekannten
    plist[ukey]['m'] = m0 * math.sqrt(Q[i,i]) # mi = m0 * sqrt(Q(i,i))

# Resultate
print("-" * 77)
header = ["Punkt", "H [m]", "H0 [m]", "m [mm]"]
print("| {:^16s} | {:^16s} | {:^16s} | {:^16s} |".format(*header))
print("-" * 77)
for key, obj in plist.items():
    if obj['h'] is not None and obj['h0'] is None:
        print(f"| {obj['name']:^16s} | {obj['h']:^16.3f} | {'Festpunkt':^16s} | {'Festpunkt':^16s} |")
    elif obj['h'] is not None and obj['h0'] is not None:
        print(f"| {obj['name']:^16s} | {obj['h']:^16.3f} | {obj['h0']:^16.3f} | +/- {obj['m']*1000:^12.3f} |")
print("-" * 77)
